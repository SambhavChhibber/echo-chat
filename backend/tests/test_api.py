"""End-to-end API tests against the MOCK key pool (no network, no keys needed)."""

from __future__ import annotations

import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "test.db"),
        cache_type="local",
        cooldown_seconds=1,
        rate_limit_per_minute=0,
        MOCK_MODE=True,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        c.headers["X-Client-Id"] = "test-" + uuid.uuid4().hex[:12]
        yield c


def new_session(client, **body):
    r = client.post("/api/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def send(client, sid, content, **extra):
    r = client.post(f"/api/sessions/{sid}/messages", json={"content": content, **extra})
    return r


def test_health_and_pool(client):
    assert client.get("/api/health").json()["mock_mode"] is True
    deps = client.get("/api/pool").json()["deployments"]
    assert [d["id"] for d in deps] == ["mock-primary", "mock-backup"]
    assert all("api_key" not in d for d in deps)


def test_reply_has_full_metadata(client):
    sid = new_session(client)
    r = send(client, sid, "Hello " + uuid.uuid4().hex)
    assert r.status_code == 200, r.text
    meta = r.json()["message"]["meta"]
    assert meta["source"] == "LIVE_INFERENCE"
    assert meta["model"] == "mock-model-alpha"
    assert meta["deployment_id"] == "mock-primary"
    assert isinstance(meta["latency_ms"], int)
    assert meta["prompt_tokens"] > 0 and meta["completion_tokens"] > 0


def test_multi_turn_history_is_sent(client):
    sid = new_session(client)
    send(client, sid, "My name is Ada " + uuid.uuid4().hex)
    r = send(client, sid, "What is my name?")
    body = r.json()
    # mock echoes how many messages of context it received: system + 2 history + new user = 4
    assert "received 4 messages" in body["message"]["content"]
    assert body["message"]["meta"]["context_messages"] == 2
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_tokens_are_per_request_not_cumulative(client):
    sid = new_session(client)
    a = send(client, sid, "short " + uuid.uuid4().hex).json()["message"]["meta"]
    b = send(client, sid, "short " + uuid.uuid4().hex).json()["message"]["meta"]
    # second prompt is larger (it carries history) but completion counts are independent
    assert b["prompt_tokens"] > a["prompt_tokens"]
    assert b["prompt_tokens"] < a["prompt_tokens"] + a["completion_tokens"] + b["prompt_tokens"]


def test_identical_request_is_served_from_cache(client):
    q = "Cache me " + uuid.uuid4().hex
    first = send(client, new_session(client), q).json()["message"]
    time.sleep(0.2)  # LiteLLM writes the cache entry asynchronously
    second = send(client, new_session(client), q).json()["message"]
    assert first["meta"]["source"] == "LIVE_INFERENCE"
    assert second["meta"]["source"] == "LLM_CACHE"
    assert second["content"] == first["content"]
    # bypass_cache forces fresh inference
    third = send(client, new_session(client), q, bypass_cache=True).json()["message"]
    assert third["meta"]["source"] == "LIVE_INFERENCE"


def test_failover_on_429(client):
    sid = new_session(client)
    r = send(client, sid, "fail over " + uuid.uuid4().hex, simulate_failures={"mock-primary": "rate_limit"})
    assert r.status_code == 200, r.text
    meta = r.json()["message"]["meta"]
    assert meta["deployment_id"] == "mock-backup"
    assert meta["model"] == "mock-model-beta"
    failed = [a for a in meta["attempts"] if a["outcome"] == "failed"]
    assert failed[0]["deployment_id"] == "mock-primary"
    assert failed[0]["error_code"] == "rate_limited"
    # the failing key is now in cooldown and skipped by the next request
    pool = {d["id"]: d for d in client.get("/api/pool").json()["deployments"]}
    assert pool["mock-primary"]["status"] == "cooldown"
    nxt = send(client, sid, "next " + uuid.uuid4().hex).json()["message"]["meta"]
    assert nxt["deployment_id"] == "mock-backup"
    assert [a["outcome"] for a in nxt["attempts"]] == ["served"]


@pytest.mark.parametrize("fault,code", [("quota", "quota_exceeded"), ("server_error", "provider_error"),
                                        ("unavailable", "provider_error"), ("timeout", "timeout")])
def test_failover_on_other_errors(client, fault, code):
    r = send(client, new_session(client), uuid.uuid4().hex, simulate_failures={"mock-primary": fault})
    meta = r.json()["message"]["meta"]
    assert meta["deployment_id"] == "mock-backup"
    assert meta["attempts"][0]["error_code"] == code


def test_all_keys_down_is_graceful(client):
    sid = new_session(client)
    r = send(client, sid, "doomed " + uuid.uuid4().hex,
             simulate_failures={"mock-primary": "rate_limit", "mock-backup": "server_error"})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "all_keys_failed"
    assert err["retryable"] is True
    assert {a["deployment_id"] for a in err["attempts"]} == {"mock-primary", "mock-backup"}
    # nothing half-saved
    assert client.get(f"/api/sessions/{sid}").json()["messages"] == []
    # recovers after cooldown
    time.sleep(1.2)
    ok = send(client, sid, "back " + uuid.uuid4().hex)
    assert ok.status_code == 200


def test_sse_stream(client):
    sid = new_session(client)
    with client.stream("POST", f"/api/sessions/{sid}/messages/stream", json={"content": "stream " + uuid.uuid4().hex}) as r:
        raw = "".join(r.iter_text())
    events = [blk.split("\n")[0].removeprefix("event: ") for blk in raw.strip().split("\n\n")]
    assert events[0] == "start" and events[-1] == "done" and "delta" in events
    done = json.loads(raw.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert done["message"]["meta"]["source"] == "LIVE_INFERENCE"
    assert done["session_title"].startswith("stream")


def test_system_prompt_and_isolation(client):
    sid = new_session(client, system_prompt="You are a pirate.")
    assert client.get(f"/api/sessions/{sid}").json()["system_prompt"] == "You are a pirate."
    r = client.patch(f"/api/sessions/{sid}", json={"system_prompt": "Be brief."})
    assert r.json()["system_prompt"] == "Be brief."
    other = client.get(f"/api/sessions/{sid}", headers={"X-Client-Id": "someone-else-123"})
    assert other.status_code == 404


def test_validation_errors_are_json(client):
    sid = new_session(client)
    assert send(client, sid, "   ").json()["error"]["code"] == "empty_message"
    assert send(client, sid, "x" * 9000).json()["error"]["code"] == "message_too_long"
    assert client.get("/api/sessions", headers={"X-Client-Id": ""}).status_code == 400


def test_failover_when_key_fails_after_stream_opens(client):
    """A provider error on the first chunk (after the stream opened) must still fail over."""
    sid = new_session(client)
    r = send(client, sid, "midstream " + uuid.uuid4().hex, simulate_failures={"mock-primary": "midstream_error"})
    assert r.status_code == 200, r.text
    meta = r.json()["message"]["meta"]
    assert meta["deployment_id"] == "mock-backup"
    assert [a["outcome"] for a in meta["attempts"]] == ["failed", "served"]
    assert meta["attempts"][0]["deployment_id"] == "mock-primary"
    pool = {d["id"]: d for d in client.get("/api/pool").json()["deployments"]}
    assert pool["mock-primary"]["status"] == "cooldown"
