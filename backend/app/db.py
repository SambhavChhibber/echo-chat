"""SQLite persistence (aiosqlite, WAL mode). Sessions are scoped to an anonymous client id."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    client_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions(client_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    source            TEXT,           -- LLM_CACHE | LIVE_INFERENCE (assistant only)
    model             TEXT,
    deployment_id     TEXT,
    deployment_label  TEXT,
    latency_ms        INTEGER,
    ttft_ms           INTEGER,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    tokens_estimated  INTEGER,
    attempts_json     TEXT,
    context_messages  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA foreign_keys=ON")
        await self.conn.executescript(SCHEMA)
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self.conn is not None, "Database not connected"
        return self.conn

    # ------------------------------------------------------------------ sessions
    async def create_session(self, client_id: str, title: str, system_prompt: str) -> dict[str, Any]:
        ts = now_iso()
        sid = new_id()
        await self.db.execute(
            "INSERT INTO sessions (id, client_id, title, system_prompt, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (sid, client_id, title, system_prompt, ts, ts),
        )
        await self.db.commit()
        return await self.get_session(sid, client_id)  # type: ignore[return-value]

    async def list_sessions(self, client_id: str) -> list[dict[str, Any]]:
        cur = await self.db.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
               FROM sessions s WHERE s.client_id = ? ORDER BY s.updated_at DESC LIMIT 200""",
            (client_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_session(self, session_id: str, client_id: str) -> dict[str, Any] | None:
        cur = await self.db.execute("SELECT * FROM sessions WHERE id = ? AND client_id = ?", (session_id, client_id))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def update_session(self, session_id: str, client_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {k: v for k, v in fields.items() if k in ("title", "system_prompt") and v is not None}
        if allowed:
            sets = ", ".join(f"{k} = ?" for k in allowed)
            await self.db.execute(
                f"UPDATE sessions SET {sets}, updated_at = ? WHERE id = ? AND client_id = ?",
                (*allowed.values(), now_iso(), session_id, client_id),
            )
            await self.db.commit()
        return await self.get_session(session_id, client_id)

    async def delete_session(self, session_id: str, client_id: str) -> bool:
        cur = await self.db.execute("DELETE FROM sessions WHERE id = ? AND client_id = ?", (session_id, client_id))
        await self.db.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ messages
    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        cur = await self.db.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at, rowid", (session_id,))
        return [serialize_message(dict(r)) for r in await cur.fetchall()]

    async def save_turn(self, session_id: str, user_content: str, assistant: dict[str, Any], title: str | None) -> tuple[dict, dict]:
        """Persist the user message and the reply atomically (no half-saved turns)."""
        user_id, asst_id = new_id(), new_id()
        user_ts = assistant.pop("user_created_at")
        asst_ts = now_iso()
        await self.db.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (user_id, session_id, "user", user_content, user_ts),
        )
        await self.db.execute(
            """INSERT INTO messages (id, session_id, role, content, created_at, source, model, deployment_id,
                   deployment_label, latency_ms, ttft_ms, prompt_tokens, completion_tokens, tokens_estimated,
                   attempts_json, context_messages)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                asst_id, session_id, "assistant", assistant["content"], asst_ts, assistant["source"],
                assistant["model"], assistant["deployment_id"], assistant["deployment_label"],
                assistant["latency_ms"], assistant["ttft_ms"], assistant["prompt_tokens"],
                assistant["completion_tokens"], int(assistant["tokens_estimated"]),
                json.dumps(assistant["attempts"]), assistant["context_messages"],
            ),
        )
        if title:
            await self.db.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, asst_ts, session_id))
        else:
            await self.db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (asst_ts, session_id))
        await self.db.commit()
        cur = await self.db.execute("SELECT * FROM messages WHERE id IN (?, ?)", (user_id, asst_id))
        rows = {r["id"]: serialize_message(dict(r)) for r in await cur.fetchall()}
        return rows[user_id], rows[asst_id]

    async def stats(self, client_id: str) -> dict[str, Any]:
        cur = await self.db.execute(
            """SELECT COUNT(*) AS replies,
                      SUM(m.source = 'LLM_CACHE') AS cache_hits,
                      SUM(CASE WHEN m.source = 'LLM_CACHE' THEN m.prompt_tokens + m.completion_tokens ELSE 0 END) AS tokens_saved,
                      SUM(CASE WHEN m.source = 'LIVE_INFERENCE' THEN m.prompt_tokens + m.completion_tokens ELSE 0 END) AS tokens_billed,
                      AVG(CASE WHEN m.source = 'LIVE_INFERENCE' THEN m.latency_ms END) AS avg_live_ms,
                      AVG(CASE WHEN m.source = 'LLM_CACHE' THEN m.latency_ms END) AS avg_cache_ms,
                      SUM(m.attempts_json LIKE '%"failed"%') AS failovers
               FROM messages m JOIN sessions s ON s.id = m.session_id
               WHERE s.client_id = ? AND m.role = 'assistant'""",
            (client_id,),
        )
        row = dict(await cur.fetchone())
        replies = row["replies"] or 0
        hits = row["cache_hits"] or 0
        return {
            "replies": replies,
            "cache_hits": hits,
            "cache_hit_rate": round(hits / replies, 3) if replies else 0.0,
            "tokens_saved": row["tokens_saved"] or 0,
            "tokens_billed": row["tokens_billed"] or 0,
            "avg_live_ms": round(row["avg_live_ms"]) if row["avg_live_ms"] else None,
            "avg_cache_ms": round(row["avg_cache_ms"]) if row["avg_cache_ms"] else None,
            "failovers": row["failovers"] or 0,
        }


def serialize_message(row: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
    if row["role"] == "assistant":
        msg["meta"] = {
            "source": row["source"],
            "model": row["model"],
            "deployment_id": row["deployment_id"],
            "deployment_label": row["deployment_label"],
            "latency_ms": row["latency_ms"],
            "ttft_ms": row["ttft_ms"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "tokens_estimated": bool(row["tokens_estimated"]),
            "attempts": json.loads(row["attempts_json"] or "[]"),
            "context_messages": row["context_messages"],
        }
    return msg
