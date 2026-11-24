"""LiteLLM gateway: key pool, ordered failover, response cache, per-request trace.

How it works
------------
* Every API key is a LiteLLM Router *deployment* in one model group ("chat"),
  with an explicit ``order``. The router tries order 1, and on a 429 / quota /
  5xx / timeout / auth error falls back to order 2, 3 ... automatically.
  Failing keys are put in cooldown (circuit breaker) so later requests skip them.
* LiteLLM's built-in cache (disk or in-memory) is enabled on the router. Because
  all keys share one model group, a cached answer is reused no matter which key
  produced it.
* A per-request ``RequestTrace`` lives in a ContextVar. LiteLLM's own hooks write
  into it: which deployments were attempted, which failed and why, and whether
  the cache lookup hit. That is how each reply gets reliable metadata.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

import litellm
from litellm import Router
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger

from app.config import Deployment, Settings

litellm.suppress_debug_info = True
litellm.drop_params = True  # ignore params a provider does not support instead of failing

MODEL_GROUP = "chat"
LLM_CACHE = "LLM_CACHE"
LIVE_INFERENCE = "LIVE_INFERENCE"

FAULT_TYPES = {
    "rate_limit": "HTTP 429 · rate limited",
    "quota": "HTTP 429 · quota exhausted",
    "server_error": "HTTP 500 · server error",
    "unavailable": "HTTP 503 · unavailable",
    "timeout": "Timeout",
}


# --------------------------------------------------------------------------- trace
@dataclass
class AttemptError:
    code: str
    status: int | None
    message: str


@dataclass
class RequestTrace:
    faults: dict[str, str] = field(default_factory=dict)
    attempts: list[str] = field(default_factory=list)
    failures: dict[str, AttemptError] = field(default_factory=dict)
    cache_lookups: list[bool] = field(default_factory=list)


TRACE: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("llm_trace", default=None)


def classify_error(exc: BaseException) -> AttemptError:
    """Map any provider/router exception to a stable code + friendly message."""
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    text = str(exc)
    low = text.lower()
    if isinstance(exc, litellm.Timeout) or "timeout" in type(exc).__name__.lower():
        code = "timeout"
    elif isinstance(exc, litellm.RateLimitError) or status == 429:
        code = "quota_exceeded" if ("quota" in low or "resource_exhausted" in low) else "rate_limited"
    elif isinstance(exc, litellm.AuthenticationError) or status in (401, 403):
        code = "auth_error"
    elif isinstance(exc, litellm.ContextWindowExceededError):
        code = "context_too_long"
    elif "no deployments available" in low or "cooldown" in low:
        code = "all_keys_cooling_down"
    elif isinstance(exc, litellm.BadRequestError) or status == 400:
        code = "bad_request"
    elif isinstance(exc, litellm.NotFoundError) or status == 404:
        code = "model_not_found"
    elif status is not None and status >= 500:
        code = "provider_error"
    else:
        code = "provider_error"
    # Strip LiteLLM's noisy prefixes, keep it short and key-free.
    message = text.replace("litellm.", "").split("\n")[0][:240]
    return AttemptError(code=code, status=status, message=message)


def _build_fault(kind: str, model: str) -> Exception:
    provider = "simulated"
    if kind == "rate_limit":
        return litellm.RateLimitError(message="Simulated: 429 Too Many Requests", llm_provider=provider, model=model)
    if kind == "quota":
        return litellm.RateLimitError(
            message="Simulated: 429 RESOURCE_EXHAUSTED - quota exceeded for this key", llm_provider=provider, model=model
        )
    if kind == "server_error":
        return litellm.InternalServerError(message="Simulated: 500 Internal Server Error", llm_provider=provider, model=model)
    if kind == "unavailable":
        return litellm.ServiceUnavailableError(message="Simulated: 503 Service Unavailable", llm_provider=provider, model=model)
    return litellm.Timeout(message="Simulated: request timed out", llm_provider=provider, model=model)


def _mock_reply(messages: list[dict[str, Any]]) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    turns = sum(1 for m in messages if m["role"] == "user")
    return (
        f"**Mock mode** (no API keys configured). You said: “{last[:300]}”.\n\n"
        f"This is turn {turns} of the conversation; I received {len(messages)} messages of context. "
        "Add `GEMINI_API_KEYS` / `GROQ_API_KEYS` to get real answers."
    )


class _ProbeCache(Cache):
    """LiteLLM's own Cache, plus a note in the trace when a lookup hits."""

    async def async_get_cache(self, *args: Any, **kwargs: Any) -> Any:
        value = await super().async_get_cache(*args, **kwargs)
        trace = TRACE.get()
        if trace is not None:
            trace.cache_lookups.append(value is not None)
        return value


class _GatewayHooks(CustomLogger):
    def __init__(self, gateway: LLMGateway) -> None:
        super().__init__()
        self.gateway = gateway

    @staticmethod
    def _deployment_id(kwargs: dict[str, Any]) -> str | None:
        info = kwargs.get("model_info") or ((kwargs.get("litellm_params") or {}).get("model_info")) or {}
        if not info:
            info = ((kwargs.get("metadata") or {}).get("model_info")) or {}
            if not info:
                info = (((kwargs.get("litellm_params") or {}).get("metadata") or {}).get("model_info")) or {}
        return info.get("id")

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: Any) -> dict[str, Any]:
        """Runs after the router picked a key, before the cache lookup + provider call."""
        dep_id = self._deployment_id(kwargs)
        dep = self.gateway.deployments.get(dep_id or "")
        trace = TRACE.get()
        if trace is not None and dep_id:
            trace.attempts.append(dep_id)
            fault = trace.faults.get(dep_id)
            if fault and self.gateway.settings.enable_fault_injection:
                # Raised inside LiteLLM's provider path → identical handling to a real error.
                kwargs["mock_response"] = _build_fault(fault, str(kwargs.get("model")))
                return kwargs
        if dep is not None and dep.mock_response is not None:
            kwargs["mock_response"] = _mock_reply(kwargs.get("messages") or [])
        return kwargs

    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        trace = TRACE.get()
        dep_id = self._deployment_id(kwargs)
        exc = kwargs.get("exception")
        if trace is not None and dep_id and exc is not None:
            trace.failures[dep_id] = classify_error(exc)


# --------------------------------------------------------------------------- results
@dataclass
class Attempt:
    deployment_id: str
    label: str
    model: str
    outcome: str  # "failed" | "served" | "cache"
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class ChatResult:
    content: str
    source: str  # LLM_CACHE | LIVE_INFERENCE
    model: str
    deployment_id: str | None
    deployment_label: str | None
    latency_ms: int
    ttft_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool
    attempts: list[Attempt]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GatewayError(Exception):
    def __init__(self, code: str, message: str, attempts: list[Attempt], latency_ms: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.attempts = attempts
        self.latency_ms = latency_ms


@dataclass
class DeploymentStats:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    last_error: str | None = None
    last_error_at: float | None = None
    last_latency_ms: int | None = None


# --------------------------------------------------------------------------- gateway
class LLMGateway:
    def __init__(self, settings: Settings, pool: list[Deployment]) -> None:
        if not pool:
            raise ValueError("Key pool is empty")
        self.settings = settings
        self.pool = sorted(pool, key=lambda d: d.order)
        self.deployments = {d.id: d for d in self.pool}
        self.stats = {d.id: DeploymentStats() for d in self.pool}
        self.mock_mode = all(d.mock_response is not None for d in self.pool)

        if settings.cache_type == "disk":
            litellm.cache = _ProbeCache(type="disk", disk_cache_dir=settings.cache_dir, ttl=settings.cache_ttl_seconds)
        else:
            litellm.cache = _ProbeCache(type="local", ttl=settings.cache_ttl_seconds)
        self._hooks = _GatewayHooks(self)
        litellm.callbacks = [self._hooks]

        model_list = []
        for d in self.pool:
            params: dict[str, Any] = {
                "model": d.model,
                "api_key": d.api_key,
                "order": d.order,
                "timeout": settings.request_timeout_seconds,
                "stream_timeout": settings.request_timeout_seconds,
                "max_retries": 0,  # never retry inside the SDK: fail over to the next key instead
                **d.extra_params,
            }
            if d.api_base:
                params["api_base"] = d.api_base
            model_list.append({"model_name": MODEL_GROUP, "litellm_params": params, "model_info": {"id": d.id}})

        self.router = Router(
            model_list=model_list,
            num_retries=0,  # the next key *is* the retry
            retry_after=0,
            cooldown_time=settings.cooldown_seconds,
            allowed_fails=0,
            cache_responses=True,
        )
        # LiteLLM lazily imports modules on the first cache hit; warm them so hits are ~ms.
        try:
            import importlib

            importlib.import_module("litellm.llms.anthropic.pass_through.messages.response_cache")
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------ health
    async def pool_status(self) -> list[dict[str, Any]]:
        active: dict[str, Any] = {}
        try:
            rows = await self.router.cooldown_cache.async_get_active_cooldowns(list(self.deployments), None)
            active = {dep_id: value for dep_id, value in rows}
        except Exception:
            active = {}
        now = time.time()
        out = []
        for d in self.pool:
            cd = active.get(d.id)
            remaining = 0.0
            if cd:
                remaining = max(0.0, float(cd.get("timestamp", now)) + float(cd.get("cooldown_time", 0)) - now)
            out.append(
                {
                    "id": d.id,
                    "label": d.label,
                    "provider": d.provider,
                    "model": d.model_name,
                    "order": d.order,
                    "key_hint": d.key_hint,
                    "status": "cooldown" if remaining > 0 else "healthy",
                    "cooldown_remaining_s": round(remaining, 1),
                    **asdict(self.stats[d.id]),
                }
            )
        return out

    # ------------------------------------------------------------------ chat
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        faults: dict[str, str] | None = None,
        bypass_cache: bool = False,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yields ("delta", str) chunks, then ("done", ChatResult). Raises GatewayError.

        Must be consumed inside a single task (the ContextVar trace is task-local).
        """
        trace = RequestTrace(faults={k: v for k, v in (faults or {}).items() if v in FAULT_TYPES})
        token = TRACE.set(trace)
        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        usage = None
        model_reported: str | None = None
        stream = None
        try:
            kwargs: dict[str, Any] = {
                "model": MODEL_GROUP,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.max_output_tokens,
                "include_fallback_errors": True,
                "metadata": {"request_id": uuid.uuid4().hex},
            }
            if bypass_cache:
                kwargs["cache"] = {"no-cache": True}
            stream = await self.router.acompletion(**kwargs)
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if getattr(chunk, "model", None):
                    model_reported = chunk.model
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    parts.append(text)
                    yield "delta", text
            await asyncio.sleep(0)  # let LiteLLM's failure callbacks flush into the trace
        except Exception as exc:  # every failure mode ends here, never as an unhandled 500
            await asyncio.sleep(0)
            latency = int((time.perf_counter() - started) * 1000)
            err = classify_error(exc)
            attempts = self._attempts(trace, served=None, cache_hit=False)
            for a in attempts:
                self._record_failure(a)
            if err.code in ("context_too_long", "bad_request"):
                code, message = err.code, err.message
            elif attempts and all(a.outcome == "failed" for a in attempts):
                code = "all_keys_failed"
                message = "Every API key in the pool failed for this request."
            elif not attempts:
                code = err.code if err.code == "all_keys_cooling_down" else "all_keys_failed"
                message = "All API keys are cooling down after recent failures. Please retry in a few seconds."
            else:
                code, message = err.code, err.message
            raise GatewayError(code, message, attempts, latency) from exc
        finally:
            TRACE.reset(token)

        latency = int((time.perf_counter() - started) * 1000)
        hidden = getattr(stream, "_hidden_params", None) or {}
        served_id = hidden.get("model_id")
        cache_hit = bool(hidden.get("cache_hit")) or (bool(trace.cache_lookups) and trace.cache_lookups[-1])
        attempts = self._attempts(trace, served=served_id, cache_hit=cache_hit)
        for a in attempts:
            if a.outcome == "failed":
                self._record_failure(a)
        dep = self.deployments.get(served_id or "")
        content = "".join(parts)

        estimated = False
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        if not prompt_tokens or completion_tokens is None:
            estimated = True
            prompt_tokens = litellm.token_counter(model="gpt-4o", messages=messages)
            completion_tokens = litellm.token_counter(model="gpt-4o", text=content, count_response_tokens=True)

        if dep is not None and not cache_hit:
            s = self.stats[dep.id]
            s.requests += 1
            s.successes += 1
            s.last_latency_ms = latency

        model_name = (model_reported or (dep.model_name if dep else "unknown")).split("/")[-1]
        return_value = ChatResult(
            content=content,
            source=LLM_CACHE if cache_hit else LIVE_INFERENCE,
            model=model_name,
            deployment_id=dep.id if dep else None,
            deployment_label=dep.label if dep else None,
            latency_ms=latency,
            ttft_ms=int((first_token_at - started) * 1000) if first_token_at else None,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            tokens_estimated=estimated,
            attempts=attempts,
        )
        yield "done", return_value

    # ------------------------------------------------------------------ helpers
    def _attempts(self, trace: RequestTrace, *, served: str | None, cache_hit: bool) -> list[Attempt]:
        seen: list[str] = []
        for dep_id in trace.attempts:
            if dep_id not in seen:
                seen.append(dep_id)
        out: list[Attempt] = []
        for dep_id in seen:
            dep = self.deployments.get(dep_id)
            label = dep.label if dep else dep_id
            model = dep.model_name if dep else ""
            if dep_id == served:
                out.append(Attempt(dep_id, label, model, "cache" if cache_hit else "served"))
                continue
            err = trace.failures.get(dep_id)
            if err is None and dep_id in trace.faults:
                err = AttemptError(trace.faults[dep_id], None, FAULT_TYPES[trace.faults[dep_id]])
            out.append(
                Attempt(
                    dep_id,
                    label,
                    model,
                    "failed",
                    error_code=err.code if err else "provider_error",
                    error_message=err.message if err else "Request failed",
                )
            )
        return out

    def _record_failure(self, attempt: Attempt) -> None:
        if attempt.outcome != "failed" or attempt.deployment_id not in self.stats:
            return
        s = self.stats[attempt.deployment_id]
        s.requests += 1
        s.failures += 1
        s.last_error = f"{attempt.error_code}: {attempt.error_message}"
        s.last_error_at = time.time()


def attempts_to_json(attempts: list[Attempt]) -> str:
    return json.dumps([asdict(a) for a in attempts])
