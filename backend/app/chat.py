"""Chat turn orchestration: history → context window → gateway → persistence.

Each turn runs in its own background task that pushes events into a queue.
The HTTP response (SSE or JSON) just reads that queue, so if the browser
disconnects mid-answer the turn still completes, gets cached and is saved.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

import litellm

from app.config import Settings
from app.db import Database, now_iso
from app.llm.gateway import GatewayError, LLMGateway

FRIENDLY_ERRORS = {
    "all_keys_failed": "All model providers are unavailable right now. Your message was not lost — please retry.",
    "all_keys_cooling_down": "All API keys are cooling down after recent failures. Please retry in a few seconds.",
    "context_too_long": "This conversation is too long for the model. Start a new chat or shorten your message.",
    "bad_request": "The model rejected this request.",
}


def build_context(
    system_prompt: str, history: list[dict[str, Any]], new_message: str, settings: Settings
) -> tuple[list[dict[str, str]], int]:
    """System prompt + as much recent history as fits the token budget + the new message."""
    turns = [{"role": m["role"], "content": m["content"]} for m in history][-settings.max_history_messages :]
    system = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    user = [{"role": "user", "content": new_message}]

    def count(msgs: list[dict[str, str]]) -> int:
        return litellm.token_counter(model="gpt-4o", messages=msgs)

    while turns and count(system + turns + user) > settings.max_context_tokens:
        turns = turns[2:] if len(turns) >= 2 else []  # drop the oldest user/assistant pair
    # Never start the window with an orphan assistant reply.
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    return system + turns + user, len(turns)


class ChatService:
    def __init__(self, db: Database, gateway: LLMGateway, settings: Settings) -> None:
        self.db = db
        self.gateway = gateway
        self.settings = settings
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: set[asyncio.Task] = set()

    def start_turn(
        self,
        session: dict[str, Any],
        content: str,
        *,
        faults: dict[str, str] | None,
        bypass_cache: bool,
    ) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._run_turn(session, content, faults, bypass_cache, queue))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return queue

    async def _run_turn(
        self,
        session: dict[str, Any],
        content: str,
        faults: dict[str, str] | None,
        bypass_cache: bool,
        queue: asyncio.Queue,
    ) -> None:
        sid = session["id"]
        async with self._locks[sid]:  # one turn at a time per session keeps history ordered
            user_created_at = now_iso()
            try:
                history = await self.db.get_messages(sid)
                messages, context_count = build_context(session["system_prompt"], history, content, self.settings)
                await queue.put(("start", {"context_messages": context_count}))
                result = None
                async for kind, payload in self.gateway.stream_chat(messages, faults=faults, bypass_cache=bypass_cache):
                    if kind == "delta":
                        await queue.put(("delta", {"text": payload}))
                    else:
                        result = payload
                assert result is not None
                reply = result.to_dict()
                reply["attempts"] = [dict(a) for a in reply["attempts"]]
                reply["context_messages"] = context_count
                reply["user_created_at"] = user_created_at
                title = _title_from(content) if not history else None
                user_msg, asst_msg = await self.db.save_turn(sid, content, reply, title)
                await queue.put(("done", {"user_message": user_msg, "message": asst_msg, "session_title": title}))
            except GatewayError as err:
                await queue.put(
                    (
                        "error",
                        {
                            "code": err.code,
                            "message": FRIENDLY_ERRORS.get(err.code, err.message),
                            "retryable": err.code not in ("context_too_long", "bad_request"),
                            "latency_ms": err.latency_ms,
                            "attempts": [asdict(a) for a in err.attempts],
                        },
                    )
                )
            except Exception as exc:  # defensive: the client must always get a clean event
                await queue.put(
                    ("error", {"code": "internal_error", "message": "Something went wrong on our side. Please retry.",
                               "retryable": True, "attempts": [], "detail": type(exc).__name__})
                )
            finally:
                await queue.put(None)

    @staticmethod
    async def drain(queue: asyncio.Queue) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


def _title_from(text: str) -> str:
    line = " ".join(text.split())
    return (line[:60] + "…") if len(line) > 60 else (line or "New chat")
