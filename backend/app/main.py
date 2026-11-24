"""FastAPI entrypoint: REST + SSE API for the chatbot."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.chat import ChatService
from app.config import Settings, get_settings
from app.db import Database
from app.llm.gateway import FAULT_TYPES, LLMGateway

log = logging.getLogger("echo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
VERSION = "1.0.0"


# --------------------------------------------------------------------------- schemas
class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    system_prompt: str | None = None


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    system_prompt: str | None = None


class MessageIn(BaseModel):
    content: str = Field(min_length=1)
    bypass_cache: bool = False
    # Optional fault injection for the demo: {"gemini-1": "rate_limit"}
    simulate_failures: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_path)
        await db.connect()
        pool = settings.build_pool()
        gateway = LLMGateway(settings, pool)
        app.state.settings = settings
        app.state.db = db
        app.state.gateway = gateway
        app.state.chat = ChatService(db, gateway, settings)
        log.info("Key pool: %s", ", ".join(f"{d.order}:{d.id}({d.model})" for d in gateway.pool))
        if gateway.mock_mode:
            log.warning("No API keys configured – running in MOCK mode")
        yield
        await db.close()

    app = FastAPI(title="Echo API", version=VERSION, lifespan=lifespan)
    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- errors
    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Invalid request.",
                                                                  "details": json.loads(json.dumps(exc.errors(), default=str))}})

    @app.exception_handler(HTTPException)
    async def http_handler(_: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail}, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "Something went wrong. Please retry."}})

    # ---------------------------------------------------------------- deps
    rate_buckets: dict[str, deque] = defaultdict(deque)

    def client_id(x_client_id: str | None = Header(default=None)) -> str:
        cid = (x_client_id or "").strip()
        if not (8 <= len(cid) <= 64) or not all(c.isalnum() or c in "-_" for c in cid):
            raise HTTPException(400, {"code": "missing_client_id", "message": "X-Client-Id header is required."})
        return cid

    def rate_limit(request: Request, cid: str = Depends(client_id)) -> str:
        limit = settings.rate_limit_per_minute
        if limit <= 0:
            return cid
        key = f"{cid}:{request.client.host if request.client else ''}"
        bucket, now = rate_buckets[key], time.monotonic()
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= limit:
            retry = int(60 - (now - bucket[0])) + 1
            raise HTTPException(429, {"code": "rate_limited", "message": f"Too many messages. Try again in {retry}s."},
                                headers={"Retry-After": str(retry)})
        bucket.append(now)
        return cid

    async def owned_session(session_id: str, request: Request, cid: str = Depends(client_id)) -> dict[str, Any]:
        session = await request.app.state.db.get_session(session_id, cid)
        if session is None:
            raise HTTPException(404, {"code": "session_not_found", "message": "Chat not found."})
        return session

    # ---------------------------------------------------------------- system
    @app.get("/")
    async def root():
        return {"name": "Echo API", "version": VERSION, "docs": "/docs", "health": "/api/health"}

    @app.get("/api/health")
    async def health(request: Request):
        gw: LLMGateway = request.app.state.gateway
        return {"status": "ok", "version": VERSION, "pool_size": len(gw.pool), "mock_mode": gw.mock_mode,
                "cache": settings.cache_type}

    @app.get("/api/config")
    async def config(request: Request):
        gw: LLMGateway = request.app.state.gateway
        return {
            "mock_mode": gw.mock_mode,
            "cache_type": settings.cache_type,
            "cache_ttl_seconds": settings.cache_ttl_seconds,
            "cooldown_seconds": settings.cooldown_seconds,
            "default_system_prompt": settings.default_system_prompt,
            "max_message_chars": settings.max_message_chars,
            "fault_injection": settings.enable_fault_injection,
            "fault_types": FAULT_TYPES,
        }

    @app.get("/api/pool")
    async def pool(request: Request):
        return {"deployments": await request.app.state.gateway.pool_status()}

    @app.get("/api/stats")
    async def stats(request: Request, cid: str = Depends(client_id)):
        return await request.app.state.db.stats(cid)

    # ---------------------------------------------------------------- sessions
    @app.get("/api/sessions")
    async def list_sessions(request: Request, cid: str = Depends(client_id)):
        return {"sessions": await request.app.state.db.list_sessions(cid)}

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: SessionCreate, request: Request, cid: str = Depends(client_id)):
        prompt = settings.default_system_prompt if body.system_prompt is None else body.system_prompt
        _check_prompt(prompt)
        return await request.app.state.db.create_session(cid, body.title or "New chat", prompt)

    @app.get("/api/sessions/{session_id}")
    async def get_session(request: Request, session: dict = Depends(owned_session)):
        return {**session, "messages": await request.app.state.db.get_messages(session["id"])}

    @app.patch("/api/sessions/{session_id}")
    async def update_session(body: SessionUpdate, request: Request, session: dict = Depends(owned_session)):
        if body.system_prompt is not None:
            _check_prompt(body.system_prompt)
        return await request.app.state.db.update_session(session["id"], session["client_id"], **body.model_dump())

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(request: Request, session: dict = Depends(owned_session)):
        await request.app.state.db.delete_session(session["id"], session["client_id"])

    def _check_prompt(prompt: str) -> None:
        if len(prompt) > settings.max_system_prompt_chars:
            raise HTTPException(422, {"code": "system_prompt_too_long",
                                      "message": f"System prompt is limited to {settings.max_system_prompt_chars} characters."})

    def _check_message(body: MessageIn) -> str:
        content = body.content.strip()
        if not content:
            raise HTTPException(422, {"code": "empty_message", "message": "Message is empty."})
        if len(content) > settings.max_message_chars:
            raise HTTPException(422, {"code": "message_too_long",
                                      "message": f"Messages are limited to {settings.max_message_chars} characters."})
        return content

    # ---------------------------------------------------------------- chat
    @app.post("/api/sessions/{session_id}/messages/stream")
    async def send_message_stream(body: MessageIn, request: Request, session: dict = Depends(owned_session),
                                  _: str = Depends(rate_limit)):
        """Server-Sent Events: start → delta* → done | error."""
        content = _check_message(body)
        chat: ChatService = request.app.state.chat
        queue = chat.start_turn(session, content, faults=body.simulate_failures, bypass_cache=body.bypass_cache)

        async def events():
            async for event, data in chat.drain(queue):
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/sessions/{session_id}/messages")
    async def send_message(body: MessageIn, request: Request, session: dict = Depends(owned_session),
                           _: str = Depends(rate_limit)):
        """Non-streaming variant (handy for curl / tests). Same pipeline, same cache."""
        content = _check_message(body)
        chat: ChatService = request.app.state.chat
        queue = chat.start_turn(session, content, faults=body.simulate_failures, bypass_cache=body.bypass_cache)
        async for event, data in chat.drain(queue):
            if event == "done":
                return data
            if event == "error":
                return JSONResponse(status_code=503, content={"error": data})
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "No response."}})

    return app


app = create_app()
