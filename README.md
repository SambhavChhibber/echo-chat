# Echo: LLM chatbot with key failover and response caching

**Echo** is a full-stack chatbot (named for the cache that echoes back answers it has already given): **React (Vite + TypeScript)** frontend, **FastAPI** backend, **SQLite** storage, and a **LiteLLM Router** gateway that keeps answering when an API key fails and serves repeated questions from cache.

- **Live frontend:** `<add Vercel URL>`
- **Live backend:** `<add Render URL>` (API docs at `/docs`)
- **Demo video:** `<add link>`

## What you'll see

Every assistant reply shows its own metadata:

| Field | Example | Where it comes from |
|---|---|---|
| Resolution source | `LLM_CACHE` / `LIVE_INFERENCE` | LiteLLM's cache lookup, recorded per request |
| Model used | `gemini-3.5-flash-lite` | the model that actually answered (on a failover, the backup's model) |
| Latency | `812 ms` (hover: time to first token) | measured server-side around the LiteLLM call |
| Token usage | `412 in · 96 out` | the provider's `usage` for **this request only**; cache hits are marked *not billed* |
| Failover trail | `Gemini · key 1 ✕ 429 → Groq · key 1 ✓` | shown only when a key failed during this request |

Other features:

- Multi-turn chat with history stored per session in SQLite.
- Chats are scoped to an anonymous per-browser id.
- A system prompt per chat.
- Streaming replies.
- "Regenerate (skip cache)".
- A **Key pool** panel showing live key health, cooldown countdowns and cache stats.
- A **failure simulator**, so failover can be demonstrated on demand.

## Architecture

```
React (Vercel)                FastAPI (Render, Docker)
┌────────────┐  SSE / JSON   ┌──────────────────────────────────────────────┐
│ Chat UI    │ ────────────▶ │ routes (main.py)                              │
│ metadata   │               │   └▶ ChatService (chat.py)                    │
│ key pool   │ ◀──────────── │        history → token-budgeted context       │
└────────────┘   events      │        └▶ LLMGateway (llm/gateway.py)         │
                             │             LiteLLM Router, model group "chat"│
                             │             order 1: Gemini key 1  ─┐ 429/5xx │
                             │             order 2: Gemini key 2  ◀┘ timeout │
                             │             order 3: Groq key 1      → next   │
                             │             + LiteLLM disk cache              │
                             │        └▶ SQLite (db.py): sessions, messages  │
                             └──────────────────────────────────────────────┘
```

### Key pool and automatic failover

- Every API key is a LiteLLM Router deployment in one model group, `chat`, with an explicit `order`.
- The router tries order 1 first. On **HTTP 429, quota exhaustion, any 5xx, a timeout or an auth error**, it moves to the next order level automatically.
- `num_retries=0` and the provider SDK's `max_retries=0` are deliberate: the next key *is* the retry. Without them, the SDK would hammer a rate-limited key with backoff before failing over.
- A key that fails is put into **cooldown** (default 30 s), a circuit breaker. Later requests skip it instead of paying its latency again, and the UI shows the countdown.
- **Nothing can crash the UI:**
  - Every exception is classified into a stable code such as `rate_limited`, `quota_exceeded`, `provider_error`, `timeout` or `all_keys_failed`.
  - The client receives it as an `error` event or a JSON body, never as a raw stack trace.
  - The UI shows a friendly message, the failover trail and a Retry button.
  - A turn is saved only when it succeeds, so failures never leave a half-written history.
  - A React error boundary and a "server is waking up" screen cover the frontend side.

### Caching

- LiteLLM's built-in cache is enabled on the router (`cache_responses=True`).
  - The default is a **disk** cache (`diskcache`) with a 24 h TTL, so it survives restarts.
  - `CACHE_TYPE=local` switches it to in-memory.
- The cache key is the model **group** plus the exact messages. So:
  - A cached answer is reused no matter which key produced it.
  - The cache is conversation-aware: the same question with different history or a different system prompt is a different request, and gets a fresh answer.
  - The same opening question in a new chat is a cache hit (try it).
- **How `LLM_CACHE` is detected reliably:** a `ContextVar` request trace plus two of LiteLLM's own extension points:
  1. A `Cache` subclass that notes when a lookup hits.
  2. `CustomLogger` hooks that record which deployments were attempted and why they failed.

  The response's `_hidden_params["cache_hit"]` is also checked.
- On a cache hit the reply still reports the request's token counts, labelled *not billed* (the provider wasn't called). The Key pool panel shows the tokens the cache saved.
- All calls run in streaming mode, so the streaming and JSON endpoints share one cache.

### Failure simulator (for the demo)

- Pick "Simulate HTTP 429" (or quota, 500, 503, timeout) on any key in the **Key pool** panel. The next messages from *your* browser get that error from that key.
- The fault is injected inside LiteLLM's provider path (`async_pre_call_deployment_hook` → `mock_response=<real LiteLLM exception>`). So it travels the exact path a real provider error would: cooldown, fallback, logging.
- One realistic detail: a cached question is still answered even when key 1 is "down". The cache is checked before the provider is called, as it would be in production.
- Turn it off with `ENABLE_FAULT_INJECTION=false`.

### Other decisions

- **Context window:**
  - History is trimmed oldest-first to `MAX_CONTEXT_TOKENS`; the system prompt is always kept.
  - Each reply records how many history messages were sent.
- **Concurrency:**
  - One turn at a time per session (an `asyncio.Lock`), so history stays ordered.
  - Each turn runs in its own task, so if the browser disconnects mid-answer, the reply still finishes, is cached and is saved.
- **Abuse protection:** per-client rate limit, message and system-prompt length limits, and CORS locked to the frontend origin.
- **Supply-chain safety:**
  - LiteLLM's PyPI package was compromised twice in 2026 (1.82.7/1.82.8 in March, 1.107.0 in October). It is pinned to **1.106.2**, which the maintainers named as the last clean release.
  - The Docker image installs from a fully **hash-pinned** lock (`pip install --require-hashes`).
- **Gemini gotcha:** Gemini quotas are per Google Cloud *project*, not per key. Two keys from the same project hit 429 together, so use keys from different projects, or mix providers (Gemini + Groq).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + pool size |
| GET | `/api/config` | UI config (fault types, limits) |
| GET | `/api/pool` | key health, cooldowns, per-key stats (keys are never exposed, only `…last4`) |
| GET | `/api/stats` | cache hit rate, tokens saved, failovers (per client) |
| GET/POST | `/api/sessions` | list / create chats (`system_prompt` optional) |
| GET/PATCH/DELETE | `/api/sessions/{id}` | read with messages / update title or system prompt / delete |
| POST | `/api/sessions/{id}/messages/stream` | send a message (SSE events: `start`, `delta`, `done`, `error`) |
| POST | `/api/sessions/{id}/messages` | same, as one JSON response |

All requests send an `X-Client-Id` header (an anonymous UUID that the frontend keeps in localStorage).

```bash
curl -s -X POST $API/api/sessions -H 'X-Client-Id: demo-client-123' -H 'Content-Type: application/json' -d '{}'
curl -s -X POST $API/api/sessions/<id>/messages -H 'X-Client-Id: demo-client-123' \
     -H 'Content-Type: application/json' -d '{"content":"Hi!"}' | jq .message.meta
```

## Run locally

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt && pip install -r requirements-dev.txt
cp .env.example .env            # add your keys (or set MOCK_MODE=true)
uvicorn app.main:app --reload   # http://localhost:8000/docs
pytest -q                       # 14 tests: multi-turn, cache, failover per error type, all-keys-down, SSE

# frontend
cd frontend
npm install
cp .env.example .env            # VITE_API_URL=http://localhost:8000
npm run dev                     # http://localhost:5173
```

With no keys configured, the backend runs in **mock mode**: two simulated keys. You can still see caching, failover and the metadata end to end.

## Deploy

1. **Push to GitHub** (public repo).
2. **Backend on Render:**
   1. New → **Blueprint** → pick the repo. `render.yaml` creates a Docker web service from `backend/`.
   2. Set `GEMINI_API_KEYS` (comma-separated), `GROQ_API_KEYS`, and `CORS_ORIGINS` (your Vercel URL).
   3. Check that `https://<service>.onrender.com/api/health` responds.
3. **Frontend on Vercel:**
   1. New Project → import the repo → **Root directory `frontend`** (framework: Vite).
   2. Set the env var `VITE_API_URL=https://<service>.onrender.com`, then deploy.
   3. Put the Vercel URL into Render's `CORS_ORIGINS` and redeploy the backend.

Notes:

- The Render free tier sleeps after ~15 min idle. The UI shows "Waking up the server…" and retries automatically.
- Free-tier disks are ephemeral, so SQLite and the disk cache reset on redeploy. On a paid plan, attach a disk at `/app/data` for persistence; no code changes needed.

## Project layout

```
backend/
  app/main.py          FastAPI app, routes, validation, rate limit, error handlers
  app/chat.py          turn orchestration, context trimming, background task + event queue
  app/llm/gateway.py   LiteLLM Router, cache, failover trace, fault injection, pool health
  app/db.py            SQLite schema + queries (aiosqlite, WAL)
  app/config.py        settings + key-pool parsing
  tests/test_api.py    end-to-end tests on the mock pool
frontend/
  src/api.ts           typed client + SSE parser
  src/App.tsx          UI: chats, messages, metadata chips, key pool panel
render.yaml · docker-compose.yml · .github/workflows/ci.yml
```
