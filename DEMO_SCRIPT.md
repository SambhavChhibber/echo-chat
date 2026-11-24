# 2–3 minute demo script

Before recording:

- Open the live frontend in a fresh window.
- Make sure the backend is awake (open it once).
- Clear any simulated failures.

**0:00 – Intro (15 s)**
"This is a full-stack chatbot. React frontend on Vercel, FastAPI and SQLite on Render, with LiteLLM routing across a pool of API keys." Point at the Key pool panel on the right: "Here are my keys in failover order. Each one is healthy."

**0:15 – Multi-turn and metadata (35 s)**
1. Ask: *"My name is Priya and I'm learning FastAPI. Suggest a first project."*
2. Point at the chips under the reply: **LIVE_INFERENCE**, the model name, latency in ms, and tokens in/out for this request only.
3. Ask: *"What's my name, and what am I learning?"* It remembers, because history is stored per session in SQLite.
4. Note that the input tokens grew (history is sent), but each request reports only its own tokens.

**0:50 – Caching (30 s)**
1. Click **New chat** and ask the exact first question again.
2. The reply comes back as **LLM_CACHE** in a few milliseconds, marked *not billed*.
3. "LiteLLM's built-in disk cache. Its key includes the conversation, so a different history gets a fresh answer."
4. Optionally click **Regenerate (skip cache)** to force live inference.

**1:20 – Failover (40 s)**
1. In Key pool, set key 1 to **Simulate HTTP 429 · rate limited**.
2. Ask a new question. It still answers: the failover chip shows `key 1 ✕ 429 → key 2 ✓`, and the model chip shows the backup model.
3. Key 1 now shows **cooling 30s**. Ask another question: it goes straight to key 2, with no wasted call. That's the circuit breaker.
4. "The fault is injected inside LiteLLM's provider path, so it's handled exactly like a real 429."

**2:00 – Total outage is graceful (25 s)**
1. Set every key to fail (e.g. 500), then ask something.
2. No broken screen: a clear message, the trail of what was tried, and a **Retry** button.
3. Clear the failures, click Retry, and it answers.

**2:25 – Wrap (15 s)**
Briefly show the repo: hash-pinned LiteLLM 1.106.2 (after this year's PyPI compromises), 14 tests in CI, and the README explaining each decision.
