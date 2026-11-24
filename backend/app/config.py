"""Application settings and key-pool configuration.

All configuration comes from environment variables (or a local `.env` file),
so the same image runs locally, in CI and in production.

The key pool can be described in two ways:

1. Provider shorthand (easiest on hosting dashboards)::

       GEMINI_API_KEYS=key-a,key-b
       GROQ_API_KEYS=key-c
       PROVIDER_ORDER=gemini,groq          # failover order across providers

2. Full JSON (any LiteLLM model string, explicit order and labels)::

       LLM_POOL='[{"label": "Gemini A", "model": "gemini/gemini-3.5-flash-lite", "api_key": "..."},
                  {"label": "Groq",     "model": "groq/llama-3.3-70b-versatile", "api_key": "..."}]'

If no keys are configured the app starts in MOCK mode with two simulated
deployments, so the UI, caching and failover can be exercised without keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider shorthand: env prefix -> default LiteLLM model + display name.
PROVIDERS: dict[str, dict[str, str]] = {
    "gemini": {"default_model": "gemini/gemini-3.5-flash-lite", "label": "Gemini"},
    "groq": {"default_model": "groq/llama-3.3-70b-versatile", "label": "Groq"},
    "openrouter": {"default_model": "openrouter/openai/gpt-oss-20b:free", "label": "OpenRouter"},
    "openai": {"default_model": "openai/gpt-4o-mini", "label": "OpenAI"},
    "anthropic": {"default_model": "anthropic/claude-haiku-4-5", "label": "Anthropic"},
    "mistral": {"default_model": "mistral/mistral-small-latest", "label": "Mistral"},
}


@dataclass(frozen=True)
class Deployment:
    """One API key on one model: a single entry in the failover chain."""

    id: str  # stable, URL-safe id, e.g. "gemini-1"
    label: str  # human readable, e.g. "Gemini · key 1"
    provider: str
    model: str  # LiteLLM model string, e.g. "gemini/gemini-3.5-flash-lite"
    api_key: str = field(repr=False)
    order: int = 1  # 1 = tried first
    api_base: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    mock_response: str | None = None  # only set in MOCK mode

    @property
    def key_hint(self) -> str:
        """Last four characters of the key, never the key itself."""
        if self.mock_response is not None:
            return "mock"
        return f"…{self.api_key[-4:]}" if len(self.api_key) >= 8 else "…"

    @property
    def model_name(self) -> str:
        """Model name without the provider prefix, for display."""
        return self.model.split("/", 1)[1] if "/" in self.model else self.model


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Echo"
    environment: str = Field(default="development", alias="APP_ENV")

    # Comma-separated list of allowed browser origins ("*" for any).
    cors_origins: str = "*"

    database_path: str = "./data/chat.db"

    # LiteLLM response cache
    cache_type: str = "disk"  # "disk" (persists across restarts) or "local" (in-memory)
    cache_dir: str = "./data/llm_cache"
    cache_ttl_seconds: int = 60 * 60 * 24

    # Gateway behaviour
    cooldown_seconds: float = 30.0  # how long a failing key is benched
    request_timeout_seconds: float = 45.0
    temperature: float = 0.7
    max_output_tokens: int = 1024
    max_context_tokens: int = 8000  # history budget sent to the model
    max_history_messages: int = 40

    # Input limits & abuse protection
    max_message_chars: int = 8000
    max_system_prompt_chars: int = 4000
    rate_limit_per_minute: int = 30

    default_system_prompt: str = "You are a helpful, concise assistant. Use Markdown when it helps readability."

    # Lets the UI inject simulated provider failures to demo failover.
    enable_fault_injection: bool = True

    # Key pool (see module docstring)
    llm_pool: str = ""
    provider_order: str = "gemini,groq,openrouter,openai,anthropic,mistral"
    force_mock: bool = Field(default=False, alias="MOCK_MODE")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def build_pool(self) -> list[Deployment]:
        if self.force_mock:
            return mock_pool()
        pool = _pool_from_json(self.llm_pool) if self.llm_pool.strip() else _pool_from_provider_env(self.provider_order)
        return pool or mock_pool()


def _pool_from_json(raw: str) -> list[Deployment]:
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:  # fail loudly at boot, never at request time
        raise ValueError(f"LLM_POOL is not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError("LLM_POOL must be a JSON list")

    pool: list[Deployment] = []
    counters: dict[str, int] = {}
    for index, entry in enumerate(entries, start=1):
        model = str(entry["model"])
        provider = model.split("/", 1)[0] if "/" in model else "openai"
        counters[provider] = counters.get(provider, 0) + 1
        dep_id = entry.get("id") or f"{provider}-{counters[provider]}"
        label = entry.get("label") or f"{PROVIDERS.get(provider, {}).get('label', provider.title())} · key {counters[provider]}"
        known = {"id", "label", "model", "api_key", "api_base", "order"}
        pool.append(
            Deployment(
                id=_slug(dep_id),
                label=label,
                provider=provider,
                model=model,
                api_key=str(entry.get("api_key", "")),
                order=int(entry.get("order", index)),
                api_base=entry.get("api_base"),
                extra_params={k: v for k, v in entry.items() if k not in known},
            )
        )
    return sorted(pool, key=lambda d: d.order)


def _pool_from_provider_env(provider_order: str) -> list[Deployment]:
    pool: list[Deployment] = []
    order = 0
    for provider in [p.strip().lower() for p in provider_order.split(",") if p.strip()]:
        meta = PROVIDERS.get(provider)
        if meta is None:
            continue
        prefix = provider.upper()
        keys = os.getenv(f"{prefix}_API_KEYS") or os.getenv(f"{prefix}_API_KEY") or ""
        model = os.getenv(f"{prefix}_MODEL") or meta["default_model"]
        if "/" not in model:
            model = f"{provider}/{model}"
        for n, key in enumerate([k.strip() for k in keys.split(",") if k.strip()], start=1):
            order += 1
            pool.append(
                Deployment(
                    id=f"{provider}-{n}",
                    label=f"{meta['label']} · key {n}",
                    provider=provider,
                    model=model,
                    api_key=key,
                    order=order,
                )
            )
    return pool


def mock_pool() -> list[Deployment]:
    """Two simulated deployments so the app is fully usable without keys."""
    return [
        Deployment(
            id="mock-primary",
            label="Mock · primary key",
            provider="mock",
            model="openai/mock-model-alpha",
            api_key="sk-mock-primary",
            order=1,
            mock_response="__echo__",
        ),
        Deployment(
            id="mock-backup",
            label="Mock · backup key",
            provider="mock",
            model="openai/mock-model-beta",
            api_key="sk-mock-backup",
            order=2,
            mock_response="__echo__",
        ),
    ]


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in value.lower()).strip("-") or "key"


@lru_cache
def get_settings() -> Settings:
    return Settings()
