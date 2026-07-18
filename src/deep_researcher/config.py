"""Runtime settings loaded from environment variables.

All external dependencies are optional at import time; missing keys only fail
when the corresponding feature is actually exercised. This keeps offline /
cached replay working without any credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # dotenv is convenient but not mandatory
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a soft dependency
    pass


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default OpenRouter model slugs per role. The design doc calls for deliberate
# heterogeneity: perspectives share one model (differences come from the prompt),
# the critic runs on a different family to decorrelate blind spots, and the
# judge gets the longest-context model.
DEFAULT_MODELS: dict[str, str] = {
    "planner": "meta-llama/llama-3.1-8b-instruct",
    "perspective": "meta-llama/llama-3.3-70b-instruct",
    "critic": "mistralai/mistral-small-3.2-24b-instruct",
    "judge": "google/gemini-2.5-flash",
    "fallback": "openai/gpt-4o-mini",
}

_ENV_MODEL_KEYS = {
    "planner": "DR_MODEL_PLANNER",
    "perspective": "DR_MODEL_PERSPECTIVE",
    "critic": "DR_MODEL_CRITIC",
    "judge": "DR_MODEL_JUDGE",
    "fallback": "DR_MODEL_FALLBACK",
}

# Ordered provider fallback chain (design doc: Groq -> Gemini -> Mistral). With
# OpenRouter as the single gateway these are model slugs tried in order after a
# role's primary model fails.
DEFAULT_FALLBACK_CHAIN: list[str] = [
    "google/gemini-2.5-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-4o-mini",
]


def _resolve_models() -> dict[str, str]:
    resolved = dict(DEFAULT_MODELS)
    for role, env_key in _ENV_MODEL_KEYS.items():
        override = os.getenv(env_key)
        if override:
            resolved[role] = override
    return resolved


def _resolve_fallback_chain() -> list[str]:
    raw = os.getenv("DR_FALLBACK_CHAIN")
    if raw:
        chain = [m.strip() for m in raw.split(",") if m.strip()]
        if chain:
            return chain
    return list(DEFAULT_FALLBACK_CHAIN)


@dataclass
class Settings:
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    semantic_scholar_api_key: str | None = field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    resend_api_key: str | None = field(
        default_factory=lambda: os.getenv("RESEND_API_KEY")
    )
    resend_from: str = field(
        default_factory=lambda: os.getenv(
            "RESEND_FROM", "Deep Researcher <onboarding@resend.dev>"
        )
    )
    # Gmail SMTP (preferred when set): sends as a real Gmail address via an
    # App Password, so reports can be delivered from @gmail.com without
    # needing a verified custom domain (which Resend requires). Takes
    # priority over Resend when both are configured.
    gmail_sender_email: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_SENDER_EMAIL")
    )
    gmail_app_password: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD")
    )

    models: dict[str, str] = field(default_factory=_resolve_models)
    fallback_chain: list[str] = field(default_factory=_resolve_fallback_chain)

    # Observability (opt-in). When langsmith_api_key is set we enable LangChain
    # tracing so the run shows up as a live trace.
    langsmith_api_key: str | None = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "deep-researcher")
    )

    top_n: int = field(default_factory=lambda: int(os.getenv("DR_TOP_N", "8")))
    # 40s (not 15s): the planner/judge structured-output calls fall back to a
    # manual JSON round-trip on models without native tool-calling, which is
    # slow. A tight timeout was silently tripping the fallback plan (arXiv
    # preprints for a news topic) and "judge model unavailable" synthesis.
    api_timeout: int = field(default_factory=lambda: int(os.getenv("DR_API_TIMEOUT", "40")))
    # Cap on simultaneous in-flight LLM calls, so the parallel debate fan-out
    # stays under provider rate limits (e.g. Groq free tier ~30 req/min) and
    # under constrained networks (corporate proxies/TLS inspection have been
    # observed to drop connections - SSLEOFError - above ~4-6 concurrent
    # HTTPS calls to the same host).
    max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("DR_MAX_CONCURRENCY", "4"))
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DR_CACHE_DIR", "data/cache"))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DR_REPORTS_DIR", "data/reports"))
    )
    checkpoint_db: str = field(
        default_factory=lambda: os.getenv("DR_CHECKPOINT_DB", "data/checkpoints.sqlite")
    )

    def model_for(self, role: str) -> str:
        return self.models.get(role, self.models["fallback"])

    def candidate_models(self, role: str) -> list[str]:
        """Primary model for the role, followed by the fallback chain.

        De-duplicated while preserving order, so a model that is both a role
        primary and part of the chain is only tried once.
        """
        ordered = [self.model_for(role), *self.fallback_chain]
        seen: set[str] = set()
        result: list[str] = []
        for m in ordered:
            if m and m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def enable_tracing(self) -> bool:
        """Turn on LangChain/LangSmith tracing if a key is configured."""
        if not self.langsmith_api_key:
            return False
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", self.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)
        return True

    def require_openrouter(self) -> str:
        if not self.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to your .env "
                "(copy .env.example) to run live LLM calls."
            )
        return self.openrouter_api_key

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force a re-read of the environment (useful in tests / the UI)."""
    global _settings
    _settings = Settings()
    return _settings
