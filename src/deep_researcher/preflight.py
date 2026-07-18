"""Preflight / doctor: report which capabilities are live vs degraded.

Run `deep-researcher --doctor` before a demo to see, at a glance, what will run
for real and what will fall back. Nothing here makes network calls -- it only
inspects configured credentials and installed packages.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .config import get_settings


@dataclass
class Capability:
    name: str
    ok: bool
    detail: str


def _has_module(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def check() -> list[Capability]:
    settings = get_settings()
    caps: list[Capability] = []

    # LLM gateway (the only hard requirement for a live run).
    caps.append(
        Capability(
            "LLM gateway (OpenRouter)",
            bool(settings.openrouter_api_key),
            "OPENROUTER_API_KEY set" if settings.openrouter_api_key
            else "MISSING - live runs will fail; offline replay still works",
        )
    )

    # Search sources.
    caps.append(
        Capability(
            "Web search (Tavily)",
            bool(settings.tavily_api_key),
            "TAVILY_API_KEY set" if settings.tavily_api_key
            else "no key - web/report/blog queries skipped (academic + GDELT still work)",
        )
    )
    for label, mod in [("arXiv", "arxiv"), ("full-text extraction", "trafilatura")]:
        caps.append(
            Capability(
                f"Retrieval: {label}",
                _has_module(mod),
                "available (no key needed)" if _has_module(mod) else f"install '{mod}'",
            )
        )
    caps.append(
        Capability(
            "Retrieval: Semantic Scholar / GDELT / Crossref",
            True,
            "keyless HTTP APIs"
            + (" (+S2 key for higher limits)" if settings.semantic_scholar_api_key else ""),
        )
    )

    # Email.
    gmail_ready = bool(settings.gmail_sender_email and settings.gmail_app_password)
    caps.append(
        Capability(
            "Email (Gmail SMTP)",
            gmail_ready,
            f"from {settings.gmail_sender_email}" if gmail_ready
            else "not configured - set GMAIL_SENDER_EMAIL + GMAIL_APP_PASSWORD",
        )
    )
    caps.append(
        Capability(
            "Email (Resend, fallback)",
            bool(settings.resend_api_key),
            f"from {settings.resend_from}" if settings.resend_api_key
            else "no key - used only if Gmail SMTP isn't configured either",
        )
    )

    # Observability + persistence.
    caps.append(
        Capability(
            "Tracing (LangSmith)",
            bool(settings.langsmith_api_key),
            f"project '{settings.langsmith_project}'" if settings.langsmith_api_key
            else "no key - tracing disabled (optional)",
        )
    )
    caps.append(
        Capability(
            "Checkpointing / HITL (SQLite)",
            _has_module("aiosqlite"),
            "available" if _has_module("aiosqlite") else "install 'aiosqlite' for --approve/resume",
        )
    )

    # Config summary.
    caps.append(
        Capability(
            "Model routing",
            True,
            "; ".join(f"{r}={m}" for r, m in settings.models.items()),
        )
    )
    caps.append(
        Capability(
            "Concurrency / top-N",
            True,
            f"max_concurrency={settings.max_concurrency}, top_n={settings.top_n}",
        )
    )
    return caps


def can_run_live() -> bool:
    return bool(get_settings().openrouter_api_key)


def print_report() -> int:
    caps = check()
    print("Deep Researcher - preflight\n" + "=" * 30)
    for c in caps:
        mark = "OK " if c.ok else "-- "
        print(f"[{mark}] {c.name}: {c.detail}")
    print("=" * 30)
    if can_run_live():
        print("Ready for a live run.")
        return 0
    print("Offline only (no OPENROUTER_API_KEY). Use --replay after seeding a demo.")
    return 0
