"""Run-level caching + replay.

Full runs are cached keyed on (topic, filter_hash) so a rehearsed demo topic
skips the flaky network entirely, and a `--replay` path can stream the cached
run through the UI at realistic speed if wifi dies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import get_settings
from .models import (
    ArticleAnalysis,
    ArticleBundle,
    JudgedState,
    SearchPlan,
)
from .state import ResearchState


def filter_hash(topic: str, filters: str) -> str:
    return hashlib.sha1(f"{topic}||{filters}".encode()).hexdigest()[:16]


def run_cache_path(topic: str, filters: str) -> Path:
    settings = get_settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / f"run_{filter_hash(topic, filters)}.json"


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


def save_run(topic: str, filters: str, state: ResearchState) -> Path:
    payload = {
        "topic": topic,
        "filters": filters,
        "search_plan": _dump(state.get("search_plan")),
        "raw_count": len(state.get("raw_docs", [])),
        "bundles": _dump(state.get("bundles", [])),
        "analyses": _dump(state.get("analyses", [])),
        "judged": _dump(state.get("judged")),
        "report_md": state.get("report_md", ""),
        "report_html": state.get("report_html", ""),
        "delivery_status": state.get("delivery_status", {}),
        "status_log": state.get("status_log", []),
        "errors": state.get("errors", []),
        "metrics": state.get("metrics", []),
    }
    path = run_cache_path(topic, filters)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_run(topic: str, filters: str) -> dict | None:
    path = run_cache_path(topic, filters)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def hydrate_state(payload: dict) -> ResearchState:
    """Reconstruct a ResearchState (with model objects) from a cached run."""
    state: ResearchState = {
        "topic": payload.get("topic", ""),
        "filters": payload.get("filters", ""),
        "report_md": payload.get("report_md", ""),
        "report_html": payload.get("report_html", ""),
        "delivery_status": payload.get("delivery_status", {}),
        "status_log": payload.get("status_log", []),
        "errors": payload.get("errors", []),
        "metrics": payload.get("metrics", []),
    }
    if payload.get("search_plan"):
        state["search_plan"] = SearchPlan.model_validate(payload["search_plan"])
    if payload.get("bundles"):
        state["bundles"] = [ArticleBundle.model_validate(b) for b in payload["bundles"]]
    if payload.get("analyses"):
        state["analyses"] = [
            ArticleAnalysis.model_validate(a) for a in payload["analyses"]
        ]
    if payload.get("judged"):
        state["judged"] = JudgedState.model_validate(payload["judged"])
    return state
