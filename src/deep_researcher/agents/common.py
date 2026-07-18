"""Shared helpers for agent nodes."""

from __future__ import annotations

import time
from typing import Any


def event(agent: str, message: str, **extra: Any) -> dict[str, Any]:
    """Build a status-log entry for the live UI board."""
    return {"ts": time.time(), "agent": agent, "message": message, **extra}


def truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"
