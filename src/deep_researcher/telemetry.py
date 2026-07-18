"""Lightweight timing telemetry for graph nodes.

`timed` wraps an async node so every invocation records a {node, seconds}
entry into the additive `metrics` channel. This powers the run-timing panel in
the UI and the latency summary in the CLI, and complements LangSmith tracing
(enabled via config when a key is present).
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Awaitable, Callable

NodeFn = Callable[[dict], Awaitable[dict]]


def timed(name: str) -> Callable[[NodeFn], NodeFn]:
    """Decorator: measure wall-clock time of a node and append a metrics entry."""

    def wrapper(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def inner(state: dict) -> dict:
            start = time.perf_counter()
            result = await fn(state)
            elapsed = time.perf_counter() - start
            entry = {"node": name, "seconds": round(elapsed, 3), "ts": time.time()}
            if isinstance(result, dict):
                merged = dict(result)
                merged["metrics"] = [*merged.get("metrics", []), entry]
                return merged
            return {"metrics": [entry]}

        return inner

    return wrapper


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw per-invocation metrics into per-node and total stats."""
    per_node: dict[str, dict[str, float]] = {}
    total = 0.0
    for m in metrics or []:
        node = m.get("node", "?")
        secs = float(m.get("seconds", 0.0))
        total += secs
        agg = per_node.setdefault(node, {"calls": 0, "total_seconds": 0.0})
        agg["calls"] += 1
        agg["total_seconds"] = round(agg["total_seconds"] + secs, 3)
    return {"per_node": per_node, "total_seconds": round(total, 3)}
