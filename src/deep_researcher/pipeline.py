"""High-level orchestration: run the graph, stream state, cache the result.

`arun` streams the full state object after every superstep (stream_mode
"values"), which is exactly the "watch the typed state mutate" demo. It caches
the completed run keyed on (topic, filters) and can serve a cached run back
without touching the network.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Awaitable, Callable

from .cache import load_run, save_run
from .config import get_settings
from .graph import build_graph
from .models import ArticleBundle
from .state import ResearchState

EventCB = Callable[[ResearchState], Awaitable[None] | None]
# Given the triaged bundles, return the approved subset (or None to accept all).
ApproveCB = Callable[[list[ArticleBundle]], "Awaitable[list[ArticleBundle] | None] | list[ArticleBundle] | None"]


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        await value


async def _maybe_await_value(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _make_checkpointer():
    """Return an AsyncSqliteSaver context manager, or None if unavailable."""
    settings = get_settings()
    settings.ensure_dirs()
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        return AsyncSqliteSaver.from_conn_string(settings.checkpoint_db)
    except Exception:
        return None


async def arun(
    topic: str,
    filters: str = "",
    recipient_email: str | None = None,
    use_cache: bool = True,
    on_event: EventCB | None = None,
    on_approve: ApproveCB | None = None,
) -> ResearchState:
    settings = get_settings()
    settings.enable_tracing()  # opt-in LangSmith trace when a key is configured

    if use_cache:
        cached = load_run(topic, filters)
        if cached:
            from .cache import hydrate_state

            state = hydrate_state(cached)
            if on_event:
                await _maybe_await(on_event(state))
            return state

    inputs: ResearchState = {
        "topic": topic,
        "filters": filters,
        "recipient_email": recipient_email,
        "analyses": [],
        "status_log": [],
        "errors": [],
        "metrics": [],
    }
    # Unique thread per invocation so independent runs never resume/merge from a
    # prior checkpoint (the checkpointer still enables within-run HITL/resume).
    config = {
        "configurable": {"thread_id": f"{topic}:{filters}:{uuid.uuid4().hex[:8]}"},
        "recursion_limit": 50,
    }

    cm = await _make_checkpointer()
    final_state: ResearchState = dict(inputs)  # type: ignore[assignment]

    async def _stream(graph, data):
        nonlocal final_state
        async for chunk in graph.astream(data, config, stream_mode="values"):
            final_state = chunk  # type: ignore[assignment]
            if on_event:
                await _maybe_await(on_event(chunk))

    if on_approve is not None and cm is not None:
        # Human-in-the-loop: pause after triage, let the human edit the article
        # set, then resume. Requires a checkpointer to persist the paused state.
        async with cm as saver:
            graph = build_graph(saver, interrupt_before=["gate"])
            await _stream(graph, inputs)  # runs up to the interrupt before `gate`
            snapshot = await graph.aget_state(config)
            bundles = snapshot.values.get("bundles", [])
            approved = await _maybe_await_value(on_approve(bundles))
            if approved is not None:
                await graph.aupdate_state(config, {"bundles": list(approved)})
            await _stream(graph, None)  # resume through debate -> report -> dispatch
    elif cm is not None:
        async with cm as saver:
            await _stream(build_graph(saver), inputs)
    else:
        if on_approve is not None:
            final_state.setdefault("errors", []).append(
                "approval requested but no checkpointer available; ran without it"
            )
        await _stream(build_graph(None), inputs)

    try:
        save_run(topic, filters, final_state)
    except Exception:
        pass
    return final_state


def run(
    topic: str,
    filters: str = "",
    recipient_email: str | None = None,
    use_cache: bool = True,
) -> ResearchState:
    """Synchronous convenience wrapper."""
    return asyncio.run(
        arun(topic, filters, recipient_email, use_cache=use_cache)
    )
