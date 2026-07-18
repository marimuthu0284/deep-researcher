"""The LangGraph shared state object.

Agents mutate this typed dict as they run. `analyses` uses an additive reducer
so the parallel per-article map branches accumulate their results without
clobbering each other; everything else defaults to last-value semantics.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .models import (
    ArticleAnalysis,
    ArticleBundle,
    JudgedState,
    SearchPlan,
)


class ResearchState(TypedDict, total=False):
    # --- inputs ---
    topic: str
    filters: str  # raw human filter string
    recipient_email: str | None

    # --- planning / retrieval ---
    search_plan: SearchPlan
    raw_docs: list[ArticleBundle]
    bundles: list[ArticleBundle]  # top-N after triage

    # --- debate (map) ---
    # Additive reducer: parallel debate_article branches each append one analysis.
    analyses: Annotated[list[ArticleAnalysis], operator.add]

    # --- reduce ---
    judged: JudgedState
    report_md: str
    report_html: str
    delivery_status: dict[str, Any]

    # --- UI / telemetry ---
    # Additive so concurrent branches can each record their status events.
    status_log: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    # Per-node timing entries ({node, seconds, ts}); additive across branches.
    metrics: Annotated[list[dict[str, Any]], operator.add]
