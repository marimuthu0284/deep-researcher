"""Agent 1: Query Planner. Reasoning-before-retrieval.

Decomposes the topic + filters into 4-8 faceted search queries and translates
the human filter string into machine parameters. Includes a fallback plan so
the pipeline still runs if the LLM/structured call fails.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..llm import structured
from ..models import FilterSpec, SearchPlan, SearchQuery
from ..prompts import QUERY_PLANNER
from ..state import ResearchState
from .common import event


async def query_planner(state: ResearchState) -> dict:
    topic = state["topic"]
    filters = state.get("filters", "")
    today = datetime.now(timezone.utc).date().isoformat()

    try:
        plan = await structured(
            "planner",
            SearchPlan,
            system=QUERY_PLANNER.format(topic=topic, filters=filters, today=today),
            user=f"TOPIC: {topic}\nFILTERS: {filters}\nProduce the SearchPlan.",
        )
        if not plan.queries:
            raise ValueError("empty plan")
    except Exception as exc:  # noqa: BLE001 - graceful degradation
        plan = _fallback_plan(topic, filters)
        return {
            "search_plan": plan,
            "status_log": [
                event(
                    "Query Planner",
                    f"used fallback plan ({len(plan.queries)} queries)",
                    error=str(exc),
                )
            ],
            "errors": [f"query_planner fallback: {exc}"],
        }

    return {
        "search_plan": plan,
        "status_log": [
            event(
                "Query Planner",
                f"planned {len(plan.queries)} queries across "
                f"{len({q.source_type for q in plan.queries})} source types",
            )
        ],
    }


_ACADEMIC_HINTS = (
    "peer-review",
    "peer reviewed",
    "peer-reviewed",
    "journal",
    "paper",
    "study",
    "studies",
    "academic",
    "preprint",
    "arxiv",
    "clinical",
    "trial",
    "scholar",
)
_NEWS_HINTS = ("news", "press", "media", "headline", "coverage", "report", "market")


def _fallback_plan(topic: str, filters: str) -> SearchPlan:
    """A deterministic minimal plan when the LLM is unavailable.

    Filter-aware on purpose: the old fallback always led with peer_reviewed
    queries, so a "major news" topic (e.g. "Dubai real estate") silently
    returned unrelated arXiv preprints. We now route by the filter intent so a
    degraded run still retrieves on-topic sources.
    """
    f = (filters or "").lower()
    wants_academic = any(h in f for h in _ACADEMIC_HINTS)
    wants_news = any(h in f for h in _NEWS_HINTS)
    # Default to the general web + news when the filter is silent or news-ish;
    # only lead with academic sources when the filter clearly asks for them.
    academic_only = wants_academic and not wants_news

    queries: list[SearchQuery] = []
    if academic_only:
        queries += [
            SearchQuery(query_str=topic, source_type="peer_reviewed", facet="core"),
            SearchQuery(
                query_str=f"{topic} outcomes evidence",
                source_type="peer_reviewed",
                facet="outcomes",
            ),
            SearchQuery(
                query_str=f"{topic} limitations", source_type="preprint", facet="mechanism"
            ),
        ]
    else:
        queries += [
            SearchQuery(query_str=f"{topic} latest news", source_type="news", facet="news"),
            SearchQuery(query_str=topic, source_type="report", facet="core"),
            SearchQuery(
                query_str=f"{topic} analysis outlook", source_type="report", facet="outlook"
            ),
        ]
        if wants_academic:
            queries.append(
                SearchQuery(query_str=topic, source_type="peer_reviewed", facet="research")
            )

    # Always include a query that seeks the critical / skeptical view.
    queries.append(
        SearchQuery(
            query_str=f"{topic} criticism risks problems skeptic",
            source_type="news" if not academic_only else "report",
            facet="criticism",
        )
    )
    return SearchPlan(queries=queries, filter_spec=FilterSpec(raw=filters))
