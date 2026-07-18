"""Regression tests for the Query Planner's degraded fallback plan.

The original fallback always led with peer_reviewed queries, so a "major news"
topic (e.g. "Dubai real estate") silently returned unrelated arXiv preprints.
These tests pin the filter-aware routing so a degraded run stays on-topic.
"""

from __future__ import annotations

from deep_researcher.agents.query_planner import _fallback_plan


def _types(plan):
    return {q.source_type for q in plan.queries}


def test_news_filter_avoids_academic_sources():
    plan = _fallback_plan("Dubai real estate", "major news, last 120 days")
    types = _types(plan)
    assert "news" in types
    # A news topic must NOT be routed to arXiv/Semantic Scholar in fallback.
    assert "peer_reviewed" not in types
    assert "preprint" not in types
    assert plan.queries, "fallback must still produce queries"


def test_empty_filter_defaults_to_web_and_news_not_arxiv():
    plan = _fallback_plan("Dubai real estate", "")
    types = _types(plan)
    assert "peer_reviewed" not in types
    assert "preprint" not in types
    assert {"news", "report"} & types


def test_academic_filter_uses_scholarly_sources():
    plan = _fallback_plan("GLP-1 receptor agonists", "peer-reviewed papers only")
    types = _types(plan)
    assert "peer_reviewed" in types


def test_mixed_filter_keeps_news_and_adds_research():
    plan = _fallback_plan("mRNA vaccines", "major news and peer-reviewed studies")
    types = _types(plan)
    assert "news" in types
    assert "peer_reviewed" in types


def test_always_includes_a_criticism_query():
    plan = _fallback_plan("Dubai real estate", "major news")
    facets = {q.facet for q in plan.queries}
    assert "criticism" in facets
