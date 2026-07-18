"""Human-in-the-loop approval test (offline).

Runs the graph with an approval callback that trims the article set, and
asserts the trim is honored (only the approved article is debated).
"""

from __future__ import annotations

import pytest

import deep_researcher.agents.critical_analysis as ca_mod
import deep_researcher.agents.judge as judge_mod
import deep_researcher.agents.perspective as persp_mod
import deep_researcher.agents.query_planner as qp_mod
import deep_researcher.agents.retriever as ret_mod
import deep_researcher.agents.triage as triage_mod
from deep_researcher.agents.judge import _JudgePerArticle, _Synthesis, _SynthFinding
from deep_researcher.agents.triage import _TriageResult
from deep_researcher.models import (
    Claim,
    CritiqueReport,
    FilterSpec,
    PerspectiveBrief,
    ScoredComponent,
    SearchPlan,
    SearchQuery,
)


def _sc(score: float) -> ScoredComponent:
    return ScoredComponent(score=score, justification="fixture")


async def fake_structured(role, schema, system, user, temperature=0.2):
    name = schema.__name__
    if name == "SearchPlan":
        return SearchPlan(
            queries=[SearchQuery(query_str="q", source_type="news")],
            filter_spec=FilterSpec(),
        )
    if name == "_TriageResult":
        return _TriageResult(items=[])
    if name == "PerspectiveBrief":
        return PerspectiveBrief(
            article_id="x",
            stance="advocate",
            claims=[Claim(claim_id="c1", text="c", cited_chunk_ids=["x-c0"], strength=3)],
        )
    if name == "CritiqueReport":
        return CritiqueReport(
            article_id="x",
            source_credibility=_sc(8),
            evidence_strength=_sc(7),
            internal_consistency=_sc(8),
            evidence_sufficiency="sufficient",
        )
    if name == "_JudgePerArticle":
        return _JudgePerArticle(
            article_id="x",
            resolved_position="p",
            dissent_note=None,
            corroboration=_sc(6),
            recency_relevance=_sc(7),
        )
    if name == "_Synthesis":
        return _Synthesis(
            executive_summary="s",
            findings=[_SynthFinding(statement="f", supporting_article_ids=[], single_thread=True)],
        )
    raise AssertionError(name)


def fake_run_query(query, filter_spec, settings=None):
    return [
        {"title": "One", "url": "https://a/1", "source_name": "a", "source_type": "news",
         "snippet": "s", "full_text": None, "published_at": None, "citation_count": None},
        {"title": "Two", "url": "https://b/2", "source_name": "b", "source_type": "news",
         "snippet": "s", "full_text": None, "published_at": None, "citation_count": None},
        {"title": "Three", "url": "https://c/3", "source_name": "c", "source_type": "news",
         "snippet": "s", "full_text": None, "published_at": None, "citation_count": None},
    ]


def fake_fetch_full_text(url, timeout=15):
    return " ".join(["evidence"] * 200)


@pytest.mark.asyncio
async def test_approval_trims_articles(monkeypatch, tmp_path):
    aiosqlite = pytest.importorskip("aiosqlite")  # noqa: F841

    for mod in (qp_mod, triage_mod, persp_mod, ca_mod, judge_mod):
        monkeypatch.setattr(mod, "structured", fake_structured)
    monkeypatch.setattr(ret_mod, "run_query", fake_run_query)
    monkeypatch.setattr(ret_mod, "fetch_full_text", fake_fetch_full_text)

    from deep_researcher.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "cp.sqlite"))

    from deep_researcher.pipeline import arun

    async def approve(bundles):
        # Keep only the first article.
        return bundles[:1]

    state = await arun(
        "Topic", "", use_cache=False, on_approve=approve
    )

    assert len(state.get("analyses", [])) == 1
    assert state.get("report_html")
