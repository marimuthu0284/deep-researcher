"""End-to-end graph smoke test that runs fully offline.

Monkeypatches the LLM structured-output calls and the network search/extract so
the Send fan-out, additive reducer, judge reduce, report render, and dispatcher
all exercise without any API keys.
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
from deep_researcher.graph import build_graph
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
            queries=[
                SearchQuery(query_str="q1", source_type="peer_reviewed"),
                SearchQuery(query_str="q2", source_type="news"),
            ],
            filter_spec=FilterSpec(),
        )
    if name == "_TriageResult":
        # Empty -> triage falls back to the deterministic heuristic selector.
        return _TriageResult(items=[])
    if name == "PerspectiveBrief":
        return PerspectiveBrief(
            article_id="x",
            stance="advocate",
            claims=[
                Claim(claim_id="c1", text="a claim", cited_chunk_ids=["x-c0"], strength=3)
            ],
        )
    if name == "CritiqueReport":
        return CritiqueReport(
            article_id="x",
            uncited_claims_rejected=[],
            source_credibility=_sc(8),
            evidence_strength=_sc(7),
            internal_consistency=_sc(8),
            evidence_sufficiency="sufficient",
        )
    if name == "_JudgePerArticle":
        return _JudgePerArticle(
            article_id="x",
            resolved_position="position",
            dissent_note="dissent",
            corroboration=_sc(6),
            recency_relevance=_sc(7),
        )
    if name == "_Synthesis":
        return _Synthesis(
            executive_summary="summary",
            trajectory="stable",
            findings=[
                _SynthFinding(statement="a finding", supporting_article_ids=[], single_thread=True)
            ],
            disagreements=[],
        )
    raise AssertionError(f"unexpected schema {name}")


def fake_run_query(query, filter_spec, settings=None):
    return [
        {
            "title": "Fake trial reports durable efficacy",
            "url": "https://example.org/trial",
            "source_name": "example.org",
            "source_type": "peer_reviewed",
            "snippet": "abstract",
            "full_text": None,
            "published_at": None,
            "citation_count": 12,
        },
        {
            "title": "News covers the trial",
            "url": "https://news.example.com/story",
            "source_name": "news.example.com",
            "source_type": "news",
            "snippet": "abstract",
            "full_text": None,
            "published_at": None,
            "citation_count": None,
        },
    ]


def fake_fetch_full_text(url, timeout=15):
    return " ".join(["evidence"] * 200)


@pytest.mark.asyncio
async def test_pipeline_runs_offline(monkeypatch, tmp_path):
    for mod in (qp_mod, triage_mod, persp_mod, ca_mod, judge_mod):
        monkeypatch.setattr(mod, "structured", fake_structured)
    monkeypatch.setattr(ret_mod, "run_query", fake_run_query)
    monkeypatch.setattr(ret_mod, "fetch_full_text", fake_fetch_full_text)

    from deep_researcher.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")

    graph = build_graph(None)
    inputs = {
        "topic": "Test Topic",
        "filters": "",
        "recipient_email": None,
        "analyses": [],
        "status_log": [],
        "errors": [],
    }
    final = None
    async for chunk in graph.astream(inputs, {"recursion_limit": 50}, stream_mode="values"):
        final = chunk

    assert final is not None
    assert final.get("report_html")
    assert final.get("report_md")
    assert len(final.get("analyses", [])) >= 1
    assert final.get("judged") is not None
    assert final["delivery_status"]["local_link"]
