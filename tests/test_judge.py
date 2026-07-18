"""Regression tests for the Judge node.

The Judge previously called `structured()` without the required `user`
argument, which raised a TypeError that was silently swallowed into the
fallback path (every report said "judge model unavailable"). These tests use a
fake `structured` whose `user` parameter is REQUIRED, so any regression to a
missing-argument call fails loudly instead of degrading silently.
"""

from __future__ import annotations

import pytest

import deep_researcher.agents.judge as J
from deep_researcher.agents.judge import _JudgePerArticle, _Synthesis, _SynthFinding
from deep_researcher.models import (
    ArticleAnalysis,
    ArticleBundle,
    Chunk,
    Claim,
    CritiqueReport,
    PerspectiveBrief,
    ScoredComponent,
)

FALLBACK_MARKER = "judge model unavailable"


def _sc(v):
    return ScoredComponent(score=v, justification="fixture")


def _analysis(aid):
    bundle = ArticleBundle(
        article_id=aid, title=aid, url=f"https://x/{aid}", source_name="x",
        source_type="peer_reviewed", chunks=[Chunk(chunk_id=f"{aid}-c0", text="t")],
    )
    return ArticleAnalysis(
        article_id=aid,
        bundle=bundle,
        advocate=PerspectiveBrief(article_id=aid, stance="advocate",
                                  claims=[Claim(claim_id="a", text="c", strength=3)]),
        skeptic=PerspectiveBrief(article_id=aid, stance="skeptic",
                                 claims=[Claim(claim_id="s", text="c", strength=3)]),
        critique=CritiqueReport(
            article_id=aid,
            source_credibility=_sc(8), evidence_strength=_sc(7),
            internal_consistency=_sc(8),
        ),
    )


async def fake_structured(role, schema, system, user, temperature=0.2):
    # `user` is required here on purpose: a missing-user regression -> TypeError.
    assert user, "structured() must receive a non-empty user message"
    if schema is _JudgePerArticle:
        return _JudgePerArticle(
            article_id="x", resolved_position="A calibrated resolution.",
            dissent_note="the skeptic had a point", corroboration=_sc(7),
            recency_relevance=_sc(8),
        )
    if schema is _Synthesis:
        return _Synthesis(
            executive_summary="Real synthesis of the corpus.",
            trajectory="improving",
            findings=[_SynthFinding(statement="f", supporting_article_ids=["a0", "a1"])],
            disagreements=[],
        )
    raise AssertionError(schema)


@pytest.mark.asyncio
async def test_judge_uses_model_not_fallback(monkeypatch):
    monkeypatch.setattr(J, "structured", fake_structured)
    state = {"topic": "T", "filters": "", "analyses": [_analysis("a0"), _analysis("a1")]}

    out = await J.judge(state)
    judged = out["judged"]

    assert judged.executive_summary == "Real synthesis of the corpus."
    assert judged.verdicts, "expected verdicts"
    for v in judged.verdicts:
        assert FALLBACK_MARKER not in v.resolved_position
        assert v.dissent_note == "the skeptic had a point"
    # Two independent supporting articles -> aggregated finding confidence.
    assert judged.findings[0].confidence > 0
    # A healthy run must not be flagged degraded.
    assert judged.synthesis_degraded is False
    assert judged.degraded_articles == 0


async def all_models_fail(role, schema, system, user, temperature=0.2):
    assert user, "structured() must receive a non-empty user message"
    raise RuntimeError("all models failed (simulated outage)")


@pytest.mark.asyncio
async def test_judge_flags_degraded_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(J, "structured", all_models_fail)
    state = {"topic": "T", "filters": "", "analyses": [_analysis("a0"), _analysis("a1")]}

    out = await J.judge(state)
    judged = out["judged"]

    # Every article fell back and synthesis failed -> the run is flagged degraded
    # so the report can warn instead of presenting fallbacks as real findings.
    assert judged.synthesis_degraded is True
    assert judged.degraded_articles == 2
    for v in judged.verdicts:
        assert FALLBACK_MARKER in v.resolved_position

    # The real cause must be surfaced, not just the generic fallback text -
    # a degraded run with no clue why (network error vs bad key vs rate
    # limit) is undebuggable from the report alone.
    assert out["errors"], "expected the real exception(s) in state['errors']"
    assert all("simulated outage" in e for e in out["errors"])
