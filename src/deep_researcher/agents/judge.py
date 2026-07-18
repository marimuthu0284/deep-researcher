"""Agent 7: Judge / Synthesis (the reduce step).

Per article: resolves contradictions, writes a calibrated position + dissent
note, and scores corroboration + recency (the two rubric components the critic
did not own). Code then computes the confidence total from the full rubric.
Across articles: builds the cross-source synthesis and aggregates finding-level
confidence via noisy-OR.
"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from ..llm import structured
from ..models import (
    ArticleAnalysis,
    ArticleVerdict,
    Disagreement,
    Finding,
    JudgedState,
    ScoredComponent,
)
from ..prompts import JUDGE_PER_ARTICLE, JUDGE_SYNTHESIS
from ..scoring import (
    aggregate_finding_confidence,
    band,
    band_short,
    compute_confidence,
    independence_damp,
)
from ..state import ResearchState
from .common import event, truncate

# Marker text used by the per-article fallback; also let us count degraded
# verdicts so the report can flag a degraded (LLM-unavailable) run.
_ARTICLE_FALLBACK_MARKER = (
    "Automated fallback: see per-perspective briefs; judge model unavailable."
)


class _JudgePerArticle(BaseModel):
    article_id: str
    resolved_position: str
    dissent_note: str | None = None
    corroboration: ScoredComponent
    recency_relevance: ScoredComponent


class _SynthFinding(BaseModel):
    statement: str
    supporting_article_ids: list[str] = Field(default_factory=list)
    single_thread: bool = False


class _SynthDisagreement(BaseModel):
    topic: str
    summary: str
    article_ids: list[str] = Field(default_factory=list)


class _Synthesis(BaseModel):
    executive_summary: str = ""
    trajectory: str = ""
    findings: list[_SynthFinding] = Field(default_factory=list)
    disagreements: list[_SynthDisagreement] = Field(default_factory=list)


async def judge(state: ResearchState) -> dict:
    analyses: list[ArticleAnalysis] = list(state.get("analyses", []))
    filters = state.get("filters", "")
    topic = state["topic"]

    corpus_titles = "\n".join(
        f"- {a.article_id}: {a.bundle.title[:100]} ({a.bundle.source_name})"
        for a in analyses
    )

    judge_errors: list[str] = []
    verdicts = await asyncio.gather(
        *[_judge_article(a, filters, corpus_titles, judge_errors) for a in analyses]
    )
    verdicts = [v for v in verdicts if v is not None]
    degraded_articles = sum(
        1 for v in verdicts if v.resolved_position == _ARTICLE_FALLBACK_MARKER
    )

    judged = await _synthesize(topic, filters, verdicts, analyses, judge_errors)
    judged.degraded_articles = degraded_articles
    # If every article's verdict fell back, treat the whole synthesis as degraded
    # even if the synthesis call itself happened to return something.
    if verdicts and degraded_articles == len(verdicts):
        judged.synthesis_degraded = True

    return {
        "judged": judged,
        "errors": judge_errors,
        "status_log": [
            event(
                "Judge",
                f"scored {len(verdicts)} articles; "
                f"{len(judged.findings)} findings, "
                f"{len(judged.disagreements)} open disagreements",
            )
        ],
    }


async def _judge_article(
    analysis: ArticleAnalysis,
    filters: str,
    corpus_titles: str,
    judge_errors: list[str],
) -> ArticleVerdict | None:
    bundle = analysis.bundle
    critique = analysis.critique
    try:
        jr = await structured(
            "judge",
            _JudgePerArticle,
            user="Resolve this article's debate and produce the JSON.",
            system=JUDGE_PER_ARTICLE.format(
                filters=filters,
                article_id=bundle.article_id,
                title=bundle.title,
                source_type=bundle.source_type,
                published_at=bundle.published_at,
                syndication_count=bundle.syndication_count,
                critique=truncate(json.dumps(critique.model_dump(), default=str), 3000),
                advocate=truncate(json.dumps(analysis.advocate.model_dump(), default=str), 2000),
                skeptic=truncate(json.dumps(analysis.skeptic.model_dump(), default=str), 2000),
                corpus_titles=corpus_titles,
            ),
        )
        corroboration = jr.corroboration.score
        recency = jr.recency_relevance.score
        resolved = jr.resolved_position
        dissent = jr.dissent_note
    except Exception as exc:  # noqa: BLE001
        # Fallback: derive components from critique-only signals.
        judge_errors.append(f"judge per-article ({bundle.article_id}): {exc}")
        corroboration = min(10.0, 3.0 + 2.0 * (bundle.syndication_count - 1))
        recency = 6.0
        resolved = _ARTICLE_FALLBACK_MARKER
        dissent = None

    breakdown = {
        "source_credibility": critique.source_credibility.score,
        "evidence_strength": critique.evidence_strength.score,
        "corroboration": corroboration,
        "internal_consistency": critique.internal_consistency.score,
        "recency_relevance": recency,
    }
    confidence = compute_confidence(breakdown)

    return ArticleVerdict(
        article_id=bundle.article_id,
        confidence_score=confidence,
        score_breakdown=breakdown,
        band=band(confidence),
        resolved_position=resolved,
        dissent_note=dissent,
    )


async def _synthesize(
    topic: str,
    filters: str,
    verdicts: list[ArticleVerdict],
    analyses: list[ArticleAnalysis],
    judge_errors: list[str],
) -> JudgedState:
    verdict_blob = json.dumps(
        [
            {
                "article_id": v.article_id,
                "confidence": v.confidence_score,
                "position": v.resolved_position,
            }
            for v in verdicts
        ],
        default=str,
    )
    synthesis_degraded = False
    try:
        synth = await structured(
            "judge",
            _Synthesis,
            user="Write the cross-source synthesis and produce the JSON.",
            system=JUDGE_SYNTHESIS.format(
                topic=topic, filters=filters, verdicts=truncate(verdict_blob, 6000)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        judge_errors.append(f"judge synthesis: {exc}")
        synthesis_degraded = True
        synth = _Synthesis(
            executive_summary=(
                f"Synthesis unavailable: the judge model could not be reached, so "
                f"the {len(verdicts)} article(s) below are shown without a "
                f"cross-source synthesis for '{topic}'. Re-run when the LLM "
                f"gateway is available for calibrated findings."
            ),
            findings=[
                _SynthFinding(
                    statement=v.resolved_position[:200],
                    supporting_article_ids=[v.article_id],
                    single_thread=True,
                )
                for v in verdicts[:5]
            ],
        )

    # Aggregate finding confidence via noisy-OR over supporting articles.
    verdict_by_id = {v.article_id: v for v in verdicts}
    damp_by_id = {
        a.article_id: independence_damp(a.bundle.syndication_count) for a in analyses
    }

    findings: list[Finding] = []
    for f in synth.findings:
        supporting = [
            (verdict_by_id[aid].confidence_score, damp_by_id.get(aid, 1.0))
            for aid in f.supporting_article_ids
            if aid in verdict_by_id
        ]
        conf = aggregate_finding_confidence(supporting)
        distinct_origins = sum(
            1 for aid in f.supporting_article_ids if damp_by_id.get(aid, 1.0) == 1.0
        )
        findings.append(
            Finding(
                statement=f.statement,
                confidence=conf,
                band=band_short(conf),
                supporting_article_ids=f.supporting_article_ids,
                single_thread=f.single_thread or distinct_origins <= 1,
            )
        )
    findings.sort(key=lambda x: x.confidence, reverse=True)

    disagreements = [
        Disagreement(topic=d.topic, summary=d.summary, article_ids=d.article_ids)
        for d in synth.disagreements
    ]

    return JudgedState(
        verdicts=sorted(verdicts, key=lambda v: v.confidence_score, reverse=True),
        findings=findings,
        disagreements=disagreements,
        executive_summary=synth.executive_summary,
        trajectory=synth.trajectory,
        synthesis_degraded=synthesis_degraded,
    )
