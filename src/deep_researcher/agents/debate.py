"""Per-article debate node (the map step).

Runs Advocate + Skeptic concurrently and blind to each other, then the Critical
Analyst. If the critique flags insufficient evidence, performs exactly one
bounded re-retrieval and re-runs the debate (the single dashed back-edge from
the workflow diagram, implemented as an internal loop for determinism).

This node is fanned out via Send(); its input state is the payload dict
{"article": ArticleBundle}. Its return appends one ArticleAnalysis through the
additive reducer on `analyses`.
"""

from __future__ import annotations

import asyncio

from ..models import (
    ArticleAnalysis,
    ArticleBundle,
    CritiqueReport,
    PerspectiveBrief,
    ScoredComponent,
)
from .common import event
from .critical_analysis import run_critical_analysis
from .perspective import run_advocate, run_skeptic
from .retriever import reretrieve_article


async def _debate_once(
    bundle: ArticleBundle,
) -> tuple[PerspectiveBrief, PerspectiveBrief, CritiqueReport]:
    advocate, skeptic = await asyncio.gather(
        run_advocate(bundle), run_skeptic(bundle)
    )
    critique = await run_critical_analysis(bundle, advocate, skeptic)
    return advocate, skeptic, critique


async def debate_article(state: dict) -> dict:
    bundle: ArticleBundle = state["article"]
    reretrieved = False
    events = []

    try:
        advocate, skeptic, critique = await _debate_once(bundle)

        if critique.evidence_sufficiency == "insufficient":
            events.append(
                event(
                    "Critical Analysis",
                    f"insufficient evidence for {bundle.article_id}; "
                    "triggering one bounded re-retrieval",
                    article_id=bundle.article_id,
                )
            )
            bundle = await reretrieve_article(bundle)
            reretrieved = True
            advocate, skeptic, critique = await _debate_once(bundle)

        analysis = ArticleAnalysis(
            article_id=bundle.article_id,
            bundle=bundle,
            advocate=advocate,
            skeptic=skeptic,
            critique=critique,
            reretrieved=reretrieved,
        )
        events.append(
            event(
                "Debate",
                f"{bundle.article_id}: {len(advocate.claims)} advocate / "
                f"{len(skeptic.claims)} skeptic claims, "
                f"{len(critique.uncited_claims_rejected)} rejected, "
                f"sufficiency={critique.evidence_sufficiency}",
                article_id=bundle.article_id,
            )
        )
        return {"analyses": [analysis], "status_log": events}

    except Exception as exc:  # noqa: BLE001 - never let one article kill the map
        analysis = _degraded_analysis(bundle, str(exc))
        return {
            "analyses": [analysis],
            "status_log": [
                event(
                    "Debate",
                    f"{bundle.article_id}: degraded (error)",
                    article_id=bundle.article_id,
                    error=str(exc),
                )
            ],
            "errors": [f"debate_article {bundle.article_id}: {exc}"],
        }


def _degraded_analysis(bundle: ArticleBundle, err: str) -> ArticleAnalysis:
    empty_brief = lambda stance: PerspectiveBrief(  # noqa: E731
        article_id=bundle.article_id, stance=stance, claims=[]
    )
    critique = CritiqueReport(
        article_id=bundle.article_id,
        evidence_sufficiency="insufficient",
        source_credibility=ScoredComponent(score=2, justification=f"debate failed: {err}"),
        evidence_strength=ScoredComponent(score=1, justification="no claims analyzed"),
        internal_consistency=ScoredComponent(score=5, justification="not assessed"),
        claims_retained_pct=0.0,
    )
    return ArticleAnalysis(
        article_id=bundle.article_id,
        bundle=bundle,
        advocate=empty_brief("advocate"),
        skeptic=empty_brief("skeptic"),
        critique=critique,
    )
