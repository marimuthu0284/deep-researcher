"""Agent 6: Critical Analysis (the quality gate).

Audits citations (entailment), detects and classifies contradictions, validates
the source, judges evidence sufficiency, and scores three rubric components.
Runs on a different model family than the perspectives to decorrelate blind
spots.
"""

from __future__ import annotations

import json

from ..llm import structured
from ..models import (
    ArticleBundle,
    CritiqueReport,
    PerspectiveBrief,
)
from ..prompts import CRITICAL_ANALYSIS
from .common import truncate


def _render_chunks(bundle: ArticleBundle) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}] {truncate(c.text, 1000)}" for c in bundle.chunks
    ) or "(no chunks available)"


def _brief_json(brief: PerspectiveBrief) -> str:
    return json.dumps(brief.model_dump(), indent=2, default=str)


async def run_critical_analysis(
    bundle: ArticleBundle,
    advocate: PerspectiveBrief,
    skeptic: PerspectiveBrief,
) -> CritiqueReport:
    critique = await structured(
        "critic",
        CritiqueReport,
        system=CRITICAL_ANALYSIS.format(
            article_id=bundle.article_id,
            title=bundle.title,
            source_type=bundle.source_type,
            source_name=bundle.source_name,
            syndication_count=bundle.syndication_count,
            citation_count=bundle.citation_count,
            chunks=_render_chunks(bundle),
            advocate=_brief_json(advocate),
            skeptic=_brief_json(skeptic),
        ),
        user="Produce the CritiqueReport JSON.",
    )
    critique.article_id = bundle.article_id
    return critique
