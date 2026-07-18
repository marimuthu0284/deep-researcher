"""Agents 4 & 5: Perspective A (Advocate) and Perspective B (Skeptic).

Both run on the same model and see identical evidence, blind to each other, so
the difference in their briefs comes from the stance prompt, not model
personality -- which is what makes downstream contradiction detection genuine.
"""

from __future__ import annotations

from ..llm import structured
from ..models import ArticleBundle, PerspectiveBrief
from ..prompts import PERSPECTIVE_ADVOCATE, PERSPECTIVE_SKEPTIC
from .common import truncate


def _render_chunks(bundle: ArticleBundle) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}] {truncate(c.text, 1200)}" for c in bundle.chunks
    ) or "(no chunks available)"


async def run_advocate(bundle: ArticleBundle) -> PerspectiveBrief:
    brief = await structured(
        "perspective",
        PerspectiveBrief,
        system=PERSPECTIVE_ADVOCATE.format(
            article_id=bundle.article_id,
            title=bundle.title,
            chunk_ids=", ".join(bundle.chunk_ids()) or "(none)",
            chunks=_render_chunks(bundle),
        ),
        user="Produce the advocate PerspectiveBrief JSON.",
        temperature=0.3,
    )
    brief.article_id = bundle.article_id
    brief.stance = "advocate"
    return brief


async def run_skeptic(bundle: ArticleBundle) -> PerspectiveBrief:
    brief = await structured(
        "perspective",
        PerspectiveBrief,
        system=PERSPECTIVE_SKEPTIC.format(
            article_id=bundle.article_id,
            title=bundle.title,
            chunk_ids=", ".join(bundle.chunk_ids()) or "(none)",
            chunks=_render_chunks(bundle),
        ),
        user="Produce the skeptic PerspectiveBrief JSON.",
        temperature=0.3,
    )
    brief.article_id = bundle.article_id
    brief.stance = "skeptic"
    return brief
