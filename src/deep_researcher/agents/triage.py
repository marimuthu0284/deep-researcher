"""Agent 3: Triage & Dedup.

Deterministic dedup (exact URL, near-identical titles, syndicated wire stories)
with syndication_count accounting, then an LLM scores relevance / clusters /
selects the top-N. A heuristic fallback keeps the stage working offline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

from ..config import get_settings
from ..llm import structured
from ..models import ArticleBundle
from ..prompts import TRIAGE
from ..state import ResearchState
from .common import event
from pydantic import BaseModel

_SOURCE_TIER = {
    "peer_reviewed": 5,
    "report": 4,
    "news": 3,
    "preprint": 2,
    "blog": 1,
}


class _TriageItem(BaseModel):
    article_id: str
    relevance: float = 3.0
    cluster: str = "general"
    triage_rationale: str = ""


class _TriageResult(BaseModel):
    items: list[_TriageItem]


async def triage(state: ResearchState) -> dict:
    settings = get_settings()
    docs: list[ArticleBundle] = list(state.get("raw_docs", []))
    n = settings.top_n

    deduped, dup_count = _dedup(docs)

    # LLM scoring + selection.
    try:
        chosen = await _llm_select(state["topic"], deduped, n)
    except Exception:
        chosen = _heuristic_select(deduped, n)

    return {
        "bundles": chosen,
        "status_log": [
            event(
                "Triage",
                f"{len(docs)} raw -> {len(deduped)} after dedup "
                f"({dup_count} duplicates) -> kept top {len(chosen)}",
            )
        ],
    }


# --------------------------------------------------------------------------- #
# Deterministic dedup
# --------------------------------------------------------------------------- #
def _dedup(docs: list[ArticleBundle]) -> tuple[list[ArticleBundle], int]:
    # Exact URL collapse first.
    seen_url: dict[str, ArticleBundle] = {}
    for d in docs:
        if d.url not in seen_url:
            seen_url[d.url] = d
    unique = list(seen_url.values())

    # Title-similarity clustering to catch syndicated wire stories.
    clusters: list[list[ArticleBundle]] = []
    for d in unique:
        placed = False
        for cluster in clusters:
            if _title_cosine(d.title, cluster[0].title) > 0.9:
                cluster.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])

    kept: list[ArticleBundle] = []
    duplicates_removed = 0
    for cluster in clusters:
        representative = _most_authoritative(cluster)
        representative = representative.model_copy(deep=True)
        representative.syndication_count = len(cluster)
        kept.append(representative)
        duplicates_removed += len(cluster) - 1

    return kept, duplicates_removed


def _most_authoritative(cluster: list[ArticleBundle]) -> ArticleBundle:
    def key(b: ArticleBundle):
        tier = _SOURCE_TIER.get(b.source_type, 0)
        earliest = b.published_at or datetime.max.replace(tzinfo=timezone.utc)
        # higher tier wins; earlier date wins
        return (-tier, earliest)

    return sorted(cluster, key=key)[0]


def _tokenize(text: str) -> Counter:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return Counter(words)


def _title_cosine(a: str, b: str) -> float:
    va, vb = _tokenize(a), _tokenize(b)
    if not va or not vb:
        return 0.0
    common = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in common)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
async def _llm_select(
    topic: str, docs: list[ArticleBundle], n: int
) -> list[ArticleBundle]:
    candidates = "\n".join(
        f"- {d.article_id} | {d.source_type} | {d.source_name} | "
        f"syndication={d.syndication_count} | {d.title[:120]}"
        for d in docs
    )
    result = await structured(
        "planner",
        _TriageResult,
        system=TRIAGE.format(topic=topic, n=n, candidates=candidates),
        user="Score and select the top articles. Return JSON.",
    )
    by_id = {d.article_id: d for d in docs}
    chosen: list[ArticleBundle] = []
    for item in result.items[:n]:
        base = by_id.get(item.article_id)
        if not base:
            continue
        b = base.model_copy(deep=True)
        b.relevance = item.relevance
        b.cluster = item.cluster
        b.triage_rationale = item.triage_rationale
        chosen.append(b)
    if not chosen:
        raise ValueError("LLM triage returned no valid ids")
    return chosen


def _heuristic_select(docs: list[ArticleBundle], n: int) -> list[ArticleBundle]:
    now = datetime.now(timezone.utc)

    def score(b: ArticleBundle) -> float:
        recency = 1.0
        if b.published_at:
            age_days = max(0.0, (now - _aware(b.published_at)).days)
            recency = math.exp(-age_days / 120.0)
        tier = _SOURCE_TIER.get(b.source_type, 1) / 5.0
        cites = math.log1p(b.citation_count or 0) / 5.0
        return 0.5 * recency + 0.3 * tier + 0.2 * cites

    ranked = sorted(docs, key=score, reverse=True)

    # Enforce a diversity bonus: avoid an all-one-type top-N.
    chosen: list[ArticleBundle] = []
    seen_types: Counter = Counter()
    for b in ranked:
        if len(chosen) >= n:
            break
        if seen_types[b.source_type] >= max(2, n // 2) and len(ranked) > n:
            continue
        c = b.model_copy(deep=True)
        c.cluster = c.cluster or b.source_type
        c.triage_rationale = c.triage_rationale or f"kept: {b.source_type} source"
        chosen.append(c)
        seen_types[b.source_type] += 1
    for b in ranked:  # backfill if diversity filter left us short
        if len(chosen) >= n:
            break
        if b.article_id not in {c.article_id for c in chosen}:
            chosen.append(b.model_copy(deep=True))
    return chosen[:n]


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
