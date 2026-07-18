"""Agent 2: Contextual Retriever.

Deterministic tool-calling: execute the SearchPlan across the mapped sources,
fetch full text where possible (else keep abstract + mark full_text=False),
normalize into ArticleBundles with citation-anchored chunk_ids. Never fabricate
metadata. Does NOT filter (that is Triage's job).
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ..config import get_settings
from ..models import ArticleBundle, Chunk, SearchPlan, SourceType
from ..state import ResearchState
from ..tools.extract import chunk_text, fetch_full_text
from ..tools.search import run_query
from .common import event

_MIN_CHUNK_WORDS = 40
_VALID_SOURCE_TYPES = {"peer_reviewed", "preprint", "news", "report", "blog"}


def _coerce_source_type(value) -> SourceType:
    """Map any unexpected source_type to a safe default."""
    return value if value in _VALID_SOURCE_TYPES else "report"


async def retriever(state: ResearchState) -> dict:
    settings = get_settings()
    plan: SearchPlan = state["search_plan"]

    # 1. Execute all queries (each source wrapper is fault-tolerant).
    raw_lists = await asyncio.gather(
        *[
            asyncio.to_thread(run_query, q, plan.filter_spec, settings)
            for q in plan.queries
        ]
    )
    raw: list[dict] = [r for sub in raw_lists for r in sub]

    # 2. Collapse by URL (true dedup/syndication accounting happens in Triage).
    by_url: dict[str, dict] = {}
    for r in raw:
        url = r.get("url")
        if not url:
            continue
        if url not in by_url:
            by_url[url] = r

    # 3. Build bundles, fetching full text concurrently.
    items = list(by_url.values())
    bundles = await asyncio.gather(
        *[asyncio.to_thread(_build_bundle, i, r, settings.api_timeout) for i, r in enumerate(items)]
    )
    bundles = [b for b in bundles if b is not None]

    return {
        "raw_docs": bundles,
        "status_log": [
            event(
                "Retriever",
                f"retrieved {len(bundles)} documents "
                f"({sum(b.full_text for b in bundles)} with full text)",
            )
        ],
    }


def _build_bundle(idx: int, r: dict, timeout: int) -> ArticleBundle | None:
    url = r.get("url")
    if not url:
        return None
    article_id = f"a{idx}"
    source_type: SourceType = _coerce_source_type(r.get("source_type", "news"))

    full_text = r.get("full_text")
    have_full = bool(full_text and len(full_text.split()) >= _MIN_CHUNK_WORDS)
    if not have_full:
        fetched = fetch_full_text(url, timeout=timeout)
        if fetched and len(fetched.split()) >= _MIN_CHUNK_WORDS:
            full_text = fetched
            have_full = True

    if have_full:
        chunks = chunk_text(full_text, article_id)
    else:
        snippet = r.get("snippet") or r.get("title") or ""
        chunks = (
            [Chunk(chunk_id=f"{article_id}-c0", text=snippet)] if snippet else []
        )

    return ArticleBundle(
        article_id=article_id,
        title=r.get("title") or "(untitled)",
        url=url,
        source_name=r.get("source_name") or "web",
        published_at=_parse_dt(r.get("published_at")),
        source_type=source_type,
        full_text=have_full,
        chunks=chunks,
        citation_count=r.get("citation_count"),
    )


async def reretrieve_article(bundle: ArticleBundle) -> ArticleBundle:
    """One bounded, targeted re-retrieval for a single thin article.

    Searches the open web for the article's title and appends any newly
    extracted chunks. Returns a new bundle (never mutates in place).
    """
    from ..models import FilterSpec, SearchQuery

    settings = get_settings()
    query = SearchQuery(query_str=bundle.title, source_type="report", facet="refill")
    try:
        results = await asyncio.to_thread(
            run_query, query, FilterSpec(), settings
        )
    except Exception:
        results = []

    extra: list[Chunk] = []
    start = len(bundle.chunks)
    for r in results[:3]:
        text = r.get("full_text") or await asyncio.to_thread(
            fetch_full_text, r.get("url", ""), settings.api_timeout
        )
        if text and len(text.split()) >= _MIN_CHUNK_WORDS:
            for c in chunk_text(text, bundle.article_id):
                extra.append(
                    Chunk(chunk_id=f"{bundle.article_id}-c{start + len(extra)}", text=c.text)
                )
        if len(extra) >= 3:
            break

    merged = bundle.model_copy(deep=True)
    merged.chunks = bundle.chunks + extra
    if extra:
        merged.full_text = True
    return merged


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
