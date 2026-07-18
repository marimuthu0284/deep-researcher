"""Search API wrappers with per-query caching and graceful degradation.

Each source is wrapped so a failure (missing key, timeout, rate limit) returns
an empty list rather than crashing the pipeline -- every external dependency
has a fallback. Results are normalized into plain dicts that the Retriever
turns into ArticleBundles.

Sources:
  - Tavily        (web search, requires TAVILY_API_KEY)
  - arXiv         (preprints, no key)
  - Semantic Scholar (peer-reviewed + citation counts, no key)
  - GDELT 2.0     (real-time news incl. source-country filter, no key)
  - Crossref      (venue metadata / DOI resolution, no key)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..config import Settings, get_settings
from ..models import FilterSpec, SearchQuery, SourceType

_UA = {"User-Agent": "DeepResearcher/0.1 (hackathon; mailto:research@example.com)"}


# --------------------------------------------------------------------------- #
# Per-query disk cache (survives flaky wifi on repeated demo runs)
# --------------------------------------------------------------------------- #
def _cache_path(source: str, query: str, params: dict) -> Path:
    settings = get_settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        json.dumps([source, query, params], sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return settings.cache_dir / f"q_{source}_{key}.json"


def _read_cache(path: Path) -> list[dict] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(path: Path, data: list[dict]) -> None:
    try:
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception:
        pass


def _cached(source: str, query: str, params: dict, fn) -> list[dict]:
    path = _cache_path(source, query, params)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    try:
        data = fn()
    except Exception:
        data = []
    _write_cache(path, data)
    return data


def _norm(
    *,
    title: str,
    url: str,
    source_name: str,
    source_type: SourceType,
    snippet: str = "",
    full_text: str | None = None,
    published_at: str | None = None,
    citation_count: int | None = None,
) -> dict[str, Any]:
    return {
        "title": (title or "").strip(),
        "url": url,
        "source_name": source_name,
        "source_type": source_type,
        "snippet": snippet or "",
        "full_text": full_text,
        "published_at": published_at,
        "citation_count": citation_count,
    }


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def tavily_search(query: str, params: dict, settings: Settings) -> list[dict]:
    if not settings.tavily_api_key:
        return []

    def _run() -> list[dict]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        kwargs: dict[str, Any] = {
            "query": query,
            "max_results": params.get("max_results", 6),
            "search_depth": params.get("search_depth", "advanced"),
        }
        allow = params.get("include_domains")
        if allow:
            kwargs["include_domains"] = allow
        resp = client.search(**kwargs)
        out = []
        for r in resp.get("results", []):
            out.append(
                _norm(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    source_name=_domain(r.get("url", "")),
                    source_type=params.get("source_type", "news"),
                    snippet=r.get("content", ""),
                    full_text=r.get("raw_content"),
                    published_at=r.get("published_date"),
                )
            )
        return out

    return _cached("tavily", query, params, _run)


def arxiv_search(query: str, params: dict) -> list[dict]:
    def _run() -> list[dict]:
        import arxiv

        search = arxiv.Search(
            query=query,
            max_results=params.get("max_results", 5),
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        out = []
        for r in arxiv.Client().results(search):
            out.append(
                _norm(
                    title=r.title,
                    url=r.entry_id,
                    source_name="arXiv",
                    source_type="preprint",
                    snippet=r.summary,
                    published_at=r.published.isoformat() if r.published else None,
                )
            )
        return out

    return _cached("arxiv", query, params, _run)


def semantic_scholar_search(query: str, params: dict, settings: Settings) -> list[dict]:
    def _run() -> list[dict]:
        headers = dict(_UA)
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": params.get("max_results", 6),
                "fields": "title,abstract,url,year,publicationDate,venue,"
                "citationCount,isOpenAccess,openAccessPdf,externalIds",
            },
            headers=headers,
            timeout=settings.api_timeout,
        )
        resp.raise_for_status()
        out = []
        for p in resp.json().get("data", []) or []:
            pub = p.get("publicationDate")
            if not pub and p.get("year"):
                pub = f"{p['year']}-01-01"
            out.append(
                _norm(
                    title=p.get("title", ""),
                    url=(p.get("openAccessPdf") or {}).get("url")
                    or p.get("url", ""),
                    source_name=p.get("venue") or "Semantic Scholar",
                    source_type="peer_reviewed",
                    snippet=p.get("abstract") or "",
                    published_at=pub,
                    citation_count=p.get("citationCount"),
                )
            )
        return out

    return _cached("s2", query, params, _run)


def gdelt_search(query: str, params: dict, settings: Settings) -> list[dict]:
    def _run() -> list[dict]:
        q = query
        country = params.get("country")
        if country:
            q = f"{query} sourcecountry:{country}"
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": q,
                "mode": "ArtList",
                "maxrecords": params.get("max_results", 8),
                "format": "json",
                "sort": "DateDesc",
                "timespan": params.get("timespan", "3m"),
            },
            headers=_UA,
            timeout=settings.api_timeout,
        )
        resp.raise_for_status()
        out = []
        for a in resp.json().get("articles", []) or []:
            out.append(
                _norm(
                    title=a.get("title", ""),
                    url=a.get("url", ""),
                    source_name=a.get("domain", _domain(a.get("url", ""))),
                    source_type="news",
                    snippet=a.get("title", ""),
                    published_at=_gdelt_date(a.get("seendate")),
                )
            )
        return out

    return _cached("gdelt", query, params, _run)


def crossref_lookup(title: str, settings: Settings) -> dict | None:
    """Resolve venue metadata for a title (used to enrich credibility signals)."""

    def _run() -> list[dict]:
        resp = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 1},
            headers=_UA,
            timeout=settings.api_timeout,
        )
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        return items[:1]

    items = _cached("crossref", title, {}, _run)
    return items[0] if items else None


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def run_query(
    query: SearchQuery, filter_spec: FilterSpec, settings: Settings | None = None
) -> list[dict]:
    """Execute one planned query against the source(s) mapped to its type."""
    settings = settings or get_settings()
    params = dict(query.api_params)
    # The query's validated source_type is authoritative; never let a stray
    # api_params value (e.g. an LLM-emitted "web") override it.
    params["source_type"] = query.source_type
    if filter_spec.domain_allowlist:
        params.setdefault("include_domains", filter_spec.domain_allowlist)
    if filter_spec.country:
        params.setdefault("country", filter_spec.country)

    results: list[dict] = []
    st = query.source_type
    if st == "peer_reviewed":
        results += semantic_scholar_search(query.query_str, params, settings)
        results += arxiv_search(query.query_str, params)
    elif st == "preprint":
        results += arxiv_search(query.query_str, params)
        results += semantic_scholar_search(query.query_str, params, settings)
    elif st == "news":
        results += gdelt_search(query.query_str, params, settings)
        results += tavily_search(query.query_str, params, settings)
    else:  # report, blog -> general web
        results += tavily_search(query.query_str, params, settings)

    return _apply_date_filter(results, filter_spec)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).netloc.replace("www.", "") or "web"
    except Exception:
        return "web"


def _gdelt_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except Exception:
        return None


def _apply_date_filter(results: list[dict], filter_spec: FilterSpec) -> list[dict]:
    if not filter_spec.published_after:
        return results
    cutoff = filter_spec.published_after
    kept = []
    for r in results:
        pub = r.get("published_at")
        if not pub:
            kept.append(r)  # never guess; keep undated items, flag later
            continue
        try:
            dt = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            cmp_cut = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=timezone.utc)
            if dt >= cmp_cut:
                kept.append(r)
        except Exception:
            kept.append(r)
    return kept
