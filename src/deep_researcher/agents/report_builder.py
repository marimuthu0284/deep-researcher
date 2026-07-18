"""Agent 8: Report Builder.

Renders the JudgedState into the final Markdown + email-safe HTML. Rendering is
deterministic (from typed state) so every citation resolves and every score
shows its rubric breakdown -- no hallucinated numbers.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import ArticleAnalysis, JudgedState, SearchPlan
from ..report.templates import ReportContext, build_html, build_markdown
from ..state import ResearchState
from .common import event


async def report_builder(state: ResearchState) -> dict:
    settings = get_settings()
    settings.ensure_dirs()

    analyses: list[ArticleAnalysis] = list(state.get("analyses", []))
    judged: JudgedState = state.get("judged", JudgedState())
    plan: SearchPlan | None = state.get("search_plan")
    raw_docs = state.get("raw_docs", [])
    bundles = state.get("bundles", [])

    ctx = ReportContext(
        topic=state["topic"],
        filters=state.get("filters", ""),
        judged=judged,
        analyses=analyses,
        n_queries=len(plan.queries) if plan else 0,
        n_retrieved=len(raw_docs),
        n_after_triage=len(bundles),
        dedup_count=max(0, len(raw_docs) - len(bundles)),
    )

    report_md = build_markdown(ctx)
    report_html = build_html(ctx)

    # Persist so the Dispatcher's fallback link has something to point at.
    (settings.reports_dir / "report.md").write_text(report_md, encoding="utf-8")
    (settings.reports_dir / "report.html").write_text(report_html, encoding="utf-8")

    return {
        "report_md": report_md,
        "report_html": report_html,
        "status_log": [
            event(
                "Report Builder",
                f"rendered report ({len(analyses)} article cards, "
                f"{len(judged.findings)} findings)",
            )
        ],
    }
