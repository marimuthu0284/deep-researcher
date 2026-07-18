"""Deterministic report rendering.

Rendering from the typed JudgedState (rather than free LLM prose) guarantees
every citation resolves to a real URL and every confidence score shows its
rubric breakdown. Produces both Markdown and email-safe inline-CSS HTML.
"""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass

from ..models import ArticleAnalysis, JudgedState
from ..scoring import WEIGHTS, band_short


@dataclass
class ReportContext:
    topic: str
    filters: str
    judged: JudgedState
    analyses: list[ArticleAnalysis]
    n_queries: int
    n_retrieved: int
    n_after_triage: int
    dedup_count: int


_COMPONENT_LABELS = {
    "source_credibility": "Source credibility",
    "evidence_strength": "Evidence strength",
    "corroboration": "Corroboration",
    "internal_consistency": "Internal consistency",
    "recency_relevance": "Recency & relevance",
}


def _citation_index(ctx: ReportContext) -> dict[str, int]:
    return {a.article_id: i + 1 for i, a in enumerate(ctx.analyses)}


# --------------------------------------------------------------------------- #
# Insights data (shared by the Markdown text-bars and the HTML SVG charts)
# --------------------------------------------------------------------------- #
_BAND_ORDER = ["Strong", "Moderate", "Emerging", "Weak"]
# Reuses the same severity colors already used for confidence badges elsewhere
# in this report, so the Insights section speaks the same visual language
# rather than introducing a second, unrelated palette.
_BAND_COLORS = {
    "Strong": "#1a7f37",
    "Moderate": "#9a6700",
    "Emerging": "#bc4c00",
    "Weak": "#a40e26",
}
_HIST_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]


def _band_counts(judged: JudgedState) -> dict[str, int]:
    counts = {b: 0 for b in _BAND_ORDER}
    for v in judged.verdicts:
        counts[band_short(v.confidence_score)] += 1
    return counts


def _confidence_histogram(judged: JudgedState) -> list[tuple[str, int]]:
    counts = [0] * len(_HIST_BINS)
    for v in judged.verdicts:
        idx = min(int(v.confidence_score // 20), len(_HIST_BINS) - 1)
        counts[idx] += 1
    return [(f"{lo}-{hi}", n) for (lo, hi), n in zip(_HIST_BINS, counts)]


def _source_counts(analyses: list[ArticleAnalysis]) -> list[tuple[str, int]]:
    c = Counter(a.bundle.source_type.replace("_", " ") for a in analyses)
    return sorted(c.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def build_markdown(ctx: ReportContext) -> str:
    j = ctx.judged
    cite = _citation_index(ctx)
    by_id = {a.article_id: a for a in ctx.analyses}
    verdict_by_id = {v.article_id: v for v in j.verdicts}

    lines: list[str] = []
    lines.append(f"# Deep Research Report: {ctx.topic}\n")
    lines.append(f"*Filters: {ctx.filters or 'none'}*\n")

    if j.synthesis_degraded:
        lines.append(
            "> ⚠️ **Degraded run — this is NOT a full analysis.** The LLM synthesis "
            "layer was unavailable, so the findings below are automated fallbacks "
            "rather than a cross-source judgement. Confidence scores are not "
            "reliable. Re-run once the LLM gateway is reachable (check "
            "`deep-researcher --doctor`).\n"
        )

    lines.append("## Executive summary\n")
    lines.append((j.executive_summary or "_No summary available._") + "\n")
    if j.trajectory:
        lines.append(f"**Evidence trajectory:** {j.trajectory}\n")

    lines.append("## Key findings\n")
    if not j.findings:
        lines.append("_No findings synthesized._\n")
    for f in j.findings:
        cites = " ".join(
            f"[{cite[aid]}]" for aid in f.supporting_article_ids if aid in cite
        )
        thread = " _(single-thread)_" if f.single_thread else ""
        lines.append(
            f"- **[{f.band} · {f.confidence:.0f}]** {f.statement} {cites}{thread}"
        )
    lines.append("")

    lines.append("## Points of genuine disagreement\n")
    if not j.disagreements:
        lines.append("_None flagged._\n")
    for d in j.disagreements:
        cites = " ".join(f"[{cite[aid]}]" for aid in d.article_ids if aid in cite)
        lines.append(f"- **{d.topic}** — {d.summary} {cites}")
    lines.append("")

    lines.append("## Per-article analysis\n")
    for a in ctx.analyses:
        v = verdict_by_id.get(a.article_id)
        n = cite[a.article_id]
        b = a.bundle
        lines.append(f"### [{n}] {b.title}\n")
        lines.append(
            f"- Source: {b.source_name} ({b.source_type}) · "
            f"Published: {b.published_at.date() if b.published_at else 'unknown'} · "
            f"[link]({b.url})"
        )
        if v:
            lines.append(f"- **Confidence: {v.confidence_score:.0f} — {v.band}**")
            lines.append("- Rubric breakdown:")
            for comp, label in _COMPONENT_LABELS.items():
                score = v.score_breakdown.get(comp, 0.0)
                weight = WEIGHTS[comp]
                lines.append(f"    - {label}: {score:.1f}/10 (weight {weight:.0%})")
            lines.append(f"- Resolved position: {v.resolved_position}")
            if v.dissent_note:
                lines.append(f"- Dissent note: {v.dissent_note}")
        if a.critique.uncited_claims_rejected:
            lines.append(
                f"- Rejected (failed citation audit): "
                f"{', '.join(a.critique.uncited_claims_rejected)}"
            )
        if a.critique.credibility_flags:
            flags = ", ".join(fl.label for fl in a.critique.credibility_flags)
            lines.append(f"- Credibility flags: {flags}")
        lines.append("")

    lines.append("## Methodology appendix\n")
    lines.append(f"- Queries run: {ctx.n_queries}")
    lines.append(f"- Documents retrieved: {ctx.n_retrieved}")
    lines.append(f"- Duplicates removed: {ctx.dedup_count}")
    lines.append(f"- Articles analyzed after triage: {ctx.n_after_triage}")
    lines.append("- Scoring rubric weights:")
    for comp, label in _COMPONENT_LABELS.items():
        lines.append(f"    - {label}: {WEIGHTS[comp]:.0%}")
    lines.append(
        "- Confidence = Σ(component × weight) × 10; finding-level confidence "
        "aggregated via noisy-OR damped by source independence."
    )
    lines.append("")

    lines.append("## References\n")
    for a in ctx.analyses:
        n = cite[a.article_id]
        lines.append(f"{n}. [{a.bundle.title}]({a.bundle.url}) — {a.bundle.source_name}")
    lines.append("")

    lines.extend(_insights_markdown(ctx))

    return "\n".join(lines)


def _text_bar(value: float, max_value: float, width: int = 20) -> str:
    if max_value <= 0:
        return "░" * width
    filled = round((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


def _insights_markdown(ctx: ReportContext) -> list[str]:
    j = ctx.judged
    lines: list[str] = ["## Insights\n"]
    if j.synthesis_degraded:
        lines.append(
            "_Based on fallback confidence scores from a degraded run — "
            "treat these distributions as illustrative, not reliable._\n"
        )

    band_counts = _band_counts(j)
    total = sum(band_counts.values())
    lines.append("**Confidence band distribution**\n")
    if total:
        for b in _BAND_ORDER:
            n = band_counts[b]
            if not n:
                continue
            pct = round(100 * n / total)
            lines.append(f"- {b:<9} {_text_bar(n, total)}  {n} ({pct}%)")
    else:
        lines.append("_No verdicts to summarize._")
    lines.append("")

    lines.append("**Confidence score spread**\n")
    hist = _confidence_histogram(j)
    max_h = max((v for _, v in hist), default=0)
    for label, v in hist:
        lines.append(f"- {label:<7} {_text_bar(v, max_h)}  {v}")
    lines.append("")

    lines.append("**Source mix**\n")
    src = _source_counts(ctx.analyses)
    max_s = max((v for _, v in src), default=0)
    for label, v in src:
        lines.append(f"- {label:<15} {_text_bar(v, max_s)}  {v}")
    lines.append("")

    return lines


# --------------------------------------------------------------------------- #
# HTML (inline CSS, email-safe)
# --------------------------------------------------------------------------- #
def _badge_color(confidence: float) -> str:
    if confidence >= 80:
        return "#1a7f37"
    if confidence >= 60:
        return "#9a6700"
    if confidence >= 40:
        return "#bc4c00"
    return "#a40e26"


def _e(text: str) -> str:
    return html.escape(str(text))


def _donut_svg(counts: dict[str, int]) -> str:
    """Confidence-band donut, built as stroked arc segments (no JS/libs)."""
    total = sum(counts.values())
    if not total:
        return "<p style='color:#656d76;font-size:13px'><em>No verdicts to chart.</em></p>"

    cx, cy, r = 70, 70, 50
    circumference = 2 * 3.14159265 * r
    gap = 3.0  # px gap between segments, so touching slices stay visually distinct
    offset = 0.0
    segments = []
    for b in _BAND_ORDER:
        n = counts[b]
        if not n:
            continue
        seg_len = (n / total) * circumference
        dash = max(seg_len - gap, 0.0)
        segments.append(
            f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{_BAND_COLORS[b]}' "
            f"stroke-width='22' stroke-dasharray='{dash:.2f} {circumference - dash:.2f}' "
            f"stroke-dashoffset='{-offset:.2f}' transform='rotate(-90 {cx} {cy})'/>"
        )
        offset += seg_len

    center = (
        f"<text x='{cx}' y='{cy - 4}' text-anchor='middle' font-size='20' "
        f"font-weight='600' fill='#1f2328'>{total}</text>"
        f"<text x='{cx}' y='{cy + 14}' text-anchor='middle' font-size='11' "
        f"fill='#656d76'>article{'s' if total != 1 else ''}</text>"
    )
    svg = (
        f"<svg width='140' height='140' viewBox='0 0 140 140' role='img' "
        f"aria-label='Confidence band distribution'>{''.join(segments)}{center}</svg>"
    )

    legend = []
    table_rows = []
    for b in _BAND_ORDER:
        n = counts[b]
        if not n:
            continue
        pct = round(100 * n / total)
        legend.append(
            "<div style='display:flex;align-items:center;gap:6px;margin:4px 0;font-size:13px'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
            f"background:{_BAND_COLORS[b]}'></span><span>{_e(b)} — {n} ({pct}%)</span></div>"
        )
        table_rows.append(f"<tr><td>{_e(b)}</td><td>{n}</td><td>{pct}%</td></tr>")

    table = (
        "<details style='margin-top:8px;font-size:12px;color:#656d76'>"
        "<summary>Table view</summary>"
        "<table style='border-collapse:collapse;margin-top:6px'>"
        "<tr style='text-align:left'><th style='padding:2px 8px 2px 0'>Band</th>"
        "<th style='padding:2px 8px'>Count</th><th style='padding:2px 8px'>Share</th></tr>"
        + "".join(table_rows) + "</table></details>"
    )

    return (
        "<div style='display:flex;align-items:center;gap:24px;flex-wrap:wrap'>"
        f"<div>{svg}</div><div>{''.join(legend)}</div></div>{table}"
    )


def _hbar_svg(items: list[tuple[str, int]], color: str = "#2a78d6") -> str:
    """A minimal single-series horizontal bar/histogram - no external libs."""
    if not items:
        return "<p style='color:#656d76;font-size:13px'><em>No data to chart.</em></p>"

    max_v = max(v for _, v in items) or 1
    bar_h, gap, label_w, chart_w = 18, 10, 90, 220
    row_h = bar_h + gap
    height = row_h * len(items) + gap
    rows = []
    y = gap / 2
    for label, v in items:
        w = (v / max_v) * chart_w
        text_y = y + bar_h * 0.72
        rows.append(
            f"<text x='{label_w - 8}' y='{text_y:.1f}' text-anchor='end' font-size='12' "
            f"fill='#1f2328'>{_e(label)}</text>"
        )
        rows.append(
            f"<rect x='{label_w}' y='{y:.1f}' width='{w:.1f}' height='{bar_h}' "
            f"rx='4' fill='{color}'/>"
        )
        rows.append(
            f"<text x='{label_w + w + 6:.1f}' y='{text_y:.1f}' font-size='12' "
            f"fill='#656d76'>{v}</text>"
        )
        y += row_h

    total_w = label_w + chart_w + 40
    return (
        f"<svg width='100%' viewBox='0 0 {total_w} {height:.1f}' style='max-width:420px' "
        f"role='img' aria-label='bar chart'>{''.join(rows)}</svg>"
    )


def _insights_html(ctx: ReportContext) -> str:
    j = ctx.judged
    parts = ["<h2>Insights</h2>"]
    if j.synthesis_degraded:
        parts.append(
            "<p style='color:#656d76;font-size:13px'><em>Based on fallback confidence "
            "scores from a degraded run — treat these distributions as illustrative, "
            "not reliable.</em></p>"
        )
    parts.append("<h3 style='font-size:15px;margin:16px 0 8px'>Confidence band distribution</h3>")
    parts.append(_donut_svg(_band_counts(j)))
    parts.append("<h3 style='font-size:15px;margin:20px 0 8px'>Confidence score spread</h3>")
    parts.append(_hbar_svg(_confidence_histogram(j)))
    parts.append("<h3 style='font-size:15px;margin:20px 0 8px'>Source mix</h3>")
    parts.append(_hbar_svg(_source_counts(ctx.analyses)))
    return "".join(parts)


def build_html(ctx: ReportContext) -> str:
    j = ctx.judged
    cite = _citation_index(ctx)
    verdict_by_id = {v.article_id: v for v in j.verdicts}

    p = []
    p.append(
        "<div style=\"font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"
        "sans-serif;max-width:760px;margin:0 auto;color:#1f2328;line-height:1.5\">"
    )
    p.append(f"<h1 style='font-size:24px'>Deep Research Report: {_e(ctx.topic)}</h1>")
    p.append(f"<p style='color:#656d76'>Filters: {_e(ctx.filters or 'none')}</p>")

    if j.synthesis_degraded:
        p.append(
            "<div style='background:#fff1f0;border:1px solid #ffccc7;border-left:4px "
            "solid #a40e26;border-radius:6px;padding:12px 16px;margin:12px 0;"
            "color:#a40e26'>"
            "<strong>⚠️ Degraded run — this is NOT a full analysis.</strong> "
            "The LLM synthesis layer was unavailable, so the findings below are "
            "automated fallbacks rather than a cross-source judgement. Confidence "
            "scores are not reliable. Re-run once the LLM gateway is reachable "
            "(check <code>deep-researcher --doctor</code>)."
            "</div>"
        )

    p.append("<h2>Executive summary</h2>")
    p.append(f"<p>{_e(j.executive_summary or 'No summary available.')}</p>")
    if j.trajectory:
        p.append(f"<p><strong>Evidence trajectory:</strong> {_e(j.trajectory)}</p>")

    p.append("<h2>Key findings</h2><ul>")
    for f in j.findings:
        color = _badge_color(f.confidence)
        cites = " ".join(
            f"<sup>[{cite[aid]}]</sup>" for aid in f.supporting_article_ids if aid in cite
        )
        badge = (
            f"<span style='background:{color};color:#fff;border-radius:4px;"
            f"padding:1px 6px;font-size:12px'>{_e(f.band)} · {f.confidence:.0f}</span>"
        )
        thread = " <em>(single-thread)</em>" if f.single_thread else ""
        p.append(f"<li>{badge} {_e(f.statement)} {cites}{thread}</li>")
    p.append("</ul>")

    p.append("<h2>Points of genuine disagreement</h2><ul>")
    if not j.disagreements:
        p.append("<li><em>None flagged.</em></li>")
    for d in j.disagreements:
        p.append(f"<li><strong>{_e(d.topic)}</strong> — {_e(d.summary)}</li>")
    p.append("</ul>")

    p.append("<h2>Per-article analysis</h2>")
    for a in ctx.analyses:
        v = verdict_by_id.get(a.article_id)
        n = cite[a.article_id]
        b = a.bundle
        conf = v.confidence_score if v else 0
        color = _badge_color(conf)
        p.append(
            "<div style='border:1px solid #d0d7de;border-radius:8px;"
            "padding:12px 16px;margin:12px 0'>"
        )
        p.append(
            f"<h3 style='margin:0 0 4px'>[{n}] "
            f"<a href='{_e(b.url)}' style='color:#0969da'>{_e(b.title)}</a></h3>"
        )
        p.append(
            f"<p style='color:#656d76;margin:0 0 8px;font-size:13px'>"
            f"{_e(b.source_name)} · {_e(b.source_type)} · "
            f"{b.published_at.date() if b.published_at else 'date unknown'}</p>"
        )
        if v:
            p.append(
                f"<p><span style='background:{color};color:#fff;border-radius:4px;"
                f"padding:2px 8px'>Confidence {v.confidence_score:.0f} — "
                f"{_e(v.band)}</span></p>"
            )
            p.append(
                "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
            )
            for comp, label in _COMPONENT_LABELS.items():
                score = v.score_breakdown.get(comp, 0.0)
                p.append(
                    "<tr>"
                    f"<td style='padding:2px 6px;border-bottom:1px solid #eee'>{_e(label)}</td>"
                    f"<td style='padding:2px 6px;border-bottom:1px solid #eee;text-align:right'>"
                    f"{score:.1f}/10</td>"
                    f"<td style='padding:2px 6px;border-bottom:1px solid #eee;"
                    f"text-align:right;color:#656d76'>weight {WEIGHTS[comp]:.0%}</td>"
                    "</tr>"
                )
            p.append("</table>")
            p.append(f"<p><strong>Resolved:</strong> {_e(v.resolved_position)}</p>")
            if v.dissent_note:
                p.append(
                    f"<p style='color:#656d76'><strong>Dissent:</strong> "
                    f"{_e(v.dissent_note)}</p>"
                )
        if a.critique.uncited_claims_rejected:
            p.append(
                "<p style='color:#a40e26;font-size:13px'>Rejected (citation audit): "
                f"{_e(', '.join(a.critique.uncited_claims_rejected))}</p>"
            )
        p.append("</div>")

    p.append("<h2>Methodology appendix</h2><ul>")
    p.append(f"<li>Queries run: {ctx.n_queries}</li>")
    p.append(f"<li>Documents retrieved: {ctx.n_retrieved}</li>")
    p.append(f"<li>Duplicates removed: {ctx.dedup_count}</li>")
    p.append(f"<li>Articles analyzed after triage: {ctx.n_after_triage}</li>")
    p.append("</ul>")

    p.append("<h2>References</h2><ol>")
    for a in ctx.analyses:
        p.append(
            f"<li><a href='{_e(a.bundle.url)}'>{_e(a.bundle.title)}</a> — "
            f"{_e(a.bundle.source_name)}</li>"
        )
    p.append("</ol>")

    p.append(_insights_html(ctx))

    p.append("</div>")
    return "".join(p)
