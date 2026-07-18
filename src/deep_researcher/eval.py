"""Evaluation harness.

Computes quality metrics over a completed run (live state or a cached run):

  - Citation audit: how much of the debate survived the entailment check --
    the core anti-hallucination signal.
  - Confidence distribution: per-band counts and mean, to sanity-check that the
    rubric is discriminating rather than clumping.
  - Calibration checks: structural invariants the scoring pipeline should hold
    (findings ordered by confidence, single-thread findings not rated Strong,
    verdicts within 0-100, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import ArticleAnalysis, JudgedState
from .scoring import band_short


@dataclass
class CitationMetrics:
    total_claims: int
    rejected_claims: int
    retained_claims: int
    retained_pct: float
    articles: int
    articles_with_rejections: int
    reretrieved_articles: int


@dataclass
class ConfidenceMetrics:
    n_verdicts: int
    mean_confidence: float
    band_counts: dict[str, int]
    per_article: dict[str, float]
    n_findings: int
    single_thread_findings: int


def citation_metrics(analyses: list[ArticleAnalysis]) -> CitationMetrics:
    total_claims = 0
    rejected = 0
    with_rejections = 0
    reretrieved = 0
    for a in analyses:
        claims = len(a.advocate.claims) + len(a.skeptic.claims)
        total_claims += claims
        r = len(a.critique.uncited_claims_rejected)
        rejected += r
        if r:
            with_rejections += 1
        if a.reretrieved:
            reretrieved += 1
    retained = max(0, total_claims - rejected)
    pct = round(100.0 * retained / total_claims, 1) if total_claims else 0.0
    return CitationMetrics(
        total_claims=total_claims,
        rejected_claims=rejected,
        retained_claims=retained,
        retained_pct=pct,
        articles=len(analyses),
        articles_with_rejections=with_rejections,
        reretrieved_articles=reretrieved,
    )


def confidence_metrics(judged: JudgedState) -> ConfidenceMetrics:
    verdicts = judged.verdicts
    band_counts = {"Strong": 0, "Moderate": 0, "Emerging": 0, "Weak": 0}
    per_article: dict[str, float] = {}
    for v in verdicts:
        band_counts[band_short(v.confidence_score)] += 1
        per_article[v.article_id] = v.confidence_score
    mean = (
        round(sum(v.confidence_score for v in verdicts) / len(verdicts), 1)
        if verdicts
        else 0.0
    )
    return ConfidenceMetrics(
        n_verdicts=len(verdicts),
        mean_confidence=mean,
        band_counts=band_counts,
        per_article=per_article,
        n_findings=len(judged.findings),
        single_thread_findings=sum(1 for f in judged.findings if f.single_thread),
    )


def calibration_checks(judged: JudgedState) -> list[dict[str, Any]]:
    """Return a list of {check, passed, detail} structural invariants."""
    checks: list[dict[str, Any]] = []

    confs = [f.confidence for f in judged.findings]
    ordered = confs == sorted(confs, reverse=True)
    checks.append(
        {
            "check": "findings_ordered_by_confidence",
            "passed": ordered,
            "detail": f"{confs}",
        }
    )

    bad_single_thread = [
        f.statement[:60]
        for f in judged.findings
        if f.single_thread and f.confidence >= 80
    ]
    checks.append(
        {
            "check": "single_thread_not_strong",
            "passed": not bad_single_thread,
            "detail": bad_single_thread or "none",
        }
    )

    out_of_range = [
        v.article_id for v in judged.verdicts if not (0.0 <= v.confidence_score <= 100.0)
    ]
    checks.append(
        {
            "check": "verdicts_in_range",
            "passed": not out_of_range,
            "detail": out_of_range or "all within 0-100",
        }
    )

    missing_breakdown = [
        v.article_id for v in judged.verdicts if not v.score_breakdown
    ]
    checks.append(
        {
            "check": "verdicts_have_breakdown",
            "passed": not missing_breakdown,
            "detail": missing_breakdown or "all present",
        }
    )
    return checks


def evaluate_state(state: dict) -> dict[str, Any]:
    """Run all evaluators over a (live or hydrated) ResearchState."""
    analyses: list[ArticleAnalysis] = list(state.get("analyses", []))
    judged: JudgedState = state.get("judged") or JudgedState()
    checks = calibration_checks(judged)
    return {
        "citation": asdict(citation_metrics(analyses)),
        "confidence": asdict(confidence_metrics(judged)),
        "calibration": checks,
        "calibration_passed": all(c["passed"] for c in checks),
    }
