"""Confidence scoring (design doc section 6).

The LLM scores the rubric *components* (0-10, each with a justification); this
module does the arithmetic, which keeps the totals honest and reproducible.
"""

from __future__ import annotations

from math import prod

# Rubric weights (must sum to 1.0).
WEIGHTS: dict[str, float] = {
    "source_credibility": 0.25,
    "evidence_strength": 0.25,
    "corroboration": 0.20,
    "internal_consistency": 0.15,
    "recency_relevance": 0.15,
}

COMPONENT_ORDER = list(WEIGHTS.keys())


def compute_confidence(components: dict[str, float]) -> float:
    """confidence = sum(component_i * weight_i) * 10, clamped to 0-100.

    Components are on a 0-10 scale. Missing components are treated as 0.
    """
    total = 0.0
    for name, weight in WEIGHTS.items():
        total += float(components.get(name, 0.0)) * weight
    return round(max(0.0, min(100.0, total * 10.0)), 1)


def band(confidence: float) -> str:
    """Map a 0-100 confidence to a calibration band label."""
    if confidence >= 80:
        return "Strong - multiple independent, high-quality sources"
    if confidence >= 60:
        return "Moderate - credible but limited corroboration"
    if confidence >= 40:
        return "Emerging - early or contested"
    return "Weak - single-thread, preprint, or unresolved contradictions"


def band_short(confidence: float) -> str:
    if confidence >= 80:
        return "Strong"
    if confidence >= 60:
        return "Moderate"
    if confidence >= 40:
        return "Emerging"
    return "Weak"


def aggregate_finding_confidence(
    supporting: list[tuple[float, float]],
) -> float:
    """Noisy-OR over supporting articles, damped by source independence.

    finding_conf = 1 - prod(1 - s_i * d_i)

    where each tuple is (article_confidence_0_100, independence_damp). d_i is
    1.0 for an independent originating source and 0.3 for a syndicated copy.
    Two independent 70s beat one 85.
    """
    if not supporting:
        return 0.0
    terms = []
    for conf_100, damp in supporting:
        s = max(0.0, min(1.0, conf_100 / 100.0))
        terms.append(1.0 - s * damp)
    finding = 1.0 - prod(terms)
    return round(max(0.0, min(100.0, finding * 100.0)), 1)


def independence_damp(syndication_count: int) -> float:
    """1.0 for an independent origin, 0.3 for a syndicated copy."""
    return 1.0 if syndication_count <= 1 else 0.3
