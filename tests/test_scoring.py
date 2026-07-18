from deep_researcher.scoring import (
    WEIGHTS,
    aggregate_finding_confidence,
    band_short,
    compute_confidence,
    independence_damp,
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_compute_confidence_full_marks():
    components = {k: 10.0 for k in WEIGHTS}
    assert compute_confidence(components) == 100.0


def test_compute_confidence_zero():
    assert compute_confidence({}) == 0.0


def test_compute_confidence_weighted():
    # Only source_credibility (weight .25) at 8/10 -> 8*.25*10 = 20
    assert compute_confidence({"source_credibility": 8.0}) == 20.0


def test_two_independent_beat_one_strong():
    two_independent = aggregate_finding_confidence([(70.0, 1.0), (70.0, 1.0)])
    one_strong = aggregate_finding_confidence([(85.0, 1.0)])
    assert two_independent > one_strong


def test_syndication_damping_reduces_contribution():
    independent = aggregate_finding_confidence([(80.0, 1.0)])
    syndicated = aggregate_finding_confidence([(80.0, 0.3)])
    assert independent > syndicated


def test_independence_damp():
    assert independence_damp(1) == 1.0
    assert independence_damp(5) == 0.3


def test_bands():
    assert band_short(90) == "Strong"
    assert band_short(70) == "Moderate"
    assert band_short(50) == "Emerging"
    assert band_short(20) == "Weak"
