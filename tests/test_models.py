import pytest
from pydantic import ValidationError

from deep_researcher.models import (
    ArticleBundle,
    Chunk,
    Claim,
    CritiqueReport,
    PerspectiveBrief,
    ScoredComponent,
)


def test_article_bundle_roundtrip():
    b = ArticleBundle(
        article_id="a0",
        title="T",
        url="https://example.com",
        source_name="Example",
        source_type="news",
        chunks=[Chunk(chunk_id="a0-c0", text="hello")],
    )
    again = ArticleBundle.model_validate(b.model_dump())
    assert again.chunk_ids() == ["a0-c0"]
    assert again.syndication_count == 1


def test_claim_strength_bounds():
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", text="x", strength=6)


def test_scored_component_requires_justification():
    with pytest.raises(ValidationError):
        ScoredComponent(score=8, justification="")


def test_perspective_brief_stance_literal():
    brief = PerspectiveBrief(article_id="a0", stance="advocate", claims=[])
    assert brief.stance == "advocate"
    with pytest.raises(ValidationError):
        PerspectiveBrief(article_id="a0", stance="cheerleader", claims=[])


def test_critique_report_defaults():
    c = CritiqueReport(
        article_id="a0",
        source_credibility=ScoredComponent(score=7, justification="tier 1"),
        evidence_strength=ScoredComponent(score=6, justification="primary data"),
        internal_consistency=ScoredComponent(score=8, justification="coherent"),
    )
    assert c.evidence_sufficiency == "sufficient"
    assert c.claims_retained_pct == 100.0
