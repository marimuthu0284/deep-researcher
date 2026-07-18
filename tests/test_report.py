from deep_researcher.models import (
    ArticleAnalysis,
    ArticleBundle,
    ArticleVerdict,
    Chunk,
    CritiqueReport,
    Finding,
    JudgedState,
    PerspectiveBrief,
    ScoredComponent,
)
from deep_researcher.report.templates import ReportContext, build_html, build_markdown


def _fixture():
    bundle = ArticleBundle(
        article_id="a0",
        title="A durable efficacy trial",
        url="https://nejm.org/x",
        source_name="NEJM",
        source_type="peer_reviewed",
        chunks=[Chunk(chunk_id="a0-c0", text="15% weight loss at 104 weeks")],
    )
    critique = CritiqueReport(
        article_id="a0",
        uncited_claims_rejected=["c3"],
        source_credibility=ScoredComponent(score=9, justification="tier 1"),
        evidence_strength=ScoredComponent(score=9, justification="primary data"),
        internal_consistency=ScoredComponent(score=8, justification="coherent"),
    )
    analysis = ArticleAnalysis(
        article_id="a0",
        bundle=bundle,
        advocate=PerspectiveBrief(article_id="a0", stance="advocate", claims=[]),
        skeptic=PerspectiveBrief(article_id="a0", stance="skeptic", claims=[]),
        critique=critique,
    )
    judged = JudgedState(
        verdicts=[
            ArticleVerdict(
                article_id="a0",
                confidence_score=85.0,
                score_breakdown={
                    "source_credibility": 9,
                    "evidence_strength": 9,
                    "corroboration": 7,
                    "internal_consistency": 8,
                    "recency_relevance": 10,
                },
                band="Strong",
                resolved_position="Efficacy robust.",
                dissent_note="Adherence support matters.",
            )
        ],
        findings=[
            Finding(
                statement="Durable efficacy at 2 years",
                confidence=85.0,
                band="Strong",
                supporting_article_ids=["a0"],
            )
        ],
        executive_summary="Strong evidence for durable efficacy.",
    )
    return ReportContext(
        topic="Ozempic",
        filters="last 30 days",
        judged=judged,
        analyses=[analysis],
        n_queries=6,
        n_retrieved=31,
        n_after_triage=1,
        dedup_count=30,
    )


def test_markdown_contains_key_sections():
    md = build_markdown(_fixture())
    assert "Executive summary" in md
    assert "Key findings" in md
    assert "Methodology appendix" in md
    assert "Confidence: 85" in md
    assert "[1]" in md  # citation index


def test_html_renders_and_escapes():
    html = build_html(_fixture())
    assert "<h1" in html
    assert "Confidence 85" in html
    assert "nejm.org" in html


def test_no_degraded_banner_on_healthy_run():
    ctx = _fixture()
    assert "Degraded run" not in build_markdown(ctx)
    assert "Degraded run" not in build_html(ctx)


def test_degraded_banner_shown_when_synthesis_degraded():
    ctx = _fixture()
    ctx.judged.synthesis_degraded = True
    md = build_markdown(ctx)
    html = build_html(ctx)
    assert "Degraded run" in md
    assert "Degraded run" in html
    assert "not a full analysis" in md.lower()
    assert "not a full analysis" in html.lower()
