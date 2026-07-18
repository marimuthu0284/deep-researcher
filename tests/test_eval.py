from deep_researcher.eval import (
    calibration_checks,
    citation_metrics,
    confidence_metrics,
    evaluate_state,
)
from deep_researcher.models import (
    ArticleAnalysis,
    ArticleBundle,
    ArticleVerdict,
    Chunk,
    Claim,
    CritiqueReport,
    Finding,
    JudgedState,
    PerspectiveBrief,
    ScoredComponent,
)


def _analysis(aid, rejected):
    bundle = ArticleBundle(
        article_id=aid, title=aid, url=f"https://x/{aid}", source_name="x",
        chunks=[Chunk(chunk_id=f"{aid}-c0", text="t")],
    )
    return ArticleAnalysis(
        article_id=aid,
        bundle=bundle,
        advocate=PerspectiveBrief(
            article_id=aid, stance="advocate",
            claims=[Claim(claim_id=f"{aid}-a1", text="c", strength=3)],
        ),
        skeptic=PerspectiveBrief(
            article_id=aid, stance="skeptic",
            claims=[Claim(claim_id=f"{aid}-s1", text="c", strength=3)],
        ),
        critique=CritiqueReport(
            article_id=aid,
            uncited_claims_rejected=rejected,
            source_credibility=ScoredComponent(score=7, justification="j"),
            evidence_strength=ScoredComponent(score=7, justification="j"),
            internal_consistency=ScoredComponent(score=7, justification="j"),
        ),
    )


def _state():
    analyses = [_analysis("a0", ["a0-a1"]), _analysis("a1", [])]
    judged = JudgedState(
        verdicts=[
            ArticleVerdict(
                article_id="a0", confidence_score=85.0,
                score_breakdown={"source_credibility": 9}, band="Strong",
                resolved_position="p",
            ),
            ArticleVerdict(
                article_id="a1", confidence_score=60.0,
                score_breakdown={"source_credibility": 6}, band="Moderate",
                resolved_position="p",
            ),
        ],
        findings=[
            Finding(statement="high", confidence=85.0, band="Strong",
                    supporting_article_ids=["a0", "a1"]),
            Finding(statement="low", confidence=34.0, band="Weak",
                    supporting_article_ids=["a1"], single_thread=True),
        ],
    )
    return {"analyses": analyses, "judged": judged}


def test_citation_metrics():
    m = citation_metrics(_state()["analyses"])
    assert m.total_claims == 4
    assert m.rejected_claims == 1
    assert m.retained_pct == 75.0
    assert m.articles_with_rejections == 1


def test_confidence_metrics():
    m = confidence_metrics(_state()["judged"])
    assert m.n_verdicts == 2
    assert m.mean_confidence == 72.5
    assert m.band_counts["Strong"] == 1
    assert m.band_counts["Moderate"] == 1


def test_calibration_passes_on_wellformed():
    checks = calibration_checks(_state()["judged"])
    assert all(c["passed"] for c in checks)


def test_calibration_flags_single_thread_strong():
    judged = _state()["judged"]
    judged.findings[1].confidence = 90.0  # single-thread but Strong -> should fail
    judged.findings[1].single_thread = True
    checks = calibration_checks(judged)
    by_name = {c["check"]: c["passed"] for c in checks}
    assert by_name["single_thread_not_strong"] is False


def test_evaluate_state_shape():
    report = evaluate_state(_state())
    assert "citation" in report and "confidence" in report and "calibration" in report
    assert report["calibration_passed"] is True
