"""Seed a coherent, offline cached run so the replay demo works with no keys.

This writes a synthetic-but-realistic "Ozempic" run (modeled on the design
doc's section 11 walkthrough) into the run cache. After seeding, the Streamlit
"Replay cached run" toggle and `deep-researcher ... --replay` work immediately
without any API keys or network.

The data is illustrative demo content, not real research output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_researcher.cache import run_cache_path, save_run  # noqa: E402
from deep_researcher.models import (  # noqa: E402
    ArticleAnalysis,
    ArticleBundle,
    ArticleVerdict,
    Chunk,
    Claim,
    Contradiction,
    CritiqueReport,
    Disagreement,
    Finding,
    Flag,
    JudgedState,
    PerspectiveBrief,
    ScoredComponent,
)
from deep_researcher.report.templates import (  # noqa: E402
    ReportContext,
    build_html,
    build_markdown,
)
from deep_researcher.scoring import (  # noqa: E402
    aggregate_finding_confidence,
    band,
    band_short,
    compute_confidence,
    independence_damp,
)

TOPIC = "Ozempic"
FILTERS = "peer-reviewed papers and major news, last 30 days"


def _analysis(
    aid: str,
    title: str,
    source_name: str,
    source_type: str,
    url: str,
    breakdown: dict[str, float],
    syndication: int = 1,
    rejected: list[str] | None = None,
    flags: list[str] | None = None,
    contradictions: list[Contradiction] | None = None,
) -> tuple[ArticleAnalysis, ArticleVerdict]:
    chunks = [
        Chunk(chunk_id=f"{aid}-c{i}", text=f"Illustrative evidence chunk {i} for {title}.")
        for i in range(6)
    ]
    bundle = ArticleBundle(
        article_id=aid,
        title=title,
        url=url,
        source_name=source_name,
        source_type=source_type,
        syndication_count=syndication,
        chunks=chunks,
        cluster="efficacy" if source_type == "peer_reviewed" else "coverage",
        triage_rationale="kept: primary data in window"
        if source_type == "peer_reviewed"
        else "kept: tier-1 coverage",
    )
    advocate = PerspectiveBrief(
        article_id=aid,
        stance="advocate",
        claims=[
            Claim(
                claim_id=f"{aid}-a1",
                text="Durable efficacy demonstrated with primary endpoint met.",
                cited_chunk_ids=[f"{aid}-c2", f"{aid}-c5"],
                strength=5,
            ),
            Claim(
                claim_id=f"{aid}-a2",
                text="Effect size is clinically meaningful.",
                cited_chunk_ids=[f"{aid}-c1"],
                strength=4,
            ),
        ],
    )
    skeptic = PerspectiveBrief(
        article_id=aid,
        stance="skeptic",
        claims=[
            Claim(
                claim_id=f"{aid}-s1",
                text="Real-world dropout likely understated relative to trial support.",
                cited_chunk_ids=[f"{aid}-c3"],
                strength=4,
            ),
            Claim(
                claim_id=f"{aid}-s2",
                text="Funding and author conflicts warrant caution.",
                cited_chunk_ids=[f"{aid}-c4"],
                strength=5,
            ),
        ],
    )
    critique = CritiqueReport(
        article_id=aid,
        contradictions=contradictions or [],
        credibility_flags=[Flag(label=f) for f in (flags or [])],
        uncited_claims_rejected=rejected or [],
        evidence_sufficiency="sufficient",
        source_credibility=ScoredComponent(
            score=breakdown["source_credibility"], justification="venue tier and peer-review status"
        ),
        evidence_strength=ScoredComponent(
            score=breakdown["evidence_strength"], justification="primary data and audit survival"
        ),
        internal_consistency=ScoredComponent(
            score=breakdown["internal_consistency"], justification="coherent internal argument"
        ),
        claims_retained_pct=100.0 * (4 - len(rejected or [])) / 4,
    )
    analysis = ArticleAnalysis(
        article_id=aid,
        bundle=bundle,
        advocate=advocate,
        skeptic=skeptic,
        critique=critique,
    )
    confidence = compute_confidence(breakdown)
    verdict = ArticleVerdict(
        article_id=aid,
        confidence_score=confidence,
        score_breakdown=breakdown,
        band=band(confidence),
        resolved_position=(
            "Findings robust for efficacy; real-world durability remains open given "
            "adherence support; COI noted but methodology independently sound."
        ),
        dissent_note="The skeptic's adherence-support caveat is the key limitation.",
    )
    return analysis, verdict


def build_state() -> dict:
    specs = [
        dict(
            aid="a0",
            title="Semaglutide durable efficacy at 104 weeks (RCT)",
            source_name="NEJM",
            source_type="peer_reviewed",
            url="https://www.nejm.org/example-semaglutide-rct",
            breakdown={
                "source_credibility": 9,
                "evidence_strength": 9,
                "corroboration": 7,
                "internal_consistency": 8,
                "recency_relevance": 10,
            },
            rejected=["a0-a2"],
            flags=["COI disclosed", "industry-funded"],
            contradictions=[
                Contradiction(
                    claim_a_id="a0-a1",
                    claim_b_id="a0-s1",
                    nature="interpretive",
                    resolvable=True,
                    note="generalizability weighting differs; both defensible",
                )
            ],
        ),
        dict(
            aid="a1",
            title="GLP-1 receptor agonists and lean-mass loss",
            source_name="Semantic Scholar",
            source_type="peer_reviewed",
            url="https://www.semanticscholar.org/example-glp1-leanmass",
            breakdown={
                "source_credibility": 8,
                "evidence_strength": 8,
                "corroboration": 7,
                "internal_consistency": 8,
                "recency_relevance": 8,
            },
            flags=["single-cohort"],
        ),
        dict(
            aid="a2",
            title="Ozempic supply shortage and compounding pharmacies",
            source_name="reuters.com",
            source_type="news",
            url="https://www.reuters.com/example-ozempic-supply",
            breakdown={
                "source_credibility": 6,
                "evidence_strength": 5,
                "corroboration": 8,
                "internal_consistency": 7,
                "recency_relevance": 9,
            },
            syndication=5,
            flags=["wire syndication"],
        ),
        dict(
            aid="a3",
            title="Preprint: possible cognitive side-effects of semaglutide",
            source_name="arXiv",
            source_type="preprint",
            url="https://arxiv.org/abs/example-semaglutide-cognition",
            breakdown={
                "source_credibility": 3,
                "evidence_strength": 3,
                "corroboration": 2,
                "internal_consistency": 6,
                "recency_relevance": 7,
            },
            flags=["preprint", "not peer-reviewed", "single-source"],
        ),
    ]

    analyses = []
    verdicts = []
    for s in specs:
        a, v = _analysis(**s)
        analyses.append(a)
        verdicts.append(v)

    verdict_by_id = {v.article_id: v for v in verdicts}
    damp_by_id = {a.article_id: independence_damp(a.bundle.syndication_count) for a in analyses}

    def finding(statement: str, ids: list[str], single_thread: bool = False) -> Finding:
        supporting = [(verdict_by_id[i].confidence_score, damp_by_id[i]) for i in ids]
        conf = aggregate_finding_confidence(supporting)
        distinct = sum(1 for i in ids if damp_by_id[i] == 1.0)
        return Finding(
            statement=statement,
            confidence=conf,
            band=band_short(conf),
            supporting_article_ids=ids,
            single_thread=single_thread or distinct <= 1,
        )

    findings = [
        finding("Semaglutide shows durable weight-loss efficacy sustained to ~2 years.", ["a0", "a1"]),
        finding("Lean-mass loss is a real but manageable trade-off of GLP-1 therapy.", ["a1"]),
        finding("Supply shortages persist, driving compounding-pharmacy substitution.", ["a2"]),
        finding("Possible cognitive side-effects are unconfirmed and single-thread.", ["a3"], single_thread=True),
    ]
    findings.sort(key=lambda f: f.confidence, reverse=True)

    judged = JudgedState(
        verdicts=sorted(verdicts, key=lambda v: v.confidence_score, reverse=True),
        findings=findings,
        disagreements=[
            Disagreement(
                topic="Real-world durability",
                summary="Trial efficacy vs. adherence-dependent real-world persistence; "
                "preserved as an open question rather than resolved.",
                article_ids=["a0", "a2"],
            )
        ],
        executive_summary=(
            "Semaglutide's weight-loss efficacy is strongly supported by peer-reviewed, "
            "independent sources, with durable results to two years. Lean-mass loss is a "
            "manageable trade-off. Supply constraints remain a live logistics issue. A "
            "preprint's cognitive-side-effect claim is single-thread and treated as an open "
            "question. Overall the evidence trends positive but with real caveats."
        ),
        trajectory="Evidence over the window trends toward confirmed efficacy with sharpening safety nuance.",
    )

    ctx = ReportContext(
        topic=TOPIC,
        filters=FILTERS,
        judged=judged,
        analyses=analyses,
        n_queries=6,
        n_retrieved=31,
        n_after_triage=len(analyses),
        dedup_count=31 - len(analyses),
    )
    report_md = build_markdown(ctx)
    report_html = build_html(ctx)

    now = time.time()
    status_log = [
        {"ts": now + 0, "agent": "Query Planner", "message": "planned 6 queries across 3 source types"},
        {"ts": now + 1, "agent": "Retriever", "message": "retrieved 31 documents (24 with full text)"},
        {"ts": now + 2, "agent": "Triage", "message": "31 raw -> 22 after dedup (9 duplicates) -> kept top 4"},
        {"ts": now + 3, "agent": "Debate", "message": "a0: 2 advocate / 2 skeptic claims, 1 rejected, sufficiency=sufficient"},
        {"ts": now + 4, "agent": "Critical Analysis", "message": "citation audit caught a mismatched dose-arm claim"},
        {"ts": now + 5, "agent": "Judge", "message": "scored 4 articles; 4 findings, 1 open disagreement"},
        {"ts": now + 6, "agent": "Report Builder", "message": "rendered report (4 article cards, 4 findings)"},
        {"ts": now + 7, "agent": "Dispatcher", "message": "report available at local link"},
    ]
    metrics = [
        {"node": "query_planner", "seconds": 1.1, "ts": now},
        {"node": "retriever", "seconds": 8.4, "ts": now},
        {"node": "triage", "seconds": 2.0, "ts": now},
        {"node": "debate_article", "seconds": 6.2, "ts": now},
        {"node": "debate_article", "seconds": 6.7, "ts": now},
        {"node": "debate_article", "seconds": 5.9, "ts": now},
        {"node": "debate_article", "seconds": 6.1, "ts": now},
        {"node": "judge", "seconds": 4.3, "ts": now},
        {"node": "report_builder", "seconds": 0.2, "ts": now},
        {"node": "dispatcher", "seconds": 0.3, "ts": now},
    ]

    return {
        "topic": TOPIC,
        "filters": FILTERS,
        "analyses": analyses,
        "judged": judged,
        "bundles": [a.bundle for a in analyses],
        "raw_docs": [a.bundle for a in analyses],
        "report_md": report_md,
        "report_html": report_html,
        "delivery_status": {
            "delivered": False,
            "channel": "fallback_link",
            "reason": "seeded demo",
        },
        "status_log": status_log,
        "errors": [],
        "metrics": metrics,
    }


def main() -> int:
    state = build_state()
    path = save_run(TOPIC, FILTERS, state)
    print(f"Seeded demo run for '{TOPIC}' -> {path}")
    print(f"Replay with:\n  deep-researcher \"{TOPIC}\" --filters \"{FILTERS}\" --replay")
    print(f"Cache key path: {run_cache_path(TOPIC, FILTERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
