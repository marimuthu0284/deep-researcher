"""Typed inter-agent contracts (the actual protocol between agents).

Every hop between agents is one of these schema-validated Pydantic objects,
never free-form prose. This is what makes the pipeline debuggable, resumable,
and demo-safe, and lets the UI render the live state object mutating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["peer_reviewed", "preprint", "news", "report", "blog"]
Stance = Literal["advocate", "skeptic"]
ContradictionKind = Literal["factual", "interpretive", "scope"]
EvidenceSufficiency = Literal["sufficient", "thin", "insufficient"]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
class SearchQuery(BaseModel):
    query_str: str
    source_type: SourceType
    # Free-form machine params passed to the mapped search tool
    # (e.g. {"published_after": "2026-06-18", "venue": "NEJM"}).
    api_params: dict = Field(default_factory=dict)
    facet: str = Field(
        default="",
        description="Which facet of the topic this query targets (mechanism, "
        "outcomes, controversy, regulation, economics, criticism).",
    )


class FilterSpec(BaseModel):
    published_after: datetime | None = None
    published_before: datetime | None = None
    source_types: list[SourceType] = Field(default_factory=list)
    domain_allowlist: list[str] = Field(default_factory=list)
    country: str | None = None
    peer_reviewed_only: bool = False
    raw: str = Field(default="", description="Original human filter string.")


class SearchPlan(BaseModel):
    queries: list[SearchQuery]
    filter_spec: FilterSpec


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    chunk_id: str  # {article_id}-c{n} — the citation anchor
    text: str


class ArticleBundle(BaseModel):
    article_id: str
    title: str
    url: str
    source_name: str
    published_at: datetime | None = None
    source_type: SourceType = "news"
    full_text: bool = True
    chunks: list[Chunk] = Field(default_factory=list)

    # Retrieval / triage annotations
    citation_count: int | None = None  # from Semantic Scholar where available
    syndication_count: int = 1  # distinct originating copies (corroboration signal)
    cluster: str | None = None
    relevance: float | None = None  # LLM-scored 1-5
    triage_rationale: str | None = None

    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    claim_id: str
    text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    strength: int = Field(ge=1, le=5)


class PerspectiveBrief(BaseModel):
    article_id: str
    stance: Stance
    claims: list[Claim]


class Contradiction(BaseModel):
    claim_a_id: str
    claim_b_id: str
    nature: ContradictionKind
    resolvable: bool
    note: str = ""


class Flag(BaseModel):
    label: str  # e.g. "single-source", "preprint", "COI disclosed"
    detail: str = ""


class ScoredComponent(BaseModel):
    """A rubric component: the LLM supplies value (0-10) and a justification.

    The justification is required by the schema so unjustified scores are
    rejected before they can reach the (code-computed) total.
    """

    score: float = Field(ge=0, le=10)
    justification: str = Field(min_length=1)


class CritiqueReport(BaseModel):
    article_id: str
    contradictions: list[Contradiction] = Field(default_factory=list)
    credibility_flags: list[Flag] = Field(default_factory=list)
    uncited_claims_rejected: list[str] = Field(default_factory=list)
    evidence_sufficiency: EvidenceSufficiency = "sufficient"

    # Rubric components scored by the Critical Analyst (§6). Corroboration and
    # recency are filled by Judge/Triage respectively; kept optional here.
    source_credibility: ScoredComponent
    evidence_strength: ScoredComponent
    internal_consistency: ScoredComponent
    claims_retained_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Percent of submitted claims that survived the citation audit.",
    )


class ArticleAnalysis(BaseModel):
    """The reduced per-article unit: bundle + both briefs + critique."""

    article_id: str
    bundle: ArticleBundle
    advocate: PerspectiveBrief
    skeptic: PerspectiveBrief
    critique: CritiqueReport
    reretrieved: bool = False


# --------------------------------------------------------------------------- #
# Judgement & synthesis
# --------------------------------------------------------------------------- #
class ArticleVerdict(BaseModel):
    article_id: str
    confidence_score: float  # 0-100, computed in code from score_breakdown
    score_breakdown: dict[str, float]  # rubric components (0-10), shown in report
    band: str = ""  # calibration band label
    resolved_position: str
    dissent_note: str | None = None


class Finding(BaseModel):
    statement: str
    confidence: float  # 0-100, aggregated via noisy-OR over supporting articles
    band: str = ""
    supporting_article_ids: list[str] = Field(default_factory=list)
    single_thread: bool = False


class Disagreement(BaseModel):
    topic: str
    summary: str
    article_ids: list[str] = Field(default_factory=list)


class JudgedState(BaseModel):
    verdicts: list[ArticleVerdict] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    executive_summary: str = ""
    trajectory: str = ""  # how the evidence trends across the filtered window
    # True when the Judge could not run real synthesis (LLM unavailable) and the
    # findings below are automated fallbacks, not analysis. Surfaced as a banner
    # in the report so a degraded run is never mistaken for a real one.
    synthesis_degraded: bool = False
    degraded_articles: int = 0  # per-article verdicts that fell back
