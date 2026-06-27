"""Shared Pydantic schemas for the Policy Think Tank.

These models are the FROZEN contract between the four workstreams. Do not change a
field without team sign-off (see ARCHITECTURE.md / the refactor plan). Every public
seam function consumes and produces objects defined here.

Adapted from TAU Group's ThinkTank (MIT). The legacy meeting models are retained at
the bottom of this file only to keep transitional imports working; they are not part
of the policy contract and will be removed once the old entrypoints are deleted.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Source / evidence layer
# ---------------------------------------------------------------------------

SourceType = Literal[
    "government_report",
    "open_data",
    "academic",
    "case_study",
    "uploaded_pdf",
    "structured_csv",
    "other",
]

Severity = Literal["low", "medium", "high"]

# Worker agent types the Policy Director dispatches. The Director also assigns each
# task a skill set (from skills_registry) — skills are orchestrator-decided, not
# hardcoded on the agent.
AgentType = Literal["research", "stakeholder", "data_analyst"]


class EvidenceItem(BaseModel):
    """A single retrieved, citable chunk of evidence with provenance metadata."""

    source_id: str
    title: str
    organization: Optional[str] = None
    source_type: SourceType = "other"
    publication_date: Optional[str] = None
    geography: Optional[str] = None
    page: Optional[int] = None
    text: str
    relevance_score: float = 0.0
    credibility_score: float = 0.0


class Finding(BaseModel):
    """A claim grounded in evidence, with explicit assumptions and limitations."""

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request / planning layer
# ---------------------------------------------------------------------------


class PolicyRequest(BaseModel):
    """The user's policy question and framing — the single entry point."""

    question: str
    geography: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    # Paths to optional supporting files the user uploaded (PDF/CSV in workspace/).
    uploaded_files: list[str] = Field(default_factory=list)


class PolicyObjective(BaseModel):
    """The Policy Director's structured interpretation of the request."""

    statement: str
    success_metrics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)


class StakeholderProfile(BaseModel):
    """A research personality / perspective spun up from the shared agent."""

    key: str  # stable id, e.g. "community_equity"
    name: str  # display name, e.g. "Community & Equity"
    perspective: str  # short description of the lens
    priorities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)  # skill keys to load


class PolicyTask(BaseModel):
    """A task delegated by the Director to a worker agent.

    `agent_type` says which worker handles it; `skills` are the skill keys the
    Director decided this task needs (assigned at plan time from the registry).
    `stakeholder_key` is only meaningful for stakeholder tasks.
    """

    task_id: str
    description: str
    agent_type: AgentType = "stakeholder"
    skills: list[str] = Field(default_factory=list)
    stakeholder_key: str = ""
    queries: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Research results layer
# ---------------------------------------------------------------------------


class StakeholderResearchResult(BaseModel):
    """One stakeholder agent's findings from its assigned perspective."""

    stakeholder: str
    findings: list[Finding] = Field(default_factory=list)
    likely_position: str = ""
    concerns: list[str] = Field(default_factory=list)
    proposed_mitigations: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    # Compact handoff summary stored alongside the full result (context strategy).
    handoff_summary: str = ""


class ResearchBrief(BaseModel):
    """Output of a Research agent: objective, cited findings on one sub-topic
    (no stakeholder perspective). Feeds the stakeholders and the synthesis."""

    topic: str
    findings: list[Finding] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    skills_used: list[str] = Field(default_factory=list)


class ResearchSynthesis(BaseModel):
    """Cross-stakeholder synthesis produced before recommendation."""

    summary: str
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    key_findings: list[Finding] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Recommendation / implementation layer
# ---------------------------------------------------------------------------


class ImpactAnalysis(BaseModel):
    """Economic / equity / political / operational / legal feasibility view."""

    economic: str = ""
    equity: str = ""
    political: str = ""
    operational: str = ""
    legal: str = ""
    notes: list[str] = Field(default_factory=list)


class PolicyAlternative(BaseModel):
    """A candidate policy option compared against the recommendation."""

    name: str
    description: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class ImplementationStep(BaseModel):
    """One concrete phase of the rollout plan."""

    phase: str
    actions: list[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    owner: Optional[str] = None


class ImplementationPlan(BaseModel):
    """Phased rollout plus monitoring requirements."""

    steps: list[ImplementationStep] = Field(default_factory=list)
    success_metrics: list[str] = Field(default_factory=list)
    monitoring: list[str] = Field(default_factory=list)


class PolicyRecommendation(BaseModel):
    """The Implementation agent's recommended policy design."""

    summary: str
    recommended_actions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    equity_effects: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    impact: Optional[ImpactAnalysis] = None
    implementation_plan: Optional[ImplementationPlan] = None
    alternatives_detail: list[PolicyAlternative] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Red-team layer
# ---------------------------------------------------------------------------


class CritiqueResult(BaseModel):
    """The Red Team's structured challenge to a recommendation."""

    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unintended_consequences: list[str] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)
    severity: Severity = "low"


# ---------------------------------------------------------------------------
# Forecasting layer (deterministic — no LLM arithmetic)
# ---------------------------------------------------------------------------


class ForecastParameters(BaseModel):
    """Scenario inputs for the congestion-pricing demo forecaster.

    Kept domain-specific but extensible: unknown levers go in `extra`.
    """

    daily_fee: float = 0.0
    affected_trips_per_day: int = 0
    behavioral_response_rate: float = 0.0  # fraction of trips deterred
    exemption_rate: float = 0.0
    enforcement_effectiveness: float = 1.0
    operating_days_per_year: int = 250
    implementation_cost: float = 0.0
    transit_reinvestment_pct: float = 0.0
    extra: dict[str, float] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Deterministic outputs for one scenario (baseline/conservative/...)."""

    name: str
    trip_reduction: float = 0.0
    gross_revenue: float = 0.0
    net_revenue: float = 0.0
    transit_demand_increase: float = 0.0
    emissions_change: float = 0.0
    commuter_burden: float = 0.0
    equity_risk_index: float = 0.0
    inputs: dict[str, float] = Field(default_factory=dict)


ForecastMode = Literal["numeric", "qualitative"]


class ForecastResult(BaseModel):
    """Forecast output, in one of two modes:

    - "numeric": a deterministic domain module produced the four scenarios below.
    - "qualitative": no domain module matched, so only a directional outlook is
      given (`qualitative`) with NO fabricated numbers. Scenario fields stay None.
    """

    mode: ForecastMode = "numeric"
    domain: str = "generic"
    baseline: Optional[ScenarioResult] = None
    conservative: Optional[ScenarioResult] = None
    expected: Optional[ScenarioResult] = None
    optimistic: Optional[ScenarioResult] = None
    qualitative: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sensitivity: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observability + final result
# ---------------------------------------------------------------------------


class ModelEvent(BaseModel):
    """One model call, logged for the model-selection dashboard."""

    agent: str
    model: str
    latency_ms: int = 0
    schema_valid: bool = True
    escalated: bool = False
    error: Optional[str] = None


class PolicyRunResult(BaseModel):
    """The complete output of one policy analysis — what the UI renders."""

    run_id: str
    request: PolicyRequest
    objective: Optional[PolicyObjective] = None
    stakeholders: list[StakeholderProfile] = Field(default_factory=list)
    tasks: list[PolicyTask] = Field(default_factory=list)
    research_briefs: list[ResearchBrief] = Field(default_factory=list)
    research: list[StakeholderResearchResult] = Field(default_factory=list)
    synthesis: Optional[ResearchSynthesis] = None
    recommendation: Optional[PolicyRecommendation] = None
    critiques: list[CritiqueResult] = Field(default_factory=list)
    revisions: int = 0
    forecast: Optional[ForecastResult] = None
    forecast_parameters: Optional[ForecastParameters] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    model_events: list[ModelEvent] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)  # human-readable activity log


# ---------------------------------------------------------------------------
# Legacy meeting models (transitional — to be removed with old entrypoints)
# ---------------------------------------------------------------------------


class FileData(BaseModel):
    filename: str
    content: str  # base64-encoded file content


class FileReference(BaseModel):
    original_name: str
    size: int


class Expert(BaseModel):
    title: str
    expertise: str
    goal: str
    role: str


class Meeting(BaseModel):
    project_name: str
    experts: list[Expert]
    vector_store: Optional[list[list[FileData]]] = Field(default_factory=list)
    file_references: Optional[dict[str, list[FileReference]]] = Field(default_factory=dict)
    session_id: Optional[str] = None
    meeting_topic: str
    rounds: int
    timestamp: Optional[int] = None
    transcript: Optional[list[dict[str, str]]] = Field(default_factory=list)
    summary: Optional[str] = ""

    def serialize(self) -> dict:
        return {
            "project_name": self.project_name,
            "experts": [e.model_dump() for e in self.experts],
            "vector_store": self.vector_store,
            "meeting_topic": self.meeting_topic,
            "rounds": self.rounds,
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "summary": self.summary,
        }


class Project(BaseModel):
    title: str
    description: str
    meetings: list[Meeting]

    def serialize(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "meetings": [m.serialize() for m in self.meetings],
        }
