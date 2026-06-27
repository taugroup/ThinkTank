"""Policy implementation & recommendation agent.

OWNER: Person 3 (impl/forecast/evals). Synthesizes stakeholder findings, analyzes
feasibility, compares alternatives, and produces a recommendation with a phased
implementation plan and success metrics.

Real path (POLICY_MOCK_ANALYSIS=0) follows the Director template and falls back to
the deterministic mock on any failure.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_builder import run_structured
from config import MOCK_ANALYSIS
from context_builder import build_packet
from models import (
    Finding,
    ImpactAnalysis,
    ImplementationPlan,
    ImplementationStep,
    PolicyAlternative,
    PolicyRecommendation,
    PolicyRequest,
    ResearchSynthesis,
    StakeholderResearchResult,
)


def _evidence_ids(research: list[StakeholderResearchResult]) -> list[str]:
    return sorted({eid for r in research for f in r.findings for eid in f.evidence_ids})


# --- Synthesis -------------------------------------------------------------
class _SynthOut(BaseModel):
    summary: str
    consensus_points: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)


def _mock_synthesis(
    request: PolicyRequest, research: list[StakeholderResearchResult]
) -> ResearchSynthesis:
    all_findings: list[Finding] = [f for r in research for f in r.findings]
    return ResearchSynthesis(
        summary=f"Across {len(research)} stakeholder perspectives, there is qualified "
        f"support for action on '{request.question}', contingent on equity safeguards.",
        consensus_points=[
            "Some intervention is justified by the evidence.",
            "Mitigations are required to protect vulnerable groups.",
        ],
        disagreements=["Stakeholders differ on the acceptable level of cost."],
        key_findings=all_findings[:5],
        evidence_ids=_evidence_ids(research),
        data_gaps=sorted({g for r in research for g in r.data_gaps}),
    )


def synthesize_research(
    request: PolicyRequest, research: list[StakeholderResearchResult]
) -> ResearchSynthesis:
    """Combine stakeholder findings into a cross-cutting synthesis."""
    if MOCK_ANALYSIS:
        return _mock_synthesis(request, research)

    prior = [f"{r.stakeholder}: {r.handoff_summary}" for r in research]
    packet = build_packet(
        request,
        perspective="You synthesize multiple stakeholder findings.",
        prior_findings=[Finding(claim=p) for p in prior],
        skill_keys=["policy-analysis"],
        output_schema_name="_SynthOut",
    )
    prompt = packet.to_prompt() + (
        "\n\nReturn JSON with keys: summary (string), consensus_points[], "
        "disagreements[]."
    )
    out, _ = run_structured("synthesis", prompt, _SynthOut)
    if out is None:
        return _mock_synthesis(request, research)
    base = _mock_synthesis(request, research)  # reuse programmatic findings/evidence
    base.summary = out.summary or base.summary
    base.consensus_points = out.consensus_points or base.consensus_points
    base.disagreements = out.disagreements or base.disagreements
    return base


# --- Recommendation --------------------------------------------------------
class _StepOut(BaseModel):
    phase: str
    actions: list[str] = Field(default_factory=list)
    timeline: str | None = None


class _RecOut(BaseModel):
    summary: str
    recommended_actions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    equity_effects: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    steps: list[_StepOut] = Field(default_factory=list)


def _default_plan(request: PolicyRequest) -> ImplementationPlan:
    return ImplementationPlan(
        steps=[
            ImplementationStep(
                phase="Phase 1: Design & Authorization",
                actions=[
                    "Confirm legal authority and finalize the policy design",
                    "Define eligibility, scope, and protective carve-outs",
                ],
                timeline="Months 0-6",
                owner="Lead agency",
            ),
            ImplementationStep(
                phase="Phase 2: Pilot & Build",
                actions=[
                    "Stand up the operational and data infrastructure",
                    "Run a limited pilot with monitoring and feedback",
                ],
                timeline="Months 6-18",
                owner="Lead agency + delivery partners",
            ),
            ImplementationStep(
                phase="Phase 3: Rollout & Reinvestment",
                actions=[
                    "Scale based on pilot results",
                    "Direct gains toward affected and underserved groups",
                ],
                timeline="Months 18-36",
                owner="Oversight body",
            ),
        ],
        success_metrics=[
            "Measurable progress toward the stated objective vs. baseline",
            "Costs remain within the stated constraints",
            "No net harm to the most affected groups",
        ],
        monitoring=[
            "Quarterly outcome reporting against the success metrics",
            "Annual equity-impact audit",
        ],
    )


def _mock_recommendation(
    request: PolicyRequest, research: list[StakeholderResearchResult]
) -> PolicyRecommendation:
    topic = request.question.rstrip("?")
    return PolicyRecommendation(
        summary=f"Adopt a phased, equity-protected approach to '{topic}' in "
        f"{request.geography}, piloted before full rollout and monitored against "
        f"clear success metrics.",
        recommended_actions=[
            f"Implement the core intervention addressing: {topic}",
            "Add targeted protections for the most affected groups",
            "Direct benefits/savings toward equity and reinvestment",
        ],
        alternatives=["No action", "A narrower / lower-cost variant", "A phased pilot only"],
        benefits=[
            "Direct progress on the stated objective",
            "A dedicated mechanism to fund follow-through",
            "Reduced long-term costs of inaction",
        ],
        risks=[
            "Regressive impact if protections are poorly designed",
            "Displacement or spillover of the problem to nearby groups",
        ],
        equity_effects=[
            "Vulnerable groups could be disproportionately burdened without safeguards",
            "Targeted reinvestment can offset burden if directed to underserved groups",
        ],
        evidence_ids=_evidence_ids(research),
        confidence=0.7,
        impact=ImpactAnalysis(
            economic="Net positive if benefits are realized; transition costs for some groups.",
            equity="Risk of regressivity, mitigable via safeguards and targeted reinvestment.",
            political="Contested; visible safeguards and reinvestment build support.",
            operational="Feasible with established delivery mechanisms.",
            legal="Requires confirmation of authorizing authority.",
        ),
        implementation_plan=_default_plan(request),
        alternatives_detail=[
            PolicyAlternative(
                name="No action",
                description="Maintain the status quo.",
                pros=["No implementation cost", "No political friction"],
                cons=["The underlying problem persists", "No new capacity to act"],
            ),
            PolicyAlternative(
                name="Narrower variant",
                description="A smaller-scope, lower-cost version of the intervention.",
                pros=["Lower cost and risk", "Faster to deliver"],
                cons=["Weaker effect on the objective", "May not reach key groups"],
            ),
        ],
    )


def create_policy_recommendation(
    request: PolicyRequest, research: list[StakeholderResearchResult]
) -> PolicyRecommendation:
    """Produce a recommendation, alternatives, and a phased implementation plan."""
    if MOCK_ANALYSIS:
        return _mock_recommendation(request, research)

    prior = [f"{r.stakeholder}: {r.handoff_summary}" for r in research]
    packet = build_packet(
        request,
        perspective="You design the recommended policy and implementation plan.",
        prior_findings=[Finding(claim=p) for p in prior],
        skill_keys=["policy-analysis"],
        output_schema_name="_RecOut",
    )
    prompt = packet.to_prompt() + (
        "\n\nReturn JSON with keys: summary, recommended_actions[], alternatives[], "
        "benefits[], risks[], equity_effects[], confidence (0-1), steps (array of "
        "{phase, actions[], timeline}). Include at least 3 implementation steps and "
        "discuss equity_effects explicitly."
    )
    out, _ = run_structured("implementation_agent", prompt, _RecOut)
    if out is None or not out.recommended_actions:
        return _mock_recommendation(request, research)

    plan = (
        ImplementationPlan(
            steps=[
                ImplementationStep(phase=s.phase, actions=s.actions, timeline=s.timeline)
                for s in out.steps
            ],
            success_metrics=_default_plan(request).success_metrics,
            monitoring=_default_plan(request).monitoring,
        )
        if out.steps
        else _default_plan(request)
    )
    return PolicyRecommendation(
        summary=out.summary,
        recommended_actions=out.recommended_actions,
        alternatives=out.alternatives,
        benefits=out.benefits,
        risks=out.risks,
        equity_effects=out.equity_effects or _mock_recommendation(request, research).equity_effects,
        evidence_ids=_evidence_ids(research),
        confidence=out.confidence,
        impact=_mock_recommendation(request, research).impact,
        implementation_plan=plan,
        alternatives_detail=_mock_recommendation(request, research).alternatives_detail,
    )
