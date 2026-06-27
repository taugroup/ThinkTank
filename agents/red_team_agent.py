"""Red-Team agent.

OWNER: Person 3 (impl/forecast/evals). Challenges the recommendation for weak claims,
unintended consequences, missing perspectives, and equity blind spots, and decides
whether a revision is required. Domain-neutral (forecast parameters belong to the
matched domain module in `forecasters/`).

Real path (POLICY_MOCK_ANALYSIS=0) follows the Director template; falls back to mock.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent_builder import run_structured
from config import MOCK_ANALYSIS
from context_builder import build_packet
from models import CritiqueResult, Finding, PolicyRecommendation, PolicyRequest


class _CritiqueOut(BaseModel):
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unintended_consequences: list[str] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)
    required_revisions: list[str] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"


def _mock_review(
    request: PolicyRequest, recommendation: PolicyRecommendation
) -> CritiqueResult:
    unsupported = [
        a for a in recommendation.recommended_actions if not recommendation.evidence_ids
    ]
    has_equity = bool(recommendation.equity_effects)
    return CritiqueResult(
        issues=[
            f"Key projections rest on assumptions not yet validated for {request.geography}.",
            "Second-order and displacement effects are acknowledged but not quantified.",
        ],
        unsupported_claims=unsupported
        or ["Stated confidence level lacks an accompanying sensitivity analysis."],
        unintended_consequences=[
            "Effects may spill over to adjacent populations or jurisdictions.",
            "Affected groups may adapt in ways that erode the intended benefit.",
        ],
        missing_perspectives=[
            "Frontline implementers / operators",
            "Groups most exposed to the downside risks",
        ],
        required_revisions=[
            "Add a concrete equity-protection mechanism before rollout"
            if not has_equity
            else "Define monitoring triggers for the identified second-order effects",
        ],
        severity="high" if not has_equity else "medium",
    )


def red_team_review(
    request: PolicyRequest, recommendation: PolicyRecommendation
) -> CritiqueResult:
    """Challenge the recommendation for weak claims, risks, and gaps."""
    if MOCK_ANALYSIS:
        return _mock_review(request, recommendation)

    packet = build_packet(
        request,
        perspective="You are an adversarial red-team reviewer.",
        prior_findings=[Finding(claim=recommendation.summary)]
        + [Finding(claim=a) for a in recommendation.recommended_actions],
        output_schema_name="_CritiqueOut",
    )
    prompt = packet.to_prompt() + (
        "\n\nChallenge the recommendation above. Return JSON with keys: issues[], "
        "unsupported_claims[], unintended_consequences[], missing_perspectives[], "
        "required_revisions[], severity ('low'|'medium'|'high'). Mark severity 'high' "
        "only for material flaws (e.g. unaddressed inequitable effects)."
    )
    out, _ = run_structured("red_team", prompt, _CritiqueOut)
    if out is None or not out.issues:
        return _mock_review(request, recommendation)
    return CritiqueResult(
        issues=out.issues,
        unsupported_claims=out.unsupported_claims,
        unintended_consequences=out.unintended_consequences,
        missing_perspectives=out.missing_perspectives,
        required_revisions=out.required_revisions,
        severity=out.severity,
    )
