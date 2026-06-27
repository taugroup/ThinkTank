"""Policy Director agent.

OWNER: Person 1 (orchestration). Interprets the request into a structured objective,
defines the stakeholder roster + research tasks, and revises the recommendation when
the Red Team demands it.

Real path (POLICY_MOCK_DIRECTOR=0) calls the shared `run_structured` wrapper against
a local model; on any failure it falls back to the deterministic mock so the app
keeps working. This module is the TEMPLATE the other agents follow.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_builder import run_structured
from config import MOCK_DIRECTOR
from context_builder import build_packet
from models import (
    CritiqueResult,
    PolicyObjective,
    PolicyRecommendation,
    PolicyRequest,
    PolicyTask,
    StakeholderProfile,
)
from skills_registry import skill_catalog_text, valid_skills

# Default skill assignments per agent type (used by the mock and as a safety net).
_DEFAULT_SKILLS = {
    "research": ["policy-research"],
    "stakeholder": ["policy-research"],
    "data_analyst": ["policy-analysis"],
}

# Default domain-neutral stakeholder roster. Used by the mock and as a fallback.
_DEFAULT_STAKEHOLDERS = [
    StakeholderProfile(
        key="government_regulatory",
        name="Government & Regulatory",
        perspective="Legal authority, fiscal responsibility, and regulatory feasibility.",
        priorities=["legal authority", "budget", "enforceability"],
        skills=["policy-research"],
    ),
    StakeholderProfile(
        key="community_equity",
        name="Community & Equity",
        perspective="Distributional fairness and impact on vulnerable populations.",
        priorities=["equity", "affordability", "access"],
        skills=["policy-research"],
    ),
    StakeholderProfile(
        key="economic_fiscal",
        name="Economic & Fiscal",
        perspective="Effects on the economy, employment, costs, and public finances.",
        priorities=["economic impact", "cost", "competitiveness"],
        skills=["policy-research", "policy-analysis"],  # needs cost/benefit analysis too
    ),
    StakeholderProfile(
        key="subject_matter_expert",
        name="Subject-Matter & Implementation Expert",
        perspective="Technical evidence and operational feasibility for this issue.",
        priorities=["evidence base", "operational feasibility", "unintended effects"],
        skills=["policy-research"],
    ),
]


# --- LLM output schemas (internal; mapped onto the frozen models) ----------
class _StakeholderOut(BaseModel):
    key: str
    name: str
    perspective: str
    priorities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)  # Director-chosen, from catalog


class _PlanOut(BaseModel):
    objective: str
    success_metrics: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    research_topics: list[str] = Field(default_factory=list)
    stakeholders: list[_StakeholderOut] = Field(default_factory=list)


def _research_tasks(request: PolicyRequest, topics: list[str]) -> list[PolicyTask]:
    skills = valid_skills(_DEFAULT_SKILLS["research"])
    return [
        PolicyTask(
            task_id=f"research_{i}",
            description=f"Gather objective evidence on: {topic}",
            agent_type="research",
            skills=skills,
            queries=[f"{topic} {request.geography}", topic],
            required_outputs=["cited findings"],
        )
        for i, topic in enumerate(topics)
    ]


def _stakeholder_tasks(
    request: PolicyRequest, stakeholders: list[StakeholderProfile]
) -> list[PolicyTask]:
    return [
        PolicyTask(
            task_id=f"task_{s.key}",
            description=f"Analyze {request.question} from the {s.name} perspective.",
            agent_type="stakeholder",
            skills=s.skills or valid_skills(_DEFAULT_SKILLS["stakeholder"]),
            stakeholder_key=s.key,
            queries=[f"{request.question} {p} {request.geography}" for p in s.priorities],
            required_outputs=["cited findings", "likely position", "concerns"],
        )
        for s in stakeholders
    ]


def _mock_plan(
    request: PolicyRequest,
) -> tuple[PolicyObjective, list[StakeholderProfile], list[PolicyTask]]:
    objective = PolicyObjective(
        statement=request.objective or f"Evaluate: {request.question}",
        success_metrics=[
            "Measurable progress toward the stated objective",
            "Costs justified by benefits, within the stated constraints",
            "No disproportionate burden on vulnerable or affected groups",
        ],
        constraints=list(request.constraints),
        out_of_scope=["Issues outside the stated geography or objective"],
    )
    stakeholders = list(_DEFAULT_STAKEHOLDERS)
    topics = [
        f"benefits and effectiveness of addressing {request.question.rstrip('?')}",
        "risks, costs, and comparable cases from other jurisdictions",
    ]
    tasks = _research_tasks(request, topics) + _stakeholder_tasks(request, stakeholders)
    return objective, stakeholders, tasks


def plan_policy(
    request: PolicyRequest,
) -> tuple[PolicyObjective, list[StakeholderProfile], list[PolicyTask]]:
    """Define objective + success criteria, pick stakeholders, assign each agent its
    skills, and delegate research + stakeholder tasks."""
    if MOCK_DIRECTOR:
        return _mock_plan(request)

    packet = build_packet(
        request,
        task=None,
        perspective="You are the Policy Director planning the analysis.",
        output_schema_name="_PlanOut",
    )
    prompt = (
        packet.to_prompt()
        + "\n\nAVAILABLE SKILLS (assign each stakeholder the skills it needs):\n"
        + skill_catalog_text()
        + "\n\nReturn JSON with keys: objective (string), success_metrics (string[]), "
        "out_of_scope (string[]), research_topics (string[] — 2-3 objective evidence "
        "topics to investigate), stakeholders (array of {key, name, perspective, "
        "priorities[], skills[]}). Choose 3-5 stakeholders whose perspectives best fit "
        "THIS question, and for each choose skills ONLY from the available list above."
    )
    plan, _ = run_structured("policy_director", prompt, _PlanOut)
    if plan is None or not plan.stakeholders:
        return _mock_plan(request)  # fallback keeps the app working

    objective = PolicyObjective(
        statement=plan.objective or request.objective or request.question,
        success_metrics=plan.success_metrics or [],
        constraints=list(request.constraints),
        out_of_scope=plan.out_of_scope or [],
    )
    stakeholders = [
        StakeholderProfile(
            key=s.key or s.name.lower().replace(" ", "_"),
            name=s.name,
            perspective=s.perspective,
            priorities=s.priorities or ["impact"],
            # Director-chosen skills, validated against the registry (fallback default).
            skills=valid_skills(s.skills) or valid_skills(_DEFAULT_SKILLS["stakeholder"]),
        )
        for s in plan.stakeholders
    ]
    topics = plan.research_topics or [
        f"evidence on {request.question.rstrip('?')}",
        "risks and comparable cases",
    ]
    tasks = _research_tasks(request, topics) + _stakeholder_tasks(request, stakeholders)
    return objective, stakeholders, tasks


# --- Revision --------------------------------------------------------------
class _RevisionOut(BaseModel):
    summary: str
    added_actions: list[str] = Field(default_factory=list)
    added_risks: list[str] = Field(default_factory=list)


def _mock_revise(
    recommendation: PolicyRecommendation, critique: CritiqueResult
) -> PolicyRecommendation:
    revised = recommendation.model_copy(deep=True)
    revised.summary = "[Revised after red-team review] " + recommendation.summary
    for fix in critique.required_revisions:
        revised.recommended_actions.append(f"Address red-team concern: {fix}")
    for consequence in critique.unintended_consequences:
        if consequence not in revised.risks:
            revised.risks.append(consequence)
    revised.confidence = max(0.0, recommendation.confidence - 0.1)
    return revised


def revise_recommendation(
    request: PolicyRequest,
    recommendation: PolicyRecommendation,
    critique: CritiqueResult,
) -> PolicyRecommendation:
    """Apply the Red Team's required revisions to the recommendation."""
    if MOCK_DIRECTOR:
        return _mock_revise(recommendation, critique)

    prompt = (
        f"POLICY QUESTION: {request.question}\n\n"
        f"CURRENT RECOMMENDATION SUMMARY: {recommendation.summary}\n\n"
        f"RED-TEAM REQUIRED REVISIONS:\n- " + "\n- ".join(critique.required_revisions)
        + "\n\nRevise the recommendation to address these. Return JSON with keys: "
        "summary (string), added_actions (string[]), added_risks (string[])."
    )
    out, _ = run_structured("policy_director", prompt, _RevisionOut)
    if out is None:
        return _mock_revise(recommendation, critique)

    revised = recommendation.model_copy(deep=True)
    revised.summary = out.summary or recommendation.summary
    revised.recommended_actions.extend(out.added_actions)
    for risk in out.added_risks:
        if risk not in revised.risks:
            revised.risks.append(risk)
    revised.confidence = max(0.0, recommendation.confidence - 0.1)
    return revised
