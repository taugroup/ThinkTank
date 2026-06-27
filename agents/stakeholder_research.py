"""Stakeholder research agent system.

OWNER: Person 2 (research/RAG). One shared agent implementation, dynamically
configured into different policy perspectives. Retrieves evidence and returns cited
findings, likely position, concerns, mitigations, and data gaps.

Real path (POLICY_MOCK_RESEARCH=0) follows the Director template: retrieve evidence,
build a skill-loaded BriefingPacket, call `run_structured`, fall back to mock on any
failure. Public function works without Streamlit/LangGraph by contract.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_builder import run_structured
from config import DEFAULT_TOP_K, MOCK_RESEARCH
from context_builder import build_packet
from models import (
    Finding,
    PolicyRequest,
    PolicyTask,
    StakeholderProfile,
    StakeholderResearchResult,
)
from retrieval import retrieve_policy_evidence


# --- LLM output schema (internal) -----------------------------------------
class _ResearchOut(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    likely_position: str = ""
    concerns: list[str] = Field(default_factory=list)
    proposed_mitigations: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    handoff_summary: str = ""


def _mock_result(
    request: PolicyRequest, task: PolicyTask, profile: StakeholderProfile
) -> StakeholderResearchResult:
    evidence = retrieve_policy_evidence(
        task.queries, geography=request.geography, top_k=DEFAULT_TOP_K
    )
    evidence_ids = [e.source_id for e in evidence]
    findings = [
        Finding(
            claim=f"From the {profile.name} view, {request.question.rstrip('?')} "
            f"primarily affects {profile.priorities[0]}.",
            evidence_ids=evidence_ids[:2],
            confidence=0.7,
            assumptions=[f"Local conditions resemble those in source {evidence_ids[0]}."]
            if evidence_ids
            else ["No local evidence available; reasoning by analogy."],
            limitations=["Limited city-specific data in the curated corpus."],
        ),
    ]
    return StakeholderResearchResult(
        stakeholder=profile.name,
        findings=findings,
        likely_position=f"{profile.name} is cautiously supportive if {profile.priorities[0]} "
        "concerns are addressed.",
        concerns=[f"Potential negative impact on {p}" for p in profile.priorities[1:]],
        proposed_mitigations=[f"Targeted measures to protect {profile.priorities[-1]}"],
        data_gaps=[f"Need recent {request.geography} data on {profile.priorities[0]}"],
        handoff_summary=f"{profile.name}: cautiously supportive; key concern = "
        f"{profile.priorities[-1]}; {len(evidence_ids)} sources cited.",
    )


def _real_result(
    request: PolicyRequest, task: PolicyTask, profile: StakeholderProfile
) -> StakeholderResearchResult:
    evidence = retrieve_policy_evidence(
        task.queries, geography=request.geography, top_k=DEFAULT_TOP_K
    )
    packet = build_packet(
        request,
        task=task,
        perspective=profile.perspective,
        evidence=evidence,
        skill_keys=task.skills or profile.skills,  # orchestrator-assigned skills
        output_schema_name="_ResearchOut",
    )
    prompt = (
        packet.to_prompt()
        + "\n\nReturn JSON with keys: findings (array of {claim, evidence_ids[], "
        "confidence (0-1), assumptions[], limitations[]}), likely_position (string), "
        "concerns[], proposed_mitigations[], data_gaps[], handoff_summary (string). "
        "Every numeric or factual claim must cite evidence_ids from the evidence above; "
        "otherwise list it under assumptions."
    )
    out, _ = run_structured(f"research:{profile.key}", prompt, _ResearchOut)
    if out is None or not out.findings:
        return _mock_result(request, task, profile)
    return StakeholderResearchResult(
        stakeholder=profile.name,
        findings=out.findings,
        likely_position=out.likely_position,
        concerns=out.concerns,
        proposed_mitigations=out.proposed_mitigations,
        data_gaps=out.data_gaps,
        handoff_summary=out.handoff_summary
        or f"{profile.name}: {len(out.findings)} findings.",
    )


def run_stakeholder_research(
    request: PolicyRequest,
    tasks: list[PolicyTask],
    stakeholders: list[StakeholderProfile],
) -> list[StakeholderResearchResult]:
    """Run each stakeholder's research task and return cited results."""
    by_key = {s.key: s for s in stakeholders}
    build = _mock_result if MOCK_RESEARCH else _real_result
    results = []
    for task in tasks:
        if task.agent_type != "stakeholder":
            continue  # research tasks are handled by the Research agent
        profile = by_key.get(task.stakeholder_key)
        if profile is None:
            continue
        results.append(build(request, task, profile))
    return results
