"""Research agent.

OWNER: Person 2 (research/RAG) for the retrieval internals; the dispatch + skill
assignment is orchestration. Gathers OBJECTIVE, cited evidence on a sub-topic (no
stakeholder perspective) — the shared evidence base the stakeholders and the data
analyst build on. Loads exactly the skills the Director assigned to the task.

Real path (POLICY_MOCK_RESEARCH=0) follows the Director template; mock fallback.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_builder import run_structured
from config import DEFAULT_TOP_K, MOCK_RESEARCH
from context_builder import build_packet
from models import Finding, PolicyRequest, PolicyTask, ResearchBrief
from retrieval import retrieve_policy_evidence


class _ResearchOut(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""


def _mock_brief(request: PolicyRequest, task: PolicyTask) -> ResearchBrief:
    evidence = retrieve_policy_evidence(
        task.queries, geography=request.geography, top_k=DEFAULT_TOP_K
    )
    ids = [e.source_id for e in evidence]
    finding = Finding(
        claim=f"Evidence on '{task.description}' shows mixed but informative results.",
        evidence_ids=ids[:2],
        confidence=0.6,
        assumptions=[] if ids else ["No matching evidence; flagged as a data gap."],
        limitations=["Placeholder evidence in mock mode."],
    )
    return ResearchBrief(
        topic=task.description,
        findings=[finding],
        evidence_ids=ids,
        summary=f"{len(ids)} sources gathered on this topic.",
        skills_used=task.skills,
    )


def _real_brief(request: PolicyRequest, task: PolicyTask) -> ResearchBrief:
    evidence = retrieve_policy_evidence(
        task.queries, geography=request.geography, top_k=DEFAULT_TOP_K
    )
    packet = build_packet(
        request,
        task=task,
        perspective="You are an objective research analyst (no advocacy).",
        evidence=evidence,
        skill_keys=task.skills,  # orchestrator-assigned
        output_schema_name="_ResearchOut",
    )
    prompt = packet.to_prompt() + (
        "\n\nReturn JSON with keys: findings (array of {claim, evidence_ids[], "
        "confidence (0-1), assumptions[], limitations[]}), summary (string). Cite "
        "evidence_ids for every factual/numeric claim; otherwise list as an assumption."
    )
    out, _ = run_structured(f"research:{task.task_id}", prompt, _ResearchOut)
    if out is None or not out.findings:
        return _mock_brief(request, task)
    return ResearchBrief(
        topic=task.description,
        findings=out.findings,
        evidence_ids=sorted({eid for f in out.findings for eid in f.evidence_ids}),
        summary=out.summary,
        skills_used=task.skills,
    )


def run_research(request: PolicyRequest, tasks: list[PolicyTask]) -> list[ResearchBrief]:
    """Run every research-typed task and return objective, cited briefs."""
    build = _mock_brief if MOCK_RESEARCH else _real_brief
    return [build(request, t) for t in tasks if t.agent_type == "research"]
