"""Centralized briefing-packet builder.

OWNER: Person 1 (orchestration). This is the antidote to the old "pass the whole
transcript to every agent" pattern. Each agent receives only a compact
`BriefingPacket`: the question, objective, its task, the relevant prior findings,
the top retrieved chunks, the skill instructions it needs, and the schema it must
return. Full state lives in LangGraph / SQLite / the vector store, not in prompts.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field

from config import SKILLS_DIR
from models import (
    EvidenceItem,
    Finding,
    PolicyObjective,
    PolicyRequest,
    PolicyTask,
)


class BriefingPacket(BaseModel):
    """The ONLY context an agent should receive. Keep it compact."""

    question: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    task: str = ""
    perspective: str = ""
    prior_findings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    skill_instructions: str = ""
    output_schema_name: str = ""

    def to_prompt(self) -> str:
        """Render the packet as a compact prompt string for a local model."""
        lines = [
            f"POLICY QUESTION: {self.question}",
            f"OBJECTIVE: {self.objective}",
        ]
        if self.constraints:
            lines.append("CONSTRAINTS:\n- " + "\n- ".join(self.constraints))
        if self.perspective:
            lines.append(f"YOUR PERSPECTIVE: {self.perspective}")
        if self.task:
            lines.append(f"YOUR TASK: {self.task}")
        if self.prior_findings:
            lines.append("RELEVANT PRIOR FINDINGS:\n- " + "\n- ".join(self.prior_findings))
        if self.evidence:
            ev = "\n".join(
                f"[{e.source_id}] ({e.source_type}, cred={e.credibility_score:.2f}) {e.text}"
                for e in self.evidence
            )
            lines.append("TOP EVIDENCE (cite by source_id):\n" + ev)
        if self.skill_instructions:
            lines.append("SKILL GUIDANCE:\n" + self.skill_instructions)
        if self.output_schema_name:
            lines.append(f"Return ONLY valid JSON matching the {self.output_schema_name} schema.")
        return "\n\n".join(lines)


def load_skill(skill_key: str) -> str:
    """Load a skill's SKILL.md body so agents include only relevant guidance."""
    path = os.path.join(SKILLS_DIR, skill_key, "SKILL.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def _summarize_findings(findings: list[Finding], limit: int = 5) -> list[str]:
    """Compact prior findings to one line each — never dump full objects."""
    return [f.claim for f in findings[:limit]]


def build_packet(
    request: PolicyRequest,
    *,
    objective: Optional[PolicyObjective] = None,
    task: Optional[PolicyTask] = None,
    perspective: str = "",
    prior_findings: Optional[list[Finding]] = None,
    evidence: Optional[list[EvidenceItem]] = None,
    skill_keys: Optional[list[str]] = None,
    output_schema_name: str = "",
) -> BriefingPacket:
    """Assemble a compact briefing packet for a single agent call."""
    objective_text = objective.statement if objective else request.objective
    constraints = list(objective.constraints) if objective else list(request.constraints)
    skills_text = "\n\n".join(load_skill(k) for k in (skill_keys or []) if load_skill(k))

    return BriefingPacket(
        question=request.question,
        objective=objective_text,
        constraints=constraints,
        task=task.description if task else "",
        perspective=perspective,
        prior_findings=_summarize_findings(prior_findings or []),
        evidence=evidence or [],
        skill_instructions=skills_text,
        output_schema_name=output_schema_name,
    )
