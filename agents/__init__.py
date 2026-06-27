"""Agent package for the Policy Think Tank.

Roster: a Policy Director that plans and ASSIGNS each worker its agent type + skills,
three worker agent types, and a Red-Team for the revision loop.

  - policy_director   (P1): plan_policy, revise_recommendation  (also assigns skills)
  - research_agent    (P2): run_research            (objective, cited evidence)
  - stakeholder_research (P2): run_stakeholder_research  (per-perspective analysis)
  - data_analyst      (P3): synthesize_research, create_policy_recommendation
  - red_team_agent    (P3): red_team_review         (domain-neutral)

All ship MOCK implementations with deterministic fallbacks (config.MOCK_*), so the
workstreams can develop in parallel before any real model wiring exists.
"""

from agents.data_analyst import create_policy_recommendation, synthesize_research
from agents.policy_director import plan_policy, revise_recommendation
from agents.red_team_agent import red_team_review
from agents.research_agent import run_research
from agents.stakeholder_research import run_stakeholder_research

__all__ = [
    "plan_policy",
    "revise_recommendation",
    "run_research",
    "run_stakeholder_research",
    "synthesize_research",
    "create_policy_recommendation",
    "red_team_review",
]
