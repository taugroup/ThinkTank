"""Data Analyst agent.

OWNER: Person 3 (impl/forecast/evals). Analyzes the research + stakeholder findings,
compares alternatives, and produces the synthesis and the policy recommendation with
a phased implementation plan. This is the "data analyst" role in the roster.

The implementation currently lives in `implementation_agent.py` (kept for history);
this module is the canonical name the orchestrator and roster use.
"""

from agents.implementation_agent import (
    create_policy_recommendation,
    synthesize_research,
)

__all__ = ["synthesize_research", "create_policy_recommendation"]
