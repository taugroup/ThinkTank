"""End-to-end mock workflow tests for the orchestrator + graph."""

import config
from models import PolicyRequest, PolicyRunResult
from orchestrator import run_policy_analysis, should_revise

REQ = PolicyRequest(
    question="Should Boston implement congestion pricing downtown?",
    geography="Boston, MA",
    objective="Reduce downtown congestion while protecting equity",
    constraints=["Protect low-income commuters"],
    timeline="3 years",
)


def test_end_to_end_produces_complete_result():
    res = run_policy_analysis(REQ)
    assert isinstance(res, PolicyRunResult)
    assert res.objective is not None
    assert len(res.stakeholders) >= 3  # multiple perspectives
    # stakeholder results match the stakeholder-typed tasks; research briefs come
    # from the research-typed tasks the Director also created.
    stakeholder_tasks = [t for t in res.tasks if t.agent_type == "stakeholder"]
    assert len(res.research) == len(stakeholder_tasks)
    assert len(res.research_briefs) >= 1
    assert all(t.skills for t in res.tasks)  # Director assigned skills to every task
    assert res.recommendation is not None
    assert res.synthesis is not None
    assert res.forecast is not None
    assert res.critiques  # red team ran
    assert res.model_events  # model calls were logged
    assert res.events  # human-readable activity log populated


def test_revision_loop_bounded():
    res = run_policy_analysis(REQ)
    assert res.revisions <= config.MAX_REVISION_LOOPS


def test_should_revise_respects_limit():
    from models import CritiqueResult

    state = {"critiques": [CritiqueResult(severity="high")], "revisions": config.MAX_REVISION_LOOPS}
    assert should_revise(state) == "forecast"  # at the cap -> stop revising
    state2 = {"critiques": [CritiqueResult(severity="high")], "revisions": 0}
    assert should_revise(state2) == "revise"


def test_severe_critique_changes_recommendation():
    """When the Red Team flags a high-severity issue, the recommendation revises."""
    from agents.policy_director import revise_recommendation
    from agents.red_team_agent import red_team_review
    from models import PolicyRecommendation

    # A recommendation with NO equity_effects -> red team returns severity "high".
    weak = PolicyRecommendation(
        summary="Charge a flat fee.",
        recommended_actions=["Charge $9/day"],
        evidence_ids=["TFL-2008-CC", "TRB-2021-CP"],
        equity_effects=[],
    )
    critique = red_team_review(REQ, weak)
    assert critique.severity == "high"
    revised = revise_recommendation(REQ, weak, critique)
    assert revised.summary != weak.summary
    assert len(revised.recommended_actions) > len(weak.recommended_actions)


def test_fallback_executor_matches_topology():
    from graph import _run_fallback

    state = {
        "run_id": "t",
        "request": REQ,
        "model_events": [],
        "events": [],
        "critiques": [],
        "revisions": 0,
    }
    out = _run_fallback(state)
    assert out["recommendation"] is not None and out["forecast"] is not None
