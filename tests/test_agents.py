"""Tests for the agent roster: skill registry, Director skill assignment, and the
Research agent. Mock mode (no model needed)."""

from agents import plan_policy, run_research
from models import PolicyRequest
from skills_registry import list_skills, valid_skills

REQ = PolicyRequest(
    question="Should the city fund universal pre-K?",
    geography="Ohio",
    objective="improve early-education outcomes",
)


def test_registry_reads_skill_md_files():
    skills = list_skills()
    assert "policy-research" in skills
    assert "policy-analysis" in skills
    assert skills["policy-research"]  # has a description


def test_valid_skills_filters_unknown():
    assert valid_skills(["policy-research", "made-up-skill"]) == ["policy-research"]


def test_director_assigns_agent_types_and_skills():
    _obj, _stk, tasks = plan_policy(REQ)
    types = {t.agent_type for t in tasks}
    assert "research" in types and "stakeholder" in types
    # every task gets a non-empty, valid skill set chosen by the Director
    for t in tasks:
        assert t.skills
        assert valid_skills(t.skills) == t.skills


def test_director_can_assign_different_skills_per_stakeholder():
    _obj, stk, _tasks = plan_policy(REQ)
    skill_sets = {tuple(s.skills) for s in stk}
    # at least one stakeholder has a different skill set than another (economic gets
    # policy-analysis on top of policy-research)
    assert len(skill_sets) >= 2


def test_research_agent_produces_briefs_only_for_research_tasks():
    _obj, _stk, tasks = plan_policy(REQ)
    briefs = run_research(REQ, tasks)
    assert len(briefs) == sum(1 for t in tasks if t.agent_type == "research")
    assert all(b.findings for b in briefs)
