"""Orchestrator: node functions + the single public entry point.

OWNER: Person 1 (orchestration). The LangGraph state graph (graph.py) wires these
node functions together. Each node reads the shared state, calls a public seam
function (mockable), logs a ModelEvent, appends a human-readable event, and returns
its state update. Full data lives in state/SQLite/vector store — NOT in prompts.

Public function (frozen):
    run_policy_analysis(request: PolicyRequest) -> PolicyRunResult
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from config import (
    LOCAL_MODEL,
    MAX_REVISION_LOOPS,
    MOCK_MODE,
    REVISE_ON_SEVERITY,
)
from logger import log_model_event
from models import ModelEvent, PolicyRequest, PolicyRunResult

from agents import (
    create_policy_recommendation,
    plan_policy,
    red_team_review,
    revise_recommendation,
    run_research,
    run_stakeholder_research,
    synthesize_research,
)
from forecasters import detect_domain
from forecasting import run_forecast

PolicyState = dict[str, Any]


def _emit(state: PolicyState, agent: str, *, escalated: bool = False, started: float | None = None) -> None:
    """Record a model event for the dashboard (mock latency in MOCK_MODE)."""
    latency = int((time.time() - started) * 1000) if started else (1 if MOCK_MODE else 0)
    event = ModelEvent(
        agent=agent,
        model="mock" if MOCK_MODE else LOCAL_MODEL,
        latency_ms=latency,
        schema_valid=True,
        escalated=escalated,
        error=None,
    )
    state.setdefault("model_events", []).append(event)
    log_model_event(event)


def _log(state: PolicyState, message: str) -> None:
    state.setdefault("events", []).append(message)


# --- Graph nodes -----------------------------------------------------------


def node_plan_policy(state: PolicyState) -> PolicyState:
    request: PolicyRequest = state["request"]
    objective, stakeholders, tasks = plan_policy(request)
    state["objective"] = objective
    state["stakeholders"] = stakeholders
    state["tasks"] = tasks
    _emit(state, "policy_director")
    _log(state, f"Policy Director created research plan with {len(tasks)} tasks.")
    return state


def node_research(state: PolicyState) -> PolicyState:
    request: PolicyRequest = state["request"]
    briefs = run_research(request, state["tasks"])
    state["research_briefs"] = briefs
    n_research = sum(1 for t in state["tasks"] if t.agent_type == "research")
    _emit(state, "research")
    _log(state, f"{len(briefs)} research agent(s) gathered evidence ({n_research} tasks).")
    return state


def node_stakeholder_research(state: PolicyState) -> PolicyState:
    request: PolicyRequest = state["request"]
    research = run_stakeholder_research(request, state["tasks"], state["stakeholders"])
    state["research"] = research
    # Collect cited evidence so the UI can show the corpus actually used.
    from retrieval import retrieve_policy_evidence

    state["evidence"] = retrieve_policy_evidence(
        [request.question], geography=request.geography
    )
    _emit(state, "stakeholder_research")
    _log(state, f"{len(research)} stakeholder agents completed research.")
    return state


def node_synthesize(state: PolicyState) -> PolicyState:
    state["synthesis"] = synthesize_research(state["request"], state["research"])
    _emit(state, "synthesis")
    _log(state, "Research synthesized across stakeholders.")
    return state


def node_recommend(state: PolicyState) -> PolicyState:
    state["recommendation"] = create_policy_recommendation(
        state["request"], state["research"]
    )
    _emit(state, "implementation_agent")
    _log(state, "Implementation plan and recommendation drafted.")
    return state


def node_red_team(state: PolicyState) -> PolicyState:
    critique = red_team_review(state["request"], state["recommendation"])
    state.setdefault("critiques", []).append(critique)
    _emit(state, "red_team")
    _log(state, f"Red Team review complete (severity={critique.severity}).")
    return state


def node_revise(state: PolicyState) -> PolicyState:
    critique = state["critiques"][-1]
    state["recommendation"] = revise_recommendation(
        state["request"], state["recommendation"], critique
    )
    state["revisions"] = state.get("revisions", 0) + 1
    _emit(state, "policy_director", escalated=False)
    _log(state, f"Recommendation revised (revision #{state['revisions']}).")
    return state


def node_forecast(state: PolicyState) -> PolicyState:
    request: PolicyRequest = state["request"]
    recommendation = state["recommendation"]
    module = detect_domain(request)
    if module is not None:
        # Record the parameters used so the UI scenario simulator can re-run them.
        state["forecast_parameters"] = module.derive_parameters(recommendation, request)
        _log(state, f"Numeric forecast generated ({module.DOMAIN} domain).")
    else:
        state["forecast_parameters"] = None
        _log(state, "No domain model matched; qualitative outlook generated (no numbers).")
    state["forecast"] = run_forecast(recommendation, request)
    _emit(state, "forecasting")
    return state


def node_finalize(state: PolicyState) -> PolicyState:
    _log(state, "Final recommendation assembled.")
    return state


# --- Conditional routing ---------------------------------------------------


def should_revise(state: PolicyState) -> str:
    """Return 'revise' or 'forecast' after red-team review."""
    critique = state["critiques"][-1]
    revisions = state.get("revisions", 0)
    severities = {"low": 0, "medium": 1, "high": 2}
    needs = severities.get(critique.severity, 0) >= severities.get(REVISE_ON_SEVERITY, 2)
    if needs and revisions < MAX_REVISION_LOOPS:
        return "revise"
    return "forecast"


# --- Public entry point ----------------------------------------------------


def _build_result(state: PolicyState) -> PolicyRunResult:
    return PolicyRunResult(
        run_id=state["run_id"],
        request=state["request"],
        objective=state.get("objective"),
        stakeholders=state.get("stakeholders", []),
        tasks=state.get("tasks", []),
        research_briefs=state.get("research_briefs", []),
        research=state.get("research", []),
        synthesis=state.get("synthesis"),
        recommendation=state.get("recommendation"),
        critiques=state.get("critiques", []),
        revisions=state.get("revisions", 0),
        forecast=state.get("forecast"),
        forecast_parameters=state.get("forecast_parameters"),
        evidence=state.get("evidence", []),
        model_events=state.get("model_events", []),
        events=state.get("events", []),
    )


def run_policy_analysis(request: PolicyRequest) -> PolicyRunResult:
    """Run the full policy workflow and return a complete PolicyRunResult.

    Uses the LangGraph state graph when available, falling back to an equivalent
    sequential executor so the workflow runs even without langgraph installed.
    """
    state: PolicyState = {
        "run_id": uuid.uuid4().hex[:12],
        "request": request,
        "model_events": [],
        "events": [],
        "critiques": [],
        "revisions": 0,
    }

    from graph import run_graph

    final_state = run_graph(state)
    result = _build_result(final_state)

    # Best-effort persistence (P4 owns the schema; failure must not break a run).
    try:
        from storage import save_run

        save_run(result)
    except Exception:  # pragma: no cover - persistence is non-critical here
        pass

    return result
