"""Policy Think Tank — Streamlit UI.

OWNER: Person 4 (UI/persistence). Refactored from the original scientific-meeting
interface into a local-first policy think tank. Five sections: Intake, Agent
Activity, Stakeholder Views, Final Recommendation, and a Scenario Simulator that
recalculates forecasts without rerunning the agents.

Until Person 1's graph is fully wired the UI can render examples/sample_result.json
(mock mode), and `execute_policy_analysis` is the single adapter the UI calls.

Adapted from TAU Group's ThinkTank (MIT). See ARCHITECTURE.md for attribution.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from forecasters import detect_domain
from models import PolicyRequest, PolicyRunResult
from storage import list_runs, load_run

EXAMPLE_RESULT = os.path.join("examples", "sample_result.json")


# --- Single adapter the UI calls (mock-capable) ----------------------------
def execute_policy_analysis(request: PolicyRequest) -> PolicyRunResult:
    """Run a full analysis. Delegates to the orchestrator; falls back to the
    bundled example result if the orchestrator is unavailable."""
    try:
        from orchestrator import run_policy_analysis

        return run_policy_analysis(request)
    except Exception as exc:  # pragma: no cover - UI resilience
        st.warning(f"Falling back to bundled example result: {exc}")
        with open(EXAMPLE_RESULT, encoding="utf-8") as f:
            return PolicyRunResult.model_validate(json.load(f))


def _load_example() -> PolicyRunResult:
    with open(EXAMPLE_RESULT, encoding="utf-8") as f:
        return PolicyRunResult.model_validate(json.load(f))


# --- Rendering -------------------------------------------------------------
def render_activity(res: PolicyRunResult):
    st.subheader("🛠️ Agent Activity")
    for ev in res.events:
        st.markdown(f"- {ev}")
    local = sum(1 for m in res.model_events if not m.escalated)
    esc = sum(1 for m in res.model_events if m.escalated)
    st.caption(f"Model calls — local: {local} · escalated: {esc} · revisions: {res.revisions}")


def render_research(res: PolicyRunResult):
    if not res.research_briefs:
        return
    st.subheader("🔎 Research")
    for b in res.research_briefs:
        with st.expander(f"{b.topic}  ·  skills: {', '.join(b.skills_used) or '—'}"):
            st.caption(b.summary)
            for f in b.findings:
                st.markdown(f"- {f.claim}  _[{', '.join(f.evidence_ids) or 'no citation'}]_")


def render_stakeholders(res: PolicyRunResult):
    st.subheader("👥 Stakeholder Views")
    # Map stakeholder name -> assigned skills (Director-decided) for display.
    skills_by_name = {s.name: s.skills for s in res.stakeholders}
    for r in res.research:
        skills = skills_by_name.get(r.stakeholder, [])
        label = f"{r.stakeholder} — {r.likely_position}"
        with st.expander(label):
            if skills:
                st.caption("Skills assigned by orchestrator: " + ", ".join(skills))
            for f in r.findings:
                st.markdown(f"**Finding:** {f.claim}")
                st.caption(
                    f"Evidence: {', '.join(f.evidence_ids) or 'none'} · "
                    f"confidence {f.confidence:.0%}"
                )
                if f.assumptions:
                    st.caption("Assumptions: " + "; ".join(f.assumptions))
            if r.concerns:
                st.markdown("**Concerns:** " + "; ".join(r.concerns))
            if r.data_gaps:
                st.markdown("**Data gaps:** " + "; ".join(r.data_gaps))


def render_recommendation(res: PolicyRunResult):
    rec = res.recommendation
    if not rec:
        return
    st.subheader("📋 Final Recommendation")
    st.markdown(f"**Executive summary:** {rec.summary}")
    st.markdown(f"_Confidence: {rec.confidence:.0%}_")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommended actions**")
        for a in rec.recommended_actions:
            st.markdown(f"- {a}")
        st.markdown("**Benefits**")
        for b in rec.benefits:
            st.markdown(f"- {b}")
    with col2:
        st.markdown("**Risks**")
        for r in rec.risks:
            st.markdown(f"- {r}")
        st.markdown("**Equity effects**")
        for e in rec.equity_effects:
            st.markdown(f"- {e}")
    if rec.implementation_plan:
        st.markdown("**Implementation plan**")
        for step in rec.implementation_plan.steps:
            st.markdown(f"- *{step.phase}* ({step.timeline or 'TBD'}): " + "; ".join(step.actions))
    if rec.evidence_ids:
        st.caption("Supporting evidence: " + ", ".join(sorted(set(rec.evidence_ids))))
    if res.critiques:
        with st.expander("🔴 Red Team review"):
            c = res.critiques[-1]
            st.markdown(f"Severity: **{c.severity}**")
            for i in c.issues:
                st.markdown(f"- {i}")


def _scenario_table(fc) -> dict:
    return {
        s.name: {
            "trip_reduction": s.trip_reduction,
            "net_revenue": s.net_revenue,
            "transit_increase": s.transit_demand_increase,
            "emissions_change": s.emissions_change,
            "equity_risk": s.equity_risk_index,
        }
        for s in (fc.baseline, fc.conservative, fc.expected, fc.optimistic)
        if s is not None
    }


def render_forecast(res: PolicyRunResult):
    fc = res.forecast
    if not fc:
        return
    st.subheader("🔮 Forecast")
    if fc.mode == "qualitative":
        st.info(
            "No deterministic model matched this policy domain, so only a "
            "directional outlook is shown — no numeric figures."
        )
        for line in fc.qualitative:
            st.markdown(f"- {line}")
    else:
        st.caption(f"Numeric forecast — {fc.domain} domain.")
        st.dataframe(_scenario_table(fc))
    with st.expander("Assumptions & limitations"):
        for a in fc.assumptions:
            st.markdown(f"- {a}")
        st.markdown("**Limitations**")
        for limit in fc.limitations:
            st.markdown(f"- {limit}")


def render_simulator(res: PolicyRunResult):
    """Only shown for numeric, domain-matched forecasts (e.g. transportation)."""
    module = detect_domain(res.request)
    if module is None or not res.forecast or res.forecast.mode != "numeric":
        return
    if res.forecast_parameters is None:
        return

    st.subheader("🎛️ Scenario Simulator")
    st.caption("Adjust levers and recalculate forecasts — agents are NOT rerun.")
    base = res.forecast_parameters
    col1, col2 = st.columns(2)
    with col1:
        fee = st.slider("Daily fee ($)", 0.0, 30.0, float(base.daily_fee), 0.5)
        exemption = st.slider("Exemption rate", 0.0, 1.0, float(base.exemption_rate), 0.05)
    with col2:
        enforcement = st.slider(
            "Enforcement effectiveness", 0.0, 1.0, float(base.enforcement_effectiveness), 0.05
        )
        reinvest = st.slider(
            "Transit reinvestment", 0.0, 1.0, float(base.transit_reinvestment_pct), 0.05
        )
    params = base.model_copy(
        update={
            "daily_fee": fee,
            "exemption_rate": exemption,
            "enforcement_effectiveness": enforcement,
            "transit_reinvestment_pct": reinvest,
        }
    )
    fc = module.forecast(res.recommendation, params)
    st.dataframe(_scenario_table(fc))


def main():
    st.set_page_config(page_title="Policy Think Tank", layout="wide")
    st.markdown("<h1 style='text-align:center;'>🏛️ Policy Think Tank</h1>", unsafe_allow_html=True)
    st.caption("Local-first multi-agent policy analysis · adapted from TAU Group's ThinkTank")

    st.sidebar.header("Policy Intake")
    question = st.sidebar.text_area(
        "Policy question",
        "Should our city introduce a guaranteed basic income pilot?",
        help="Any policy question, in any domain.",
    )
    geography = st.sidebar.text_input("Geography", "")
    objective = st.sidebar.text_input("Objective", "")
    constraints = st.sidebar.text_area("Constraints (one per line)", "")
    timeline = st.sidebar.text_input("Timeline", "")
    st.sidebar.file_uploader("Supporting files (optional)", accept_multiple_files=True)

    run_clicked = st.sidebar.button("Run Policy Analysis", type="primary")
    show_example = st.sidebar.button("Load example result")

    # Past runs (persisted in SQLite by the orchestrator).
    try:
        history = list_runs()
    except Exception:
        history = []
    if history:
        st.sidebar.markdown("---")
        labels = ["—"] + [f"{h['question'][:40]} ({h['run_id'][:6]})" for h in history]
        chosen = st.sidebar.selectbox("Past runs", labels)
        if chosen != "—":
            picked = history[labels.index(chosen) - 1]
            loaded = load_run(picked["run_id"])
            if loaded is not None:
                st.session_state.result = loaded

    if "result" not in st.session_state:
        st.session_state.result = None

    if run_clicked:
        req = PolicyRequest(
            question=question,
            geography=geography,
            objective=objective,
            constraints=[c.strip() for c in constraints.splitlines() if c.strip()],
            timeline=timeline or None,
        )
        with st.spinner("Running policy analysis..."):
            st.session_state.result = execute_policy_analysis(req)
    elif show_example:
        st.session_state.result = _load_example()

    res = st.session_state.result
    if res is None:
        st.info("Enter a policy question and click **Run Policy Analysis**, or load the example.")
        return

    render_activity(res)
    render_research(res)
    render_stakeholders(res)
    render_recommendation(res)
    render_forecast(res)
    render_simulator(res)

    # Export the brief (lazy import so docx/pandoc deps don't block app startup).
    try:
        from utils import export_policy_brief

        st.download_button(
            "⬇️ Download policy brief (DOCX)",
            export_policy_brief(res),
            file_name=f"policy_brief_{res.run_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except Exception as exc:  # pragma: no cover - export is non-critical
        st.caption(f"Brief export unavailable: {exc}")


if __name__ == "__main__":
    main()
