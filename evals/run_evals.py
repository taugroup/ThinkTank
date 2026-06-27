"""Assertion-based evaluation harness for policy runs.

OWNER: Person 3 (impl/forecast/evals). Runs each case through run_policy_analysis
and checks the hackathon assertions, recording pass rate, latency, schema failures,
local-model call count, escalation count, and errors.

Usage:
    python evals/run_evals.py                 # local-only, current mode
    python evals/run_evals.py --out evals/skilled_results.json

This works in MOCK_MODE today; the same assertions apply to real runs later, which
is how "without skills vs with skills" and "local-only vs fallback" are compared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PolicyRequest, PolicyRunResult  # noqa: E402
from orchestrator import run_policy_analysis  # noqa: E402

CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")


def assertions(res: PolicyRunResult) -> list[tuple[str, bool]]:
    """Each (name, passed). Mirrors the hackathon eval requirements."""
    rec = res.recommendation
    crit = res.critiques[-1] if res.critiques else None
    fc = res.forecast
    material_findings = [f for r in res.research for f in r.findings]
    impl_steps = rec.implementation_plan.steps if rec and rec.implementation_plan else []
    # Forecasting is domain-gated: numeric scenarios when a domain model matches,
    # otherwise a qualitative directional outlook with NO fabricated numbers.
    numeric_ok = bool(fc) and fc.mode == "numeric" and all([fc.conservative, fc.expected, fc.optimistic])
    qualitative_ok = bool(fc) and fc.mode == "qualitative" and bool(fc.qualitative) and not fc.sensitivity
    return [
        ("findings_have_evidence_ids", all(f.evidence_ids for f in material_findings)),
        ("recommendation_has_2plus_sources", bool(rec) and len(set(rec.evidence_ids)) >= 2),
        ("multiple_stakeholder_perspectives", len(res.research) >= 3),
        ("equity_discussed", bool(rec) and len(rec.equity_effects) >= 1),
        ("implementation_has_3plus_steps", len(impl_steps) >= 3),
        ("red_team_two_plus_weaknesses", bool(crit) and (len(crit.issues) + len(crit.unintended_consequences)) >= 2),
        ("forecast_present_in_correct_mode", numeric_ok or qualitative_ok),
        ("forecast_assumptions_visible", bool(fc) and bool(fc.assumptions) and bool(fc.limitations)),
        ("revisions_bounded", res.revisions <= 2),
    ]


def run(out_path: str) -> dict:
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    case_reports = []
    total_assertions = passed_assertions = 0
    schema_failures = local_calls = escalations = errors = 0

    for case in cases:
        t0 = time.time()
        error = None
        checks: list[tuple[str, bool]] = []
        try:
            res = run_policy_analysis(PolicyRequest(**case["request"]))
            PolicyRunResult.model_validate(res.model_dump())  # schema validation
            checks = assertions(res)
            local_calls += sum(1 for m in res.model_events if not m.escalated)
            escalations += sum(1 for m in res.model_events if m.escalated)
            schema_failures += sum(1 for m in res.model_events if not m.schema_valid)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            errors += 1
        latency_ms = int((time.time() - t0) * 1000)
        cp = sum(1 for _, ok in checks if ok)
        total_assertions += len(checks)
        passed_assertions += cp
        case_reports.append(
            {
                "id": case["id"],
                "latency_ms": latency_ms,
                "assertions": {name: ok for name, ok in checks},
                "passed": cp,
                "total": len(checks),
                "error": error,
            }
        )

    report = {
        "summary": {
            "cases": len(cases),
            "assertion_pass_rate": round(passed_assertions / total_assertions, 3)
            if total_assertions
            else 0.0,
            "passed_assertions": passed_assertions,
            "total_assertions": total_assertions,
            "local_model_calls": local_calls,
            "escalations": escalations,
            "schema_failures": schema_failures,
            "errors": errors,
        },
        "cases": case_reports,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "baseline_results.json"))
    args = parser.parse_args()
    rep = run(args.out)
    s = rep["summary"]
    print(json.dumps(s, indent=2))
    print(f"\nWrote {args.out}")
