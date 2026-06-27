# Agent Development Life Cycle (ADLC)

How we build, evaluate, and harden the Policy Think Tank agents.

## 1. Design
- Freeze shared schemas (`models.py`) and public seam contracts before splitting work.
- Each agent gets a compact `BriefingPacket` and a required output schema.

## 2. Build (mock-first)
- Every public seam ships a `MOCK_MODE` implementation returning schema-valid
  objects from `examples/*.json`. This unblocks parallel development.
- Real implementations swap in behind the unchanged signature.

## 3. Structured output + validation
- Agents must return JSON matching their Pydantic schema.
- On validation failure: retry locally up to `MAX_SCHEMA_RETRIES`, then optionally
  escalate to the frontier model (if enabled), then fail loudly with a logged error.

## 4. Evaluate
- `evals/run_evals.py` runs cases in `evals/cases.json` and checks assertions:
  findings have evidence IDs; recommendation cites ≥2 sources; ≥3 stakeholder
  perspectives; equity discussed; ≥3 implementation steps; red team finds ≥2
  weaknesses; recommendation changes on severe critique; conservative/expected/
  optimistic forecasts present with visible assumptions; structured outputs validate.
- Records: assertion pass rate, latency, schema failures, local-model call count,
  escalation count, errors → `evals/baseline_results.json` / `skilled_results.json`.
- Comparisons to run: **without skills vs with skills**, **local-only vs local +
  fallback**.

## 5. Observe
- Every model call logs a `ModelEvent`. The UI surfaces local vs escalated counts.

## 6. Iterate
- Skills (`skills/*/SKILL.md`) encode the rules; tune them and re-run evals to show
  measurable improvement (skilled vs baseline).

## Guardrails
- No LLM arithmetic in forecasts. No unsupported numeric claims (label as assumption
  or cite). No secrets in code. Local Ollama support preserved. Works for any policy
  domain; numeric forecasts only where a deterministic domain module exists,
  otherwise a qualitative outlook (no fabricated numbers).
