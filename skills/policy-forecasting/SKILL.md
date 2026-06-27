---
name: policy-forecasting
description: Translate a policy into deterministic scenario parameters and explain outputs with honest uncertainty.
owner: Person 3 (impl/forecast/evals)
---

# Policy Forecasting Skill

Use this skill when converting a recommendation into forecast scenarios.

## Hard rule: no LLM arithmetic
- The model only sets `ForecastParameters`. ALL numbers are computed by
  `forecasting.run_forecast` in deterministic Python. Never compute forecast
  figures yourself.

## Forecasting is optional & domain-gated
- Numeric forecasting runs ONLY when a deterministic domain module matches the
  question (see `forecasters/`). For any other domain, produce a **qualitative
  directional outlook** (e.g. "likely upward pressure on rents") with **no numbers**.
- Never invent numeric magnitudes for a domain that has no calibrated model.

## Scenarios (numeric mode — produce all four)
- **Baseline** — status quo, no policy in effect.
- **Conservative** — reduced behavioral response and enforcement.
- **Expected** — modeled central estimate.
- **Optimistic** — stronger behavioral response.

## Explicit assumptions
- Every assumption that drives the numbers must appear in `ForecastResult.assumptions`.

## Sensitivity analysis
- Report how a key output moves with a key input (e.g. net revenue per +10% fee).

## Limitations & confidence language
- List modeling limitations in `ForecastResult.limitations`.
- Use ranges and scenario language. Never claim an exact prediction of the future.

## Rule against false precision
- Present figures as scenario estimates, not guarantees. State what is NOT modeled
  (induced demand, boundary effects, economic feedback).
