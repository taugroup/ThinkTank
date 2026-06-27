---
name: policy-analysis
description: Turn stakeholder research into a comparable, feasible, equitable policy recommendation with a phased plan.
owner: Person 3 (impl/forecast/evals)
---

# Policy Analysis Skill

Use this skill when synthesizing research into a recommendation.

## Alternative comparison
- Define at least two alternatives plus "no action"; list pros/cons for each.

## Cost and benefit analysis
- Summarize expected benefits and costs qualitatively; defer hard numbers to the
  deterministic forecaster (`forecasting.py`).

## Equity analysis (required)
- Identify who bears costs vs. who gains. Name affected vulnerable groups.
- Propose concrete mitigations (exemptions, credits, reinvestment).

## Stakeholder analysis
- Reflect each stakeholder's likely position and key concerns.

## Feasibility analysis
- Assess economic, equity, political, operational, and legal feasibility
  (`ImpactAnalysis`).

## Implementation planning
- Provide a phased `ImplementationPlan` with at least three concrete steps,
  timelines, and owners.

## Success metrics & monitoring
- Define measurable success metrics and monitoring requirements.

## Required recommendation structure
Return a `PolicyRecommendation`: `summary`, `recommended_actions`, `alternatives`,
`benefits`, `risks`, `equity_effects`, `evidence_ids` (>= 2 supporting sources),
`confidence`, `impact`, `implementation_plan`, `alternatives_detail`.
