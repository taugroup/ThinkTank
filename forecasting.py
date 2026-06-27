"""Forecasting dispatcher — domain-neutral.

OWNER: Person 3 (impl/forecast/evals). Forecasting is OPTIONAL and domain-gated:

  - If a deterministic domain module matches the policy question (see `forecasters/`),
    run its numeric 4-scenario forecast.
  - Otherwise return a QUALITATIVE directional outlook derived from the
    recommendation — directional language only, NEVER fabricated numbers.

This keeps the "no LLM arithmetic / no unsupported numbers" rule while supporting
any policy domain. The numeric math lives in the per-domain modules.

Public interface:
    run_forecast(recommendation, request) -> ForecastResult
"""

from __future__ import annotations

from forecasters import detect_domain
from models import ForecastResult, PolicyRecommendation, PolicyRequest


def qualitative_outlook(recommendation: PolicyRecommendation) -> ForecastResult:
    """Directional, number-free outlook when no domain forecaster applies."""
    directions: list[str] = []
    for benefit in recommendation.benefits:
        directions.append(f"Likely positive movement: {benefit.lower()}")
    for risk in recommendation.risks:
        directions.append(f"Potential adverse pressure: {risk.lower()}")
    if not directions:
        directions.append(
            "Direction of effects is uncertain; insufficient basis for a signed outlook."
        )
    return ForecastResult(
        mode="qualitative",
        domain="generic",
        qualitative=directions,
        assumptions=[
            "No deterministic model exists for this policy domain, so only the "
            "direction of effects is offered.",
            "Directions are inferred from the recommendation's stated benefits and risks.",
        ],
        limitations=[
            "No quantitative magnitudes are provided and none should be inferred.",
            "Directional claims are qualitative judgments, not measured forecasts.",
            "Adding a deterministic domain module would enable numeric scenarios.",
        ],
    )


def run_forecast(
    recommendation: PolicyRecommendation,
    request: PolicyRequest,
) -> ForecastResult:
    """Numeric forecast if a domain module matches; else a qualitative outlook."""
    module = detect_domain(request)
    if module is None:
        return qualitative_outlook(recommendation)
    parameters = module.derive_parameters(recommendation, request)
    return module.forecast(recommendation, parameters)
