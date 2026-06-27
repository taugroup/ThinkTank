"""Housing-supply domain forecaster.

OWNER: Person 3 (impl/forecast/evals). A second pluggable, LLM-free domain module,
demonstrating the registry pattern. Estimates new housing units and rent effects
from upzoning / supply-expansion policies. Illustrative constants, not predictions.

Exposes the domain-module interface: matches / derive_parameters / forecast.
"""

from __future__ import annotations

from models import (
    ForecastParameters,
    ForecastResult,
    PolicyRecommendation,
    PolicyRequest,
    ScenarioResult,
)

DOMAIN = "housing"
KEYWORDS = [
    "housing",
    "zoning",
    "upzoning",
    "rent",
    "affordable",
    "affordability",
    "dwelling",
    "adu",
    "density",
    "multi-family",
    "multifamily",
]

# Illustrative elasticity: % rent change per 1% growth in housing stock.
RENT_ELASTICITY = -0.25

_SCENARIO_UPTAKE = {
    "baseline": 0.0,
    "conservative": 0.5,
    "expected": 1.0,
    "optimistic": 1.5,
}


def matches(request: PolicyRequest) -> bool:
    text = f"{request.question} {request.objective}".lower()
    return any(k in text for k in KEYWORDS)


def derive_parameters(
    recommendation: PolicyRecommendation, request: PolicyRequest
) -> ForecastParameters:
    """Map a housing policy to generic levers via the `extra` dict (demo defaults)."""
    return ForecastParameters(
        extra={
            "existing_units": 200_000.0,
            "eligible_parcels": 50_000.0,
            "units_per_parcel": 2.0,
            "annual_uptake_rate": 0.04,  # share of eligible parcels developed / yr
            "years": 5.0,
            "avg_rent": 2000.0,
        }
    )


def _run_scenario(name: str, p: ForecastParameters) -> ScenarioResult:
    e = p.extra
    uptake = e.get("annual_uptake_rate", 0.0) * _SCENARIO_UPTAKE[name]
    new_units = e.get("eligible_parcels", 0.0) * uptake * e.get("units_per_parcel", 1.0) * e.get("years", 1.0)
    existing = e.get("existing_units", 1.0) or 1.0
    stock_growth_pct = (new_units / existing) * 100.0
    rent_change_pct = stock_growth_pct * RENT_ELASTICITY
    avg_rent = e.get("avg_rent", 0.0)
    rent_change_dollars = avg_rent * (rent_change_pct / 100.0)
    return ScenarioResult(
        name=name,
        inputs={
            "new_units": round(new_units, 0),
            "stock_growth_pct": round(stock_growth_pct, 2),
            "rent_change_pct": round(rent_change_pct, 2),
            "rent_change_dollars_per_month": round(rent_change_dollars, 2),
            "effective_uptake_rate": round(uptake, 3),
        },
    )


def forecast(
    recommendation: PolicyRecommendation, parameters: ForecastParameters
) -> ForecastResult:
    return ForecastResult(
        mode="numeric",
        domain=DOMAIN,
        baseline=_run_scenario("baseline", parameters),
        conservative=_run_scenario("conservative", parameters),
        expected=_run_scenario("expected", parameters),
        optimistic=_run_scenario("optimistic", parameters),
        assumptions=[
            f"Rent elasticity = {RENT_ELASTICITY}% rent change per 1% stock growth.",
            "Baseline assumes the status quo (no upzoning uptake).",
            "Conservative/optimistic assume 0.5x / 1.5x of the modeled uptake rate.",
            "All eligible parcels develop at the same average density.",
        ],
        limitations=[
            "Figures are scenario estimates, not predictions of actual outcomes.",
            "Elasticity and uptake constants are illustrative, not locally calibrated.",
            "Displacement, construction lag, and demand shifts are not modeled.",
        ],
    )
