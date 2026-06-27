"""Transportation domain forecaster (congestion-pricing style).

OWNER: Person 3 (impl/forecast/evals). This is ONE pluggable domain module, not the
product. It is LLM-free: every number comes from explicit Python arithmetic so
forecasts are reproducible and testable. Other domains add sibling modules; policy
questions with no matching module fall back to a qualitative outlook (forecasting.py).

A domain module must expose:
    KEYWORDS: list[str]
    matches(request) -> bool
    derive_parameters(recommendation, request) -> ForecastParameters
    forecast(recommendation, parameters) -> ForecastResult
"""

from __future__ import annotations

from models import (
    ForecastParameters,
    ForecastResult,
    PolicyRecommendation,
    PolicyRequest,
    ScenarioResult,
)

DOMAIN = "transportation"
KEYWORDS = [
    "congestion",
    "traffic",
    "toll",
    "tolling",
    "transit",
    "parking",
    "bus lane",
    "vehicle",
    "road pricing",
    "commute",
    "commuter",
]

# --- Tunable physical/behavioral constants (documented assumptions) --------
EMISSIONS_KG_PER_TRIP = 4.6  # avg CO2 kg avoided per deterred car trip
MODE_SHIFT_TO_TRANSIT = 0.6  # fraction of deterred trips that become transit trips

_SCENARIO_PROFILES: dict[str, dict[str, float]] = {
    "baseline": {"response_mult": 0.0, "enforcement_mult": 0.0},
    "conservative": {"response_mult": 0.7, "enforcement_mult": 0.85},
    "expected": {"response_mult": 1.0, "enforcement_mult": 1.0},
    "optimistic": {"response_mult": 1.3, "enforcement_mult": 1.0},
}


def matches(request: PolicyRequest) -> bool:
    """True if this domain module can forecast the request (keyword match)."""
    text = f"{request.question} {request.objective}".lower()
    return any(k in text for k in KEYWORDS)


def derive_parameters(
    recommendation: PolicyRecommendation,
    request: PolicyRequest,
) -> ForecastParameters:
    """Translate the policy into deterministic forecast inputs (demo defaults)."""
    return ForecastParameters(
        daily_fee=9.0,
        affected_trips_per_day=120_000,
        behavioral_response_rate=0.18,
        exemption_rate=0.15,
        enforcement_effectiveness=0.9,
        operating_days_per_year=250,
        implementation_cost=45_000_000.0,
        transit_reinvestment_pct=0.6,
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _run_scenario(name: str, p: ForecastParameters) -> ScenarioResult:
    profile = _SCENARIO_PROFILES[name]
    response = _clamp(p.behavioral_response_rate * profile["response_mult"])
    enforcement = _clamp(p.enforcement_effectiveness * profile["enforcement_mult"])

    # Baseline = status quo: no charge is in effect, so no revenue and no change.
    fee = 0.0 if name == "baseline" else p.daily_fee

    chargeable_trips = p.affected_trips_per_day * (1.0 - _clamp(p.exemption_rate))
    deterred_per_day = chargeable_trips * response * enforcement
    paying_per_day = max(0.0, chargeable_trips - deterred_per_day)
    days = p.operating_days_per_year

    trip_reduction = deterred_per_day * days
    gross_revenue = paying_per_day * fee * days
    net_revenue = (gross_revenue - p.implementation_cost) if name != "baseline" else 0.0
    transit_demand_increase = deterred_per_day * MODE_SHIFT_TO_TRANSIT * days
    emissions_change = -(trip_reduction * EMISSIONS_KG_PER_TRIP)  # negative = reduction
    commuter_burden = paying_per_day * fee

    reinvest = _clamp(p.transit_reinvestment_pct)
    exemption = _clamp(p.exemption_rate)
    equity_risk_index = round(_clamp((1.0 - exemption) * (1.0 - reinvest)), 3)

    return ScenarioResult(
        name=name,
        trip_reduction=round(trip_reduction, 1),
        gross_revenue=round(gross_revenue, 2),
        net_revenue=round(net_revenue, 2),
        transit_demand_increase=round(transit_demand_increase, 1),
        emissions_change=round(emissions_change, 1),
        commuter_burden=round(commuter_burden, 2),
        equity_risk_index=equity_risk_index,
        inputs={
            "effective_response_rate": round(response, 3),
            "effective_enforcement": round(enforcement, 3),
            "chargeable_trips_per_day": round(chargeable_trips, 1),
        },
    )


def _sensitivity(p: ForecastParameters) -> dict[str, float]:
    base = _run_scenario("expected", p).net_revenue
    bumped = p.model_copy(update={"daily_fee": p.daily_fee * 1.1})
    return {"net_revenue_per_10pct_fee_increase": round(_run_scenario("expected", bumped).net_revenue - base, 2)}


def forecast(
    recommendation: PolicyRecommendation,
    parameters: ForecastParameters,
) -> ForecastResult:
    """Deterministic numeric forecast for the transportation domain."""
    return ForecastResult(
        mode="numeric",
        domain=DOMAIN,
        baseline=_run_scenario("baseline", parameters),
        conservative=_run_scenario("conservative", parameters),
        expected=_run_scenario("expected", parameters),
        optimistic=_run_scenario("optimistic", parameters),
        assumptions=[
            f"Average CO2 avoided per deterred trip = {EMISSIONS_KG_PER_TRIP} kg.",
            f"Mode shift to transit = {int(MODE_SHIFT_TO_TRANSIT * 100)}% of deterred trips.",
            "Baseline scenario assumes the status quo (no pricing in effect).",
            "Conservative assumes 70% of the modeled behavioral response and 85% enforcement.",
            "Optimistic assumes 130% of the modeled behavioral response.",
            "Exempt trips never pay and are never deterred.",
        ],
        limitations=[
            "Figures are scenario estimates, not predictions of actual outcomes.",
            "Behavioral response and mode-shift constants are illustrative, not city-calibrated.",
            "Induced demand, boundary effects, and economic feedback are not modeled.",
        ],
        sensitivity=_sensitivity(parameters),
    )
