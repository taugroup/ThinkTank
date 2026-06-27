"""Forecasting tests — deterministic numeric (transportation domain) + the
domain-gated dispatcher (numeric when matched, qualitative otherwise)."""

from forecasters import detect_domain
from forecasters.transportation import EMISSIONS_KG_PER_TRIP, forecast
from forecasting import run_forecast
from models import ForecastParameters, PolicyRecommendation, PolicyRequest

REC = PolicyRecommendation(
    summary="test",
    benefits=["less congestion"],
    risks=["regressive impact"],
)
PARAMS = ForecastParameters(
    daily_fee=10.0,
    affected_trips_per_day=100_000,
    behavioral_response_rate=0.2,
    exemption_rate=0.1,
    enforcement_effectiveness=1.0,
    operating_days_per_year=250,
    implementation_cost=10_000_000.0,
    transit_reinvestment_pct=0.5,
)

TRANSPORT_REQ = PolicyRequest(
    question="Should the city add congestion pricing downtown?",
    geography="Metro",
    objective="reduce traffic",
)
OTHER_REQ = PolicyRequest(
    question="Should the city fund universal pre-kindergarten?",
    geography="Metro",
    objective="improve early education outcomes",
)


# --- deterministic transportation math ------------------------------------
def test_baseline_is_status_quo():
    fc = forecast(REC, PARAMS)
    assert fc.baseline.trip_reduction == 0.0
    assert fc.baseline.gross_revenue == 0.0


def test_expected_scenario_is_deterministic_and_exact():
    fc = forecast(REC, PARAMS)
    assert fc.expected.trip_reduction == 4_500_000.0
    assert fc.expected.gross_revenue == 180_000_000.0
    assert fc.expected.net_revenue == 180_000_000.0 - 10_000_000.0
    assert fc.expected.emissions_change == round(-4_500_000.0 * EMISSIONS_KG_PER_TRIP, 1)


def test_optimistic_reduces_more_than_conservative():
    fc = forecast(REC, PARAMS)
    assert fc.optimistic.trip_reduction > fc.conservative.trip_reduction


def test_repeatable():
    assert forecast(REC, PARAMS) == forecast(REC, PARAMS)


# --- domain-gated dispatcher ----------------------------------------------
def test_detect_domain():
    assert detect_domain(TRANSPORT_REQ) is not None
    assert detect_domain(OTHER_REQ) is None


def test_dispatcher_numeric_for_matched_domain():
    fc = run_forecast(REC, TRANSPORT_REQ)
    assert fc.mode == "numeric"
    assert fc.domain == "transportation"
    assert fc.expected is not None
    assert fc.assumptions and fc.limitations


def test_dispatcher_qualitative_for_unmatched_domain():
    fc = run_forecast(REC, OTHER_REQ)
    assert fc.mode == "qualitative"
    assert fc.baseline is None and fc.expected is None
    assert fc.qualitative  # directional statements present
    # No fabricated numbers: qualitative outlook carries no scenario figures.
    assert not fc.sensitivity
    assert fc.assumptions and fc.limitations


# --- second domain: housing ------------------------------------------------
HOUSING_REQ = PolicyRequest(
    question="Should the city relax zoning to allow more multi-family housing?",
    geography="Metro",
    objective="increase housing affordability",
)


def test_housing_domain_detected_and_numeric():
    fc = run_forecast(REC, HOUSING_REQ)
    assert fc.mode == "numeric" and fc.domain == "housing"
    # baseline = status quo => no new units
    assert fc.baseline.inputs["new_units"] == 0
    # optimistic builds more than conservative
    assert fc.optimistic.inputs["new_units"] > fc.conservative.inputs["new_units"]
    # more supply => rent goes down (negative change) in the expected case
    assert fc.expected.inputs["rent_change_pct"] < 0
