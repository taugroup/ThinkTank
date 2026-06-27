"""Schema contract tests. These guard the FROZEN models against accidental drift."""

import json
import os

import pytest

from models import (
    CritiqueResult,
    EvidenceItem,
    Finding,
    ForecastResult,
    PolicyRecommendation,
    PolicyRequest,
    PolicyRunResult,
    StakeholderResearchResult,
)

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as f:
        return json.load(f)


def test_list_defaults_are_independent():
    """Field(default_factory=list) must not share state between instances."""
    a = Finding(claim="a")
    b = Finding(claim="b")
    a.evidence_ids.append("x")
    assert b.evidence_ids == []


def test_policy_request_minimal():
    req = PolicyRequest(question="Q?", geography="Boston", objective="reduce")
    assert req.constraints == [] and req.timeline is None


def test_fixtures_validate():
    PolicyRequest.model_validate(_load("sample_request.json"))
    PolicyRecommendation.model_validate(_load("sample_recommendation.json"))
    ForecastResult.model_validate(_load("sample_forecast.json"))
    PolicyRunResult.model_validate(_load("sample_result.json"))
    for r in _load("sample_research.json"):
        StakeholderResearchResult.model_validate(r)


def test_evidence_item_roundtrip():
    e = EvidenceItem(source_id="S1", title="T", text="body")
    assert EvidenceItem.model_validate_json(e.model_dump_json()) == e


def test_critique_severity_literal():
    with pytest.raises(Exception):
        CritiqueResult(severity="catastrophic")
