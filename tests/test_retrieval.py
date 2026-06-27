"""Tests for the policy evidence retrieval seam (mock mode)."""

from models import EvidenceItem
from retrieval import retrieve_policy_evidence
from source_scoring import score_source


def test_returns_evidence_items():
    items = retrieve_policy_evidence(["congestion pricing"], geography="Boston, MA")
    assert items and all(isinstance(e, EvidenceItem) for e in items)


def test_top_k_respected():
    items = retrieve_policy_evidence(["x"], top_k=2)
    assert len(items) <= 2


def test_source_type_filter():
    items = retrieve_policy_evidence(["x"], source_types=["academic"])
    assert all(e.source_type == "academic" for e in items)


def test_ranked_by_relevance_times_credibility():
    items = retrieve_policy_evidence(["x"], top_k=10)
    scores = [e.relevance_score * e.credibility_score for e in items]
    assert scores == sorted(scores, reverse=True)


def test_source_scoring_prefers_government_and_recency():
    gov = score_source("government_report", "2024")
    other = score_source("other", "2001")
    assert gov > other
