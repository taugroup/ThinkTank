"""Deterministic source credibility scoring.

OWNER: Person 2 (research/RAG). LLM-free: credibility is a function of source type,
recency, and geography match so it is reproducible and auditable. This is a
foundation stub with a sane default scheme; P2 may refine the weights.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# Base credibility by source type (0-1).
_TYPE_WEIGHTS = {
    "government_report": 0.9,
    "academic": 0.85,
    "open_data": 0.8,
    "case_study": 0.7,
    "uploaded_pdf": 0.6,
    "structured_csv": 0.6,
    "other": 0.4,
}


def _recency_bonus(publication_date: Optional[str]) -> float:
    """Up to +0.1 for sources within the last 5 years; older decays toward 0."""
    if not publication_date:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            year = datetime.strptime(publication_date, fmt).year
            break
        except ValueError:
            continue
    else:
        return 0.0
    age = datetime.now().year - year
    if age <= 0:
        return 0.1
    return max(0.0, 0.1 * (1 - age / 5))


def score_source(
    source_type: str,
    publication_date: Optional[str] = None,
    geography: Optional[str] = None,
    target_geography: Optional[str] = None,
) -> float:
    """Return a credibility score in [0, 1] for a source."""
    base = _TYPE_WEIGHTS.get(source_type, _TYPE_WEIGHTS["other"])
    score = base + _recency_bonus(publication_date)
    if geography and target_geography and geography.lower() == target_geography.lower():
        score += 0.05
    return round(min(1.0, score), 3)
