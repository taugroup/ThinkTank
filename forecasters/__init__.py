"""Pluggable deterministic domain forecasters.

Register a domain module here to enable numeric forecasting for that domain. Any
policy question that matches no module falls back to a qualitative outlook
(see forecasting.qualitative_outlook) — the system never fabricates numbers.

OWNER: Person 3 (impl/forecast/evals).
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional

from models import PolicyRequest

from forecasters import housing, transportation

# Ordered registry; first match wins.
REGISTRY: list[ModuleType] = [transportation, housing]


def detect_domain(request: PolicyRequest) -> Optional[ModuleType]:
    """Return the first domain module that can forecast this request, or None."""
    for module in REGISTRY:
        if module.matches(request):
            return module
    return None
