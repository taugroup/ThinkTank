"""Agent framework: the shared local-model wrapper.

OWNER: Person 1 (orchestration). This is the keystone the other agents reuse:
`run_structured()` calls a local Ollama model, forces JSON output, validates it
against a Pydantic schema, retries on failure, optionally escalates to a frontier
model, and logs a ModelEvent. It NEVER raises on model/parse failure — it returns
`(None, event)` so callers can fall back to their mock. That guarantees the app
keeps working whether or not Ollama is running.

The legacy `build_local_agent` (agno) is retained for the old meeting code.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import (
    ENABLE_FRONTIER_FALLBACK,
    FRONTIER_MODEL,
    LOCAL_MODEL,
    MAX_SCHEMA_RETRIES,
)
from logger import log_model_event
from models import ModelEvent

T = TypeVar("T", bound=BaseModel)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> str:
    """Best-effort: return the first {...} block (handles stray prose/think tags)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = _JSON_OBJ.search(text)
    return match.group(0) if match else text.strip()


def _ollama_call(model: str, prompt: str, temperature: float) -> str:
    """Call a local Ollama model in JSON mode. Imported lazily so the module loads
    (and mock mode runs) even when the `ollama` package isn't installed."""
    import ollama  # lazy

    resp = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise policy analyst. Respond with a single "
                "valid JSON object only — no prose, no markdown, no code fences.",
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": temperature},
    )
    return resp["message"]["content"]


def _frontier_call(model: str, prompt: str, temperature: float) -> str:
    """Optional frontier fallback (Anthropic). Imported lazily; off by default."""
    import anthropic  # lazy

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system="Respond with a single valid JSON object only.",
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def run_structured(
    agent_name: str,
    prompt: str,
    schema: Type[T],
    *,
    temperature: float = 0.2,
    model: Optional[str] = None,
    call: Optional[Callable[[str, str, float], str]] = None,
) -> tuple[Optional[T], ModelEvent]:
    """Run a local model and return a validated `schema` instance (or None).

    Strategy: try the local model up to MAX_SCHEMA_RETRIES + 1 times, validating the
    JSON each time. If all local attempts fail and frontier fallback is enabled, try
    once more with the frontier model. Always logs and returns a ModelEvent; never
    raises. `call` is injectable for tests (defaults to the real Ollama call).

    Returns (validated_obj_or_None, model_event).
    """
    model = model or LOCAL_MODEL
    caller = call or _ollama_call
    started = time.time()
    last_error: Optional[str] = None

    for attempt in range(MAX_SCHEMA_RETRIES + 1):
        try:
            raw = caller(model, prompt, temperature)
            obj = schema.model_validate_json(_extract_json(raw))
            event = ModelEvent(
                agent=agent_name,
                model=model,
                latency_ms=int((time.time() - started) * 1000),
                schema_valid=True,
                escalated=False,
                error=None,
            )
            log_model_event(event)
            return obj, event
        except (ValidationError, json.JSONDecodeError, KeyError, Exception) as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:300]

    # Optional frontier escalation after local attempts are exhausted.
    if ENABLE_FRONTIER_FALLBACK and FRONTIER_MODEL:
        try:
            raw = _frontier_call(FRONTIER_MODEL, prompt, temperature)
            obj = schema.model_validate_json(_extract_json(raw))
            event = ModelEvent(
                agent=agent_name,
                model=FRONTIER_MODEL,
                latency_ms=int((time.time() - started) * 1000),
                schema_valid=True,
                escalated=True,
                error=None,
            )
            log_model_event(event)
            return obj, event
        except Exception as exc:  # noqa: BLE001
            last_error = f"frontier {type(exc).__name__}: {exc}"[:300]

    event = ModelEvent(
        agent=agent_name,
        model=model,
        latency_ms=int((time.time() - started) * 1000),
        schema_valid=False,
        escalated=ENABLE_FRONTIER_FALLBACK and bool(FRONTIER_MODEL),
        error=last_error,
    )
    log_model_event(event)
    return None, event


# ---------------------------------------------------------------------------
# Legacy agno agent builder (used by the retired meeting code; kept for transition)
# ---------------------------------------------------------------------------

def build_local_agent(
    *,
    name: str,
    description: str,
    role: str,
    temperature: float = 0.2,
    memory=None,
    storage=None,
    enable_agentic_memory: bool = False,
    **extra_agent_kwargs,
):
    """Return an agno Agent backed by a local Ollama model. Imported lazily so the
    new policy stack doesn't require agno to be installed."""
    from agno.agent import Agent
    from agno.models.ollama import Ollama

    model = Ollama(id=LOCAL_MODEL, options={"temperature": temperature})
    return Agent(
        name=name,
        description=description,
        role=role,
        model=model,
        markdown=True,
        memory=memory,
        enable_agentic_memory=enable_agentic_memory,
        storage=storage,
        add_history_to_messages=True,
        num_history_runs=3,
        **extra_agent_kwargs,
    )
