"""Tests for run_structured — validation, retry, and fallback logic.

Uses an injected `call` stub so these run with no Ollama installed.
"""

from pydantic import BaseModel

from agent_builder import _extract_json, run_structured


class _Out(BaseModel):
    name: str
    value: int


def test_valid_first_try():
    obj, ev = run_structured(
        "t", "p", _Out, call=lambda m, p, t: '{"name": "a", "value": 1}'
    )
    assert obj == _Out(name="a", value=1)
    assert ev.schema_valid and not ev.escalated and ev.error is None


def test_retries_then_succeeds():
    seq = iter(['not json', '{"bad": true}', '{"name": "ok", "value": 2}'])
    obj, ev = run_structured("t", "p", _Out, call=lambda m, p, t: next(seq))
    assert obj == _Out(name="ok", value=2)
    assert ev.schema_valid


def test_all_invalid_returns_none():
    obj, ev = run_structured("t", "p", _Out, call=lambda m, p, t: "garbage")
    assert obj is None
    assert ev.schema_valid is False
    assert ev.error is not None


def test_model_exception_is_caught_not_raised():
    def boom(m, p, t):
        raise RuntimeError("ollama down")

    obj, ev = run_structured("t", "p", _Out, call=boom)
    assert obj is None and ev.schema_valid is False


def test_extract_json_strips_think_and_prose():
    raw = 'thinking...<think>noise</think> here: {"name": "x", "value": 3} done'
    assert _extract_json(raw).strip().startswith("{")
    obj = _Out.model_validate_json(_extract_json(raw))
    assert obj.value == 3
