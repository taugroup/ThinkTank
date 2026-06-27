# Model Selection

The Policy Think Tank is **local-first**. It runs entirely on local Ollama models by
default, with an optional, narrowly-scoped frontier fallback.

## Local models (default)
| Use | Model (default) |
|---|---|
| Stakeholder analysis, synthesis, critique, structured JSON | `qwen3:8b` (`config.LOCAL_MODEL`) |
| Embeddings (RAG) | `nomic-embed-text` (`config.EMBEDDING_MODEL`) |

Used for: document extraction, query generation, classification, evidence
summarization, stakeholder analysis, structured JSON generation, initial critique.

## Deterministic Python (never the LLM)
Forecast calculations, source credibility scoring, schema validation, metrics, and
sensitivity analysis. The LLM may *set* `ForecastParameters` but never computes
forecast numbers.

## Frontier fallback (optional, OFF by default)
Enable with `POLICY_ENABLE_FALLBACK=1` and set `POLICY_FRONTIER_MODEL`. Triggered
**only** when:
- local structured output repeatedly fails schema validation
  (after `config.MAX_SCHEMA_RETRIES` retries),
- required citations are missing,
- the model repeatedly fails the requested schema, or
- the Policy Director must resolve materially conflicting conclusions.

The latest Claude models are appropriate fallbacks (e.g. Claude Opus 4.8 /
`claude-opus-4-8`); supply credentials via env, never in code.

## Observability
Every model call logs a `ModelEvent`:
```json
{"agent":"research_agent","model":"qwen3:8b","latency_ms":2000,"schema_valid":true,"escalated":false,"error":null}
```
Written to `logs/model_events.jsonl` and persisted per-run in SQLite. The evals
harness reports local-call count vs. escalation count so "local-only" vs
"local + fallback" can be compared directly.
