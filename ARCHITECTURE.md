# Policy Think Tank — Architecture

A local-first, multi-agent **policy think tank** for **any policy question** in any
domain (transportation, housing, education, health, fiscal policy, …). The user
submits a broad policy question (e.g. *"Should Boston implement congestion pricing
downtown?"* or *"Should we adopt a guaranteed basic income pilot?"*) and the system
runs a Policy Director → stakeholder research → synthesis → recommendation →
red-team → forecast pipeline, returning a recommendation, evidence, implementation
plan, risks, and forecast scenarios. Transportation is just the first worked example.

> **Attribution.** This project is adapted from **TAU Group's ThinkTank**
> (Texas A&M University), MIT-licensed. See `LICENSE`. The original framework was a
> scientific virtual-lab meeting simulator; we refactored its agent + RAG plumbing
> into a policy-analysis workflow.

## Workflow (LangGraph state graph)

```
START → plan_policy → research → stakeholder_research → synthesize_research
      → implementation_and_recommendation → red_team_review
      → should_revise? ──yes─→ revise_recommendation ─→ red_team_review
                      └──no──→ run_forecast → finalize_result → END
```

Revision loops are bounded by `config.MAX_REVISION_LOOPS` (default 1).
`graph.py` falls back to an equivalent sequential executor if LangGraph is absent.

## Agent roster
A **Policy Director** that plans and **assigns each worker its agent type + skills**,
three worker agent types, and a **Red-Team** for the revision loop:
- **Policy Director** (`agents/policy_director.py`) — defines objective, picks the
  stakeholder roster, creates research + stakeholder tasks, and **chooses each task's
  skills** from the skill registry. Also revises the recommendation.
- **Research agent** (`agents/research_agent.py`) — gathers objective, cited evidence
  on sub-topics (no perspective). Output: `ResearchBrief`s.
- **Stakeholder agent** (`agents/stakeholder_research.py`) — one shared implementation
  instantiated into multiple perspectives; cited findings via RAG.
- **Data Analyst** (`agents/data_analyst.py`) — synthesis + recommendation + phased
  plan.
- **Red-Team** (`agents/red_team_agent.py`) — adversarial critique driving revision.

## Orchestrator-assigned skills
Skills are **not** hardcoded on agents. `skills_registry.py` reads `skills/*/SKILL.md`
into a catalog; the Director chooses, per task, which skills the worker needs and
attaches them to the `PolicyTask`. Each worker loads exactly those via
`build_packet(skill_keys=...)`. So the same stakeholder agent runs with different
skills depending on what the Director decides the task requires.

## Context strategy
No full transcript is passed to agents. `context_builder.build_packet()` assembles a
compact `BriefingPacket` (question, objective, task, relevant prior findings, top
evidence chunks, skill guidance, output schema). Full data lives in LangGraph state,
SQLite (`storage.py`), and the vector store — never in prompts.

## Determinism boundary
- **LLM**: extraction, query generation, classification, summarization, stakeholder
  analysis, structured JSON, initial critique.
- **Python (deterministic)**: forecasts (`forecasters/`), source scoring
  (`source_scoring.py`), schema validation, metrics, sensitivity analysis.

## Forecasting is optional & domain-gated
`forecasting.run_forecast(recommendation, request)` dispatches via a registry of
deterministic domain modules (`forecasters/`):
- If a module **matches** the question (e.g. `forecasters/transportation.py`), it
  runs a numeric 4-scenario forecast (`mode="numeric"`).
- If **no** module matches, the system returns a **qualitative directional outlook**
  (`mode="qualitative"`) with NO fabricated numbers — honoring "no unsupported
  numeric claims" for domains without a calibrated model.
Adding a new domain = add one module exposing `matches/derive_parameters/forecast`.
No LLM ever computes forecast arithmetic.

## Model strategy
Local Ollama by default (`config.LOCAL_MODEL`). Optional frontier fallback
(`ENABLE_FRONTIER_FALLBACK`) only on repeated schema/citation failure or to resolve
materially conflicting conclusions. Every model call logs a `ModelEvent`
(`logger.log_model_event`). The system runs local-only by default; `MOCK_MODE=1`
needs no model at all.

## Public seams (frozen contracts)
| Function | Module | Owner |
|---|---|---|
| `run_policy_analysis(request)` | `orchestrator.py` | P1 |
| `run_stakeholder_research(request, tasks, stakeholders)` | `agents/stakeholder_research.py` | P2 |
| `retrieve_policy_evidence(queries, geography, source_types, top_k)` | `retrieval.py` | P2 |
| `synthesize_research` / `create_policy_recommendation` | `agents/implementation_agent.py` | P3 |
| `run_forecast(recommendation, request)` + `forecasters/` modules | `forecasting.py` | P3 |
| `execute_policy_analysis(request)` | `app.py` | P4 |

All shared schemas live in `models.py` and are **frozen** — change only with team
sign-off.

## Scope
General-purpose: works for any policy question. Forecasting is numeric only where a
deterministic domain module exists (transportation today); all other domains get a
qualitative outlook until a module is added. New domains add a curated evidence
collection + a `forecasters/<domain>.py` module.
