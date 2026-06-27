---
name: policy-research
description: Retrieve, evaluate, and cite policy evidence with explicit credibility and assumption labeling.
owner: Person 2 (research/RAG)
---

# Policy Research Skill

Use this skill when researching a policy question from a stakeholder perspective.

## Trusted-source hierarchy (highest to lowest)
1. Government reports and official open-data exports
2. Peer-reviewed academic research
3. Comparable-city case studies
4. User-uploaded PDFs / structured CSVs
5. Anything else (label clearly)

## Retrieval strategy
- Generate 2-4 targeted queries from your assigned perspective and priorities.
- Call `retrieve_policy_evidence(queries, geography, source_types, top_k)`.
- Prefer evidence matching the request geography; otherwise use analogous cities and say so.

## Citation requirements
- Every material claim MUST list the `source_id`s that support it in `evidence_ids`.
- Page-level attribution where available.

## Source credibility scoring
- Credibility is computed deterministically (`source_scoring.py`) from source type,
  recency, and geography match. Do not invent credibility numbers.

## Handling missing evidence
- If no evidence supports a claim, either drop the claim or move it to `assumptions`.
- Record what you could not find in `data_gaps`.

## Numerical-claim rule
- Any number in a finding must trace to an `evidence_id`, OR be listed in
  `assumptions`. Never present an unsupported number as fact.

## Required structured output
Return a `StakeholderResearchResult`: `findings` (each a `Finding` with
`evidence_ids`, `confidence`, `assumptions`, `limitations`), `likely_position`,
`concerns`, `proposed_mitigations`, `data_gaps`, and a one-line `handoff_summary`.
