"""Central configuration for the Policy Think Tank.

Sections are namespaced by workstream so the four branches make additive edits
with minimal merge conflicts. Everything has a local-first default.
"""

import os

# --- Legacy / RAG plumbing (P2) -------------------------------------------
LOG_DIR = "./logs"
DB_PATH = "chroma_db"

# Text splitter configurations
CHUNK_SIZE = 512
CHUNK_OVERLAP = 120
BATCH_SIZE = 5

# HNSW configurations
HNSW_SPACE = "cosine"

EMBEDDING_MODEL = "nomic-embed-text"

# Single shared evidence collection (replaces per-agent collections). Domains are
# distinguished by metadata filtering, not by separate collections.
EVIDENCE_COLLECTION = "policy_evidence"

# --- Model strategy (P1) ---------------------------------------------------
# Local-only by default. Frontier fallback is OFF unless explicitly enabled.
LOCAL_MODEL = os.getenv("POLICY_LOCAL_MODEL", "qwen3:8b")
FRONTIER_MODEL = os.getenv("POLICY_FRONTIER_MODEL", "")  # empty => disabled
ENABLE_FRONTIER_FALLBACK = os.getenv("POLICY_ENABLE_FALLBACK", "0") == "1"


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


# Global default. When true (the safe default), agents return fixture-backed mocks
# and never call Ollama. Each component below can be flipped independently so the
# three workstreams can turn their own piece "real" without forcing the others.
MOCK_MODE = _flag("POLICY_MOCK_MODE", True)
MOCK_DIRECTOR = _flag("POLICY_MOCK_DIRECTOR", MOCK_MODE)   # Person A
MOCK_RESEARCH = _flag("POLICY_MOCK_RESEARCH", MOCK_MODE)   # Person B
MOCK_ANALYSIS = _flag("POLICY_MOCK_ANALYSIS", MOCK_MODE)   # Person C (impl + red-team)

MAX_SCHEMA_RETRIES = 2  # local re-asks before considering escalation

# --- Orchestration (P1) ----------------------------------------------------
MAX_REVISION_LOOPS = 1  # red-team -> revise -> red-team, at most this many
REVISE_ON_SEVERITY = "high"  # severity that forces a revision loop

# --- Retrieval (P2) --------------------------------------------------------
DEFAULT_TOP_K = 6
MAX_CHUNKS_PER_SOURCE = 2

# --- Storage (P4) ----------------------------------------------------------
RUNS_DB_PATH = "policy_runs.db"  # SQLite: runs, tasks, outputs, forecasts, events
WORKSPACE_DIR = "workspace"  # original uploaded PDFs/CSVs

# --- Paths -----------------------------------------------------------------
DATA_DIR = "data"  # each domain gets a subfolder, e.g. data/transportation (sample)
SAMPLE_MANIFEST = os.path.join(DATA_DIR, "transportation", "source_manifest.json")
EXAMPLES_DIR = "examples"
SKILLS_DIR = "skills"
