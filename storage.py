"""SQLite persistence for policy runs.

OWNER: Person 4 (UI/persistence). Stores the full structured result plus model
events so nothing important lives only inside LangGraph state. Foundation version:
a single `runs` table holding the JSON-serialized PolicyRunResult, plus a
`model_events` table for the dashboard. P4 may normalize further (tasks, outputs,
forecasts as separate tables) without changing the public functions.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Optional

from config import RUNS_DB_PATH
from models import PolicyRunResult


def _connect(db_path: str = RUNS_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            question TEXT,
            geography TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            result_json TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS model_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            agent TEXT,
            model TEXT,
            latency_ms INTEGER,
            schema_valid INTEGER,
            escalated INTEGER,
            error TEXT
        )"""
    )
    return conn


def save_run(result: PolicyRunResult, db_path: str = RUNS_DB_PATH) -> None:
    """Persist a complete run plus its model events."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, question, geography, result_json) "
            "VALUES (?, ?, ?, ?)",
            (
                result.run_id,
                result.request.question,
                result.request.geography,
                result.model_dump_json(),
            ),
        )
        conn.execute("DELETE FROM model_events WHERE run_id = ?", (result.run_id,))
        for ev in result.model_events:
            conn.execute(
                "INSERT INTO model_events "
                "(run_id, agent, model, latency_ms, schema_valid, escalated, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    ev.agent,
                    ev.model,
                    ev.latency_ms,
                    int(ev.schema_valid),
                    int(ev.escalated),
                    ev.error,
                ),
            )


def load_run(run_id: str, db_path: str = RUNS_DB_PATH) -> Optional[PolicyRunResult]:
    """Load a previously persisted run by id, or None if not found."""
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT result_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if not row:
        return None
    return PolicyRunResult.model_validate(json.loads(row[0]))


def list_runs(db_path: str = RUNS_DB_PATH) -> list[dict]:
    """Return run metadata (newest first) for a history selector."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT run_id, question, geography, created_at FROM runs "
            "ORDER BY created_at DESC"
        ).fetchall()
    return [
        {"run_id": r[0], "question": r[1], "geography": r[2], "created_at": r[3]}
        for r in rows
    ]
