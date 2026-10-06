
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class AuditStore:
    """
    Deployment audit store.

    PostgreSQL is recommended for multi-worker deployment. SQLite remains available
    only for single-node research/demo operation.
    """
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv(
            "TRUST_SLEEP_DATABASE_URL",
            "sqlite:///trust_sleep_audit.sqlite3",
        )
        self.backend = (
            "postgres"
            if self.database_url.startswith(("postgres://", "postgresql://"))
            else "sqlite"
        )
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        if self.backend == "postgres":
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError(
                    "Install psycopg[binary] for PostgreSQL deployment."
                ) from exc
            connection = psycopg.connect(self.database_url)
        else:
            path = self.database_url.removeprefix("sqlite:///")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        auto_id = (
            "BIGSERIAL PRIMARY KEY"
            if self.backend == "postgres"
            else "INTEGER PRIMARY KEY AUTOINCREMENT"
        )
        with self.connection() as connection:
            cursor = connection.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id {auto_id},
                    timestamp DOUBLE PRECISION NOT NULL,
                    case_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_checksum TEXT NOT NULL,
                    prediction_json TEXT NOT NULL
                )
            """)
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS clinician_feedback (
                    id {auto_id},
                    timestamp DOUBLE PRECISION NOT NULL,
                    case_id TEXT NOT NULL,
                    clinician_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    corrected_subtype TEXT,
                    artifact INTEGER NOT NULL DEFAULT 0,
                    second_review INTEGER NOT NULL DEFAULT 0,
                    note TEXT,
                    model_version TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_case
                ON audit_events(case_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_case
                ON clinician_feedback(case_id)
            """)

    def log_prediction(
        self,
        *,
        case_id: str,
        session_id: str,
        model_version: str,
        input_checksum: str,
        prediction: Dict[str, Any],
    ) -> None:
        placeholder = "%s" if self.backend == "postgres" else "?"
        values = ",".join([placeholder] * 6)
        with self.connection() as connection:
            connection.cursor().execute(
                f"""
                INSERT INTO audit_events
                (timestamp, case_id, session_id, model_version,
                 input_checksum, prediction_json)
                VALUES ({values})
                """,
                (
                    time.time(),
                    case_id,
                    session_id,
                    model_version,
                    input_checksum,
                    json.dumps(prediction),
                ),
            )

    def add_feedback(
        self,
        *,
        case_id: str,
        clinician_id: str,
        action: str,
        model_version: str,
        corrected_subtype: Optional[str] = None,
        artifact: bool = False,
        second_review: bool = False,
        note: Optional[str] = None,
    ) -> None:
        allowed = {
            "confirm_event",
            "reject_event",
            "change_subtype",
            "mark_artifact",
            "request_second_review",
        }
        if action not in allowed:
            raise ValueError(f"Unsupported feedback action: {action}")
        placeholder = "%s" if self.backend == "postgres" else "?"
        values = ",".join([placeholder] * 9)
        with self.connection() as connection:
            connection.cursor().execute(
                f"""
                INSERT INTO clinician_feedback
                (timestamp, case_id, clinician_id, action, corrected_subtype,
                 artifact, second_review, note, model_version)
                VALUES ({values})
                """,
                (
                    time.time(),
                    case_id,
                    clinician_id,
                    action,
                    corrected_subtype,
                    int(artifact),
                    int(second_review),
                    note,
                    model_version,
                ),
            )
