"""SQLite persistence for the fleet control service."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)
DATABASE_SCHEMA_VERSION = 3

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    runner_id TEXT NOT NULL,
    task_class TEXT NOT NULL CHECK (task_class IN ('manual', 'canary', 'sweep')),
    priority INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'claimed', 'accepted', 'completed', 'failed', 'cancelled')
    ),
    submitted_at TEXT NOT NULL,
    submitted_by TEXT NOT NULL,
    canary_id TEXT,
    envelope_json TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    claimed_at TEXT,
    lease_expires TEXT,
    claim_token TEXT,
    accepted_at TEXT,
    completed_at TEXT,
    outcome_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_runner_state_priority
    ON tasks(runner_id, state, priority DESC, submitted_at ASC);
CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);

CREATE TABLE IF NOT EXISTS runner_status (
    runner_id TEXT PRIMARY KEY,
    status_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    task_id TEXT,
    runner_id TEXT,
    old_state TEXT,
    new_state TEXT,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_task ON audit_log(task_id);
CREATE INDEX IF NOT EXISTS idx_audit_runner ON audit_log(runner_id);
"""


_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS canary_schedule (
    runner_id   TEXT    NOT NULL,
    profile_id  TEXT    NOT NULL,
    utc_date    TEXT    NOT NULL,
    state       TEXT    NOT NULL CHECK (
        state IN ('created', 'missed', 'expired')
    ),
    task_id     TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (runner_id, profile_id, utc_date)
);
"""


_SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS canary_observations (
    runner_id        TEXT    NOT NULL,
    profile_id       TEXT    NOT NULL,
    profile_version  INTEGER NOT NULL,
    utc_date         TEXT    NOT NULL,
    task_id          TEXT    NOT NULL,
    score            REAL,
    completed_at     TEXT,
    phase            TEXT    NOT NULL CHECK (
        phase IN ('observation', 'calibrating', 'ready')
    ),
    accepted         INTEGER NOT NULL DEFAULT 1 CHECK (accepted IN (0, 1)),
    rejection_reason TEXT,
    ref_median       REAL,
    ref_mad          REAL,
    delta_pct        REAL,
    series_ordinal        INTEGER,
    ref_sample_count      INTEGER,
    candidate_warning_pct REAL,
    candidate_alarm_pct   REAL,
    candidate_signal      TEXT    CHECK (
        candidate_signal IN ('insufficient-data', 'within', 'warning', 'alarm')
    ),
    actionable            INTEGER NOT NULL DEFAULT 0 CHECK (actionable = 0),
    window_start     TEXT,
    window_end       TEXT,
    environment_json TEXT,
    env_change_annotation TEXT,
    provenance_schema_version INTEGER,
    created_at       TEXT    NOT NULL,
    PRIMARY KEY (runner_id, profile_id, profile_version, utc_date, task_id)
);
CREATE INDEX IF NOT EXISTS idx_canary_obs_series
    ON canary_observations(runner_id, profile_id, profile_version, utc_date);
CREATE INDEX IF NOT EXISTS idx_canary_obs_accepted
    ON canary_observations(runner_id, profile_id, profile_version, accepted, utc_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_canary_obs_one_accepted_per_date
    ON canary_observations(runner_id, profile_id, profile_version, utc_date)
    WHERE accepted = 1;

CREATE TABLE IF NOT EXISTS canary_calibration_reports (
    report_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    runner_id                 TEXT    NOT NULL,
    profile_id                TEXT    NOT NULL,
    profile_version           INTEGER NOT NULL,
    sample_count              INTEGER NOT NULL,
    date_range_start          TEXT    NOT NULL,
    date_range_end            TEXT    NOT NULL,
    median_score              REAL    NOT NULL,
    mad                       REAL    NOT NULL,
    robust_sigma              REAL    NOT NULL,
    variability_floor_pct     REAL    NOT NULL,
    candidate_warning_pct     REAL,
    candidate_alarm_pct       REAL,
    recommended_warning_pct   REAL    NOT NULL,
    recommended_alarm_pct     REAL    NOT NULL,
    status                    TEXT    NOT NULL CHECK (
        status IN ('ready-for-review', 'accepted', 'rejected')
    ),
    report_json               TEXT    NOT NULL,
    created_at                TEXT    NOT NULL,
    UNIQUE (runner_id, profile_id, profile_version)
);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: Optional[datetime] = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ControlDatabase:
    def __init__(self, path: Path, audit_jsonl_path: Path):
        self.path = path
        self.audit_jsonl_path = audit_jsonl_path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        os.chmod(self.path, 0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = connection.execute("SELECT MAX(version) AS version FROM schema_migrations").fetchone()
            version = row["version"] or 0
            if version > DATABASE_SCHEMA_VERSION:
                raise RuntimeError(f"database schema {version} is newer than supported {DATABASE_SCHEMA_VERSION}")
            if version < 1:
                connection.executescript(_SCHEMA_V1)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, utc_text()),
                )
            if version < 2:
                connection.executescript(_SCHEMA_V2)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (2, utc_text()),
                )
            if version < 3:
                connection.executescript(_SCHEMA_V3)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (3, utc_text()),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def insert_audit(
        self,
        connection: sqlite3.Connection,
        *,
        actor: str,
        action: str,
        task_id: Optional[str] = None,
        runner_id: Optional[str] = None,
        old_state: Optional[str] = None,
        new_state: Optional[str] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp": utc_text(),
            "actor": actor,
            "action": action,
            "task_id": task_id,
            "runner_id": runner_id,
            "old_state": old_state,
            "new_state": new_state,
            "detail": detail,
        }
        connection.execute(
            "INSERT INTO audit_log(timestamp, actor, action, task_id, runner_id, old_state, new_state, detail_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["timestamp"],
                actor,
                action,
                task_id,
                runner_id,
                old_state,
                new_state,
                json.dumps(detail, sort_keys=True) if detail is not None else None,
            ),
        )
        return record

    def append_audit_jsonl(self, record: dict[str, Any]) -> None:
        """Best-effort mirror; SQLite audit_log remains authoritative."""
        try:
            self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.warning("unable to append audit JSONL", exc_info=True)
