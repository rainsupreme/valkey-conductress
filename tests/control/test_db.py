import json
import sqlite3
import stat

import pytest

from conductress.control.db import DATABASE_SCHEMA_VERSION, ControlDatabase


def test_database_initializes_wal_schema_and_reopens(control_env):
    database = control_env["database"]
    with database.read() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == DATABASE_SCHEMA_VERSION
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {
        "tasks",
        "runner_status",
        "audit_log",
        "schema_migrations",
        "canary_schedule",
        "canary_observations",
        "canary_calibration_reports",
    } <= tables
    assert stat.S_IMODE(database.path.stat().st_mode) == 0o600

    ControlDatabase(database.path, database.audit_jsonl_path).initialize()


def test_transaction_rolls_back(control_env):
    database = control_env["database"]
    with pytest.raises(RuntimeError):
        with database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO runner_status(runner_id, status_json, updated_at) VALUES ('x', '{}', 'now')"
            )
            raise RuntimeError("rollback")

    with database.read() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runner_status").fetchone()[0] == 0


def test_audit_is_stored_and_mirrored(control_env):
    database = control_env["database"]
    with database.transaction(immediate=True) as connection:
        record = database.insert_audit(
            connection,
            actor="operator:test",
            action="test.action",
            task_id="task-1",
            new_state="queued",
        )
    database.append_audit_jsonl(record)

    with database.read() as connection:
        row = connection.execute("SELECT * FROM audit_log").fetchone()
    assert row["actor"] == "operator:test"
    assert row["new_state"] == "queued"
    assert json.loads(database.audit_jsonl_path.read_text(encoding="utf-8"))["action"] == "test.action"


def test_read_context_closes_connection(control_env):
    database = control_env["database"]
    with database.read() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def test_v4_adds_batch_submitter_columns_and_new_states(control_env):
    database = control_env["database"]
    with database.read() as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "batch_id" in columns
    assert "submitter_login" in columns
    # The two new lifecycle states are accepted by the state CHECK constraint.
    now = "2026-01-01T00:00:00Z"
    for state in ("pending-approval", "cancel-requested"):
        with database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO tasks(task_id, runner_id, task_class, priority, state, submitted_at, "
                "submitted_by, envelope_json, batch_id, submitter_login, created_at, updated_at) "
                "VALUES (?, 'armbench', 'manual', 100, ?, ?, 'rain', '{}', 'batch-1', 'rain', ?, ?)",
                (f"task-{state}", state, now, now, now),
            )
        with database.read() as connection:
            row = connection.execute(
                "SELECT state, batch_id, submitter_login FROM tasks WHERE task_id=?", (f"task-{state}",)
            ).fetchone()
        assert row["state"] == state
        assert row["batch_id"] == "batch-1"
        assert row["submitter_login"] == "rain"


def test_v4_rejects_unknown_state(control_env):
    database = control_env["database"]
    now = "2026-01-01T00:00:00Z"
    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction(immediate=True) as connection:
            connection.execute(
                "INSERT INTO tasks(task_id, runner_id, task_class, priority, state, submitted_at, "
                "submitted_by, envelope_json, created_at, updated_at) "
                "VALUES ('bad', 'armbench', 'manual', 100, 'nonsense', ?, 'rain', '{}', ?, ?)",
                (now, now, now),
            )


def test_v3_database_upgrades_to_v4_preserving_rows(tmp_path):
    """A database left at schema 3 gains the v4 columns and keeps its rows."""
    from conductress.control.db import _SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3, utc_text

    db_path = tmp_path / "legacy.db"
    audit_path = tmp_path / "audit.jsonl"
    # Build a v3 database by hand and seed one task row.
    connection = sqlite3.connect(db_path, isolation_level=None)
    connection.executescript("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);")
    connection.executescript(_SCHEMA_V1)
    connection.executescript(_SCHEMA_V2)
    connection.executescript(_SCHEMA_V3)
    for version in (1, 2, 3):
        connection.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, utc_text()))
    connection.execute(
        "INSERT INTO tasks(task_id, runner_id, task_class, priority, state, submitted_at, "
        "submitted_by, envelope_json, created_at, updated_at) "
        "VALUES ('legacy-1', 'armbench', 'manual', 100, 'queued', '2026-01-01T00:00:00Z', "
        "'rain', '{\"x\":1}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
    )
    connection.close()

    ControlDatabase(db_path, audit_path).initialize()

    reopened = sqlite3.connect(db_path)
    reopened.row_factory = sqlite3.Row
    version = reopened.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()["v"]
    columns = {row[1] for row in reopened.execute("PRAGMA table_info(tasks)").fetchall()}
    row = reopened.execute("SELECT * FROM tasks WHERE task_id='legacy-1'").fetchone()
    reopened.close()
    assert version == DATABASE_SCHEMA_VERSION
    assert {"batch_id", "submitter_login"} <= columns
    assert row["state"] == "queued"
    assert row["batch_id"] is None
    assert row["envelope_json"] == '{"x":1}'
