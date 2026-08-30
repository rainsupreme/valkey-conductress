"""Tests for remote task-ID preservation through the full lifecycle.

The canary scheduler produces deterministic task IDs like
``canary:armbench:default-throughput:2026-08-30``.  These must survive
unchanged through: envelope → claim → import → queue file → from_file
reload → FileProtocol execution → RunnerMailbox owns/stage/outcome →
remote outcome reporting.

Ordinary local tasks with no override must keep their timestamp-derived
IDs and remain serialization-compatible.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from conductress import task_queue
from conductress.file_protocol import FileProtocol
from conductress.fleet_client import FleetClientError
from conductress.runner_mailbox import RunnerMailbox
from conductress.task_envelope import build_task_envelope, serialize_task
from conductress.task_queue import BaseTaskData, TaskQueue
from conductress.utility import datetime_to_task_id

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden_tasks" / "PerfTaskData.json"

CANARY_TASK_ID = "canary:armbench:default-throughput:2026-08-30"


@pytest.fixture(autouse=True)
def valid_source(monkeypatch):
    monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])


def _golden_task_doc():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _canary_task_doc():
    """Simulate a canary scheduler envelope's inner task document."""
    doc = _golden_task_doc()
    doc["task_type"] = "CanaryPerfTaskData"
    doc["keyspace"] = 3_000_000
    doc["seed"] = 42
    doc["note"] = "canary default-throughput v1 (2026-08-30)"
    return doc


def _canary_envelope(task_id=CANARY_TASK_ID, runner_id="armbench"):
    """Full remote envelope as the control plane would build it."""
    task_doc = _canary_task_doc()
    return {
        "schema_version": 1,
        "task_id": task_id,
        "runner_id": runner_id,
        "task_class": "canary",
        "priority": 50,
        "submitted_at": "2026-08-30T12:00:00Z",
        "submitted_by": "canary-scheduler",
        "canary_id": "default-throughput:2026-08-30",
        "task": task_doc,
    }


def _canary_claim_document(task_id=CANARY_TASK_ID, runner_id="armbench"):
    envelope = _canary_envelope(task_id, runner_id)
    return {
        "schema_version": 1,
        "claim": {
            "task": {
                "task_id": task_id,
                "runner_id": runner_id,
                "envelope": envelope,
            },
            "claim_token": "canary-claim-token",
            "lease_expires": "2026-08-30T12:05:00Z",
        },
    }


# ---------------------------------------------------------------------------
# BaseTaskData override tests
# ---------------------------------------------------------------------------


class TestOverrideTaskId:
    def test_no_override_uses_timestamp_derived_id(self):
        task = BaseTaskData.from_dict(_golden_task_doc())
        assert task.task_id == datetime_to_task_id(task.timestamp)
        assert task._override_task_id is None

    def test_envelope_task_id_overrides_timestamp(self):
        task = BaseTaskData.from_dict(_golden_task_doc(), envelope_task_id=CANARY_TASK_ID)
        assert task.task_id == CANARY_TASK_ID
        assert task._override_task_id == CANARY_TASK_ID

    def test_override_survives_save_and_reload(self, tmp_path):
        task = BaseTaskData.from_dict(_golden_task_doc(), envelope_task_id=CANARY_TASK_ID)
        filepath = tmp_path / "task_test.json"
        task.save_to_file(filepath)

        # Verify persisted JSON has the sidecar key
        persisted = json.loads(filepath.read_text(encoding="utf-8"))
        assert persisted["__envelope_task_id"] == CANARY_TASK_ID
        assert "_override_task_id" not in persisted

        # Reload
        reloaded = BaseTaskData.from_file(filepath)
        assert reloaded.task_id == CANARY_TASK_ID
        assert reloaded._override_task_id == CANARY_TASK_ID

    def test_no_override_does_not_persist_sidecar(self, tmp_path):
        task = BaseTaskData.from_dict(_golden_task_doc())
        filepath = tmp_path / "task_test.json"
        task.save_to_file(filepath)
        persisted = json.loads(filepath.read_text(encoding="utf-8"))
        assert "__envelope_task_id" not in persisted
        assert "_override_task_id" not in persisted

    def test_explicit_envelope_wins_over_persisted_sidecar(self):
        """Defence-in-depth: if both an explicit kwarg and sidecar exist,
        the explicit kwarg from the trusted caller wins."""
        doc = _golden_task_doc()
        doc["__envelope_task_id"] = "persisted:old:id"
        task = BaseTaskData.from_dict(doc, envelope_task_id=CANARY_TASK_ID)
        assert task.task_id == CANARY_TASK_ID

    def test_untrusted_document_cannot_inject_override_without_caller(self):
        """An untrusted document with __envelope_task_id is honoured only
        because it was written by our own save_to_file.  There's no way
        for a remote attacker to supply it via the envelope because
        RunnerMailbox always passes the explicit envelope_task_id kwarg."""
        doc = _golden_task_doc()
        doc["__envelope_task_id"] = "injected:attack"
        task = BaseTaskData.from_dict(doc)
        # Persisted sidecar is honoured for restart recovery
        assert task.task_id == "injected:attack"

    def test_serialize_task_strips_override(self):
        """The wire-format task body never contains internal fields."""
        task = BaseTaskData.from_dict(_golden_task_doc(), envelope_task_id=CANARY_TASK_ID)
        serialized = serialize_task(task)
        assert "_override_task_id" not in serialized
        assert "__envelope_task_id" not in serialized

    def test_build_envelope_uses_override_id(self):
        """build_task_envelope uses task.task_id which should return the override."""
        task = BaseTaskData.from_dict(_golden_task_doc(), envelope_task_id="custom:task:id")
        env = build_task_envelope(task, runner_id="g4bench")
        assert env["task_id"] == "custom:task:id"
        assert "_override_task_id" not in env["task"]


# ---------------------------------------------------------------------------
# TaskQueue tests
# ---------------------------------------------------------------------------


class TestTaskQueueOverride:
    def test_import_with_envelope_id_creates_correct_filename(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        task = queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        assert task.task_id == CANARY_TASK_ID
        expected_file = tmp_path / "queue" / f"task_{CANARY_TASK_ID}.json"
        assert expected_file.exists()

    def test_import_persists_sidecar_and_reloads(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)

        # Reload from queue
        all_tasks = queue.get_all_tasks()
        assert len(all_tasks) == 1
        assert all_tasks[0].task_id == CANARY_TASK_ID

    def test_import_idempotent_with_override(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        t1 = queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        t2 = queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        assert t1.task_id == t2.task_id

    def test_import_rejects_conflicting_content_for_same_id(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        changed = {**doc, "note": "different canary"}
        with pytest.raises(ValueError, match="different content"):
            queue.import_task(changed, envelope_task_id=CANARY_TASK_ID)

    def test_has_task_and_remove_with_override_id(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        assert queue.has_task(CANARY_TASK_ID)
        assert queue.remove_task(CANARY_TASK_ID)
        assert not queue.has_task(CANARY_TASK_ID)

    def test_finish_task_with_override_id(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        task = queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        queue.finish_task(task)
        assert not queue.has_task(CANARY_TASK_ID)

    def test_local_task_without_override_unchanged(self, tmp_path):
        """Ordinary locally-created tasks retain timestamp-derived IDs."""
        queue = TaskQueue(tmp_path / "queue")
        doc = _golden_task_doc()
        task = queue.import_task(doc)
        expected_id = datetime_to_task_id(task.timestamp)
        assert task.task_id == expected_id
        assert queue.has_task(expected_id)
        # No sidecar persisted
        persisted = json.loads(queue.task_path(expected_id).read_text())
        assert "__envelope_task_id" not in persisted

    def test_get_next_task_returns_override_id(self, tmp_path):
        queue = TaskQueue(tmp_path / "queue")
        doc = _canary_task_doc()
        queue.import_task(doc, envelope_task_id=CANARY_TASK_ID)
        next_task = queue.get_next_task()
        assert next_task is not None
        assert next_task.task_id == CANARY_TASK_ID


# ---------------------------------------------------------------------------
# FileProtocol tests
# ---------------------------------------------------------------------------


class TestFileProtocolOverride:
    def test_file_protocol_uses_override_id(self, tmp_path):
        task = BaseTaskData.from_dict(_canary_task_doc(), envelope_task_id=CANARY_TASK_ID)
        fp = FileProtocol(task.task_id, role_id="client", base_dir=tmp_path)
        assert fp.task_id == CANARY_TASK_ID
        assert fp.work_dir == tmp_path / f"benchmark_{CANARY_TASK_ID}"


# ---------------------------------------------------------------------------
# RunnerMailbox full lifecycle tests
# ---------------------------------------------------------------------------


class FakeRunnerClient:
    def __init__(self, claim_doc=None, *, accept_failures=0, outcome_failures=0):
        self.claims = [claim_doc or _canary_claim_document()]
        self.accept_failures = accept_failures
        self.outcome_failures = outcome_failures
        self.accepted = []
        self.outcomes = []
        self.statuses = []

    def claim_task(self, runner_id):
        return self.claims.pop(0) if self.claims else None

    def accept_task(self, task_id, claim_token):
        if self.accept_failures:
            self.accept_failures -= 1
            raise FleetClientError("CONTROL_UNREACHABLE", "offline", 3)
        self.accepted.append((task_id, claim_token))
        return {"schema_version": 1, "task": {"task_id": task_id}, "changed": True}

    def report_outcome(self, task_id, outcome):
        if self.outcome_failures:
            self.outcome_failures -= 1
            raise FleetClientError("CONTROL_UNREACHABLE", "offline", 3)
        self.outcomes.append((task_id, outcome))
        return {"schema_version": 1, "task": {"task_id": task_id}, "changed": True}

    def push_status(self, runner_id, status):
        self.statuses.append((runner_id, status))
        return {"schema_version": 1, "updated": True}


@pytest.fixture()
def mailbox_env(monkeypatch, tmp_path):
    monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])
    monkeypatch.setattr(
        "conductress.runner_mailbox.get_runner_config",
        lambda: SimpleNamespace(runner_id="armbench"),
    )
    monkeypatch.setattr("conductress.runner_mailbox.CONDUCTRESS_OUTPUT", tmp_path / "output.jsonl")
    monkeypatch.setattr("conductress.runner_mailbox.CONDUCTRESS_FAILED_LOG", tmp_path / "failed.jsonl")
    return tmp_path


def _make_mailbox(tmp_path, client, mode="live"):
    return RunnerMailbox(
        tmp_path / "delivery.json",
        mode=mode,
        client_factory=lambda: client,
    )


class TestCanaryLifecycle:
    """Full lifecycle: scheduler envelope → poll → import → accept → execute → outcome."""

    def test_canary_full_lifecycle(self, mailbox_env):
        """End-to-end: canary task_id preserved through claim/import/accept/
        execution identity/stage_success/outcome reporting."""
        tmp_path = mailbox_env
        client = FakeRunnerClient()
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        # Poll claims the canary task
        task = mailbox.poll(queue)
        assert task is not None
        assert task.task_id == CANARY_TASK_ID

        # Queue file uses the canary ID
        assert queue.has_task(CANARY_TASK_ID)
        queue_file = queue.task_path(CANARY_TASK_ID)
        assert queue_file.exists()

        # Journal active uses the canary ID
        assert mailbox.journal.active["task_id"] == CANARY_TASK_ID
        assert mailbox.journal.active["stage"] == "accepted"

        # owns() works with the canary ID
        assert mailbox.owns(CANARY_TASK_ID)

        # FileProtocol would use the override ID
        fp = FileProtocol(task.task_id, role_id="client")
        assert fp.task_id == CANARY_TASK_ID

        # Accept was called with the canary ID
        assert client.accepted == [(CANARY_TASK_ID, "canary-claim-token")]

        # Complete with success
        result = {
            "task_id": CANARY_TASK_ID,
            "method": "perf-get",
            "score": 200000,
            "commit_hash": "fcd8bc3",
        }
        mailbox.stage_success(task, result=result)
        queue.finish_task(task)
        assert mailbox.flush_pending_outcome()

        # Outcome reported with canary ID
        assert len(client.outcomes) == 1
        reported_id, reported_outcome = client.outcomes[0]
        assert reported_id == CANARY_TASK_ID
        assert reported_outcome["task_id"] == CANARY_TASK_ID
        assert reported_outcome["state"] == "completed"

    def test_canary_reload_after_restart_preserves_id(self, mailbox_env):
        """Simulate runner restart: journal + queue file survive, task_id preserved."""
        tmp_path = mailbox_env
        client = FakeRunnerClient(accept_failures=1)
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        # First poll: claim + import but accept fails
        assert mailbox.poll(queue) is None
        assert mailbox.blocks_execution()
        assert queue.has_task(CANARY_TASK_ID)

        # Simulate restart
        client2 = FakeRunnerClient()
        client2.claims = []  # No new claims on restart
        mailbox2 = _make_mailbox(tmp_path, client2)
        queue2 = TaskQueue(tmp_path / "queue")

        # Reconcile recovers the task with the canary ID
        task = mailbox2.poll(queue2)
        assert task is not None
        assert task.task_id == CANARY_TASK_ID
        assert queue2.has_task(CANARY_TASK_ID)

    def test_canary_expired_claim_clears_local_state(self, mailbox_env):
        """Stale claim with expired token: local queue file and journal cleared,
        subsequent poll can reclaim."""
        tmp_path = mailbox_env

        def reject_accept(_task_id, _claim_token):
            raise FleetClientError("CLAIM_EXPIRED", "expired", 4)

        client = FakeRunnerClient()
        client.accept_task = reject_accept
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        # Poll claims but accept is rejected
        assert mailbox.poll(queue) is None
        assert mailbox.journal.active is None
        assert queue.get_queue_length() == 0
        assert not queue.has_task(CANARY_TASK_ID)

        # Subsequent poll with fresh claim succeeds
        client2 = FakeRunnerClient()
        mailbox2 = _make_mailbox(tmp_path, client2)
        task = mailbox2.poll(queue)
        assert task is not None
        assert task.task_id == CANARY_TASK_ID

    def test_canary_failure_reports_with_canary_id(self, mailbox_env):
        tmp_path = mailbox_env
        client = FakeRunnerClient()
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        task = mailbox.poll(queue)
        assert task is not None
        mailbox.stage_failure(task, "build failed")
        queue.finish_task(task)
        assert mailbox.flush_pending_outcome()

        reported_id, reported_outcome = client.outcomes[0]
        assert reported_id == CANARY_TASK_ID
        assert reported_outcome["task_id"] == CANARY_TASK_ID
        assert reported_outcome["state"] == "failed"

    def test_existing_result_detected_with_canary_id(self, mailbox_env):
        """If a result with the canary task_id exists in output.jsonl,
        reconcile detects it and does not re-execute."""
        tmp_path = mailbox_env
        client = FakeRunnerClient()
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        task = mailbox.poll(queue)
        assert task is not None

        # Write a result with the canary ID
        output = tmp_path / "output.jsonl"
        output.write_text(json.dumps({"task_id": CANARY_TASK_ID, "method": "perf-get", "score": 200000}) + "\n")

        # Restart: reconcile finds result, stages success, clears
        client2 = FakeRunnerClient()
        client2.claims = []
        mailbox2 = _make_mailbox(tmp_path, client2)
        queue2 = TaskQueue(tmp_path / "queue")
        assert mailbox2.reconcile(queue2) is None
        assert not queue2.has_task(CANARY_TASK_ID)
        assert mailbox2.journal.active is None


class TestLocalTaskUnchanged:
    """Verify that ordinary local tasks (no envelope override) behave identically
    to the pre-change baseline."""

    def test_local_task_uses_timestamp_id(self, mailbox_env):
        tmp_path = mailbox_env
        local_doc = _golden_task_doc()
        task_id = datetime_to_task_id(datetime.fromisoformat(local_doc["timestamp"]))

        # Manual envelope with timestamp-derived ID (as build_task_envelope produces)
        claim = {
            "schema_version": 1,
            "claim": {
                "task": {
                    "task_id": task_id,
                    "runner_id": "armbench",
                    "envelope": {
                        "schema_version": 1,
                        "task_id": task_id,
                        "runner_id": "armbench",
                        "task_class": "manual",
                        "priority": 100,
                        "submitted_at": "2026-08-29T00:00:00Z",
                        "submitted_by": "rain",
                        "canary_id": None,
                        "task": local_doc,
                    },
                },
                "claim_token": "manual-token",
                "lease_expires": "2026-08-29T00:05:00Z",
            },
        }

        client = FakeRunnerClient(claim_doc=claim)
        mailbox = _make_mailbox(tmp_path, client)
        queue = TaskQueue(tmp_path / "queue")

        task = mailbox.poll(queue)
        assert task is not None
        assert task.task_id == task_id
        assert task._override_task_id == task_id  # envelope_task_id matches timestamp
        assert queue.has_task(task_id)

    def test_local_submit_and_finish(self, mailbox_env):
        tmp_path = mailbox_env
        queue = TaskQueue(tmp_path / "queue")
        doc = _golden_task_doc()
        task = BaseTaskData.from_dict(doc)
        queue.submit_task(task)
        assert queue.has_task(task.task_id)
        assert task._override_task_id is None
        queue.finish_task(task)
        assert not queue.has_task(task.task_id)


class TestQueueFileConsistency:
    """Ensure queue files with and without overrides are correctly sorted
    and retrieved."""

    def test_mixed_local_and_remote_tasks_coexist(self, mailbox_env):
        tmp_path = mailbox_env
        queue = TaskQueue(tmp_path / "queue")

        # Import a canary task
        canary_doc = _canary_task_doc()
        queue.import_task(canary_doc, envelope_task_id=CANARY_TASK_ID)

        # Submit a local task
        local_doc = _golden_task_doc()
        local_task = BaseTaskData.from_dict(local_doc)
        queue.submit_task(local_task)

        all_tasks = queue.get_all_tasks()
        assert len(all_tasks) == 2
        ids = {t.task_id for t in all_tasks}
        assert CANARY_TASK_ID in ids
        assert local_task.task_id in ids
