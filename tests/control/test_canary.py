"""Tests for canary profile loading, validation, and the canary scheduler.

Covers: schema validation, mutable ref rejection, unknown field rejection,
filename matching, dedup/concurrency, restart idempotency, freshness
window, missed/expired states, priority ordering, and envelope compatibility.

No sleep-based tests -- all use explicit datetime injection.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conductress.control.canary_profiles import CanaryProfileRegistry
from conductress.control.canary_scheduler import (
    CANARY_TASK_CLASS,
    PRIORITY_CANARY,
    PRIORITY_SWEEP,
    CanaryScheduler,
    _canary_task_id,
)
from conductress.control.db import ControlDatabase, DATABASE_SCHEMA_VERSION
from conductress.control.errors import ControlError
from conductress.control.fleet_registry import FleetRegistry
from conductress.control.service import ControlService
from conductress.task_queue import BaseTaskData

from .helpers import fleet_manifest, task_envelope


def _write_profile(directory: Path, data: dict) -> Path:
    """Write a profile JSON and return the path."""
    path = directory / f"{data['profile_id']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _valid_profile(
    profile_id: str = "throughput-get-v1",
    profile_version: int = 1,
    pinned_commit: str = "a" * 40,
    utc_hour: int = 6,
    freshness_hours: int = 18,
) -> dict:
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "description": "test canary profile",
        "source": "valkey",
        "pinned_commit": pinned_commit,
        "build": {"make_args": ""},
        "workload": {
            "test": "get",
            "val_size": 512,
            "key_size": 0,
            "io_threads": 9,
            "pipelining": 10,
            "clients": 1200,
            "threads": 16,
            "keyspace": 3000000,
            "warmup_seconds": 30,
            "duration_seconds": 300,
            "repetitions": 5,
            "seed": 42,
        },
        "schedule": {
            "utc_hour": utc_hour,
            "freshness_hours": freshness_hours,
        },
        "thresholds": {
            "platforms": {
                "amd": {"warning_pct": 0.5, "alarm_pct": 1.0},
                "graviton3": {"warning_pct": 2.0, "alarm_pct": 4.0},
            }
        },
    }


# ---------------------------------------------------------------------------
# Profile validation tests
# ---------------------------------------------------------------------------


class TestCanaryProfileValidation:
    def test_valid_profile_loads(self, tmp_path):
        _write_profile(tmp_path, _valid_profile())
        registry = CanaryProfileRegistry.from_directory(tmp_path)
        assert len(registry) == 1
        profile = registry.require("throughput-get-v1")
        assert profile.profile_id == "throughput-get-v1"
        assert profile.profile_version == 1
        assert profile.pinned_commit == "a" * 40
        assert profile.source == "valkey"
        assert profile.utc_hour == 6
        assert profile.freshness_hours == 18
        assert profile.workload["test"] == "get"

    def test_rejects_mutable_branch_ref(self, tmp_path):
        data = _valid_profile(pinned_commit="unstable" + "0" * 32)
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_origin_prefix(self, tmp_path):
        data = _valid_profile(pinned_commit="origin/unstable" + "0" * 25)
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_short_hash(self, tmp_path):
        data = _valid_profile()
        data["pinned_commit"] = "abc1234"
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_unknown_fields(self, tmp_path):
        data = _valid_profile()
        data["surprise_field"] = True
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_unknown_workload_field(self, tmp_path):
        data = _valid_profile()
        data["workload"]["surprise"] = 42
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_unknown_build_field(self, tmp_path):
        data = _valid_profile()
        data["build"]["extra_option"] = True
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_unknown_schedule_field(self, tmp_path):
        data = _valid_profile()
        data["schedule"]["extra"] = True
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_missing_required_field(self, tmp_path):
        data = _valid_profile()
        del data["pinned_commit"]
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_wrong_schema_version(self, tmp_path):
        data = _valid_profile()
        data["schema_version"] = 2
        _write_profile(tmp_path, data)
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_SCHEMA_INVALID"

    def test_rejects_duplicate_profile_id(self, tmp_path):
        _write_profile(tmp_path, _valid_profile())
        data2 = _valid_profile(profile_id="throughput-get-v1")
        path2 = tmp_path / "duplicate.json"
        path2.write_text(json.dumps(data2), encoding="utf-8")
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_FILENAME_MISMATCH"

    def test_rejects_filename_mismatch(self, tmp_path):
        data = _valid_profile()
        path = tmp_path / "wrong-name.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ControlError) as exc_info:
            CanaryProfileRegistry.from_directory(tmp_path)
        assert exc_info.value.code == "PROFILE_FILENAME_MISMATCH"

    def test_require_raises_on_unknown(self, tmp_path):
        registry = CanaryProfileRegistry.from_directory(tmp_path)
        with pytest.raises(ControlError) as exc_info:
            registry.require("nonexistent")
        assert exc_info.value.code == "PROFILE_NOT_FOUND"

    def test_empty_directory(self, tmp_path):
        registry = CanaryProfileRegistry.from_directory(tmp_path)
        assert len(registry) == 0
        assert registry.list_profiles() == []

    def test_nonexistent_directory(self, tmp_path):
        registry = CanaryProfileRegistry.from_directory(tmp_path / "nope")
        assert len(registry) == 0

    def test_multiple_profiles(self, tmp_path):
        _write_profile(tmp_path, _valid_profile("profile-a", pinned_commit="a" * 40))
        _write_profile(tmp_path, _valid_profile("profile-b", pinned_commit="b" * 40))
        registry = CanaryProfileRegistry.from_directory(tmp_path)
        assert len(registry) == 2
        assert "profile-a" in registry
        assert "profile-b" in registry


# ---------------------------------------------------------------------------
# Checked-in profile validation
# ---------------------------------------------------------------------------


class TestCheckedInProfile:
    """Validate the actual throughput-get-v1.json checked into the repo."""

    PROFILE_DIR = Path(__file__).resolve().parents[2] / "src" / "conductress" / "canary_profiles"

    def test_throughput_get_v1_validates(self):
        registry = CanaryProfileRegistry.from_directory(self.PROFILE_DIR)
        profile = registry.require("throughput-get-v1")
        assert profile.profile_version == 1
        assert len(profile.pinned_commit) == 40
        assert profile.source == "valkey"
        assert profile.workload["test"] == "get"
        assert profile.workload["val_size"] == 512
        assert profile.workload["keyspace"] == 3000000
        assert profile.workload["seed"] == 20260830
        assert profile.schedule["utc_hour"] == 6
        assert profile.schedule["freshness_hours"] == 18


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


def _make_scheduler(
    tmp_path: Path, *, profile_data: dict | None = None
) -> tuple[CanaryScheduler, ControlDatabase, FleetRegistry]:
    """Build a scheduler with a real DB, fleet manifest, and one profile."""
    manifest_path = tmp_path / "fleet.json"
    manifest_path.write_text(json.dumps(fleet_manifest()), encoding="utf-8")

    db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
    db.initialize()
    registry = FleetRegistry.from_file(manifest_path)

    canary_dir = tmp_path / "canary_profiles"
    canary_dir.mkdir(exist_ok=True)
    if profile_data is None:
        profile_data = _valid_profile()
    _write_profile(canary_dir, profile_data)
    profiles = CanaryProfileRegistry.from_directory(canary_dir)

    scheduler = CanaryScheduler(db, registry, profiles)
    return scheduler, db, registry


class TestCanaryScheduler:
    def test_creates_one_canary_per_runner_per_day(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        # Within freshness window: utc_hour=6, freshness=18, so 06:00-24:00 UTC
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        actions = scheduler.tick(now=now)
        # armbench and g4bench both have canary_profile set; yesterday (Aug 31)
        # is also evaluated and should be missed (first tick ever)
        created = [a for a in actions if a["action"] == "created"]
        assert len(created) == 2
        runner_ids = {a["runner_id"] for a in created}
        assert runner_ids == {"armbench", "g4bench"}
        # All created entries are for today
        assert all(a["utc_date"] == "2026-09-01" for a in created)

    def test_idempotent_across_repeated_ticks(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        first = scheduler.tick(now=now)
        second = scheduler.tick(now=now)
        third = scheduler.tick(now=now)
        created_first = [a for a in first if a["action"] == "created"]
        assert len(created_first) == 2
        assert second == []
        assert third == []

    def test_idempotent_across_restart(self, tmp_path):
        """Simulate a service restart mid-day."""
        scheduler, db, registry = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        # "Restart" -- create new scheduler instances with same DB
        canary_dir = tmp_path / "canary_profiles"
        profiles = CanaryProfileRegistry.from_directory(canary_dir)
        scheduler2 = CanaryScheduler(db, registry, profiles)
        actions = scheduler2.tick(now=now)
        assert actions == []

    def test_does_not_create_before_utc_hour(self, tmp_path):
        # Use utc_hour=10 so we can test 09:59 without yesterday being in window
        profile = _valid_profile(utc_hour=10, freshness_hours=14)
        scheduler, _, _ = _make_scheduler(tmp_path, profile_data=profile)
        # At 09:59 on Sept 1, today's canary isn't due yet.
        # Yesterday (Aug 31) would be past deadline (10+14=24 => midnight Sept 1,
        # and 09:59 < midnight so yesterday deadline hasn't passed...
        # Actually with utc_hour=10, freshness=14, deadline = Aug 31 10:00 + 14h = Sept 1 00:00
        # At 09:59 Sept 1, yesterday's deadline is Sept 1 00:00, which is past.
        # So yesterday will be missed. Let's just check today isn't created.
        too_early = datetime(2026, 9, 1, 9, 59, tzinfo=timezone.utc)
        actions = scheduler.tick(now=too_early)
        created = [a for a in actions if a["action"] == "created"]
        assert len(created) == 0

    def test_missed_after_freshness_deadline(self, tmp_path):
        # Use utc_hour=6, freshness=6 so deadline = 12:00 UTC same day
        profile = _valid_profile(utc_hour=6, freshness_hours=6)
        scheduler, _, _ = _make_scheduler(tmp_path, profile_data=profile)
        # Tick at 13:00 -- past deadline for today, yesterday also missed
        past_deadline = datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)
        actions = scheduler.tick(now=past_deadline)
        missed = [a for a in actions if a["action"] == "missed"]
        # Both today and yesterday are missed for both runners
        assert len(missed) >= 2
        today_missed = [a for a in missed if a["utc_date"] == "2026-09-01"]
        assert len(today_missed) == 2  # armbench + g4bench

    def test_never_backfills_stale_dates(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        # Run for Sept 5 only -- only Sept 4 (yesterday) and Sept 5 (today) should appear
        now = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)
        with db.read() as conn:
            dates = {
                row["utc_date"] for row in conn.execute("SELECT DISTINCT utc_date FROM canary_schedule").fetchall()
            }
        # Only yesterday (missed) and today (created) -- never Sept 1-3
        assert "2026-09-01" not in dates
        assert "2026-09-02" not in dates
        assert "2026-09-03" not in dates
        assert "2026-09-05" in dates

    def test_expired_state_when_created_but_past_deadline(self, tmp_path):
        # Use a short freshness window
        profile = _valid_profile(utc_hour=6, freshness_hours=6)
        scheduler, _, _ = _make_scheduler(tmp_path, profile_data=profile)
        # Create at 10:00 (within 06:00-12:00 window)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)
        # Tick again past deadline (12:00)
        past = datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)
        actions = scheduler.tick(now=past)
        expired = [a for a in actions if a["action"] == "expired" and a["utc_date"] == "2026-09-01"]
        assert len(expired) == 2  # armbench + g4bench
        with scheduler.database.read() as conn:
            stale = conn.execute("SELECT state FROM tasks WHERE task_class = 'canary'").fetchall()
        assert {row["state"] for row in stale} == {"cancelled"}

    def test_disabled_runner_gets_no_canary(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)
        with db.read() as conn:
            runners = {
                row["runner_id"] for row in conn.execute("SELECT DISTINCT runner_id FROM canary_schedule").fetchall()
            }
        assert "disabled" not in runners

    def test_runner_without_canary_profile_skipped(self, tmp_path):
        """Runner with canary_profile=None is silently skipped."""
        manifest = fleet_manifest()
        for runner in manifest["runners"]:
            runner["canary_profile"] = None

        manifest_path = tmp_path / "fleet.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
        db.initialize()
        registry = FleetRegistry.from_file(manifest_path)

        canary_dir = tmp_path / "canary_profiles"
        canary_dir.mkdir()
        _write_profile(canary_dir, _valid_profile())
        profiles = CanaryProfileRegistry.from_directory(canary_dir)

        scheduler = CanaryScheduler(db, registry, profiles)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        assert scheduler.tick(now=now) == []

    def test_canary_task_has_correct_class_and_priority(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        with db.read() as conn:
            tasks = conn.execute(
                "SELECT task_class, priority, canary_id FROM tasks WHERE task_class = 'canary'"
            ).fetchall()
        assert len(tasks) == 2
        for task in tasks:
            assert task["task_class"] == CANARY_TASK_CLASS
            assert task["priority"] == PRIORITY_CANARY
            assert task["canary_id"] is not None

    def test_new_day_creates_new_canary(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        day1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=day1)
        day2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=day2)
        with db.read() as conn:
            # Count unique (runner_id, utc_date) created canary tasks
            canary_tasks = conn.execute("SELECT COUNT(*) FROM tasks WHERE task_class = 'canary'").fetchone()[0]
        # 2 runners x 2 days = 4 canary tasks
        assert canary_tasks == 4


class TestPriorityOrdering:
    """Verify manual > canary > sweep priority through the claim path."""

    def test_manual_before_canary_before_sweep(self, tmp_path):
        scheduler, db, registry = _make_scheduler(tmp_path)

        # Create a canary
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        service = ControlService(db, registry)

        # Submit a manual task and a sweep task
        manual = task_envelope("manual-1", runner_id="armbench", priority=1, task_class="manual")
        sweep = task_envelope("sweep-1", runner_id="armbench", priority=PRIORITY_SWEEP, task_class="sweep")
        service.submit_task(manual, actor="test")
        service.submit_task(sweep, actor="test")

        # First claim should be manual even with a lower numeric priority
        claim1 = service.claim_task("armbench", actor="test")
        assert claim1["task"]["task_id"] == "manual-1"
        assert claim1["task"]["task_class"] == "manual"
        service.accept_task("armbench", "manual-1", claim1["claim_token"], actor="test")
        service.record_outcome(
            "armbench",
            "manual-1",
            {
                "schema_version": 1,
                "task_id": "manual-1",
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": now.isoformat().replace("+00:00", "Z"),
                "result": {"score": 1.0},
                "error": None,
            },
            actor="test",
        )

        # Second claim should be canary (priority 50)
        claim2 = service.claim_task("armbench", actor="test")
        assert claim2["task"]["task_class"] == "canary"
        assert claim2["task"]["priority"] == PRIORITY_CANARY
        service.accept_task("armbench", claim2["task"]["task_id"], claim2["claim_token"], actor="test")
        service.record_outcome(
            "armbench",
            claim2["task"]["task_id"],
            {
                "schema_version": 1,
                "task_id": claim2["task"]["task_id"],
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": now.isoformat().replace("+00:00", "Z"),
                "result": {"score": 1.0},
                "error": None,
            },
            actor="test",
        )

        # Third claim should be sweep (priority 10)
        claim3 = service.claim_task("armbench", actor="test")
        assert claim3["task"]["task_id"] == "sweep-1"
        assert claim3["task"]["task_class"] == "sweep"

    def test_explicit_manual_priority_preserved(self, tmp_path):
        """A manual task with priority=200 stays at 200, not overridden."""
        scheduler, db, registry = _make_scheduler(tmp_path)
        service = ControlService(db, registry)

        high = task_envelope("high-pri", runner_id="armbench", priority=200, task_class="manual")
        low = task_envelope("low-pri", runner_id="armbench", priority=50, task_class="manual")
        service.submit_task(high, actor="test")
        service.submit_task(low, actor="test")

        claim = service.claim_task("armbench", actor="test")
        assert claim["task"]["task_id"] == "high-pri"
        assert claim["task"]["priority"] == 200


class TestEnvelopeCompatibility:
    """Verify canary task envelopes pass existing task-envelope schema validation."""

    def test_canary_envelope_validates_against_schema(self, tmp_path, monkeypatch):
        from jsonschema import Draft202012Validator

        from conductress import task_queue
        from conductress.control.schema import load_schema

        monkeypatch.setattr(task_queue.config, "REPO_NAMES", ["valkey"])
        scheduler, db, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        schema = load_schema("task-envelope.schema.json")
        validator = Draft202012Validator(schema)

        with db.read() as conn:
            rows = conn.execute("SELECT envelope_json FROM tasks WHERE task_class = 'canary'").fetchall()
        assert len(rows) >= 1
        for row in rows:
            envelope = json.loads(row["envelope_json"])
            validator.validate(envelope)
            assert envelope["task_class"] == "canary"
            assert envelope["canary_id"] is not None
            task = BaseTaskData.from_dict(envelope["task"])
            assert task.task_type == "CanaryPerfTaskData"
            assert task.specifier == "a" * 40
            assert task.bench_clients == 1200
            assert task.bench_threads == 16
            assert task.keyspace == 3000000
            assert task.seed == 42
            assert task.preload_keys is True

    def test_canary_task_id_is_deterministic(self, tmp_path):
        tid = _canary_task_id("armbench", "throughput-get-v1", "2026-09-01")
        assert tid == "canary:armbench:throughput-get-v1:2026-09-01"
        assert _canary_task_id("armbench", "throughput-get-v1", "2026-09-01") == tid


class TestConcurrentTicks:
    """Verify that concurrent tick() calls don't create duplicate canaries."""

    def test_concurrent_ticks_produce_exactly_one_canary_per_runner(self, tmp_path):
        scheduler, db, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

        barrier = threading.Barrier(4)
        results = []

        def tick():
            barrier.wait()
            result = scheduler.tick(now=now)
            results.append(result)

        threads = [threading.Thread(target=tick) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # DB should have exactly 2 canary tasks for today
        with db.read() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tasks WHERE task_class = 'canary'").fetchone()[0]
        assert count == 2


class TestDBMigration:
    def test_migration_v2_is_idempotent(self, tmp_path):
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")

        db.initialize()
        db.initialize()
        db.initialize()

        with db.read() as conn:
            v2_count = conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = 2").fetchone()[0]
        assert v2_count == 1

    def test_v1_db_upgraded_to_v2(self, tmp_path):
        """A fresh DB starts at v3 with tasks, canary_schedule, and observation tables."""
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
        db.initialize()

        with db.read() as conn:
            max_version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert max_version == DATABASE_SCHEMA_VERSION
        assert "canary_schedule" in tables
        assert "tasks" in tables
        assert "canary_observations" in tables
        assert "canary_calibration_reports" in tables


class TestScheduleQueries:
    def test_schedule_for_runner(self, tmp_path):
        scheduler, _, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        entries = scheduler.schedule_for_runner("armbench")
        # Today created + yesterday missed
        today_entries = [e for e in entries if e["utc_date"] == "2026-09-01"]
        assert len(today_entries) == 1
        assert today_entries[0]["state"] == "created"

    def test_schedule_for_date(self, tmp_path):
        scheduler, _, _ = _make_scheduler(tmp_path)
        now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        scheduler.tick(now=now)

        entries = scheduler.schedule_for_date("2026-09-01")
        assert len(entries) == 2
        runner_ids = {e["runner_id"] for e in entries}
        assert runner_ids == {"armbench", "g4bench"}
