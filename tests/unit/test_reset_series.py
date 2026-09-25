"""Unit tests for ``conductress sweep reset-series`` core logic.

The CLI wrapper is a thin argparse shim; the behaviour lives in
``conductress.sweep.reset_series.reset_series``, exercised here with ``tmp_path``
stand-ins for the state directory and the task queue so no runner is involved.
"""

import json
from pathlib import Path

import pytest

from conductress.sweep.reset_series import (
    ResetSeriesResult,
    default_state_dir,
    is_memory_workload,
    memory_series_workloads,
    reset_all_memory_series,
    reset_series,
    series_label,
    service_is_active,
    state_file_for,
)


def _write_state(state_dir: Path, workload: str, engine=None) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_file_for(workload, engine, state_dir=state_dir)
    path.write_text(json.dumps({"points": {"a": {"value": 1}}, "merge_commits": ["a"]}))
    return path


def _write_task(queue_dir: Path, name: str, note: str, source: str = "valkey", task_type="CachecannonTaskData") -> Path:
    queue_dir.mkdir(parents=True, exist_ok=True)
    path = queue_dir / f"task_{name}.json"
    path.write_text(json.dumps({"note": note, "source": source, "task_type": task_type}))
    return path


def _memory_note(shape: str) -> str:
    """The memory coordinator's note names the shape only; the engine is in ``source``."""
    return f"[memory-sweep:{shape}] Backfill: 12 commits in gap"


def _note(workload: str, engine=None, source="valkey") -> str:
    return f"[cachecannon-sweep-v3:{source}/{series_label(workload, engine)}] backfill"


class TestSeriesLabelAndPath:
    def test_valkey_label_is_unprefixed(self):
        assert series_label("get-k16-v16-t7-p1", None) == "get-k16-v16-t7-p1"

    def test_engine_label_is_prefixed(self):
        assert series_label("get-k16-v16-t7-p1", "redis") == "redis-get-k16-v16-t7-p1"

    def test_state_file_name(self, tmp_path):
        path = state_file_for("get-k16-v16-t7-p1", None, state_dir=tmp_path)
        assert path.name == "state_cachecannon-v3_get-k16-v16-t7-p1.json"

    def test_engine_state_file_name(self, tmp_path):
        path = state_file_for("get-k16-v16-t7-p1", "redis", state_dir=tmp_path)
        assert path.name == "state_cachecannon-v3_redis-get-k16-v16-t7-p1.json"

    def test_tls_series_state_file_name(self, tmp_path):
        """reset-series must resolve the -tls label's state file, same as any other."""
        path = state_file_for("get-k16-v16-t7-p10-tls", None, state_dir=tmp_path)
        assert path.name == "state_cachecannon-v3_get-k16-v16-t7-p10-tls.json"

    def test_tls_series_note_prefix_matches_only_its_own_cells(self):
        """The TLS series' queued-task note prefix is distinct from the plaintext GET series."""
        assert _note("get-k16-v16-t7-p10-tls").startswith("[cachecannon-sweep-v3:valkey/get-k16-v16-t7-p10-tls]")
        assert not _note("get-k16-v16-t7-p10").startswith("[cachecannon-sweep-v3:valkey/get-k16-v16-t7-p10-tls]")


class TestStateBackupAndDelete:
    def test_state_file_backed_up_then_deleted(self, tmp_path):
        state_dir = tmp_path / "v3"
        queue_dir = tmp_path / "queue"
        state = _write_state(state_dir, "get-k16-v16-t7-p1")

        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=state_dir,
            queue_dir=queue_dir,
            stamp="20260921T170000Z",
            service_active=False,
        )

        assert result.state_existed is True
        assert not state.exists(), "original state file must be deleted"
        assert result.state_backup is not None
        assert result.state_backup.exists(), "backup must be written"
        assert result.state_backup.name == "state_cachecannon-v3_get-k16-v16-t7-p1.json.bak-20260921T170000Z"
        # The backup preserves the original contents.
        assert json.loads(result.state_backup.read_text())["merge_commits"] == ["a"]

    def test_missing_state_file_is_reported_not_fatal(self, tmp_path):
        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=tmp_path / "v3",
            queue_dir=tmp_path / "queue",
            service_active=False,
        )
        assert result.state_existed is False
        assert result.state_backup is None


class TestQueuedTaskRelocation:
    def test_matching_tasks_relocated_others_untouched(self, tmp_path):
        state_dir = tmp_path / "v3"
        queue_dir = tmp_path / "queue"
        _write_state(state_dir, "get-k16-v16-t7-p1")
        mine_a = _write_task(queue_dir, "0001", _note("get-k16-v16-t7-p1"))
        mine_b = _write_task(queue_dir, "0002", _note("get-k16-v16-t7-p1"))
        other_series = _write_task(queue_dir, "0003", _note("get-k16-v16-t7-p10"))
        other_manual = _write_task(queue_dir, "0004", "[manual] a diagnostic cell")

        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=state_dir,
            queue_dir=queue_dir,
            stamp="20260921T170000Z",
            service_active=False,
        )

        assert not mine_a.exists() and not mine_b.exists(), "series' queued cells must be moved out"
        assert other_series.exists(), "another series' cell must be untouched"
        assert other_manual.exists(), "a manual cell must be untouched"
        reset_dir = queue_dir / "reset-20260921T170000Z"
        assert (reset_dir / "task_0001.json").exists()
        assert (reset_dir / "task_0002.json").exists()
        assert len(result.relocated_tasks) == 2

    def test_prefix_match_is_exact_not_substring(self, tmp_path):
        """A longer workload label starting with the target must not be swept in."""
        state_dir = tmp_path / "v3"
        queue_dir = tmp_path / "queue"
        _write_state(state_dir, "get-k16-v16-t7-p1")
        # This note's series label starts with the target label but is a different series.
        sibling = _write_task(queue_dir, "0005", _note("get-k16-v16-t7-p1-r100k"))

        reset_series(
            "get-k16-v16-t7-p1",
            state_dir=state_dir,
            queue_dir=queue_dir,
            stamp="s",
            service_active=False,
        )
        # The prefix includes the closing bracket, so "...p1]" does not match "...p1-r100k]".
        assert sibling.exists()

    def test_engine_prefix_matches_only_engine_series(self, tmp_path):
        state_dir = tmp_path / "v3"
        queue_dir = tmp_path / "queue"
        _write_state(state_dir, "get-k16-v16-t7-p1", engine="redis")
        redis_cell = _write_task(queue_dir, "0006", _note("get-k16-v16-t7-p1", engine="redis", source="redis"))
        valkey_cell = _write_task(queue_dir, "0007", _note("get-k16-v16-t7-p1", source="valkey"))

        result = reset_series(
            "get-k16-v16-t7-p1",
            engine="redis",
            state_dir=state_dir,
            queue_dir=queue_dir,
            stamp="s",
            service_active=False,
        )

        assert not redis_cell.exists()
        assert valkey_cell.exists(), "the Valkey series' cell must be left alone"
        assert result.label == "redis-get-k16-v16-t7-p1"


class TestDryRun:
    def test_dry_run_touches_nothing(self, tmp_path):
        state_dir = tmp_path / "v3"
        queue_dir = tmp_path / "queue"
        state = _write_state(state_dir, "get-k16-v16-t7-p1")
        cell = _write_task(queue_dir, "0001", _note("get-k16-v16-t7-p1"))

        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=state_dir,
            queue_dir=queue_dir,
            dry_run=True,
            stamp="s",
            service_active=False,
        )

        assert state.exists(), "dry-run must not delete the state file"
        assert cell.exists(), "dry-run must not move task files"
        assert not (queue_dir / "reset-s").exists(), "dry-run must not create the reset dir"
        # It still reports what it WOULD do.
        assert result.state_existed is True
        assert len(result.relocated_tasks) == 1
        assert result.dry_run is True


class TestServiceGuard:
    def test_refuses_when_service_active_without_force(self, tmp_path):
        state = _write_state(tmp_path / "v3", "get-k16-v16-t7-p1")
        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=tmp_path / "v3",
            queue_dir=tmp_path / "queue",
            service_active=True,
        )
        assert result.refused_reason is not None
        assert "conductress.service" in result.refused_reason
        assert state.exists(), "a refused reset must touch nothing"

    def test_force_overrides_active_service(self, tmp_path):
        state = _write_state(tmp_path / "v3", "get-k16-v16-t7-p1")
        result = reset_series(
            "get-k16-v16-t7-p1",
            state_dir=tmp_path / "v3",
            queue_dir=tmp_path / "queue",
            force=True,
            service_active=True,
            stamp="s",
        )
        assert result.refused_reason is None
        assert not state.exists()

    def test_unknown_epoch_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="v3"):
            reset_series(
                "get-k16-v16-t7-p1",
                epoch="v2",
                state_dir=tmp_path / "v3",
                queue_dir=tmp_path / "queue",
                service_active=False,
            )


class TestServiceProbe:
    def test_missing_systemctl_is_not_active(self, monkeypatch):
        import conductress.sweep.reset_series as mod

        def _boom(*_a, **_k):
            raise FileNotFoundError("systemctl")

        monkeypatch.setattr(mod.subprocess, "run", _boom)
        assert service_is_active() is False

    def test_active_output_is_active(self, monkeypatch):
        import conductress.sweep.reset_series as mod

        class _R:
            stdout = "active\n"

        monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: _R())
        assert service_is_active() is True

    def test_unknown_output_is_not_active(self, monkeypatch):
        import conductress.sweep.reset_series as mod

        class _R:
            stdout = "unknown\n"

        monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: _R())
        assert service_is_active() is False


class TestResultSummary:
    def test_summary_reports_refusal(self):
        result = ResetSeriesResult(
            label="get-k16-v16-t7-p1",
            state_file=Path("x"),
            dry_run=False,
            refused_reason="conductress.service is active",
        )
        assert any("Refused" in line for line in result.summary_lines())


class TestMemorySeries:
    """Memory series resolve to the memory coordinator's state files and note format."""

    def test_memory_prefix_selects_memory_metric(self):
        assert is_memory_workload("memory-sadd-m20")
        assert not is_memory_workload("get-k16-v16-t7-p1")

    def test_memory_state_file_matches_coordinator_naming(self, tmp_path):
        from conductress.sweep.memory_coordinator import MEMORY_WORKLOADS

        wl = next(w for w in MEMORY_WORKLOADS if w.label == "sadd-m20")
        valkey = state_file_for("memory-sadd-m20", None, state_dir=tmp_path)
        redis = state_file_for("memory-sadd-m20", "redis", state_dir=tmp_path)
        assert valkey.name == wl.state_file_for_engine(None).name == "memory_state_sadd-m20.json"
        assert redis.name == "memory_state_redis-sadd-m20.json"

    def test_default_state_dir_by_metric(self):
        from conductress import config
        from conductress.sweep.coordinator_v3 import V3_STATE_DIR

        assert default_state_dir("memory-sadd-m20") == config.MEMORY_STATE_DIR
        assert default_state_dir("get-k16-v16-t7-p1") == V3_STATE_DIR

    def test_label_carries_engine_after_metric_prefix(self, tmp_path):
        result = reset_series(
            "memory-sadd-m20", engine="redis", state_dir=tmp_path, queue_dir=tmp_path / "q", service_active=False
        )
        assert result.label == "memory-redis-sadd-m20"

    def test_memory_cells_matched_on_note_and_source(self, tmp_path):
        state_dir = tmp_path / "mem"
        queue_dir = tmp_path / "queue"
        state_dir.mkdir()
        state_file_for("memory-sadd-m20", None, state_dir=state_dir).write_text("{}")
        valkey_cell = _write_task(queue_dir, "0001", _memory_note("sadd-m20"), source="valkey", task_type="MemTaskData")
        redis_cell = _write_task(queue_dir, "0002", _memory_note("sadd-m20"), source="redis", task_type="MemTaskData")
        other_shape = _write_task(
            queue_dir, "0003", _memory_note("sadd-m200"), source="valkey", task_type="MemTaskData"
        )
        throughput = _write_task(queue_dir, "0004", _note("get-k16-v16-t7-p1"))

        result = reset_series(
            "memory-sadd-m20", state_dir=state_dir, queue_dir=queue_dir, stamp="s", service_active=False
        )

        assert not valkey_cell.exists(), "the Valkey memory cell belongs to the series"
        assert redis_cell.exists(), "the same shape under another engine is another series"
        assert other_shape.exists(), "the note prefix match includes the closing bracket"
        assert throughput.exists()
        assert result.state_existed and result.state_backup is not None
        assert len(result.relocated_tasks) == 1

    def test_engine_memory_series_takes_engine_cells(self, tmp_path):
        queue_dir = tmp_path / "queue"
        valkey_cell = _write_task(queue_dir, "0001", _memory_note("sadd-m20"), source="valkey", task_type="MemTaskData")
        redis_cell = _write_task(queue_dir, "0002", _memory_note("sadd-m20"), source="redis", task_type="MemTaskData")

        reset_series(
            "memory-sadd-m20", engine="redis", state_dir=tmp_path, queue_dir=queue_dir, stamp="s", service_active=False
        )

        assert valkey_cell.exists()
        assert not redis_cell.exists()


class TestResetAllMemory:
    def test_roster_is_every_memory_workload(self):
        from conductress.sweep.memory_coordinator import MEMORY_WORKLOADS

        assert memory_series_workloads() == [f"memory-{wl.label}" for wl in MEMORY_WORKLOADS]
        assert "memory-set-k16-v16-expire" in memory_series_workloads()

    def test_resets_every_series_under_one_stamp(self, tmp_path):
        state_dir = tmp_path / "mem"
        queue_dir = tmp_path / "queue"
        state_dir.mkdir()
        for workload in memory_series_workloads():
            state_file_for(workload, None, state_dir=state_dir).write_text("{}")
        cell = _write_task(queue_dir, "0001", _memory_note("hset-f16-v16"), task_type="MemTaskData")

        results = reset_all_memory_series(state_dir=state_dir, queue_dir=queue_dir, stamp="s", service_active=False)

        assert len(results) == len(memory_series_workloads())
        assert all(r.state_existed and not r.refused_reason for r in results)
        assert not any(state_file_for(w, None, state_dir=state_dir).exists() for w in memory_series_workloads())
        assert all(r.state_backup is not None and r.state_backup.name.endswith(".bak-s") for r in results)
        assert (queue_dir / "reset-s" / cell.name).exists()

    def test_engine_batch_leaves_valkey_state_alone(self, tmp_path):
        state_dir = tmp_path / "mem"
        state_dir.mkdir()
        valkey_file = state_file_for("memory-sadd-m20", None, state_dir=state_dir)
        redis_file = state_file_for("memory-sadd-m20", "redis", state_dir=state_dir)
        valkey_file.write_text("{}")
        redis_file.write_text("{}")

        results = reset_all_memory_series(
            engine="redis", state_dir=state_dir, queue_dir=tmp_path / "q", stamp="s", service_active=False
        )

        assert valkey_file.exists()
        assert not redis_file.exists()
        assert {r.label for r in results} == {f"memory-redis-{w[len('memory-'):]}" for w in memory_series_workloads()}

    def test_active_service_refuses_whole_batch(self, tmp_path):
        state_dir = tmp_path / "mem"
        state_dir.mkdir()
        state_file = state_file_for("memory-sadd-m20", None, state_dir=state_dir)
        state_file.write_text("{}")

        results = reset_all_memory_series(state_dir=state_dir, queue_dir=tmp_path / "q", service_active=True)

        assert all(r.refused_reason for r in results)
        assert state_file.exists()

    def test_force_overrides_probe(self, tmp_path):
        state_dir = tmp_path / "mem"
        state_dir.mkdir()
        state_file = state_file_for("memory-sadd-m20", None, state_dir=state_dir)
        state_file.write_text("{}")

        results = reset_all_memory_series(
            force=True, state_dir=state_dir, queue_dir=tmp_path / "q", stamp="s", service_active=True
        )

        assert not any(r.refused_reason for r in results)
        assert not state_file.exists()
