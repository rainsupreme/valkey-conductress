"""``queue add-mixed`` and ``queue add-latency`` submit cachecannon tasks.

Both commands keep their names and their positional/required arguments, but
the cell they queue is a ``CachecannonTaskData`` whose defaults are the
epoch-3 sweep shape, so a cell queued without overrides is comparable with
the sweep's points.  ``add-latency`` drives a fixed request rate with no
pipelining and scores p99; ``add-mixed`` scores throughput.
"""

import json
from unittest.mock import patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.task_queue import TaskQueue


@pytest.fixture(autouse=True)
def _ensure_valid_source(monkeypatch):
    """Another module may have left REPO_NAMES without 'valkey'; the commands validate against it."""
    if "valkey" not in config.REPO_NAMES:
        monkeypatch.setattr(config, "REPO_NAMES", config.REPO_NAMES + ["valkey"])


@pytest.fixture
def queue_path(tmp_path):
    path = tmp_path / "queue"
    path.mkdir()
    _Orig = TaskQueue

    class _Isolated(_Orig):
        def __init__(self, queue_dir_override=None):
            super().__init__(queue_dir=path)

    with patch("conductress.cli.TaskQueue", _Isolated):
        yield path


def _tasks(queue_path) -> list[dict]:
    return [json.loads(p.read_text()) for p in queue_path.glob("task_*.json")]


class TestAddMixed:
    def _run(self, *extra):
        base = ["queue", "add-mixed", "--source", "valkey", "--specifier", "unstable", "--set-ratio", "20"]
        return main(base + list(extra))

    def test_queues_a_cachecannon_cell_in_the_sweep_shape(self, queue_path):
        assert self._run() == 0
        (d,) = _tasks(queue_path)
        assert d["task_type"] == "CachecannonTaskData"
        assert d["set_ratio"] == 20 and d["test"] == "get"
        assert d["val_size"] == config.SWEEP_V3_VAL_SIZE
        assert d["io_threads"] == config.SWEEP_V3_IO_THREADS
        assert d["pipelining"] == config.SWEEP_V3_PIPELINING
        assert d["connections"] == config.SWEEP_V3_CONNECTIONS
        assert d["threads"] == config.SWEEP_V3_CLIENT_THREADS
        assert d["keyspace_count"] == config.SWEEP_V3_KEYSPACE
        assert d["warmup"] == config.SWEEP_V3_WARMUP
        assert d["duration"] == config.SWEEP_V3_DURATION
        assert d["repetitions"] == config.SWEEP_V3_REPETITIONS
        assert d["rate_limit"] == 0
        assert d["score_metric"] == "throughput"
        assert d["sweep_commit"] == "", "a manual cell must never be absorbed into the sweep series"

    @pytest.mark.parametrize(("value", "expected"), [("12s", 12), ("0s", 0)])
    def test_warmup_serialized(self, queue_path, value, expected):
        assert self._run("--warmup", value) == 0
        assert _tasks(queue_path)[0]["warmup"] == expected

    def test_cartesian_product_over_sizes_io_threads_and_pipelining(self, queue_path):
        assert self._run("--sizes", "16,512", "--io-threads", "7,9", "--pipelining", "10") == 0
        tasks = _tasks(queue_path)
        assert len(tasks) == 4
        assert {(t["val_size"], t["io_threads"]) for t in tasks} == {(16, 7), (16, 9), (512, 7), (512, 9)}

    def test_connections_and_threads_override(self, queue_path):
        assert self._run("--connections", "1200", "--threads", "24") == 0
        d = _tasks(queue_path)[0]
        assert d["connections"] == 1200 and d["threads"] == 24

    def test_perf_stat_note_and_server_args_carried(self, queue_path):
        assert (
            self._run("--perf-stat", "--note", "regression check", "--server-args", "--io-threads-ownership yes") == 0
        )
        d = _tasks(queue_path)[0]
        assert d["perf_stat_enabled"] is True
        assert d["note"] == "regression check"
        assert d["server_args"] == "--io-threads-ownership yes"

    def test_invalid_ratio_rejected(self, queue_path, capsys):
        assert main(["queue", "add-mixed", "--source", "valkey", "--specifier", "unstable", "--set-ratio", "150"]) == 1
        assert "set-ratio must be 0-100" in capsys.readouterr().err
        assert _tasks(queue_path) == []

    @pytest.mark.parametrize(("flag", "value"), [("--connections", "0"), ("--threads", "0"), ("--keyspace", "0")])
    def test_zero_concurrency_or_keyspace_rejected(self, queue_path, flag, value, capsys):
        assert self._run(flag, value) == 1
        assert "must be >= 1" in capsys.readouterr().err
        assert _tasks(queue_path) == []

    def test_memtier_flags_are_gone(self, queue_path):
        with pytest.raises(SystemExit):
            self._run("--memtier-threads", "24")
        with pytest.raises(SystemExit):
            self._run("--key-sizes", "32")

    def test_invalid_source_rejected(self, queue_path, capsys):
        assert (
            main(["queue", "add-mixed", "--source", "nosuchrepo", "--specifier", "unstable", "--set-ratio", "20"]) == 1
        )
        assert "Invalid source" in capsys.readouterr().err


class TestAddLatency:
    def _run(self, *extra, rps="100000"):
        return main(["queue", "add-latency", "valkey", "unstable", rps] + list(extra))

    def test_queues_a_rate_limited_p99_cell_in_the_sweep_shape(self, queue_path):
        assert self._run() == 0
        (d,) = _tasks(queue_path)
        assert d["task_type"] == "CachecannonTaskData"
        assert d["rate_limit"] == 100000
        assert d["score_metric"] == "p99"
        assert d["pipelining"] == config.SWEEP_V3_LATENCY_PIPELINING == 1
        assert d["test"] == "get" and d["set_ratio"] == 0
        assert d["val_size"] == config.SWEEP_V3_VAL_SIZE
        assert d["io_threads"] == config.SWEEP_V3_IO_THREADS
        assert d["connections"] == config.SWEEP_V3_CONNECTIONS
        assert d["threads"] == config.SWEEP_V3_CLIENT_THREADS
        assert d["keyspace_count"] == config.SWEEP_V3_KEYSPACE
        assert d["warmup"] == config.SWEEP_V3_WARMUP
        assert d["duration"] == config.SWEEP_V3_DURATION
        assert d["repetitions"] == config.SWEEP_V3_LATENCY_REPETITIONS
        assert d["sweep_commit"] == ""
        assert d["note"] == "manual latency @ 100000 rps"

    def test_matches_the_v3_latency_sweep_cell_except_for_ownership(self, queue_path, tmp_path):
        """The manual default and the sweep's own cell agree on every workload field."""
        from conductress.sweep.coordinator_v3 import CachecannonLatencySweepCoordinatorV3
        from conductress.sweep.planner import SweepTask

        assert self._run() == 0
        (manual,) = _tasks(queue_path)
        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                coord = CachecannonLatencySweepCoordinatorV3(tmp_path)
        sweep_cell = coord._create_task(SweepTask(commit="abc123", reason="test", date="2026-01-01", priority=1))
        workload_fields = (
            "test",
            "set_ratio",
            "val_size",
            "io_threads",
            "pipelining",
            "connections",
            "threads",
            "warmup",
            "duration",
            "keyspace_count",
            "distribution",
            "rate_limit",
            "score_metric",
        )
        for field in workload_fields:
            assert manual[field] == getattr(sweep_cell, field), field

    def test_set_ratio_and_value_size_override(self, queue_path):
        assert self._run("--set-ratio", "20", "--value-size", "512") == 0
        d = _tasks(queue_path)[0]
        assert d["set_ratio"] == 20 and d["val_size"] == 512
        assert d["note"] == "manual latency @ 100000 rps, SET=20%"

    def test_concurrency_and_repetitions_override(self, queue_path):
        assert self._run("--connections", "64", "--threads", "4", "--repetitions", "3", "--io-threads", "9") == 0
        d = _tasks(queue_path)[0]
        assert (d["connections"], d["threads"], d["repetitions"], d["io_threads"]) == (64, 4, 3, 9)

    def test_zero_rate_rejected(self, queue_path, capsys):
        assert self._run(rps="0") == 1
        assert "target_rps must be > 0" in capsys.readouterr().err
        assert _tasks(queue_path) == []

    def test_invalid_ratio_rejected(self, queue_path, capsys):
        assert self._run("--set-ratio", "101") == 1
        assert "set-ratio must be 0-100" in capsys.readouterr().err
