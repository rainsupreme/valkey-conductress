"""Unit tests for the mixed GET/SET throughput task."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.task_queue import TaskQueue
from conductress.tasks.task_mixed import (
    MixedTaskData,
    MixedTaskRunner,
    parse_memtier_total_rps,
    set_ratio_to_memtier_ratio,
)


class TestSetRatioConversion:
    """Tests for set_ratio_to_memtier_ratio conversion."""

    def test_pure_get(self):
        assert set_ratio_to_memtier_ratio(0) == "0:1"

    def test_pure_set(self):
        assert set_ratio_to_memtier_ratio(100) == "1:0"

    def test_20_percent_set(self):
        # 20% SET / 80% GET -> gcd(20,80)=20 -> 1:4
        assert set_ratio_to_memtier_ratio(20) == "1:4"

    def test_50_percent_set(self):
        # 50/50 -> gcd(50,50)=50 -> 1:1
        assert set_ratio_to_memtier_ratio(50) == "1:1"

    def test_10_percent_set(self):
        # 10/90 -> gcd(10,90)=10 -> 1:9
        assert set_ratio_to_memtier_ratio(10) == "1:9"

    def test_33_percent_set(self):
        # 33/67 -> gcd(33,67)=1 -> 33:67
        assert set_ratio_to_memtier_ratio(33) == "33:67"

    def test_75_percent_set(self):
        # 75/25 -> gcd(75,25)=25 -> 3:1
        assert set_ratio_to_memtier_ratio(75) == "3:1"


class TestMemtierOutputParsing:
    """Tests for memtier stdout parsing."""

    def test_parse_totals_line(self):
        output = """Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec
----------------------------------------------------------------------------------------------------------------------------------------
Sets        250000.12          ---          ---         0.123         0.100         0.500         1.000      1234.56
Gets       1000000.34    1000000.34         0.00         0.089         0.080         0.400         0.900      5678.90
Totals     1250000.46    1000000.34         0.00         0.096         0.085         0.420         0.920      6913.46"""
        result = parse_memtier_total_rps(output)
        assert result == 1250000.46

    def test_parse_no_totals(self):
        output = "some random output\nno totals here"
        result = parse_memtier_total_rps(output)
        assert result is None

    def test_parse_empty_output(self):
        result = parse_memtier_total_rps("")
        assert result is None


class TestMixedTaskDataValidation:
    """Tests for MixedTaskData construction and validation."""

    @pytest.fixture(autouse=True)
    def patch_sources(self):
        with (
            patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
            patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
            patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
        ):
            yield

    def test_valid_ratio_0(self):
        task = MixedTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            set_ratio=0,
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=30,
        )
        assert task.set_ratio == 0
        assert task.task_type == "MixedTaskData"

    def test_valid_ratio_100(self):
        task = MixedTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            set_ratio=100,
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=30,
        )
        assert task.set_ratio == 100

    def test_invalid_ratio_negative(self):
        with pytest.raises(ValueError, match="set_ratio must be 0-100"):
            MixedTaskData(
                source="valkey",
                specifier="unstable",
                make_args="",
                replicas=0,
                note="",
                requirements={},
                set_ratio=-1,
                val_size=512,
                io_threads=9,
                pipelining=10,
                duration=30,
            )

    def test_invalid_ratio_over_100(self):
        with pytest.raises(ValueError, match="set_ratio must be 0-100"):
            MixedTaskData(
                source="valkey",
                specifier="unstable",
                make_args="",
                replicas=0,
                note="",
                requirements={},
                set_ratio=101,
                val_size=512,
                io_threads=9,
                pipelining=10,
                duration=30,
            )

    def test_short_description(self):
        task = MixedTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            set_ratio=20,
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=30,
        )
        desc = task.short_description()
        assert "20%SET" in desc
        assert "80%GET" in desc
        assert "io=9" in desc

    def test_serialization_round_trip(self, tmp_path):
        """Task can be saved to JSON and reloaded."""
        task = MixedTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="test note",
            requirements={},
            set_ratio=20,
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=30,
            repetitions=5,
            perf_stat_enabled=True,
        )
        filepath = tmp_path / "task.json"
        task.save_to_file(filepath)

        from conductress.task_queue import BaseTaskData

        loaded = BaseTaskData.from_file(filepath)
        assert isinstance(loaded, MixedTaskData)
        assert loaded.set_ratio == 20
        assert loaded.val_size == 512
        assert loaded.io_threads == 9
        assert loaded.pipelining == 10
        assert loaded.duration == 30
        assert loaded.repetitions == 5
        assert loaded.perf_stat_enabled is True
        assert loaded.note == "test note"


class TestCliAddMixed:
    """Tests for 'queue add-mixed' CLI subcommand."""

    @pytest.fixture(autouse=True)
    def isolate_queue(self, tmp_path):
        """Patch TaskQueue to use temp dir."""
        queue_path = tmp_path / "queue"
        queue_path.mkdir()

        _OriginalTaskQueue = TaskQueue

        class _IsolatedTaskQueue(_OriginalTaskQueue):
            def __init__(self, queue_dir_override=None):
                super().__init__(queue_dir=queue_path)

        with patch("conductress.cli.TaskQueue", _IsolatedTaskQueue):
            self.queue_path = queue_path
            yield

    @pytest.fixture(autouse=True)
    def patch_sources(self):
        with (
            patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
            patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
            patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
        ):
            yield

    def test_basic_add_mixed(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--sizes",
                "512",
                "--io-threads",
                "9",
                "--pipelining",
                "10",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        assert len(tasks) == 1

        data = json.loads(tasks[0].read_text())
        assert data["task_type"] == "MixedTaskData"
        assert data["set_ratio"] == 20
        assert data["val_size"] == 512
        assert data["io_threads"] == 9
        assert data["pipelining"] == 10

    def test_invalid_ratio_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "150",
            ]
        )
        assert exit_code == 1
        assert "set-ratio must be 0-100" in capsys.readouterr().err

    def test_cartesian_product(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "50",
                "--sizes",
                "16,512",
                "--io-threads",
                "7,9",
                "--pipelining",
                "10",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        # 2 sizes * 2 io-threads * 1 pipeline * 1 key-size = 4
        assert len(tasks) == 4

    def test_invalid_source_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "nosuchrepo",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
            ]
        )
        assert exit_code == 1
        assert "Invalid source" in capsys.readouterr().err

    def test_perf_stat_flag(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "30",
                "--perf-stat",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["perf_stat_enabled"] is True

    def test_note_stored(self):
        exit_code = main(
            [
                "queue",
                "add-mixed",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--set-ratio",
                "20",
                "--note",
                "regression check",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["note"] == "regression check"
