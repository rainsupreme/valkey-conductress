"""Unit tests for the pathological-workload scenario task."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress import config
from conductress.cli import main
from conductress.task_queue import TaskQueue
from conductress.tasks.task_scenario import (
    OVERLAY_CLIENTS,
    OVERLAY_THREADS,
    SCENARIO_CHOICES,
    ScenarioTaskData,
    ScenarioTaskRunner,
    build_overlay_command,
    compute_dip_metrics,
    encode_resp_array,
    generate_multi_exec_resp_payload,
    parse_memtier_json_intervals,
    validate_scenario,
)


class TestValidateScenario:
    """Tests for scenario name validation."""

    def test_valid_scenarios(self):
        for name in SCENARIO_CHOICES:
            assert validate_scenario(name) is True

    def test_invalid_scenario(self):
        assert validate_scenario("nonexistent") is False
        assert validate_scenario("") is False
        assert validate_scenario("eval_storm") is False  # underscore not dash


class TestComputeDipMetrics:
    """Tests for throughput dip analysis."""

    def test_no_dip(self):
        # Steady at 1M rps, no dip
        baseline = 1_000_000.0
        intervals = [1_000_000.0] * 30
        result = compute_dip_metrics(baseline, intervals)
        assert result["dip_depth_pct"] == 0.0
        assert result["dip_duration_seconds"] == 0
        assert result["min_rps"] == 1_000_000.0
        assert result["recovery_seconds"] == 0

    def test_full_dip(self):
        # Drop to zero for 5 seconds
        baseline = 1_000_000.0
        intervals = [1_000_000.0] * 10 + [0.0] * 5 + [1_000_000.0] * 15
        result = compute_dip_metrics(baseline, intervals)
        assert result["dip_depth_pct"] == 100.0
        assert result["dip_duration_seconds"] == 5
        assert result["min_rps"] == 0.0

    def test_partial_dip(self):
        # Drop to 50% for 3 seconds
        baseline = 1_000_000.0
        intervals = [1_000_000.0] * 10 + [500_000.0] * 3 + [1_000_000.0] * 17
        result = compute_dip_metrics(baseline, intervals)
        assert result["dip_depth_pct"] == 50.0
        assert result["dip_duration_seconds"] == 3
        assert result["min_rps"] == 500_000.0

    def test_empty_intervals(self):
        result = compute_dip_metrics(1_000_000.0, [])
        assert result["dip_depth_pct"] == 0.0

    def test_zero_baseline(self):
        result = compute_dip_metrics(0.0, [100.0, 200.0])
        assert result["dip_depth_pct"] == 0.0

    def test_recovery_time(self):
        # Drop below 80%, then recover above 90% after 4 seconds
        baseline = 1_000_000.0
        intervals = (
            [1_000_000.0] * 5
            + [700_000.0]  # below 80% (800K threshold)
            + [750_000.0]  # still below 80%
            + [850_000.0]  # below 90% (900K threshold)
            + [950_000.0]  # above 90% -> recovered
            + [1_000_000.0] * 21
        )
        result = compute_dip_metrics(baseline, intervals)
        assert result["recovery_seconds"] == 4  # from first dip to recovery


class TestBuildOverlayCommand:
    """Tests for overlay command construction."""

    def test_eval_storm(self):
        cmd = build_overlay_command("eval-storm", "127.0.0.1", 6379, 30, 3_000_000, 512)
        assert "valkey-benchmark" in cmd
        assert "EVAL" in cmd
        assert "-h 127.0.0.1" in cmd
        assert "-p 6379" in cmd
        assert "-r 3000000" in cmd

    def test_scan_churn(self):
        cmd = build_overlay_command("scan-churn", "127.0.0.1", 6379, 30, 3_000_000, 512)
        assert "valkey-cli" in cmd
        assert "--scan" in cmd
        assert "SECONDS" in cmd  # bash time loop

    def test_multi_exec(self):
        cmd = build_overlay_command("multi-exec", "127.0.0.1", 6379, 30, 3_000_000, 512)
        assert "valkey-cli" in cmd
        assert "--pipe" in cmd
        assert "/tmp/multi_exec_payload.resp" in cmd
        assert "valkey-benchmark" not in cmd
        # Should loop-replay until duration expires
        assert "SECONDS" in cmd
        # Should clean up payload file at end of loop
        assert "rm -f" in cmd

    def test_flushall_spike(self):
        cmd = build_overlay_command("flushall-spike", "127.0.0.1", 6379, 30, 3_000_000, 512)
        assert "FLUSHALL ASYNC" in cmd
        assert "memtier_benchmark" in cmd  # re-prefill
        assert "sleep" in cmd

    def test_expiry_heavy(self):
        cmd = build_overlay_command("expiry-heavy", "127.0.0.1", 6379, 30, 3_000_000, 512)
        assert "valkey-benchmark" in cmd
        assert "EX 3" in cmd
        assert "-r 3000000" in cmd

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            build_overlay_command("bogus", "127.0.0.1", 6379, 30, 3_000_000, 512)

    def test_overlay_connections(self):
        """Overlay uses limited connections to avoid dominating."""
        cmd = build_overlay_command("eval-storm", "127.0.0.1", 6379, 30, 3_000_000, 512)
        expected_conns = OVERLAY_THREADS * OVERLAY_CLIENTS
        assert f"-c {expected_conns}" in cmd


class TestEncodeRespArray:
    """Tests for RESP array encoding."""

    def test_single_arg(self):
        result = encode_resp_array("PING")
        assert result == b"*1\r\n$4\r\nPING\r\n"

    def test_multi_arg(self):
        result = encode_resp_array("SET", "mykey", "myvalue")
        assert result == b"*3\r\n$3\r\nSET\r\n$5\r\nmykey\r\n$7\r\nmyvalue\r\n"

    def test_multi_command(self):
        result = encode_resp_array("MULTI")
        assert result == b"*1\r\n$5\r\nMULTI\r\n"

    def test_exec_command(self):
        result = encode_resp_array("EXEC")
        assert result == b"*1\r\n$4\r\nEXEC\r\n"

    def test_empty_value(self):
        result = encode_resp_array("SET", "key", "")
        assert result == b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$0\r\n\r\n"

    def test_byte_exact_transaction(self):
        """Verify a full MULTI/GET/SET/EXEC transaction is byte-correct."""
        parts = []
        parts.append(encode_resp_array("MULTI"))
        parts.append(encode_resp_array("GET", "memtier-42"))
        parts.append(encode_resp_array("SET", "memtier-42", "xx"))
        parts.append(encode_resp_array("EXEC"))
        full = b"".join(parts)
        expected = (
            b"*1\r\n$5\r\nMULTI\r\n"
            b"*2\r\n$3\r\nGET\r\n$10\r\nmemtier-42\r\n"
            b"*3\r\n$3\r\nSET\r\n$10\r\nmemtier-42\r\n$2\r\nxx\r\n"
            b"*1\r\n$4\r\nEXEC\r\n"
        )
        assert full == expected


class TestGenerateMultiExecRespPayload:
    """Tests for MULTI/EXEC RESP payload generation."""

    def test_basic_structure(self):
        """Generated payload has correct number of commands (4 per transaction)."""
        payload = generate_multi_exec_resp_payload(num_transactions=5, keyspace=100, val_size=8)
        # Each transaction: MULTI + GET + SET + EXEC = 4 commands
        # Count MULTI occurrences
        assert payload.count(b"MULTI") == 5
        assert payload.count(b"EXEC") == 5
        assert payload.count(b"GET") == 5
        assert payload.count(b"SET") == 5

    def test_key_format(self):
        """Keys use memtier-<N> format."""
        payload = generate_multi_exec_resp_payload(num_transactions=10, keyspace=1000, val_size=4)
        # All keys should have memtier- prefix
        assert b"memtier-" in payload

    def test_value_size(self):
        """Value is exactly val_size bytes."""
        payload = generate_multi_exec_resp_payload(num_transactions=1, keyspace=100, val_size=16)
        # Value should be 16 'x' characters, encoded as $16\r\nxxxxxxxxxxxxxxxx\r\n
        assert b"$16\r\n" + b"x" * 16 + b"\r\n" in payload

    def test_keyspace_bounds(self):
        """All keys should be within the specified keyspace."""
        import random

        random.seed(42)
        payload = generate_multi_exec_resp_payload(num_transactions=100, keyspace=50, val_size=4)
        # Extract key numbers from the payload
        decoded = payload.decode()
        import re

        keys = re.findall(r"memtier-(\d+)", decoded)
        for k in keys:
            assert 1 <= int(k) <= 50

    def test_valid_resp_format(self):
        """Each line follows RESP protocol conventions."""
        payload = generate_multi_exec_resp_payload(num_transactions=2, keyspace=100, val_size=4)
        # Should start with *1\r\n (MULTI is a 1-arg command)
        assert payload.startswith(b"*1\r\n$5\r\nMULTI\r\n")
        # Should end with EXEC
        assert payload.endswith(b"*1\r\n$4\r\nEXEC\r\n")

    def test_payload_not_empty(self):
        payload = generate_multi_exec_resp_payload(num_transactions=1, keyspace=10, val_size=1)
        assert len(payload) > 0

    def test_deterministic_with_seed(self):
        """Same seed produces same payload."""
        import random

        random.seed(123)
        p1 = generate_multi_exec_resp_payload(num_transactions=5, keyspace=100, val_size=8)
        random.seed(123)
        p2 = generate_multi_exec_resp_payload(num_transactions=5, keyspace=100, val_size=8)
        assert p1 == p2


class TestParseMemtierJsonIntervals:
    """Tests for memtier JSON interval parsing."""

    def test_empty_content(self):
        assert parse_memtier_json_intervals("/tmp/test.json", "") == []

    def test_invalid_json(self):
        assert parse_memtier_json_intervals("/tmp/test.json", "not json") == []

    def test_no_all_stats(self):
        assert parse_memtier_json_intervals("/tmp/test.json", "{}") == []

    def test_with_second_entries(self):
        data = {
            "ALL STATS": {
                "Second 1": {"Ops/sec": 100000.5},
                "Second 2": {"Ops/sec": 95000.0},
                "Second 3": {"Ops/sec": 110000.0},
                "Totals": {"Ops/sec": 101666.8},
            }
        }
        result = parse_memtier_json_intervals("/tmp/test.json", json.dumps(data))
        assert result == [100000.5, 95000.0, 110000.0]


class TestScenarioTaskDataValidation:
    """Tests for ScenarioTaskData construction and validation."""

    @pytest.fixture(autouse=True)
    def patch_sources(self):
        with (
            patch.object(config, "REPO_NAMES", ["valkey", "testrepo"]),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "testrepo"]),
            patch.object(config, "MANUALLY_UPLOADED", "manually_uploaded"),
            patch("conductress.task_queue.config.MANUALLY_UPLOADED", "manually_uploaded"),
        ):
            yield

    def test_valid_scenario(self):
        task = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            scenario="eval-storm",
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=30,
        )
        assert task.scenario == "eval-storm"
        assert task.task_type == "ScenarioTaskData"

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            ScenarioTaskData(
                source="valkey",
                specifier="unstable",
                make_args="",
                replicas=0,
                note="",
                requirements={},
                scenario="bogus-scenario",
                val_size=512,
                io_threads=9,
                pipelining=10,
                duration=30,
            )

    def test_all_valid_scenarios_accepted(self):
        for scenario in SCENARIO_CHOICES:
            task = ScenarioTaskData(
                source="valkey",
                specifier="unstable",
                make_args="",
                replicas=0,
                note="",
                requirements={},
                scenario=scenario,
                val_size=512,
                io_threads=9,
                pipelining=10,
                duration=30,
            )
            assert task.scenario == scenario

    def test_short_description(self):
        task = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            scenario="flushall-spike",
            val_size=512,
            io_threads=9,
            pipelining=10,
            duration=60,
        )
        desc = task.short_description()
        assert "flushall-spike" in desc
        assert "io=9" in desc

    def test_serialization_round_trip(self, tmp_path):
        """Task can be saved to JSON and reloaded."""
        task = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="regression test",
            requirements={},
            scenario="expiry-heavy",
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
        assert isinstance(loaded, ScenarioTaskData)
        assert loaded.scenario == "expiry-heavy"
        assert loaded.val_size == 512
        assert loaded.io_threads == 9
        assert loaded.pipelining == 10
        assert loaded.duration == 30
        assert loaded.repetitions == 5
        assert loaded.perf_stat_enabled is True
        assert loaded.note == "regression test"


class TestCliAddScenario:
    """Tests for 'queue add-scenario' CLI subcommand."""

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

    def test_basic_add_scenario(self):
        exit_code = main(
            [
                "queue",
                "add-scenario",
                "--scenario",
                "eval-storm",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        assert len(tasks) == 1

        data = json.loads(tasks[0].read_text())
        assert data["task_type"] == "ScenarioTaskData"
        assert data["scenario"] == "eval-storm"

    def test_all_scenarios_accepted(self):
        for scenario in SCENARIO_CHOICES:
            exit_code = main(
                [
                    "queue",
                    "add-scenario",
                    "--scenario",
                    scenario,
                    "--source",
                    "valkey",
                    "--specifier",
                    "unstable",
                ]
            )
            assert exit_code == 0

    def test_invalid_scenario_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "queue",
                    "add-scenario",
                    "--scenario",
                    "nonexistent",
                    "--source",
                    "valkey",
                    "--specifier",
                    "unstable",
                ]
            )
        assert exc_info.value.code != 0

    def test_invalid_source_rejected(self, capsys):
        exit_code = main(
            [
                "queue",
                "add-scenario",
                "--scenario",
                "eval-storm",
                "--source",
                "nosuchrepo",
                "--specifier",
                "unstable",
            ]
        )
        assert exit_code == 1
        assert "Invalid source" in capsys.readouterr().err

    def test_perf_stat_flag(self):
        exit_code = main(
            [
                "queue",
                "add-scenario",
                "--scenario",
                "scan-churn",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
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
                "add-scenario",
                "--scenario",
                "multi-exec",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--note",
                "testing concurrency",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["note"] == "testing concurrency"

    def test_custom_duration_and_reps(self):
        exit_code = main(
            [
                "queue",
                "add-scenario",
                "--scenario",
                "eval-storm",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--duration",
                "2m",
                "--repetitions",
                "5",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["duration"] == 120
        assert data["repetitions"] == 5

    def test_custom_io_threads_and_pipelining(self):
        exit_code = main(
            [
                "queue",
                "add-scenario",
                "--scenario",
                "expiry-heavy",
                "--source",
                "valkey",
                "--specifier",
                "unstable",
                "--io-threads",
                "7",
                "--pipelining",
                "50",
            ]
        )
        assert exit_code == 0
        tasks = list(self.queue_path.glob("task_*.json"))
        data = json.loads(tasks[0].read_text())
        assert data["io_threads"] == 7
        assert data["pipelining"] == 50
