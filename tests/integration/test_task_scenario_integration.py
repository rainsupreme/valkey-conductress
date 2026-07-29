"""Tier 2: Scenario task end-to-end integration test.

Requires: server-keyfile.pem + real valkey-server binary + memtier_benchmark.
Runs ONE cheap scenario (scan-churn, 5s, 1 rep) through the full task runner
and validates result structure, timeseries presence, and process cleanup.
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.config import ServerInfo
from conductress.file_protocol import FileProtocol
from conductress.tasks.task_scenario import ScenarioTaskData

pytestmark = pytest.mark.requires_server


class TestScenarioEndToEnd:
    """Integration test: run a full scenario task and validate results."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for task output files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with (
                patch("conductress.file_protocol.CONDUCTRESS_OUTPUT", tmp_path / "output.jsonl"),
                patch("conductress.file_protocol.CONDUCTRESS_RESULTS", tmp_path / "results"),
            ):
                yield tmp_path

    @patch("conductress.config.REPO_NAMES", ["valkey"])
    @patch("conductress.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_scan_churn_scenario_e2e(self, temp_dir):
        """Run scan-churn scenario: cheapest overlay (no valkey-benchmark binary needed)."""
        task_data = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="integration test",
            requirements={},
            scenario="scan-churn",
            val_size=128,
            io_threads=4,
            pipelining=10,
            duration=5,
            warmup=0,
            repetitions=1,
            perf_stat_enabled=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="", name="localhost")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_scenario"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # Validate results file was written
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists(), "No output.jsonl written"

        # Parse the final result record
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) >= 1, "output.jsonl is empty"
        result = json.loads(lines[-1])

        # Validate result record structure
        assert "data" in result, "Missing 'data' in result"
        data = result["data"]

        # Core fields from ScenarioTaskRunner.run()
        assert data["scenario"] == "scan-churn"
        assert data["duration"] == 5
        assert data["io_threads"] == 4
        assert data["pipeline"] == 10
        assert data["size"] == 128
        assert data["repetitions"] == 1

        # RPS data
        assert "per_run_rps" in data
        assert len(data["per_run_rps"]) == 1
        assert data["per_run_rps"][0] > 0, "RPS must be positive"
        assert "mean_rps" in data
        assert data["mean_rps"] > 0

        # Scenario metrics
        assert "scenario_metrics" in data
        sm = data["scenario_metrics"]
        assert "scenario" in sm
        assert sm["scenario"] == "scan-churn"
        # Dip keys must exist (compute_dip_metrics output)
        assert "per_rep" in sm
        if sm["per_rep"]:
            rep_metrics = sm["per_rep"][0]
            # These keys come from compute_dip_metrics
            for dip_key in ("dip_depth_pct", "dip_duration_seconds", "min_rps", "recovery_seconds"):
                assert dip_key in rep_metrics, f"Missing dip key '{dip_key}' in per_rep metrics"

        # Interval timeseries (best-effort: may be empty if JSON parse failed)
        if sm["per_rep"] and "interval_rps" in sm["per_rep"][0]:
            intervals = sm["per_rep"][0]["interval_rps"]
            assert isinstance(intervals, list)
            assert len(intervals) > 0, "interval_rps is present but empty"
            for val in intervals:
                assert val > 0, f"interval RPS value {val} is not positive"

        # Validate status shows completion
        status_file = runner.file_protocol.status_file
        assert status_file.exists(), "status file not written"
        status = json.loads(status_file.read_text())
        assert status["state"] == "completed"

    @patch("conductress.config.REPO_NAMES", ["valkey"])
    @patch("conductress.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_no_leftover_processes(self, temp_dir):
        """After scenario completes, no overlay/memtier/valkey-benchmark zombies remain."""
        task_data = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="cleanup test",
            requirements={},
            scenario="scan-churn",
            val_size=128,
            io_threads=4,
            pipelining=10,
            duration=5,
            warmup=0,
            repetitions=1,
            perf_stat_enabled=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="", name="localhost")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_cleanup"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # Check no leftover benchmark/overlay processes
        for proc_name in ("memtier_benchmark", "valkey-benchmark", "valkey-cli"):
            result = subprocess.run(
                ["pgrep", "-f", proc_name],
                capture_output=True,
                text=True,
            )
            # pgrep returns 0 if matches found, 1 if none
            assert (
                result.returncode != 0
            ), f"Leftover {proc_name} process(es) found after task completion: {result.stdout.strip()}"

    @patch("conductress.config.REPO_NAMES", ["valkey"])
    @patch("conductress.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_no_leftover_tmp_files(self, temp_dir):
        """After scenario completes, no /tmp payload files remain."""
        task_data = ScenarioTaskData(
            source="valkey",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="tmp cleanup test",
            requirements={},
            scenario="multi-exec",
            val_size=128,
            io_threads=4,
            pipelining=10,
            duration=5,
            warmup=0,
            repetitions=1,
            perf_stat_enabled=False,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="", name="localhost")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_tmp"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # multi-exec scenario creates /tmp/multi_exec_payload.resp -- should be cleaned up
        assert not Path("/tmp/multi_exec_payload.resp").exists(), "Leftover RESP payload file in /tmp"
        assert not Path("/tmp/multi_exec_payload.b64").exists(), "Leftover b64 payload file in /tmp"
