"""Integration test for perf stat collection through the sweep task path.

Requires a benchmark host with:
- valkey-server buildable from source
- perf stat accessible (perf_event_paranoid <= 1 or CAP_PERFMON)
- Runs with: pytest tests/integration/test_perf_stat_sweep_integration.py -m requires_server
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.config import ServerInfo
from conductress.file_protocol import FileProtocol
from conductress.sweep.exporter import _compute_metric, export_perf_metrics
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState
from conductress.tasks.task_perf_benchmark import PerfTaskData


class TestPerfStatSweepIntegration:
    """End-to-end test: task with perf_stat_enabled → counters in output → export."""

    pytestmark = pytest.mark.requires_server

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
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
    async def test_perf_stat_collected_in_sweep_task(self, temp_dir):
        """Run a benchmark with perf_stat_enabled=True and verify counters are recorded."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="perf stat integration test",
            requirements={},
            make_args="USE_FAST_FLOAT=yes",
            test="get",
            val_size=16,
            io_threads=1,
            pipelining=10,
            warmup=5,
            duration=10,
            profiling_sample_rate=0,
            perf_stat_enabled=True,
            has_expire=False,
            preload_keys=True,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_get_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # Parse output
        output_file = temp_dir / "output.jsonl"
        assert output_file.exists(), "output.jsonl not created"
        results = json.loads(output_file.read_text().strip().splitlines()[-1])

        # Verify perf counters are present
        data = results["data"]
        assert "perf_counters" in data, f"perf_counters missing from output. Keys: {list(data.keys())}"
        counters = data["perf_counters"]

        # Verify essential counters exist and are positive
        assert "instructions" in counters, f"instructions missing. Available: {list(counters.keys())}"
        assert "cycles" in counters, f"cycles missing. Available: {list(counters.keys())}"
        assert counters["instructions"] > 0, f"instructions = {counters['instructions']}"
        assert counters["cycles"] > 0, f"cycles = {counters['cycles']}"

        # Verify IPC is reasonable (0.5 - 5.0 for any modern CPU)
        ipc = counters["instructions"] / counters["cycles"]
        assert 0.5 < ipc < 5.0, f"IPC {ipc} outside reasonable range"

        # Verify perf_duration_seconds is recorded
        assert "perf_duration_seconds" in data
        assert data["perf_duration_seconds"] > 0

        # Verify throughput was also recorded (perf stat didn't break the benchmark)
        assert results["score"] > 0, "Benchmark produced no throughput"

    @patch("conductress.config.REPO_NAMES", ["valkey"])
    @patch("conductress.task_queue.config.REPO_NAMES", ["valkey"])
    @pytest.mark.asyncio
    async def test_perf_counters_export_pipeline(self, temp_dir):
        """Run benchmark → extract counters → export as normalized series files."""
        task_data = PerfTaskData(
            source="valkey",
            specifier="unstable",
            replicas=0,
            note="export pipeline test",
            requirements={},
            make_args="USE_FAST_FLOAT=yes",
            test="get",
            val_size=16,
            io_threads=1,
            pipelining=10,
            warmup=5,
            duration=10,
            profiling_sample_rate=0,
            perf_stat_enabled=True,
            has_expire=False,
            preload_keys=True,
        )

        server_info = ServerInfo(ip="127.0.0.1", username="test", name="test_server")
        runner = task_data.prepare_task_runner([server_info])
        task_name = f"{task_data.timestamp.strftime('%Y.%m.%d_%H.%M.%S.%f')}_get_perf"
        runner.file_protocol = FileProtocol(task_name, "client", temp_dir)

        await runner.run()

        # Parse output and build a BenchmarkPoint
        results = json.loads((temp_dir / "output.jsonl").read_text().strip().splitlines()[-1])
        data = results["data"]

        point = BenchmarkPoint(
            commit="test123",
            date="2024-01-01",
            value=results["score"],
            cv=0.5,
            reps=1,
            status=PointStatus.COMPLETED,
            perf_counters=data["perf_counters"],
            perf_duration_seconds=data["perf_duration_seconds"],
            perf_rps=results["score"],
        )

        # Verify normalization produces valid values
        ipc = _compute_metric(point, "ipc")
        assert ipc is not None and 0.5 < ipc < 5.0, f"IPC: {ipc}"

        icache = _compute_metric(point, "icache-mpki")
        if icache is not None:  # May not be available on all platforms
            assert 0 < icache < 100, f"icache MPKI: {icache}"

        insn_per_req = _compute_metric(point, "instructions-per-req")
        assert insn_per_req is not None and insn_per_req > 100, f"insn/req: {insn_per_req}"

        # Export to series files
        state = SweepState(merge_commits=["test123"], commit_dates={"test123": "2024-01-01"})
        state.points["test123"] = point

        export_dir = temp_dir / "export"
        exported = export_perf_metrics(state, export_dir, platform="test", workload="get16b-t1-p10")

        assert "ipc" in exported, f"IPC not exported. Exported: {exported}"
        assert exported["ipc"] == 1

        # Verify exported file is valid JSON with correct structure
        ipc_file = export_dir / "series-test-get16b-t1-p10-ipc.json"
        assert ipc_file.exists()
        series = json.loads(ipc_file.read_text())
        assert series["metadata"]["metric"] == "ipc"
        assert len(series["points"]) == 1
        assert 0.5 < series["points"][0]["value"] < 5.0
