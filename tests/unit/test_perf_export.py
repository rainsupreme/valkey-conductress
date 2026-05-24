"""Tests for perf metric export functionality."""

import json
from pathlib import Path

import pytest

from conductress.sweep.exporter import PERF_METRICS, _compute_metric, export_manifest, export_perf_metrics
from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepState


@pytest.fixture
def sample_state():
    """Create a SweepState with perf counter data."""
    state = SweepState(
        merge_commits=["aaa", "bbb", "ccc"],
        commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        commit_prs={"bbb": 42},
        commit_titles={"bbb": "Optimize hash table"},
        landmarks=[Landmark(commit="aaa", date="2024-01-01", label="8.0")],
    )
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa",
        date="2024-01-01",
        value=2000000.0,
        cv=0.5,
        reps=5,
        status=PointStatus.COMPLETED,
        perf_counters={
            "instructions": 900_000_000_000,
            "cycles": 300_000_000_000,
            "L1-icache-load-misses": 30_000_000,
            "branch-misses": 5_000_000,
            "branches": 100_000_000_000,
            "stalled-cycles-frontend": 15_000_000_000,
            "stalled-cycles-backend": 75_000_000_000,
            "LLC-load-misses": 9_000_000,
        },
        perf_duration_seconds=30.0,
        perf_rps=2000000.0,
    )
    state.points["bbb"] = BenchmarkPoint(
        commit="bbb",
        date="2024-02-01",
        value=1900000.0,
        cv=0.4,
        reps=5,
        status=PointStatus.COMPLETED,
        perf_counters={
            "instructions": 950_000_000_000,
            "cycles": 320_000_000_000,
            "L1-icache-load-misses": 35_000_000,
            "branch-misses": 6_000_000,
            "branches": 105_000_000_000,
            "stalled-cycles-frontend": 18_000_000_000,
            "stalled-cycles-backend": 80_000_000_000,
            "LLC-load-misses": 10_000_000,
        },
        perf_duration_seconds=30.0,
        perf_rps=1900000.0,
    )
    # Point without perf counters (should be skipped)
    state.points["ccc"] = BenchmarkPoint(
        commit="ccc",
        date="2024-03-01",
        value=1950000.0,
        cv=0.3,
        reps=5,
        status=PointStatus.COMPLETED,
    )
    return state


class TestComputeMetric:
    def test_ipc(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "ipc")
        assert result == pytest.approx(3.0, rel=1e-6)

    def test_icache_mpki(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "icache-mpki")
        # 30M / 900B * 1000 = 0.0333...
        assert result == pytest.approx(0.0333, rel=1e-2)

    def test_branch_mpki(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "branch-mpki")
        # 5M / 900B * 1000 = 0.00556
        assert result == pytest.approx(0.00556, rel=1e-2)

    def test_frontend_stall_pct(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "frontend-stall-pct")
        # 15B / 300B * 100 = 5.0%
        assert result == pytest.approx(5.0, rel=1e-6)

    def test_backend_stall_pct(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "backend-stall-pct")
        # 75B / 300B * 100 = 25.0%
        assert result == pytest.approx(25.0, rel=1e-6)

    def test_llc_mpki(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "llc-mpki")
        # 9M / 900B * 1000 = 0.01
        assert result == pytest.approx(0.01, rel=1e-6)

    def test_instructions_per_req(self, sample_state):
        point = sample_state.points["aaa"]
        result = _compute_metric(point, "instructions-per-req")
        # 900B / (2M * 30) = 15000
        assert result == pytest.approx(15000.0, rel=1e-6)

    def test_no_counters_returns_none(self):
        point = BenchmarkPoint(commit="x", date="2024-01-01", status=PointStatus.COMPLETED)
        assert _compute_metric(point, "ipc") is None

    def test_missing_event_returns_none(self, sample_state):
        point = sample_state.points["aaa"]
        point.perf_counters = {"instructions": 100}  # no cycles
        assert _compute_metric(point, "ipc") is None


class TestExportPerfMetrics:
    def test_exports_metric_files(self, sample_state, tmp_path):
        exported = export_perf_metrics(sample_state, tmp_path, platform="amd64", workload="get16b-t7-p10")
        assert "ipc" in exported
        assert exported["ipc"] == 2  # aaa and bbb have counters, ccc doesn't

        ipc_file = tmp_path / "series-amd64-get16b-t7-p10-ipc.json"
        assert ipc_file.exists()
        data = json.loads(ipc_file.read_text())
        assert data["metadata"]["metric"] == "ipc"
        assert data["metadata"]["platform"] == "amd64"
        assert len(data["points"]) == 2
        assert data["points"][0]["value"] == pytest.approx(3.0, rel=1e-3)

    def test_includes_pr_info(self, sample_state, tmp_path):
        export_perf_metrics(sample_state, tmp_path, platform="amd64", workload="get16b-t7-p10")
        ipc_file = tmp_path / "series-amd64-get16b-t7-p10-ipc.json"
        data = json.loads(ipc_file.read_text())
        # bbb has PR info
        bbb_point = [p for p in data["points"] if p["commit"] == "bbb"][0]
        assert bbb_point["pr"] == 42
        assert bbb_point["pr_title"] == "Optimize hash table"

    def test_includes_landmarks(self, sample_state, tmp_path):
        export_perf_metrics(sample_state, tmp_path, platform="amd64", workload="get16b-t7-p10")
        ipc_file = tmp_path / "series-amd64-get16b-t7-p10-ipc.json"
        data = json.loads(ipc_file.read_text())
        assert len(data["landmarks"]) == 1
        assert data["landmarks"][0]["label"] == "8.0"

    def test_skips_metrics_without_required_events(self, tmp_path):
        """If counters lack LLC-load-misses, llc-mpki should not be exported."""
        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=1000000.0,
            status=PointStatus.COMPLETED,
            perf_counters={"instructions": 100, "cycles": 50},  # no LLC data
            perf_duration_seconds=10.0,
            perf_rps=1000000.0,
        )
        exported = export_perf_metrics(state, tmp_path, platform="arm64", workload="get16b-t9-p10")
        assert "llc-mpki" not in exported
        assert "ipc" in exported

    def test_empty_state_returns_empty(self, tmp_path):
        state = SweepState()
        exported = export_perf_metrics(state, tmp_path, platform="amd64", workload="test")
        assert exported == {}


class TestExportManifest:
    def test_writes_manifest(self, tmp_path):
        export_manifest(tmp_path, platforms=["amd64", "arm64"], workloads=["get16b-t7-p10"])
        manifest_file = tmp_path / "manifest.json"
        assert manifest_file.exists()
        data = json.loads(manifest_file.read_text())
        assert data["version"] == 1
        assert "amd64" in data["platforms"]
        group_ids = [g["id"] for g in data["groups"]]
        assert "throughput" in group_ids
        assert "efficiency" in group_ids
        assert "cache" in group_ids
        assert "pipeline" in group_ids
        assert "branching" in group_ids
