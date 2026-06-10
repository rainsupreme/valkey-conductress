"""Tests for exporter breakdown and categories metadata."""

import json
from pathlib import Path

import pytest

from conductress.sweep.exporter import export_series
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState


@pytest.fixture
def state_with_breakdown():
    state = SweepState(
        merge_commits=["aaa", "bbb"],
        commit_dates={"aaa": "2024-01-01", "bbb": "2024-06-01"},
    )
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa",
        date="2024-01-01",
        value=84.15,
        cv=0.0,
        reps=1,
        status=PointStatus.COMPLETED,
        breakdown={"embedded_val": 40.0, "sds": 24.0, "hashtable": 0, "dict": 24.0, "other": 0.3},
    )
    state.points["bbb"] = BenchmarkPoint(
        commit="bbb",
        date="2024-06-01",
        value=54.92,
        cv=0.0,
        reps=1,
        status=PointStatus.COMPLETED,
        breakdown={"embedded_val": 56.0, "sds": 0, "hashtable": 12.8, "dict": 0, "other": 0.3},
    )
    return state


@pytest.fixture
def state_without_breakdown():
    state = SweepState(
        merge_commits=["aaa"],
        commit_dates={"aaa": "2024-01-01"},
    )
    state.points["aaa"] = BenchmarkPoint(
        commit="aaa",
        date="2024-01-01",
        value=84.15,
        cv=0.0,
        reps=1,
        status=PointStatus.COMPLETED,
    )
    return state


class TestExportBreakdown:
    def test_includes_breakdown_in_points(self, state_with_breakdown, tmp_path):
        output = tmp_path / "series.json"
        export_series(state_with_breakdown, output, platform="arm64", workload="memory")

        data = json.loads(output.read_text())
        assert data["points"][0]["breakdown"]["embedded_val"] == 40.0
        assert data["points"][1]["breakdown"]["hashtable"] == 12.8

    def test_includes_categories_in_metadata(self, state_with_breakdown, tmp_path):
        output = tmp_path / "series.json"
        export_series(state_with_breakdown, output, platform="arm64", workload="memory")

        data = json.loads(output.read_text())
        assert "categories" in data["metadata"]
        assert "embedded_val" in data["metadata"]["categories"]
        assert "other" in data["metadata"]["categories"]

    def test_no_categories_without_breakdown(self, state_without_breakdown, tmp_path):
        output = tmp_path / "series.json"
        export_series(state_without_breakdown, output, platform="arm64", workload="memory")

        data = json.loads(output.read_text())
        assert "categories" not in data["metadata"]

    def test_no_breakdown_key_without_data(self, state_without_breakdown, tmp_path):
        output = tmp_path / "series.json"
        export_series(state_without_breakdown, output, platform="arm64", workload="memory")

        data = json.loads(output.read_text())
        assert "breakdown" not in data["points"][0]

    def test_recategorizes_from_raw_stacks_at_export(self, tmp_path):
        """When num_keys is provided, export uses raw_stacks instead of stored breakdown."""
        state = SweepState(
            merge_commits=["aaa"],
            commit_dates={"aaa": "2024-01-01"},
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=50.0,
            cv=0.0,
            reps=1,
            status=PointStatus.COMPLETED,
            breakdown={"sds": 999.0},  # stale breakdown — should be ignored
            raw_stacks=[
                [["je_malloc", "sdsnewlen", "setCommand"], 40_000_000],
                [["je_malloc", "hashtableExpand", "dbAdd"], 10_000_000],
            ],
        )
        output = tmp_path / "series.json"
        export_series(state, output, platform="arm64", workload="memory", num_keys=5_000_000)

        data = json.loads(output.read_text())
        bd = data["points"][0]["breakdown"]
        assert bd["sds"] == 8.0  # 40M / 5M
        assert bd["hashtable"] == 2.0  # 10M / 5M
        # Stale breakdown value (999.0) was NOT used
        assert bd["sds"] != 999.0

    def test_falls_back_to_breakdown_without_raw_stacks(self, tmp_path):
        """Points without raw_stacks use stored breakdown (backward compat)."""
        state = SweepState(
            merge_commits=["aaa"],
            commit_dates={"aaa": "2024-01-01"},
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=50.0,
            cv=0.0,
            reps=1,
            status=PointStatus.COMPLETED,
            breakdown={"embedded_val": 40.0, "hashtable": 10.0},
        )
        output = tmp_path / "series.json"
        export_series(state, output, platform="arm64", workload="memory", num_keys=5_000_000)

        data = json.loads(output.read_text())
        assert data["points"][0]["breakdown"]["embedded_val"] == 40.0
