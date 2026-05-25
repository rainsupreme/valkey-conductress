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
        breakdown={"embedded_obj": 40.0, "sds": 24.0, "hashtable": 0, "dict": 24.0, "other": 0.3},
    )
    state.points["bbb"] = BenchmarkPoint(
        commit="bbb",
        date="2024-06-01",
        value=54.92,
        cv=0.0,
        reps=1,
        status=PointStatus.COMPLETED,
        breakdown={"embedded_obj": 56.0, "sds": 0, "hashtable": 12.8, "dict": 0, "other": 0.3},
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
        assert data["points"][0]["breakdown"]["embedded_obj"] == 40.0
        assert data["points"][1]["breakdown"]["hashtable"] == 12.8

    def test_includes_categories_in_metadata(self, state_with_breakdown, tmp_path):
        output = tmp_path / "series.json"
        export_series(state_with_breakdown, output, platform="arm64", workload="memory")

        data = json.loads(output.read_text())
        assert "categories" in data["metadata"]
        assert "embedded_obj" in data["metadata"]["categories"]
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
