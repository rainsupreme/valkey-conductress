"""Unit tests for the sweep exporter."""

import json
import tempfile
from pathlib import Path

import pytest

from conductress.sweep.exporter import NotableSource, export_notable, export_series
from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepState


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def make_state_with_results() -> SweepState:
    """Create a state with some benchmark results for export testing."""
    state = SweepState(
        merge_commits=["a", "b", "c", "d", "e"],
        commit_dates={
            "a": "2024-03-20",
            "b": "2024-04-15",
            "c": "2024-05-10",
            "d": "2024-06-01",
            "e": "2024-07-01",
        },
        landmarks=[Landmark(commit="c", date="2024-05-10", label="8.0.0")],
    )
    state.points["a"] = BenchmarkPoint(
        commit="a",
        date="2024-03-20",
        value=148000,
        cv=0.19,
        status=PointStatus.COMPLETED,
    )
    state.points["b"] = BenchmarkPoint(
        commit="b",
        date="2024-04-15",
        value=139000,
        cv=0.21,
        pr=1847,
        pr_title="Refactor output buffer",
        status=PointStatus.COMPLETED,
    )
    state.points["c"] = BenchmarkPoint(
        commit="c",
        date="2024-05-10",
        value=140000,
        cv=0.18,
        status=PointStatus.COMPLETED,
    )
    state.points["e"] = BenchmarkPoint(
        commit="e",
        date="2024-07-01",
        value=152000,
        cv=0.20,
        status=PointStatus.COMPLETED,
    )
    return state


class TestExporter:
    """Tests for series.json export."""

    def test_export_creates_file(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        assert output.exists()

    def test_export_valid_json(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        assert "metadata" in data
        assert "points" in data
        assert "landmarks" in data
        assert "annotations" in data

    def test_export_metadata(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output, platform="test-platform", workload="GET_16B_t1_p1")
        data = json.loads(output.read_text())
        assert data["metadata"]["repo"] == "valkey-io/valkey"
        assert data["metadata"]["branch"] == "unstable"
        assert data["metadata"]["platform"] == "test-platform"

    def test_export_points_ordered(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        points = data["points"]
        # Should be in commit order (a, b, c, e — d has no result)
        commits = [p["commit"] for p in points]
        assert commits == ["a", "b", "c", "e"]

    def test_export_includes_pr_info(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        # Point "b" has PR info
        b_point = next(p for p in data["points"] if p["commit"] == "b")
        assert b_point["pr"] == 1847

    def test_export_landmarks(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        assert len(data["landmarks"]) == 1
        assert data["landmarks"][0]["label"] == "8.0.0"

    def test_export_annotations_for_pinpointed_changes(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        # a->b is adjacent (index 0->1), delta = (139000-148000)/148000 = -6.08%
        annotations = data["annotations"]
        assert len(annotations) >= 1
        regression = next((a for a in annotations if a["commit"] == "b"), None)
        assert regression is not None
        assert regression["type"] == "decrease"
        assert regression["good"] is False
        assert regression["delta"] < 0
        assert regression["pr"] == 1847

    def test_export_annotations_include_commit_date(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        regression = next(a for a in data["annotations"] if a["commit"] == "b")
        assert regression["date"] == "2024-04-15"

    def test_export_empty_state(self, tmp_path):
        state = SweepState()
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        assert data["points"] == []
        assert data["annotations"] == []

    def test_export_creates_parent_dirs(self, tmp_path):
        state = make_state_with_results()
        output = tmp_path / "nested" / "dir" / "series.json"
        export_series(state, output)
        assert output.exists()

    def test_export_skips_build_failures(self, tmp_path):
        state = make_state_with_results()
        state.points["d"] = BenchmarkPoint(
            commit="d",
            date="2024-06-01",
            status=PointStatus.BUILD_FAILED,
        )
        output = tmp_path / "series.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        commits = [p["commit"] for p in data["points"]]
        assert "d" not in commits


def make_memory_state() -> SweepState:
    """Create a state whose adjacent-commit change is an increase (bad for memory)."""
    state = SweepState(
        merge_commits=["a", "b", "c"],
        commit_dates={"a": "2024-05-01", "b": "2024-08-20", "c": "2024-09-01"},
    )
    state.points["a"] = BenchmarkPoint(
        commit="a",
        date="2024-05-01",
        value=100.0,
        status=PointStatus.COMPLETED,
    )
    state.points["b"] = BenchmarkPoint(
        commit="b",
        date="2024-08-20",
        value=110.0,
        pr=2001,
        pr_title="Grow per-key metadata",
        status=PointStatus.COMPLETED,
    )
    return state


class TestExportNotable:
    """Tests for the combined notable-changes export."""

    def test_aggregates_across_workloads_and_metrics(self, tmp_path):
        sources = [
            NotableSource(state=make_state_with_results(), workload="get-16b", metric="throughput"),
            NotableSource(state=make_memory_state(), workload="set-m20", metric="memory", lower_is_better=True),
        ]
        output = tmp_path / "notable.json"
        export_notable(sources, output, platform="test-platform")
        data = json.loads(output.read_text())
        assert data["metadata"]["platform"] == "test-platform"
        pairs = {(a["workload"], a["metric"]) for a in data["annotations"]}
        assert ("get-16b", "throughput") in pairs
        assert ("set-m20", "memory") in pairs

    def test_entries_sorted_newest_first(self, tmp_path):
        sources = [
            NotableSource(state=make_state_with_results(), workload="get-16b", metric="throughput"),
            NotableSource(state=make_memory_state(), workload="set-m20", metric="memory", lower_is_better=True),
        ]
        output = tmp_path / "notable.json"
        export_notable(sources, output, platform="test-platform")
        data = json.loads(output.read_text())
        dates = [a["date"] for a in data["annotations"]]
        assert dates == sorted(dates, reverse=True)
        # Memory entry (2024-08-20) is newer than throughput entry (2024-04-15)
        assert data["annotations"][0]["metric"] == "memory"

    def test_lower_is_better_polarity(self, tmp_path):
        sources = [
            NotableSource(state=make_memory_state(), workload="set-m20", metric="memory", lower_is_better=True),
        ]
        output = tmp_path / "notable.json"
        export_notable(sources, output, platform="test-platform")
        data = json.loads(output.read_text())
        entry = next(a for a in data["annotations"] if a["commit"] == "b")
        # +10% memory is an increase and NOT good
        assert entry["type"] == "increase"
        assert entry["good"] is False
        assert entry["pr"] == 2001

    def test_empty_sources_produce_valid_file(self, tmp_path):
        output = tmp_path / "notable.json"
        export_notable([], output, platform="test-platform")
        data = json.loads(output.read_text())
        assert data["annotations"] == []
        assert data["metadata"]["platform"] == "test-platform"

    def test_source_without_pinpointed_changes_contributes_nothing(self, tmp_path):
        state = SweepState(
            merge_commits=["a", "b"],
            commit_dates={"a": "2024-01-01", "b": "2024-02-01"},
        )
        # Only one completed point — no adjacent pair, no annotations
        state.points["a"] = BenchmarkPoint(commit="a", date="2024-01-01", value=100.0, status=PointStatus.COMPLETED)
        output = tmp_path / "notable.json"
        export_notable([NotableSource(state=state, workload="w", metric="throughput")], output, platform="p")
        data = json.loads(output.read_text())
        assert data["annotations"] == []
