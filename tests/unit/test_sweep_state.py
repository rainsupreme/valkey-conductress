"""Unit tests for sweep state persistence (SweepState.save/load)."""

import json
import tempfile
from pathlib import Path

import pytest

from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepState


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestStatePersistence:
    """Tests for save/load round-trip."""

    def test_empty_state_round_trip(self, tmp_path):
        path = tmp_path / "state.json"
        state = SweepState()
        state.save(path)
        loaded = SweepState.load(path)
        assert loaded.threshold == 0.02
        assert loaded.merge_commits == []
        assert loaded.points == {}

    def test_full_state_round_trip(self, tmp_path):
        path = tmp_path / "state.json"
        state = SweepState(
            threshold=0.02,
            last_benchmarked_head="abc123",
            merge_commits=["a", "b", "c"],
            commit_dates={"a": "2024-01-01", "b": "2024-02-01", "c": "2024-03-01"},
            landmarks=[Landmark(commit="b", date="2024-02-01", label="8.0.0")],
        )
        state.points["a"] = BenchmarkPoint(
            commit="a",
            date="2024-01-01",
            value=150000,
            cv=0.19,
            reps=3,
            pr=1234,
            pr_title="Optimize dict",
            status=PointStatus.COMPLETED,
        )
        state.points["c"] = BenchmarkPoint(
            commit="c",
            date="2024-03-01",
            status=PointStatus.BUILD_FAILED,
        )

        state.save(path)
        loaded = SweepState.load(path)

        assert loaded.threshold == 0.02
        assert loaded.last_benchmarked_head == "abc123"
        assert loaded.merge_commits == ["a", "b", "c"]
        assert loaded.commit_dates == state.commit_dates
        assert len(loaded.landmarks) == 1
        assert loaded.landmarks[0].label == "8.0.0"
        assert loaded.points["a"].value == 150000
        assert loaded.points["a"].pr == 1234
        assert loaded.points["a"].status == PointStatus.COMPLETED
        assert loaded.points["c"].status == PointStatus.BUILD_FAILED

    def test_cpu_stacks_round_trip(self, tmp_path):
        """Regression: cpu_stacks_main/io must survive save+load.

        Bug found Jun 20 2026: save() and load() omitted these fields, so CPU
        profile stacks were set in memory by the coordinator but silently
        dropped on the next state write — state files had 0 CPU points despite
        active collection, and export_cpu_profile() produced nothing.
        """
        path = tmp_path / "state.json"
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2024-01-01"})
        main_stacks = [["valkey-server;main;processCommand;getCommand", 100]]
        io_stacks = [["io_thd_1;IOThreadMain;IOJobQueue_availableJobs", 500]]
        state.points["a"] = BenchmarkPoint(
            commit="a",
            date="2024-01-01",
            value=2_000_000,
            status=PointStatus.COMPLETED,
            cpu_stacks_main=main_stacks,
            cpu_stacks_io=io_stacks,
        )

        state.save(path)
        loaded = SweepState.load(path)

        assert loaded.points["a"].cpu_stacks_main == main_stacks
        assert loaded.points["a"].cpu_stacks_io == io_stacks

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        state = SweepState.load(path)
        assert state.merge_commits == []
        assert state.points == {}

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "state.json"
        state = SweepState(merge_commits=["a"])
        state.save(path)
        assert path.exists()

    def test_json_is_human_readable(self, tmp_path):
        path = tmp_path / "state.json"
        state = SweepState(merge_commits=["abc123"])
        state.save(path)
        content = path.read_text()
        data = json.loads(content)
        assert "merge_commits" in data
        # Check it's indented (human-readable)
        assert "\n" in content
