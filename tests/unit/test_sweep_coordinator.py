"""Unit tests for the sweep coordinator."""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from conductress.sweep.coordinator import SWEEP_IO_THREADS, SWEEP_TEST, SWEEP_VAL_SIZE, SweepCoordinator
from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepPlanner, SweepState


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_repo(tmp_dir):
    """Create a minimal git repo for testing."""
    import subprocess

    repo = tmp_dir / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
    )
    # Create initial commit
    (repo / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Initial"],
        capture_output=True,
        check=True,
    )
    return repo


class TestSweepCoordinatorInit:
    """Tests for SweepCoordinator initialization."""

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("conductress.sweep.coordinator.get_merge_commits")
    def test_initialize_populates_commits(self, mock_commits, mock_state_file, tmp_dir):
        from conductress.sweep.git_ops import MergeCommit

        mock_state_file.__class__ = Path
        state_file = tmp_dir / "state.json"

        mock_commits.return_value = [
            MergeCommit(hash="aaa111", date="2024-03-20", pr=100, pr_title="First PR"),
            MergeCommit(hash="bbb222", date="2024-04-15", pr=200, pr_title="Second PR"),
        ]

        with (
            patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file),
            patch("conductress.sweep.coordinator.get_release_branch_points", return_value=[]),
        ):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()

        assert len(coordinator.state.merge_commits) == 2
        assert coordinator.state.merge_commits[0] == "aaa111"
        assert coordinator.state.commit_dates["bbb222"] == "2024-04-15"

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    def test_initialize_skips_if_already_populated(self, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        # Pre-populate state
        state = SweepState(
            merge_commits=["abc", "def"],
            commit_dates={"abc": "2024-01-01", "def": "2024-02-01"},
        )
        state.save(state_file)

        with (
            patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file),
            patch("conductress.sweep.coordinator.get_merge_commits") as mock_git,
        ):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            # Should NOT call git since commits already populated
            mock_git.assert_not_called()


class TestSweepCoordinatorTaskGeneration:
    """Tests for sweep task generation."""

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("conductress.sweep.coordinator.get_head")
    def test_generates_task_for_new_head(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={
                "aaa": "2024-01-01",
                "bbb": "2024-02-01",
                "ccc": "2024-03-01",
            },
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=100000,
            cv=0.2,
            status=PointStatus.COMPLETED,
        )
        state.save(state_file)
        mock_head.return_value = "ccc"

        with (
            patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file),
            patch("conductress.task_queue.config.REPO_NAMES", ["valkey", "valkey-rainfall"]),
        ):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            task = coordinator.get_next_sweep_task()

        assert task is not None
        assert task.specifier == "ccc"
        assert task.test == SWEEP_TEST
        assert task.val_size == SWEEP_VAL_SIZE
        assert task.io_threads == SWEEP_IO_THREADS
        assert "[sweep]" in task.note

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("conductress.sweep.coordinator.get_head")
    def test_returns_none_when_all_done(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01"},
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa",
            date="2024-01-01",
            value=100000,
            cv=0.2,
            status=PointStatus.COMPLETED,
        )
        state.points["bbb"] = BenchmarkPoint(
            commit="bbb",
            date="2024-02-01",
            value=100000,
            cv=0.2,
            status=PointStatus.COMPLETED,
        )
        state.save(state_file)
        mock_head.return_value = "bbb"
        state.last_benchmarked_head = "bbb"
        state.save(state_file)

        with patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            task = coordinator.get_next_sweep_task()

        assert task is None


class TestSweepCoordinatorResults:
    """Tests for recording results and cleanup."""

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("conductress.sweep.coordinator.get_head")
    def test_record_result_persists(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01"},
        )
        state.save(state_file)
        mock_head.return_value = "bbb"

        with patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            coordinator.record_result("aaa", value=150000, cv=0.19, reps=3)

        # Reload and verify
        loaded = SweepState.load(state_file)
        assert "aaa" in loaded.points
        assert loaded.points["aaa"].value == 150000
        assert loaded.points["aaa"].status == PointStatus.COMPLETED

    @patch("conductress.sweep.coordinator.SWEEP_STATE_FILE")
    def test_record_build_failure_persists(self, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa"],
            commit_dates={"aaa": "2024-01-01"},
        )
        state.save(state_file)

        with patch("conductress.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            coordinator.record_build_failure("aaa")

        loaded = SweepState.load(state_file)
        assert loaded.points["aaa"].status == PointStatus.BUILD_FAILED


class TestTaskRunnerSweepIntegration:
    """Tests for TaskRunner with sweep mode."""

    def test_task_runner_accepts_sweep_flag(self):
        """Verify TaskRunner can be instantiated with sweep=False without errors."""
        from conductress.task_runner import TaskRunner

        runner = TaskRunner(sweep=False)
        assert runner._subscribers == []

    @patch("conductress.sweep.coordinator.SweepCoordinator.initialize")
    @patch("conductress.sweep.coordinator.get_merge_commits", return_value=[])
    @patch("conductress.sweep.coordinator.get_release_branch_points", return_value=[])
    def test_task_runner_sweep_mode_creates_sweep_coordinator(self, mock_tags, mock_commits, mock_init, tmp_dir):
        from conductress.task_runner import TaskRunner

        runner = TaskRunner(sweep=True, repo_path=tmp_dir)
        # throughput + latency + 4 memory = 6
        assert len(runner._subscribers) == 7


class TestUrgencyScore:
    """Tests for get_urgency_score priority scheduling."""

    def test_empty_series_returns_infinity(self, tmp_dir):
        """A series with <2 points should have infinite urgency (top priority)."""
        state_file = tmp_dir / "state.json"
        state = SweepState(threshold=0.02)
        state.merge_commits = ["aaa", "bbb", "ccc"]
        state.commit_dates = {"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"}
        state.save(state_file)

        with patch.object(SweepCoordinator, "initialize"):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.state = state
            coord.state_file = state_file
            coord.planner = SweepPlanner(state)

        score = coord.get_urgency_score()
        assert score == float("inf")

    def test_fully_resolved_returns_zero(self, tmp_dir):
        """A series with no work left should return 0."""
        state_file = tmp_dir / "state.json"
        state = SweepState(threshold=0.02)
        state.merge_commits = ["aaa", "bbb"]
        state.commit_dates = {"aaa": "2024-01-01", "bbb": "2024-02-01"}
        state.points = {
            "aaa": BenchmarkPoint(commit="aaa", date="2024-01-01", value=1000.0, cv=0.5, reps=3),
            "bbb": BenchmarkPoint(commit="bbb", date="2024-02-01", value=1010.0, cv=0.5, reps=3),
        }
        state.save(state_file)

        with patch.object(SweepCoordinator, "initialize"):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.state = state
            coord.state_file = state_file
            coord.planner = SweepPlanner(state)

        score = coord.get_urgency_score()
        assert score == 0.0

    def test_large_delta_scores_higher(self, tmp_dir):
        """A segment with a large delta should score higher than a small one."""
        state_file = tmp_dir / "state.json"
        state = SweepState(threshold=0.02)
        commits = [f"c{i:03d}" for i in range(20)]
        state.merge_commits = commits
        state.commit_dates = {c: f"2024-01-{i+1:02d}" for i, c in enumerate(commits)}
        # 10% jump in the middle
        state.points = {
            commits[0]: BenchmarkPoint(commit=commits[0], date="2024-01-01", value=1000.0, cv=0.5, reps=3),
            commits[10]: BenchmarkPoint(commit=commits[10], date="2024-01-11", value=1100.0, cv=0.5, reps=3),
            commits[19]: BenchmarkPoint(commit=commits[19], date="2024-01-20", value=1100.0, cv=0.5, reps=3),
        }
        state.save(state_file)

        with patch.object(SweepCoordinator, "initialize"):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.state = state
            coord.state_file = state_file
            coord.planner = SweepPlanner(state)

        score = coord.get_urgency_score()
        assert score > 0


class TestFetchAndRefresh:
    """Tests for _fetch_and_refresh stale commit detection."""

    def test_refreshes_when_head_not_in_commit_list(self, tmp_dir):
        """Regression: stale state with HEAD not in merge_commits must trigger refresh."""
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        )
        state.save(state_file)

        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.repo_path = tmp_dir
            coord.state_file = state_file
            coord.state = state
            coord.planner = SweepPlanner(state)
            coord._last_fetch_time = 0.0

        new_head = "ddd"  # Not in merge_commits

        with (
            patch("conductress.sweep.coordinator.get_head", return_value=new_head),
            patch("conductress.sweep.coordinator.fetch_ref"),
            patch.object(coord, "_refresh_commits") as mock_refresh,
        ):
            coord._fetch_and_refresh()
            mock_refresh.assert_called_once()

    def test_skips_refresh_when_head_in_commit_list_and_unchanged(self, tmp_dir):
        """No refresh needed when HEAD is already in the commit list and didn't move."""
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        )
        state.save(state_file)

        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.repo_path = tmp_dir
            coord.state_file = state_file
            coord.state = state
            coord.planner = SweepPlanner(state)
            coord._last_fetch_time = 0.0

        with (
            patch("conductress.sweep.coordinator.get_head", return_value="ccc"),
            patch("conductress.sweep.coordinator.fetch_ref"),
            patch.object(coord, "_refresh_commits") as mock_refresh,
        ):
            coord._fetch_and_refresh()
            mock_refresh.assert_not_called()
