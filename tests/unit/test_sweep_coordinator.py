"""Unit tests for the sweep coordinator."""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepPlanner, SweepState
from src.sweep.coordinator import SweepCoordinator, SWEEP_TEST, SWEEP_VAL_SIZE, SWEEP_IO_THREADS


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
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test.com"],
                   capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                   capture_output=True, check=True)
    # Create initial commit
    (repo / "file.txt").write_text("hello")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Initial"],
                   capture_output=True, check=True)
    return repo


class TestSweepCoordinatorInit:
    """Tests for SweepCoordinator initialization."""

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("src.sweep.coordinator.get_merge_commits")
    def test_initialize_populates_commits(self, mock_commits, mock_state_file, tmp_dir):
        from src.sweep.git_ops import MergeCommit
        mock_state_file.__class__ = Path
        state_file = tmp_dir / "state.json"

        mock_commits.return_value = [
            MergeCommit(hash="aaa111", date="2024-03-20", pr=100, pr_title="First PR"),
            MergeCommit(hash="bbb222", date="2024-04-15", pr=200, pr_title="Second PR"),
        ]

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file), \
             patch("src.sweep.coordinator.get_release_branch_points", return_value=[]):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()

        assert len(coordinator.state.merge_commits) == 2
        assert coordinator.state.merge_commits[0] == "aaa111"
        assert coordinator.state.commit_dates["bbb222"] == "2024-04-15"

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    def test_initialize_skips_if_already_populated(self, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        # Pre-populate state
        state = SweepState(merge_commits=["abc", "def"], commit_dates={"abc": "2024-01-01", "def": "2024-02-01"})
        state.save(state_file)

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file), \
             patch("src.sweep.coordinator.get_merge_commits") as mock_git:
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            # Should NOT call git since commits already populated
            mock_git.assert_not_called()


class TestSweepCoordinatorTaskGeneration:
    """Tests for sweep task generation."""

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("src.sweep.coordinator.get_head")
    def test_generates_task_for_new_head(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb", "ccc"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01", "ccc": "2024-03-01"},
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=100000, cv=0.2, status=PointStatus.COMPLETED
        )
        state.save(state_file)
        mock_head.return_value = "ccc"

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file), \
             patch("src.task_queue.config.REPO_NAMES", ["valkey", "valkey-rainfall"]):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            task = coordinator.get_next_sweep_task()

        assert task is not None
        assert task.specifier == "ccc"
        assert task.test == SWEEP_TEST
        assert task.val_size == SWEEP_VAL_SIZE
        assert task.io_threads == SWEEP_IO_THREADS
        assert "[sweep]" in task.note

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("src.sweep.coordinator.get_head")
    def test_returns_none_when_all_done(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01"},
        )
        state.points["aaa"] = BenchmarkPoint(
            commit="aaa", date="2024-01-01", value=100000, cv=0.2, status=PointStatus.COMPLETED
        )
        state.points["bbb"] = BenchmarkPoint(
            commit="bbb", date="2024-02-01", value=100000, cv=0.2, status=PointStatus.COMPLETED
        )
        state.save(state_file)
        mock_head.return_value = "bbb"
        state.last_benchmarked_head = "bbb"
        state.save(state_file)

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            task = coordinator.get_next_sweep_task()

        assert task is None


class TestSweepCoordinatorResults:
    """Tests for recording results and cleanup."""

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    @patch("src.sweep.coordinator.get_head")
    def test_record_result_persists(self, mock_head, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa", "bbb"],
            commit_dates={"aaa": "2024-01-01", "bbb": "2024-02-01"},
        )
        state.save(state_file)
        mock_head.return_value = "bbb"

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            coordinator.record_result("aaa", value=150000, cv=0.19, reps=3)

        # Reload and verify
        loaded = SweepState.load(state_file)
        assert "aaa" in loaded.points
        assert loaded.points["aaa"].value == 150000
        assert loaded.points["aaa"].status == PointStatus.COMPLETED

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    def test_record_build_failure_persists(self, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(
            merge_commits=["aaa"],
            commit_dates={"aaa": "2024-01-01"},
        )
        state.save(state_file)

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            coordinator.record_build_failure("aaa")

        loaded = SweepState.load(state_file)
        assert loaded.points["aaa"].status == PointStatus.BUILD_FAILED

    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    def test_delete_cached_binary(self, mock_state_file, tmp_dir):
        state_file = tmp_dir / "state.json"
        state = SweepState(merge_commits=["abc123"], commit_dates={"abc123": "2024-01-01"})
        state.save(state_file)

        # Create a fake cache dir
        cache_dir = tmp_dir / "build_cache" / "valkey" / "abc123"
        cache_dir.mkdir(parents=True)
        (cache_dir / "valkey-server").write_text("fake binary")

        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file), \
             patch("pathlib.Path.home", return_value=tmp_dir):
            coordinator = SweepCoordinator(tmp_dir / "repo")
            coordinator.initialize()
            coordinator.delete_cached_binary("abc123")

        assert not cache_dir.exists()


class TestTaskRunnerSweepIntegration:
    """Tests for TaskRunner with sweep mode."""

    def test_task_runner_accepts_sweep_flag(self):
        """Verify TaskRunner can be instantiated with sweep=False without errors."""
        from src.task_runner import TaskRunner
        runner = TaskRunner(sweep=False)
        assert runner._subscribers == []

    @patch("src.sweep.coordinator.SweepCoordinator.initialize")
    @patch("src.sweep.coordinator.get_merge_commits", return_value=[])
    @patch("src.sweep.coordinator.get_release_branch_points", return_value=[])
    @patch("src.sweep.coordinator.SWEEP_STATE_FILE")
    def test_task_runner_sweep_mode_creates_sweep_coordinator(
        self, mock_state, mock_tags, mock_commits, mock_init, tmp_dir
    ):
        from src.task_runner import TaskRunner
        state_file = tmp_dir / "state.json"
        with patch("src.sweep.coordinator.SWEEP_STATE_FILE", state_file):
            runner = TaskRunner(sweep=True, repo_path=tmp_dir)
        assert len(runner._subscribers) == 1
