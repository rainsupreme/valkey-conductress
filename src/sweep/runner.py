"""Sweep runner: integrates the sweep planner with the Conductress task runner.

When --sweep is active, the runner checks the planner for work whenever the
manual queue is empty. After each sweep task completes, results are recorded
back into the planner state and the cached binary is deleted.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT
from src.sweep.git_ops import MergeCommit, get_head, get_merge_commits, get_release_tags
from src.sweep.planner import Landmark, SweepPlanner, SweepState, SweepTask
from src.sweep.state import load_state, save_state
from src.tasks.task_perf_benchmark import PerfTaskData

logger = logging.getLogger(__name__)

# Sweep configuration
SWEEP_STATE_DIR = PROJECT_ROOT / "sweep_data"
SWEEP_STATE_FILE = SWEEP_STATE_DIR / "state.json"

# Default sweep benchmark parameters (GET 16B, single-threaded, pipeline=1)
SWEEP_SOURCE = "valkey"
SWEEP_TEST = "get"
SWEEP_VAL_SIZE = 16
SWEEP_IO_THREADS = 1
SWEEP_PIPELINING = 1
SWEEP_WARMUP = 5
SWEEP_DURATION = 30
SWEEP_REPETITIONS = 3
SWEEP_MAKE_ARGS = "USE_FAST_FLOAT=yes"


class SweepRunner:
    """Manages sweep state and generates tasks for the runner."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.state = load_state(SWEEP_STATE_FILE)
        self.planner = SweepPlanner(self.state)

    def initialize(self) -> None:
        """Initialize or update the merge commit list from git history."""
        if not self.state.merge_commits:
            logger.info("Initializing sweep: enumerating merge commits...")
            self._populate_commits()
            self._populate_landmarks()
            save_state(self.state, SWEEP_STATE_FILE)
            # Rebuild planner index after populating
            self.planner = SweepPlanner(self.state)
            logger.info("Sweep initialized: %d merge commits, %d landmarks",
                        len(self.state.merge_commits), len(self.state.landmarks))

    def get_next_sweep_task(self) -> Optional[PerfTaskData]:
        """Get the next sweep task as a PerfTaskData ready for the runner.

        Returns None if no sweep work is available.
        """
        # Check for new HEAD
        try:
            current_head = get_head(self.repo_path)
        except Exception as e:
            logger.warning("Failed to get HEAD: %s", e)
            current_head = None

        sweep_task = self.planner.get_next_task(current_head)
        if sweep_task is None:
            return None

        logger.info("Sweep task: %s (%s)", sweep_task.commit[:8], sweep_task.reason)
        return self._sweep_task_to_perf_task(sweep_task)

    def record_result(self, commit: str, rps: float, cv: float, reps: int) -> None:
        """Record a completed benchmark result and persist state."""
        # Try to get PR info from the commit's merge message
        pr = None
        pr_title = None
        for mc in _find_commit_in_list(self.state.merge_commits, self.state, commit):
            pr = mc.pr
            pr_title = mc.pr_title
            break

        self.planner.record_result(commit, rps, cv, reps, pr=pr, pr_title=pr_title)

        # Update last_benchmarked_head if this was HEAD
        try:
            head = get_head(self.repo_path)
            if commit == head:
                self.state.last_benchmarked_head = commit
        except Exception:
            pass

        save_state(self.state, SWEEP_STATE_FILE)
        logger.info("Sweep result recorded: %s -> %.0f rps (CV %.2f%%)", commit[:8], rps, cv)

    def record_build_failure(self, commit: str) -> None:
        """Record a build failure and persist state."""
        self.planner.record_build_failure(commit)
        save_state(self.state, SWEEP_STATE_FILE)
        logger.info("Sweep build failure: %s", commit[:8])

    def delete_cached_binary(self, commit: str) -> None:
        """Delete the cached binary for a sweep task to save disk space."""
        # Build cache is at ~/build_cache/{source}/{commit_hash}/
        cache_base = Path.home() / "build_cache" / SWEEP_SOURCE
        commit_dir = cache_base / commit
        if commit_dir.exists():
            shutil.rmtree(commit_dir)
            logger.info("Deleted cached build: %s", commit_dir)

    def _populate_commits(self) -> None:
        """Populate merge_commits from git history."""
        commits = get_merge_commits(self.repo_path)
        self.state.merge_commits = [c.hash for c in commits]
        self.state.commit_dates = {c.hash: c.date for c in commits}
        # Store PR info for later annotation
        for c in commits:
            if c.pr is not None:
                # We'll use this when recording results
                pass

    def _populate_landmarks(self) -> None:
        """Populate landmarks from release tags."""
        try:
            tags = get_release_tags(self.repo_path)
            commit_set = set(self.state.merge_commits)
            for commit_hash, date, tag_name in tags:
                if commit_hash in commit_set:
                    self.state.landmarks.append(
                        Landmark(commit=commit_hash, date=date, label=tag_name)
                    )
        except Exception as e:
            logger.warning("Failed to enumerate release tags: %s", e)

    def _sweep_task_to_perf_task(self, task: SweepTask) -> PerfTaskData:
        """Convert a SweepTask to a PerfTaskData for the runner."""
        return PerfTaskData(
            source=SWEEP_SOURCE,
            specifier=task.commit,
            make_args=SWEEP_MAKE_ARGS,
            replicas=0,
            note=f"[sweep] {task.reason}",
            requirements={},
            test=SWEEP_TEST,
            val_size=SWEEP_VAL_SIZE,
            io_threads=SWEEP_IO_THREADS,
            pipelining=SWEEP_PIPELINING,
            warmup=SWEEP_WARMUP,
            duration=SWEEP_DURATION,
            profiling_sample_rate=0,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            key_size=0,
            repetitions=SWEEP_REPETITIONS,
        )


def _find_commit_in_list(merge_commits: list, state: SweepState,
                         commit: str) -> list:
    """Find PR info for a commit by re-parsing (lightweight helper)."""
    from src.sweep.git_ops import _parse_merge_subject
    # We don't store full MergeCommit objects in state, so return empty
    # PR info will be populated during git_ops enumeration
    return []
