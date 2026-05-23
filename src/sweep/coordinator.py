"""Sweep coordinators: bridge the sweep planner with the Conductress task runner.

BaseSweepCoordinator handles git history, state management, queue interaction,
and the pub/sub protocol. Subclasses define task creation and result extraction
for specific metrics (throughput, memory, etc.).
"""

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.config import CONDUCTRESS_RESULTS, PROJECT_ROOT
from src.sweep.git_ops import get_head, get_merge_commits, get_release_branch_points
from src.sweep.planner import Landmark, SweepPlanner, SweepState, SweepTask
from src.task_queue import BaseTaskData, TaskQueue
from src.tasks.task_perf_benchmark import PerfTaskData

logger = logging.getLogger(__name__)

# Git configuration (shared across all sweep types)
SWEEP_SOURCE = "valkey"
SWEEP_REF = "origin/unstable"


class BaseSweepCoordinator(ABC):
    """Abstract base for sweep coordinators.

    Handles: git history enumeration, state persistence, queue management,
    TaskSubscriber protocol (on_task_completed, on_task_failed, on_queue_empty).

    Subclasses define: task creation, result extraction, task filtering.
    """

    def __init__(self, repo_path: Path, state_file: Path):
        self.repo_path = repo_path
        self.state_file = state_file
        self.state = SweepState.load(state_file)
        self.planner = SweepPlanner(self.state)

    def initialize(self) -> None:
        """Initialize or update the merge commit list from git history."""
        if not self.state.merge_commits:
            logger.info("Initializing sweep: enumerating merge commits...")
            self._populate_commits()
            self._populate_landmarks()
            self.state.save(self.state_file)
            self.planner = SweepPlanner(self.state)
            logger.info(
                "Sweep initialized: %d merge commits, %d landmarks",
                len(self.state.merge_commits),
                len(self.state.landmarks),
            )

    def record_result(self, commit: str, value: float, cv: float, reps: int) -> None:
        """Record a completed benchmark result and persist state."""
        self.planner.record_result(commit, value, cv, reps)
        try:
            head = get_head(self.repo_path, ref=SWEEP_REF)
            if commit == head:
                self.state.last_benchmarked_head = commit
        except Exception:
            pass
        self.state.save(self.state_file)
        logger.info("Sweep result recorded: %s -> %.0f (CV %.2f%%)", commit[:8], value, cv)

    def record_build_failure(self, commit: str) -> None:
        """Record a build failure and persist state."""
        self.planner.record_build_failure(commit)
        self.state.save(self.state_file)
        logger.info("Sweep build failure: %s", commit[:8])

    def delete_cached_binary(self, commit: str) -> None:
        """Delete the cached binary for a sweep task to save disk space."""
        cache_base = Path.home() / "build_cache" / SWEEP_SOURCE
        commit_dir = cache_base / commit
        if commit_dir.exists():
            shutil.rmtree(commit_dir)
            logger.info("Deleted cached build: %s", commit_dir)

    # --- TaskSubscriber protocol ---

    def on_queue_empty(self) -> None:
        """Called when the task queue is empty."""
        self.queue_next_if_needed()

    def on_task_completed(self, task: BaseTaskData) -> None:
        """Called on every task completion. Filters to own tasks."""
        if not self._is_my_task(task):
            return
        result = self._extract_result(task)
        if result:
            value, cv, reps = result
            self.record_result(task.sweep_commit, value, cv, reps)  # type: ignore[attr-defined]
            self.delete_cached_binary(task.sweep_commit)  # type: ignore[attr-defined]
        else:
            commit = getattr(task, "sweep_commit", "?")
            logger.warning("Could not extract result for sweep commit %s", commit[:8])
        self.queue_next_if_needed()

    def on_task_failed(self, task: BaseTaskData) -> None:
        """Called on every task failure. Filters to own tasks."""
        if not self._is_my_task(task):
            return
        commit = getattr(task, "sweep_commit", "")
        self.record_build_failure(commit)
        self.delete_cached_binary(commit)
        self.queue_next_if_needed()

    def queue_next_if_needed(self) -> bool:
        """Queue the next sweep task if none is already pending."""
        queue = TaskQueue()
        for queued in queue.get_all_tasks():
            if self._is_my_task(queued):
                return False

        sweep_task = self._get_next_task()
        if sweep_task is None:
            return False

        task = self._create_task(sweep_task)
        task.sweep_commit = sweep_task.commit  # type: ignore[attr-defined]
        queue.submit_task(task)
        print(f"[sweep] Queued: {sweep_task.commit[:8]} - {sweep_task.reason}")
        return True

    # --- Abstract methods (subclass defines) ---

    # --- Abstract properties (subclass defines) ---

    @property
    @abstractmethod
    def metric_id(self) -> str:
        """Identifier for this metric (e.g. 'throughput', 'memory')."""
        ...

    @property
    @abstractmethod
    def metric_unit(self) -> str:
        """Display unit (e.g. 'ops/sec', 'bytes/item')."""
        ...

    # --- Export ---

    def export(self, output_path: Path, platform: str) -> int:
        """Export this coordinator's data to a series JSON file. Returns point count."""
        from src.sweep.exporter import export_series

        export_series(self.state, output_path, platform=platform, workload=self.metric_id)
        return sum(1 for p in self.state.points.values() if p.value is not None)

    # --- Abstract methods (subclass defines) ---

    @abstractmethod
    def _create_task(self, sweep_task: SweepTask) -> BaseTaskData:
        """Create a concrete task from a SweepTask."""
        ...

    @abstractmethod
    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        """Extract (value, cv, reps) from a completed task. Returns None on failure."""
        ...

    @abstractmethod
    def _is_my_task(self, task: BaseTaskData) -> bool:
        """Return True if this task belongs to this coordinator."""
        ...

    # --- Private helpers ---

    def _get_next_task(self) -> Optional[SweepTask]:
        """Get the next sweep task from the planner."""
        try:
            current_head = get_head(self.repo_path, ref=SWEEP_REF)
        except Exception as e:
            logger.warning("Failed to get HEAD: %s", e)
            current_head = None
        return self.planner.get_next_task(current_head)

    def _populate_commits(self) -> None:
        """Populate merge_commits from git history (Valkey-era only)."""
        from src.sweep.git_ops import find_fork_point

        fork_point = find_fork_point(self.repo_path)
        commits = get_merge_commits(self.repo_path, since_commit=fork_point, ref=SWEEP_REF)
        self.state.merge_commits = [c.hash for c in commits]
        self.state.commit_dates = {c.hash: c.date for c in commits}
        self.state.commit_prs = {c.hash: c.pr for c in commits if c.pr is not None}
        self.state.commit_titles = {c.hash: c.pr_title for c in commits if c.pr_title is not None}

    def _populate_landmarks(self) -> None:
        """Populate landmarks from release branch points on unstable."""
        PRE_FORK_LANDMARKS = [
            Landmark(
                commit="2b8cde71bb553713cf93794a0fb30a1618c0c955",
                date="2023-08-16",
                label="Valkey created",
            ),
            Landmark(
                commit="f7b1d0287d62ec9fac72bf14cf789e350d14e52b",
                date="2024-01-09",
                label="7.2.4",
            ),
        ]
        try:
            points = get_release_branch_points(self.repo_path)
            commit_set = set(self.state.merge_commits)
            for lm in PRE_FORK_LANDMARKS:
                if lm.commit in commit_set:
                    self.state.landmarks.append(lm)
            for commit_hash, date, label in points:
                if commit_hash in commit_set:
                    self.state.landmarks.append(Landmark(commit=commit_hash, date=date, label=label))
        except Exception as e:
            logger.warning("Failed to enumerate release branch points: %s", e)


# =============================================================================
# Concrete implementation: throughput sweep
# =============================================================================

SWEEP_STATE_DIR = PROJECT_ROOT / "sweep_data"
SWEEP_STATE_FILE = SWEEP_STATE_DIR / "state.json"

SWEEP_TEST = "get"
SWEEP_VAL_SIZE = 16
SWEEP_IO_THREADS = 7
SWEEP_PIPELINING = 10
SWEEP_WARMUP = 5
SWEEP_DURATION = 30
SWEEP_REPETITIONS = 3
SWEEP_MAX_REPS = 7
SWEEP_TARGET_CV = 0.5
SWEEP_MAKE_ARGS = "USE_FAST_FLOAT=yes"


class SweepCoordinator(BaseSweepCoordinator):
    """Throughput sweep coordinator (GET 16B, io-threads=7, P=10)."""

    metric_id = "throughput"
    metric_unit = "ops/sec"

    def __init__(self, repo_path: Path):
        super().__init__(repo_path, SWEEP_STATE_FILE)

    def get_next_sweep_task(self) -> Optional[PerfTaskData]:
        """Legacy interface: get next task directly."""
        sweep_task = self._get_next_task()
        if sweep_task is None:
            return None
        logger.info("Sweep task: %s (%s)", sweep_task.commit[:8], sweep_task.reason)
        return self._create_task(sweep_task)

    def _create_task(self, sweep_task: SweepTask) -> PerfTaskData:
        return PerfTaskData(
            source=SWEEP_SOURCE,
            specifier=sweep_task.commit,
            make_args=SWEEP_MAKE_ARGS,
            replicas=0,
            note=f"[sweep] {sweep_task.reason}",
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
            max_reps=SWEEP_MAX_REPS,
            target_cv=SWEEP_TARGET_CV,
        )

    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        import json as _json
        from statistics import stdev

        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None

        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = _json.loads(line)
                if entry.get("task_id") == task.task_id:
                    rps = entry.get("score")
                    per_run = entry.get("data", {}).get("per_run_rps", [])
                    cv = (stdev(per_run) / rps) * 100 if len(per_run) >= 2 and rps else 0.0
                    reps = len(per_run) if per_run else 3
                    return (rps, cv, reps) if rps else None
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return isinstance(task, PerfTaskData) and bool(task.sweep_commit)
