"""Latency sweep coordinator: measures per-request latency at 70% of max throughput.

Depends on throughput sweep data -- only queues tasks for commits that have
a completed throughput measurement. Bisects on p99 latency with 10% threshold.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

from conductress.config import (
    CONDUCTRESS_RESULTS,
    LATENCY_DETECTION_THRESHOLD,
    LATENCY_IO_THREADS,
    LATENCY_LOAD_FRACTION,
    LATENCY_MAKE_ARGS,
    LATENCY_STATE_FILE,
    SWEEP_SOURCE,
    SWEEP_STATE_DIR,
)
from conductress.sweep.coordinator import BaseSweepCoordinator
from conductress.sweep.planner import SweepState, SweepTask, TaskPriority
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_latency import LATENCY_PIPELINE, LATENCY_REPS, LATENCY_VAL_SIZE, LatencyTaskData

logger = logging.getLogger(__name__)


class LatencySweepCoordinator(BaseSweepCoordinator):
    """Latency sweep: measures p99 latency at 70% of max throughput across history.

    Key difference from throughput/memory coordinators:
    - Operates only on commits with completed throughput measurements
    - Computes target_rps from throughput result × load_fraction
    - Uses p99 as bisection metric (lower is better)
    - Priority dampened by 0.5x (fills in behind throughput)
    """

    metric_unit = "µs"
    lower_is_better = True

    def __init__(self, repo_path: Path, throughput_state_file: Path):
        self._throughput_state_file = throughput_state_file
        self._throughput_state: Optional[SweepState] = None
        super().__init__(repo_path, LATENCY_STATE_FILE)

    @property
    def metric_id(self) -> str:  # type: ignore[override]
        return "latency"

    @property
    def workload_id(self) -> str:  # type: ignore[override]
        return f"get{LATENCY_VAL_SIZE}b-t{LATENCY_IO_THREADS}-p{LATENCY_PIPELINE}"

    @property
    def throughput_state(self) -> SweepState:
        """Load throughput state (cached, refreshed on each queue_next cycle)."""
        if self._throughput_state is None:
            self._throughput_state = SweepState.load(self._throughput_state_file)
        return self._throughput_state

    def _refresh_throughput_state(self) -> None:
        """Reload throughput state from disk."""
        self._throughput_state = SweepState.load(self._throughput_state_file)

    def _get_candidate_commits(self) -> list[str]:
        """Only commits with completed throughput measurements."""
        commit_order = {c: i for i, c in enumerate(self.throughput_state.merge_commits)}
        return sorted(
            [c for c, p in self.throughput_state.points.items() if p.value is not None],
            key=lambda c: commit_order.get(c, 0),
        )

    def _get_throughput_for_commit(self, commit: str) -> Optional[float]:
        """Get the throughput (rps) for a commit from throughput state."""
        point = self.throughput_state.points.get(commit)
        if point and point.value is not None:
            return point.value
        return None

    def get_urgency_score(self) -> float:
        """Priority score, dampened by 0.5x relative to throughput.

        Returns 0 if no throughput data exists (can't run without it).
        """
        self._refresh_throughput_state()
        candidates = self._get_candidate_commits()
        if len(candidates) < 2:
            return 0.0  # Need at least 2 throughput points to bisect between

        completed = sum(1 for p in self.state.points.values() if p.value is not None)
        if completed < 2:
            return float("inf")  # New series, top priority

        # Normal bisection urgency, dampened
        base = super().get_urgency_score()
        return base * 0.5

    def _get_next_task(self) -> Optional[SweepTask]:
        """Override to select from throughput-measured commits only."""
        candidates = self._get_candidate_commits()
        if len(candidates) < 2:
            return None

        # Find the largest gap between adjacent measured commits that don't have latency data
        # or use the planner's normal bisection on the candidate subset
        task = self.planner.get_next_task(current_head=None)
        if task is None:
            return None

        # Verify the task's commit has throughput data
        if self._get_throughput_for_commit(task.commit) is None:
            # Skip commits without throughput -- find next candidate
            for commit in candidates:
                if commit not in self.state.points:
                    date = self.throughput_state.commit_dates.get(commit, "")
                    return SweepTask(
                        commit=commit,
                        date=date,
                        priority=TaskPriority.BACKFILL,
                        reason="backfill (has throughput data)",
                    )
            return None

        return task

    def _create_task(self, sweep_task: SweepTask) -> LatencyTaskData:
        throughput_rps = self._get_throughput_for_commit(sweep_task.commit)
        if throughput_rps is None:
            raise ValueError(f"No throughput data for {sweep_task.commit[:8]}")

        target_rps = int(throughput_rps * LATENCY_LOAD_FRACTION)

        return LatencyTaskData(
            source=SWEEP_SOURCE,
            specifier=sweep_task.commit,
            make_args=LATENCY_MAKE_ARGS,
            replicas=0,
            note=f"[latency-sweep] {sweep_task.reason} (target {target_rps} rps)",
            requirements={},
            target_rps=target_rps,
            load_fraction=LATENCY_LOAD_FRACTION,
            io_threads=LATENCY_IO_THREADS,
        )

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return isinstance(task, LatencyTaskData) and bool(getattr(task, "sweep_commit", ""))

    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        """Extract p99 latency as the primary metric for bisection."""
        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None

        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = json.loads(line)
                if entry.get("task_id") == task.task_id:
                    score = entry.get("score")  # p99_us
                    data = entry.get("data", {})
                    reps = data.get("reps", LATENCY_REPS)
                    # CV not meaningful for latency (inherent variance)
                    cv = 0.0
                    if score and score > 0:
                        return (score, cv, reps)
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def on_task_completed(self, task: BaseTaskData) -> None:
        """Override to store full latency data (all percentiles + histogram) on the point."""
        if not self._is_my_task(task):
            return

        result = self._extract_result(task)
        if result:
            value, cv, reps = result
            self.record_result(task.sweep_commit, value, cv, reps)  # type: ignore[attr-defined]

            # Store full latency data for export (p50, p99.9, histogram, rps)
            latency_data = self._extract_full_data(task)
            if latency_data and task.sweep_commit in self.state.points:  # type: ignore[attr-defined]
                self.state.points[task.sweep_commit].latency_data = latency_data  # type: ignore[attr-defined]
                self.state.save(self.state_file)
        else:
            commit = getattr(task, "sweep_commit", "?")
            logger.warning("Could not extract latency result for %s", commit[:8])
        self.queue_next_if_needed()

    def _extract_full_data(self, task: BaseTaskData) -> Optional[dict]:
        """Extract full latency data (all percentiles + histogram) from output.jsonl."""
        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None

        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = json.loads(line)
                if entry.get("task_id") == task.task_id:
                    return entry.get("data")
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def export(self, output_path: Path, platform: str) -> int:
        """Export latency data to a series JSON file. Returns point count."""
        from conductress.sweep.exporter import export_latency

        return export_latency(
            self.state,
            output_path,
            platform=platform,
            workload=self.workload_id,
            load_fraction=LATENCY_LOAD_FRACTION,
        )
