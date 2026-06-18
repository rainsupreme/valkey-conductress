"""Latency sweep coordinator: measures per-request latency at a flat 100K rps.

Independent of throughput sweep -- operates on full commit history.
Bisects on p99 latency with 10% threshold.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from conductress.config import (
    CONDUCTRESS_RESULTS,
    LATENCY_MAKE_ARGS,
    LATENCY_STATE_FILE,
    LATENCY_TARGET_RPS,
    SWEEP_IO_THREADS,
    SWEEP_SOURCE,
)
from conductress.sweep.coordinator import BaseSweepCoordinator
from conductress.sweep.planner import SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_latency import LATENCY_REPS, LatencyTaskData

logger = logging.getLogger(__name__)


class LatencySweepCoordinator(BaseSweepCoordinator):
    """Latency sweep: measures p99 latency at a flat 100K rps across history.

    Key differences from throughput coordinator:
    - Uses flat rate (no throughput dependency)
    - Uses p99 as bisection metric (lower is better)
    - Priority dampened by 0.5x (fills in behind throughput)
    """

    metric_unit = "µs"
    lower_is_better = True

    def __init__(self, repo_path: Path):
        super().__init__(repo_path, LATENCY_STATE_FILE)

    @property
    def metric_id(self) -> str:  # type: ignore[override]
        return "latency"

    @property
    def workload_id(self) -> str:  # type: ignore[override]
        return "get-k16-v16"

    def get_urgency_score(self) -> float:
        """Priority score, dampened by 0.5x relative to throughput."""
        completed = sum(1 for p in self.state.points.values() if p.value is not None)
        if completed < 2:
            return float("inf")  # New series, top priority

        base = super().get_urgency_score()
        return base * 0.5

    def _create_task(self, sweep_task: SweepTask) -> LatencyTaskData:
        return LatencyTaskData(
            source=SWEEP_SOURCE,
            specifier=sweep_task.commit,
            make_args=LATENCY_MAKE_ARGS,
            replicas=0,
            note=f"[latency-sweep] {sweep_task.reason} (100K rps flat)",
            requirements={},
            target_rps=LATENCY_TARGET_RPS,
            io_threads=SWEEP_IO_THREADS,
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
                    per_rep = data.get("per_rep_p99", [])
                    if len(per_rep) >= 2 and score and score > 0:
                        from statistics import mean, stdev

                        cv = (stdev(per_rep) / mean(per_rep)) * 100
                    else:
                        cv = 0.0
                    if score and score > 0:
                        return (score, cv, reps)
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def on_task_completed(self, task: BaseTaskData) -> None:
        """Store full latency data (all percentiles + histogram) on the point."""
        if not self._is_my_task(task):
            return

        result = self._extract_result(task)
        if result:
            value, cv, reps = result
            self.record_result(task.sweep_commit, value, cv, reps)  # type: ignore[attr-defined]

            # Store full latency data for export
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
            target_rps=LATENCY_TARGET_RPS,
        )
