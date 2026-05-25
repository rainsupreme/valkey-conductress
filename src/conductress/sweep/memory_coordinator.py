"""Memory sweep coordinator: tracks per-item memory overhead across Valkey history."""

import logging
from pathlib import Path
from typing import Optional

from conductress.config import CONDUCTRESS_RESULTS, PROJECT_ROOT
from conductress.heap_profiler import JEMALLOC_PROF_MAKE_ARGS
from conductress.sweep.coordinator import (
    SWEEP_IO_THREADS,
    SWEEP_PIPELINING,
    SWEEP_SOURCE,
    SWEEP_TEST,
    SWEEP_VAL_SIZE,
    BaseSweepCoordinator,
)
from conductress.sweep.planner import SweepState, SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_mem_efficiency import MemTaskData

logger = logging.getLogger(__name__)

MEMORY_STATE_DIR = PROJECT_ROOT / "sweep_data"
MEMORY_STATE_FILE = MEMORY_STATE_DIR / "memory_state.json"

# Memory sweep parameters
MEMORY_TEST = "set"  # SET key value — most common pattern
MEMORY_VAL_SIZE = 64  # bytes


class MemorySweepCoordinator(BaseSweepCoordinator):
    """Memory efficiency sweep: tracks bytes/item overhead across history."""

    metric_id = "memory"
    metric_unit = "bytes/item"
    lower_is_better = True
    workload_id = f"{SWEEP_TEST}{SWEEP_VAL_SIZE}b-t{SWEEP_IO_THREADS}-p{SWEEP_PIPELINING}"

    def __init__(self, repo_path: Path):
        super().__init__(repo_path, MEMORY_STATE_FILE)

    def _create_task(self, sweep_task: SweepTask) -> MemTaskData:
        return MemTaskData(
            source=SWEEP_SOURCE,
            specifier=sweep_task.commit,
            make_args=JEMALLOC_PROF_MAKE_ARGS,
            replicas=0,
            note=f"[memory-sweep] {sweep_task.reason}",
            requirements={},
            type=MEMORY_TEST,
            val_sizes=[MEMORY_VAL_SIZE],
            has_expire=False,
            enable_profiling=True,
        )

    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        """Extract bytes_per_item from output. CV=0 (memory is deterministic)."""
        import json as _json

        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None

        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = _json.loads(line)
                if entry.get("task_id") == task.task_id:
                    score = entry.get("score")
                    if score and score > 0:
                        return (score, 0.0, 1)  # (bytes_per_item, cv=0, reps=1)
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def _extract_breakdown(self, task: BaseTaskData) -> Optional[dict[str, float]]:
        """Extract per-category breakdown from the task output."""
        import json as _json

        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None

        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = _json.loads(line)
                if entry.get("task_id") == task.task_id:
                    data = entry.get("data", {})
                    results = data.get("results", [])
                    if results and results[0].get("breakdown"):
                        return results[0]["breakdown"]
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def on_task_completed(self, task: BaseTaskData) -> None:
        """Override to attach breakdown data to the recorded point."""
        if not self._is_my_task(task):
            return

        result = self._extract_result(task)
        if result:
            value, cv, reps = result
            self.record_result(task.sweep_commit, value, cv, reps)  # type: ignore[attr-defined]

            # Attach breakdown to the point (separate extraction, no shared mutable state)
            breakdown = self._extract_breakdown(task)
            if breakdown and task.sweep_commit in self.state.points:  # type: ignore[attr-defined]
                self.state.points[task.sweep_commit].breakdown = breakdown  # type: ignore[attr-defined]
                self.state.save(self.state_file)
                logger.info("Recorded memory breakdown for %s", task.sweep_commit[:8])  # type: ignore[attr-defined]
        else:
            commit = getattr(task, "sweep_commit", "?")
            logger.warning("Could not extract result for sweep commit %s", commit[:8])
        self.queue_next_if_needed()

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return isinstance(task, MemTaskData) and bool(task.sweep_commit)
