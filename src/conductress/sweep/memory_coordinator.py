"""Memory sweep coordinator: tracks per-item memory overhead across Valkey history."""

import logging
from pathlib import Path
from typing import Optional

from conductress.config import CONDUCTRESS_RESULTS, PROJECT_ROOT
from conductress.sweep.coordinator import SWEEP_SOURCE, BaseSweepCoordinator
from conductress.sweep.planner import SweepState, SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_mem_efficiency import MemTaskData

logger = logging.getLogger(__name__)

MEMORY_STATE_DIR = PROJECT_ROOT / "sweep_data"
MEMORY_STATE_FILE = MEMORY_STATE_DIR / "memory_state.json"

# Memory sweep parameters
MEMORY_TEST = "set"  # SET key value — most common pattern
MEMORY_VAL_SIZE = 64  # bytes
MEMORY_MAKE_ARGS = "USE_FAST_FLOAT=yes"


class MemorySweepCoordinator(BaseSweepCoordinator):
    """Memory efficiency sweep: tracks bytes/item overhead across history."""

    metric_id = "memory"
    metric_unit = "bytes/item"

    def __init__(self, repo_path: Path):
        super().__init__(repo_path, MEMORY_STATE_FILE)

    def _create_task(self, sweep_task: SweepTask) -> MemTaskData:
        return MemTaskData(
            source=SWEEP_SOURCE,
            specifier=sweep_task.commit,
            make_args=MEMORY_MAKE_ARGS,
            replicas=0,
            note=f"[memory-sweep] {sweep_task.reason}",
            requirements={},
            type=MEMORY_TEST,
            val_sizes=[MEMORY_VAL_SIZE],
            has_expire=False,
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

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return isinstance(task, MemTaskData) and bool(task.sweep_commit)
