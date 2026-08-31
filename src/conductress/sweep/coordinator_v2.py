"""Profile-aware sweep coordinators (v2).

These coordinators use the generator profile system to run benchmarks with
explicit, reproducible benchmark-client binaries.  They maintain completely
isolated state files and export namespaces from the legacy v1 coordinators
to prevent cross-contamination.

Terminology correction: "P10" = pipeline depth 10 (not percentile 10).
Canonical value size: 16B.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import stdev
from typing import Optional

from conductress import config
from conductress.config import (
    CONDUCTRESS_RESULTS,
    SWEEP_DURATION,
    SWEEP_IO_THREADS,
    SWEEP_KEY_SIZE,
    SWEEP_MAX_REPS,
    SWEEP_PIPELINING,
    SWEEP_REPETITIONS,
    SWEEP_STATE_DIR,
    SWEEP_TARGET_CV,
    SWEEP_WARMUP,
)
from conductress.generator_profiles import SCALABLE_V2
from conductress.sweep.coordinator import BaseSweepCoordinator
from conductress.sweep.planner import SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_mixed import MixedTaskData
from conductress.tasks.task_perf_benchmark import PerfTaskData

logger = logging.getLogger(__name__)

# v2 state directory: isolated from v1
V2_STATE_DIR = SWEEP_STATE_DIR / "v2"


def _ensure_v2_state_dir() -> Path:
    V2_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return V2_STATE_DIR


# =============================================================================
# Throughput sweep coordinator (v2) — profile-aware
# =============================================================================


class ThroughputSweepCoordinatorV2(BaseSweepCoordinator):
    """Profile-aware throughput sweep using a named generator profile.

    Differences from v1 SweepCoordinator:
    - Tasks carry an explicit ``generator_profile``.
    - State files live under ``sweep_data/v2/`` to prevent cross-contamination.
    - Export namespacing comes from ``epoch_id``; workload IDs remain canonical.
    - Task discrimination uses ``generator_profile`` field matching.
    """

    metric_id = "throughput"
    metric_unit = "ops/sec"

    def __init__(
        self,
        repo_path: Path,
        profile_name: str = SCALABLE_V2.name,
        val_size: int = 16,
        test: str = "get",
        io_threads: int = SWEEP_IO_THREADS,
        pipelining: int = SWEEP_PIPELINING,
        engine: Optional[config.SweepEngine] = None,
    ):
        self._profile_name = profile_name
        self._val_size = val_size
        self._test = test
        self._io_threads = io_threads
        self._pipelining = pipelining

        engine_prefix = f"{engine.source}-" if engine and engine.source != "valkey" else ""
        self._label = f"{engine_prefix}{test}-k{SWEEP_KEY_SIZE}-v{val_size}-t{io_threads}-p{pipelining}"

        _ensure_v2_state_dir()
        state_file = V2_STATE_DIR / f"state_{self._profile_name}_{self._label}.json"
        super().__init__(repo_path, state_file, engine=engine)

    @property
    def epoch_id(self) -> str:
        return "v2"

    @property
    def workload_id(self) -> str:  # type: ignore[override]
        return self._label

    def _create_task(self, sweep_task: SweepTask) -> PerfTaskData:
        task = PerfTaskData(
            source=self._sweep_source,
            specifier=sweep_task.commit,
            make_args=self._sweep_make_args,
            replicas=0,
            note=f"[perf-sweep-v2:{self._sweep_source}/{self.workload_id}] {sweep_task.reason}",
            requirements={},
            test=self._test,
            val_size=self._val_size,
            io_threads=self._io_threads,
            pipelining=self._pipelining,
            warmup=SWEEP_WARMUP,
            duration=SWEEP_DURATION,
            perf_stat_enabled=True,
            has_expire=False,
            preload_keys=True,
            key_size=0,
            repetitions=SWEEP_REPETITIONS,
            max_reps=SWEEP_MAX_REPS,
            target_cv=SWEEP_TARGET_CV,
        )
        # Attach profile metadata
        task.generator_profile = self._profile_name  # type: ignore[attr-defined]
        # bench_binary will be resolved at execution time via resolve_bench_binary
        return task

    def _find_task_entry(self, task: BaseTaskData) -> Optional[dict]:
        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None
        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = json.loads(line)
                if entry.get("task_id") == task.task_id:
                    return entry
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        entry = self._find_task_entry(task)
        if not entry:
            return None
        rps = entry.get("score")
        per_run = entry.get("data", {}).get("per_run_rps", [])
        cv = (stdev(per_run) / rps) * 100 if len(per_run) >= 2 and rps else 0.0
        reps = len(per_run) if per_run else 3
        return (rps, cv, reps) if rps else None

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return (
            isinstance(task, PerfTaskData)
            and bool(getattr(task, "sweep_commit", ""))
            and task.source == self._sweep_source
            and task.val_size == self._val_size
            and task.test == self._test
            and task.io_threads == self._io_threads
            and task.pipelining == self._pipelining
            and getattr(task, "generator_profile", "") == self._profile_name
        )


# =============================================================================
# Mixed sweep coordinator (v2) — wraps MixedTaskData for set_ratio=20
# =============================================================================


class MixedSweepCoordinatorV2(BaseSweepCoordinator):
    """Versioned 80:20 GET/SET sweep using pinned memtier_benchmark.

    The v2 identity is a sweep protocol/export epoch, not a valkey-benchmark
    generator profile.  MixedTaskRunner continues to use the separately pinned
    memtier binary.  Default parameters are 20% SET, 16B values and P10.
    """

    metric_id = "throughput"
    metric_unit = "ops/sec"

    def __init__(
        self,
        repo_path: Path,
        set_ratio: int = 20,
        val_size: int = 16,
        io_threads: int = SWEEP_IO_THREADS,
        pipelining: int = SWEEP_PIPELINING,
        engine: Optional[config.SweepEngine] = None,
        schedule_p50: bool = False,
    ):
        self._set_ratio = set_ratio
        self._val_size = val_size
        self._io_threads = io_threads
        self._pipelining = pipelining
        self._schedule_p50 = schedule_p50

        engine_prefix = f"{engine.source}-" if engine and engine.source != "valkey" else ""
        self._label = f"{engine_prefix}mixed-s{set_ratio}-k{SWEEP_KEY_SIZE}-v{val_size}-t{io_threads}-p{pipelining}"

        _ensure_v2_state_dir()
        state_file = V2_STATE_DIR / f"state_v2_{self._label}.json"
        super().__init__(repo_path, state_file, engine=engine)

    @property
    def epoch_id(self) -> str:
        return "v2"

    @property
    def workload_id(self) -> str:  # type: ignore[override]
        return self._label

    def _create_task(self, sweep_task: SweepTask) -> MixedTaskData:
        task = MixedTaskData(
            source=self._sweep_source,
            specifier=sweep_task.commit,
            make_args=self._sweep_make_args,
            replicas=0,
            note=f"[mixed-sweep-v2:{self._sweep_source}/{self.workload_id}] {sweep_task.reason}",
            requirements={},
            set_ratio=self._set_ratio,
            val_size=self._val_size,
            io_threads=self._io_threads,
            pipelining=self._pipelining,
            duration=SWEEP_DURATION,
            warmup=SWEEP_WARMUP,
            repetitions=SWEEP_REPETITIONS,
        )
        return task

    def _find_task_entry(self, task: BaseTaskData) -> Optional[dict]:
        output_file = CONDUCTRESS_RESULTS / "output.jsonl"
        if not output_file.exists():
            return None
        for line in reversed(output_file.read_text().strip().splitlines()):
            try:
                entry = json.loads(line)
                if entry.get("task_id") == task.task_id:
                    return entry
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def _extract_result(self, task: BaseTaskData) -> Optional[tuple[float, float, int]]:
        entry = self._find_task_entry(task)
        if not entry:
            return None
        rps = entry.get("score")
        per_run = entry.get("data", {}).get("per_run_rps", [])
        cv = (stdev(per_run) / rps) * 100 if len(per_run) >= 2 and rps else 0.0
        reps = len(per_run) if per_run else 3
        return (rps, cv, reps) if rps else None

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return (
            isinstance(task, MixedTaskData)
            and bool(getattr(task, "sweep_commit", ""))
            and task.source == self._sweep_source
            and task.set_ratio == self._set_ratio
            and task.val_size == self._val_size
            and task.io_threads == self._io_threads
            and task.pipelining == self._pipelining
        )
