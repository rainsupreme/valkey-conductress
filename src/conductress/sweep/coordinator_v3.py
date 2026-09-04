"""Cachecannon sweep coordinators (v3 epoch).

The v3 epoch replaces the retired ``scalable-v2`` sweep epoch. Two things forced
the replacement:

* The v2 mixed workload could not run at all -- it emitted a ``--warmup-period``
  flag that memtier has never implemented, and memtier has no warmup feature to
  wire it to. Cachecannon has native same-process warmup that keeps connections,
  pipelines, and key-generator state alive across the boundary and atomically
  resets statistics, which is the semantics a warmup actually requires.
* The v2 GET series came from a patched valkey-benchmark whose arrival shape
  differs from stock by roughly 5% even when both are server-bound, so those
  points could never share a chart line with cachecannon output.

State files, published filenames, and ownership predicates are disjoint from
both legacy v1 and the retired v2 namespace. Ownership is additionally disjoint
by construction: cachecannon tasks are ``CachecannonTaskData``, a sibling of
``PerfTaskData``/``MixedTaskData`` under ``BaseTaskData``, so the v1 and v2
``isinstance`` predicates cannot match a v3 task even accidentally.

Protocol values live in ``config`` and were confirmed by measurement -- see
``SWEEP_V3_*`` and ``conductress-cachecannon-v3/epoch-specification.md``.
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
    SWEEP_KEY_SIZE,
    SWEEP_STATE_DIR,
    SWEEP_V3_CLIENT_THREADS,
    SWEEP_V3_CONNECTIONS,
    SWEEP_V3_DISTRIBUTION,
    SWEEP_V3_DURATION,
    SWEEP_V3_EPOCH_ID,
    SWEEP_V3_IO_THREADS,
    SWEEP_V3_KEYSPACE,
    SWEEP_V3_MAX_REPS,
    SWEEP_V3_PIPELINING,
    SWEEP_V3_REPETITIONS,
    SWEEP_V3_SET_RATIO,
    SWEEP_V3_TARGET_CV,
    SWEEP_V3_VAL_SIZE,
    SWEEP_V3_WARMUP,
)
from conductress.sweep.coordinator import BaseSweepCoordinator
from conductress.sweep.planner import SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_cachecannon import CachecannonTaskData

logger = logging.getLogger(__name__)

# v3 state directory: isolated from both v1 (unqualified) and v2.
V3_STATE_DIR = SWEEP_STATE_DIR / "v3"


def _ensure_v3_state_dir() -> Path:
    V3_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return V3_STATE_DIR


class BaseCachecannonSweepCoordinatorV3(BaseSweepCoordinator):
    """Shared behaviour for cachecannon-driven v3 sweeps.

    Subclasses differ only in their workload label and the ``set_ratio`` /
    ``test`` pair that defines the workload.
    """

    metric_id = "throughput"
    metric_unit = "ops/sec"

    def __init__(
        self,
        repo_path: Path,
        label: str,
        test: str,
        set_ratio: int,
        val_size: int = SWEEP_V3_VAL_SIZE,
        io_threads: int = SWEEP_V3_IO_THREADS,
        pipelining: int = SWEEP_V3_PIPELINING,
        connections: int = SWEEP_V3_CONNECTIONS,
        threads: int = SWEEP_V3_CLIENT_THREADS,
        distribution: str = SWEEP_V3_DISTRIBUTION,
        engine: Optional[config.SweepEngine] = None,
    ):
        self._test = test
        self._set_ratio = set_ratio
        self._val_size = val_size
        self._io_threads = io_threads
        self._pipelining = pipelining
        self._connections = connections
        self._threads = threads
        self._distribution = distribution

        engine_prefix = f"{engine.source}-" if engine and engine.source != "valkey" else ""
        self._label = f"{engine_prefix}{label}"

        _ensure_v3_state_dir()
        state_file = V3_STATE_DIR / f"state_cachecannon-v3_{self._label}.json"
        super().__init__(repo_path, state_file, engine=engine)

    @property
    def epoch_id(self) -> str:
        return SWEEP_V3_EPOCH_ID

    @property
    def workload_id(self) -> str:  # type: ignore[override]
        return self._label

    def _create_task(self, sweep_task: SweepTask) -> CachecannonTaskData:
        return CachecannonTaskData(
            source=self._sweep_source,
            specifier=sweep_task.commit,
            make_args=self._sweep_make_args,
            replicas=0,
            note=f"[cachecannon-sweep-v3:{self._sweep_source}/{self.workload_id}] {sweep_task.reason}",
            requirements={},
            test=self._test,
            set_ratio=self._set_ratio,
            val_size=self._val_size,
            io_threads=self._io_threads,
            pipelining=self._pipelining,
            connections=self._connections,
            threads=self._threads,
            warmup=SWEEP_V3_WARMUP,
            duration=SWEEP_V3_DURATION,
            keyspace_count=SWEEP_V3_KEYSPACE,
            distribution=self._distribution,
            repetitions=SWEEP_V3_REPETITIONS,
            max_reps=SWEEP_V3_MAX_REPS,
            target_cv=SWEEP_V3_TARGET_CV,
        )

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
        # Adaptive reps mean the count is not knowable from configuration; it
        # must come from the recorded run list.
        reps = len(per_run) if per_run else SWEEP_V3_REPETITIONS
        return (rps, cv, reps) if rps else None

    def _is_my_task(self, task: BaseTaskData) -> bool:
        return (
            isinstance(task, CachecannonTaskData)
            # A manually queued cachecannon cell carries no sweep_commit and must
            # never be absorbed into the sweep series.
            and bool(getattr(task, "sweep_commit", ""))
            and task.source == self._sweep_source
            and task.test == self._test
            and task.set_ratio == self._set_ratio
            and task.val_size == self._val_size
            and task.io_threads == self._io_threads
            and task.pipelining == self._pipelining
            and task.connections == self._connections
            and task.threads == self._threads
            and task.distribution == self._distribution
        )


class CachecannonThroughputSweepCoordinatorV3(BaseCachecannonSweepCoordinatorV3):
    """Canonical v3 pure-GET throughput sweep."""

    def __init__(
        self,
        repo_path: Path,
        test: str = "get",
        val_size: int = SWEEP_V3_VAL_SIZE,
        io_threads: int = SWEEP_V3_IO_THREADS,
        pipelining: int = SWEEP_V3_PIPELINING,
        connections: int = SWEEP_V3_CONNECTIONS,
        threads: int = SWEEP_V3_CLIENT_THREADS,
        distribution: str = SWEEP_V3_DISTRIBUTION,
        engine: Optional[config.SweepEngine] = None,
    ):
        label = f"{test}-k{SWEEP_KEY_SIZE}-v{val_size}-t{io_threads}-p{pipelining}"
        if distribution != SWEEP_V3_DISTRIBUTION:
            label += f"-{distribution}"
        super().__init__(
            repo_path,
            label=label,
            test=test,
            set_ratio=0,
            val_size=val_size,
            io_threads=io_threads,
            pipelining=pipelining,
            connections=connections,
            threads=threads,
            distribution=distribution,
            engine=engine,
        )


class CachecannonMixedSweepCoordinatorV3(BaseCachecannonSweepCoordinatorV3):
    """Canonical v3 mixed GET/SET throughput sweep (default 80:20)."""

    def __init__(
        self,
        repo_path: Path,
        set_ratio: int = SWEEP_V3_SET_RATIO,
        val_size: int = SWEEP_V3_VAL_SIZE,
        io_threads: int = SWEEP_V3_IO_THREADS,
        pipelining: int = SWEEP_V3_PIPELINING,
        connections: int = SWEEP_V3_CONNECTIONS,
        threads: int = SWEEP_V3_CLIENT_THREADS,
        distribution: str = SWEEP_V3_DISTRIBUTION,
        engine: Optional[config.SweepEngine] = None,
    ):
        label = f"mixed-s{set_ratio}-k{SWEEP_KEY_SIZE}-v{val_size}-t{io_threads}-p{pipelining}"
        if distribution != SWEEP_V3_DISTRIBUTION:
            label += f"-{distribution}"
        super().__init__(
            repo_path,
            label=label,
            test="get",
            set_ratio=set_ratio,
            val_size=val_size,
            io_threads=io_threads,
            pipelining=pipelining,
            connections=connections,
            threads=threads,
            distribution=distribution,
            engine=engine,
        )


def create_v3_coordinators(repo_path: Path) -> list[BaseCachecannonSweepCoordinatorV3]:
    """Build the v3 coordinator roster.

    The roster opens deliberately small: it multiplies directly against a
    full-history backfill across four platforms.
    """
    return [
        CachecannonThroughputSweepCoordinatorV3(repo_path),
        CachecannonMixedSweepCoordinatorV3(repo_path),
    ]
