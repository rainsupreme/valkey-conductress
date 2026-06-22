"""Exporter: generates series.json for the dashboard from sweep state."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from conductress.config import ANNOTATION_THRESHOLD
from conductress.heap_profiler import recategorize_from_stacks
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepPlanner, SweepState

# =============================================================================
# Perf metric definitions and normalization
# =============================================================================

PERF_METRICS: dict[str, dict[str, Any]] = {
    "ipc": {
        "label": "IPC",
        "unit": "instructions/cycle",
        "compute": lambda c, **_: c["instructions"] / c["cycles"] if c.get("cycles") else None,
    },
    "instructions-per-req": {
        "label": "Instructions per Request",
        "unit": "instructions/request",
        # Raw counters are SUMMED across all collected reps (see task_perf_benchmark
        # run loop), but rps/duration describe a single rep, so the summed instruction
        # total must be divided by the rep count to recover per-request instructions.
        # Omitting this caused a bimodal artifact (value scaled linearly with rep
        # count: 3x for fixed 3-rep runs, up to 10x for adaptive runs). Unlike ratio
        # metrics (IPC, MPKI, stall%), the rep factor does not cancel here.
        "compute": lambda c, rps=0, duration=0, reps=1, **_: (
            c["instructions"] / (rps * duration * (reps or 1)) if rps and duration and c.get("instructions") else None
        ),
    },
    "icache-mpki": {
        "label": "I-Cache MPKI",
        "unit": "misses/1Ki",
        "compute": lambda c, **_: (
            c["L1-icache-load-misses"] / c["instructions"] * 1000
            if c.get("instructions") and c.get("L1-icache-load-misses")
            else None
        ),
    },
    "branch-mpki": {
        "label": "Branch MPKI",
        "unit": "misses/1Ki",
        "compute": lambda c, **_: (
            c["branch-misses"] / c["instructions"] * 1000 if c.get("instructions") and c.get("branch-misses") else None
        ),
    },
    "frontend-stall-pct": {
        "label": "Frontend Stall %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["stalled-cycles-frontend"] / c["cycles"] * 100
            if c.get("cycles") and c.get("stalled-cycles-frontend")
            else None
        ),
    },
    "backend-stall-pct": {
        "label": "Backend Stall %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["stalled-cycles-backend"] / c["cycles"] * 100
            if c.get("cycles") and c.get("stalled-cycles-backend")
            else None
        ),
    },
    "llc-mpki": {
        "label": "LLC MPKI",
        "unit": "misses/1Ki",
        "compute": lambda c, **_: (
            c["LLC-load-misses"] / c["instructions"] * 1000
            if c.get("instructions") and c.get("LLC-load-misses")
            else None
        ),
    },
    "tma-retiring-pct": {
        "label": "Retiring %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["topdown-retiring"] / c["slots"] * 100 if c.get("slots") and c.get("topdown-retiring") else None
        ),
    },
    "tma-fe-bound-pct": {
        "label": "Frontend Bound %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["topdown-fe-bound"] / c["slots"] * 100 if c.get("slots") and c.get("topdown-fe-bound") else None
        ),
    },
    "tma-be-bound-pct": {
        "label": "Backend Bound %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["topdown-be-bound"] / c["slots"] * 100 if c.get("slots") and c.get("topdown-be-bound") else None
        ),
    },
    "tma-bad-spec-pct": {
        "label": "Bad Speculation %",
        "unit": "%",
        "compute": lambda c, **_: (
            c["topdown-bad-spec"] / c["slots"] * 100 if c.get("slots") and c.get("topdown-bad-spec") else None
        ),
    },
}

PERF_GROUPS: list[dict[str, Any]] = [
    {
        "id": "efficiency",
        "title": "Execution Efficiency",
        "series": ["ipc", "instructions-per-req"],
        "y_axes": [
            {"id": "left", "label": "IPC", "series": ["ipc"]},
            {"id": "right", "label": "insn/request", "series": ["instructions-per-req"]},
        ],
    },
    {
        "id": "cache",
        "title": "Cache Pressure",
        "series": ["icache-mpki", "llc-mpki"],
        "y_axes": [{"id": "left", "label": "misses per 1K instructions", "series": ["icache-mpki", "llc-mpki"]}],
    },
    {
        "id": "pipeline",
        "title": "Pipeline Stalls",
        "series": ["frontend-stall-pct", "backend-stall-pct"],
        "y_axes": [
            {"id": "left", "label": "% of cycles stalled", "series": ["frontend-stall-pct", "backend-stall-pct"]}
        ],
    },
    {
        "id": "tma",
        "title": "Pipeline Breakdown (TMA)",
        "series": ["tma-retiring-pct", "tma-fe-bound-pct", "tma-be-bound-pct", "tma-bad-spec-pct"],
        "y_axes": [
            {
                "id": "left",
                "label": "% of pipeline slots",
                "series": ["tma-retiring-pct", "tma-fe-bound-pct", "tma-be-bound-pct", "tma-bad-spec-pct"],
            }
        ],
    },
    {
        "id": "branching",
        "title": "Branch Prediction",
        "series": ["branch-mpki"],
        "y_axes": [{"id": "left", "label": "misses per 1K instructions", "series": ["branch-mpki"]}],
    },
    {
        "id": "cpu-main",
        "title": "CPU Profile — Main Thread",
        "series": ["cpu-main"],
        "stacked": True,
        "y_axes": [{"id": "left", "label": "% of samples", "series": ["cpu-main"]}],
    },
    {
        "id": "cpu-io",
        "title": "CPU Profile — IO Threads",
        "series": ["cpu-io"],
        "stacked": True,
        "y_axes": [{"id": "left", "label": "% of samples", "series": ["cpu-io"]}],
    },
]

# Counter-based groups that can be split by thread (everything except the
# cpu-main/cpu-io flamegraph groups, which are already thread-specific).
_PER_THREAD_BASE_GROUP_IDS = ("efficiency", "cache", "pipeline", "tma", "branching")
_PER_THREAD_SUFFIXES = (("-main", "Main Thread"), ("-io", "IO Threads"))


def _build_per_thread_groups() -> list[dict[str, Any]]:
    """Derive per-thread (main/io) variants of the counter-based perf groups.

    For each splittable base group (e.g. ``efficiency``) this produces
    ``efficiency-main`` and ``efficiency-io`` groups whose series/axes reference
    the per-thread metric variants (``ipc-main`` / ``ipc-io`` etc.), mirroring the
    cpu-main/cpu-io split. The original process-wide groups are left untouched.
    """
    base_by_id: dict[str, dict[str, Any]] = {g["id"]: g for g in PERF_GROUPS}
    out: list[dict[str, Any]] = []
    for base_id in _PER_THREAD_BASE_GROUP_IDS:
        base = base_by_id.get(base_id)
        if base is None:
            continue
        for suffix, label in _PER_THREAD_SUFFIXES:
            axes: list[dict[str, Any]] = base["y_axes"]
            group: dict[str, Any] = {
                "id": f"{base_id}{suffix}",
                "title": f"{base['title']} — {label}",
                "series": [f"{s}{suffix}" for s in base["series"]],
                "y_axes": [
                    {
                        "id": axis["id"],
                        "label": axis["label"],
                        "series": [f"{s}{suffix}" for s in axis["series"]],
                    }
                    for axis in axes
                ],
                "per_thread": suffix.lstrip("-"),  # "main" | "io" — dashboard hint
            }
            out.append(group)
    return out


PER_THREAD_PERF_GROUPS = _build_per_thread_groups()


def export_series(
    state: SweepState,
    output_path: Path,
    platform: str = "arm64/c7g.metal/graviton3",
    workload: str = "",
    lower_is_better: bool = False,
    num_keys: int = 0,
    repo: str = "valkey-io/valkey",
    branch: str = "unstable",
) -> None:
    """Export sweep state to dashboard-ready series.json.

    Args:
        state: The current sweep state with benchmark results.
        output_path: Where to write series.json.
        platform: Platform identifier string.
        workload: Workload identifier string.
        num_keys: If provided and point has raw_stacks, recompute breakdown at export time.
    """
    if not workload:
        from conductress.config import SWEEP_IO_THREADS, SWEEP_PIPELINING, SWEEP_TEST, SWEEP_VAL_SIZE

        workload = f"{SWEEP_TEST.upper()}_{SWEEP_VAL_SIZE}B_t{SWEEP_IO_THREADS}_p{SWEEP_PIPELINING}"

    planner = SweepPlanner(state)
    commit_index = planner._commit_index

    # Build ordered points list
    points: list[dict[str, Any]] = []
    has_breakdown = False
    for point in planner._get_ordered_completed_points():
        entry: dict[str, Any] = {
            "commit": point.commit,
            "date": point.date,
            "commit_index": commit_index.get(point.commit, 0),
            "results": {
                workload: {
                    "rps": point.value,
                    "cv": point.cv,
                    "reps": point.reps,
                }
            },
        }
        pr = state.commit_prs.get(point.commit) or point.pr
        pr_title = state.commit_titles.get(point.commit)
        if pr is not None:
            entry["pr"] = pr
        if pr_title is not None:
            entry["pr_title"] = pr_title
        if point.raw_stacks and num_keys > 0:
            entry["breakdown"] = recategorize_from_stacks(point.raw_stacks, num_keys)
            has_breakdown = True
        elif point.breakdown:
            entry["breakdown"] = point.breakdown
            has_breakdown = True
        points.append(entry)

    # Build landmarks list
    landmarks = [
        {
            "commit": lm.commit,
            "date": lm.date,
            "label": lm.label,
            "type": "release",
            "commit_index": commit_index.get(lm.commit, 0),
        }
        for lm in state.landmarks
    ]

    # Build annotations from segments that have been bisected down to 1 commit
    annotations = _build_annotations(state, planner, workload, lower_is_better)

    series: dict[str, Any] = {
        "metadata": {
            "repo": repo,
            "branch": branch,
            "platform": platform,
            "workload": workload,
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_commits": len(state.merge_commits),
        },
        "workloads": [workload],
        "points": points,
        "landmarks": landmarks,
        "annotations": annotations,
    }

    # Add category metadata if any points have breakdown data
    if has_breakdown:
        from conductress.heap_profiler import CATEGORY_NAMES

        series["metadata"]["categories"] = CATEGORY_NAMES

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(series, indent=2))


def _build_annotations(
    state: SweepState, planner: SweepPlanner, workload: str, lower_is_better: bool = False
) -> list[dict[str, Any]]:
    """Build annotations for commits where a change was pinpointed."""
    annotations: list[dict[str, Any]] = []

    # Find segments of exactly 1 commit (fully bisected)
    completed = planner._get_ordered_completed_points()
    commit_index = planner._commit_index

    for i in range(len(completed) - 1):
        left = completed[i]
        right = completed[i + 1]
        left_idx = commit_index.get(left.commit, 0)
        right_idx = commit_index.get(right.commit, 0)

        # Adjacent commits — this is a pinpointed change
        if right_idx - left_idx == 1:
            if left.value is None or right.value is None:
                continue
            delta = (right.value - left.value) / left.value
            if abs(delta) >= ANNOTATION_THRESHOLD:
                annotation: dict[str, Any] = {
                    "commit": right.commit,
                    "delta": round(delta, 4),
                    "workload": workload,
                    "type": "increase" if delta > 0 else "decrease",
                    "good": (delta < 0) if lower_is_better else (delta > 0),
                }
                pr = state.commit_prs.get(right.commit) or right.pr
                pr_title = state.commit_titles.get(right.commit) or right.pr_title
                if pr is not None:
                    annotation["pr"] = pr
                if pr_title is not None:
                    annotation["pr_title"] = pr_title
                # Check for manual note in point
                annotations.append(annotation)

    return annotations


# =============================================================================
# Perf metric series export
# =============================================================================


def _effective_perf_reps(point: BenchmarkPoint) -> int:
    """Number of reps whose raw counters were summed into this point's perf data.

    Raw perf counters are summed across reps, so absolute per-request metrics must
    divide by this count. ``perf_rep_count`` is the exact summed-rep count recorded
    going forward; historical points predate that field and fall back to ``reps``
    (which equalled the perf-rep count for all collected sweep data — verified).
    """
    return point.perf_rep_count or point.reps or 1


def _compute_metric_from_counters(
    counters: Optional[dict], metric_id: str, rps: float, duration: float, reps: int = 1
) -> Any:
    """Compute a normalized metric value from a raw perf-counter dict."""
    if not counters:
        return None
    metric_def = PERF_METRICS.get(metric_id)
    if not metric_def:
        return None
    try:
        return metric_def["compute"](counters, rps=rps, duration=duration, reps=reps)
    except (ZeroDivisionError, KeyError, TypeError):
        return None


def _compute_metric(point: BenchmarkPoint, metric_id: str) -> Any:
    """Compute a normalized metric value from a point's process-wide perf counters."""
    return _compute_metric_from_counters(
        point.perf_counters,
        metric_id,
        point.perf_rps or 0,
        point.perf_duration_seconds or 0,
        _effective_perf_reps(point),
    )


def export_perf_metrics(
    state: SweepState,
    output_dir: Path,
    platform: str,
    workload: str = "",
    repo: str = "valkey-io/valkey",
    branch: str = "unstable",
) -> dict[str, int]:
    """Export perf stat metrics as individual series files.

    Returns a dict of metric_id -> point_count for metrics that had data.
    """
    if not workload:
        from conductress.config import SWEEP_IO_THREADS, SWEEP_PIPELINING, SWEEP_TEST, SWEEP_VAL_SIZE

        workload = f"{SWEEP_TEST.lower()}{SWEEP_VAL_SIZE}b-t{SWEEP_IO_THREADS}-p{SWEEP_PIPELINING}"

    planner = SweepPlanner(state)
    commit_index = planner._commit_index
    completed = planner._get_ordered_completed_points()

    # Filter to points that have perf counters
    perf_points = [p for p in completed if p.perf_counters]
    if not perf_points:
        return {}

    # Build landmarks (shared across all metric files)
    landmarks = [
        {
            "commit": lm.commit,
            "date": lm.date,
            "label": lm.label,
            "type": "release",
            "commit_index": commit_index.get(lm.commit, 0),
        }
        for lm in state.landmarks
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, int] = {}

    # Each metric is emitted in up to three variants: process-wide (no suffix),
    # main-thread ("-main"), and IO-threads ("-io"). The per-thread variants read
    # the point's per-thread counter dicts (sparse/forward-only — only points that
    # collected per-thread data contribute). All variants reuse the same compute
    # lambdas via _compute_metric_from_counters.
    variants: list[tuple[str, str]] = [
        ("", "perf_counters"),
        ("-main", "perf_counters_main"),
        ("-io", "perf_counters_io"),
    ]

    for metric_id, metric_def in PERF_METRICS.items():
        for suffix, counter_attr in variants:
            points: list[dict[str, Any]] = []
            for point in perf_points:
                counters = getattr(point, counter_attr, None)
                if not counters:
                    continue
                value = _compute_metric_from_counters(
                    counters,
                    metric_id,
                    point.perf_rps or 0,
                    point.perf_duration_seconds or 0,
                    _effective_perf_reps(point),
                )
                if value is None:
                    continue
                entry: dict[str, Any] = {
                    "commit": point.commit,
                    "commit_index": commit_index.get(point.commit, 0),
                    "date": point.date,
                    "value": round(value, 6),
                }
                pr = state.commit_prs.get(point.commit) or point.pr
                pr_title = state.commit_titles.get(point.commit)
                if pr is not None:
                    entry["pr"] = pr
                if pr_title is not None:
                    entry["pr_title"] = pr_title
                points.append(entry)

            if not points:
                continue

            metric_key = f"{metric_id}{suffix}"
            series: dict[str, Any] = {
                "metadata": {
                    "repo": "valkey-io/valkey",
                    "branch": "unstable",
                    "platform": platform,
                    "workload": workload,
                    "metric": metric_key,
                    "unit": metric_def["unit"],
                    "generated": datetime.now(timezone.utc).isoformat(),
                    "total_commits": len(state.merge_commits),
                },
                "points": points,
                "landmarks": landmarks,
            }

            filename = f"series-{platform}-{workload}-{metric_key}.json"
            (output_dir / filename).write_text(json.dumps(series, indent=2))
            exported[metric_key] = len(points)

    return exported


def export_latency(
    state: SweepState,
    output_path: Path,
    platform: str,
    workload: str,
    target_rps: int = 100_000,
    tool_version: str = "d52544b1",
    repo: str = "valkey-io/valkey",
    branch: str = "unstable",
) -> int:
    """Export latency sweep data to a dashboard-ready series file.

    Reads extended latency data (all percentiles + histogram) from output.jsonl
    since the state only stores p99 as the primary value.

    Returns the number of exported points.
    """

    planner = SweepPlanner(state)
    commit_index = planner._commit_index
    completed = planner._get_ordered_completed_points()

    if not completed:
        return 0

    # Build points
    points: list[dict[str, Any]] = []
    for point in completed:
        entry: dict[str, Any] = {
            "commit": point.commit,
            "commit_index": commit_index.get(point.commit, 0),
            "date": point.date,
            "p99_us": point.value,  # primary metric stored in state
        }
        # Include full latency data if available
        if point.latency_data:
            entry["p50_us"] = point.latency_data.get("p50_us")
            entry["p99_9_us"] = point.latency_data.get("p99_9_us")
            entry["p100_us"] = point.latency_data.get("p100_us")
            entry["target_rps"] = point.latency_data.get("target_rps")
            entry["actual_rps"] = point.latency_data.get("actual_rps")
            entry["histogram"] = point.latency_data.get("histogram")
        pr = state.commit_prs.get(point.commit) or point.pr
        pr_title = state.commit_titles.get(point.commit) or point.pr_title
        if pr is not None:
            entry["pr"] = pr
        if pr_title is not None:
            entry["pr_title"] = pr_title
        points.append(entry)

    # Build landmarks
    landmarks = [
        {
            "commit": lm.commit,
            "date": lm.date,
            "label": lm.label,
            "type": "release",
            "commit_index": commit_index.get(lm.commit, 0),
        }
        for lm in state.landmarks
    ]

    # Build annotations (same logic as throughput but with lower_is_better=True)
    annotations = _build_annotations(state, planner, workload, lower_is_better=True)

    series: dict[str, Any] = {
        "metadata": {
            "repo": repo,
            "branch": branch,
            "platform": platform,
            "workload": workload,
            "metric": "latency",
            "unit": "µs",
            "load_fraction": None,
            "target_rps": target_rps,
            "pipeline": 1,
            "tool": "memtier_benchmark",
            "tool_version": tool_version,
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_commits": len(state.merge_commits),
        },
        "points": points,
        "landmarks": landmarks,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(series, indent=2))
    return len(points)


def export_cpu_profile(
    state: SweepState,
    output_dir: Path,
    platform: str,
    workload: str,
    repo: str = "valkey-io/valkey",
    branch: str = "unstable",
) -> dict[str, int]:
    """Export per-thread CPU profile breakdowns as stacked-area series files.

    Produces two files per workload:
      series-{platform}-{workload}-cpu-main.json  (main thread categories)
      series-{platform}-{workload}-cpu-io.json    (IO thread categories)

    Each point's ``breakdown`` is a category -> percentage map (sums to ~100,
    including an ``idle`` band). Forward-only: only points that actually have
    collapsed stacks contribute, so the series is sparse until sweep tasks
    accumulate CPU data. Returns {metric_id: point_count} for non-empty series.
    """
    from conductress.cpu_profiler import (
        CPU_CATEGORIES_IO,
        CPU_CATEGORIES_MAIN,
        CPU_CATEGORY_NAMES_IO,
        CPU_CATEGORY_NAMES_MAIN,
        categorize_cpu_stacks,
    )

    planner = SweepPlanner(state)
    commit_index = planner._commit_index
    completed = planner._get_ordered_completed_points()

    landmarks = [
        {
            "commit": lm.commit,
            "date": lm.date,
            "label": lm.label,
            "type": "release",
            "commit_index": commit_index.get(lm.commit, 0),
        }
        for lm in state.landmarks
    ]

    variants = [
        ("cpu-main", "cpu_stacks_main", CPU_CATEGORIES_MAIN, CPU_CATEGORY_NAMES_MAIN, "Main thread"),
        ("cpu-io", "cpu_stacks_io", CPU_CATEGORIES_IO, CPU_CATEGORY_NAMES_IO, "IO threads"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, int] = {}

    for metric_id, attr, categories, category_names, thread_label in variants:
        points: list[dict[str, Any]] = []
        for point in completed:
            stacks = getattr(point, attr, None)
            if not stacks:
                continue
            breakdown = categorize_cpu_stacks(stacks, categories)
            # Round for compact JSON; drop near-zero noise categories.
            breakdown = {cat: round(pct, 4) for cat, pct in breakdown.items() if pct >= 0.05}
            entry: dict[str, Any] = {
                "commit": point.commit,
                "commit_index": commit_index.get(point.commit, 0),
                "date": point.date,
                "breakdown": breakdown,
            }
            pr = state.commit_prs.get(point.commit) or point.pr
            pr_title = state.commit_titles.get(point.commit) or point.pr_title
            if pr is not None:
                entry["pr"] = pr
            if pr_title is not None:
                entry["pr_title"] = pr_title
            points.append(entry)

        if not points:
            continue

        series: dict[str, Any] = {
            "metadata": {
                "repo": repo,
                "branch": branch,
                "platform": platform,
                "workload": workload,
                "metric": metric_id,
                "thread": thread_label,
                "unit": "%",
                "categories": category_names,
                "generated": datetime.now(timezone.utc).isoformat(),
                "total_commits": len(state.merge_commits),
            },
            "points": points,
            "landmarks": landmarks,
        }

        filename = f"series-{platform}-{workload}-{metric_id}.json"
        (output_dir / filename).write_text(json.dumps(series, indent=2))
        exported[metric_id] = len(points)

    return exported


def export_cpu_stacks_raw(
    state: SweepState,
    output_dir: Path,
    platform: str,
    workload: str,
    repo: str = "valkey-io/valkey",
    branch: str = "unstable",
) -> dict[str, int]:
    """Export raw per-commit CPU collapsed stacks for future flamegraph drill-down.

    For each completed point that carries collapsed stacks, writes one file:
      series-{platform}-{workload}-cpu-stacks-{commit}.json

    containing the raw ``cpu_stacks_main`` / ``cpu_stacks_io`` arrays
    (shape ``[[stack_string, sample_count], ...]``) preserved exactly, plus
    minimal metadata. These files are intentionally separate from the small
    ``cpu-main`` / ``cpu-io`` percentage series so the dashboard load stays fast;
    a future flamegraph page fetches a single raw-stacks file lazily by commit.

    A lightweight per-workload index is also written:
      series-{platform}-{workload}-cpu-stacks-index.json

    listing ``[{commit, commit_index, date}, ...]`` for every commit that has a
    raw-stacks file, so a future page can discover available stacks without a
    directory listing on the static host.

    Forward-only/sparse: points without stacks are skipped. Idempotent: per-commit
    files that already exist are not rewritten (raw IO arrays are multi-MB, so
    re-serializing the full history every publish would be wasteful). The index is
    always rebuilt from the points that have stacks.

    Returns ``{"files_written": n, "indexed": m}``.
    """
    planner = SweepPlanner(state)
    commit_index = planner._commit_index
    completed = planner._get_ordered_completed_points()

    output_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    index_entries: list[dict[str, Any]] = []

    for point in completed:
        if not point.cpu_stacks_main and not point.cpu_stacks_io:
            continue

        idx = commit_index.get(point.commit, 0)
        index_entries.append({"commit": point.commit, "commit_index": idx, "date": point.date})

        filename = f"series-{platform}-{workload}-cpu-stacks-{point.commit}.json"
        file_path = output_dir / filename
        # Idempotent: raw IO arrays are multi-MB; never rewrite an existing file.
        if file_path.exists():
            continue

        metadata: dict[str, Any] = {
            "repo": repo,
            "branch": branch,
            "platform": platform,
            "workload": workload,
            "metric": "cpu-stacks",
            "commit": point.commit,
            "commit_index": idx,
            "date": point.date,
            "generated": datetime.now(timezone.utc).isoformat(),
        }
        pr = state.commit_prs.get(point.commit) or point.pr
        pr_title = state.commit_titles.get(point.commit) or point.pr_title
        if pr is not None:
            metadata["pr"] = pr
        if pr_title is not None:
            metadata["pr_title"] = pr_title

        payload: dict[str, Any] = {
            "metadata": metadata,
            "cpu_stacks_main": point.cpu_stacks_main or [],
            "cpu_stacks_io": point.cpu_stacks_io or [],
        }
        file_path.write_text(json.dumps(payload))
        files_written += 1

    if not index_entries:
        return {"files_written": 0, "indexed": 0}

    index = {
        "metadata": {
            "repo": repo,
            "branch": branch,
            "platform": platform,
            "workload": workload,
            "metric": "cpu-stacks-index",
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "commits": index_entries,
    }
    index_path = output_dir / f"series-{platform}-{workload}-cpu-stacks-index.json"
    index_path.write_text(json.dumps(index, indent=2))

    return {"files_written": files_written, "indexed": len(index_entries)}


def export_manifest(output_dir: Path, platforms: list[str], workloads: list[tuple[str, str]]) -> None:
    """Write per-platform manifest for dashboard auto-discovery."""
    from conductress.publisher import detect_platform

    platform_id, _ = detect_platform()
    throughput = [w for w, m in workloads if m == "throughput"]
    memory = [w for w, m in workloads if m == "memory"]
    latency = [w for w, m in workloads if m == "latency"]
    manifest: dict[str, Any] = {
        "version": 2,
        "platform": platform_id,
        "throughput_workloads": throughput,
        "memory_workloads": memory,
        "latency_workloads": latency,
        "groups": [
            {
                "id": "throughput",
                "title": "Throughput",
                "series": ["throughput"],
                "y_axes": [{"id": "left", "label": "requests/sec", "series": ["throughput"]}],
            },
            {
                "id": "memory",
                "title": "Memory Overhead",
                "series": ["memory"],
                "y_axes": [{"id": "left", "label": "bytes/key", "series": ["memory"]}],
            },
        ]
        + PERF_GROUPS
        + PER_THREAD_PERF_GROUPS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"manifest-{platform_id}.json").write_text(json.dumps(manifest, indent=2))
