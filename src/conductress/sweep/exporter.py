"""Exporter: generates series.json for the dashboard from sweep state."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        "compute": lambda c, rps=0, duration=0, **_: (
            c["instructions"] / (rps * duration) if rps and duration and c.get("instructions") else None
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

PERF_GROUPS = [
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
]


def export_series(
    state: SweepState,
    output_path: Path,
    platform: str = "arm64/c7g.metal/graviton3",
    workload: str = "",
    lower_is_better: bool = False,
) -> None:
    """Export sweep state to dashboard-ready series.json.

    Args:
        state: The current sweep state with benchmark results.
        output_path: Where to write series.json.
        platform: Platform identifier string.
        workload: Workload identifier string.
    """
    if not workload:
        from conductress.sweep.coordinator import SWEEP_IO_THREADS, SWEEP_PIPELINING, SWEEP_TEST, SWEEP_VAL_SIZE

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
        if point.breakdown:
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
            "repo": "valkey-io/valkey",
            "branch": "unstable",
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
            noise_floor = max(left.cv or 0.0, right.cv or 0.0) / 100.0
            if abs(delta) >= max(state.threshold, noise_floor):
                annotation: dict[str, Any] = {
                    "commit": right.commit,
                    "delta": round(delta, 4),
                    "workload": workload,
                    "type": "regression" if (delta > 0 if lower_is_better else delta < 0) else "improvement",
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


def _compute_metric(point: BenchmarkPoint, metric_id: str) -> Any:
    """Compute a normalized metric value from raw perf counters."""
    if not point.perf_counters:
        return None
    metric_def = PERF_METRICS.get(metric_id)
    if not metric_def:
        return None
    try:
        return metric_def["compute"](
            point.perf_counters,
            rps=point.perf_rps or 0,
            duration=point.perf_duration_seconds or 0,
        )
    except (ZeroDivisionError, KeyError, TypeError):
        return None


def export_perf_metrics(
    state: SweepState,
    output_dir: Path,
    platform: str,
    workload: str = "",
) -> dict[str, int]:
    """Export perf stat metrics as individual series files.

    Returns a dict of metric_id -> point_count for metrics that had data.
    """
    if not workload:
        from conductress.sweep.coordinator import SWEEP_IO_THREADS, SWEEP_PIPELINING, SWEEP_TEST, SWEEP_VAL_SIZE

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

    for metric_id, metric_def in PERF_METRICS.items():
        points: list[dict[str, Any]] = []
        for point in perf_points:
            value = _compute_metric(point, metric_id)
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

        series: dict[str, Any] = {
            "metadata": {
                "repo": "valkey-io/valkey",
                "branch": "unstable",
                "platform": platform,
                "workload": workload,
                "metric": metric_id,
                "unit": metric_def["unit"],
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


def export_manifest(output_dir: Path, platforms: list[str], workloads: list[str]) -> None:
    """Write the manifest.json grouping file for the dashboard."""
    manifest: dict[str, Any] = {
        "version": 1,
        "platforms": platforms,
        "workloads": [{"id": w, "label": w} for w in workloads],
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
        + PERF_GROUPS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
