"""Exporter: generates series.json for the dashboard from sweep state."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.sweep.planner import PointStatus, SweepPlanner, SweepState


def export_series(state: SweepState, output_path: Path,
                  platform: str = "arm64/c7g.metal/graviton3",
                  workload: str = "GET_16B_t1_p1") -> None:
    """Export sweep state to dashboard-ready series.json.

    Args:
        state: The current sweep state with benchmark results.
        output_path: Where to write series.json.
        platform: Platform identifier string.
        workload: Workload identifier string.
    """
    planner = SweepPlanner(state)

    # Build ordered points list
    points: list[dict[str, Any]] = []
    for point in planner._get_ordered_completed_points():
        entry: dict[str, Any] = {
            "commit": point.commit,
            "date": point.date,
            "results": {
                workload: {
                    "rps": point.rps,
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
        points.append(entry)

    # Build landmarks list
    landmarks = [
        {"commit": lm.commit, "date": lm.date, "label": lm.label, "type": "release"}
        for lm in state.landmarks
    ]

    # Build annotations from segments that have been bisected down to 1 commit
    annotations = _build_annotations(state, planner, workload)

    series: dict[str, Any] = {
        "metadata": {
            "repo": "valkey-io/valkey",
            "branch": "unstable",
            "platform": platform,
            "workload": workload,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
        "workloads": [workload],
        "points": points,
        "landmarks": landmarks,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(series, indent=2))


def _build_annotations(state: SweepState, planner: SweepPlanner,
                       workload: str) -> list[dict[str, Any]]:
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
            assert left.rps is not None and right.rps is not None
            delta = (right.rps - left.rps) / left.rps
            noise_floor = max(left.cv or 0.0, right.cv or 0.0) / 100.0
            if abs(delta) >= max(state.threshold, noise_floor):
                annotation: dict[str, Any] = {
                    "commit": right.commit,
                    "delta": round(delta, 4),
                    "workload": workload,
                    "type": "regression" if delta < 0 else "improvement",
                }
                if right.pr is not None:
                    annotation["pr"] = right.pr
                if right.pr_title is not None:
                    annotation["pr_title"] = right.pr_title
                # Check for manual note in point
                annotations.append(annotation)

    return annotations
