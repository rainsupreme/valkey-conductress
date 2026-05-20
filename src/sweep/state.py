"""State persistence for the sweep planner.

Handles serialization/deserialization of SweepState to/from JSON.
"""

import json
from pathlib import Path
from typing import Any

from src.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepState


def save_state(state: SweepState, path: Path) -> None:
    """Serialize sweep state to a JSON file."""
    data: dict[str, Any] = {
        "threshold": state.threshold,
        "last_benchmarked_head": state.last_benchmarked_head,
        "merge_commits": state.merge_commits,
        "commit_dates": state.commit_dates,
        "landmarks": [
            {"commit": lm.commit, "date": lm.date, "label": lm.label}
            for lm in state.landmarks
        ],
        "points": {
            commit: {
                "commit": p.commit,
                "date": p.date,
                "rps": p.rps,
                "cv": p.cv,
                "reps": p.reps,
                "pr": p.pr,
                "pr_title": p.pr_title,
                "status": p.status.name,
            }
            for commit, p in state.points.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_state(path: Path) -> SweepState:
    """Deserialize sweep state from a JSON file."""
    if not path.exists():
        return SweepState()

    data = json.loads(path.read_text())
    state = SweepState(
        threshold=data.get("threshold", 0.01),
        last_benchmarked_head=data.get("last_benchmarked_head"),
        merge_commits=data.get("merge_commits", []),
        commit_dates=data.get("commit_dates", {}),
    )

    for lm_data in data.get("landmarks", []):
        state.landmarks.append(Landmark(
            commit=lm_data["commit"],
            date=lm_data["date"],
            label=lm_data["label"],
        ))

    for commit, p_data in data.get("points", {}).items():
        state.points[commit] = BenchmarkPoint(
            commit=p_data["commit"],
            date=p_data.get("date", ""),
            rps=p_data.get("rps"),
            cv=p_data.get("cv"),
            reps=p_data.get("reps", 3),
            pr=p_data.get("pr"),
            pr_title=p_data.get("pr_title"),
            status=PointStatus[p_data.get("status", "PENDING")],
        )

    return state
