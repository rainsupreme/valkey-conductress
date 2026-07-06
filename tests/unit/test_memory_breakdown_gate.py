"""Tests for the memory-breakdown export gate.

Mirrors the CPU-stacks opt-out: engines that opt out of internal profiling (Redis)
keep total memory but must not export the jemalloc allocation breakdown (which
exposes the binary's allocation-site symbols). include_breakdown=False also stops
pre-existing breakdowns in state from being re-published.
"""

import json

from conductress.sweep.exporter import export_series
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState


def _state_with_breakdown():
    state = SweepState(merge_commits=["abc123"], commit_dates={"abc123": "2026-07-01"})
    state.points["abc123"] = BenchmarkPoint(
        commit="abc123",
        date="2026-07-01",
        value=1000.0,
        cv=0.5,
        reps=3,
        breakdown={"robj": 40.0, "sds": 20.0},
        status=PointStatus.COMPLETED,
    )
    return state


def _export(tmp_path, include_breakdown):
    out = tmp_path / "series.json"
    export_series(
        _state_with_breakdown(),
        out,
        platform="arm64",
        workload="mem-set",
        num_keys=0,
        include_breakdown=include_breakdown,
    )
    return json.loads(out.read_text())


def _points(data):
    return data.get("points") or data.get("data") or []


class TestMemoryBreakdownExportGate:
    def test_breakdown_included_by_default(self, tmp_path):
        pts = _points(_export(tmp_path, include_breakdown=True))
        assert pts and "breakdown" in pts[0]

    def test_breakdown_omitted_when_gated(self, tmp_path):
        data = _export(tmp_path, include_breakdown=False)
        pts = _points(data)
        assert pts, "point (with total memory) must still be exported"
        assert "breakdown" not in pts[0], "breakdown must be omitted for opted-out engines"
        # Total memory (rps/results) is still present.
        assert pts[0]["results"]["mem-set"]["rps"] == 1000.0
