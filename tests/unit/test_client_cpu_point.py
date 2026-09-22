"""Client-CPU utilization on the sweep point: recording, export, round-trip.

Every cachecannon v3 result row carries a ``data.client_cpu`` block (built by
``utility.summarize_client_cpu``) describing the load generator's CPU use.
These tests pin the four contracts that lift it onto the published point:

* the v3 coordinator completion records the four ``client_*`` fields from a row
  that carries ``client_cpu`` and leaves them None on a row that does not;
* a saturated point logs a WARNING naming the workload, commit, utilization and
  allocated cores, and a non-saturated point does not;
* the exporter emits the four keys for a point that has them and omits them for
  one that does not;
* the fields survive a state save -> load round-trip, and a legacy state file
  lacking the keys loads them as None.
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.sweep.exporter import export_series
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState, SweepTask


@pytest.fixture(autouse=True)
def _ensure_valid_source(monkeypatch):
    """Other test modules patch ``config.REPO_NAMES`` at import time; the v3
    coordinator creates tasks with source ``valkey``, so keep it valid here."""
    import conductress.config as cfg

    if "valkey" not in cfg.REPO_NAMES:
        monkeypatch.setattr(cfg, "REPO_NAMES", cfg.REPO_NAMES + ["valkey"])


def _v3_coordinator(tmp_path: Path):
    from conductress.sweep import coordinator_v3
    from conductress.sweep.coordinator_v3 import CachecannonThroughputSweepCoordinatorV3

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
        with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
            coord = CachecannonThroughputSweepCoordinatorV3(tmp_path, test="get")
    return coord, results


def _complete_with_row(coord, results: Path, row: dict):
    commit = row["_commit"]
    coord.state.merge_commits = [commit]
    coord.state.commit_dates = {commit: "2026-01-01"}
    coord.state.points[commit] = BenchmarkPoint(commit=commit, date="2026-01-01")
    task = coord._create_task(SweepTask(commit=commit, reason="x", date="2026-01-01", priority=1))
    task.sweep_commit = commit
    payload = {"task_id": task.task_id, "score": row["score"], "data": row["data"]}
    (results / "output.jsonl").write_text(json.dumps(payload) + "\n")
    with patch("conductress.sweep.coordinator.get_head", return_value="z" * 40):
        coord.on_task_completed(task)
    return coord.state.points[commit]


class TestV3CoordinatorLiftsClientCpu:
    def test_row_with_client_cpu_records_all_four_fields(self, tmp_path: Path, monkeypatch):
        from conductress.sweep import coordinator_v3

        coord, results = _v3_coordinator(tmp_path)
        monkeypatch.setattr(coordinator_v3, "CONDUCTRESS_RESULTS", results)
        commit = "a" * 40
        point = _complete_with_row(
            coord,
            results,
            {
                "_commit": commit,
                "score": 1_221_594.0,
                "data": {
                    "per_run_rps": [1_221_594.0, 1_210_000.0],
                    "client_cpu": {
                        "cores_busy_per_rep": [15.0, 15.06],
                        "allocated_cores": 16,
                        "utilization": 0.941,
                        "saturated": True,
                    },
                },
            },
        )
        assert point.client_cores_busy == 15.06  # max over reps
        assert point.client_allocated_cores == 16
        assert point.client_utilization == 0.941
        assert point.client_saturated is True

    def test_row_without_client_cpu_leaves_fields_none(self, tmp_path: Path, monkeypatch):
        from conductress.sweep import coordinator_v3

        coord, results = _v3_coordinator(tmp_path)
        monkeypatch.setattr(coordinator_v3, "CONDUCTRESS_RESULTS", results)
        commit = "b" * 40
        point = _complete_with_row(
            coord,
            results,
            {"_commit": commit, "score": 3_000_000.0, "data": {"per_run_rps": [3_000_000.0, 3_010_000.0]}},
        )
        assert point.client_cores_busy is None
        assert point.client_allocated_cores is None
        assert point.client_utilization is None
        assert point.client_saturated is None

    def test_saturated_logs_warning(self, tmp_path: Path, monkeypatch, caplog):
        from conductress.sweep import coordinator_v3

        coord, results = _v3_coordinator(tmp_path)
        monkeypatch.setattr(coordinator_v3, "CONDUCTRESS_RESULTS", results)
        commit = "c" * 40
        with caplog.at_level(logging.WARNING, logger="conductress.sweep.coordinator_v3"):
            _complete_with_row(
                coord,
                results,
                {
                    "_commit": commit,
                    "score": 1_221_594.0,
                    "data": {
                        "per_run_rps": [1_221_594.0],
                        "client_cpu": {
                            "cores_busy_per_rep": [15.06],
                            "allocated_cores": 16,
                            "utilization": 0.941,
                            "saturated": True,
                        },
                    },
                },
            )
        assert "saturated client" in caplog.text
        assert commit[:8] in caplog.text
        assert "0.941" in caplog.text
        assert "16" in caplog.text

    def test_not_saturated_does_not_warn(self, tmp_path: Path, monkeypatch, caplog):
        from conductress.sweep import coordinator_v3

        coord, results = _v3_coordinator(tmp_path)
        monkeypatch.setattr(coordinator_v3, "CONDUCTRESS_RESULTS", results)
        commit = "d" * 40
        with caplog.at_level(logging.WARNING, logger="conductress.sweep.coordinator_v3"):
            _complete_with_row(
                coord,
                results,
                {
                    "_commit": commit,
                    "score": 3_000_000.0,
                    "data": {
                        "per_run_rps": [3_000_000.0],
                        "client_cpu": {
                            "cores_busy_per_rep": [4.0],
                            "allocated_cores": 8,
                            "utilization": 0.5,
                            "saturated": False,
                        },
                    },
                },
            )
        assert "saturated client" not in caplog.text


class TestClientCpuExport:
    def _point(self, **extra) -> BenchmarkPoint:
        return BenchmarkPoint(
            commit="a",
            date="2026-01-01",
            value=1_221_594.0,
            cv=0.3,
            reps=2,
            status=PointStatus.COMPLETED,
            **extra,
        )

    def test_exports_the_keys_when_present(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = self._point(
            client_cores_busy=15.06,
            client_allocated_cores=16,
            client_utilization=0.941,
            client_saturated=True,
        )
        out = tmp_path / "series.json"
        export_series(state, out, workload="get-k16-v16-t7-p1")
        result = json.loads(out.read_text())["points"][0]["results"]["get-k16-v16-t7-p1"]
        assert result["client_utilization"] == 0.941
        assert result["client_saturated"] is True
        assert result["client_cores_busy"] == 15.06
        assert result["client_allocated_cores"] == 16

    def test_omits_the_keys_when_absent(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = self._point()
        out = tmp_path / "series.json"
        export_series(state, out, workload="get-k16-v64-t7-p10")
        result = json.loads(out.read_text())["points"][0]["results"]["get-k16-v64-t7-p10"]
        assert "client_utilization" not in result
        assert "client_saturated" not in result
        assert "client_cores_busy" not in result
        assert "client_allocated_cores" not in result


class TestClientCpuStateRoundTrip:
    def test_fields_round_trip(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = BenchmarkPoint(
            commit="a",
            date="2026-01-01",
            value=1_221_594.0,
            cv=0.3,
            reps=2,
            client_cores_busy=15.06,
            client_allocated_cores=16,
            client_utilization=0.941,
            client_saturated=True,
            status=PointStatus.COMPLETED,
        )
        path = tmp_path / "s.json"
        state.save(path)
        loaded = SweepState.load(path)
        p = loaded.points["a"]
        assert p.client_cores_busy == 15.06
        assert p.client_allocated_cores == 16
        assert p.client_utilization == 0.941
        assert p.client_saturated is True

    def test_legacy_state_without_the_keys_loads_as_none(self, tmp_path: Path):
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({"merge_commits": ["a"], "points": {"a": {"commit": "a", "value": 1.0, "status": "COMPLETED"}}})
        )
        loaded = SweepState.load(path)
        p = loaded.points["a"]
        assert p.client_cores_busy is None
        assert p.client_allocated_cores is None
        assert p.client_utilization is None
        assert p.client_saturated is None
