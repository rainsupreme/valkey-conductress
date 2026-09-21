"""score_aggregate and the per-rep score spread (score_min/score_max).

Four contracts:

* ``CachecannonTaskData.score_aggregate`` is ``mean`` by default (every existing
  task unchanged) and validated against the registry; ``median`` records the
  median of the per-rep score series while leaving ``cv`` as the coefficient of
  variation of that series (taken against the mean).
* Every cachecannon task records ``score_min`` / ``score_max`` -- the min/max of
  the per-rep score series -- regardless of aggregate.
* ``BenchmarkPoint`` carries ``score_min`` / ``score_max`` as real dataclass
  fields that survive a state save -> load round-trip.
* The v3 export publishes ``score_min`` / ``score_max`` beside rps/cv/reps on a
  point that has them, and omits the keys on a v1 point that does not.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conductress import config
from conductress.sweep.exporter import export_series
from conductress.sweep.planner import BenchmarkPoint, PointStatus, SweepState
from conductress.tasks.task_cachecannon import SCORE_AGGREGATES, CachecannonTaskData, score_from_parsed
from conductress.topology import TopologySpec


@pytest.fixture(autouse=True)
def _ensure_valid_source(monkeypatch):
    # Some other test modules replace config.REPO_NAMES wholesale; make sure the
    # "valkey" source these tasks use is always valid regardless of test order.
    if "valkey" not in config.REPO_NAMES:
        monkeypatch.setattr(config, "REPO_NAMES", list(config.REPO_NAMES) + ["valkey"])


def _task(**overrides) -> CachecannonTaskData:
    fields = dict(
        source="valkey",
        specifier="abc123",
        make_args="",
        topology=TopologySpec.standalone(),
        note="",
        requirements={},
    )
    fields.update(overrides)
    return CachecannonTaskData(**fields)


def _parsed(throughput: float, p99_ms: float = 1.0) -> dict:
    return {
        "throughput_rps": throughput,
        "error_pct": 0.0,
        "hit_rate": {"percent": 100.0, "hits": 1.0, "misses": 0.0},
        "latency": {
            "command": "GET",
            "p50_ms": p99_ms / 2,
            "p99_ms": p99_ms,
            "p999_ms": p99_ms * 2,
            "max_ms": p99_ms * 8,
        },
    }


class TestScoreAggregateField:
    def test_default_is_mean_and_serializes(self, tmp_path):
        task = _task()
        assert task.score_aggregate == "mean"
        path = tmp_path / "t.json"
        task.save_to_file(path)
        assert json.loads(path.read_text())["score_aggregate"] == "mean"

    def test_documents_without_the_field_load_as_mean(self, tmp_path):
        from conductress.task_queue import BaseTaskData

        task = _task()
        path = tmp_path / "t.json"
        task.save_to_file(path)
        doc = json.loads(path.read_text())
        del doc["score_aggregate"]
        path.write_text(json.dumps(doc))
        assert BaseTaskData.from_file(path).score_aggregate == "mean"

    def test_registry_is_mean_and_median(self):
        assert SCORE_AGGREGATES == {"mean", "median"}

    def test_unknown_aggregate_rejected(self):
        with pytest.raises(ValueError, match="score_aggregate"):
            _task(score_aggregate="p50")


class TestMedianScoring:
    @pytest.fixture
    def _server(self):
        server = MagicMock()

        async def _lscpu(_cmd):
            return ("lscpu", "")

        server.run_host_command = _lscpu
        server.server_cpus = "0-3"
        return server

    @pytest.mark.asyncio
    async def test_median_records_the_median_but_leaves_cv_on_the_mean(self, _server):
        task = _task(repetitions=5, score_aggregate="median")
        runner = task.prepare_task_runner([config.ServerInfo(ip="127.0.0.1")])
        runner.file_protocol = MagicMock()
        runner.management_cpus = []
        # One slow restart (the 900k) should not move a median the way it moves a mean.
        rps = [3_000_000.0, 3_010_000.0, 2_990_000.0, 3_005_000.0, 900_000.0]
        results = [_parsed(r) for r in rps]
        await runner._record_result(_server, rps, results, "toml", None, rps)
        recorded = runner.file_protocol.write_results.call_args.args[0]
        assert recorded.score == pytest.approx(3_000_000.0)  # median, robust to the outlier
        # cv is stdev/mean of the whole series -- the outlier is still visible there.
        import statistics

        mean = statistics.mean(rps)
        assert recorded.cv == pytest.approx((statistics.stdev(rps) / mean) * 100, abs=0.01)
        assert recorded.data["score_aggregate"] == "median"

    @pytest.mark.asyncio
    async def test_mean_is_unchanged_and_still_the_default(self, _server):
        task = _task(repetitions=2)
        runner = task.prepare_task_runner([config.ServerInfo(ip="127.0.0.1")])
        runner.file_protocol = MagicMock()
        runner.management_cpus = []
        rps = [3_000_000.0, 3_010_000.0]
        await runner._record_result(_server, rps, [_parsed(r) for r in rps], "toml", None, rps)
        recorded = runner.file_protocol.write_results.call_args.args[0]
        assert recorded.score == pytest.approx(3_005_000.0)
        assert recorded.data["score_aggregate"] == "mean"

    @pytest.mark.asyncio
    async def test_score_min_max_recorded_for_all_tasks(self, _server):
        task = _task(repetitions=3)
        runner = task.prepare_task_runner([config.ServerInfo(ip="127.0.0.1")])
        runner.file_protocol = MagicMock()
        runner.management_cpus = []
        rps = [3_000_000.0, 3_100_000.0, 2_900_000.0]
        await runner._record_result(_server, rps, [_parsed(r) for r in rps], "toml", None, rps)
        data = runner.file_protocol.write_results.call_args.args[0].data
        assert data["score_min"] == 2_900_000.0
        assert data["score_max"] == 3_100_000.0

    @pytest.mark.asyncio
    async def test_score_bounds_track_the_score_metric_for_latency(self, _server):
        task = _task(rate_limit=100_000, pipelining=1, score_metric="p99", repetitions=3)
        runner = task.prepare_task_runner([config.ServerInfo(ip="127.0.0.1")])
        runner.file_protocol = MagicMock()
        runner.management_cpus = []
        results = [_parsed(100_000.0, 0.40), _parsed(100_000.0, 0.42), _parsed(100_000.0, 0.41)]
        rps = [r["throughput_rps"] for r in results]
        scores = [score_from_parsed(r, "p99") for r in results]
        await runner._record_result(_server, rps, results, "toml", None, scores)
        data = runner.file_protocol.write_results.call_args.args[0].data
        assert data["score_min"] == pytest.approx(400.0)
        assert data["score_max"] == pytest.approx(420.0)


class TestBenchmarkPointScoreBounds:
    def test_fields_round_trip_through_state(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = BenchmarkPoint(
            commit="a",
            date="2026-01-01",
            value=3_000_000.0,
            cv=0.3,
            reps=5,
            score_min=2_900_000.0,
            score_max=3_100_000.0,
            status=PointStatus.COMPLETED,
        )
        path = tmp_path / "s.json"
        state.save(path)
        loaded = SweepState.load(path)
        assert loaded.points["a"].score_min == 2_900_000.0
        assert loaded.points["a"].score_max == 3_100_000.0

    def test_old_state_without_the_fields_loads_as_none(self, tmp_path: Path):
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({"merge_commits": ["a"], "points": {"a": {"commit": "a", "value": 1.0, "status": "COMPLETED"}}})
        )
        loaded = SweepState.load(path)
        assert loaded.points["a"].score_min is None
        assert loaded.points["a"].score_max is None


class TestScoreBoundsExport:
    def test_v3_point_exports_the_spread(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = BenchmarkPoint(
            commit="a",
            date="2026-01-01",
            value=3_000_000.0,
            cv=0.3,
            reps=5,
            score_min=2_900_000.0,
            score_max=3_100_000.0,
            status=PointStatus.COMPLETED,
        )
        out = tmp_path / "series.json"
        export_series(state, out, workload="get-k16-v16-t7-p10")
        result = json.loads(out.read_text())["points"][0]["results"]["get-k16-v16-t7-p10"]
        assert result["rps"] == 3_000_000.0
        assert result["score_min"] == 2_900_000.0
        assert result["score_max"] == 3_100_000.0

    def test_v1_point_omits_the_keys(self, tmp_path: Path):
        state = SweepState(merge_commits=["a"], commit_dates={"a": "2026-01-01"})
        state.points["a"] = BenchmarkPoint(
            commit="a", date="2026-01-01", value=148000.0, cv=0.19, status=PointStatus.COMPLETED
        )
        out = tmp_path / "series.json"
        export_series(state, out, workload="get-k16-v64-t7-p10")
        result = json.loads(out.read_text())["points"][0]["results"]["get-k16-v64-t7-p10"]
        assert "score_min" not in result
        assert "score_max" not in result


class TestV3CoordinatorLiftsScoreBounds:
    def test_completion_sets_score_min_max_on_the_point(self, tmp_path: Path, monkeypatch):
        from conductress.sweep import coordinator_v3
        from conductress.sweep.coordinator_v3 import CachecannonThroughputSweepCoordinatorV3

        results = tmp_path / "results"
        results.mkdir()
        monkeypatch.setattr(coordinator_v3, "CONDUCTRESS_RESULTS", results)

        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                coord = CachecannonThroughputSweepCoordinatorV3(
                    tmp_path, test="get", pipelining=1, score_aggregate="median"
                )
        commit = "a" * 40
        coord.state.merge_commits = [commit]
        coord.state.commit_dates = {commit: "2026-01-01"}
        coord.state.points[commit] = BenchmarkPoint(commit=commit, date="2026-01-01")
        from conductress.sweep.planner import SweepTask

        task = coord._create_task(SweepTask(commit=commit, reason="x", date="2026-01-01", priority=1))
        task.sweep_commit = commit
        row = {
            "task_id": task.task_id,
            "score": 2_500_000.0,
            "data": {
                "per_run_rps": [2_500_000.0, 2_510_000.0, 900_000.0],
                "score_min": 900_000.0,
                "score_max": 2_510_000.0,
            },
        }
        (results / "output.jsonl").write_text(json.dumps(row) + "\n")
        with patch("conductress.sweep.coordinator.get_head", return_value="z" * 40):
            coord.on_task_completed(task)
        point = coord.state.points[commit]
        assert point.score_min == 900_000.0
        assert point.score_max == 2_510_000.0
