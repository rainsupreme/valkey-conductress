"""Engines on two axes: how a binary is provisioned, and how much history is measured.

A ``history`` engine (Valkey) bisects and backfills every merge commit.  A
``release-and-tip`` engine (Redis) measures its latest release once and its tip
at most once per interval, which is all the engine comparison reads.  These
tests pin the config model, the planner switch, the landmark reduction, the
tip cadence, the sample tagging on points and in the export, the retirement of
the epoch-1 Redis mirror, and the Redis v3 roster.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from conductress import config
from conductress.config import (
    SWEEP_ENGINES,
    SWEEP_THROUGHPUT_WORKLOADS,
    SWEEP_V1_RETIRED_SERIES,
    SweepEngine,
    engine_repo_slug,
    engine_series_prefix,
    engine_tip_interval_seconds,
    engine_tracks_history,
    get_sweep_engine,
    sweep_throughput_label,
)
from conductress.sweep.coordinator import SweepCoordinator, latest_release_landmark
from conductress.sweep.exporter import export_series
from conductress.sweep.planner import BenchmarkPoint, Landmark, PointStatus, SweepPlanner, SweepState, TaskPriority

REDIS = get_sweep_engine("redis")
assert REDIS is not None
DAY = 24 * 3600.0


@pytest.fixture(autouse=True)
def _ensure_valid_sources(monkeypatch):
    monkeypatch.setattr(config, "REPO_NAMES", ["valkey", "redis"])


def _release_and_tip(**overrides) -> SweepEngine:
    fields = dict(source="redis", ref="origin/unstable", binary_name="redis-server", scope="release-and-tip")
    fields.update(overrides)
    return SweepEngine(**fields)


# ---------------------------------------------------------------------------
# Config: the two axes
# ---------------------------------------------------------------------------


class TestEngineModel:
    def test_defaults_are_the_valkey_shape(self):
        e = SweepEngine(source="valkey", ref="origin/unstable", binary_name="valkey-server")
        assert e.provisioning == "built-from-git"
        assert e.scope == "history"
        assert e.tracks_history is True

    def test_redis_is_release_and_tip_once_a_day(self):
        assert REDIS.scope == "release-and-tip"
        assert REDIS.tracks_history is False
        assert REDIS.tip_interval_seconds == DAY
        assert REDIS.provisioning == "built-from-git"

    def test_valkey_tracks_history(self):
        valkey = get_sweep_engine("valkey")
        assert valkey is not None and valkey.tracks_history is True

    def test_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            _release_and_tip(scope="tip-only")

    def test_unknown_provisioning_is_rejected(self):
        with pytest.raises(ValueError, match="provisioning"):
            _release_and_tip(provisioning="apt")

    def test_prebuilt_release_is_declared_but_not_implemented(self):
        # The axis exists so a release-only engine slots in later; constructing
        # one today must fail loudly rather than fall back to a source build.
        assert "prebuilt-release" in config.ENGINE_PROVISIONING
        with pytest.raises(NotImplementedError):
            _release_and_tip(source="dragonfly", binary_name="dragonfly", provisioning="prebuilt-release")

    def test_tip_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="tip_interval_hours"):
            _release_and_tip(tip_interval_hours=0)

    def test_helpers_treat_no_engine_as_the_valkey_sweep(self):
        assert engine_tracks_history(None) is True
        assert engine_tip_interval_seconds(None) == 0.0
        assert engine_series_prefix(None) == ""
        assert engine_repo_slug(None) == "valkey-io/valkey"

    def test_helpers_read_the_redis_engine(self):
        assert engine_tracks_history(REDIS) is False
        assert engine_tip_interval_seconds(REDIS) == DAY
        assert engine_series_prefix(REDIS) == "redis-"
        assert engine_repo_slug(REDIS) == "redis/redis"

    def test_history_engine_has_no_tip_interval(self):
        # A history engine measures every new tip; the interval is a
        # release-and-tip lever only.
        valkey = get_sweep_engine("valkey")
        assert engine_tip_interval_seconds(valkey) == 0.0

    def test_repo_slug_comes_from_the_repositories_table(self):
        assert engine_repo_slug(get_sweep_engine("valkey")) == "valkey-io/valkey"
        unknown = _release_and_tip(source="nowhere")
        assert engine_repo_slug(unknown) == "valkey-io/valkey"

    def test_throughput_label_matches_the_coordinator(self, tmp_path):
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            coord = SweepCoordinator(tmp_path, val_size=64, engine=REDIS)
        assert coord.workload_id == sweep_throughput_label(val_size=64, engine=REDIS) == "redis-get-k16-v64-t7-p10"


# ---------------------------------------------------------------------------
# Retirement of the epoch-1 Redis mirror
# ---------------------------------------------------------------------------


class TestRedisMirrorRetired:
    def test_every_redis_throughput_mirror_is_in_the_registry(self):
        expected = {f"throughput:{sweep_throughput_label(engine=REDIS)}"}
        for wl in SWEEP_THROUGHPUT_WORKLOADS:
            label = sweep_throughput_label(
                test=wl.get("test", "get"),
                val_size=wl["val_size"],
                io_threads=wl.get("io_threads", config.SWEEP_IO_THREADS),
                pipelining=wl.get("pipelining", config.SWEEP_PIPELINING),
                engine=REDIS,
            )
            expected.add(f"throughput:{label}")
        assert expected <= SWEEP_V1_RETIRED_SERIES
        assert len(expected) == 1 + len(SWEEP_THROUGHPUT_WORKLOADS)

    def test_redis_v1_coordinators_are_retired(self, tmp_path):
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            assert SweepCoordinator(tmp_path, engine=REDIS).retired is True
            assert SweepCoordinator(tmp_path, val_size=128, test="set", pipelining=1, engine=REDIS).retired is True

    def test_valkey_v1_throughput_workloads_are_all_retired(self, tmp_path):
        # The v3 roster now covers every v1 Valkey throughput workload, so the
        # non-default series are retired too (see test_epoch3 for the full set).
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            assert SweepCoordinator(tmp_path, val_size=64).retired is True
            assert SweepCoordinator(tmp_path, val_size=16, test="set").retired is True

    def test_redis_v3_series_are_not_retired(self, tmp_path):
        from conductress.sweep.coordinator_v3 import create_v3_coordinators

        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                coords = create_v3_coordinators(tmp_path, engine=REDIS)
        assert coords and all(c.retired is False for c in coords)

    def test_redis_memory_is_not_retired(self, tmp_path, monkeypatch):
        import conductress.sweep.memory_coordinator as mc

        monkeypatch.setattr(mc, "MEMORY_STATE_DIR", tmp_path)
        coords = mc.create_memory_coordinators(tmp_path / "redis", engine=REDIS)
        assert coords and all(c.retired is False for c in coords)


# ---------------------------------------------------------------------------
# The Redis v3 roster
# ---------------------------------------------------------------------------


class TestRedisV3Roster:
    @pytest.fixture
    def roster(self, tmp_path):
        from conductress.sweep.coordinator_v3 import create_v3_coordinators

        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                return create_v3_coordinators(tmp_path, engine=REDIS)

    def test_same_six_series_under_the_engine_prefix(self, roster):
        assert [c.workload_id for c in roster] == [
            "redis-get-k16-v16-t7-p10",
            "redis-mixed-s20-k16-v16-t7-p10",
            "redis-set-k16-v16-t7-p10",
            "redis-get-k16-v16-t7-p1",
            "redis-get-k16-v1024-t7-p10",
            "redis-get-k16-v16-t7-p1-r100k",
        ]
        assert all(c.epoch_id == "v3" for c in roster)

    def test_valkey_large_value_floor_is_not_applied_to_a_release_and_tip_engine(self, roster):
        """A Valkey tag means nothing in the Redis repo; the engine floor stands."""
        assert all(c.floor_tag is None for c in roster)
        assert {c._floor_tag for c in roster} == {REDIS.floor_tag}

    def test_cells_are_sourced_from_redis(self, roster):
        from conductress.sweep.planner import SweepTask

        task = roster[0]._create_task(
            SweepTask(commit="a" * 40, date="2026-01-01", priority=TaskPriority.NIGHTLY, reason="x")
        )
        assert task.source == "redis"
        assert task.perf_stat_enabled is True  # aggregate counters stay; the flamegraph is gated by profile_internals

    def test_planner_is_release_and_tip(self, roster):
        for coord in roster:
            assert coord.planner.tracks_history is False
            assert coord.planner.tip_interval_seconds == DAY

    def test_valkey_roster_is_unchanged(self, tmp_path):
        from conductress.sweep.coordinator_v3 import create_v3_coordinators

        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                coords = create_v3_coordinators(tmp_path)
        assert [c.workload_id for c in coords] == [
            "get-k16-v16-t7-p10",
            "mixed-s20-k16-v16-t7-p10",
            "set-k16-v16-t7-p10",
            "get-k16-v16-t7-p1",
            "get-k16-v1024-t7-p10",
            "get-k16-v16-t7-p1-r100k",
        ]
        assert all(c.planner.tracks_history for c in coords)


# ---------------------------------------------------------------------------
# Planner: scope switch and tip cadence
# ---------------------------------------------------------------------------


def _state(commits: List[str], points: Optional[Dict[str, float]] = None, **kw) -> SweepState:
    state = SweepState(
        merge_commits=commits, commit_dates={c: f"2026-01-{i+1:02d}" for i, c in enumerate(commits)}, **kw
    )
    for commit, value in (points or {}).items():
        state.points[commit] = BenchmarkPoint(
            commit=commit, date=state.commit_dates[commit], value=value, cv=0.2, status=PointStatus.COMPLETED
        )
    return state


class TestPlannerScope:
    COMMITS = [f"c{i:02d}" for i in range(20)]

    def test_history_planner_bisects_a_gap(self):
        state = _state(self.COMMITS, {"c00": 100.0, "c19": 50.0}, last_benchmarked_head="c19")
        task = SweepPlanner(state).get_next_task("c19")
        assert task is not None and task.priority in (TaskPriority.BISECTION, TaskPriority.BACKFILL)

    def test_release_and_tip_planner_never_bisects_or_backfills(self):
        state = _state(self.COMMITS, {"c00": 100.0, "c19": 50.0}, last_benchmarked_head="c19")
        assert SweepPlanner(state, tracks_history=False).get_next_task("c19") is None

    def test_release_and_tip_planner_still_measures_the_landmark(self):
        state = _state(self.COMMITS, landmarks=[Landmark(commit="c10", date="2026-01-11", label="8.10")])
        task = SweepPlanner(state, tracks_history=False).get_next_task(None)
        assert task is not None and task.priority == TaskPriority.LANDMARK and task.commit == "c10"

    def test_release_and_tip_planner_measures_an_unmeasured_tip(self):
        state = _state(self.COMMITS)
        task = SweepPlanner(state, tracks_history=False, tip_interval_seconds=DAY).get_next_task("c19")
        assert task is not None and task.priority == TaskPriority.NIGHTLY and task.commit == "c19"

    def test_tip_is_spaced_by_the_interval(self):
        now = [1_000_000.0]
        state = _state(self.COMMITS, {"c10": 100.0}, last_benchmarked_head="c10", last_benchmarked_head_at=now[0])
        planner = SweepPlanner(state, tracks_history=False, tip_interval_seconds=DAY, clock=lambda: now[0])
        now[0] += DAY / 2
        assert planner.tip_due("c19") is False
        assert planner.get_next_task("c19") is None
        now[0] += DAY / 2
        assert planner.tip_due("c19") is True
        task = planner.get_next_task("c19")
        assert task is not None and task.commit == "c19"

    def test_history_planner_ignores_the_interval(self):
        state = _state(self.COMMITS, {"c10": 100.0}, last_benchmarked_head="c10", last_benchmarked_head_at=1_000_000.0)
        planner = SweepPlanner(state, clock=lambda: 1_000_001.0)
        assert planner.tip_due("c19") is True

    def test_an_already_measured_tip_is_not_due(self):
        state = _state(self.COMMITS, {"c19": 100.0})
        assert SweepPlanner(state, tracks_history=False).tip_due("c19") is False
        assert SweepPlanner(state).tip_due(None) is False


class TestLatestReleaseLandmark:
    def test_keeps_the_newest_release_only(self):
        commits = ["a", "b", "c", "d"]
        lms = [
            Landmark(commit="a", date="", label="8.0"),
            Landmark(commit="c", date="", label="8.10"),
            Landmark(commit="b", date="", label="8.2"),
            Landmark(commit="d", date="", label="First benchmarkable"),
        ]
        assert latest_release_landmark(lms, commits) == [lms[1]]

    def test_empty_when_no_release_exists(self):
        assert latest_release_landmark([Landmark(commit="a", date="", label="marker")], ["a"]) == []
        assert latest_release_landmark([], []) == []

    def test_ignores_a_release_that_is_not_on_the_sweep_ref(self):
        lms = [Landmark(commit="zz", date="", label="9.9"), Landmark(commit="a", date="", label="8.0")]
        assert latest_release_landmark(lms, ["a"]) == [lms[1]]


# ---------------------------------------------------------------------------
# Coordinator: landmarks, sample tagging, tip check
# ---------------------------------------------------------------------------


def _coord(tmp_path: Path, engine: Optional[SweepEngine], commits: List[str]) -> SweepCoordinator:
    with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
        coord = SweepCoordinator(tmp_path / "repo", engine=engine)
    coord.state.merge_commits = list(commits)
    coord.state.commit_dates = {c: "2026-01-01" for c in commits}
    coord.planner = coord._new_planner()
    return coord


RELEASE_POINTS = [("c02", "2025-01-01", "8.0"), ("c05", "2025-06-01", "8.2"), ("c08", "2026-01-01", "8.10")]


class TestCoordinatorLandmarks:
    COMMITS = [f"c{i:02d}" for i in range(10)]

    def test_landmarks_are_rebuilt_not_appended(self, tmp_path):
        coord = _coord(tmp_path, None, self.COMMITS)
        with patch("conductress.sweep.coordinator.get_release_branch_points", return_value=RELEASE_POINTS):
            coord._populate_landmarks()
            coord._populate_landmarks()
            coord._populate_landmarks()
        assert [lm.label for lm in coord.state.landmarks] == ["8.0", "8.2", "8.10"]

    def test_release_and_tip_engine_keeps_the_latest_release_only(self, tmp_path):
        coord = _coord(tmp_path, REDIS, self.COMMITS)
        with patch("conductress.sweep.coordinator.get_release_branch_points", return_value=RELEASE_POINTS):
            coord._populate_landmarks()
        assert [(lm.commit, lm.label) for lm in coord.state.landmarks] == [("c08", "8.10")]

    def test_history_engine_keeps_every_release(self, tmp_path):
        coord = _coord(tmp_path, get_sweep_engine("valkey"), self.COMMITS)
        with patch("conductress.sweep.coordinator.get_release_branch_points", return_value=RELEASE_POINTS):
            coord._populate_landmarks()
        assert [lm.label for lm in coord.state.landmarks] == ["8.0", "8.2", "8.10"]


class TestSampleTagging:
    COMMITS = [f"c{i:02d}" for i in range(10)]

    def test_release_tip_and_history_are_tagged(self, tmp_path):
        coord = _coord(tmp_path, REDIS, self.COMMITS)
        coord.state.landmarks = [Landmark(commit="c05", date="2026-01-01", label="8.10")]
        with (
            patch("conductress.sweep.coordinator.get_head", return_value="c09"),
            patch("conductress.sweep.coordinator.time.time", return_value=5_000.0),
        ):
            coord.record_result("c05", 100.0, 0.5, 5)
            coord.record_result("c09", 101.0, 0.5, 5)
            coord.record_result("c07", 99.0, 0.5, 5)
        assert coord.state.points["c05"].sample == "release"
        assert coord.state.points["c09"].sample == "tip"
        assert coord.state.points["c07"].sample == "history"
        assert coord.state.last_benchmarked_head == "c09"
        assert coord.state.last_benchmarked_head_at == 5_000.0

    def test_tagging_survives_a_head_lookup_failure(self, tmp_path):
        coord = _coord(tmp_path, None, self.COMMITS)
        with patch("conductress.sweep.coordinator.get_head", side_effect=RuntimeError("no repo")):
            coord.record_result("c03", 100.0, 0.5, 5)
        assert coord.state.points["c03"].sample == "history"
        assert coord.state.last_benchmarked_head_at is None


class TestHasNightlyTask:
    COMMITS = [f"c{i:02d}" for i in range(10)]

    def test_redis_tip_waits_for_the_interval(self, tmp_path):
        coord = _coord(tmp_path, REDIS, self.COMMITS)
        coord.state.last_benchmarked_head = "c05"
        coord.state.last_benchmarked_head_at = 1_000.0
        coord.state.points["c05"] = BenchmarkPoint(commit="c05", date="", value=1.0, status=PointStatus.COMPLETED)
        coord.planner = coord._new_planner()
        coord.planner._clock = lambda: 1_000.0 + DAY / 2
        with patch("conductress.sweep.coordinator.get_head", return_value="c09"):
            assert coord.has_nightly_task() is False
        coord.planner._clock = lambda: 1_000.0 + DAY
        with patch("conductress.sweep.coordinator.get_head", return_value="c09"):
            assert coord.has_nightly_task() is True

    def test_valkey_tip_is_measured_as_soon_as_it_moves(self, tmp_path):
        coord = _coord(tmp_path, None, self.COMMITS)
        coord.state.last_benchmarked_head = "c08"
        coord.state.last_benchmarked_head_at = 1_000.0
        coord.state.points["c08"] = BenchmarkPoint(commit="c08", date="", value=1.0, status=PointStatus.COMPLETED)
        with patch("conductress.sweep.coordinator.get_head", return_value="c09"):
            assert coord.has_nightly_task() is True

    def test_a_failed_tip_is_retried(self, tmp_path):
        coord = _coord(tmp_path, REDIS, self.COMMITS)
        coord.state.points["c09"] = BenchmarkPoint(commit="c09", date="", status=PointStatus.BUILD_FAILED)
        with patch("conductress.sweep.coordinator.get_head", return_value="c09"):
            assert coord.has_nightly_task() is True

    def test_a_tip_outside_the_commit_list_is_not_a_task(self, tmp_path):
        coord = _coord(tmp_path, REDIS, self.COMMITS)
        with (
            patch("conductress.sweep.coordinator.get_head", return_value="zz"),
            patch.object(coord, "_fetch_and_refresh"),
        ):
            assert coord.has_nightly_task() is False


# ---------------------------------------------------------------------------
# State and export
# ---------------------------------------------------------------------------


class TestStateAndExport:
    def test_state_round_trips_the_new_fields(self, tmp_path):
        state = _state(["a", "b"], {"a": 1.0}, last_benchmarked_head="a", last_benchmarked_head_at=123.5)
        state.points["a"].sample = "tip"
        path = tmp_path / "s.json"
        state.save(path)
        loaded = SweepState.load(path)
        assert loaded.last_benchmarked_head_at == 123.5
        assert loaded.points["a"].sample == "tip"

    def test_old_state_files_load_without_the_fields(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(
            json.dumps({"merge_commits": ["a"], "points": {"a": {"commit": "a", "value": 1.0, "status": "COMPLETED"}}})
        )
        loaded = SweepState.load(path)
        assert loaded.last_benchmarked_head_at is None
        assert loaded.points["a"].sample is None

    def test_export_names_the_engine_and_scope(self, tmp_path):
        state = _state(["a"], {"a": 1.0})
        state.points["a"].sample = "release"
        out = tmp_path / "series.json"
        export_series(state, out, workload="redis-get", repo=engine_repo_slug(REDIS), engine=REDIS)
        data = json.loads(out.read_text())
        assert data["metadata"]["engine"] == "redis"
        assert data["metadata"]["scope"] == "release-and-tip"
        assert data["metadata"]["repo"] == "redis/redis"
        assert data["points"][0]["sample"] == "release"

    def test_export_defaults_to_the_valkey_history_sweep(self, tmp_path):
        state = _state(["a"], {"a": 1.0})
        out = tmp_path / "series.json"
        export_series(state, out, workload="get")
        data = json.loads(out.read_text())
        assert data["metadata"]["engine"] == "valkey"
        assert data["metadata"]["scope"] == "history"
        assert "sample" not in data["points"][0]

    def test_redis_memory_export_links_the_redis_repo(self, tmp_path, monkeypatch):
        import conductress.sweep.memory_coordinator as mc

        monkeypatch.setattr(mc, "MEMORY_STATE_DIR", tmp_path)
        coord = mc.create_memory_coordinators(tmp_path / "redis", engine=REDIS)[0]
        coord.state.merge_commits = ["a"]
        coord.state.commit_dates = {"a": "2026-01-01"}
        coord.state.points["a"] = BenchmarkPoint(
            commit="a", date="2026-01-01", value=80.0, status=PointStatus.COMPLETED
        )
        out = tmp_path / "mem.json"
        coord.export(out, platform="test")
        data = json.loads(out.read_text())
        assert data["metadata"]["repo"] == "redis/redis"
        assert data["metadata"]["engine"] == "redis"

    def test_engines_table_has_exactly_one_history_engine(self):
        assert [e.source for e in SWEEP_ENGINES if e.tracks_history] == ["valkey"]
