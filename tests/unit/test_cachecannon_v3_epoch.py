"""Tests for the cachecannon-v3 sweep epoch.

Three defect classes are guarded here, each of which has actually shipped before:

* ``sweep_commit`` declared as a dynamic attribute instead of a dataclass field,
  so ``asdict()`` silently dropped it and the coordinator rejected its own
  reloaded completion (the PR #178 defect).
* An epoch's ownership predicate matching another epoch's tasks, which
  contaminated 20 v1 GET points (the PR #177 defect).
* A declared task field that does not affect execution -- a "fake lever" that
  reports a precision target which never took effect.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from conductress.config import (
    SWEEP_EPOCHS,
    SWEEP_V3_CONNECTIONS,
    SWEEP_V3_DURATION,
    SWEEP_V3_KEYSPACE,
    SWEEP_V3_MAX_REPS,
    SWEEP_V3_REPETITIONS,
    SWEEP_V3_TARGET_CV,
    SWEEP_V3_WARMUP,
)
from conductress.sweep.planner import SweepTask
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_cachecannon import CachecannonTaskData


@pytest.fixture(autouse=True)
def _ensure_valid_source(monkeypatch):
    import conductress.config as cfg

    if "valkey" not in cfg.REPO_NAMES:
        monkeypatch.setattr(cfg, "REPO_NAMES", cfg.REPO_NAMES + ["valkey"])


def _sweep_task(commit: str = "a" * 40) -> SweepTask:
    return SweepTask(commit=commit, reason="unit test", date="2026-01-01", priority=1)


def _v3_get(tmp_path: Path):
    from conductress.sweep.coordinator_v3 import CachecannonThroughputSweepCoordinatorV3

    with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
        with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
            return CachecannonThroughputSweepCoordinatorV3(tmp_path)


def _v3_mixed(tmp_path: Path):
    from conductress.sweep.coordinator_v3 import CachecannonMixedSweepCoordinatorV3

    with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
        with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
            return CachecannonMixedSweepCoordinatorV3(tmp_path)


class TestV3TaskSerialization:
    """sweep_commit and the adaptive controls must survive a real round-trip."""

    def test_sweep_commit_is_a_real_dataclass_field(self):
        from dataclasses import fields

        names = {f.name for f in fields(CachecannonTaskData)}
        assert "sweep_commit" in names
        assert "max_reps" in names
        assert "target_cv" in names

    def test_sweep_commit_survives_save_load_roundtrip(self, tmp_path: Path):
        """The PR #178 defect: a dynamic attribute is dropped by asdict()."""
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        task.sweep_commit = "a" * 40

        path = tmp_path / "task.json"
        task.save_to_file(path)
        reloaded = BaseTaskData.from_file(path)

        assert isinstance(reloaded, CachecannonTaskData)
        assert reloaded.sweep_commit == "a" * 40

    def test_protocol_values_survive_roundtrip(self, tmp_path: Path):
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        task.sweep_commit = "a" * 40

        path = tmp_path / "task.json"
        task.save_to_file(path)
        reloaded = BaseTaskData.from_file(path)

        assert reloaded.warmup == SWEEP_V3_WARMUP
        assert reloaded.duration == SWEEP_V3_DURATION
        assert reloaded.connections == SWEEP_V3_CONNECTIONS
        assert reloaded.keyspace_count == SWEEP_V3_KEYSPACE
        assert reloaded.repetitions == SWEEP_V3_REPETITIONS
        assert reloaded.max_reps == SWEEP_V3_MAX_REPS
        assert reloaded.target_cv == SWEEP_V3_TARGET_CV

    def test_coordinator_claims_its_own_reloaded_completion(self, tmp_path: Path):
        """The exact failure mode of PR #178: reloaded task not recognised."""
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        task.sweep_commit = "a" * 40

        path = tmp_path / "task.json"
        task.save_to_file(path)
        reloaded = BaseTaskData.from_file(path)

        assert coord._is_my_task(reloaded)

    def test_pre_existing_queued_document_still_loads(self, tmp_path: Path):
        """Documents queued before these fields existed must not fail to load.

        The three new fields are additive with defaults, so a task already
        sitting in a runner mailbox at deploy time deserializes unchanged. This
        is the real backward-compatibility property; the golden-fixture test
        only pins the current schema's round-trip.
        """
        import json

        legacy = {
            "source": "valkey",
            "specifier": "abc123",
            "replicas": 0,
            "note": "queued before v3",
            "requirements": {},
            "make_args": "",
            "task_type": "CachecannonTaskData",
            "timestamp": "2026-08-29T00:00:00.123456",
            "test": "get",
            "val_size": 512,
            "pipelining": 10,
            "connections": 1200,
            "threads": 16,
            "warmup": 5,
            "duration": 30,
            "repetitions": 3,
            "keyspace_count": 3_000_000,
            "io_threads": 9,
            "set_ratio": 0,
            "distribution": "uniform",
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))

        task = BaseTaskData.from_file(path)
        assert isinstance(task, CachecannonTaskData)
        assert task.sweep_commit == ""
        assert task.max_reps == 0
        assert task.target_cv == 0.0
        # And it must not be absorbed into the v3 sweep series.
        assert not _v3_get(tmp_path)._is_my_task(task)


class TestV3ProtocolValues:
    """Stage 2 measured these; a silent change would redefine the series."""

    def test_warmup_is_ten_not_thirty(self):
        assert SWEEP_V3_WARMUP == 10

    def test_connections_is_four_hundred(self):
        assert SWEEP_V3_CONNECTIONS == 400

    def test_created_task_uses_measured_protocol(self, tmp_path: Path):
        task = _v3_get(tmp_path)._create_task(_sweep_task())
        assert task.warmup == 10
        assert task.duration == 30
        assert task.connections == 400
        assert task.threads == 8
        assert task.io_threads == 7
        assert task.pipelining == 10
        assert task.val_size == 16
        assert task.keyspace_count == 3_000_000
        assert task.distribution == "uniform"


class TestV3EpochIsolation:
    """v3 must be disjoint from legacy v1 and the retired v2 namespace."""

    def test_state_dir_is_isolated(self):
        from conductress.sweep.coordinator_v3 import V3_STATE_DIR

        assert V3_STATE_DIR.name == "v3"

    def test_epoch_id_and_state_file_name(self, tmp_path: Path):
        coord = _v3_get(tmp_path)
        assert coord.epoch_id == "v3"
        assert "v3" in coord.state_file.name

    def test_workload_ids_are_canonical(self, tmp_path: Path):
        assert _v3_get(tmp_path).workload_id == "get-k16-v16-t7-p10"
        assert _v3_mixed(tmp_path).workload_id == "mixed-s20-k16-v16-t7-p10"

    def test_v1_does_not_claim_a_v3_task(self, tmp_path: Path):
        from conductress.sweep.coordinator import SweepCoordinator

        task = _v3_get(tmp_path)._create_task(_sweep_task())
        task.sweep_commit = "a" * 40
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            v1 = SweepCoordinator(tmp_path)
        assert not v1._is_my_task(task)

    def test_retired_v2_coordinators_do_not_claim_a_v3_task(self, tmp_path: Path):
        from conductress.sweep.coordinator_v2 import MixedSweepCoordinatorV2, ThroughputSweepCoordinatorV2

        task = _v3_get(tmp_path)._create_task(_sweep_task())
        task.sweep_commit = "a" * 40
        with patch("conductress.sweep.coordinator_v2._ensure_v2_state_dir"):
            with patch("conductress.sweep.coordinator_v2.V2_STATE_DIR", tmp_path):
                assert not ThroughputSweepCoordinatorV2(tmp_path)._is_my_task(task)
                assert not MixedSweepCoordinatorV2(tmp_path)._is_my_task(task)

    def test_v3_does_not_claim_v1_or_v2_tasks(self, tmp_path: Path):
        from conductress.tasks.task_mixed import MixedTaskData
        from conductress.tasks.task_perf_benchmark import PerfTaskData

        perf = PerfTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="v1 task",
            requirements={},
            make_args="",
            test="get",
            val_size=16,
            io_threads=7,
            pipelining=10,
            warmup=5,
            duration=30,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )
        perf.sweep_commit = "abc123"
        mixed = MixedTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="v2 mixed task",
            requirements={},
            make_args="",
            set_ratio=20,
            val_size=16,
            io_threads=7,
            pipelining=10,
            duration=30,
            warmup=5,
        )
        mixed.sweep_commit = "abc123"

        coord = _v3_get(tmp_path)
        assert not coord._is_my_task(perf)
        assert not coord._is_my_task(mixed)

    def test_get_and_mixed_v3_coordinators_do_not_claim_each_other(self, tmp_path: Path):
        get_coord, mixed_coord = _v3_get(tmp_path), _v3_mixed(tmp_path)
        get_task = get_coord._create_task(_sweep_task())
        get_task.sweep_commit = "a" * 40
        mixed_task = mixed_coord._create_task(_sweep_task())
        mixed_task.sweep_commit = "a" * 40

        assert get_coord._is_my_task(get_task)
        assert not get_coord._is_my_task(mixed_task)
        assert mixed_coord._is_my_task(mixed_task)
        assert not mixed_coord._is_my_task(get_task)

    def test_manual_cell_without_sweep_commit_is_not_absorbed(self, tmp_path: Path):
        """Manually queued diagnostic cells must never join the sweep series."""
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        assert task.sweep_commit == ""
        assert not coord._is_my_task(task)

    def test_v3_completion_does_not_reach_v1_result_extraction(self, tmp_path: Path):
        from conductress.sweep.coordinator import SweepCoordinator

        task = _v3_get(tmp_path)._create_task(_sweep_task())
        task.sweep_commit = "a" * 40
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            v1 = SweepCoordinator(tmp_path)
        with patch.object(v1, "_extract_result", return_value=None) as extract:
            v1.on_task_completed(task)
        extract.assert_not_called()

    def test_v3_failure_does_not_mark_v1_build_failure(self, tmp_path: Path):
        from conductress.sweep.coordinator import SweepCoordinator

        task = _v3_get(tmp_path)._create_task(_sweep_task())
        task.sweep_commit = "a" * 40
        with patch("conductress.sweep.coordinator.SWEEP_STATE_DIR", tmp_path):
            v1 = SweepCoordinator(tmp_path)
        with patch.object(v1, "record_build_failure") as record:
            v1.on_task_failed(task)
        record.assert_not_called()


class TestAdaptiveRepetitionsAreReal:
    """target_cv/max_reps must affect execution, not merely be accepted."""

    def test_runner_receives_adaptive_controls(self, tmp_path: Path):
        task = _v3_get(tmp_path)._create_task(_sweep_task())
        runner = task.prepare_task_runner([])
        assert runner.max_reps == SWEEP_V3_MAX_REPS
        assert runner.target_cv == SWEEP_V3_TARGET_CV

    def test_step_budget_uses_adaptive_ceiling(self, tmp_path: Path):
        """Progress must not exceed 100% when extra reps are needed."""
        task = _v3_get(tmp_path)._create_task(_sweep_task())
        runner = task.prepare_task_runner([])
        assert runner.status.steps_total == (SWEEP_V3_WARMUP + SWEEP_V3_DURATION) * SWEEP_V3_MAX_REPS

    def test_target_cv_without_headroom_is_rejected(self):
        """A target that can never fire is a fake lever, not a default."""
        with pytest.raises(ValueError, match="no-op"):
            CachecannonTaskData(
                source="valkey",
                specifier="abc123",
                replicas=0,
                note="n",
                requirements={},
                make_args="",
                repetitions=5,
                max_reps=5,
                target_cv=0.5,
            )

    def test_max_reps_below_repetitions_is_rejected(self):
        with pytest.raises(ValueError, match="max_reps"):
            CachecannonTaskData(
                source="valkey",
                specifier="abc123",
                replicas=0,
                note="n",
                requirements={},
                make_args="",
                repetitions=5,
                max_reps=3,
            )

    def test_negative_target_cv_is_rejected(self):
        with pytest.raises(ValueError, match="target_cv"):
            CachecannonTaskData(
                source="valkey",
                specifier="abc123",
                replicas=0,
                note="n",
                requirements={},
                make_args="",
                target_cv=-1.0,
            )

    def test_fixed_reps_remain_the_default(self):
        """Manual cachecannon cells keep their previous fixed-rep behaviour."""
        task = CachecannonTaskData(
            source="valkey",
            specifier="abc123",
            replicas=0,
            note="n",
            requirements={},
            make_args="",
            repetitions=3,
        )
        assert task.max_reps == 0
        assert task.target_cv == 0.0
        runner = task.prepare_task_runner([])
        assert runner.status.steps_total == (task.warmup + task.duration) * 3

    def test_adaptive_stop_fires_only_after_minimum_reps(self):
        from conductress.tasks.task_cachecannon import _should_stop_adaptive

        identical = [1_000_000.0, 1_000_000.0, 1_000_000.0]
        # rep index 1 is below the 5-rep minimum even with zero variance
        assert not _should_stop_adaptive(identical, 1, 5, 0.5)
        assert _should_stop_adaptive(identical + [1_000_000.0, 1_000_000.0], 4, 5, 0.5)

    def test_adaptive_stop_does_not_fire_when_disabled(self):
        from conductress.tasks.task_cachecannon import _should_stop_adaptive

        identical = [1_000_000.0] * 5
        assert not _should_stop_adaptive(identical, 4, 5, 0.0)

    def test_noisy_results_do_not_stop_early(self):
        from conductress.tasks.task_cachecannon import _should_stop_adaptive

        # AMD/1200c-like spread: ~6% between restarts must not satisfy 0.5%
        noisy = [2_100_000.0, 2_230_000.0, 2_090_000.0, 2_240_000.0, 2_150_000.0]
        assert not _should_stop_adaptive(noisy, 4, 5, 0.5)

    @pytest.mark.parametrize(
        "label,mean,cv_pct,expect_stop",
        [
            ("G4 400c", 3.11e6, 0.280, True),
            ("G4 1200c", 3.00e6, 0.297, True),
            ("AMD 400c", 2.11e6, 0.296, True),
            # The rejected config: 2.520% CV is a 3.13% CI95 half-width at n=5,
            # so every cell would burn the full 10-rep ceiling.
            ("AMD 1200c", 2.21e6, 2.520, False),
        ],
    )
    def test_measured_stage2_variance_converges_at_minimum_reps(self, label, mean, cv_pct, expect_stop):
        """The chosen protocol must actually converge on the measured hardware.

        Variance figures are the Stage 2 between-restart CVs (5 fresh starts per
        cell). Note the target bounds the 95% CI half-width, which at n=5 is
        about 1.24x the CV -- so 0.30% CV is really ~0.37%, inside the 0.5%
        target but with less margin than the raw CV suggests.
        """
        import statistics

        from conductress.tasks.task_cachecannon import _should_stop_adaptive

        base = [-2.0, -1.0, 0.0, 1.0, 2.0]
        scale = (mean * cv_pct / 100) / statistics.stdev(base)
        samples = [mean + b * scale for b in base]

        assert _should_stop_adaptive(samples, 4, SWEEP_V3_REPETITIONS, SWEEP_V3_TARGET_CV) is expect_stop, label


class TestV3ResultExtraction:
    def test_reps_come_from_recorded_runs_not_configuration(self, tmp_path: Path):
        """With adaptive reps the count is only knowable from the result."""
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        entry = {"score": 3_104_882.0, "data": {"per_run_rps": [3_104_882.0, 3_100_100.0, 3_102_000.0]}}
        with patch.object(coord, "_find_task_entry", return_value=entry):
            value, cv, reps = coord._extract_result(task)
        assert value == 3_104_882.0
        assert reps == 3
        assert cv > 0

    def test_missing_entry_returns_none(self, tmp_path: Path):
        coord = _v3_get(tmp_path)
        task = coord._create_task(_sweep_task())
        with patch.object(coord, "_find_task_entry", return_value=None):
            assert coord._extract_result(task) is None


class TestEpochRegistry:
    def test_v3_is_registered_with_cachecannon_generator(self):
        assert SWEEP_EPOCHS["v3"]["generator"] == "cachecannon"

    def test_publisher_labels_v3_distinctly(self):
        from conductress.publisher import DashboardPublisher

        v1 = DashboardPublisher._epoch_def("v1")
        v2 = DashboardPublisher._epoch_def("v2")
        v3 = DashboardPublisher._epoch_def("v3")
        assert v1["id"] == "v1" and v2["id"] == "v2" and v3["id"] == "v3"
        # The old binary v1-or-else expression labelled v3 as "Scalable v2".
        assert v3["label"] != v2["label"]
        assert len({v1["label"], v2["label"], v3["label"]}) == 3

    def test_unregistered_epoch_gets_generic_label(self):
        from conductress.publisher import DashboardPublisher

        entry = DashboardPublisher._epoch_def("v99")
        assert entry["generator"] == "unknown"
        assert entry["label"] != SWEEP_EPOCHS["v2"]["label"]

    def test_v3_series_filename_is_epoch_qualified(self):
        from conductress.publisher import DashboardPublisher

        base = Path("series-graviton4-get-k16-v16-t7-p10-throughput.json")
        assert DashboardPublisher._epoch_path(base, "v3").name == (
            "series-graviton4-get-k16-v16-t7-p10-throughput.epoch-v3.json"
        )

    def test_v1_paths_remain_unqualified(self):
        from conductress.publisher import DashboardPublisher

        base = Path("series-graviton4-get-k16-v16-t7-p10-throughput.json")
        assert DashboardPublisher._epoch_path(base, "v1") == base


class TestV3Roster:
    def test_roster_opens_with_two_workloads(self, tmp_path: Path):
        from conductress.sweep.coordinator_v3 import create_v3_coordinators

        with patch("conductress.sweep.coordinator_v3._ensure_v3_state_dir"):
            with patch("conductress.sweep.coordinator_v3.V3_STATE_DIR", tmp_path):
                coords = create_v3_coordinators(tmp_path)
        assert [c.workload_id for c in coords] == [
            "get-k16-v16-t7-p10",
            "mixed-s20-k16-v16-t7-p10",
        ]
        assert all(c.epoch_id == "v3" for c in coords)

    def test_v3_is_disabled_by_default(self, monkeypatch):
        """The toggle must default off so deploys land inert.

        Checked through ``_env_bool`` rather than by reloading ``config``:
        reloading rebinds module-level state that other tests already hold
        references to.
        """
        from conductress.config import _env_bool

        monkeypatch.delenv("CONDUCTRESS_SWEEP_V3_ENABLED", raising=False)
        assert _env_bool("CONDUCTRESS_SWEEP_V3_ENABLED", False) is False

    def test_v3_toggle_accepts_explicit_enable(self, monkeypatch):
        from conductress.config import _env_bool

        monkeypatch.setenv("CONDUCTRESS_SWEEP_V3_ENABLED", "1")
        assert _env_bool("CONDUCTRESS_SWEEP_V3_ENABLED", False) is True

    def test_v3_toggle_rejects_invalid_values(self, monkeypatch):
        """Fail-closed at startup rather than silently defaulting."""
        from conductress.config import _env_bool

        monkeypatch.setenv("CONDUCTRESS_SWEEP_V3_ENABLED", "maybe")
        with pytest.raises(ValueError):
            _env_bool("CONDUCTRESS_SWEEP_V3_ENABLED", False)
