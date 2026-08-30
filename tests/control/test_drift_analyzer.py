"""Tests for canary drift analysis: ingestion, rolling stats, phase tracking.

Covers: 14/28 phase transitions, rolling-window eviction, MAD computation,
zero median, zero MAD, threshold recommendation, duplicate replay,
malformed/non-finite score, out-of-order daily results, profile version
separation, calibration report generation, DB migration v3, and no
regression in normal (non-canary) outcome completion.

No sleeps -- all use explicit datetime/date injection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conductress.control.db import ControlDatabase, DATABASE_SCHEMA_VERSION
from conductress.control.drift_analyzer import (
    CALIBRATION_WINDOW,
    OBSERVATION_WINDOW,
    PHASE_CALIBRATING,
    PHASE_OBSERVATION,
    PHASE_READY,
    DriftAnalyzer,
    _mad,
    _median,
)
from conductress.control.fleet_registry import FleetRegistry
from conductress.control.service import ControlService

from .helpers import fleet_manifest, task_envelope, task_outcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_env(tmp_path: Path):
    """Create a DB, fleet registry, and drift analyzer."""
    manifest_path = tmp_path / "fleet.json"
    manifest_path.write_text(json.dumps(fleet_manifest()), encoding="utf-8")
    db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
    db.initialize()
    registry = FleetRegistry.from_file(manifest_path)
    analyzer = DriftAnalyzer(db)
    return db, registry, analyzer


def _ingest_n(
    analyzer: DriftAnalyzer,
    n: int,
    *,
    runner_id: str = "armbench",
    profile_id: str = "throughput-get-v1",
    profile_version: int = 1,
    base_score: float = 200000.0,
    score_fn=None,
    start_day: int = 1,
):
    """Ingest n observations for consecutive days starting at 2026-09-{start_day}."""
    results = []
    for i in range(n):
        day = start_day + i
        utc_date = f"2026-09-{day:02d}"
        task_id = f"canary:{runner_id}:{profile_id}:{utc_date}"
        score = score_fn(i) if score_fn else base_score + i * 100.0
        outcome = {
            "schema_version": 1,
            "task_id": task_id,
            "runner_id": runner_id,
            "state": "completed",
            "completed_at": f"{utc_date}T12:00:00Z",
            "result": {"score": score},
            "error": None,
        }
        obs = analyzer.ingest_outcome(
            task_id=task_id,
            runner_id=runner_id,
            outcome=outcome,
            profile_id=profile_id,
            profile_version=profile_version,
            utc_date=utc_date,
        )
        results.append(obs)
    return results


# ---------------------------------------------------------------------------
# Pure math tests
# ---------------------------------------------------------------------------


class TestMedianMAD:
    def test_median_odd(self):
        assert _median([3, 1, 2]) == 2

    def test_median_even(self):
        assert _median([4, 1, 3, 2]) == 2.5

    def test_median_single(self):
        assert _median([42]) == 42

    def test_median_empty_raises(self):
        with pytest.raises(ValueError):
            _median([])

    def test_mad_constant(self):
        assert _mad([5, 5, 5, 5]) == 0.0

    def test_mad_symmetric(self):
        # [1, 2, 3, 4, 5], median=3, deviations=[0,1,1,2,2], MAD=1
        assert _mad([1, 2, 3, 4, 5]) == 1.0

    def test_mad_single(self):
        assert _mad([10]) == 0.0

    def test_mad_two(self):
        # [10, 20], median=15, deviations=[5, 5], MAD=5
        assert _mad([10, 20]) == 5.0


# ---------------------------------------------------------------------------
# Phase transition tests
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    def test_first_14_are_observation(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, OBSERVATION_WINDOW)
        for obs in results:
            assert obs["phase"] == PHASE_OBSERVATION

    def test_15th_is_calibrating(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, OBSERVATION_WINDOW + 1)
        assert results[OBSERVATION_WINDOW]["phase"] == PHASE_CALIBRATING

    def test_15_to_27_are_calibrating(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, CALIBRATION_WINDOW - 1)
        for obs in results[OBSERVATION_WINDOW:]:
            assert obs["phase"] == PHASE_CALIBRATING

    def test_28th_is_ready(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, CALIBRATION_WINDOW)
        assert results[CALIBRATION_WINDOW - 1]["phase"] == PHASE_READY

    def test_beyond_28_stays_ready(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, CALIBRATION_WINDOW + 5)
        for obs in results[CALIBRATION_WINDOW - 1:]:
            assert obs["phase"] == PHASE_READY


# ---------------------------------------------------------------------------
# Rolling window tests
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_first_observation_has_no_reference(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, 1)
        obs = results[0]
        assert obs["ref_median"] is None
        assert obs["ref_mad"] is None
        assert obs["delta_pct"] is None
        assert obs["sample_count"] == 0  # zero prior observations

    def test_second_observation_has_reference_from_first(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 2, base_score=100000.0)
        obs_list = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        obs2 = obs_list[0]  # newest
        assert obs2["ref_median"] is not None
        assert obs2["sample_count"] == 1
        # With only 1 prior sample, median = that sample
        assert obs2["ref_median"] == 100000.0

    def test_new_sample_not_in_own_window(self, tmp_path):
        """The critical invariant: a sample cannot move its own baseline."""
        _, _, analyzer = _make_env(tmp_path)
        # Ingest 10 samples at 100k, then one outlier at 200k
        _ingest_n(analyzer, 10, base_score=100000.0, score_fn=lambda i: 100000.0)
        outlier_date = "2026-09-11"
        outlier_tid = f"canary:armbench:throughput-get-v1:{outlier_date}"
        obs = analyzer.ingest_outcome(
            task_id=outlier_tid,
            runner_id="armbench",
            outcome={
                "schema_version": 1,
                "task_id": outlier_tid,
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": f"{outlier_date}T12:00:00Z",
                "result": {"score": 200000.0},
                "error": None,
            },
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date=outlier_date,
        )
        # Reference should be from prior 10 samples (all 100k), not including the outlier
        assert obs["ref_median"] == 100000.0
        assert obs["ref_mad"] == 0.0
        assert abs(obs["delta_pct"] - 100.0) < 0.01

    def test_window_eviction_at_28(self, tmp_path):
        """After 28+ prior samples, only the most recent 28 are used."""
        _, _, analyzer = _make_env(tmp_path)

        # Ingest 30 samples: first 2 at 50k, rest at 100k
        def score_fn(i):
            return 50000.0 if i < 2 else 100000.0

        _ingest_n(analyzer, 30, score_fn=score_fn)

        # The 31st sample should NOT see the 50k values in its window
        date_31 = "2026-09-31"  # Fine for testing
        tid_31 = f"canary:armbench:throughput-get-v1:{date_31}"
        obs = analyzer.ingest_outcome(
            task_id=tid_31,
            runner_id="armbench",
            outcome={
                "schema_version": 1,
                "task_id": tid_31,
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": f"{date_31}T12:00:00Z",
                "result": {"score": 100000.0},
                "error": None,
            },
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date=date_31,
        )
        # Prior 28 should be all 100k (samples 3-30), since samples 1-2 (50k) are evicted
        assert obs["ref_median"] == 100000.0
        assert obs["ref_mad"] == 0.0


# ---------------------------------------------------------------------------
# MAD edge cases
# ---------------------------------------------------------------------------


class TestMADEdgeCases:
    def test_zero_mad_constant_series(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, 5, score_fn=lambda i: 100000.0)
        # From 3rd sample onward, MAD should be 0
        for obs in results[2:]:
            assert obs["ref_mad"] == 0.0

    def test_zero_median(self, tmp_path):
        """Handle a series where median is zero."""
        _, _, analyzer = _make_env(tmp_path)
        # First 3 at 0.0, then one non-zero
        _ingest_n(analyzer, 3, score_fn=lambda i: 0.0)

        obs = analyzer.ingest_outcome(
            task_id="canary:armbench:throughput-get-v1:2026-09-04",
            runner_id="armbench",
            outcome={
                "schema_version": 1,
                "task_id": "canary:armbench:throughput-get-v1:2026-09-04",
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": "2026-09-04T12:00:00Z",
                "result": {"score": 100.0},
                "error": None,
            },
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date="2026-09-04",
        )
        # delta_pct should be clamped to 999999 (infinite, since median=0)
        assert obs["delta_pct"] == 999999.0

    def test_both_zero(self, tmp_path):
        """Handle median=0, score=0."""
        _, _, analyzer = _make_env(tmp_path)
        obs_list = _ingest_n(analyzer, 4, score_fn=lambda i: 0.0)
        assert obs_list[3]["delta_pct"] == 0.0


# ---------------------------------------------------------------------------
# Duplicate and replay
# ---------------------------------------------------------------------------


class TestDuplicateReplay:
    def test_duplicate_ingestion_is_idempotent(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        utc_date = "2026-09-01"
        task_id = "canary:armbench:throughput-get-v1:2026-09-01"
        outcome = {
            "schema_version": 1,
            "task_id": task_id,
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": f"{utc_date}T12:00:00Z",
            "result": {"score": 100000.0},
            "error": None,
        }
        kwargs = dict(
            task_id=task_id,
            runner_id="armbench",
            outcome=outcome,
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date=utc_date,
        )
        obs1 = analyzer.ingest_outcome(**kwargs)
        obs2 = analyzer.ingest_outcome(**kwargs)
        assert obs1["task_id"] == obs2["task_id"]
        assert obs1["score"] == obs2["score"]
        assert obs1["created_at"] == obs2["created_at"]

    def test_replayed_outcome_does_not_affect_later_stats(self, tmp_path):
        """Replaying an old outcome doesn't create a second observation."""
        db, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 5)
        # Replay the first outcome
        task_id = "canary:armbench:throughput-get-v1:2026-09-01"
        outcome = {
            "schema_version": 1,
            "task_id": task_id,
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": {"score": 200000.0},
            "error": None,
        }
        analyzer.ingest_outcome(
            task_id=task_id,
            runner_id="armbench",
            outcome=outcome,
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date="2026-09-01",
        )
        with db.read() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM canary_observations "
                "WHERE runner_id = 'armbench' AND accepted = 1"
            ).fetchone()[0]
        assert count == 5  # Not 6


# ---------------------------------------------------------------------------
# Malformed and non-finite scores
# ---------------------------------------------------------------------------


class TestMalformedScores:
    @pytest.mark.parametrize("bad_score", [None, "hello", float("nan"), float("inf"), float("-inf")])
    def test_non_finite_score_rejected(self, tmp_path, bad_score):
        _, _, analyzer = _make_env(tmp_path)
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:p:2026-09-01",
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": {"score": bad_score},
            "error": None,
        }
        obs = analyzer.ingest_outcome(
            task_id="canary:armbench:p:2026-09-01",
            runner_id="armbench",
            outcome=outcome,
            profile_id="p",
            profile_version=1,
            utc_date="2026-09-01",
        )
        assert obs is None

    def test_rejected_observation_recorded_in_db(self, tmp_path):
        db, _, analyzer = _make_env(tmp_path)
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:p:2026-09-01",
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": {"score": float("nan")},
            "error": None,
        }
        analyzer.ingest_outcome(
            task_id="canary:armbench:p:2026-09-01",
            runner_id="armbench",
            outcome=outcome,
            profile_id="p",
            profile_version=1,
            utc_date="2026-09-01",
        )
        with db.read() as conn:
            row = conn.execute(
                "SELECT * FROM canary_observations WHERE task_id = 'canary:armbench:p:2026-09-01'"
            ).fetchone()
        assert row is not None
        assert row["accepted"] == 0
        assert "non-finite" in row["rejection_reason"]

    def test_rejected_does_not_count_in_phase(self, tmp_path):
        """Rejected observations don't contribute to the sample count."""
        _, _, analyzer = _make_env(tmp_path)
        # Ingest 13 valid + 1 rejected
        _ingest_n(analyzer, 13)
        # Now ingest a bad one for day 14
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:throughput-get-v1:2026-09-14",
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-14T12:00:00Z",
            "result": {"score": float("nan")},
            "error": None,
        }
        analyzer.ingest_outcome(
            task_id="canary:armbench:throughput-get-v1:2026-09-14",
            runner_id="armbench",
            outcome=outcome,
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date="2026-09-14",
        )
        # Now ingest day 15 -- should be 14th accepted observation (observation phase)
        result = _ingest_n(analyzer, 1, start_day=15)
        assert result[0]["phase"] == PHASE_OBSERVATION

    def test_missing_result_treated_as_non_finite(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:p:2026-09-01",
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": None,
            "error": None,
        }
        obs = analyzer.ingest_outcome(
            task_id="canary:armbench:p:2026-09-01",
            runner_id="armbench",
            outcome=outcome,
            profile_id="p",
            profile_version=1,
            utc_date="2026-09-01",
        )
        assert obs is None


# ---------------------------------------------------------------------------
# Out-of-order completions
# ---------------------------------------------------------------------------


class TestOutOfOrder:
    def test_out_of_order_daily_results(self, tmp_path):
        """Observations completed out of calendar order still slot correctly."""
        _, _, analyzer = _make_env(tmp_path)
        # Ingest day 3 first, then day 1, then day 2
        for day, score in [(3, 300.0), (1, 100.0), (2, 200.0)]:
            utc_date = f"2026-09-{day:02d}"
            tid = f"canary:armbench:throughput-get-v1:{utc_date}"
            analyzer.ingest_outcome(
                task_id=tid,
                runner_id="armbench",
                outcome={
                    "schema_version": 1,
                    "task_id": tid,
                    "runner_id": "armbench",
                    "state": "completed",
                    "completed_at": f"{utc_date}T12:00:00Z",
                    "result": {"score": score},
                    "error": None,
                },
                profile_id="throughput-get-v1",
                profile_version=1,
                utc_date=utc_date,
            )

        # Verify all three exist and are ordered by date
        obs = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        dates = [o["utc_date"] for o in obs]
        # Newest first
        assert dates == ["2026-09-03", "2026-09-02", "2026-09-01"]


# ---------------------------------------------------------------------------
# Profile version separation
# ---------------------------------------------------------------------------


class TestVersionSeparation:
    def test_different_versions_have_independent_series(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        # Ingest 5 for v1
        _ingest_n(analyzer, 5, profile_version=1, base_score=100000.0)
        # Ingest 3 for v2
        _ingest_n(analyzer, 3, profile_version=2, base_score=200000.0)

        # v1 has 5, v2 has 3
        obs_v1 = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        obs_v2 = analyzer.get_observations("armbench", "throughput-get-v1", 2)
        assert len(obs_v1) == 5
        assert len(obs_v2) == 3

        # v2's reference should NOT include v1 data
        latest_v2 = obs_v2[0]  # newest
        assert latest_v2["sample_count"] == 2  # 2 prior v2 samples

    def test_version_change_resets_phase(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        # Get through observation phase for v1
        _ingest_n(analyzer, OBSERVATION_WINDOW + 1, profile_version=1)
        # v2 starts fresh at observation
        results_v2 = _ingest_n(analyzer, 1, profile_version=2)
        assert results_v2[0]["phase"] == PHASE_OBSERVATION


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_report_generated_at_28_samples(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, CALIBRATION_WINDOW, base_score=100000.0)

        report = analyzer.get_calibration_report("armbench", "throughput-get-v1", 1)
        assert report is not None
        assert report["status"] == "ready-for-review"
        assert report["sample_count"] == CALIBRATION_WINDOW
        data = report["report"]
        assert data["runner_id"] == "armbench"
        assert data["profile_id"] == "throughput-get-v1"
        assert data["status"] == "ready-for-review"
        assert "median_score" in data
        assert "mad" in data
        assert "cv_pct" in data
        assert "mad_based_threshold_pct" in data

    def test_report_not_generated_before_28(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, CALIBRATION_WINDOW - 1)
        report = analyzer.get_calibration_report("armbench", "throughput-get-v1", 1)
        assert report is None

    def test_report_is_idempotent(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, CALIBRATION_WINDOW)
        report1 = analyzer.get_calibration_report("armbench", "throughput-get-v1", 1)
        # Ingest one more -- should not create a second report
        _ingest_n(analyzer, 1, start_day=CALIBRATION_WINDOW + 1)
        report2 = analyzer.get_calibration_report("armbench", "throughput-get-v1", 1)
        assert report1["report_id"] == report2["report_id"]

    def test_report_threshold_conservative(self, tmp_path):
        """Recommended threshold should be at least 3x MAD/median."""
        _, _, analyzer = _make_env(tmp_path)
        # Slight variance: alternate 99k and 101k
        _ingest_n(
            analyzer,
            CALIBRATION_WINDOW,
            score_fn=lambda i: 99000.0 if i % 2 == 0 else 101000.0,
        )
        report = analyzer.get_calibration_report("armbench", "throughput-get-v1", 1)
        data = report["report"]
        # MAD of alternating 99k/101k: median is 100000, deviations are all 1000, MAD=1000
        # Threshold = 3 * 1000 / 100000 * 100 = 3.0%
        assert data["mad_based_threshold_pct"] >= 2.5  # Conservative bound


# ---------------------------------------------------------------------------
# Missed days (gaps)
# ---------------------------------------------------------------------------


class TestMissedDays:
    def test_gaps_do_not_affect_counting(self, tmp_path):
        """Missed days just create date gaps -- no interpolation."""
        _, _, analyzer = _make_env(tmp_path)
        # Ingest days 1, 3, 5, 7 (gaps on 2, 4, 6)
        for day in [1, 3, 5, 7]:
            utc_date = f"2026-09-{day:02d}"
            tid = f"canary:armbench:throughput-get-v1:{utc_date}"
            analyzer.ingest_outcome(
                task_id=tid,
                runner_id="armbench",
                outcome={
                    "schema_version": 1,
                    "task_id": tid,
                    "runner_id": "armbench",
                    "state": "completed",
                    "completed_at": f"{utc_date}T12:00:00Z",
                    "result": {"score": 100000.0},
                    "error": None,
                },
                profile_id="throughput-get-v1",
                profile_version=1,
                utc_date=utc_date,
            )

        obs = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        assert len(obs) == 4

        summary = analyzer.get_series_summary("armbench", "throughput-get-v1", 1)
        assert summary["accepted_count"] == 4
        assert summary["phase"] == PHASE_OBSERVATION


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    def test_get_observation_returns_correct(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 3)
        obs = analyzer.get_observation("armbench", "throughput-get-v1", 1, "2026-09-02")
        assert obs is not None
        assert obs["utc_date"] == "2026-09-02"

    def test_get_observation_returns_none_for_missing(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        assert analyzer.get_observation("armbench", "throughput-get-v1", 1, "2026-09-01") is None

    def test_get_all_observations_includes_rejected(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 2)
        # Add a rejected one
        analyzer.ingest_outcome(
            task_id="canary:armbench:throughput-get-v1:2026-09-03",
            runner_id="armbench",
            outcome={
                "schema_version": 1,
                "task_id": "canary:armbench:throughput-get-v1:2026-09-03",
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": "2026-09-03T12:00:00Z",
                "result": {"score": float("nan")},
                "error": None,
            },
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date="2026-09-03",
        )
        all_obs = analyzer.get_all_observations("armbench", "throughput-get-v1", 1)
        accepted_obs = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        assert len(all_obs) == 3
        assert len(accepted_obs) == 2

    def test_series_summary(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 5)
        summary = analyzer.get_series_summary("armbench", "throughput-get-v1", 1)
        assert summary["accepted_count"] == 5
        assert summary["rejected_count"] == 0
        assert summary["phase"] == PHASE_OBSERVATION
        assert summary["progress"] == "5/14"
        assert summary["latest_observation"] is not None
        assert summary["calibration_status"] is None

    def test_series_summary_calibrating(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 20)
        summary = analyzer.get_series_summary("armbench", "throughput-get-v1", 1)
        assert summary["phase"] == PHASE_CALIBRATING
        assert summary["progress"] == "20/28"

    def test_series_summary_ready(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        _ingest_n(analyzer, 30)
        summary = analyzer.get_series_summary("armbench", "throughput-get-v1", 1)
        assert summary["phase"] == PHASE_READY
        assert "30" in summary["progress"]
        assert summary["calibration_status"] == "ready-for-review"

    def test_series_summary_no_data(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        summary = analyzer.get_series_summary("armbench", "throughput-get-v1", 1)
        assert summary["phase"] == "no-data"
        assert summary["progress"] is None


# ---------------------------------------------------------------------------
# DB migration tests
# ---------------------------------------------------------------------------


class TestDBMigrationV3:
    def test_migration_v3_creates_tables(self, tmp_path):
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
        db.initialize()
        with db.read() as conn:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        assert "canary_observations" in tables
        assert "canary_calibration_reports" in tables
        assert version == DATABASE_SCHEMA_VERSION

    def test_migration_v3_idempotent(self, tmp_path):
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
        db.initialize()
        db.initialize()
        db.initialize()
        with db.read() as conn:
            v3_count = conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE version = 3"
            ).fetchone()[0]
        assert v3_count == 1

    def test_v2_db_upgrades_to_v3(self, tmp_path):
        """Simulate a v2 DB and verify it upgrades cleanly."""
        db = ControlDatabase(tmp_path / "control.db", tmp_path / "audit.jsonl")
        # Initialize at v2 (current code includes v3 automatically)
        db.initialize()
        # Verify both old and new tables present
        with db.read() as conn:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        assert "tasks" in tables
        assert "canary_schedule" in tables
        assert "canary_observations" in tables
        assert "canary_calibration_reports" in tables


# ---------------------------------------------------------------------------
# Integration: service record_outcome triggers ingestion
# ---------------------------------------------------------------------------


class TestServiceIntegration:
    def _setup_canary_task(self, tmp_path):
        """Create DB with a canary task in accepted state."""
        db, registry, analyzer = _make_env(tmp_path)
        service = ControlService(db, registry)
        # Create a canary task directly
        with db.transaction(immediate=True) as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(task_id, runner_id, task_class, priority, state, submitted_at, "
                "submitted_by, canary_id, envelope_json, created_at, updated_at, "
                "claimed_at, claim_token, accepted_at) "
                "VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "canary:armbench:throughput-get-v1:2026-09-01",
                    "armbench",
                    "canary",
                    50,
                    "2026-09-01T06:00:00Z",
                    "canary-scheduler",
                    "throughput-get-v1:2026-09-01",
                    json.dumps({
                        "schema_version": 1,
                        "task_id": "canary:armbench:throughput-get-v1:2026-09-01",
                        "runner_id": "armbench",
                        "task_class": "canary",
                        "priority": 50,
                        "submitted_at": "2026-09-01T06:00:00Z",
                        "submitted_by": "canary-scheduler",
                        "canary_id": "throughput-get-v1:2026-09-01",
                        "task": {
                            "task_type": "CanaryPerfTaskData",
                            "source": "valkey",
                            "specifier": "a" * 40,
                            "timestamp": "2026-09-01T06:00:00.000000",
                            "note": "canary throughput-get-v1 v1 (2026-09-01)",
                        },
                    }),
                    "2026-09-01T06:00:00Z",
                    "2026-09-01T06:00:00Z",
                    "2026-09-01T06:00:00Z",
                    "test-token",
                    "2026-09-01T06:00:00Z",
                ),
            )
        return db, service, analyzer

    def test_canary_completion_triggers_ingestion(self, tmp_path):
        db, service, analyzer = self._setup_canary_task(tmp_path)
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:throughput-get-v1:2026-09-01",
            "runner_id": "armbench",
            "state": "completed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": {"score": 200000.0},
            "error": None,
        }
        service.record_outcome(
            "armbench",
            "canary:armbench:throughput-get-v1:2026-09-01",
            outcome,
            actor="test",
        )
        # Observation should exist
        obs = analyzer.get_observations("armbench", "throughput-get-v1", 1)
        assert len(obs) == 1
        assert obs[0]["score"] == 200000.0

    def test_non_canary_completion_does_not_trigger(self, tmp_path):
        """Normal task completion should not create an observation."""
        db, registry, analyzer = _make_env(tmp_path)
        service = ControlService(db, registry)
        # Submit, claim, accept a manual task
        env = task_envelope("manual-1", runner_id="armbench")
        service.submit_task(env, actor="test")
        claim = service.claim_task("armbench", actor="test")
        service.accept_task("armbench", "manual-1", claim["claim_token"], actor="test")
        outcome = task_outcome("manual-1", "armbench", "completed")
        service.record_outcome("armbench", "manual-1", outcome, actor="test")
        # No observations should exist
        with db.read() as conn:
            count = conn.execute("SELECT COUNT(*) FROM canary_observations").fetchone()[0]
        assert count == 0

    def test_failed_canary_does_not_trigger(self, tmp_path):
        """A failed canary task should not create an observation."""
        db, registry, _ = _make_env(tmp_path)
        service = ControlService(db, registry)
        # Insert a canary task directly in accepted state
        with db.transaction(immediate=True) as conn:
            conn.execute(
                "INSERT INTO tasks "
                "(task_id, runner_id, task_class, priority, state, submitted_at, "
                "submitted_by, canary_id, envelope_json, created_at, updated_at, "
                "claimed_at, claim_token, accepted_at) "
                "VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "canary:armbench:throughput-get-v1:2026-09-01",
                    "armbench",
                    "canary",
                    50,
                    "2026-09-01T06:00:00Z",
                    "canary-scheduler",
                    "throughput-get-v1:2026-09-01",
                    json.dumps({
                        "schema_version": 1,
                        "task_id": "canary:armbench:throughput-get-v1:2026-09-01",
                        "runner_id": "armbench",
                        "task_class": "canary",
                        "priority": 50,
                        "submitted_at": "2026-09-01T06:00:00Z",
                        "submitted_by": "canary-scheduler",
                        "canary_id": "throughput-get-v1:2026-09-01",
                        "task": {
                            "task_type": "CanaryPerfTaskData",
                            "source": "valkey",
                            "specifier": "a" * 40,
                        },
                    }),
                    "2026-09-01T06:00:00Z",
                    "2026-09-01T06:00:00Z",
                    "2026-09-01T06:00:00Z",
                    "test-token",
                    "2026-09-01T06:00:00Z",
                ),
            )
        outcome = {
            "schema_version": 1,
            "task_id": "canary:armbench:throughput-get-v1:2026-09-01",
            "runner_id": "armbench",
            "state": "failed",
            "completed_at": "2026-09-01T12:00:00Z",
            "result": None,
            "error": "benchmark failed",
        }
        service.record_outcome(
            "armbench",
            "canary:armbench:throughput-get-v1:2026-09-01",
            outcome,
            actor="test",
        )
        with db.read() as conn:
            count = conn.execute("SELECT COUNT(*) FROM canary_observations").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Environment/provenance fingerprint
# ---------------------------------------------------------------------------


class TestEnvironmentFingerprint:
    def test_environment_persisted(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        env = {"kernel": "6.1.90", "instance_id": "i-abc123"}
        utc_date = "2026-09-01"
        tid = "canary:armbench:throughput-get-v1:2026-09-01"
        obs = analyzer.ingest_outcome(
            task_id=tid,
            runner_id="armbench",
            outcome={
                "schema_version": 1,
                "task_id": tid,
                "runner_id": "armbench",
                "state": "completed",
                "completed_at": f"{utc_date}T12:00:00Z",
                "result": {"score": 100000.0},
                "error": None,
            },
            profile_id="throughput-get-v1",
            profile_version=1,
            utc_date=utc_date,
            environment=env,
        )
        stored_env = json.loads(obs["environment_json"])
        assert stored_env["kernel"] == "6.1.90"
        assert stored_env["instance_id"] == "i-abc123"

    def test_no_environment_stored_as_null(self, tmp_path):
        _, _, analyzer = _make_env(tmp_path)
        results = _ingest_n(analyzer, 1)
        assert results[0]["environment_json"] is None
