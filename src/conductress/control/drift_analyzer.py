"""Canary drift analysis: ingestion, rolling median/MAD, phase tracking.

Ingests completed canary outcomes idempotently, persists immutable daily
observations, computes deterministic rolling statistics from prior
accepted observations, and tracks observation/calibration/ready phases.

Design invariants:
- A new sample is NEVER included in its own reference window.
- MAD=0 is handled explicitly (all prior samples identical).
- Duplicate/replayed outcomes produce the same observation (idempotent).
- Profile version changes start a fresh series.
- Missed days create gaps -- no interpolation or backfill.
- Out-of-order completions are accepted and slotted by UTC date.
- No notifications, no runner traffic, no threshold mutation.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any, Dict, List, Optional

from .db import ControlDatabase, utc_text

logger = logging.getLogger(__name__)

# Phase constants
PHASE_OBSERVATION = "observation"
PHASE_CALIBRATING = "calibrating"
PHASE_READY = "ready"

# Window sizes
OBSERVATION_WINDOW = 14
CALIBRATION_WINDOW = 28
ROLLING_WINDOW = 28

# Minimum samples for any statistical computation
MIN_SAMPLES_FOR_STATS = 1


def _median(values: List[float]) -> float:
    """Deterministic median (lower-middle for even counts)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty list")
    mid = (n - 1) // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid] + s[mid + 1]) / 2.0


def _mad(values: List[float]) -> float:
    """Median absolute deviation from the median."""
    if len(values) < 1:
        raise ValueError("MAD of empty list")
    med = _median(values)
    deviations = sorted(abs(v - med) for v in values)
    return _median(deviations) if deviations else 0.0


def _is_finite(value: Any) -> bool:
    """Check if a value is a finite number."""
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


class DriftAnalyzer:
    """Stateless analyzer that operates on the control DB.

    Call :meth:`ingest_outcome` from the task-completion path.
    Call :meth:`get_observation` / :meth:`get_observations` /
    :meth:`get_calibration_report` from query/CLI paths.
    """

    def __init__(self, database: ControlDatabase) -> None:
        self.database = database

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_outcome(
        self,
        task_id: str,
        runner_id: str,
        outcome: Dict[str, Any],
        *,
        profile_id: str,
        profile_version: int,
        utc_date: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ingest a completed canary outcome into the observations table.

        Returns the observation dict if accepted, None if rejected.
        Idempotent: replaying the same (runner, profile, version, date, task_id)
        returns the existing observation unchanged.
        """
        # Extract and validate score
        result = outcome.get("result") or {}
        score = result.get("score")

        if not _is_finite(score):
            logger.warning(
                "canary outcome %s has non-finite score %r; recording rejection",
                task_id,
                score,
            )
            self._record_rejected_observation(
                task_id=task_id,
                runner_id=runner_id,
                profile_id=profile_id,
                profile_version=profile_version,
                utc_date=utc_date,
                reason=f"non-finite score: {score!r}",
                environment=environment,
            )
            return None

        score_val = float(score)  # type: ignore[arg-type]  # guarded by _is_finite above
        completed_at = outcome.get("completed_at", utc_text())

        with self.database.transaction(immediate=True) as conn:
            # Idempotency: check for existing observation with same key
            existing = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND utc_date = ? AND task_id = ?",
                (runner_id, profile_id, profile_version, utc_date, task_id),
            ).fetchone()
            if existing is not None:
                return dict(existing)

            # Compute rolling statistics from PRIOR accepted observations only
            prior_scores = self._prior_accepted_scores(
                conn, runner_id, profile_id, profile_version, utc_date
            )

            # Determine phase
            sample_count = len(prior_scores)
            # This will be the (sample_count + 1)-th observation
            ordinal = sample_count + 1

            if ordinal <= OBSERVATION_WINDOW:
                phase = PHASE_OBSERVATION
            elif ordinal < CALIBRATION_WINDOW:
                phase = PHASE_CALIBRATING
            else:
                phase = PHASE_READY

            # Compute reference statistics from prior window
            ref_median = None
            ref_mad = None
            delta_pct = None
            window_start = None
            window_end = None

            if len(prior_scores) >= MIN_SAMPLES_FOR_STATS:
                # Use at most ROLLING_WINDOW most recent prior observations
                window_scores = prior_scores[-ROLLING_WINDOW:]
                ref_median = _median(window_scores)
                ref_mad = _mad(window_scores)

                if ref_median != 0.0:
                    delta_pct = ((score_val - ref_median) / abs(ref_median)) * 100.0
                elif score_val != 0.0:
                    # Median is zero but score is not
                    delta_pct = float("inf") if score_val > 0 else float("-inf")
                else:
                    delta_pct = 0.0

                # Clamp infinite delta_pct for storage
                if delta_pct is not None and not math.isfinite(delta_pct):
                    delta_pct = 999999.0 if delta_pct > 0 else -999999.0

                # Window date range from prior observations
                prior_dates = self._prior_accepted_dates(
                    conn, runner_id, profile_id, profile_version, utc_date
                )
                if prior_dates:
                    recent_dates = prior_dates[-ROLLING_WINDOW:]
                    window_start = recent_dates[0]
                    window_end = recent_dates[-1]

            now = utc_text()
            env_json = json.dumps(environment, sort_keys=True) if environment else None

            conn.execute(
                "INSERT INTO canary_observations "
                "(runner_id, profile_id, profile_version, utc_date, task_id, "
                "score, completed_at, phase, accepted, rejection_reason, "
                "ref_median, ref_mad, delta_pct, sample_count, "
                "window_start, window_end, environment_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    runner_id,
                    profile_id,
                    profile_version,
                    utc_date,
                    task_id,
                    score_val,
                    completed_at,
                    phase,
                    ref_median,
                    ref_mad,
                    delta_pct,
                    sample_count,  # count of PRIOR observations
                    window_start,
                    window_end,
                    env_json,
                    now,
                ),
            )

            self.database.insert_audit(
                conn,
                actor="drift-analyzer",
                action="canary.observation",
                task_id=task_id,
                runner_id=runner_id,
                detail={
                    "profile_id": profile_id,
                    "profile_version": profile_version,
                    "utc_date": utc_date,
                    "score": score_val,
                    "phase": phase,
                    "ordinal": ordinal,
                    "ref_median": ref_median,
                    "ref_mad": ref_mad,
                    "delta_pct": delta_pct,
                },
            )

            obs = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND utc_date = ? AND task_id = ?",
                (runner_id, profile_id, profile_version, utc_date, task_id),
            ).fetchone()

            # Check if we should generate a calibration report
            if ordinal == CALIBRATION_WINDOW and phase == PHASE_READY:
                self._generate_calibration_report(
                    conn, runner_id, profile_id, profile_version
                )

        return dict(obs) if obs else None

    def _record_rejected_observation(
        self,
        *,
        task_id: str,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        utc_date: str,
        reason: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a rejected (malformed/non-finite) observation."""
        with self.database.transaction(immediate=True) as conn:
            existing = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND utc_date = ? AND task_id = ?",
                (runner_id, profile_id, profile_version, utc_date, task_id),
            ).fetchone()
            if existing is not None:
                return None

            now = utc_text()
            env_json = json.dumps(environment, sort_keys=True) if environment else None
            conn.execute(
                "INSERT INTO canary_observations "
                "(runner_id, profile_id, profile_version, utc_date, task_id, "
                "score, completed_at, phase, accepted, rejection_reason, "
                "ref_median, ref_mad, delta_pct, sample_count, "
                "window_start, window_end, environment_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, 'observation', 0, ?, "
                "NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)",
                (
                    runner_id,
                    profile_id,
                    profile_version,
                    utc_date,
                    task_id,
                    now,
                    reason,
                    env_json,
                    now,
                ),
            )
            self.database.insert_audit(
                conn,
                actor="drift-analyzer",
                action="canary.rejected",
                task_id=task_id,
                runner_id=runner_id,
                detail={
                    "profile_id": profile_id,
                    "profile_version": profile_version,
                    "utc_date": utc_date,
                    "reason": reason,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Rolling window helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prior_accepted_scores(
        conn: sqlite3.Connection,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        before_date: str,
    ) -> List[float]:
        """Return accepted scores BEFORE the given date, ordered by date ASC."""
        rows = conn.execute(
            "SELECT score FROM canary_observations "
            "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
            "AND utc_date < ? AND accepted = 1 "
            "ORDER BY utc_date ASC",
            (runner_id, profile_id, profile_version, before_date),
        ).fetchall()
        return [row["score"] for row in rows]

    @staticmethod
    def _prior_accepted_dates(
        conn: sqlite3.Connection,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        before_date: str,
    ) -> List[str]:
        """Return accepted observation dates BEFORE the given date, ordered ASC."""
        rows = conn.execute(
            "SELECT utc_date FROM canary_observations "
            "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
            "AND utc_date < ? AND accepted = 1 "
            "ORDER BY utc_date ASC",
            (runner_id, profile_id, profile_version, before_date),
        ).fetchall()
        return [row["utc_date"] for row in rows]

    # ------------------------------------------------------------------
    # Calibration report
    # ------------------------------------------------------------------

    def _generate_calibration_report(
        self,
        conn: sqlite3.Connection,
        runner_id: str,
        profile_id: str,
        profile_version: int,
    ) -> None:
        """Generate and store a calibration report at the 28-sample mark."""
        # Check if report already exists
        existing = conn.execute(
            "SELECT report_id FROM canary_calibration_reports "
            "WHERE runner_id = ? AND profile_id = ? AND profile_version = ?",
            (runner_id, profile_id, profile_version),
        ).fetchone()
        if existing:
            return  # Already generated

        # Collect all accepted observations for this series
        rows = conn.execute(
            "SELECT score, utc_date FROM canary_observations "
            "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
            "AND accepted = 1 ORDER BY utc_date ASC",
            (runner_id, profile_id, profile_version),
        ).fetchall()

        scores = [row["score"] for row in rows]
        dates = [row["utc_date"] for row in rows]

        if len(scores) < CALIBRATION_WINDOW:
            return  # Not enough data

        med = _median(scores)
        mad = _mad(scores)

        # Coefficient of variation proxy: MAD / median * 100
        cv_pct = (mad / abs(med) * 100.0) if med != 0.0 else 0.0

        # Conservative recommended thresholds: max(candidate, 3*MAD/median*100)
        # The 3x MAD factor provides ~99% coverage for normal-like distributions
        mad_threshold = (3.0 * mad / abs(med) * 100.0) if med != 0.0 else 0.0

        report = {
            "runner_id": runner_id,
            "profile_id": profile_id,
            "profile_version": profile_version,
            "sample_count": len(scores),
            "date_range": {"start": dates[0], "end": dates[-1]},
            "median_score": med,
            "mad": mad,
            "cv_pct": round(cv_pct, 4),
            "min_score": min(scores),
            "max_score": max(scores),
            "spread_pct": round(((max(scores) - min(scores)) / abs(med) * 100.0) if med != 0 else 0.0, 4),
            "mad_based_threshold_pct": round(mad_threshold, 4),
            "status": "ready-for-review",
            "note": (
                "Calibration report generated from first 28 observations. "
                "Recommended thresholds are derived from observed variability (3x MAD). "
                "Review and accept before enabling drift alerts."
            ),
        }

        now = utc_text()
        conn.execute(
            "INSERT INTO canary_calibration_reports "
            "(runner_id, profile_id, profile_version, sample_count, "
            "date_range_start, date_range_end, median_score, mad, cv_pct, "
            "mad_based_threshold_pct, status, report_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runner_id,
                profile_id,
                profile_version,
                len(scores),
                dates[0],
                dates[-1],
                med,
                mad,
                round(cv_pct, 4),
                round(mad_threshold, 4),
                "ready-for-review",
                json.dumps(report, sort_keys=True),
                now,
            ),
        )

        self.database.insert_audit(
            conn,
            actor="drift-analyzer",
            action="canary.calibration_report",
            runner_id=runner_id,
            detail={
                "profile_id": profile_id,
                "profile_version": profile_version,
                "sample_count": len(scores),
                "cv_pct": round(cv_pct, 4),
            },
        )

        logger.info(
            "calibration report generated for %s/%s v%d: cv=%.4f%%, "
            "mad_threshold=%.4f%%",
            runner_id,
            profile_id,
            profile_version,
            cv_pct,
            mad_threshold,
        )

    # ------------------------------------------------------------------
    # Query helpers (for PR3 CLI consumption)
    # ------------------------------------------------------------------

    def get_observation(
        self,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        utc_date: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a single observation, or None."""
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND utc_date = ? AND accepted = 1",
                (runner_id, profile_id, profile_version, utc_date),
            ).fetchone()
        return dict(row) if row else None

    def get_observations(
        self,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        *,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get accepted observations for a series, newest first."""
        with self.database.read() as conn:
            rows = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND accepted = 1 ORDER BY utc_date DESC LIMIT ?",
                (runner_id, profile_id, profile_version, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_observations(
        self,
        runner_id: str,
        profile_id: str,
        profile_version: int,
        *,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get all observations (including rejected), newest first."""
        with self.database.read() as conn:
            rows = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "ORDER BY utc_date DESC LIMIT ?",
                (runner_id, profile_id, profile_version, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_calibration_report(
        self,
        runner_id: str,
        profile_id: str,
        profile_version: int,
    ) -> Optional[Dict[str, Any]]:
        """Get the calibration report, or None if not yet generated."""
        with self.database.read() as conn:
            row = conn.execute(
                "SELECT * FROM canary_calibration_reports "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ?",
                (runner_id, profile_id, profile_version),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["report"] = json.loads(result["report_json"])
        return result

    def get_series_summary(
        self,
        runner_id: str,
        profile_id: str,
        profile_version: int,
    ) -> Dict[str, Any]:
        """Summary for CLI status display."""
        with self.database.read() as conn:
            accepted_count = conn.execute(
                "SELECT COUNT(*) FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND accepted = 1",
                (runner_id, profile_id, profile_version),
            ).fetchone()[0]

            rejected_count = conn.execute(
                "SELECT COUNT(*) FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND accepted = 0",
                (runner_id, profile_id, profile_version),
            ).fetchone()[0]

            latest = conn.execute(
                "SELECT * FROM canary_observations "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ? "
                "AND accepted = 1 ORDER BY utc_date DESC LIMIT 1",
                (runner_id, profile_id, profile_version),
            ).fetchone()

            cal = conn.execute(
                "SELECT status FROM canary_calibration_reports "
                "WHERE runner_id = ? AND profile_id = ? AND profile_version = ?",
                (runner_id, profile_id, profile_version),
            ).fetchone()

        # Determine overall phase
        if accepted_count == 0:
            phase = "no-data"
        elif accepted_count < OBSERVATION_WINDOW:
            phase = PHASE_OBSERVATION
            progress = f"{accepted_count}/{OBSERVATION_WINDOW}"
        elif accepted_count < CALIBRATION_WINDOW:
            phase = PHASE_CALIBRATING
            progress = f"{accepted_count}/{CALIBRATION_WINDOW}"
        else:
            phase = PHASE_READY
            progress = f"{accepted_count} samples"

        return {
            "runner_id": runner_id,
            "profile_id": profile_id,
            "profile_version": profile_version,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "phase": phase,
            "progress": progress if accepted_count > 0 else None,
            "latest_observation": dict(latest) if latest else None,
            "calibration_status": cal["status"] if cal else None,
        }
