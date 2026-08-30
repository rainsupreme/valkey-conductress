"""Daily drift canary scheduler.

Creates at most one canary task per (runner_id, profile_id, UTC date).
Idempotent across restarts and concurrent ticks.  Records ``missed``
status when the freshness deadline passes without a successful creation.
Never backfills stale dates.

Canary tasks flow through the existing remote mailbox -- no new runner
management calls or in-measurement traffic.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .canary_profiles import CanaryProfile, CanaryProfileRegistry
from .db import ControlDatabase, utc_now, utc_text
from .fleet_registry import FleetRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority constants  (higher number = higher priority)
# ---------------------------------------------------------------------------
# Explicit manual priority is preserved as-is; only the *default* manual
# priority sits here.  The scheduler always uses PRIORITY_CANARY.
PRIORITY_MANUAL_DEFAULT = 100
PRIORITY_CANARY = 50
PRIORITY_SWEEP = 10

# Canary task_class and submitter identity
CANARY_TASK_CLASS = "canary"
CANARY_SUBMITTER = "canary-scheduler"


def _utc_date_str(dt: datetime) -> str:
    """YYYY-MM-DD in UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _canary_task_id(runner_id: str, profile_id: str, utc_date: str) -> str:
    """Deterministic task_id for dedup."""
    return f"canary:{runner_id}:{profile_id}:{utc_date}"


class CanaryScheduler:
    """Stateless scheduler that creates/expires canary tasks in the control DB.

    Call :meth:`tick` periodically (e.g. on service startup and at each
    boundary status push).  It is safe to call concurrently -- the DB
    uniqueness constraint prevents duplicates.
    """

    def __init__(
        self,
        database: ControlDatabase,
        registry: FleetRegistry,
        profiles: CanaryProfileRegistry,
    ) -> None:
        self.database = database
        self.registry = registry
        self.profiles = profiles

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def tick(self, *, now: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Evaluate all enabled runners and create/expire canaries for today.

        Also checks yesterday to record missed status if the freshness
        deadline passed without creating a canary.  Never looks further
        back -- no backfill.

        Returns a list of actions taken (for logging / testing).
        """
        now = now or utc_now()
        today = _utc_date_str(now)
        yesterday = _utc_date_str(now - timedelta(days=1))
        actions: list[dict[str, Any]] = []

        for runner in self.registry.list_runners(enabled_only=True):
            profile_id = runner.get("canary_profile")
            if not profile_id:
                continue
            profile = self.profiles.get(profile_id)
            if profile is None:
                logger.warning(
                    "runner %s references unknown canary profile %s",
                    runner["runner_id"],
                    profile_id,
                )
                continue

            # Check yesterday for missed/expired
            action = self._evaluate_runner(runner["runner_id"], profile, yesterday, now)
            if action:
                actions.append(action)

            # Check today
            action = self._evaluate_runner(runner["runner_id"], profile, today, now)
            if action:
                actions.append(action)

        return actions

    def _evaluate_runner(
        self,
        runner_id: str,
        profile: CanaryProfile,
        utc_date: str,
        now: datetime,
    ) -> Optional[dict[str, Any]]:
        """Create or expire a single canary for one runner/profile/day."""
        task_id = _canary_task_id(runner_id, profile.profile_id, utc_date)

        # Compute the freshness window
        due_start = datetime.strptime(utc_date, "%Y-%m-%d").replace(hour=profile.utc_hour, tzinfo=timezone.utc)
        deadline = due_start + timedelta(hours=profile.freshness_hours)

        with self.database.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT state, task_id FROM canary_schedule " "WHERE runner_id = ? AND profile_id = ? AND utc_date = ?",
                (runner_id, profile.profile_id, utc_date),
            ).fetchone()

            if row is not None:
                # Freshness applies to queued work. Once claimed, accepted, or
                # completed, the boundary decision has already been made.
                if row["state"] == "created" and now >= deadline and row["task_id"]:
                    task = conn.execute("SELECT state FROM tasks WHERE task_id = ?", (row["task_id"],)).fetchone()
                    if task is not None and task["state"] == "queued":
                        return self._mark_expired(conn, runner_id, profile.profile_id, utc_date, row["task_id"])
                return None

            # Not yet in schedule table
            if now < due_start:
                # Too early
                return None

            if now >= deadline:
                # Past freshness -- record as missed, never backfill
                return self._mark_missed(conn, runner_id, profile.profile_id, utc_date)

            # Within the freshness window -- create the canary task
            return self._create_canary(conn, runner_id, profile, utc_date, task_id, now)

    def _create_canary(
        self,
        conn: Any,
        runner_id: str,
        profile: CanaryProfile,
        utc_date: str,
        task_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Insert both a canary_schedule record and a tasks row atomically."""
        now_text = utc_text(now)
        envelope = self._build_envelope(runner_id, profile, utc_date, task_id, now_text)
        canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))

        # Idempotent: if the task already exists (from a concurrent tick), skip
        existing_task = conn.execute("SELECT task_id FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if existing_task is not None:
            # Task exists but schedule row doesn't -- fix up schedule
            conn.execute(
                "INSERT OR IGNORE INTO canary_schedule "
                "(runner_id, profile_id, utc_date, state, task_id, created_at, updated_at) "
                "VALUES (?, ?, ?, 'created', ?, ?, ?)",
                (runner_id, profile.profile_id, utc_date, task_id, now_text, now_text),
            )
            return {"action": "schedule_fixup", "runner_id": runner_id, "task_id": task_id}

        # Insert schedule row
        conn.execute(
            "INSERT INTO canary_schedule "
            "(runner_id, profile_id, utc_date, state, task_id, created_at, updated_at) "
            "VALUES (?, ?, ?, 'created', ?, ?, ?)",
            (runner_id, profile.profile_id, utc_date, task_id, now_text, now_text),
        )

        # Insert task into the main tasks table
        conn.execute(
            "INSERT INTO tasks "
            "(task_id, runner_id, task_class, priority, state, submitted_at, "
            "submitted_by, canary_id, envelope_json, idempotency_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                runner_id,
                CANARY_TASK_CLASS,
                PRIORITY_CANARY,
                now_text,
                CANARY_SUBMITTER,
                f"{profile.profile_id}:{utc_date}",
                canonical,
                task_id,  # idempotency_key = task_id for canaries
                now_text,
                now_text,
            ),
        )

        self.database.insert_audit(
            conn,
            actor=CANARY_SUBMITTER,
            action="canary.created",
            task_id=task_id,
            runner_id=runner_id,
            new_state="queued",
            detail={
                "profile_id": profile.profile_id,
                "profile_version": profile.profile_version,
                "utc_date": utc_date,
            },
        )

        logger.info(
            "created canary task %s for %s/%s on %s",
            task_id,
            runner_id,
            profile.profile_id,
            utc_date,
        )
        return {
            "action": "created",
            "runner_id": runner_id,
            "profile_id": profile.profile_id,
            "utc_date": utc_date,
            "task_id": task_id,
        }

    def _mark_missed(self, conn: Any, runner_id: str, profile_id: str, utc_date: str) -> dict[str, Any]:
        now_text = utc_text()
        conn.execute(
            "INSERT OR IGNORE INTO canary_schedule "
            "(runner_id, profile_id, utc_date, state, task_id, created_at, updated_at) "
            "VALUES (?, ?, ?, 'missed', NULL, ?, ?)",
            (runner_id, profile_id, utc_date, now_text, now_text),
        )
        self.database.insert_audit(
            conn,
            actor=CANARY_SUBMITTER,
            action="canary.missed",
            runner_id=runner_id,
            detail={"profile_id": profile_id, "utc_date": utc_date},
        )
        logger.info("canary missed for %s/%s on %s", runner_id, profile_id, utc_date)
        return {
            "action": "missed",
            "runner_id": runner_id,
            "profile_id": profile_id,
            "utc_date": utc_date,
        }

    def _mark_expired(
        self,
        conn: Any,
        runner_id: str,
        profile_id: str,
        utc_date: str,
        task_id: str,
    ) -> dict[str, Any]:
        now_text = utc_text()
        cancelled = conn.execute(
            "UPDATE tasks SET state = 'cancelled', updated_at = ? " "WHERE task_id = ? AND state = 'queued'",
            (now_text, task_id),
        )
        if cancelled.rowcount != 1:
            raise RuntimeError(f"canary task {task_id} was no longer queued during expiry")
        conn.execute(
            "UPDATE canary_schedule SET state = 'expired', updated_at = ? "
            "WHERE runner_id = ? AND profile_id = ? AND utc_date = ?",
            (now_text, runner_id, profile_id, utc_date),
        )
        self.database.insert_audit(
            conn,
            actor=CANARY_SUBMITTER,
            action="canary.expired",
            runner_id=runner_id,
            task_id=task_id,
            old_state="queued",
            new_state="cancelled",
            detail={"profile_id": profile_id, "utc_date": utc_date},
        )
        logger.info("canary expired for %s/%s on %s", runner_id, profile_id, utc_date)
        return {
            "action": "expired",
            "runner_id": runner_id,
            "profile_id": profile_id,
            "utc_date": utc_date,
        }

    # ------------------------------------------------------------------
    # Envelope builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_envelope(
        runner_id: str,
        profile: CanaryProfile,
        utc_date: str,
        task_id: str,
        submitted_at: str,
    ) -> dict[str, Any]:
        """Build a task envelope from a canary profile."""
        wl = profile.workload
        return {
            "schema_version": 1,
            "task_id": task_id,
            "runner_id": runner_id,
            "task_class": CANARY_TASK_CLASS,
            "priority": PRIORITY_CANARY,
            "submitted_at": submitted_at,
            "submitted_by": CANARY_SUBMITTER,
            "canary_id": f"{profile.profile_id}:{utc_date}",
            "task": {
                "task_type": "CanaryPerfTaskData",
                "source": profile.source,
                "specifier": profile.pinned_commit,
                "timestamp": submitted_at.removesuffix("Z").removesuffix("+00:00"),
                "replicas": 0,
                "requirements": {},
                "make_args": profile.build["make_args"],
                "note": f"canary {profile.profile_id} v{profile.profile_version} ({utc_date})",
                "test": wl["test"],
                "val_size": wl["val_size"],
                "key_size": wl["key_size"],
                "io_threads": wl["io_threads"],
                "pipelining": wl["pipelining"],
                "warmup": wl["warmup_seconds"],
                "duration": wl["duration_seconds"],
                "perf_stat_enabled": False,
                "has_expire": False,
                "preload_keys": True,
                "repetitions": wl["repetitions"],
                "bench_clients": wl["clients"],
                "bench_threads": wl["threads"],
                "keyspace": wl["keyspace"],
                "seed": wl["seed"],
            },
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def schedule_for_runner(self, runner_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
        """Return recent canary schedule entries for a runner."""
        with self.database.read() as conn:
            rows = conn.execute(
                "SELECT * FROM canary_schedule WHERE runner_id = ? " "ORDER BY utc_date DESC LIMIT ?",
                (runner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def schedule_for_date(self, utc_date: str) -> list[dict[str, Any]]:
        """Return all canary schedule entries for a given UTC date."""
        with self.database.read() as conn:
            rows = conn.execute(
                "SELECT * FROM canary_schedule WHERE utc_date = ? ORDER BY runner_id",
                (utc_date,),
            ).fetchall()
        return [dict(row) for row in rows]
