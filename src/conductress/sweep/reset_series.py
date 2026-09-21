"""Clear one v3 sweep series' history so it can restart cleanly at deploy.

A series' client budget (its connection count and client-thread count) is part
of its identity.  When the budget changes -- for example the P1 GET series
moving from 8 to 16 client threads -- the old points were measured against a
different client and must not share a chart line with the new ones.  This module
resets one series by:

1. backing up and deleting the coordinator's state file, so the next boundary
   publish writes a fresh, empty state file for the series; and
2. relocating (not deleting) any queued task files that belong to the series,
   so a cell queued at the old budget cannot complete after the deploy and land
   a stale point on the fresh line.

It refuses to run while the runner service is active -- a running runner may be
mid-task on this series -- unless the caller forces it.  Everything it does is
local to the runner's project directory; nothing is published or sent anywhere.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from conductress import config
from conductress.sweep.coordinator_v3 import V3_STATE_DIR


def _utc_stamp() -> str:
    """A filesystem-safe UTC timestamp for backup and relocation names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def series_label(workload: str, engine: Optional[str]) -> str:
    """The engine-prefixed series label, matching the coordinator's own rule.

    A comparison engine's series carry a ``<source>-`` prefix (e.g.
    ``redis-get-k16-v16-t7-p1``); the default Valkey series carry none.  This
    mirrors ``BaseCachecannonSweepCoordinatorV3.__init__``.
    """
    if engine and engine != "valkey":
        return f"{engine}-{workload}"
    return workload


def state_file_for(workload: str, engine: Optional[str], state_dir: Path = V3_STATE_DIR) -> Path:
    """Path to the v3 coordinator state file for one series."""
    return state_dir / f"state_cachecannon-v3_{series_label(workload, engine)}.json"


def _note_prefix(workload: str, engine: Optional[str]) -> str:
    """The queued-task note prefix a series' cells carry.

    ``BaseCachecannonSweepCoordinatorV3._create_task`` writes the note as
    ``[cachecannon-sweep-v3:<source>/<label>] <reason>`` where ``<label>`` is
    the engine-prefixed series label and ``<source>`` is the engine source
    ('valkey' by default).
    """
    source = engine if engine else "valkey"
    return f"[cachecannon-sweep-v3:{source}/{series_label(workload, engine)}]"


def _task_note(task_file: Path) -> Optional[str]:
    """Read the ``note`` field out of a queued task file, or None if unreadable."""
    try:
        doc = json.loads(task_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    note = doc.get("note")
    return note if isinstance(note, str) else None


def service_is_active(service: str = "conductress.service") -> bool:
    """True when systemd reports the runner service as active.

    A missing ``systemctl`` or an "unknown"/inactive/failed status is treated as
    NOT active (safe to reset): a host without systemd is a dev checkout, not a
    live runner.  Only a literal "active" blocks the reset.
    """
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.stdout.strip() == "active"


@dataclass
class ResetSeriesResult:
    """What a reset did (or would do, under ``--dry-run``)."""

    label: str
    state_file: Path
    dry_run: bool
    state_existed: bool = False
    state_backup: Optional[Path] = None
    relocated_dir: Optional[Path] = None
    relocated_tasks: list[Path] = field(default_factory=list)
    refused_reason: Optional[str] = None

    def summary_lines(self) -> list[str]:
        """Human-readable description of the outcome, for the CLI to print."""
        if self.refused_reason:
            return [f"Refused: {self.refused_reason}"]
        verb = "Would" if self.dry_run else "Did"
        lines = [f"Series: {self.label}"]
        if self.state_existed:
            backup = self.state_backup.name if self.state_backup else "<backup>"
            lines.append(f"  {verb} back up state file to {backup} and delete {self.state_file.name}")
        else:
            lines.append(f"  State file {self.state_file.name} does not exist (nothing to back up)")
        if self.relocated_tasks:
            dest = self.relocated_dir.name if self.relocated_dir else "<reset-dir>"
            lines.append(f"  {verb} relocate {len(self.relocated_tasks)} queued task file(s) to {dest}/:")
            for task_file in self.relocated_tasks:
                lines.append(f"    {task_file.name}")
        else:
            lines.append("  No queued task files match this series")
        return lines


def reset_series(
    workload: str,
    *,
    epoch: str = "v3",
    engine: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    state_dir: Path = V3_STATE_DIR,
    queue_dir: Optional[Path] = None,
    stamp: Optional[str] = None,
    service_active: Optional[bool] = None,
) -> ResetSeriesResult:
    """Reset one v3 series' history.  Pure of argparse, so it is unit-testable.

    Args:
        workload: Series workload label, unprefixed (e.g. ``get-k16-v16-t7-p1``).
        epoch: Sweep epoch; only ``v3`` is supported.
        engine: Comparison engine source (e.g. ``redis``); None for Valkey.
        dry_run: When True, touch nothing -- only report what would happen.
        force: Reset even if the runner service is active.
        state_dir: Where v3 state files live (overridable for tests).
        queue_dir: The local task queue directory (defaults to config).
        stamp: UTC timestamp string for backup/relocation names (defaults to now).
        service_active: Override the systemd probe (for tests); None probes.

    Returns:
        A :class:`ResetSeriesResult` describing the outcome.  ``refused_reason``
        is set (and nothing is touched) when the service is active without
        ``--force``.
    """
    if epoch != "v3":
        raise ValueError(f"only the v3 epoch is supported, got {epoch!r}")

    label = series_label(workload, engine)
    state_file = state_file_for(workload, engine, state_dir=state_dir)
    result = ResetSeriesResult(label=label, state_file=state_file, dry_run=dry_run)

    if not force:
        active = service_is_active() if service_active is None else service_active
        if active:
            result.refused_reason = "conductress.service is active; stop it before resetting a series, or pass --force"
            return result

    stamp = stamp or _utc_stamp()
    queue_dir = Path(queue_dir) if queue_dir is not None else config.CONDUCTRESS_QUEUE

    # 1. State file: back up then delete.
    if state_file.exists():
        result.state_existed = True
        backup = state_file.with_name(f"{state_file.name}.bak-{stamp}")
        result.state_backup = backup
        if not dry_run:
            shutil.copy2(state_file, backup)
            state_file.unlink()

    # 2. Queued task files: relocate any whose note names this series.
    prefix = _note_prefix(workload, engine)
    relocate_dir = queue_dir / f"reset-{stamp}"
    if queue_dir.exists():
        for task_file in sorted(queue_dir.glob("task_*.json")):
            note = _task_note(task_file)
            if note is not None and note.startswith(prefix):
                result.relocated_tasks.append(task_file)
        if result.relocated_tasks:
            result.relocated_dir = relocate_dir
            if not dry_run:
                relocate_dir.mkdir(parents=True, exist_ok=True)
                for task_file in result.relocated_tasks:
                    task_file.rename(relocate_dir / task_file.name)

    return result
