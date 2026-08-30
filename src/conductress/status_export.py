"""Status JSON exporter for remote monitoring.

Writes a machine-readable status.json containing runner state, current task,
queue contents, and recent results. Designed to be pulled via SSH by an
aggregator on benchdev.

Performance: ``build_status`` reads output.jsonl at most **once** per call
via a bounded tail reader (last ~400 lines) and derives both duration
calibration and recent results from that shared snapshot.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import CONDUCTRESS_OUTPUT, CONDUCTRESS_TMP, PROJECT_ROOT
from .duration_estimator import estimate_task_duration_seconds, load_duration_calibration_from_lines
from .file_protocol import FileProtocol
from .runner_identity import get_runner_info
from .status import _find_runner_pid, _format_elapsed
from .tail_reader import tail_lines
from .task_queue import TaskQueue

logger = logging.getLogger(__name__)

STATUS_EXPORT_DIR = PROJECT_ROOT / "status"
STATUS_EXPORT_FILE = STATUS_EXPORT_DIR / "status.json"

# How many recent results to include
RECENT_RESULTS_COUNT = 5

# Tail read budget: enough for calibration (max_records*2 = 400) and recent
# results (RECENT_RESULTS_COUNT*2 = 10). A single tail_lines call satisfying
# both avoids any whole-file I/O.
_TAIL_BUDGET = 410


def _read_result_snapshot() -> list[str]:
    """One bounded tail read of output.jsonl shared by all build_status consumers."""
    return tail_lines(CONDUCTRESS_OUTPUT, _TAIL_BUDGET)


def build_status(
    *,
    fleet_control: Optional[dict[str, Any]] = None,
    boundary: Optional[dict[str, Any]] = None,
    _result_snapshot: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a local status snapshot without performing network activity.

    When *_result_snapshot* is provided, calibration and recent results are
    derived from those pre-read lines instead of touching the filesystem.
    This allows callers (``_publish_boundary``) to share one tail read
    across both build_status and export_status.
    """
    snapshot = _result_snapshot if _result_snapshot is not None else _read_result_snapshot()

    identity = get_runner_info()
    status: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "runner_id": identity["runner_id"],
        "platform": identity["platform"]["label"],
        "identity": identity,
        "host": _get_hostname(),
        "runner": _get_runner_info(),
        "current_task": _get_current_task(),
        "queue": _get_queue_info(snapshot),
        "recent_results": _get_recent_results(snapshot),
        "disk": _get_disk_info(),
    }

    status["eta_minutes"] = round(status["queue"]["expected_duration_sec"] / 60, 1)
    if fleet_control is not None:
        status["fleet_control"] = fleet_control
    if boundary is not None:
        status["boundary"] = boundary
        status["measurement_isolation"] = {
            "boundary_publisher_active": True,
            "status_timer_migration_required": os.environ.get("CONDUCTRESS_BOUNDARY_STATUS_ONLY") != "1",
        }
    return status


def export_status(
    publish_target: str = "",
    *,
    status: Optional[dict[str, Any]] = None,
    fleet_control: Optional[dict[str, Any]] = None,
    boundary: Optional[dict[str, Any]] = None,
    publish_attempts: int = 1,
) -> Path:
    """Write and optionally publish a status snapshot.

    The local write uses atomic rename to avoid serving a partial file to
    concurrent rsync pulls. ``publish_attempts`` permits a bounded checked
    retry while preserving the historical ``Path`` return contract.
    """
    if publish_attempts < 1:
        raise ValueError("publish_attempts must be at least 1")

    status = status or build_status(fleet_control=fleet_control, boundary=boundary)
    STATUS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to a temp file in the same directory, then rename.
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATUS_EXPORT_DIR, prefix=".status-", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, STATUS_EXPORT_FILE)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt).
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    if publish_target:
        for attempt in range(1, publish_attempts + 1):
            if _publish_status(publish_target):
                break
            if attempt < publish_attempts:
                logger.warning(
                    "Status publish to %s failed; retrying (%d/%d)",
                    publish_target,
                    attempt + 1,
                    publish_attempts,
                )

    return STATUS_EXPORT_FILE


def _publish_status(target: str) -> bool:
    """Publish canonical runner status and its legacy platform alias.

    Returns True if rsync succeeds, False on failure (logged at ERROR).
    """
    from conductress.publisher import detect_platform
    from conductress.utility import run_rsync

    platform_id, _ = detect_platform()
    identity = get_runner_info()
    # Preserve existing dashboard filenames while consumers migrate to the
    # canonical runner-specific path.
    name_map = {"arm64": "arm", "amd64": "x86", "intel": "intel"}
    legacy_filename = name_map.get(platform_id, platform_id)

    ssh_key = Path.home() / "conductress" / "server-keyfile.pem"
    if not ssh_key.exists():
        ssh_key = Path.home() / ".ssh" / "openssh-ec2-pair.pem"
    ssh_cmd = f"ssh -i {ssh_key} -F /dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=10"

    # Stage both destination paths and upload them in a single rsync process so
    # Phase 1 adds metadata without adding another network connection.
    with tempfile.TemporaryDirectory(prefix="conductress-status-") as tmp:
        root = Path(tmp)
        legacy = root / "status" / f"{legacy_filename}.json"
        canonical = root / "status" / "runners" / f"{identity['runner_id']}.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STATUS_EXPORT_FILE, legacy)
        shutil.copy2(STATUS_EXPORT_FILE, canonical)

        destination = target.rstrip("/") + "/"
        return run_rsync(
            ["rsync", "-az", "--chmod=D755,F644", "-e", ssh_cmd, f"{root}/", destination],
            destination,
            timeout=15,
        )


def _get_hostname() -> str:
    """Get a short hostname for identification."""
    import socket

    return socket.gethostname().split(".")[0]


def _get_disk_info() -> dict[str, Any]:
    """Report free space on the filesystem holding builds, benchmark output, and RDBs.

    A host's local disk can fill (perf.data captures, build trees, stray RDBs, logs)
    and stall the runner, so the dashboard surfaces a per-host disk alarm. Measured on
    PROJECT_ROOT, which shares a filesystem with the ~/valkey and ~/redis build trees.
    """
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
    except OSError:
        return {}
    free_pct = round(usage.free * 100 / usage.total) if usage.total else 0
    return {
        "path": str(PROJECT_ROOT),
        "size_bytes": usage.total,
        "used_bytes": usage.used,
        "avail_bytes": usage.free,
        "free_pct": free_pct,
    }


def _get_runner_info() -> dict[str, Any]:
    pid = _find_runner_pid()
    if pid:
        # Get uptime from /proc
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            # Approximate: use process start time
            start_time = float(stat[21]) / 100  # clock ticks to seconds
            system_uptime = float(Path("/proc/uptime").read_text().split()[0])
            uptime_sec = time.time() - (system_uptime - start_time / 100)
        except Exception:
            uptime_sec = None
        return {"pid": pid, "state": "running", "uptime_hours": round(uptime_sec / 3600, 1) if uptime_sec else None}
    else:
        return {"pid": None, "state": "stopped", "uptime_hours": None}


def _get_current_task() -> Optional[dict[str, Any]]:
    active = FileProtocol.get_active_task_ids(CONDUCTRESS_TMP)
    if not active:
        return None

    task_id, status = next(iter(active.items()))
    elapsed = time.time() - status.start_time if status.start_time else 0
    progress_pct = (status.steps_completed / status.steps_total * 100) if status.steps_total > 0 else 0

    return {
        "id": task_id,
        "type": status.task_type,
        "state": status.state,
        "progress_pct": round(progress_pct, 1),
        "elapsed_sec": round(elapsed),
        "steps": f"{status.steps_completed}/{status.steps_total}",
    }


def _get_queue_info(snapshot: Optional[list[str]] = None) -> dict[str, Any]:
    """Build queue info using shared snapshot lines for calibration."""
    queue = TaskQueue()
    tasks = queue.get_all_tasks()
    if snapshot is not None:
        calibration = load_duration_calibration_from_lines(snapshot)
    else:
        from .duration_estimator import load_duration_calibration

        calibration = load_duration_calibration(CONDUCTRESS_OUTPUT)
    expected = {task.task_id: estimate_task_duration_seconds(task, calibration) for task in tasks}
    return {
        "depth": len(tasks),
        "expected_duration_sec": sum(expected.values()),
        "tasks": [
            {
                "id": task.task_id,
                "type": task.task_type,
                "note": task.note or "",
                "source": task.source,
                "specifier": task.specifier,
                "expected_duration_sec": expected[task.task_id],
            }
            for task in tasks[:10]
        ],
    }


def _get_recent_results(snapshot: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Read the last N results, preferring the shared snapshot."""
    if snapshot is not None:
        lines = snapshot
    else:
        if not CONDUCTRESS_OUTPUT.exists():
            return []
        lines = tail_lines(CONDUCTRESS_OUTPUT, RECENT_RESULTS_COUNT * 2)

    results: list[dict[str, Any]] = []
    for line in reversed(lines[-RECENT_RESULTS_COUNT * 2 :]):
        if len(results) >= RECENT_RESULTS_COUNT:
            break
        try:
            entry = json.loads(line)
            results.append(
                {
                    "task_id": entry.get("task_id", ""),
                    "method": entry.get("method", ""),
                    "score": entry.get("score"),
                    "commit": entry.get("commit_hash", "")[:8],
                    "source": entry.get("source", ""),
                    "specifier": entry.get("specifier", ""),
                    "note": entry.get("note", ""),
                    "completed": entry.get("end_time", ""),
                    "expected_duration_sec": entry.get("expected_duration_sec"),
                    "observed_duration_sec": entry.get("observed_duration_sec"),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return results
