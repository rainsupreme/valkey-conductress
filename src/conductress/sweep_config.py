"""Runtime sweep configuration: focus, pause, and resume sweeps without restart.

Reads from sweep_config.json in PROJECT_ROOT. The file is checked on every
queue-empty cycle, so changes take effect within seconds.

Config format:
    {"mode": "normal"}                     -- all sweeps active (default)
    {"mode": "focus", "target": "memory-set-64b"}  -- only this workload runs
    {"mode": "paused", "paused": ["throughput"]}   -- these sweeps skip their turn
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from conductress.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SWEEP_CONFIG_FILE = PROJECT_ROOT / "sweep_config.json"


@dataclass
class SweepConfig:
    """Current sweep scheduling configuration."""

    mode: str = "normal"  # "normal", "focus", "paused"
    target: Optional[str] = None  # workload_id to focus on (mode=focus)
    paused: Optional[list[str]] = None  # list of workload_ids to skip (mode=paused)

    def __post_init__(self):
        if self.paused is None:
            self.paused = []

    def is_allowed(self, workload_id: str) -> bool:
        """Check if a sweep with this workload_id is allowed to queue."""
        if self.mode == "focus":
            return workload_id == self.target
        if self.mode == "paused":
            return workload_id not in (self.paused or [])
        return True  # normal mode


def load_sweep_config() -> SweepConfig:
    """Load sweep config from disk. Returns default (normal) if file missing or invalid."""
    if not SWEEP_CONFIG_FILE.exists():
        return SweepConfig()
    try:
        data = json.loads(SWEEP_CONFIG_FILE.read_text())
        return SweepConfig(
            mode=data.get("mode", "normal"),
            target=data.get("target"),
            paused=data.get("paused", []),
        )
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Invalid sweep_config.json: %s — using defaults", e)
        return SweepConfig()


def save_sweep_config(config: SweepConfig) -> None:
    """Write sweep config to disk."""
    data: dict[str, Any] = {"mode": config.mode}
    if config.target:
        data["target"] = config.target
    if config.paused:
        data["paused"] = config.paused
    SWEEP_CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")


def focus(workload_id: str) -> None:
    """Focus on a single workload — only it will run."""
    save_sweep_config(SweepConfig(mode="focus", target=workload_id))
    logger.info("Sweep focused on: %s", workload_id)


def pause(workload_ids: list[str]) -> None:
    """Pause specific workloads — they won't queue new tasks."""
    save_sweep_config(SweepConfig(mode="paused", paused=workload_ids))
    logger.info("Paused sweeps: %s", workload_ids)


def resume() -> None:
    """Resume normal operation — all sweeps active."""
    if SWEEP_CONFIG_FILE.exists():
        SWEEP_CONFIG_FILE.unlink()
    logger.info("Sweep config reset to normal")
