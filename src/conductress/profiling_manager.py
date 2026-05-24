"""Profiling and performance counter collection for Valkey servers.

Manages perf record (flamegraph) and perf stat (hardware counters)
lifecycle on remote hosts via SSH.
"""

import logging
import subprocess
from pathlib import Path
from threading import Thread
from typing import Optional

from . import config
from .ssh_host import SshHost

PERF_STATUS_FILE = "/tmp/profiling_running"
PERF_STAT_STATUS_FILE = "/tmp/perf_stat_running"

# Remote paths for profiling artifacts
_PATH_ROOT = Path("~")
PERF_DATA_PATH = _PATH_ROOT / "perf.data"
FLAMEGRAPH_PATH = _PATH_ROOT / "flamegraph.svg"
PERF_STATS_PATH = _PATH_ROOT / "perf_stat_output"
FLAMEGRAPH_DIR = _PATH_ROOT / "FlameGraph"

logger = logging.getLogger(__name__)


class ProfilingManager:
    """Manages perf record and perf stat on a remote host."""

    def __init__(self, host: SshHost) -> None:
        self._host = host
        self._profiling_thread: Optional[Thread] = None
        self._perf_stat_thread: Optional[Thread] = None
        self._target_pid: int = -1

    @property
    def target_pid(self) -> int:
        return self._target_pid

    @target_pid.setter
    def target_pid(self, pid: int) -> None:
        self._target_pid = pid

    # =========================================================================
    # PERF RECORD (flamegraph)
    # =========================================================================

    def profiling_start(self, sample_rate: int) -> None:
        """Start profiling the server using perf record."""
        if self.is_profiling():
            raise RuntimeError("Profiling already started")
        self._profiling_thread = Thread(target=self._profiling_run_sync, args=(sample_rate,))
        self._profiling_thread.start()

    def _profiling_run_sync(self, sample_rate: int) -> None:
        """Run perf record synchronously in a thread."""
        ip = self._host.ip
        command = f"touch {PERF_STATUS_FILE}"
        if ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), ip, command], check=True)

        perf_command = (
            f"sudo perf record -F {sample_rate} -a -g -o {PERF_DATA_PATH} "
            f"-- sh -c 'while [ -f {PERF_STATUS_FILE} ]; do sleep 1; done'"
        )
        if ip in ["127.0.0.1", "localhost"]:
            subprocess.run(perf_command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), ip, perf_command], check=True)

    def is_profiling(self) -> bool:
        """Check if profiling is currently running."""
        return self._profiling_thread is not None and self._profiling_thread.is_alive()

    async def profiling_stop(self) -> None:
        """Signal profiling to stop. Use profiling_wait() to block until done."""
        if self._profiling_thread is None:
            return
        await self._host.run_host_command(f"rm -f {PERF_STATUS_FILE}")

    def profiling_wait(self) -> None:
        """Block until profiling finishes."""
        if self._profiling_thread is None:
            return
        self._profiling_thread.join()
        self._profiling_thread = None

    async def profiling_report(self, result_dir: Path) -> None:
        """Generate flamegraph and copy results to local result_dir."""
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory {result_dir} must exist")

        out_perf_path = _PATH_ROOT / "out.perf"
        out_folded_path = _PATH_ROOT / "out.folded"

        if self.is_profiling():
            await self.profiling_stop()
        self.profiling_wait()

        await self._host.run_host_command(f"sudo chmod a+r {PERF_DATA_PATH}")
        await self._host.run_host_command(f"perf script -i {PERF_DATA_PATH} > {out_perf_path}")
        await self._host.run_host_command(
            f"{FLAMEGRAPH_DIR/'stackcollapse-perf.pl'} " f"{out_perf_path} > {out_folded_path}"
        )
        await self._host.run_host_command(f"{FLAMEGRAPH_DIR/'flamegraph.pl'} {out_folded_path} > {FLAMEGRAPH_PATH}")
        await self._host.run_host_command(f"rm -f {out_perf_path} {out_folded_path}")

        await self._host.get_remote_file(PERF_DATA_PATH, result_dir / "perf.data")
        await self._host.get_remote_file(FLAMEGRAPH_PATH, result_dir / "flamegraph.svg")

    # =========================================================================
    # PERF STAT (hardware counters)
    # =========================================================================

    async def perf_stat_start(self) -> None:
        """Start perf stat collection on the target process."""
        if self._perf_stat_thread and self._perf_stat_thread.is_alive():
            raise RuntimeError("Perf stat already running")
        await self._host.run_host_command(f"touch {PERF_STAT_STATUS_FILE}")
        self._perf_stat_thread = Thread(target=self._perf_stat_run_sync)
        self._perf_stat_thread.start()

    PERF_EVENTS_COMMON = [
        "instructions",
        "cycles",
        "L1-icache-load-misses",
        "L1-dcache-load-misses",
        "branch-misses",
        "branches",
        "stalled-cycles-frontend",
        "stalled-cycles-backend",
    ]

    PERF_EVENTS_X86 = PERF_EVENTS_COMMON + [
        "LLC-load-misses",
        "LLC-loads",
    ]

    def _get_perf_events(self) -> list[str]:
        """Return platform-appropriate perf events list."""
        import platform as _platform

        arch = _platform.machine()
        if arch in ("x86_64", "i686"):
            return self.PERF_EVENTS_X86
        return self.PERF_EVENTS_COMMON

    def _perf_stat_run_sync(self) -> None:
        """Run perf stat synchronously in a thread."""
        events = ",".join(self._get_perf_events())
        ip = self._host.ip
        command = (
            f"perf stat -e {events} -p {self._target_pid} -o {PERF_STATS_PATH} "
            f"-- sh -c 'while [ -f {PERF_STAT_STATUS_FILE} ]; do sleep 1; done'"
        )
        if ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), ip, command], check=True)

    async def perf_stat_stop(self) -> None:
        """Signal perf stat to stop."""
        if self._perf_stat_thread is None:
            return
        await self._host.run_host_command(f"rm -f {PERF_STAT_STATUS_FILE}")

    def perf_stat_wait(self) -> None:
        """Block until perf stat completes."""
        if self._perf_stat_thread:
            self._perf_stat_thread.join()
            self._perf_stat_thread = None

    async def perf_stat_report(self, result_dir: Path) -> dict:
        """Copy perf stat results and return parsed counters."""
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory {result_dir} must exist")
        local_path = result_dir / "perf_stat.txt"
        await self._host.get_remote_file(Path(PERF_STATS_PATH), local_path)
        return self.parse_perf_stat(local_path)

    @staticmethod
    def parse_perf_stat(path: Path) -> dict:
        """Parse perf stat output file into a dict of event_name -> count."""
        results: dict[str, int] = {}
        if not path.exists():
            return results
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "seconds time elapsed" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                count_str = parts[0].replace(",", "")
                try:
                    count = int(count_str)
                    event = parts[1].removesuffix(":u").removesuffix(":k").removesuffix(":")
                    results[event] = count
                except ValueError:
                    continue
        return results

    async def cleanup(self) -> None:
        """Remove profiling artifacts from the remote host."""
        await self._host.run_host_command(f"rm -f {PERF_DATA_PATH} {FLAMEGRAPH_PATH} {PERF_STATS_PATH}")
