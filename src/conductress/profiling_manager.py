"""Profiling and performance counter collection for Valkey/Redis servers.

Manages perf stat (hardware counters) and CPU profiling (per-thread flamegraph stacks)
lifecycle on remote hosts via SSH.
"""

import logging
import subprocess
from pathlib import Path
from threading import Thread
from typing import Optional

from . import config
from .ssh_host import SshHost

PERF_STAT_STATUS_FILE = "/tmp/perf_stat_running"
PERF_STATS_PATH = Path("~") / "perf_stat_output"
CPU_PROFILE_DATA = "/tmp/perf-cpu-profile.data"
FLAMEGRAPH_DIR = Path("~") / "FlameGraph"

logger = logging.getLogger(__name__)


class ProfilingManager:
    """Manages perf stat and CPU profile collection on a remote host."""

    def __init__(self, host: SshHost) -> None:
        self._host = host
        self._perf_stat_thread: Optional[Thread] = None
        self._cpu_profile_thread: Optional[Thread] = None
        self._target_pid: int = -1
        # Thread-group TIDs discovered at server start (main thread + IO threads).
        # Shared by the CPU-profile (flamegraph) and per-thread perf-stat paths.
        self._main_tid: Optional[str] = None
        self._io_tids: list[str] = []

    @property
    def target_pid(self) -> int:
        return self._target_pid

    @target_pid.setter
    def target_pid(self, pid: int) -> None:
        self._target_pid = pid

    # =========================================================================
    # CPU PROFILE (per-thread flamegraph stacks)
    # =========================================================================

    def cpu_profile_start(self, duration: int) -> None:
        """Start perf record with frame-pointer call-graph on target PID.

        Records for the specified duration in seconds. Non-blocking — runs in a thread.
        Call cpu_profile_collect() after the duration to retrieve results.
        Also discovers thread TIDs now (while server is alive).
        """
        if self._cpu_profile_thread and self._cpu_profile_thread.is_alive():
            raise RuntimeError("CPU profile already running")
        # Discover TIDs now while server is running (won't exist after shutdown)
        self._discover_thread_tids()
        self._cpu_profile_thread = Thread(target=self._cpu_profile_run_sync, args=(duration,))
        self._cpu_profile_thread.start()

    def _discover_thread_tids(self) -> None:
        """Classify the target process's threads into main vs IO groups.

        Reads ``/proc/PID/task/*/comm`` to find IO-thread TIDs (``io_thd_*``); the
        main-thread TID equals the PID. Must be called while the server is alive
        (the /proc entries vanish on shutdown). Shared by the CPU-profile and the
        per-thread perf-stat collection paths so both bucket on an identical
        classification. Background threads (``bio_*``, ``jemalloc_bg_thd``) are
        intentionally excluded — they do negligible work and would only add noise.
        """
        self._main_tid = str(self._target_pid)  # main thread TID = PID
        self._io_tids = []
        try:
            import os

            task_dir = f"/proc/{self._target_pid}/task"
            for tid in os.listdir(task_dir):
                comm_path = f"{task_dir}/{tid}/comm"
                try:
                    with open(comm_path) as f:
                        comm = f.read().strip()
                    if comm.startswith("io_thd_"):
                        self._io_tids.append(tid)
                except (FileNotFoundError, PermissionError):
                    pass
        except (FileNotFoundError, PermissionError):
            pass

    def _cpu_profile_run_sync(self, duration: int) -> None:
        """Run perf record synchronously in a thread."""
        ip = self._host.ip
        command = (
            f"sudo perf record -g --call-graph fp -p {self._target_pid} "
            f"-o {CPU_PROFILE_DATA} -- sleep {max(duration - 2, 5)}"
        )
        if ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), ip, command])

    async def cpu_profile_collect(self) -> tuple[list[list], list[list]]:
        """Wait for perf record and return per-thread collapsed stacks.

        Returns:
            (main_stacks, io_stacks) where each is [[stack_string, sample_count], ...].
            Returns ([], []) if collection failed.
        """
        if self._cpu_profile_thread:
            self._cpu_profile_thread.join()
            self._cpu_profile_thread = None

        main_tid = getattr(self, "_main_tid", None)
        io_tids = getattr(self, "_io_tids", [])

        if not main_tid:
            logger.warning("No main thread TID stored (cpu_profile_start not called?)")
            await self._cpu_profile_cleanup()
            return [], []

        stackcollapse = f"{FLAMEGRAPH_DIR}/stackcollapse-perf.pl"

        # Generate collapsed stacks for main thread
        main_cmd = (
            f"bash -c 'sudo perf script -i {CPU_PROFILE_DATA} --tid={main_tid} | "
            f"{stackcollapse} > /tmp/collapsed-main.txt'"
        )
        await self._host.run_host_command(main_cmd)
        main_output, _ = await self._host.run_host_command("cat /tmp/collapsed-main.txt")
        main_stacks = self._parse_collapsed(main_output)

        # Generate collapsed stacks for IO threads
        io_stacks: list[list] = []
        if io_tids:
            io_tid_str = ",".join(io_tids)
            io_cmd = (
                f"bash -c 'sudo perf script -i {CPU_PROFILE_DATA} --tid={io_tid_str} | "
                f"{stackcollapse} > /tmp/collapsed-io.txt'"
            )
            await self._host.run_host_command(io_cmd)
            io_output, _ = await self._host.run_host_command("cat /tmp/collapsed-io.txt")
            io_stacks = self._parse_collapsed(io_output)

        await self._cpu_profile_cleanup()
        return main_stacks, io_stacks

    @staticmethod
    def _parse_collapsed(output: str) -> list[list]:
        """Parse stackcollapse-perf.pl output into [[stack, count], ...] pairs."""
        stacks: list[list] = []
        for line in output.strip().splitlines():
            if not line:
                continue
            # Format: "func1;func2;func3 12345"
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    stacks.append([parts[0], int(parts[1])])
                except ValueError:
                    continue
        return stacks

    async def _cpu_profile_cleanup(self) -> None:
        """Remove CPU profile data file from remote host."""
        await self._host.run_host_command(
            f"sudo rm -f {CPU_PROFILE_DATA} /tmp/collapsed-main.txt /tmp/collapsed-io.txt"
        )

    # =========================================================================
    # PERF STAT (hardware counters)
    # =========================================================================

    async def perf_stat_start(self) -> None:
        """Start perf stat collection on the target process.

        Discovers the main/IO thread TIDs first (while the server is alive) so the
        collection can run in ``--per-thread`` mode and attribute counters to the
        main thread vs the IO-thread group, in addition to the process-wide total.
        """
        if self._perf_stat_thread and self._perf_stat_thread.is_alive():
            raise RuntimeError("Perf stat already running")
        self._discover_thread_tids()
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

    # TMA Level 1 events (Intel only) — must be collected as a group with slots
    PERF_EVENTS_TMA_GROUP = [
        "slots",
        "topdown-retiring",
        "topdown-fe-bound",
        "topdown-be-bound",
        "topdown-bad-spec",
    ]

    def _has_tma_support(self) -> bool:
        """Check if Intel TMA topdown events are available."""
        try:
            result = subprocess.run(["perf", "list"], capture_output=True, text=True, timeout=5)
            return "topdown-retiring" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_perf_events(self) -> list[str]:
        """Return platform-appropriate perf events list."""
        import platform as _platform

        arch = _platform.machine()
        if arch in ("x86_64", "i686"):
            return self.PERF_EVENTS_X86
        return self.PERF_EVENTS_COMMON

    def _build_perf_event_string(self) -> str:
        """Build the -e argument string, handling TMA event groups."""
        events = self._get_perf_events()
        base = ",".join(events)
        if self._has_tma_support():
            tma_group = "{" + ",".join(self.PERF_EVENTS_TMA_GROUP) + "}"
            return f"{tma_group},{base}"
        return base

    def _perf_stat_run_sync(self) -> None:
        """Run perf stat synchronously in a thread.

        Uses ``--per-thread`` with an explicit TID list (main + IO threads) rather
        than ``-p PID``. The explicit-TID form is required on Intel: ``-p PID
        --per-thread`` intermittently fails with "Problems finding threads of
        monitor" because perf enumerates all threads (including transient
        ``bio_*``/``jemalloc_bg_thd`` workers) and errors if one exits during setup.
        Pinning to the stable main+IO TIDs is race-free and also works with the
        Intel TMA ``{...}`` event group (validated on all 3 platforms Jun 2026).

        Falls back to the process-wide ``-p PID`` form when no IO-thread TIDs were
        discovered (e.g. io-threads disabled), preserving the original behavior.
        """
        events = self._build_perf_event_string()
        ip = self._host.ip
        target = self._build_perf_stat_target()
        command = (
            f'perf stat --per-thread -e "{events}" {target} -o {PERF_STATS_PATH} '
            f"-- sh -c 'while [ -f {PERF_STAT_STATUS_FILE} ]; do sleep 1; done'"
        )
        if ip in ["127.0.0.1", "localhost"]:
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(["ssh", "-i", str(config.SSH_KEYFILE), ip, command], check=True)

    def _build_perf_stat_target(self) -> str:
        """Build the perf-stat target selector (``-t TID,...`` or ``-p PID``).

        Prefers an explicit TID list (main + IO threads) for race-free per-thread
        collection. Falls back to ``-p PID`` when no IO TIDs are known.
        """
        tids = []
        if self._main_tid:
            tids.append(self._main_tid)
        tids.extend(self._io_tids)
        if tids:
            return f"-t {','.join(tids)}"
        return f"-p {self._target_pid}"

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
        """Copy perf stat results and return parsed per-thread counters.

        Returns a dict with three counter buckets::

            {"all": {...}, "main": {...}, "io": {...}}

        ``all`` is the process-wide total (sum of every monitored TID row, matching
        the historical single-bucket behavior), ``main`` is the main-thread row, and
        ``io`` is the summed IO-thread rows. Buckets with no data are ``{}``.
        """
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory {result_dir} must exist")
        local_path = result_dir / "perf_stat.txt"
        await self._host.get_remote_file(Path(PERF_STATS_PATH), local_path)
        return self.parse_perf_stat_per_thread(local_path, self._main_tid, self._io_tids)

    @staticmethod
    def parse_perf_stat_per_thread(
        path: Path, main_tid: Optional[str], io_tids: list[str]
    ) -> dict[str, dict[str, int]]:
        """Parse ``perf stat --per-thread`` output into main/io/all counter buckets.

        Per-thread rows look like ``comm-TID  <count>  <event>  [# ...]  [(pct%)]``
        where ``comm`` may itself contain hyphens (e.g. ``valkey-server-2770433``),
        so the TID is recovered by right-splitting the first token on ``-``. Rows
        with ``<not counted>`` / ``<not supported>`` values are skipped. Counts are
        summed per event within each bucket:

        * ``all``  — every monitored TID (process-wide total)
        * ``main`` — rows whose TID == ``main_tid``
        * ``io``   — rows whose TID is in ``io_tids``
        """
        io_set = set(io_tids or [])
        buckets: dict[str, dict[str, int]] = {"all": {}, "main": {}, "io": {}}
        if not path.exists():
            return buckets

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "seconds time elapsed" in line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            comm_tid = parts[0]
            if "-" not in comm_tid:
                continue
            tid = comm_tid.rsplit("-", 1)[1]
            if not tid.isdigit():
                continue
            count_str = parts[1].replace(",", "")
            try:
                count = int(count_str)
            except ValueError:
                continue  # <not counted> / <not supported>
            event = parts[2].removesuffix(":u").removesuffix(":k").removesuffix(":")

            targets = ["all"]
            if main_tid is not None and tid == main_tid:
                targets.append("main")
            if tid in io_set:
                targets.append("io")
            for bucket in targets:
                buckets[bucket][event] = buckets[bucket].get(event, 0) + count

        return buckets

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
        await self._host.run_host_command(f"sudo rm -f {CPU_PROFILE_DATA} {PERF_STATS_PATH}")
