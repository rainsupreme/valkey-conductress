"""CPU consistency tuning for repeatable benchmarks.

Configures CPU frequency, idle states, ASLR, and scheduler settings
to minimize variance between benchmark runs across ARM/x86 platforms.
"""

import logging
from pathlib import Path
from typing import Optional

from .platform import PlatformInfo, detect_platform
from .ssh_host import SshHost

logger = logging.getLogger(__name__)


class StabilizationManager:
    """Manages CPU/OS stabilization settings on a remote host."""

    def __init__(self, host: SshHost) -> None:
        self._host = host
        self.platform_info: Optional[PlatformInfo] = None

    async def enable(self) -> None:
        """Configure CPU settings for consistent benchmarks across ARM/x86 platforms."""
        self.platform_info = await detect_platform(lambda cmd: self._host.run_host_command(cmd, check=False))

        await self._disable_aslr()
        await self._set_thp_madvise()
        await self._apply_sysctl_tunings()
        await self._stop_noisy_timers()
        await self._configure_frequency_scaling()
        await self._configure_idle_states()
        await self._configure_scheduler()

    async def disable(self) -> None:
        """Restore default CPU settings."""
        await self._host.run_host_command("echo 2 | sudo tee /proc/sys/kernel/randomize_va_space", check=False)
        logger.info("Restored ASLR (randomize_va_space=2)")

        cpufreq_exists = await self._host.check_file_exists(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors")
        )
        if cpufreq_exists:
            governors_out, _ = await self._host.run_host_command(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors", check=False
            )
            available = governors_out.strip().split() if governors_out else []
            if "schedutil" in available:
                await self._host.run_host_command("sudo cpupower frequency-set -g schedutil")
            elif "ondemand" in available:
                await self._host.run_host_command("sudo cpupower frequency-set -g ondemand")

            for boost_path in [
                "/sys/devices/system/cpu/cpufreq/boost",
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_boost_frequencies",
            ]:
                if await self._host.check_file_exists(Path(boost_path)):
                    await self._host.run_host_command(f"echo 1 | sudo tee {boost_path}", check=False)
                    break

        idle_result, _ = await self._host.run_host_command("cpupower idle-info", check=False)
        if "No idle states" not in idle_result and "CPUidle driver: none" not in idle_result:
            for state in [1, 2, 3]:
                await self._host.run_host_command(f"sudo cpupower idle-set -e {state}", check=False)

        for path, value in [
            ("/proc/sys/kernel/sched_energy_aware", "1"),
            ("/proc/sys/kernel/sched_autogroup_enabled", "1"),
        ]:
            if await self._host.check_file_exists(Path(path)):
                await self._host.run_host_command(f"echo {value} | sudo tee {path}", check=False)

    async def verify(self) -> bool:
        """Verify stabilization settings are applied. Retry once on failure."""
        checks: list[tuple[str, str, str]] = [
            ("ASLR", "cat /proc/sys/kernel/randomize_va_space", "0"),
        ]
        cpufreq_exists = await self._host.check_file_exists(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors")
        )
        if cpufreq_exists:
            checks.append(("governor", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "performance"))
            if await self._host.check_file_exists(Path("/sys/devices/system/cpu/cpufreq/boost")):
                checks.append(("boost", "cat /sys/devices/system/cpu/cpufreq/boost", "0"))

        all_ok = True
        for name, cmd, expected in checks:
            actual, _ = await self._host.run_host_command(cmd, check=False)
            actual = actual.strip()
            if actual != expected:
                logger.warning("Stabilization FAILED: %s = %r (expected %r). Retrying...", name, actual, expected)
                await self.enable()
                actual, _ = await self._host.run_host_command(cmd, check=False)
                actual = actual.strip()
                if actual != expected:
                    logger.error("Stabilization FAILED after retry: %s = %r (expected %r)", name, actual, expected)
                    all_ok = False
                else:
                    logger.info("Stabilization recovered: %s = %s", name, actual)

        if all_ok:
            logger.info("Stabilization verified: all checks passed")
        return all_ok

    # --- Private helpers ---

    async def _disable_aslr(self) -> None:
        await self._host.run_host_command("echo 0 | sudo tee /proc/sys/kernel/randomize_va_space", check=False)
        logger.info("Disabled ASLR (randomize_va_space=0)")

    async def _set_thp_madvise(self) -> None:
        thp_path = "/sys/kernel/mm/transparent_hugepage/enabled"
        if await self._host.check_file_exists(Path(thp_path)):
            await self._host.run_host_command(f"echo madvise | sudo tee {thp_path}", check=False)
            logger.info("Set THP to madvise")

    async def _apply_sysctl_tunings(self) -> None:
        for key, value in [
            ("vm.compaction_proactiveness", "0"),
            ("kernel.watchdog", "0"),
            ("kernel.timer_migration", "0"),
            ("vm.dirty_writeback_centisecs", "0"),
        ]:
            await self._host.run_host_command(f"sudo sysctl -w {key}={value}", check=False)
        logger.info("Applied memory/scheduler sysctl tunings")

    async def _stop_noisy_timers(self) -> None:
        for timer in ["sysstat-collect.timer", "nm-cloud-setup.timer", "dnf-makecache.timer"]:
            await self._host.run_host_command(f"sudo systemctl stop {timer}", check=False)

    async def _configure_frequency_scaling(self) -> None:
        cpufreq_exists = await self._host.check_file_exists(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors")
        )
        if not cpufreq_exists:
            logger.info("No CPU frequency scaling - likely fixed-frequency processor")
            return

        governors_out, _ = await self._host.run_host_command(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors", check=False
        )
        available = governors_out.strip().split() if governors_out else []
        if "performance" in available:
            await self._host.run_host_command("sudo cpupower frequency-set -g performance")
            logger.info("Set performance governor")
        elif "userspace" in available:
            await self._host.run_host_command("sudo cpupower frequency-set -g userspace")
            logger.info("Set userspace governor (performance fallback)")

        for boost_path in [
            "/sys/devices/system/cpu/cpufreq/boost",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_boost_frequencies",
        ]:
            if await self._host.check_file_exists(Path(boost_path)):
                await self._host.run_host_command(f"echo 0 | sudo tee {boost_path}", check=False)
                logger.info("Disabled boost/turbo at %s", boost_path)
                break

    async def _configure_idle_states(self) -> None:
        idle_result, _ = await self._host.run_host_command("cpupower idle-info", check=False)
        if "No idle states" in idle_result or "CPUidle driver: none" in idle_result:
            logger.info("No CPU idle states - processor maintains consistent performance")
        else:
            for state in [1, 2, 3]:
                await self._host.run_host_command(f"sudo cpupower idle-set -d {state}", check=False)
            logger.info("Disabled CPU idle states for latency consistency")

    async def _configure_scheduler(self) -> None:
        for path, value, desc in [
            ("/proc/sys/kernel/sched_energy_aware", "0", "energy-aware scheduling"),
            ("/proc/sys/kernel/sched_autogroup_enabled", "0", "automatic process grouping"),
        ]:
            if await self._host.check_file_exists(Path(path)):
                await self._host.run_host_command(f"echo {value} | sudo tee {path}", check=False)
                logger.info("Disabled %s for consistent performance", desc)
