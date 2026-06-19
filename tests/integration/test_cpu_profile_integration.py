"""Integration test for CPU profiling (perf record + stackcollapse) pipeline.

Requires:
- valkey-server binary available (builds or pre-built)
- sudo perf record accessible
- ~/FlameGraph/stackcollapse-perf.pl installed

Run with: pytest tests/integration/test_cpu_profile_integration.py -v
"""

import asyncio
import os
import signal
import subprocess
import time

import pytest

from conductress.profiling_manager import ProfilingManager
from conductress.ssh_host import SshHost


@pytest.fixture
def valkey_server():
    """Start a valkey-server on port 6399 for testing, yield PID, then kill."""
    # Find a valkey-server binary
    candidates = [
        os.path.expanduser("~/valkey/src/valkey-server"),
        os.path.expanduser("~/build_cache/valkey/latest/valkey-server"),
    ]
    binary = None
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            binary = c
            break
    if not binary:
        pytest.skip("No valkey-server binary found")

    proc = subprocess.Popen(
        [binary, "--port", "6399", "--save", "", "--daemonize", "no", "--loglevel", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    if proc.poll() is not None:
        pytest.skip("valkey-server failed to start")

    yield proc.pid

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def profiling_manager():
    """Create a ProfilingManager connected to localhost."""
    host = SshHost("127.0.0.1")
    return ProfilingManager(host)


class TestCpuProfileIntegration:
    """End-to-end test of CPU profiling pipeline on localhost."""

    def test_cpu_profile_collects_stacks(self, valkey_server, profiling_manager):
        """Full pipeline: start perf record, collect collapsed stacks, verify format."""
        pid = valkey_server
        profiling_manager.target_pid = pid

        # Start CPU profile for 3 seconds
        profiling_manager.cpu_profile_start(3)

        # Generate some load while profiling
        subprocess.run(
            ["redis-cli", "-p", "6399", "DEBUG", "SLEEP", "0"],
            capture_output=True,
            timeout=5,
        )
        time.sleep(4)  # Wait for perf record to finish (3s + buffer)

        # Collect stacks
        async def collect():
            # Need SSH connection for run_host_command
            await profiling_manager._host.connect()
            return await profiling_manager.cpu_profile_collect()

        main_stacks, io_stacks = asyncio.run(collect())

        # Assertions
        assert len(main_stacks) > 0, "Expected non-empty main thread stacks"
        for entry in main_stacks:
            assert len(entry) == 2, f"Each stack entry should be [str, int], got {entry}"
            assert isinstance(entry[0], str), f"Stack string expected, got {type(entry[0])}"
            assert isinstance(entry[1], int), f"Sample count expected, got {type(entry[1])}"
            assert ";" in entry[0], f"Stack should have semicolons: {entry[0][:50]}"

        # Total samples should be reasonable (>1000 in 3 seconds)
        total_samples = sum(e[1] for e in main_stacks)
        assert total_samples > 1000, f"Expected >1000 samples, got {total_samples}"

        # IO stacks may be empty (no io-threads configured) — that's fine
        # But if present, same format
        for entry in io_stacks:
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], int)
