import os
import subprocess
import sys

import pytest
from hypothesis import HealthCheck, settings

from conductress.file_protocol import FileProtocol
from conductress.server import Server

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Property tests here check numerical and parsing invariants, not speed. Hypothesis
# defaults to a per-example deadline and a health check on input-generation time,
# both of which measure wall-clock and fail spuriously when the suite shares a
# CPU with other work. Disable the timing checks; per-test max_examples is kept.
settings.register_profile(
    "conductress",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("conductress")


def pytest_sessionstart(session):
    """Log additional platform information at test session start."""
    print("================================== Platform ====================================")
    try:
        # Get CPU info
        cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if cpu_info.returncode == 0:
            for line in cpu_info.stdout.split("\n"):
                if "Model name:" in line or "Architecture:" in line or "CPU(s):" in line or "NUMA node(s):" in line:
                    print(line.strip())

        # Get memory info
        mem_info = subprocess.run(["free", "-h"], capture_output=True, text=True, check=False)
        if mem_info.returncode == 0:
            lines = mem_info.stdout.split("\n")
            if len(lines) > 1:
                print(f"Memory: {lines[1].split()[1]} total")
    except Exception:
        pass  # Silently ignore platform detection failures


@pytest.fixture(autouse=True)
def reset_cpu_allocator():
    """Reset the class-level CPU allocator between tests to prevent state leaks."""
    yield
    from conductress.cpu_allocator import CpuAllocator

    Server._cpu_allocator = CpuAllocator()


@pytest.fixture(autouse=True, scope="session")
def cleanup_after_tests():
    """Automatically cleanup orphaned tasks after all tests complete."""
    yield  # Run all tests
    # Cleanup after all tests complete
    FileProtocol.cleanup_orphaned_tasks()
