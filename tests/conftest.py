import os
import subprocess
import sys

import pytest

from src.file_protocol import FileProtocol

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_sessionstart(session):
    """Log additional platform information at test session start."""
    print("================================== Platform ====================================")
    try:
        # Get CPU info
        cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if cpu_info.returncode == 0:
            for line in cpu_info.stdout.split("\n"):
                if (
                    "Model name:" in line
                    or "Architecture:" in line
                    or "CPU(s):" in line
                    or "NUMA node(s):" in line
                ):
                    print(line.strip())

        # Get memory info
        mem_info = subprocess.run(["free", "-h"], capture_output=True, text=True, check=False)
        if mem_info.returncode == 0:
            lines = mem_info.stdout.split("\n")
            if len(lines) > 1:
                print(f"Memory: {lines[1].split()[1]} total")
    except Exception:
        pass  # Silently ignore platform detection failures


@pytest.fixture(autouse=True, scope="session")
def cleanup_after_tests():
    """Automatically cleanup orphaned tasks after all tests complete."""
    yield  # Run all tests
    # Cleanup after all tests complete
    FileProtocol.cleanup_orphaned_tasks()
