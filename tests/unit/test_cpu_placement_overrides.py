"""Tests for CPU placement override feature.

Tests cover:
- cpulist validation (valid/invalid syntax)
- Round-trip persistence (save -> load -> fields preserved)
- Override-bypasses-allocator behavior (mock allocator, assert not called when override set)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress import config
from conductress.task_queue import BaseTaskData
from conductress.tasks.task_perf_benchmark import PerfTaskData, PerfTaskRunner
from conductress.utility import parse_cpulist, validate_cpulist


@pytest.fixture(autouse=True)
def patch_config(monkeypatch):
    """Patch config values for test isolation."""
    monkeypatch.setattr(config, "MANUALLY_UPLOADED", "manual")
    monkeypatch.setattr(config, "REPO_NAMES", ["repo1", "repo2", "valkey"])


class TestCpulistValidation:
    """Tests for cpulist syntax validation."""

    def test_valid_single_cpu(self):
        validate_cpulist("0")

    def test_valid_range(self):
        validate_cpulist("0-3")

    def test_valid_list(self):
        validate_cpulist("0,1,2,3")

    def test_valid_mixed(self):
        validate_cpulist("0-3,8-11,16,24-31")

    def test_valid_empty_is_ok(self):
        validate_cpulist("")

    def test_invalid_spaces(self):
        with pytest.raises(ValueError, match="Invalid cpulist"):
            validate_cpulist("0, 1, 2")

    def test_invalid_letters(self):
        with pytest.raises(ValueError, match="Invalid cpulist"):
            validate_cpulist("0-3,cpu4")

    def test_invalid_semicolons(self):
        with pytest.raises(ValueError, match="Invalid cpulist"):
            validate_cpulist("0;1;2")

    def test_invalid_brackets(self):
        with pytest.raises(ValueError, match="Invalid cpulist"):
            validate_cpulist("[0-3]")

    def test_invalid_slash(self):
        with pytest.raises(ValueError, match="Invalid cpulist"):
            validate_cpulist("0-3/2")


class TestParseCpulist:
    """Tests for cpulist parsing into int lists."""

    def test_single_cpu(self):
        assert parse_cpulist("5") == [5]

    def test_range(self):
        assert parse_cpulist("0-3") == [0, 1, 2, 3]

    def test_comma_list(self):
        assert parse_cpulist("8,10,12") == [8, 10, 12]

    def test_mixed(self):
        assert parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]

    def test_sorted_output(self):
        assert parse_cpulist("10,2,5-7") == [2, 5, 6, 7, 10]


class TestRoundTripPersistence:
    """Tests for save -> load -> use round-trip with CPU override fields."""

    def test_round_trip_with_overrides(self, tmp_path):
        """CPU override fields survive JSON serialization/deserialization."""
        task = PerfTaskData(
            source="manual",
            specifier="test-commit",
            make_args="",
            replicas=0,
            note="chiplet experiment",
            requirements={},
            test="get",
            val_size=16,
            io_threads=9,
            pipelining=10,
            warmup=30,
            duration=300,
            perf_stat_enabled=True,
            has_expire=False,
            preload_keys=True,
            server_cpu_override="0-3,8-11",
            benchmark_cpu_override="16-23",
        )

        task_file = tmp_path / "task.json"
        task.save_to_file(task_file)

        loaded = BaseTaskData.from_file(task_file)
        assert isinstance(loaded, PerfTaskData)
        assert loaded.server_cpu_override == "0-3,8-11"
        assert loaded.benchmark_cpu_override == "16-23"

    def test_round_trip_empty_overrides(self, tmp_path):
        """Empty overrides (default) survive round-trip."""
        task = PerfTaskData(
            source="manual",
            specifier="unstable",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            test="set",
            val_size=128,
            io_threads=4,
            pipelining=1,
            warmup=10,
            duration=60,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
        )

        task_file = tmp_path / "task.json"
        task.save_to_file(task_file)

        loaded = BaseTaskData.from_file(task_file)
        assert isinstance(loaded, PerfTaskData)
        assert loaded.server_cpu_override == ""
        assert loaded.benchmark_cpu_override == ""

    def test_backward_compat_missing_fields(self, tmp_path):
        """Loading a task JSON from before the feature (no override fields) works."""
        task_data = {
            "source": "manual",
            "specifier": "abc123",
            "make_args": "",
            "replicas": 0,
            "note": "",
            "requirements": {},
            "test": "get",
            "val_size": 16,
            "io_threads": 9,
            "pipelining": 10,
            "warmup": 30,
            "duration": 300,
            "perf_stat_enabled": False,
            "has_expire": False,
            "preload_keys": True,
            "task_type": "PerfTaskData",
            "timestamp": "2024-01-01T00:00:00",
        }
        task_file = tmp_path / "task_old.json"
        task_file.write_text(json.dumps(task_data))

        loaded = BaseTaskData.from_file(task_file)
        assert isinstance(loaded, PerfTaskData)
        assert loaded.server_cpu_override == ""
        assert loaded.benchmark_cpu_override == ""


class TestOverrideBypassesAllocator:
    """Tests that setting CPU overrides bypasses the topology-aware allocator."""

    def test_benchmark_override_skips_allocator(self):
        """When benchmark_cpu_override is set, _allocate_benchmark_cpus returns None
        without calling the allocator."""
        runner = PerfTaskRunner(
            task_name="test_task",
            server_infos=[config.ServerInfo(ip="127.0.0.1", username="")],
            binary_source="manual",
            specifier="test",
            io_threads=4,
            valsize=16,
            pipelining=1,
            test="get",
            warmup=5,
            duration=10,
            preload_keys=True,
            has_expire=False,
            make_args="",
            benchmark_cpu_override="16-23",
        )

        mock_client = MagicMock()
        mock_server = MagicMock()
        mock_server.ip = "127.0.0.1"

        result = runner._allocate_benchmark_cpus(mock_client, mock_server)

        # Should return None (skip allocation) and never touch the allocator
        assert result is None
        mock_client._cpu_allocator.allocate.assert_not_called()

    def test_empty_override_uses_allocator(self):
        """When benchmark_cpu_override is empty, normal allocation path runs."""
        runner = PerfTaskRunner(
            task_name="test_task",
            server_infos=[config.ServerInfo(ip="127.0.0.1", username="")],
            binary_source="manual",
            specifier="test",
            io_threads=4,
            valsize=16,
            pipelining=1,
            test="get",
            warmup=5,
            duration=10,
            preload_keys=True,
            has_expire=False,
            make_args="",
            benchmark_cpu_override="",
        )

        mock_client = MagicMock()
        mock_client._cpu_allocator.get_net_interface_numa.return_value = 0
        mock_client._cpu_allocator.allocate.return_value = [16, 17, 18, 19]
        mock_client.ip = "127.0.0.1"

        mock_server = MagicMock()
        mock_server.ip = "127.0.0.1"
        mock_server.port = 6379
        mock_server._platform_info = None

        result = runner._allocate_benchmark_cpus(mock_client, mock_server)

        # Should have called the allocator
        mock_client._cpu_allocator.allocate.assert_called_once()
        assert result is not None

    def test_benchmark_command_uses_override(self):
        """When benchmark_cpu_override is set, _build_benchmark_command uses it verbatim."""
        runner = PerfTaskRunner(
            task_name="test_task",
            server_infos=[config.ServerInfo(ip="127.0.0.1", username="")],
            binary_source="manual",
            specifier="test",
            io_threads=4,
            valsize=16,
            pipelining=1,
            test="get",
            warmup=5,
            duration=10,
            preload_keys=True,
            has_expire=False,
            make_args="",
            benchmark_cpu_override="32-39,48-55",
        )

        mock_client = MagicMock()
        mock_client._cpu_allocator.get_net_interface_numa.return_value = 0
        mock_client.ip = "127.0.0.1"

        cmd = runner._build_benchmark_command(mock_client, "127.0.0.1", None)
        assert "--physcpubind=32-39,48-55" in cmd

    def test_server_override_passed_to_runner(self):
        """PerfTaskData.prepare_task_runner passes server_cpu_override to PerfTaskRunner."""
        task = PerfTaskData(
            source="manual",
            specifier="test",
            make_args="",
            replicas=0,
            note="",
            requirements={},
            test="get",
            val_size=16,
            io_threads=9,
            pipelining=10,
            warmup=5,
            duration=10,
            perf_stat_enabled=False,
            has_expire=False,
            preload_keys=True,
            server_cpu_override="0-8",
            benchmark_cpu_override="96-191",
        )

        runner = task.prepare_task_runner([config.ServerInfo(ip="127.0.0.1", username="")])
        assert runner.server_cpu_override == "0-8"
        assert runner.benchmark_cpu_override == "96-191"


class TestOverrideMembindFollowsCpus:
    """The benchmark membind must follow the override CPUs' NUMA node(s)."""

    def test_numa_nodes_for_cpus(self):
        from conductress.cpu_allocator import CpuAllocator

        alloc = CpuAllocator()
        alloc.register_host(
            "127.0.0.1",
            all_cpus=list(range(8)),
            numa_topology={0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        )
        assert alloc.get_numa_nodes_for_cpus("127.0.0.1", [0, 1]) == [0]
        assert alloc.get_numa_nodes_for_cpus("127.0.0.1", [5]) == [1]
        assert alloc.get_numa_nodes_for_cpus("127.0.0.1", [1, 6]) == [0, 1]
        assert alloc.get_numa_nodes_for_cpus("127.0.0.1", []) == []

    def test_unknown_host_returns_empty(self):
        from conductress.cpu_allocator import CpuAllocator

        alloc = CpuAllocator()
        assert alloc.get_numa_nodes_for_cpus("10.0.0.9", [0]) == []
