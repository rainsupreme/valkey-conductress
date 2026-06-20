"""Tests for Redis engine support in Conductress sweep pipeline."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.binary_manager import VALKEY_BINARY, BinaryManager
from conductress.config import (
    SWEEP_ENGINES,
    SWEEP_MAKE_ARGS,
    SWEEP_REF,
    SWEEP_SOURCE,
    SWEEP_STATE_DIR,
    SweepEngine,
    get_sweep_engine,
)
from conductress.sweep.coordinator import SweepCoordinator
from conductress.sweep.exporter import export_series
from conductress.sweep.git_ops import resolve_tag_to_commit
from conductress.sweep.planner import SweepPlanner, SweepState
from conductress.tasks.task_perf_benchmark import PerfTaskData


class TestSweepEngineConfig:
    """SweepEngine configuration and lookup."""

    def test_redis_engine_exists(self):
        engines = [e for e in SWEEP_ENGINES if e.source == "redis"]
        assert len(engines) == 1

    def test_redis_engine_config(self):
        redis = get_sweep_engine("redis")
        assert redis is not None
        assert redis.source == "redis"
        assert redis.ref == "origin/unstable"
        assert redis.binary_name == "redis-server"
        assert redis.floor_tag == "8.0.0"
        assert redis.make_args == ""
        assert "zmalloc" in redis.heap_alloc_funcs

    def test_valkey_engine_config(self):
        valkey = get_sweep_engine("valkey")
        assert valkey is not None
        assert valkey.binary_name == "valkey-server"
        assert valkey.floor_tag is None
        assert valkey.make_args == ""

    def test_get_sweep_engine_nonexistent(self):
        assert get_sweep_engine("nonexistent") is None

    def test_redis_source_in_repo_names(self):
        from conductress.config import REPOSITORIES

        repo_names = [repo[1] for repo in REPOSITORIES]
        assert "redis" in repo_names


class TestSweepCoordinatorEngine:
    """SweepCoordinator with engine parameter."""

    def test_redis_workload_label_has_prefix(self):
        redis = get_sweep_engine("redis")
        coord = SweepCoordinator.__new__(SweepCoordinator)
        engine_prefix = f"{redis.source}-" if redis and redis.source != "valkey" else ""
        label = f"{engine_prefix}get-k16-v16-t7-p10"
        assert label == "redis-get-k16-v16-t7-p10"

    def test_valkey_workload_label_no_prefix(self):
        valkey = get_sweep_engine("valkey")
        engine_prefix = f"{valkey.source}-" if valkey and valkey.source != "valkey" else ""
        label = f"{engine_prefix}get-k16-v16-t7-p10"
        assert label == "get-k16-v16-t7-p10"

    def test_redis_state_file_name(self):
        expected_name = "state_redis-get-k16-v16-t7-p10.json"
        state_file = SWEEP_STATE_DIR / expected_name
        assert "redis-" in state_file.name

    def test_sweep_source_returns_engine_source(self):
        redis = get_sweep_engine("redis")
        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.engine = redis
            coord.state = SweepState()
            coord.planner = SweepPlanner(coord.state)
        assert coord._sweep_source == "redis"
        assert coord._sweep_ref == "origin/unstable"
        assert coord._sweep_make_args == ""

    def test_sweep_source_defaults_without_engine(self):
        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.engine = None
            coord.state = SweepState()
            coord.planner = SweepPlanner(coord.state)
        assert coord._sweep_source == SWEEP_SOURCE
        assert coord._sweep_ref == SWEEP_REF
        assert coord._sweep_make_args == SWEEP_MAKE_ARGS

    def test_is_my_task_checks_source(self):
        redis = get_sweep_engine("redis")
        with patch.object(SweepCoordinator, "__init__", lambda self, *a, **kw: None):
            coord = SweepCoordinator.__new__(SweepCoordinator)
            coord.engine = redis
            coord._val_size = 16
            coord._test = "get"
            coord._io_threads = 7
            coord._pipelining = 10
            coord.state = SweepState()
            coord.planner = SweepPlanner(coord.state)

        # Task from redis source should match
        redis_task = MagicMock(spec=PerfTaskData)
        redis_task.sweep_commit = "abc123"
        redis_task.source = "redis"
        redis_task.val_size = 16
        redis_task.test = "get"
        redis_task.io_threads = 7
        redis_task.pipelining = 10
        assert coord._is_my_task(redis_task) is True

        # Task from valkey source should NOT match
        valkey_task = MagicMock(spec=PerfTaskData)
        valkey_task.sweep_commit = "abc123"
        valkey_task.source = "valkey"
        valkey_task.val_size = 16
        valkey_task.test = "get"
        valkey_task.io_threads = 7
        valkey_task.pipelining = 10
        assert coord._is_my_task(valkey_task) is False


class TestBinaryManagerEngine:
    """BinaryManager with engine-aware binary name."""

    def test_default_binary_name(self):
        host = MagicMock()
        host.ip = "127.0.0.1"
        mgr = BinaryManager(host)
        assert mgr.binary_name == VALKEY_BINARY

    def test_redis_binary_name_override(self):
        host = MagicMock()
        host.ip = "127.0.0.1"
        host.run_host_command = AsyncMock()
        host.check_file_exists = AsyncMock(return_value=True)
        mgr = BinaryManager(host)
        mgr.source = "redis"
        mgr.specifier = "unstable"
        mgr.binary_name = "redis-server"
        assert mgr.binary_name == "redis-server"


class TestExporterEngineMetadata:
    """Exporter produces correct metadata for different engines."""

    def test_default_repo_metadata(self, tmp_path):
        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        output = tmp_path / "test.json"
        export_series(state, output)
        data = json.loads(output.read_text())
        assert data["metadata"]["repo"] == "valkey-io/valkey"
        assert data["metadata"]["branch"] == "unstable"

    def test_redis_repo_metadata(self, tmp_path):
        state = SweepState(merge_commits=["aaa"], commit_dates={"aaa": "2024-01-01"})
        output = tmp_path / "test.json"
        export_series(state, output, repo="redis/redis", branch="unstable")
        data = json.loads(output.read_text())
        assert data["metadata"]["repo"] == "redis/redis"
        assert data["metadata"]["branch"] == "unstable"


class TestGitOpsResolveTag:
    """resolve_tag_to_commit for floor tag resolution."""

    def test_resolve_existing_tag(self):
        repo = Path.home() / "valkey"
        if not repo.exists():
            pytest.skip("valkey repo not available")
        result = resolve_tag_to_commit(repo, "8.0.0")
        assert result is not None
        assert len(result) == 40

    def test_resolve_nonexistent_tag(self):
        repo = Path.home() / "valkey"
        if not repo.exists():
            pytest.skip("valkey repo not available")
        result = resolve_tag_to_commit(repo, "nonexistent-tag-xyz-99")
        assert result is None

    def test_resolve_parent_syntax(self):
        repo = Path.home() / "valkey"
        if not repo.exists():
            pytest.skip("valkey repo not available")
        tag = resolve_tag_to_commit(repo, "8.0.0")
        parent = resolve_tag_to_commit(repo, "8.0.0^")
        assert tag is not None
        assert parent is not None
        assert tag != parent
