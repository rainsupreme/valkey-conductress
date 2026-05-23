"""Unit tests for Server class pure logic (no SSH/network required)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.server import Server


class TestEnsureStr:
    """Tests for Server.__ensure_str() — output normalization."""

    def test_none_returns_empty(self):
        assert Server._Server__ensure_str(None) == ""

    def test_empty_string_returns_empty(self):
        assert Server._Server__ensure_str("") == ""

    def test_empty_bytes_returns_empty(self):
        assert Server._Server__ensure_str(b"") == ""

    def test_string_passthrough(self):
        assert Server._Server__ensure_str("hello") == "hello"

    def test_bytes_decoded(self):
        assert Server._Server__ensure_str(b"hello") == "hello"

    def test_bytearray_decoded(self):
        assert Server._Server__ensure_str(bytearray(b"hello")) == "hello"

    def test_memoryview_decoded(self):
        data = memoryview(b"hello world")
        assert Server._Server__ensure_str(data) == "hello world"

    def test_utf8_bytes(self):
        assert Server._Server__ensure_str("café".encode()) == "café"


class TestGetCachedBuildPath:
    """Tests for Server.__get_cached_build_path() — build cache path construction."""

    def test_basic_path_construction(self):
        server = Server("127.0.0.1")
        server.source = "valkey"
        server.hash = "abc123def456"
        server.make_args = "USE_FAST_FLOAT=yes"

        path = server._Server__get_cached_build_path()

        assert path.parts[-3] == "valkey"
        assert path.parts[-2] == "abc123def456"
        # make_args hash is 16 chars of md5
        assert len(path.parts[-1]) == 16

    def test_different_make_args_different_path(self):
        server = Server("127.0.0.1")
        server.source = "valkey"
        server.hash = "abc123"

        server.make_args = "USE_FAST_FLOAT=yes"
        path1 = server._Server__get_cached_build_path()

        server.make_args = ""
        path2 = server._Server__get_cached_build_path()

        assert path1 != path2

    def test_empty_make_args_valid(self):
        server = Server("127.0.0.1")
        server.source = "valkey"
        server.hash = "abc123"
        server.make_args = ""

        path = server._Server__get_cached_build_path()
        assert len(path.parts[-1]) == 16  # still a valid md5 prefix

    def test_raises_when_source_none(self):
        server = Server("127.0.0.1")
        server.source = None
        server.hash = "abc123"

        with pytest.raises(RuntimeError, match="source and hash must be set"):
            server._Server__get_cached_build_path()

    def test_raises_when_hash_none(self):
        server = Server("127.0.0.1")
        server.source = "valkey"
        server.hash = None

        with pytest.raises(RuntimeError, match="source and hash must be set"):
            server._Server__get_cached_build_path()


class TestGetSourceBinaryPath:
    """Tests for Server.__get_source_binary_path()."""

    def test_basic_path(self):
        server = Server("127.0.0.1")
        server.source = "valkey"

        path = server._Server__get_source_binary_path()
        assert str(path).endswith("valkey/src")

    def test_raises_when_source_none(self):
        server = Server("127.0.0.1")
        server.source = None

        with pytest.raises(RuntimeError, match="source must be set"):
            server._Server__get_source_binary_path()


class TestGetBuildHash:
    """Tests for Server.get_build_hash()."""

    def test_returns_hash_when_set(self):
        server = Server("127.0.0.1")
        server.hash = "deadbeef1234"
        assert server.get_build_hash() == "deadbeef1234"

    def test_returns_none_when_not_set(self):
        server = Server("127.0.0.1")
        assert server.get_build_hash() is None


class TestReleaseServerCpus:
    """Tests for Server._release_server_cpus()."""

    def test_release_clears_state(self):
        from src.cpu_allocator import AllocationTag, CpuAllocator

        allocator = CpuAllocator()
        allocator.register_host("127.0.0.1", all_cpus=list(range(16)))
        Server._cpu_allocator = allocator

        server = Server("127.0.0.1")
        tag = AllocationTag(task_id="server_127.0.0.1_6379", purpose="server")
        server._allocation_tag = tag
        server.server_cpus = allocator.allocate("127.0.0.1", tag, count=4)

        assert allocator.get_available_count("127.0.0.1") == 12

        server._release_server_cpus()

        assert server.server_cpus == []
        assert server._allocation_tag is None
        assert allocator.get_available_count("127.0.0.1") == 16

    def test_release_noop_when_no_cpus(self):
        """Release is safe to call when no CPUs are allocated."""
        server = Server("127.0.0.1")
        server.server_cpus = []
        server._allocation_tag = None
        # Should not raise
        server._release_server_cpus()


class TestServerInit:
    """Tests for Server initialization defaults."""

    def test_default_port(self):
        server = Server("10.0.0.1")
        assert server.port == 6379

    def test_custom_port(self):
        server = Server("10.0.0.1", port=7777)
        assert server.port == 7777

    def test_initial_state(self):
        server = Server("10.0.0.1")
        assert server.valkey_pid == -1
        assert server.server_cpus == []
        assert server.source is None
        assert server.hash is None
        assert server.ssh is None
