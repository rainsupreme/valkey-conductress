"""Tests for ReplicationGroup lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress.config import ServerInfo
from conductress.replication_group import ReplicationGroup


@pytest.fixture
def server_infos():
    return [ServerInfo(ip="10.0.0.1"), ServerInfo(ip="10.0.0.2")]


class TestInit:
    def test_requires_at_least_one_server(self):
        with pytest.raises(ValueError, match="At least one server"):
            ReplicationGroup([], "valkey", "main", 1)

    def test_stores_config(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "unstable", 7, "USE_FAST_FLOAT=yes")
        assert group.binary_source == "valkey"
        assert group.specifier == "unstable"
        assert group.threads == 7
        assert group.make_args == "USE_FAST_FLOAT=yes"
        assert group.primary is None
        assert group.replicas == []


class TestStart:
    @pytest.mark.asyncio
    async def test_start_sets_primary_and_replicas(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)

        mock_server1 = MagicMock()
        mock_server1.ip = "10.0.0.1"
        mock_server1.get_replicas = AsyncMock(return_value=[])
        mock_server1.replicate = AsyncMock()
        mock_server1.wait_until_ready = AsyncMock()

        mock_server2 = MagicMock()
        mock_server2.ip = "10.0.0.2"
        mock_server2.get_replicas = AsyncMock(return_value=[])
        mock_server2.replicate = AsyncMock()
        mock_server2.wait_until_ready = AsyncMock()

        with patch("conductress.replication_group.Server") as MockServer:
            MockServer.with_build = AsyncMock(side_effect=[mock_server1, mock_server2])
            await group.start()

        assert group.primary == mock_server1
        assert group.replicas == [mock_server2]


class TestBeginReplication:
    @pytest.mark.asyncio
    async def test_raises_without_primary(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        # primary is None by default
        with pytest.raises(RuntimeError, match="No primary server"):
            await group.begin_replication()

    @pytest.mark.asyncio
    async def test_replicates_from_primary(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        group.primary = MagicMock(ip="10.0.0.1")
        replica = MagicMock()
        replica.replicate = AsyncMock()
        group.replicas = [replica]

        await group.begin_replication()

        replica.replicate.assert_called_once_with("10.0.0.1")


class TestWaitForSync:
    @pytest.mark.asyncio
    async def test_raises_without_primary(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        with pytest.raises(RuntimeError, match="No primary server"):
            await group.wait_for_repl_sync()

    @pytest.mark.asyncio
    async def test_returns_immediately_with_no_replicas(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        group.primary = MagicMock()
        group.replicas = []
        # Should not hang
        await group.wait_for_repl_sync()


class TestStopAndKill:
    @pytest.mark.asyncio
    async def test_stop_all_servers(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        server1 = MagicMock()
        server1.stop = AsyncMock()
        server2 = MagicMock()
        server2.stop = AsyncMock()
        group.servers = [server1, server2]

        await group.stop_all_servers()

        server1.stop.assert_called_once()
        server2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_all_instances(self, server_infos):
        group = ReplicationGroup(server_infos, "valkey", "main", 1)
        server1 = MagicMock()
        server1.kill_all_valkey_instances_on_host = AsyncMock()
        server2 = MagicMock()
        server2.kill_all_valkey_instances_on_host = AsyncMock()
        group.servers = [server1, server2]

        await group.kill_all_valkey_instances()

        server1.kill_all_valkey_instances_on_host.assert_called_once()
        server2.kill_all_valkey_instances_on_host.assert_called_once()
