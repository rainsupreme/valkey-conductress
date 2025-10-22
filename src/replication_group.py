"""Code to manage a group of servers for replication testing."""

import asyncio
import time
from typing import Optional

from src.config import ServerInfo

from .server import Server


class ReplicationGroup:
    """A group of servers that can be used for replication testing. Cluster mode is assumed disabled."""

    # This assumes that servers are all on different hosts, and uses the standard port number.
    # It would need to be modified to support multiple instances on the same host.

    def __init__(
        self,
        server_infos: list[ServerInfo],
        binary_source: str,
        specifier: str,
        threads: int,
    ) -> None:
        """Initialize a replication group with server configuration."""
        assert len(server_infos) >= 1, "At least one server IP is required"

        self.server_infos = server_infos
        self.binary_source = binary_source
        self.specifier = specifier
        self.threads = threads

        self.servers: list[Server] = []
        self.primary: Optional[Server] = None
        self.replicas: list[Server] = []

    async def start(self) -> None:
        """Start servers in parallel (including building if necessary)"""
        self.servers: list[Server] = await asyncio.gather(
            *[self.__start_server(info.ip, info.username) for info in self.server_infos]
        )
        await self.__ensure_no_unknown_replicas(self.servers)

        self.primary = self.servers[0]
        self.replicas = self.servers[1:]

    async def __start_server(self, server_ip, username) -> Server:
        """Start a single server instance."""
        port = 6379
        server: Server = await Server.with_build(
            server_ip, port, username, self.binary_source, self.specifier, self.threads
        )
        await server.replicate(None)
        await server.wait_until_ready()
        return server

    async def __ensure_no_unknown_replicas(self, expected_servers: list[Server]) -> None:
        """Ensure that there are no unknown replicas in the group."""
        expected_ips = [server.ip for server in expected_servers]
        unexpected_ips = []
        for server in self.servers:
            replicas = await server.get_replicas()
            unexpected_ips += [replica for replica in replicas if replica not in expected_ips]

        if unexpected_ips:
            print(f"Unexpected replicas found: {unexpected_ips}")
            for unexpected in unexpected_ips:
                await Server(unexpected).replicate(None)

            # Wait for a short time to allow the servers to process and gossip
            time.sleep(0.5)

            unexpected_ips = []
            for server in self.servers:
                replicas = await server.get_replicas()
                unexpected_ips += [replica for replica in replicas if replica not in expected_servers]
            assert not unexpected_ips, f"Unexpected replicas remain after cleanup: {unexpected_ips}"

    async def begin_replication(self):
        """Set up replication among the servers in the group."""
        assert self.primary
        print("setting up replication")
        await asyncio.gather(*[replica.replicate(self.primary.ip) for replica in self.replicas])

    async def wait_for_repl_sync(self):
        """Wait for all replicas to be in sync with the primary."""
        assert self.primary
        if not self.replicas:
            print("waiting for replication sync, but there are no replicas")
            return

        for replica in self.replicas:
            while True:
                info = await replica.info("replication")
                if (
                    "master_link_status" in info
                    and info["master_link_status"] == "up"
                    and info["master_sync_in_progress"] == "0"
                ):
                    break
                time.sleep(1)

    async def end_replication(self):
        """End replication for all servers in the group."""
        await asyncio.gather(*[server.replicate(None) for server in self.servers])

    async def stop_all_servers(self) -> None:
        """Stop all server instances in this replication group."""
        await asyncio.gather(*[server.stop() for server in self.servers])

    async def kill_all_valkey_instances(self) -> None:
        """Kill server processes on all servers in group."""
        await asyncio.gather(*[server.kill_all_valkey_instances_on_host() for server in self.servers])
        await asyncio.sleep(1)
