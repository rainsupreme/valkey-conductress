"""Code to manage a group of servers for replication testing."""

import concurrent.futures
import time

from .server import Server


class ReplicationGroup:
    """A group of servers that can be used for replication testing."""

    def __init__(
        self, server_ips: list[str], binary_source: str, specifier: str, threads: int, args: list[str]
    ) -> None:
        assert len(server_ips) >= 1, "At least one server IP is required"

        self.server_ips = server_ips
        self.binary_source = binary_source
        self.specifier = specifier
        self.threads = threads
        self.args = args

        # start servers in parallel (including building if necessary)
        self.servers: list[Server] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.servers = list(executor.map(self.__start_server, self.server_ips))
        self.__ensure_no_unknown_replicas(self.servers)

        self.primary = self.servers[0]
        self.replicas = self.servers[1:]

    def __del__(self) -> None:
        """Clean up the servers when the group is deleted."""
        for server in self.servers:
            server.replicate(None)

    def __start_server(self, server_ip) -> Server:
        port = 6379
        server = Server.with_build(
            server_ip, port, self.binary_source, self.specifier, self.threads, self.args
        )
        server.replicate(None)
        server.wait_until_ready()
        return server

    def __ensure_no_unknown_replicas(self, expected_servers: list[Server]) -> None:
        """Ensure that there are no unknown replicas in the group."""
        expected_ips = [server.ip for server in expected_servers]
        unexpected_ips = []
        for server in self.servers:
            replicas = server.get_replicas()
            unexpected_ips += [replica for replica in replicas if replica not in expected_ips]

        if unexpected_ips:
            print(f"Unexpected replicas found: {unexpected_ips}")
            for unexpected in unexpected_ips:
                Server(unexpected).replicate(None)

            # Wait for a short time to allow the servers to process and gossip
            time.sleep(0.5)

            unexpected_ips = []
            for server in self.servers:
                replicas = server.get_replicas()
                unexpected_ips += [replica for replica in replicas if replica not in expected_servers]
            assert not unexpected_ips, f"Unexpected replicas remain after cleanup: {unexpected_ips}"

    def begin_replication(self):
        """Set up replication among the servers in the group."""
        if not self.replicas:
            return
        print("setting up replication")
        for replica in self.replicas:
            replica.replicate(self.primary.ip)

    def wait_for_repl_sync(self):
        """Wait for all replicas to be in sync with the primary."""
        for replica in self.replicas:
            while True:
                info = replica.info("replication")
                if (
                    "master_link_status" in info
                    and info["master_link_status"] == "up"
                    and info["master_sync_in_progress"] == "0"
                ):
                    break
                time.sleep(1)

    def end_replication(self):
        """End replication for all servers in the group."""
        for server in self.servers:
            server.run_valkey_command("replicaof no one")
