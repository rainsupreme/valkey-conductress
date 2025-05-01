"""Code to manage a group of servers for replication testing."""

import concurrent.futures

from server import Server


class ReplicationGroup:
    """A group of servers that can be used for replication testing."""

    def __init__(self, server_ips: list, binary_source: str, specifier: str, args: list) -> None:
        assert len(server_ips) >= 1, "At least one server IP is required"

        self.server_ips = server_ips
        self.binary_source = binary_source
        self.specifier = specifier
        self.args = args

        # start servers in parallel (including building if necessary)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.servers = list(executor.map(self.__create_server, self.server_ips))

        self.begin_replication()

    def __create_server(self, server_ip):
        return Server.with_build(server_ip, self.binary_source, self.specifier, self.args)

    def begin_replication(self):
        """Set up replication among the servers in the group."""
        # Assuming the first server is the primary and the rest are replicas
        self.primary = self.servers[0]
        for replica in self.servers[1:]:
            replica.run_valkey_command(f"replicaof {self.primary.ip}")

    def end_replication(self):
        """End replication for all servers in the group."""
        for server in self.servers:
            server.run_valkey_command("replicaof no one")
