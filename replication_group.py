"""Code to manage a group of servers for replication testing."""

import concurrent.futures
import time

from server import Server


class ReplicationGroup:
    """A group of servers that can be used for replication testing."""

    def __init__(
        self, server_ips: list, binary_source: str, specifier: str, threads: int, args: list
    ) -> None:
        assert len(server_ips) >= 1, "At least one server IP is required"

        self.server_ips = server_ips
        self.binary_source = binary_source
        self.specifier = specifier
        self.threads = threads
        self.args = args

        # start servers in parallel (including building if necessary)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.servers = list(executor.map(self.__start_server, self.server_ips))
        for server in self.servers:
            server.wait_until_ready()

        self.primary = self.servers[0]
        self.replicas = self.servers[1:]

    def __start_server(self, server_ip):
        return Server.with_build(server_ip, self.binary_source, self.specifier, self.threads, self.args)

    def begin_replication(self):
        """Set up replication among the servers in the group."""
        # Assuming the first server is the primary and the rest are replicas
        print("setting up replication")
        for replica in self.replicas:
            command = ["replicaof", self.primary.ip, "6379"]
            print(f"{replica.ip}: {' '.join(command)}")
            response = replica.run_valkey_command(command)
            print(f"{replica.ip}: {response}")
            assert response == "OK", f"got {repr(response)}"

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
            server.run_valkey_command(["replicaof", "no", "one"])
