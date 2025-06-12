"""Test full sync throughput"""

import time

from .replication_group import ReplicationGroup
from .utility import HumanByte, print_pretty_header


class TestFullSync:
    """Benchmark full sync throughput"""

    def __init__(
        self,
        task_name: str,
        server_ips: list,
        binary_source: str,
        specifier: str,
        io_threads: int,
        valsize: int,
        valcount: int,
    ):
        """Initialize the test with a replication group."""
        self.task_name = task_name
        self.server_ips = server_ips
        self.binary_source = binary_source
        self.specifier = specifier
        self.io_threads = io_threads
        self.valsize = valsize
        self.valcount = valcount

        assert len(self.server_ips) >= 2, "At least two server IPs are required"

        self.title = (
            f"replication test, {binary_source}:{specifier}, "
            f"io-threads={io_threads}, {HumanByte.to_human(valsize * valcount)} data, "
            f"{len(self.server_ips) - 1} replicas"
        )
        print_pretty_header(self.title)

        print(f"setting up replication group with {len(self.server_ips)} servers")
        self.replication_group = ReplicationGroup(
            self.server_ips, self.binary_source, self.specifier, self.io_threads, []
        )

        print("loading data onto primary")
        self.replication_group.primary.run_command_over_keyspace(self.valcount, f"-d {self.valsize} -t set")

    def run(self) -> None:
        """Run the full sync test."""

        print("beginning full sync...")
        self.replication_group.begin_replication()
        start = time.monotonic()
        self.replication_group.replicas[0].profiling_start(3999)

        in_sync = False
        while not in_sync:
            replica_info = self.replication_group.replicas[0].info("replication")
            in_sync = (
                "master_link_status" in replica_info
                and replica_info["master_link_status"] == "up"
                and replica_info["master_sync_in_progress"] == "0"
            )
            time.sleep(0.25)
        end = time.monotonic()
        print("full sync complete, ending profiling")
        self.replication_group.replicas[0].profiling_end()

        duration = end - start
        user_data_size = self.valsize * self.valcount
        throughput = user_data_size / duration
        print(f"full sync took {duration:.2f} seconds")
        print(
            f"full sync throughput: {HumanByte.to_human(throughput)}/s "
            f"({HumanByte.to_human(user_data_size)} total)"
        )

        print("generating flamegraph")
        self.replication_group.replicas[0].profiling_report(self.task_name)
        print("flamegraph done")

        # TODO log stats and results
