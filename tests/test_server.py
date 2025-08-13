from pathlib import Path

import pytest

from src.config import REPO_NAMES
from src.server import Server

TEST_REPO = REPO_NAMES[0]
TEST_SPECIFIER = "8.0"
DEFAULT_PORT = 6379
TEST_PORTS = 9000


class TestServerIntegration:

    def test_start_and_ping(self):
        valkey_server = Server("127.0.0.1", 6379)
        valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])
        # Server should be ready when start() returns
        pong = valkey_server.run_valkey_command("PING")
        assert pong == "PONG"

    def test_two_servers_same_host(self):
        count = 2
        servers: list[Server] = [Server("127.0.0.1", port) for port in range(TEST_PORTS, TEST_PORTS + count)]
        for instance in servers:
            print("starting server on port", instance.port)
            instance.start(TEST_REPO, TEST_SPECIFIER, [])

        for index, instance in enumerate(servers):
            instance.run_valkey_command(f"SADD server{index} {index}online")

        for index, instance in enumerate(servers):
            # ensure unique server instances
            response = instance.run_valkey_command(f"SCARD server{index}")
            assert response and response == "1"

            response = instance.run_valkey_command(f"SMEMBERS server{index}")
            assert response and response == f"{index}online"

    def test_old_state_cleaned(self):
        valkey_server = Server("127.0.0.1", 6379)
        valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])

        response = valkey_server.info("keyspace")
        assert not response
        valkey_server.run_valkey_command("SET foo bar")
        info = valkey_server.info("keyspace")
        assert "db0" in info and info["db0"] == "keys=1,expires=0,avg_ttl=0"

        # start a new server - should replace previous server
        del valkey_server
        valkey_server = Server("127.0.0.1", 6379)
        valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])
        response = valkey_server.info("keyspace")
        assert not response
