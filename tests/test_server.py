import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import REPO_NAMES
from src.server import Server

TEST_REPO = REPO_NAMES[0]
TEST_SPECIFIER = "8.0"
DEFAULT_PORT = 6379
TEST_PORTS = 9000


class TestServerIntegration:

    @classmethod
    def teardown_class(cls):
        asyncio.run(Server("127.0.0.1").kill_all_valkey_instances_on_host())

    @pytest.mark.asyncio
    async def test_start_and_ping(self):
        valkey_server = Server("127.0.0.1")
        await valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])
        # Server should be ready when start() returns
        pong = await valkey_server.run_valkey_command("PING")
        assert pong == "PONG"

    @pytest.mark.asyncio
    async def test_two_servers_same_host(self) -> None:
        count = 2
        servers: list[Server] = [Server("127.0.0.1", port) for port in range(TEST_PORTS, TEST_PORTS + count)]

        for instance in servers:
            print("starting server on port", instance.port)
            await instance.start(TEST_REPO, TEST_SPECIFIER, [])

        for index, instance in enumerate(servers):
            await instance.run_valkey_command(f"SADD server{index} {index}online")

        async def check_server(instance: Server, index: int) -> None:
            responses = await asyncio.gather(
                *(
                    instance.run_valkey_command(f"SCARD server{index}"),
                    instance.run_valkey_command(f"SMEMBERS server{index}"),
                )
            )
            assert responses[0] and responses[0] == "1"
            assert responses[1] and responses[1] == f"{index}online"

        await asyncio.gather(*(check_server(instance, index) for index, instance in enumerate(servers)))

    @pytest.mark.asyncio
    async def test_old_state_cleaned(self) -> None:
        valkey_server = Server("127.0.0.1")
        await valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])

        info: dict[str, str] = await valkey_server.info("keyspace")
        assert not info
        await valkey_server.run_valkey_command("SET foo bar")
        info = await valkey_server.info("keyspace")
        assert "db0" in info and info["db0"] == "keys=1,expires=0,avg_ttl=0"

        # start a new server - should replace previous server
        del valkey_server
        valkey_server = Server("127.0.0.1")
        await valkey_server.start(TEST_REPO, TEST_SPECIFIER, [])
        info = await valkey_server.info("keyspace")
        assert not info
