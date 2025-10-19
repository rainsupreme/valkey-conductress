import asyncio

import pytest

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
    async def test_start_and_ping(self) -> None:
        valkey_server: Server = await Server.with_build(
            "127.0.0.1",
            DEFAULT_PORT,
            "ec2-user",
            TEST_REPO,
            TEST_SPECIFIER,
            io_threads=-1,
        )
        try:
            # Server should be ready when start() returns
            pong = await valkey_server.run_valkey_command("PING")
            assert pong == "PONG"
        finally:
            await valkey_server.kill_all_valkey_instances_on_host()

    @pytest.mark.asyncio
    async def test_two_servers_same_host(self) -> None:
        count = 2

        # First ensure binary is cached (avoid concurrent git/build operations)
        await Server("127.0.0.1").kill_all_valkey_instances_on_host()
        cached_binary_path = await Server("127.0.0.1").ensure_binary_cached(
            source=TEST_REPO, specifier=TEST_SPECIFIER
        )

        # Then start multiple servers using the cached binary
        servers: list[Server] = await asyncio.gather(
            *[
                Server.with_path("127.0.0.1", port, cached_binary_path, io_threads=1)
                for port in range(TEST_PORTS, TEST_PORTS + count)
            ]
        )

        try:
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

            await asyncio.gather(
                *(
                    check_server(instance, index)
                    for index, instance in enumerate(servers)
                )
            )
        finally:
            # Clean up all servers
            await Server("127.0.0.1").kill_all_valkey_instances_on_host()

    @pytest.mark.asyncio
    async def test_old_state_cleaned(self) -> None:
        valkey_server = await Server.with_build(
            "127.0.0.1",
            DEFAULT_PORT,
            "ec2-user",
            TEST_REPO,
            TEST_SPECIFIER,
            io_threads=-1,
        )

        try:
            info: dict[str, str] = await valkey_server.info("keyspace")
            assert not info
            await valkey_server.run_valkey_command("SET foo bar")
            info = await valkey_server.info("keyspace")
            assert "db0" in info and info["db0"] == "keys=1,expires=0,avg_ttl=0"

            # start a new server - should replace previous server
            del valkey_server
            valkey_server = await Server.with_build(
                "127.0.0.1",
                DEFAULT_PORT,
                "ec2-user",
                TEST_REPO,
                TEST_SPECIFIER,
                io_threads=-1,
            )
            info = await valkey_server.info("keyspace")
            assert not info
        finally:
            await valkey_server.kill_all_valkey_instances_on_host()
