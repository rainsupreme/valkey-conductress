"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import asyncio
import datetime
import logging
from dataclasses import dataclass
from types import CoroutineType
from typing import Optional

import plotext as plt

from src.server import Server
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import (
    MILLION,
    HumanByte,
    get_host_proc_count,
    port_generator,
    print_pretty_header,
    record_task_result,
)

logger = logging.getLogger(__name__)


@dataclass
class MemTaskData(BaseTaskData):
    """data class for memory efficiency task"""

    type: str
    val_sizes: list[int]
    has_expire: bool

    def short_description(self) -> str:
        description = self.type
        if len(self.val_sizes) > 1:
            description += f" {len(self.val_sizes)} sizes"
        else:
            description += f" {HumanByte.to_human(self.val_sizes[0])}"
        if self.has_expire:
            description += " with expiration"
        return description

    def prepare_task_runner(self, server_ips: list[str]) -> "MemTaskRunner":
        """Return the task runner for this task."""
        return MemTaskRunner(
            server_ips[0],
            self.source,
            self.specifier,
            self.type,
            self.val_sizes,
            self.has_expire,
        )


class MemTaskRunner(BaseTaskRunner):
    """Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

    tests = ["set", "sadd", "zadd"]  # TODO hset? others?

    def __init__(
        self, server_ip: str, source: str, specifier: str, test: str, val_sizes: list[int], has_expire: bool
    ):

        self.logger = logging.getLogger(self.__class__.__name__ + "." + test)

        self.title = f"{test} memory efficiency, {source}:{specifier}, has_expire={has_expire}"

        assert test in self.tests, f"Test {test} is not supported. Supported tests: {self.tests}"

        # settings
        self.server_ip = server_ip
        self.source = source
        self.specifier = specifier
        self.test = test
        self.val_sizes = val_sizes
        self.has_expire = has_expire

        # test data
        self.commit_hash: Optional[str] = None

    async def run(self) -> None:
        """Run the memory efficiency test for each value size."""
        print_pretty_header(self.title)
        print(f"Val Sizes: {', '.join(HumanByte.to_human(size) for size in self.val_sizes)}")

        processor_count: int = await get_host_proc_count(self.server_ip)
        processor_count = 9 if processor_count > 9 else processor_count  # TODO max sessions typically 10

        print("Ensuring binary is ready")
        await Server(self.server_ip).kill_all_valkey_instances_on_host()
        self.cached_binary_path = await Server(self.server_ip).ensure_binary_cached(
            source=self.source, specifier=self.specifier
        )
        print(f"Binary ready! Testing with up to {processor_count} instances on server")

        port_gen = port_generator()
        semaphore = asyncio.Semaphore(processor_count)
        result_futures: list[CoroutineType] = []
        for size in self.val_sizes:
            port = next(port_gen)
            result_futures.append(self.test_single_size_overhead(size, port, semaphore))

        efficiency_map: dict[int, float] = {}
        results = []

        for future in asyncio.as_completed(result_futures):
            point_result: dict[str, float] = await future
            results.append(point_result)
            val_size = int(point_result["val_size"])
            per_item_overhead = point_result["per_item_overhead"]
            efficiency_map[val_size] = per_item_overhead
            self.plot(efficiency_map)

        # output result
        results.sort(key=lambda x: x["val_size"])
        score = -1
        if len(results) == 1:
            score = results[0]["per_item_overhead"]
        record_task_result(
            f"mem-{self.test}",
            self.source,
            self.specifier,
            self.commit_hash or "",
            score,
            datetime.datetime.now(),
            results,
        )

        await Server(self.server_ip).kill_all_valkey_instances_on_host()

    async def test_single_size_overhead(self, val_size: int, port: int, semaphore) -> dict[str, float]:
        """Test memory efficiency for a single item size."""
        async with semaphore:
            # print(f"{port} starting server")
            valkey = await Server.with_path(self.server_ip, port, self.cached_binary_path, io_threads=1)
            self.commit_hash = valkey.get_build_hash()

            # print(f"{port} get memory usage before")
            before_memory: dict[str, str] = await valkey.info("memory")

            count = 5 * MILLION
            # print(f"{port} loading {HumanByte.to_human(count * (16 + val_size))} dataset")
            await valkey.run_valkey_command_over_keyspace(count, f"-d {val_size} -t {self.test}")
            if self.has_expire:
                if self.test != "set":
                    logger.error("Expiration is only supported for sets, skipping expiration test.")
                else:
                    await valkey.run_valkey_command_over_keyspace(
                        count, f"EXPIRE key:__rand_int__ {7*24*60*60}"
                    )

            # print(f"{port} get memory usage after")
            after_memory: dict[str, str] = await valkey.info("memory")

            (item_count, expire_count) = await valkey.count_items_expires()
            assert item_count == count
            if self.has_expire:
                assert expire_count == count
            else:
                assert expire_count == 0

            keysize = 16  # "key:__rand_int__"
            before_usage = int(before_memory["used_memory"])
            after_usage = int(after_memory["used_memory"])
            total_usage = after_usage - before_usage
            per_key = float(total_usage) / count
            per_item_overhead = per_key - val_size - keysize

            result = {
                "before_memory": before_memory,
                "after_memory": after_memory,
                "has_expire": self.has_expire,
                "key_size": keysize,
                "val_size": val_size,
                "per_key_size": per_key,
                "per_item_overhead": per_item_overhead,
            }
            return result

    def plot(self, efficiency_map: dict[int, float]) -> None:
        """Plot the memory efficiency results."""
        plt.clear_terminal()
        plt.clear_figure()
        plt.canvas_color("black")
        plt.axes_color("black")
        plt.ticks_color("orange")

        plt.title(self.title)
        plt.frame(False)
        plt.xlabel("Val+Key Size (bytes)")
        plt.ylabel("Overhead (bytes)")

        keysize = 16
        sizes = [valsize + keysize for valsize in self.val_sizes]
        overheads = [efficiency_map.get(size, None) for size in self.val_sizes]

        plt.plot(sizes, overheads, marker="braille", color="orange+")
        plt.xticks(sizes, [HumanByte.to_human(size) for size in sizes])
        plt.ylim(lower=0)

        plt.show()
