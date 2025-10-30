"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import CoroutineType
from typing import Optional

import plotext as plt

from src.config import (
    MEM_TEST_EXPIRE_SECONDS,
    MEM_TEST_ITEM_COUNT,
    MEM_TEST_KEY_SIZE,
    MEM_TEST_MAX_CONCURRENT,
    ServerInfo,
)
from src.file_protocol import BenchmarkResults, BenchmarkStatus
from src.server import Server
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import HumanByte, port_generator, print_pretty_header

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

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "MemTaskRunner":
        """Return the task runner for this task."""
        return MemTaskRunner(
            self.task_id,
            server_infos[0].ip,
            self.source,
            self.specifier,
            self.type,
            self.val_sizes,
            self.has_expire,
            note=self.note,
        )


class MemTaskRunner(BaseTaskRunner):
    """Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

    tests = ["set", "sadd", "zadd"]  # TODO hset? others?

    def __init__(
        self,
        task_name: str,
        server_ip: str,
        source: str,
        specifier: str,
        test: str,
        val_sizes: list[int],
        has_expire: bool,
        note: str,
    ):
        super().__init__(task_name)

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
        self.note = note

        # test data
        self.commit_hash: Optional[str] = None
        self.cached_binary_path: Optional[Path] = None

        # Initialize status
        self.status = BenchmarkStatus(steps_total=len(self.val_sizes) + 2, task_type=f"mem-{test}")  # setup + sizes + results

    async def run(self) -> None:
        """Run the memory efficiency test for each value size."""
        print_pretty_header(self.title)
        print(f"Val Sizes: {', '.join(HumanByte.to_human(size) for size in self.val_sizes)}")

        # Write initial status
        self.file_protocol.write_status(self.status)

        avail_cpus: int = await Server(self.server_ip).get_available_cpu_count()
        avail_cpus = MEM_TEST_MAX_CONCURRENT if avail_cpus > MEM_TEST_MAX_CONCURRENT else avail_cpus

        print("Ensuring binary is ready")
        await Server(self.server_ip).kill_all_valkey_instances_on_host()
        self.cached_binary_path = await Server(self.server_ip).ensure_binary_cached(
            source=self.source, specifier=self.specifier
        )
        print(f"Binary ready! Testing with up to {avail_cpus} instances on server")

        # Update progress: setup complete
        self.status.state = "running"
        self.status.steps_completed = 1
        self.file_protocol.write_status(self.status)

        port_gen = port_generator()
        semaphore = asyncio.Semaphore(avail_cpus)
        result_futures: list[CoroutineType] = []
        for size in self.val_sizes:
            port = next(port_gen)
            result_futures.append(self.test_single_size_overhead(size, port, semaphore))

        efficiency_map: dict[int, float] = {}
        results = []

        completed_sizes = 0
        for future in asyncio.as_completed(result_futures):
            point_result: dict[str, float] = await future
            results.append(point_result)
            val_size = int(point_result["val_size"])
            per_item_overhead = point_result["per_item_overhead"]
            efficiency_map[val_size] = per_item_overhead
            self.plot(efficiency_map)

            # Log metric data point
            from src.file_protocol import MetricData

            metric = MetricData(metrics={"val_size": float(val_size), "per_item_overhead": per_item_overhead})
            self.file_protocol.append_metric(metric)

            # Update progress for each completed size
            completed_sizes += 1
            self.status.steps_completed = 1 + completed_sizes
            self.file_protocol.write_status(self.status)

        # output result
        results.sort(key=lambda x: x["val_size"])
        score = -1
        if len(results) == 1:
            score = results[0]["per_item_overhead"]
        # Write results to file protocol (replaces record_task_result)
        completion_time = datetime.datetime.now()
        results_data = BenchmarkResults(
            method=f"mem-{self.test}",
            source=self.source,
            specifier=self.specifier,
            commit_hash=self.commit_hash or "",
            score=score,
            end_time=completion_time,
            data=results,
            note=self.note,
        )
        self.file_protocol.write_results(results_data)

        # Update progress: results complete
        self.status.state = "completed"
        self.status.steps_completed = self.status.steps_total
        self.status.end_time = time.time()
        self.file_protocol.write_status(self.status)

        await Server(self.server_ip).kill_all_valkey_instances_on_host()

    async def test_single_size_overhead(self, val_size: int, port: int, semaphore) -> dict[str, float]:
        """Test memory efficiency for a single item size."""
        async with semaphore:
            # print(f"{port} starting server")
            assert (
                self.cached_binary_path is not None
            ), "cached_binary_path must be set before calling test_single_size_overhead"
            valkey = await Server.with_path(self.server_ip, port, self.cached_binary_path, io_threads=1)
            self.commit_hash = valkey.get_build_hash()

            # print(f"{port} get memory usage before")
            before_memory: dict[str, str] = await valkey.info("memory")

            count = MEM_TEST_ITEM_COUNT
            # print(f"{port} loading {HumanByte.to_human(count * (16 + val_size))} dataset")
            await valkey.run_valkey_command_over_keyspace(count, f"-d {val_size} -t {self.test}")
            if self.has_expire:
                if self.test != "set":
                    logger.error("Expiration is only supported for sets, skipping expiration test.")
                else:
                    await valkey.run_valkey_command_over_keyspace(
                        count, f"EXPIRE key:__rand_int__ {MEM_TEST_EXPIRE_SECONDS}"
                    )

            # print(f"{port} get memory usage after")
            after_memory: dict[str, str] = await valkey.info("memory")

            (item_count, expire_count) = await valkey.count_items_expires()
            assert item_count == count
            if self.has_expire:
                assert expire_count == count
            else:
                assert expire_count == 0

            keysize = MEM_TEST_KEY_SIZE
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

        keysize = MEM_TEST_KEY_SIZE
        sizes = [valsize + keysize for valsize in self.val_sizes]
        overheads = [efficiency_map.get(valsize, None) for valsize in self.val_sizes]

        plt.plot(sizes, overheads, marker="braille", color="orange+")
        plt.xticks(sizes, [HumanByte.to_human(size) for size in sizes])
        plt.ylim(lower=0)

        plt.show()
