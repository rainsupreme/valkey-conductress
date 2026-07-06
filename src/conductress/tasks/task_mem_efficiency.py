"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import asyncio
import datetime
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import plotext as plt

from conductress.config import (
    MEM_TEST_ITEM_COUNT,
    MEM_TEST_KEY_SIZE,
    MEM_TEST_MAX_CONCURRENT,
    ServerInfo,
    get_sweep_engine,
    should_profile_internals,
)
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus
from conductress.heap_profiler import JEMALLOC_PROF_ENV, HeapProfileResult, cleanup_heap_dumps, collect_heap_profile
from conductress.server import Server
from conductress.sweep.populator import populate
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.utility import HumanByte, port_generator, print_pretty_header

logger = logging.getLogger(__name__)


@dataclass
class MemTaskData(BaseTaskData):
    """data class for memory efficiency task"""

    type: str
    val_sizes: list[int]
    has_expire: bool
    sweep_commit: str = ""  # non-empty marks this as a sweep task
    enable_profiling: bool = False  # enable jemalloc heap profiling for per-struct breakdown
    key_size: int = 0  # key size in bytes (SET) or 0 (single-key commands); set by coordinator
    field_size: int = 0  # field name size (HSET only)
    user_data_bytes: int = 0  # per-item user data for overhead calculation

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
            self.make_args,
            self.note,
            self.enable_profiling,
            self.key_size,
            self.field_size,
            self.user_data_bytes,
        )


class MemTaskRunner(BaseTaskRunner):
    """Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

    tests = ["set", "sadd", "zadd", "hset"]

    def __init__(
        self,
        task_name: str,
        server_ip: str,
        source: str,
        specifier: str,
        test: str,
        val_sizes: list[int],
        has_expire: bool,
        make_args: str,
        note: str,
        enable_profiling: bool = False,
        key_size: int = 0,
        field_size: int = 0,
        user_data_bytes: int = 0,
    ):
        super().__init__(task_name)

        self.logger = logging.getLogger(self.__class__.__name__ + "." + test)

        self.title = f"{test} memory efficiency, {source}:{specifier}, has_expire={has_expire}"

        if test not in self.tests:
            raise ValueError(f"Test {test} is not supported. Supported tests: {self.tests}")

        # settings
        self.server_ip = server_ip
        self.source = source
        self.specifier = specifier
        self.test = test
        self.val_sizes = val_sizes
        self.has_expire = has_expire
        self.note = note
        self.make_args = make_args
        self.enable_profiling = enable_profiling
        self.key_size = key_size
        self.field_size = field_size
        self.user_data_bytes = user_data_bytes

        # test data
        self.commit_hash: Optional[str] = None
        self.cached_binary_path: Optional[Path] = None

        # Initialize status
        self.status = BenchmarkStatus(
            steps_total=len(self.val_sizes) + 2, task_type=f"mem-{test}"
        )  # setup + sizes + results

    async def run(self) -> None:
        """Run the memory efficiency test for each value size."""
        print_pretty_header(self.title)
        self.logger.info(f"Val Sizes: {', '.join(HumanByte.to_human(size) for size in self.val_sizes)}")

        # Write initial status
        self.file_protocol.write_status(self.status)

        num_servers: int = await Server(self.server_ip).get_available_cpu_count() // Server.get_num_cpus(1)
        num_servers = MEM_TEST_MAX_CONCURRENT if num_servers > MEM_TEST_MAX_CONCURRENT else num_servers

        self.logger.info("Ensuring binary is ready")
        await Server(self.server_ip).kill_all_valkey_instances_on_host()
        self.cached_binary_path = await Server(self.server_ip).ensure_binary_cached(
            source=self.source, specifier=self.specifier, make_args=self.make_args
        )
        self.logger.info(f"Binary ready! Testing with up to {num_servers} instances on server")

        # Update progress: setup complete
        self.status.state = "running"
        self.status.steps_completed = 1
        self.file_protocol.write_status(self.status)

        port_gen = port_generator()
        semaphore = asyncio.Semaphore(num_servers)
        result_futures: list[Coroutine] = []
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
            from conductress.file_protocol import MetricData

            metric = MetricData(
                metrics={
                    "val_size": float(val_size),
                    "per_item_overhead": per_item_overhead,
                }
            )
            self.file_protocol.append_metric(metric)

            # Update progress for each completed size
            completed_sizes += 1
            self.status.steps_completed = 1 + completed_sizes
            self.file_protocol.write_status(self.status)

        # output result
        results.sort(key=lambda x: x["val_size"])
        score: float = -1.0
        if len(results) == 1:
            score = results[0]["per_item_overhead"]
        # Write results to file protocol (replaces record_task_result)
        detailed_results = {
            "results": results,
        }
        completion_time = datetime.datetime.now()
        results_data = BenchmarkResults(
            method=f"mem-{self.test}",
            source=self.source,
            specifier=self.specifier,
            commit_hash=self.commit_hash or "",
            score=score,
            end_time=completion_time,
            data=detailed_results,
            make_args=self.make_args,
            note=self.note,
        )
        self.file_protocol.write_results(results_data)

        # Update progress: results complete
        self.status.state = "completed"
        self.status.steps_completed = self.status.steps_total
        self.status.end_time = time.time()
        self.file_protocol.write_status(self.status)

        await Server(self.server_ip).kill_all_valkey_instances_on_host()

    async def test_single_size_overhead(self, val_size: int, port: int, semaphore) -> dict[str, Any]:
        """Test memory efficiency for a single item size."""
        async with semaphore:
            if self.cached_binary_path is None:
                raise RuntimeError("cached_binary_path must be set before calling test_single_size_overhead")

            # Set up profiling env if enabled
            env_prefix = ""
            if self.enable_profiling:
                ssh_host = Server(self.server_ip)
                await cleanup_heap_dumps(ssh_host)
                env_prefix = JEMALLOC_PROF_ENV

            valkey = await Server.with_path(
                self.server_ip, port, self.cached_binary_path, io_threads=1, env_prefix=env_prefix
            )
            self.commit_hash = valkey.get_build_hash()

            before_memory: dict[str, str] = await valkey.info("memory")

            count = MEM_TEST_ITEM_COUNT
            from conductress.sweep.memory_coordinator import MemoryWorkload  # noqa: E402 (circular import)

            workload = MemoryWorkload(
                command=self.test,
                key_size=self.key_size,
                value_size=val_size,
                field_size=self.field_size,
                has_expire=self.has_expire,
                label=f"{self.test}-populator",
                item_count=count,
                user_data_bytes=self.user_data_bytes,
            )
            populate(valkey.ip, valkey.port, workload)

            after_memory: dict[str, str] = await valkey.info("memory")

            (key_count, expire_count) = await valkey.count_items_expires()
            if self.test == "set":
                # SET creates one key per item
                item_count = key_count
            else:
                # SADD/ZADD/HSET create a single collection with N members
                if key_count != 1:
                    raise RuntimeError(f"Expected 1 collection key but found {key_count} on port {port}")
                if self.test == "zadd":
                    cardinality_cmd, key_name = "ZCARD", "myzset"
                elif self.test == "hset":
                    cardinality_cmd, key_name = "HLEN", "myhash"
                else:
                    cardinality_cmd, key_name = "SCARD", "myset"
                result_str = await valkey.run_valkey_command(f"{cardinality_cmd} {key_name}")
                if result_str is None:
                    raise RuntimeError(f"{cardinality_cmd} returned None on port {port}")
                item_count = int(result_str)
            if item_count != count:
                raise RuntimeError(f"Expected {count} items but found {item_count} on port {port}")
            if self.has_expire:
                if expire_count != count:
                    raise RuntimeError(f"Expected {count} expires but found {expire_count} on port {port}")
            else:
                if expire_count != 0:
                    raise RuntimeError(f"Expected 0 expires but found {expire_count} on port {port}")

            await valkey.stop()  # required cleanup, release CPU allocations, etc

            # Collect heap profile if profiling is enabled (dump created on shutdown by prof_final).
            # Skipped for engines that opt out of internal profiling (Redis): allocation stacks
            # expose the binary's symbols. Total memory below is still recorded regardless.
            breakdown: Optional[dict[str, float]] = None
            raw_stacks: Optional[list[list]] = None
            if self.enable_profiling and should_profile_internals(get_sweep_engine(self.source)):
                profile_result = await collect_heap_profile(valkey, str(self.cached_binary_path), count)
                if profile_result:
                    breakdown = profile_result.breakdown
                    raw_stacks = profile_result.raw_stacks
                    self.logger.info("Memory breakdown collected: %s", breakdown)
                await cleanup_heap_dumps(valkey)

            # User data per item from workload config
            user_data_per_item = self.user_data_bytes

            before_usage = int(before_memory["used_memory"])
            after_usage = int(after_memory["used_memory"])
            total_usage = after_usage - before_usage
            per_key = float(total_usage) / count
            per_item_overhead = per_key - user_data_per_item

            result = {
                "before_memory": before_memory,
                "after_memory": after_memory,
                "has_expire": self.has_expire,
                "user_data_per_item": user_data_per_item,
                "val_size": val_size,
                "per_key_size": per_key,
                "per_item_overhead": per_item_overhead,
                "breakdown": breakdown,
                "raw_stacks": raw_stacks,
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
