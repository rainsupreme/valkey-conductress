"""Mixed GET/SET throughput benchmark using memtier_benchmark.

Drives a configurable ratio of GET and SET commands at maximum throughput,
measuring combined RPS across N repetitions with server restarts between reps.
"""

import datetime
import json
import logging
import re
import time
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import List, Optional

from scipy.stats import t as t_dist

from conductress.config import (
    LATENCY_KEYSPACE,
    PERF_BENCH_CLIENTS,
    PERF_BENCH_KEYSPACE,
    PERF_BENCH_THREADS,
    PROJECT_ROOT,
    ServerInfo,
    get_sweep_engine,
    should_profile_internals,
)
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus, FileProtocol, MetricData
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner

logger = logging.getLogger(__name__)

# memtier defaults
MIXED_THREADS = 8
MIXED_CLIENTS = 50  # 400 total connections (8*50)
MIXED_KEYSPACE = PERF_BENCH_KEYSPACE  # 3M keys — same as perf task


def set_ratio_to_memtier_ratio(set_pct: int) -> str:
    """Convert a SET percentage (0-100) to memtier --ratio=SET:GET string.

    memtier --ratio is SET:GET. E.g. set_pct=20 -> ratio 1:4 (1 SET per 4 GETs).
    Special cases: 0 -> 0:1 (pure GET), 100 -> 1:0 (pure SET).
    """
    if set_pct == 0:
        return "0:1"
    if set_pct == 100:
        return "1:0"
    from math import gcd

    get_pct = 100 - set_pct
    g = gcd(set_pct, get_pct)
    return f"{set_pct // g}:{get_pct // g}"


def parse_memtier_total_rps(output: str) -> Optional[float]:
    """Parse memtier_benchmark stdout for the Totals line ops/sec.

    Expected format (from memtier summary table):
    Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency  ...
    ...
    Totals      1234567.89      ...
    """
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Totals":
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def parse_memtier_interval_lines(output: str) -> List[float]:
    """Parse memtier --print-percentiles or interval output for per-second RPS.

    memtier with --json-out-file writes interval data, but stdout interval lines
    look like:
    [RUN #1  ...] ... <ops/sec> ...
    We'll look for the JSON file output instead (more reliable).
    Falls back to the Totals line if interval parsing fails.
    """
    # memtier interval lines: "[RUN #N NN%, ...]  NNN.NN KB - Gets: NNN, Sets: NNN, Ops: NNN ..."
    rps_values: List[float] = []
    for line in output.splitlines():
        # Look for interval report lines with "Ops:" pattern
        match = re.search(r"Ops:\s+(\d+)", line)
        if match:
            rps_values.append(float(match.group(1)))
    return rps_values


@dataclass
class MixedTaskData(BaseTaskData):
    """Task data for mixed GET/SET throughput measurement."""

    set_ratio: int  # percentage of SET commands (0-100)
    val_size: int
    io_threads: int
    pipelining: int
    duration: int
    warmup: int = 5
    perf_stat_enabled: bool = False
    key_size: int = 0  # 0 = standard 16B keys
    repetitions: int = 3
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""
    server_args: str = ""  # extra raw args appended to the server command line (override defaults)

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "MixedTaskData"
        if not (0 <= self.set_ratio <= 100):
            raise ValueError(f"set_ratio must be 0-100, got {self.set_ratio}")

    def short_description(self) -> str:
        from conductress.utility import HumanByte, HumanTime

        return (
            f"mixed {self.set_ratio}%SET/{100-self.set_ratio}%GET "
            f"v={HumanByte.to_human(self.val_size)} "
            f"io={self.io_threads} P={self.pipelining} "
            f"{HumanTime.to_human(self.duration)}"
            f"{' perf-stat' if self.perf_stat_enabled else ''}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "MixedTaskRunner":
        return MixedTaskRunner(
            task_name=self.task_id,
            server_infos=server_infos,
            source=self.source,
            specifier=self.specifier,
            make_args=self.make_args,
            io_threads=self.io_threads,
            val_size=self.val_size,
            pipelining=self.pipelining,
            set_ratio=self.set_ratio,
            warmup=self.warmup,
            duration=self.duration,
            repetitions=self.repetitions,
            perf_stat_enabled=self.perf_stat_enabled,
            key_size=self.key_size,
            note=self.note,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
            server_args=self.server_args,
        )


class MixedTaskRunner(BaseTaskRunner):
    """Runs memtier_benchmark with a configurable GET/SET ratio at max throughput."""

    def __init__(
        self,
        task_name: str,
        server_infos: list[ServerInfo],
        source: str,
        specifier: str,
        make_args: str,
        io_threads: int,
        val_size: int,
        pipelining: int,
        set_ratio: int,
        warmup: int,
        duration: int,
        repetitions: int,
        perf_stat_enabled: bool = False,
        key_size: int = 0,
        note: str = "",
        server_cpu_override: str = "",
        benchmark_cpu_override: str = "",
        server_args: str = "",
    ):
        super().__init__(task_name)
        self.server_infos = server_infos
        self.source = source
        self.specifier = specifier
        self.make_args = make_args
        self.io_threads = io_threads
        self.val_size = val_size
        self.pipelining = pipelining
        self.set_ratio = set_ratio
        self.warmup = warmup
        self.duration = duration
        self.repetitions = repetitions
        self.perf_stat_enabled = perf_stat_enabled
        self.key_size = key_size
        self.note = note
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override
        self.server_args = server_args

        self.commit_hash = ""
        self._profile_internals = should_profile_internals(get_sweep_engine(source))

        self.status = BenchmarkStatus(
            steps_total=repetitions * 2,
            task_type=f"perf-mixed-{set_ratio}pct-set",
        )

    async def run(self) -> None:
        """Execute N repetitions of mixed GET/SET benchmark with memtier."""
        logger.info(
            "Mixed benchmark: %d%%SET/%d%%GET, v=%d, io=%d, P=%d, %ds x %d reps",
            self.set_ratio,
            100 - self.set_ratio,
            self.val_size,
            self.io_threads,
            self.pipelining,
            self.duration,
            self.repetitions,
        )
        self.file_protocol.write_status(self.status)

        replication_group = ReplicationGroup(
            self.server_infos,
            self.source,
            self.specifier,
            self.io_threads,
            self.make_args,
            server_cpu_override=self.server_cpu_override,
            server_args=self.server_args,
        )

        per_run_rps: List[float] = []
        perf_counters: Optional[dict] = None

        try:
            for rep in range(self.repetitions):
                # Between-rep housekeeping
                if rep > 0:
                    await replication_group.stop_all_servers()
                    server = replication_group.primary or Server(self.server_infos[0].ip)
                    await server.run_host_command("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False)

                # Start server
                await replication_group.kill_all_valkey_instances()
                await replication_group.start()
                if not replication_group.primary:
                    raise RuntimeError("Server failed to start")
                server = replication_group.primary
                self.commit_hash = server.get_build_hash() or ""

                # Prefill keyspace so GETs hit (same approach as perf task)
                total_conns = MIXED_THREADS * MIXED_CLIENTS
                prefill_cmd = (
                    f"~/conductress/memtier_benchmark "
                    f"--server {server.ip} --port {server.port} --protocol redis "
                    f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
                    f"--ratio 1:0 --key-pattern P:P "
                    f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                    f"--data-size {self.val_size} "
                    f"--requests {MIXED_KEYSPACE // total_conns} "
                    f"--hide-histogram"
                )
                await server.run_host_command(prefill_cmd)
                self.status.steps_completed = rep * 2 + 1
                self.file_protocol.write_status(self.status)

                # Perf stat: start before measurement
                if self.perf_stat_enabled:
                    await server.perf_stat_start()

                # Run measurement phase
                ratio_str = set_ratio_to_memtier_ratio(self.set_ratio)
                measure_cmd = (
                    f"~/conductress/memtier_benchmark "
                    f"--server {server.ip} --port {server.port} --protocol redis "
                    f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
                    f"--ratio {ratio_str} --key-pattern R:R "
                    f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                    f"--data-size {self.val_size} "
                    f"--pipeline {self.pipelining} "
                    f"--test-time {self.duration} "
                    f"--hide-histogram"
                )
                stdout, _ = await server.run_host_command(measure_cmd)

                # Perf stat: stop and collect
                if self.perf_stat_enabled:
                    await server.perf_stat_stop()

                # Parse total RPS from memtier output
                total_rps = parse_memtier_total_rps(stdout)
                if total_rps is None:
                    raise RuntimeError(f"Failed to parse memtier output for rep {rep + 1}")

                per_run_rps.append(total_rps)
                logger.info("Rep %d/%d: %.1f ops/sec", rep + 1, self.repetitions, total_rps)

                # Write per-rep metric
                metric = MetricData(metrics={"rps": total_rps}, rep=rep + 1)
                self.file_protocol.append_metric(metric)

                self.status.steps_completed = rep * 2 + 2
                self.file_protocol.write_status(self.status)

                # Collect perf stat report
                if self.perf_stat_enabled:
                    server.perf_stat_wait()
                    result_dir = self.file_protocol.get_result_dir()
                    rep_counters = await server.perf_stat_report(result_dir)
                    if rep_counters:
                        if perf_counters is None:
                            perf_counters = rep_counters
                        else:
                            for bucket, events in rep_counters.items():
                                acc = perf_counters.setdefault(bucket, {})
                                for k, v in events.items():
                                    acc[k] = acc.get(k, 0) + v

        finally:
            await replication_group.stop_all_servers()

        if not per_run_rps:
            raise RuntimeError("No successful repetitions")

        # Compute aggregated stats
        mean_rps = mean(per_run_rps)
        ci_95 = 0.0
        if len(per_run_rps) >= 2:
            ci_95 = t_dist.ppf(0.975, len(per_run_rps) - 1) * (stdev(per_run_rps) / sqrt(len(per_run_rps)))

        # Record results
        detailed_data = {
            "set_ratio": self.set_ratio,
            "memtier_ratio": set_ratio_to_memtier_ratio(self.set_ratio),
            "duration": self.duration,
            "warmup": self.warmup,
            "io_threads": self.io_threads,
            "pipeline": self.pipelining,
            "size": self.val_size,
            "key_size": self.key_size,
            "keyspace": MIXED_KEYSPACE,
            "threads": MIXED_THREADS,
            "clients": MIXED_CLIENTS,
            "repetitions": self.repetitions,
            "per_run_rps": per_run_rps,
            "mean_rps": mean_rps,
            "ci_95": ci_95,
        }
        if perf_counters:
            detailed_data["perf_counters"] = perf_counters
            detailed_data["perf_duration_seconds"] = float(self.duration)
            detailed_data["perf_rep_count"] = len(per_run_rps)

        results = BenchmarkResults(
            method=f"perf-mixed-{self.set_ratio}set",
            source=self.source,
            specifier=self.specifier,
            commit_hash=self.commit_hash,
            score=mean_rps,
            end_time=datetime.datetime.now(),
            data=detailed_data,
            make_args=self.make_args,
            note=self.note,
        )
        self.file_protocol.write_results(results)

        self.status.state = "completed"
        self.status.end_time = time.time()
        self.status.steps_completed = self.repetitions * 2
        self.file_protocol.write_status(self.status)
