"""Pathological-workload scenario benchmark task.

Measures how badly a named pathology disturbs concurrent GET throughput.
Architecture: steady background GET load via memtier (same prefill/keyspace as
task_mixed), PLUS an overlay driver that exercises the pathological pattern.

Reports:
  - Sustained baseline throughput (per-interval RPS timeseries)
  - Scenario-specific metrics (overlay ops/s, dip depth, recovery time)
"""

import asyncio
import datetime
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import t as t_dist

from conductress.config import PERF_BENCH_KEYSPACE, ServerInfo, get_sweep_engine, should_profile_internals
from conductress.cpu_allocator import AllocationTag
from conductress.file_protocol import BenchmarkResults, BenchmarkStatus, FileProtocol, MetricData
from conductress.replication_group import ReplicationGroup
from conductress.server import Server
from conductress.task_queue import BaseTaskData, BaseTaskRunner
from conductress.tasks.task_mixed import MIXED_CLIENTS, MIXED_KEYSPACE, MIXED_THREADS, parse_memtier_total_rps

logger = logging.getLogger(__name__)

# Valid scenario names
SCENARIO_CHOICES = ("eval-storm", "scan-churn", "multi-exec", "flushall-spike", "expiry-heavy")

# Overlay thread/connection counts (kept small to avoid dominating the load)
OVERLAY_THREADS = 2
OVERLAY_CLIENTS = 4  # per-thread


def validate_scenario(name: str) -> bool:
    """Check whether a scenario name is recognized."""
    return name in SCENARIO_CHOICES


def encode_resp_array(*args: str) -> bytes:
    """Encode a single RESP command as an array of bulk strings.

    Example: encode_resp_array("SET", "key:1", "val") ->
        b"*3\r\n$3\r\nSET\r\n$5\r\nkey:1\r\n$3\r\nval\r\n"
    """
    parts = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        encoded = arg.encode()
        parts.append(f"${len(encoded)}\r\n".encode())
        parts.append(encoded)
        parts.append(b"\r\n")
    return b"".join(parts)


def generate_multi_exec_resp_payload(num_transactions: int, keyspace: int, val_size: int) -> bytes:
    """Generate a RESP payload of real MULTI/GET/SET/EXEC transactions.

    Each transaction is:
        MULTI
        GET memtier-<rand>
        SET memtier-<rand> <value>
        EXEC

    Keys use memtier's default prefix format (memtier-<N>) to hit the same
    prefilled keyspace as the background GET load.

    Returns raw RESP bytes ready for `valkey-cli --pipe`.
    """
    value = "x" * val_size
    payload_parts: List[bytes] = []
    for _ in range(num_transactions):
        key = f"memtier-{random.randint(1, keyspace)}"
        payload_parts.append(encode_resp_array("MULTI"))
        payload_parts.append(encode_resp_array("GET", key))
        payload_parts.append(encode_resp_array("SET", key, value))
        payload_parts.append(encode_resp_array("EXEC"))
    return b"".join(payload_parts)


def parse_memtier_json_intervals(json_path: str, json_content: str) -> List[float]:
    """Parse memtier --json-out-file for per-second ops/sec.

    The JSON structure has ALL STATS -> <interval> -> Ops/sec.
    Returns per-second RPS values for timeseries analysis.
    """
    try:
        data = json.loads(json_content)
    except (json.JSONDecodeError, ValueError):
        return []

    rps_values: List[float] = []
    # memtier JSON: {"ALL STATS": {"Totals": {"Ops/sec": ...}, ...}}
    # With --print-percentiles: each second has its own entry
    all_stats = data.get("ALL STATS", {})
    # Check for per-second interval data
    for key in sorted(all_stats.keys()):
        if key.startswith("Second ") or key.replace(".", "", 1).isdigit():
            interval = all_stats[key]
            if isinstance(interval, dict) and "Ops/sec" in interval:
                rps_values.append(float(interval["Ops/sec"]))
    return rps_values


def compute_dip_metrics(baseline_rps: float, interval_rps: List[float]) -> Dict[str, Any]:
    """Compute throughput dip depth and duration from interval timeseries.

    Returns:
        dip_depth_pct: maximum instantaneous drop from baseline (0-100%)
        dip_duration_seconds: number of seconds where RPS < 80% of baseline
        min_rps: lowest observed interval RPS
        recovery_seconds: time from dip start to return to 90% of baseline
    """
    if not interval_rps or baseline_rps <= 0:
        return {"dip_depth_pct": 0.0, "dip_duration_seconds": 0, "min_rps": 0.0, "recovery_seconds": 0}

    min_rps = min(interval_rps)
    dip_depth_pct = max(0.0, (1.0 - min_rps / baseline_rps) * 100.0)

    # Duration where RPS < 80% of baseline
    threshold_80 = baseline_rps * 0.80
    dip_seconds = sum(1 for r in interval_rps if r < threshold_80)

    # Recovery: find first dip below 80%, then count until back above 90%
    threshold_90 = baseline_rps * 0.90
    recovery = 0
    in_dip = False
    for r in interval_rps:
        if r < threshold_80:
            in_dip = True
        if in_dip:
            recovery += 1
            if r >= threshold_90:
                break
    if not in_dip:
        recovery = 0

    return {
        "dip_depth_pct": round(dip_depth_pct, 2),
        "dip_duration_seconds": dip_seconds,
        "min_rps": round(min_rps, 1),
        "recovery_seconds": recovery,
    }


def build_overlay_command(
    scenario: str,
    server_ip: str,
    port: int,
    duration: int,
    keyspace: int,
    val_size: int,
) -> str:
    """Build the overlay command string for a given scenario.

    Returns a shell command that drives the pathological workload.
    Uses valkey-benchmark for repeatable load generation with arbitrary commands.
    Fallback to valkey-cli loops only where multi-step transactions require it.
    """
    bench = "~/conductress/valkey-benchmark"
    cli = "~/conductress/valkey-cli"
    conns = OVERLAY_THREADS * OVERLAY_CLIENTS
    # Total requests: enough to fill the duration at moderate rate
    # Use -t for time-based where available, otherwise estimate
    n_requests = duration * 50000  # 50K ops/sec target per overlay

    if scenario == "eval-storm":
        # Small read+write Lua script at max rate on a few connections.
        # EVAL "redis.call('GET',KEYS[1]) redis.call('SET',KEYS[1],ARGV[1]) return 1" 1 key val
        lua_script = "redis.call('GET',KEYS[1]) redis.call('SET',KEYS[1],ARGV[1]) return 1"
        return (
            f"{bench} -h {server_ip} -p {port} "
            f"-c {conns} -n {n_requests} --threads {OVERLAY_THREADS} "
            f'EVAL "{lua_script}" 1 __rand_int__ value_payload -r {keyspace}'
        )

    elif scenario == "scan-churn":
        # Continuous SCAN cursor iteration from few connections.
        # valkey-benchmark doesn't support stateful SCAN (needs cursor chaining),
        # so we use a bash loop with valkey-cli.
        return (
            f"bash -c '"
            f"end=$((SECONDS + {duration})); "
            f"while [ $SECONDS -lt $end ]; do "
            f'{cli} -h {server_ip} -p {port} --scan --pattern "*" > /dev/null 2>&1; '
            f"done'"
        )

    elif scenario == "multi-exec":
        # Real MULTI/GET/SET/EXEC transactions via valkey-cli --pipe.
        # Pre-generate a RESP payload file of ~1000 transactions, then loop-replay
        # it until the overlay is terminated. This exercises true transaction
        # serialization (Tier-3 exclusive-mode pathology).
        payload_file = "/tmp/multi_exec_payload.resp"
        return (
            f"bash -c '"
            f"end=$((SECONDS + {duration})); "
            f"while [ $SECONDS -lt $end ]; do "
            f"{cli} -h {server_ip} -p {port} --pipe < {payload_file} > /dev/null 2>&1; "
            f"done; "
            f"rm -f {payload_file}'"
        )

    elif scenario == "flushall-spike":
        # Single FLUSHALL (ASYNC if supported) mid-measurement, then re-prefill.
        # We'll schedule FLUSHALL at ~40% through duration, then re-prefill.
        flush_delay = max(2, duration * 2 // 5)
        prefill_n = keyspace // (MIXED_THREADS * MIXED_CLIENTS)
        memtier = "~/conductress/memtier_benchmark"
        return (
            f"bash -c '"
            f"sleep {flush_delay}; "
            f"{cli} -h {server_ip} -p {port} FLUSHALL ASYNC; "
            f"sleep 1; "
            f"{memtier} --server {server_ip} --port {port} --protocol redis "
            f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
            f"--ratio 1:0 --key-pattern P:P "
            f"--key-minimum 1 --key-maximum {keyspace} "
            f"--data-size {val_size} "
            f"--requests {prefill_n} --hide-histogram > /dev/null 2>&1'"
        )

    elif scenario == "expiry-heavy":
        # Overlay SETs with short random TTLs so expiry churns during reads.
        # valkey-benchmark: SET with EX via custom command.
        # TTL 1-5 seconds for aggressive expiry pressure.
        return (
            f"{bench} -h {server_ip} -p {port} "
            f"-c {conns} -n {n_requests} --threads {OVERLAY_THREADS} "
            f"-r {keyspace} SET __rand_int__ value_payload EX 3"
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class ScenarioTaskData(BaseTaskData):
    """Task data for pathological-workload scenario measurement."""

    scenario: str
    val_size: int
    io_threads: int
    pipelining: int
    duration: int
    warmup: int = 5
    perf_stat_enabled: bool = False
    repetitions: int = 3
    server_cpu_override: str = ""
    benchmark_cpu_override: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.task_type = "ScenarioTaskData"
        if not validate_scenario(self.scenario):
            raise ValueError(f"Unknown scenario '{self.scenario}'. Valid: {', '.join(SCENARIO_CHOICES)}")

    def short_description(self) -> str:
        from conductress.utility import HumanByte, HumanTime

        return (
            f"scenario:{self.scenario} "
            f"v={HumanByte.to_human(self.val_size)} "
            f"io={self.io_threads} P={self.pipelining} "
            f"{HumanTime.to_human(self.duration)}"
            f"{' perf-stat' if self.perf_stat_enabled else ''}"
        )

    def prepare_task_runner(self, server_infos: list[ServerInfo]) -> "ScenarioTaskRunner":
        return ScenarioTaskRunner(
            task_name=self.task_id,
            server_infos=server_infos,
            source=self.source,
            specifier=self.specifier,
            make_args=self.make_args,
            io_threads=self.io_threads,
            val_size=self.val_size,
            pipelining=self.pipelining,
            scenario=self.scenario,
            warmup=self.warmup,
            duration=self.duration,
            repetitions=self.repetitions,
            perf_stat_enabled=self.perf_stat_enabled,
            note=self.note,
            server_cpu_override=self.server_cpu_override,
            benchmark_cpu_override=self.benchmark_cpu_override,
        )


class ScenarioTaskRunner(BaseTaskRunner):
    """Runs a pathological-workload scenario: background GET + overlay driver."""

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
        scenario: str,
        warmup: int,
        duration: int,
        repetitions: int,
        perf_stat_enabled: bool = False,
        note: str = "",
        server_cpu_override: str = "",
        benchmark_cpu_override: str = "",
    ):
        super().__init__(task_name)
        self.server_infos = server_infos
        self.source = source
        self.specifier = specifier
        self.make_args = make_args
        self.io_threads = io_threads
        self.val_size = val_size
        self.pipelining = pipelining
        self.scenario = scenario
        self.warmup = warmup
        self.duration = duration
        self.repetitions = repetitions
        self.perf_stat_enabled = perf_stat_enabled
        self.note = note
        self.server_cpu_override = server_cpu_override
        self.benchmark_cpu_override = benchmark_cpu_override

        self.commit_hash = ""
        self._profile_internals = should_profile_internals(get_sweep_engine(source))

        self.status = BenchmarkStatus(
            steps_total=repetitions * 3,  # prefill + baseline + scenario per rep
            task_type=f"scenario-{scenario}",
        )

    async def _run_overlay(self, server: Server, overlay_cmd: str) -> Optional[str]:
        """Start overlay process and return its PID for cleanup.

        Returns the remote PID as a string so we can kill it on failure.
        """
        # Run overlay in background, capture PID
        bg_cmd = f"nohup {overlay_cmd} > /tmp/overlay_out.txt 2>&1 & echo $!"
        stdout, _ = await server.run_host_command(bg_cmd, check=False)
        pid = stdout.strip().split("\n")[-1].strip()
        if pid.isdigit():
            return pid
        return None

    async def _kill_overlay(self, server: Server, pid: Optional[str]) -> Optional[str]:
        """Kill overlay process and return its stdout."""
        overlay_output = None
        if pid:
            # Read output before killing
            cat_stdout, _ = await server.run_host_command("cat /tmp/overlay_out.txt 2>/dev/null || true", check=False)
            overlay_output = cat_stdout
            # Kill the process tree
            await server.run_host_command(
                f"kill -TERM {pid} 2>/dev/null; sleep 0.5; kill -9 {pid} 2>/dev/null", check=False
            )
            # Also kill any child processes (bash loops spawn children)
            await server.run_host_command(
                f"pkill -TERM -P {pid} 2>/dev/null; sleep 0.2; pkill -9 -P {pid} 2>/dev/null", check=False
            )
        return overlay_output

    async def _write_multi_exec_payload(self, server: Server) -> None:
        """Generate and write MULTI/EXEC RESP payload to /tmp on the remote host.

        Writes ~1000 transactions worth of RESP-encoded MULTI/GET/SET/EXEC
        sequences to /tmp/multi_exec_payload.resp for valkey-cli --pipe replay.
        """
        payload = generate_multi_exec_resp_payload(
            num_transactions=1000,
            keyspace=MIXED_KEYSPACE,
            val_size=self.val_size,
        )
        # Write payload via base64 to avoid shell quoting issues with binary RESP
        import base64

        encoded = base64.b64encode(payload).decode()
        # Split into chunks to avoid argument-too-long (payload ~200KB base64)
        chunk_size = 65536
        chunks = [encoded[i : i + chunk_size] for i in range(0, len(encoded), chunk_size)]
        # Write first chunk (truncate), append rest
        await server.run_host_command(f"echo -n '{chunks[0]}' > /tmp/multi_exec_payload.b64", check=True)
        for chunk in chunks[1:]:
            await server.run_host_command(f"echo -n '{chunk}' >> /tmp/multi_exec_payload.b64", check=True)
        await server.run_host_command(
            "base64 -d /tmp/multi_exec_payload.b64 > /tmp/multi_exec_payload.resp && rm -f /tmp/multi_exec_payload.b64",
            check=True,
        )

    async def run(self) -> None:
        """Execute N repetitions of scenario benchmark."""
        logger.info(
            "Scenario benchmark: %s, v=%d, io=%d, P=%d, %ds x %d reps",
            self.scenario,
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
        )

        per_run_rps: List[float] = []
        per_run_scenario_metrics: List[Dict[str, Any]] = []
        perf_counters: Optional[dict] = None

        try:
            for rep in range(self.repetitions):
                overlay_pid: Optional[str] = None

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

                # Prefill keyspace
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
                self.status.steps_completed = rep * 3 + 1
                self.file_protocol.write_status(self.status)

                try:
                    # Perf stat: start before measurement
                    if self.perf_stat_enabled:
                        await server.perf_stat_start()

                    # Start overlay driver
                    if self.scenario == "multi-exec":
                        await self._write_multi_exec_payload(server)
                    overlay_cmd = build_overlay_command(
                        scenario=self.scenario,
                        server_ip=server.ip,
                        port=server.port,
                        duration=self.duration,
                        keyspace=MIXED_KEYSPACE,
                        val_size=self.val_size,
                    )
                    overlay_pid = await self._run_overlay(server, overlay_cmd)

                    # Brief delay to let overlay establish connections
                    await asyncio.sleep(1)

                    # Run background GET load measurement (the baseline under pathology)
                    json_out = f"/tmp/memtier_scenario_rep{rep}.json"
                    measure_cmd = (
                        f"~/conductress/memtier_benchmark "
                        f"--server {server.ip} --port {server.port} --protocol redis "
                        f"--threads {MIXED_THREADS} --clients {MIXED_CLIENTS} "
                        f"--ratio 0:1 --key-pattern R:R "
                        f"--key-minimum 1 --key-maximum {MIXED_KEYSPACE} "
                        f"--data-size {self.val_size} "
                        f"--pipeline {self.pipelining} "
                        f"--test-time {self.duration} "
                        f"--json-out-file {json_out} "
                        f"--hide-histogram"
                    )
                    stdout, _ = await server.run_host_command(measure_cmd)

                    # Collect overlay output
                    overlay_output = await self._kill_overlay(server, overlay_pid)
                    overlay_pid = None  # cleared

                    # Perf stat: stop and collect
                    if self.perf_stat_enabled:
                        await server.perf_stat_stop()

                    # Parse baseline GET RPS
                    total_rps = parse_memtier_total_rps(stdout)
                    if total_rps is None:
                        raise RuntimeError(f"Failed to parse memtier output for rep {rep + 1}")

                    # Try to get interval RPS from JSON output
                    interval_rps: List[float] = []
                    try:
                        json_stdout, _ = await server.run_host_command(f"cat {json_out}", check=False)
                        if json_stdout.strip():
                            interval_rps = parse_memtier_json_intervals(json_out, json_stdout)
                    except Exception:
                        pass  # interval data is best-effort

                    per_run_rps.append(total_rps)
                    logger.info(
                        "Rep %d/%d: %.1f baseline ops/sec under %s", rep + 1, self.repetitions, total_rps, self.scenario
                    )

                    # Compute scenario-specific metrics
                    scenario_metrics: Dict[str, Any] = {"scenario": self.scenario}

                    # Parse overlay ops/s from valkey-benchmark output (if available)
                    if overlay_output:
                        overlay_rps = self._parse_overlay_rps(overlay_output)
                        if overlay_rps is not None:
                            scenario_metrics["overlay_ops_per_sec"] = round(overlay_rps, 1)

                    # Dip metrics from interval timeseries
                    if interval_rps:
                        # Use early intervals as "undisturbed" baseline reference
                        # (first 2 seconds before overlay fully ramps)
                        ref_rps = total_rps  # use aggregate as reference
                        dip = compute_dip_metrics(ref_rps, interval_rps)
                        scenario_metrics.update(dip)
                        scenario_metrics["interval_rps"] = [round(r, 1) for r in interval_rps]

                    per_run_scenario_metrics.append(scenario_metrics)

                    # Write per-rep metric
                    metric = MetricData(
                        metrics={"rps": total_rps, "scenario_metrics": scenario_metrics},
                        rep=rep + 1,
                    )
                    self.file_protocol.append_metric(metric)

                    self.status.steps_completed = rep * 3 + 3
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
                    # Ensure overlay is always killed
                    if overlay_pid:
                        await self._kill_overlay(server, overlay_pid)
                    # Clean up multi-exec payload file if present
                    if self.scenario == "multi-exec":
                        await server.run_host_command("rm -f /tmp/multi_exec_payload.resp", check=False)

        finally:
            await replication_group.stop_all_servers()

        if not per_run_rps:
            raise RuntimeError("No successful repetitions")

        # Compute aggregated stats
        mean_rps = mean(per_run_rps)
        ci_95 = 0.0
        if len(per_run_rps) >= 2:
            ci_95 = t_dist.ppf(0.975, len(per_run_rps) - 1) * (stdev(per_run_rps) / sqrt(len(per_run_rps)))

        # Aggregate scenario metrics across reps
        agg_scenario = {"scenario": self.scenario, "per_rep": per_run_scenario_metrics}
        if per_run_scenario_metrics:
            overlay_rates = [
                m.get("overlay_ops_per_sec", 0) for m in per_run_scenario_metrics if "overlay_ops_per_sec" in m
            ]
            if overlay_rates:
                agg_scenario["mean_overlay_ops_per_sec"] = round(mean(overlay_rates), 1)
            dip_depths = [m.get("dip_depth_pct", 0) for m in per_run_scenario_metrics if "dip_depth_pct" in m]
            if dip_depths:
                agg_scenario["mean_dip_depth_pct"] = round(mean(dip_depths), 2)

        # Record results
        detailed_data = {
            "scenario": self.scenario,
            "duration": self.duration,
            "warmup": self.warmup,
            "io_threads": self.io_threads,
            "pipeline": self.pipelining,
            "size": self.val_size,
            "keyspace": MIXED_KEYSPACE,
            "threads": MIXED_THREADS,
            "clients": MIXED_CLIENTS,
            "repetitions": self.repetitions,
            "per_run_rps": per_run_rps,
            "mean_rps": mean_rps,
            "ci_95": ci_95,
            "scenario_metrics": agg_scenario,
        }
        if perf_counters:
            detailed_data["perf_counters"] = perf_counters
            detailed_data["perf_duration_seconds"] = float(self.duration)
            detailed_data["perf_rep_count"] = len(per_run_rps)

        results = BenchmarkResults(
            method=f"scenario-{self.scenario}",
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
        self.status.steps_completed = self.repetitions * 3
        self.file_protocol.write_status(self.status)

    def _parse_overlay_rps(self, output: str) -> Optional[float]:
        """Parse valkey-benchmark output for total requests/sec.

        valkey-benchmark summary line format:
          <N> requests completed in <T> seconds
        or the throughput summary line.
        """
        # Look for "requests per second" line
        for line in output.splitlines():
            # Pattern: "NNNN.NN requests per second"
            match = re.search(r"([\d.]+)\s+requests per second", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
