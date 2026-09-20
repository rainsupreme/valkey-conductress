"""cachecannon generator helpers shared by the cachecannon and replica-read tasks.

Pure functions only (TOML generation, JSON result parsing): this module must
not import any task module. Task modules are auto-imported by the task
registry, so a helper living inside one of them creates an import cycle the
moment a second task needs it.
"""

import json

from conductress.config import CACHECANNON_BINARY

# Default path to cachecannon binary on bench hosts. Sourced from config so a
# single CONDUCTRESS_CACHECANNON_BINARY env override reaches every task; kept as
# a module-level name here because tasks and the CLI import it from this module.
DEFAULT_CACHECANNON_BINARY = CACHECANNON_BINARY


def _latency_from_json(command: str, block: dict) -> dict:
    """Convert a cachecannon JSON latency object to the recorded ms schema.

    cachecannon reports latency percentiles as integer microseconds, so no unit
    inference is required.
    """
    required = ("p50_us", "p90_us", "p99_us", "p999_us", "p9999_us", "max_us")
    missing = [field for field in required if field not in block]
    if missing:
        raise ValueError(f"cachecannon JSON latency object missing {missing}: {block!r}")
    return {
        "command": command,
        "p50_ms": block["p50_us"] / 1000.0,
        "p90_ms": block["p90_us"] / 1000.0,
        "p99_ms": block["p99_us"] / 1000.0,
        "p999_ms": block["p999_us"] / 1000.0,
        "p9999_ms": block["p9999_us"] / 1000.0,
        "max_ms": block["max_us"] / 1000.0,
        "count": block.get("count", 0),
    }


def parse_json_results(output: str) -> dict:
    """Parse cachecannon's NDJSON output and return EXACT result values.

    cachecannon emits newline-delimited JSON when ``[admin] format = "json"`` is
    set. Unstructured progress lines (``[precheck]``, ringline diagnostics) are
    interleaved on the same streams and are skipped.

    Exactness is the entire point of this parser. The human ``clean`` formatter
    renders throughput through an abbreviating helper, so 3,104,882 ops/sec
    prints as ``3.10M`` -- three significant figures, a quantization step of
    roughly 0.3-0.5% at multi-million ops/sec. That is at or above the sweep
    target CV, which makes variance unmeasurable and collapses independent
    repetitions onto a single representable value. The JSON ``result`` message
    carries the exact integer instead.

    Returns dict with keys: throughput_rps, error_pct, hit_rate, latency,
    latency_get, latency_set, requests, responses, errors, duration_secs.
    """
    result_obj = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "result":
            result_obj = candidate

    if result_obj is None:
        raise ValueError(
            'No cachecannon JSON result message found. Is [admin] format = "json" set? '
            f"Output tail:\n{output[-1000:]}"
        )

    for field in ("throughput", "err_pct", "hits", "misses", "hit_pct"):
        if field not in result_obj:
            raise ValueError(f"cachecannon JSON result missing {field!r}: {result_obj!r}")

    latency_get = _latency_from_json("GET", result_obj["get"]) if "get" in result_obj else None
    latency_set = _latency_from_json("SET", result_obj["set"]) if "set" in result_obj else None

    # Primary latency is whichever command actually carried the workload. The
    # previous clean-output parser recorded the LAST latency row it matched,
    # which for a GET-only run could be an all-zero SET row.
    primary = latency_get
    if latency_set and (latency_get is None or latency_set["count"] > latency_get["count"]):
        primary = latency_set

    return {
        "throughput_rps": float(result_obj["throughput"]),
        "error_pct": float(result_obj["err_pct"]),
        "hit_rate": {
            "percent": float(result_obj["hit_pct"]),
            "hits": float(result_obj["hits"]),
            "misses": float(result_obj["misses"]),
        },
        "latency": primary,
        "latency_get": latency_get,
        "latency_set": latency_set,
        "requests": result_obj.get("requests"),
        "responses": result_obj.get("responses"),
        "errors": result_obj.get("errors"),
        "duration_secs": result_obj.get("duration_secs"),
    }


def workload_issues_gets(test: str, set_ratio: int) -> bool:
    """Whether the effective workload sends any GET commands.

    Mirrors generate_toml_config's weight mapping: set_ratio > 0 overrides
    'test' with a mixed workload (get_weight = 100 - set_ratio); set_ratio == 0
    means a pure workload per 'test'. The hit-rate prefill guard is only
    meaningful when this returns True -- pure-SET runs have hit rate 0.0 by
    definition (SETs cannot hit)."""
    if set_ratio > 0:
        return set_ratio < 100
    return test == "get"


def generate_toml_config(
    *,
    duration: int,
    warmup: int,
    threads: int,
    cpu_list: str,
    endpoint: str,
    connections: int,
    pipeline_depth: int,
    keyspace_count: int,
    val_size: int,
    test: str = "get",
    set_ratio: int = 0,
    distribution: str = "uniform",
    rate_limit: int = 0,
    prefill: bool = True,
) -> str:
    """Generate a cachecannon TOML configuration file.

    Args:
        duration: Test duration in seconds.
        warmup: Warmup duration in seconds.
        threads: Number of worker threads.
        cpu_list: Comma-separated CPU list (e.g. '8,9,10,11').
        endpoint: Server endpoint (e.g. '127.0.0.1:6379').
        connections: Total connections.
        pipeline_depth: Pipeline depth.
        keyspace_count: Number of keys in the keyspace.
        val_size: Value size in bytes.
        test: Command to bench ('get' or 'set'). Ignored when set_ratio > 0.
        set_ratio: Percentage of SET commands (0-100). When > 0, overrides
            'test' with a mixed GET/SET workload (get = 100 - set_ratio).
        distribution: Key distribution ('uniform' or 'zipf').
        rate_limit: Fixed request rate (req/s, shared across all connections).
            0 (default) = unlimited, closed-loop. Used for fixed-rate drivers
            such as the replica-read task's primary writer.
        prefill: Write every key once before warmup. Turn off for a generator
            that must not seed the keyspace itself (e.g. a reader aimed at a
            replica, which cannot accept writes).

    Returns:
        TOML configuration string.
    """
    if not 0 <= set_ratio <= 100:
        raise ValueError(f"set_ratio must be 0-100, got {set_ratio}")
    if distribution not in ("uniform", "zipf"):
        raise ValueError(f"distribution must be 'uniform' or 'zipf', got '{distribution}'")
    if rate_limit < 0:
        raise ValueError(f"rate_limit must be >= 0, got {rate_limit}")

    # Map test name / set_ratio to cachecannon command weights. ALWAYS write
    # all three weights explicitly: cachecannon applies serde per-field
    # defaults (get=80, set=20, delete=0) to any weight omitted from the
    # TOML, so a config with only 'set = 100' silently runs 80% GET.
    if set_ratio > 0:
        get_weight, set_weight = 100 - set_ratio, set_ratio
    elif test == "set":
        get_weight, set_weight = 0, 100
    else:  # 'get' (default)
        get_weight, set_weight = 100, 0
    commands_section = f"get = {get_weight}\nset = {set_weight}\ndelete = 0"
    workload_section = f"prefill = {'true' if prefill else 'false'}"
    if rate_limit > 0:
        workload_section += f"\nrate_limit = {rate_limit}"

    toml = f"""[general]
duration = "{duration}s"
warmup = "{warmup}s"
threads = {threads}
cpu_list = "{cpu_list}"
io_engine = "uring"

[target]
endpoints = ["{endpoint}"]
protocol = "resp"

[connection]
connections = {connections}
pipeline_depth = {pipeline_depth}

[workload]
{workload_section}

[workload.keyspace]
length = 16
count = {keyspace_count}
distribution = "{distribution}"

[workload.commands]
{commands_section}

[workload.values]
length = {val_size}

[timestamps]
enabled = true
mode = "userspace"

[admin]
format = "json"
"""
    return toml
