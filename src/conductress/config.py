"""Configuration for the Conductress benchmark framework"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# TODO fix paths for remote hosts?
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 1200
PERF_BENCH_THREADS = 16  # 75 connections per thread

# Default compiler arguments for Valkey builds.
# Bare make already gives O3+LTO+frame-pointer, so no extra flags are needed.
# (USE_FAST_FLOAT was a no-op — no such variable exists in the Valkey Makefile.)
DEFAULT_MAKE_ARGS = ""

# Benchmark defaults (single source of truth for CLI and TUI)
DEFAULT_IO_THREADS = 9
DEFAULT_PIPELINING = 10
DEFAULT_WARMUP = 5  # seconds
DEFAULT_DURATION = 30  # seconds
DEFAULT_REPETITIONS = 3
DEFAULT_VAL_SIZE = 512  # bytes
DEFAULT_KEY_SIZE = 0  # 0 = standard keys

# Dashboard data server (rsync target for --publish)
PUBLISH_TARGET = "ec2-user@data.conductress.rainsupreme.net:/var/www/data"

# Stable identity for this Conductress runner. runner.json is local deployment
# configuration and is intentionally not committed.
RUNNER_CONFIG_PATH = PROJECT_ROOT / "runner.json"


class Features(Enum):
    PIN_VALKEY_THREADS = "pin_valkey_threads"
    ENABLE_CPU_CONSISTENCY_MODE = "cpu_consistency_mode"
    BIND_NUMA_MEMORY = "bind_numa_memory"


FEATURE_STATES = {
    Features.PIN_VALKEY_THREADS: True,
    Features.ENABLE_CPU_CONSISTENCY_MODE: True,
    Features.BIND_NUMA_MEMORY: False,
}


def check_feature(feature: Features) -> bool:
    """Get the state of a specific feature flag."""
    return FEATURE_STATES.get(feature, False)


def get_all_features() -> dict[Features, bool]:
    """Get all feature flags and their current values."""
    return FEATURE_STATES.copy()


# Memory efficiency test configuration
MEM_TEST_ITEM_COUNT = 5_000_000  # 5 million items for memory tests
MEM_TEST_KEY_SIZE = 16  # Size of "key:__rand_int__" pattern
MEM_TEST_MEMBER_SIZE = 20  # Size of "element:__rand_int__" pattern (used by sadd/zadd)
MEM_TEST_SCORE_SIZE = 8  # Size of a double score (used by zadd)
MEM_TEST_MAX_CONCURRENT = 9  # Max concurrent server instances # TODO max session limit typically 10 by default
MEM_TEST_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days expiration

# TUI refresh interval in seconds
TUI_REFRESH_INTERVAL = 15

# =============================================================================
# RUNTIME CONSTANTS
# =============================================================================

# Task runner polls the local queue at this interval when idle (seconds)
QUEUE_POLL_INTERVAL = 4

# Fleet mailbox management happens only between tasks. The runner sleeps after
# the final boundary contact before starting benchmark work.
FLEET_IDLE_POLL_INTERVAL = 30
MANAGEMENT_SETTLE_SECONDS = 2.0
DELIVERY_JOURNAL_PATH = PROJECT_ROOT / "fleet_delivery.json"

# How often sweep fetches new commits from origin (seconds).
# Runs between jobs, not during benchmarks.
SWEEP_FETCH_INTERVAL = 3600

# =============================================================================
# Sweep configuration: throughput
# =============================================================================
SWEEP_SOURCE = "valkey"
SWEEP_REF = "origin/unstable"
SWEEP_STATE_DIR = PROJECT_ROOT / "sweep_data"
SWEEP_STATE_FILE = SWEEP_STATE_DIR / "state.json"
SWEEP_TEST = "get"
SWEEP_KEY_SIZE = 16
SWEEP_VAL_SIZE = 16
SWEEP_IO_THREADS = 7
SWEEP_PIPELINING = 10
SWEEP_WARMUP = 5
SWEEP_DURATION = 30
# Adaptive repetitions: run at least SWEEP_REPETITIONS reps, stop early once the
# 95% CI half-width of the mean is <= SWEEP_TARGET_CV (% of mean), up to
# SWEEP_MAX_REPS. Min is 5 (not 3) because Intel shows a bimodal
# between-restart distribution (docs/benchmark-precision-guide.md): with 3 reps
# there is a ~25% chance all land on one mode, freezing in a mean several % off
# with a deceptively tight interval.
SWEEP_REPETITIONS = 5
SWEEP_MAX_REPS = 10
SWEEP_TARGET_CV = 0.5
# Minimum |delta| to annotate a pinpointed change as "notable" in exported data.
# Set above binary layout noise floor (~3-4%) to reduce false positives.
ANNOTATION_THRESHOLD = 0.04
SWEEP_MAKE_ARGS = ""

# Versioned sweep epoch. Disabled by default until overlap validation is
# complete; enabling it runs additive v2 GET + 80:20 mixed coordinators while
# leaving every v1 coordinator/state file active and unchanged.
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a strict boolean environment variable.

    Unknown values fail startup rather than silently selecting the wrong sweep
    epoch. This keeps fleet enablement explicit and rollback-friendly.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_ENV_VALUES:
        return True
    if value in _FALSE_ENV_VALUES:
        return False
    allowed = ", ".join(sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES))
    raise ValueError(f"{name} must be one of: {allowed}; got {raw!r}")


SWEEP_V2_ENABLED = _env_bool("CONDUCTRESS_SWEEP_V2_ENABLED", False)
SWEEP_V2_EPOCH_ID = "v2"

SWEEP_V3_ENABLED = _env_bool("CONDUCTRESS_SWEEP_V3_ENABLED", False)
SWEEP_V3_EPOCH_ID = "v3"


def _env_epoch_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated epoch precedence list from the environment.

    An explicitly set but empty/blank value is an error rather than a silent
    fallback, matching ``_env_bool``: mis-set scheduling policy should fail
    startup, not quietly restore the default ordering.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not items:
        raise ValueError(f"{name} must be a comma-separated epoch list (e.g. 'v3,v1'); got {raw!r}")
    duplicates = {e for e in items if items.count(e) > 1}
    if duplicates:
        raise ValueError(f"{name} lists duplicate epochs: {sorted(duplicates)}")
    return items


# Scheduling precedence between measurement epochs, highest priority first.
#
# Before this existed, a newly added epoch outranked (or was outranked by) the
# legacy coordinators purely as a side effect of the order in which TaskRunner
# happened to append subscribers.  That was invisible in the scheduler and easy
# to break by moving a registration block.  Precedence is now explicit.
#
# It matters because NIGHTLY (untested HEAD) has absolute priority and returns on
# the FIRST matching coordinator, so whichever epoch is scanned first measures
# each new HEAD first.  With "v3" ahead of "v1", v3 becomes the primary epoch and
# v1 continues to run one cell later per HEAD -- deprioritized, not paused, so no
# v1 coverage is lost.
#
# Epochs absent from this list are scanned after every listed epoch, preserving
# their existing relative order.  Override without a deploy via
# CONDUCTRESS_SWEEP_EPOCH_PRECEDENCE (e.g. "v1,v3" restores v1-first).
SWEEP_EPOCH_PRECEDENCE = _env_epoch_list("CONDUCTRESS_SWEEP_EPOCH_PRECEDENCE", ("v3", "v1"))

# ---------------------------------------------------------------------------
# cachecannon-v3 sweep protocol
# ---------------------------------------------------------------------------
# Every value below was CONFIRMED by the Stage 2 ramp measurement on 2026-09-04
# (20 cells, ~1,780 per-second samples, G4 + AMD, 400c and 1200c, 5 fresh server
# starts each).  Two results are load-bearing and must not be "tidied":
#
#   * Warmup is 10s, NOT the 30s originally extrapolated from memtier.  Every
#     cell converges by second 10; cachecannon settles roughly 3x faster than
#     memtier.  Zero warmup fails only on G4/1200c, where an opening ~5% spike
#     leaks into the scored window (+0.643% median bias).
#   * Connections are 400, NOT 1200.  1200c was rejected on stability grounds:
#     AMD showed 2.520% between-restart CV across a 6% spread, which can never
#     satisfy the target below and would burn the rep ceiling on every cell.
#
# Changing any of these values defines a NEW workload identity, not a
# continuation of the series.  See conductress-cachecannon-v3/epoch-specification.md.
SWEEP_V3_CACHECANNON_COMMIT = "31a2befaa8bf7b3b7a7f7e03e2a847a2e407a3ce"
SWEEP_V3_WARMUP = 10
SWEEP_V3_DURATION = 30
SWEEP_V3_CONNECTIONS = 400
SWEEP_V3_CLIENT_THREADS = 8  # 50 connections per client thread at 400c
SWEEP_V3_IO_THREADS = 7  # server io-threads
SWEEP_V3_PIPELINING = 10
SWEEP_V3_VAL_SIZE = 16
SWEEP_V3_KEYSPACE = 3_000_000
SWEEP_V3_DISTRIBUTION = "uniform"
SWEEP_V3_SET_RATIO = 20  # the canonical mixed workload
SWEEP_V3_REPETITIONS = 5  # minimum reps
SWEEP_V3_MAX_REPS = 10  # adaptive ceiling
# Precision target for adaptive stopping.  NOTE: despite the historical field
# name, should_stop_adaptive() bounds the 95% CI half-width as a percent of the
# mean, which is a *wider* statistic than CV.  At n=5 the half-width is roughly
# 1.24x the CV, so the measured 0.28-0.30% CV at 400c corresponds to about
# 0.35-0.37% here -- inside this bound, but with less margin than the raw CV
# numbers suggest.
SWEEP_V3_TARGET_CV = 0.5

# Epoch registry: id -> dashboard metadata.  The publisher advertises these in
# every manifest so old URLs keep working while new dashboards can discover
# additional epochs.  Unknown ids fall back to a generic label rather than
# being silently mislabelled as some other epoch.
SWEEP_EPOCHS: dict[str, dict[str, str]] = {
    "v1": {"label": "Legacy v1 (stock generator)", "generator": "stock"},
    "v2": {"label": "Scalable v2 (patched generator)", "generator": "patched"},
    "v3": {"label": "Cachecannon v3 (io_uring generator)", "generator": "cachecannon"},
}

# Additional throughput workloads (each gets its own state file + series).
# Label is auto-generated as {test}-k{key_size}-v{val_size}-t{io_threads}-p{pipelining}.
# "platforms" limits the workload to specific architectures (omit for all platforms).
SWEEP_THROUGHPUT_WORKLOADS: list[dict] = [
    {"val_size": 64},
    {"val_size": 128},
    {"val_size": 16, "test": "set"},
    {"val_size": 128, "test": "set"},
    # No-pipeline workloads: single-request latency-sensitive baseline
    {"val_size": 16, "pipelining": 1},
    {"val_size": 128, "pipelining": 1},
    {"val_size": 16, "test": "set", "pipelining": 1},
    {"val_size": 128, "test": "set", "pipelining": 1},
    # Platform-optimal workloads: realistic configs for performance-sensitive users
    {"val_size": 16, "io_threads": 24, "pipelining": 100, "platforms": ["intel"]},
    {"val_size": 16, "io_threads": 24, "pipelining": 100, "test": "set", "platforms": ["intel"]},
    {"val_size": 16, "io_threads": 9, "pipelining": 50, "platforms": ["arm64", "graviton4"]},
    {"val_size": 16, "io_threads": 9, "pipelining": 50, "test": "set", "platforms": ["arm64", "graviton4"]},
]


# =============================================================================
# Sweep engine configuration: multi-engine support (Valkey, Redis, etc.)
# =============================================================================


@dataclass
class SweepEngine:
    """Configuration for a benchmarking engine (server software to sweep)."""

    source: str  # REPOSITORIES entry name (e.g. "valkey", "redis")
    ref: str  # git ref to track (e.g. "origin/unstable")
    binary_name: str  # server binary produced by build (e.g. "valkey-server", "redis-server")
    floor_tag: Optional[str] = None  # earliest tag to sweep from (None = use find_fork_point)
    make_args: str = DEFAULT_MAKE_ARGS  # compiler flags
    heap_alloc_funcs: list[str] = field(default_factory=list)  # for memory profiling
    profile_internals: bool = True  # collect CPU flamegraphs + jemalloc allocation breakdown for this engine


SWEEP_ENGINES: list[SweepEngine] = [
    SweepEngine(
        source="valkey",
        ref="origin/unstable",
        binary_name="valkey-server",
        floor_tag=None,
        make_args="",
        heap_alloc_funcs=["valkey_malloc", "valkey_calloc", "valkey_realloc"],
    ),
    SweepEngine(
        source="redis",
        ref="origin/unstable",
        binary_name="redis-server",
        floor_tag="8.0.0",
        make_args="",
        heap_alloc_funcs=["zmalloc", "zcalloc", "zrealloc"],
        # CPU flamegraphs and jemalloc allocation breakdowns expose the Redis binary's
        # symbol table (function names / call graph), which we do not analyze or surface.
        # Redis keeps aggregate performance data only: throughput, latency, total memory.
        profile_internals=False,
    ),
]


def get_sweep_engine(source: str) -> Optional["SweepEngine"]:
    """Look up a SweepEngine by source name."""
    for engine in SWEEP_ENGINES:
        if engine.source == source:
            return engine
    return None


def should_profile_internals(engine: Optional["SweepEngine"]) -> bool:
    """Whether to collect internal profiling (CPU flamegraphs, jemalloc breakdown) for an engine.

    Absent/unknown engine (e.g. a fork source, or legacy state with no engine) defaults to True
    so Valkey profiling is unaffected; an engine only opts out via profile_internals=False (Redis).
    This is the single source of truth for the policy — call it everywhere rather than inlining
    `engine.profile_internals` checks (which have to re-handle the None case and read poorly).
    """
    return engine.profile_internals if engine else True


# =============================================================================
# Sweep configuration: latency
# =============================================================================
LATENCY_STATE_FILE = SWEEP_STATE_DIR / "latency_state.json"
LATENCY_TARGET_RPS = 100_000  # flat rate, same across all platforms/commits
LATENCY_MAKE_ARGS = ""
LATENCY_DETECTION_THRESHOLD = 0.10  # 10% p99 change triggers bisection
LATENCY_THREADS = 4
LATENCY_CLIENTS = 16  # 64 total connections
LATENCY_PIPELINE = 1  # no pipelining — measures true per-request latency
LATENCY_DURATION = 60
LATENCY_KEYSPACE = 1_000_000
LATENCY_VAL_SIZE = 16
LATENCY_REPS = 3
MEMTIER_COMMIT = "d52544b1"  # pinned version for reproducible latency measurements
VALKEY_BENCHMARK_COMMIT = "d2eee78a151884518441572c53fc378bf6689e81"  # pinned valkey commit for benchmark client binary

# =============================================================================
# Sweep configuration: memory
# =============================================================================
MEMORY_STATE_DIR = SWEEP_STATE_DIR

# Benchmark metric collection interval (seconds). valkey-benchmark outputs ~4/sec.
BENCHMARK_UPDATE_INTERVAL = 0.1

# Status heartbeat interval during benchmark runs (seconds)
HEARTBEAT_INTERVAL = 5.0

# Maximum iterations for valkey-benchmark (-n flag). Set high so duration controls exit.
BENCHMARK_MAX_ITERATIONS = 2_000_000_000

# Server readiness: max attempts and delay between retries
SERVER_READY_MAX_RETRIES = 10
SERVER_READY_RETRY_DELAY = 1.0  # seconds

# Thread pinning: brief delay for scheduler to apply affinity changes
THREAD_PIN_SETTLE_DELAY = 0.1  # seconds

# when multiple valkey instances run on one host, they will start at this port number and count up
# (e.g. 9000, 9001, 9002, etc)
SERVER_PORT_RANGE_START = 9000

CONDUCTRESS_LOG = PROJECT_ROOT / "log.txt"

VALKEY_CLI = "valkey-cli"
VALKEY_BENCHMARK = "valkey-benchmark"

CONDUCTRESS_RESULTS = PROJECT_ROOT / "results"
CONDUCTRESS_OUTPUT = CONDUCTRESS_RESULTS / "output.jsonl"

CONDUCTRESS_QUEUE = PROJECT_ROOT / "benchmark_queue"
CONDUCTRESS_TMP = PROJECT_ROOT / "tmp"
CONDUCTRESS_FAILED_LOG = PROJECT_ROOT / "failed_tasks.jsonl"
CONDUCTRESS_FAILED_DIR = PROJECT_ROOT / "failed"

# ssh key to use when accessing the server
# Replace this with the path to your private key file
SSH_KEYFILE = PROJECT_ROOT / "server-keyfile.pem"

# Repositories to make available for testing
# format: (git_url, directory_name)
# Each will be cloned into ~/directory_name on each server
# The directory name is used to refer to the repo in the task queue and in results
REPOSITORIES = [
    ("https://github.com/valkey-io/valkey.git", "valkey"),
    ("https://github.com/rainsupreme/valkey.git", "rainsupreme"),
    ("https://github.com/valkey-io/valkey.git", "zuiderkwast"),
    ("https://github.com/JimB123/valkey.git", "JimB123"),
    ("https://github.com/valkey-rainfall/valkey.git", "valkey-rainfall"),
    ("https://github.com/redis/redis.git", "redis"),
]
REPO_NAMES = [repo[1] for repo in REPOSITORIES]

# unique name indicating the binary was uploaded manually
MANUALLY_UPLOADED = "manually_uploaded"
assert MANUALLY_UPLOADED not in REPO_NAMES, "MANUALLY_UPLOADED must not overlap with any repository names"


@dataclass
class ServerInfo:
    """Information about a server used in benchmarking."""

    ip: str
    """IPv4 address of the server."""
    username: str = ""
    """username to connect with"""
    name: str = ""
    """A unique descriptive name"""
    disabled: bool = False
    """Whether this server is disabled and should be skipped"""

    def __eq__(self, other) -> bool:
        if not isinstance(other, ServerInfo):
            return False

        def normalize_localhost(ip):
            return "127.0.0.1" if ip in ("localhost", "127.0.0.1", "::1") else ip

        return normalize_localhost(self.ip) == normalize_localhost(other.ip)


def load_server_ips() -> list[ServerInfo]:
    """Load server IPs from a JSON configuration file."""
    config_path = PROJECT_ROOT / "servers.json"
    default_path = PROJECT_ROOT / "servers.default.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())["valkey_servers"]
    elif default_path.exists():
        data = json.loads(default_path.read_text())["valkey_servers"]
    else:
        raise FileNotFoundError(f"No server config found at {config_path} or {default_path}")
    all_servers = [ServerInfo(**entry) for entry in data]
    return [s for s in all_servers if not s.disabled]


_SERVERS: Optional[list[ServerInfo]] = None


def get_servers() -> list[ServerInfo]:
    """Lazy accessor for server list. Loads from servers.json on first call."""
    global _SERVERS
    if _SERVERS is None:
        _SERVERS = load_server_ips()
    return _SERVERS
