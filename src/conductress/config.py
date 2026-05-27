"""Configuration for the Conductress benchmark framework"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# TODO fix paths for remote hosts?
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 1200
PERF_BENCH_THREADS = 16  # 75 connections per thread

# Default compiler arguments for Valkey builds
# Bare make already gives O3+LTO+frame-pointer; USE_FAST_FLOAT enables fast_float library
DEFAULT_MAKE_ARGS = "USE_FAST_FLOAT=yes"

# Benchmark defaults (single source of truth for CLI and TUI)
DEFAULT_IO_THREADS = 9
DEFAULT_PIPELINING = 10
DEFAULT_WARMUP = 30  # seconds
DEFAULT_DURATION = 300  # seconds (5m)
DEFAULT_REPETITIONS = 5
DEFAULT_VAL_SIZE = 512  # bytes
DEFAULT_KEY_SIZE = 0  # 0 = standard keys

# Dashboard data server (rsync target for --publish)
PUBLISH_TARGET = "ec2-user@data.conductress.rainsupreme.net:/var/www/data"


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

# Task runner polls the queue at this interval when idle (seconds)
QUEUE_POLL_INTERVAL = 4

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
SWEEP_VAL_SIZE = 16
SWEEP_IO_THREADS = 7
SWEEP_PIPELINING = 10
SWEEP_WARMUP = 5
SWEEP_DURATION = 30
SWEEP_REPETITIONS = 3
SWEEP_MAX_REPS = 7
SWEEP_TARGET_CV = 0.5
SWEEP_MAKE_ARGS = "USE_FAST_FLOAT=yes"

# =============================================================================
# Sweep configuration: latency
# =============================================================================
LATENCY_STATE_FILE = SWEEP_STATE_DIR / "latency_state.json"
LATENCY_LOAD_FRACTION = 0.70
LATENCY_IO_THREADS = 9
LATENCY_MAKE_ARGS = "USE_FAST_FLOAT=yes"
LATENCY_DETECTION_THRESHOLD = 0.10  # 10% p99 change triggers bisection
LATENCY_THREADS = 4
LATENCY_CLIENTS = 50  # 200 total connections
LATENCY_PIPELINE = 10
LATENCY_DURATION = 60
LATENCY_KEYSPACE = 1_000_000
LATENCY_VAL_SIZE = 16
LATENCY_REPS = 3
MEMTIER_COMMIT = "d52544b1"  # pinned version for reproducible latency measurements

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
