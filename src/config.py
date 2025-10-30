"""Configuration for the Conductress benchmark framework"""

import json
from pathlib import Path

from attr import dataclass

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 1200
PERF_BENCH_THREADS = 64

# Memory efficiency test configuration
MEM_TEST_ITEM_COUNT = 5_000_000  # 5 million items for memory tests
MEM_TEST_KEY_SIZE = 16  # Size of "key:__rand_int__" pattern
MEM_TEST_MAX_CONCURRENT = (
    9  # Max concurrent server instances # TODO max session limit typically 10 by default
)
MEM_TEST_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days expiration

# when multiple valkey instances run on one host, they will start at this port number and count up
# (e.g. 9000, 9001, 9002, etc)
SERVER_PORT_RANGE_START = 9000

# TODO fix paths for remote hosts?
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDUCTRESS_LOG = PROJECT_ROOT / "log.txt"

VALKEY_CLI = "valkey-cli"
VALKEY_BENCHMARK = "valkey-benchmark"

CONDUCTRESS_RESULTS = PROJECT_ROOT / "results"
CONDUCTRESS_OUTPUT = CONDUCTRESS_RESULTS / "output.jsonl"

CONDUCTRESS_QUEUE = PROJECT_ROOT / "benchmark_queue"

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


def load_server_ips() -> list[ServerInfo]:
    """Load server IPs from a JSON configuration file."""
    config_path = PROJECT_ROOT / "servers.json"
    try:
        data = json.loads(config_path.read_text())["valkey_servers"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file {config_path} not found.") from exc
    return [ServerInfo(**entry) for entry in data]


SERVERS = load_server_ips()
