"""Configuration for the Conductress benchmark framework"""

import json
from pathlib import Path

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 650
PERF_BENCH_THREADS = 64


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDUCTRESS_LOG = PROJECT_ROOT / "log.txt"
CONDUCTRESS_DATA_DUMP = PROJECT_ROOT / "testdump.txt"

VALKEY_CLI = PROJECT_ROOT / "valkey-cli"
VALKEY_BENCHMARK = PROJECT_ROOT / "valkey-benchmark"

CONDUCTRESS_RESULTS = PROJECT_ROOT / "results"
CONDUCTRESS_OUTPUT = CONDUCTRESS_RESULTS / "output.txt"

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
    ("https://github.com/SoftlyRaining/valkey.git", "SoftlyRaining"),
    ("https://github.com/valkey-io/valkey.git", "zuiderkwast"),
    ("https://github.com/JimB123/valkey.git", "JimB123"),
]
REPO_NAMES = [repo[1] for repo in REPOSITORIES]

# unique name indicating the binary was uploaded manually
MANUALLY_UPLOADED = "manually_uploaded"
assert MANUALLY_UPLOADED not in REPO_NAMES, "MANUALLY_UPLOADED must not overlap with any repository names"


def load_server_ips() -> list[str]:
    """Load server IPs from a JSON configuration file."""
    config_path = PROJECT_ROOT / "servers.json"
    try:
        return json.loads(config_path.read_text())["valkey_servers"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file {config_path} not found.") from exc


SERVERS = load_server_ips()
