"""Configuration for the Conductress benchmark framework"""

import json
from pathlib import Path

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 650
PERF_BENCH_THREADS = 64

# ssh key to use when accessing the server
# Replace this with the path to your private key file
SSH_KEYFILE = "server-keyfile.pem"

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

CONDUCTRESS_LOG = "./log.txt"
CONDUCTRESS_OUTPUT = "./results/output.txt"
CONDUCTRESS_DATA_DUMP = "./testdump.txt"


def load_server_ips() -> list:
    """Load server IPs from a JSON configuration file."""
    config_path = Path(__file__).parent / "servers.json"
    try:
        return json.loads(config_path.read_text())["valkey_servers"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file {config_path} not found.") from exc


SERVERS = load_server_ips()
