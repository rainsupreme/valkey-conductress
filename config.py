"""Configuration for the Conductress benchmark framework"""

import json
from pathlib import Path

PERF_BENCH_KEYSPACE = 3_000_000
PERF_BENCH_CLIENTS = 650
PERF_BENCH_THREADS = 64

# ssh key to use when accessing the server
# Replace this with the path to your private key file
SSH_KEYFILE = "server-keyfile.pem"

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


servers = load_server_ips()
