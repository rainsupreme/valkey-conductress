import json
from pathlib import Path

perf_bench_keyspace = 3000000 # 3 million

# ssh key to use when accessing the server
# Replace this with the path to your private key file
sshkeyfile = 'server-keyfile.pem'

conductress_log = './log.txt'
conductress_output = './results/output.txt'
conductress_data_dump = './testdump.txt'

def load_server_ips():
    config_path = Path(__file__).parent / 'servers.json'
    try:
        return json.loads(config_path.read_text())['valkey_servers']
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
servers = load_server_ips()
