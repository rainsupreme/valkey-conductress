"""Updates/installs all packages and dependencies and sets up servers for use."""

import logging
import subprocess
import sys
from pathlib import Path

import config
import utility

logger = logging.getLogger(__name__)

# Get config values
SERVERS = config.SERVERS
SSH_KEY_FILE = Path(config.SSH_KEYFILE)
REPOSITORIES = config.REPOSITORIES

CLIENT_YUM_PACKAGES = [
    "cmake",
    "cmake3",
    "git",
    "python3-pip",
    "perf",
    "js-d3-flame-graph",
]
SERVER_YUM_PACKAGES = [
    "cmake",
    "cmake3",
    "git",
    "perf",
]


def command_exists(cmd):
    """Check if a command exists in the system."""
    return subprocess.call(f"command -v {cmd} > /dev/null 2>&1", shell=True) == 0


def remove_motd():
    """Remove the insights-client motd if it exists."""
    motd_path = Path("/etc/motd.d/insights-client")
    if motd_path.exists():
        logger.info("Removing insights-client motd")
        motd_path.unlink()


def update_local_host():
    """Update the local host with the required packages and repositories."""

    logger.info("Installing/updating required distro/pip packages...")
    utility.run_command("sudo yum update -y")
    utility.run_command(["sudo", "yum", "groupinstall", "-y", "Development Tools"])
    utility.run_command("sudo yum install -y " + " ".join(CLIENT_YUM_PACKAGES))
    utility.run_command("python3 -m pip install --upgrade pip")
    utility.run_command("pip install -r requirements.txt")
    utility.run_command("pip install -r requirements-dev.txt")

    logger.info("Checking for ssh keyfile")
    if not SSH_KEY_FILE.is_file():
        logger.error("Missing SSH keyfile: '%s' (this must be manually copied to the server)", SSH_KEY_FILE)
        sys.exit(1)
    try:
        Path(SSH_KEY_FILE).chmod(0o600)
    except PermissionError:
        logger.error("Failed to set permissions on %s", SSH_KEY_FILE)
        sys.exit(1)

    logger.info("Checking for required binaries...")
    buildable_files = [Path("valkey-cli"), Path("valkey-benchmark")]
    if not all(file.is_file() for file in buildable_files):
        logger.info("something was missing - retrieving and building needed binaries")
        valkey = Path("valkey")
        if not valkey.is_dir():
            utility.run_command(f"git clone https://github.com/valkey-io/valkey.git {valkey}")

        utility.run_command("git reset --hard && git pull && make distclean && make -j", cwd=valkey)

        for file in buildable_files:
            utility.run_command(f"cp {valkey/'src'/file} .")


def update_server(server_ip):
    """Update a server with the required packages and repositories."""
    logger.info("ensure server %s is in known-hosts", server_ip)
    if not utility.run_command(f"ssh-keygen -F {server_ip}", check=False):
        logger.warning("Adding new fingerprint for %s to known_hosts...", server_ip)
        Path.home().joinpath(".ssh").mkdir(parents=True, exist_ok=True)
        utility.run_command(f"ssh-keyscan -H {server_ip} -T 10 >> ~/.ssh/known_hosts 2>/dev/null")

    if utility.run_command("exit", remote_ip=server_ip, check=False):
        logger.error("Error: Cannot connect to %s", server_ip)
        sys.exit(1)

    logger.info("Setting up packages on server %s...", server_ip)
    remote_commands = f"""
    set -e
    sudo yum update -y
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y {" ".join(SERVER_YUM_PACKAGES)}
    """
    utility.run_command([remote_commands], remote_ip=server_ip)

    logger.info("Ensuring repos cloned on (%s)...", server_ip)
    for repo_url, target_dir in REPOSITORIES:
        logger.info("Checking %s...", target_dir)
        remote_commands = f"""
            if [ ! -d "{target_dir}" ]; then
                git clone "{repo_url}" "{target_dir}"
            fi
        """
        utility.run_command([remote_commands], remote_ip=server_ip, remote_pseudo_terminal=False)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("⊹˚₊‧───Starting update/setup───‧₊˚⊹")
    remove_motd()
    update_local_host()
    for server in SERVERS:
        update_server(server)
    logger.info("Update/setup complete!")
