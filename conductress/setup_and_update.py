"""Updates/installs all packages and dependencies and sets up servers for use."""

import concurrent.futures
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Union

from . import config

ROOT = config.PROJECT_ROOT

logger = logging.getLogger(__name__)

# Get config values
SERVERS = config.SERVERS
SSH_KEYFILE = config.SSH_KEYFILE
REPOSITORIES = config.REPOSITORIES

YUM_PACKAGES = [
    "cmake",
    "cmake3",
    "git",
    "python3-pip",
    "perf",
    "js-d3-flame-graph",
    "perl-open.noarch",  # needed for brendangregg/FlameGraph in rhel
]


def run_command(
    command: Union[str, Sequence[str]],
    remote_ip: Union[str, None] = None,
    remote_pseudo_terminal: bool = True,
    cwd: Union[Path, None] = None,
    check: bool = True,
):
    """Run a console command and return its output."""
    if remote_ip is None:  # local command
        cmd_list = command.split() if isinstance(command, str) else command
        result = subprocess.run(
            cmd_list,
            check=check,
            encoding="utf-8",
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stderr:
            print(repr(result.stderr))
        return result.stdout, result.stderr

    if isinstance(command, str):
        remote_command = command
    else:
        remote_command = " ".join(command)
    if cwd is not None:
        remote_command = f"cd {str(cwd)}; {remote_command}"

    # remote command
    ssh_command = ["ssh", "-q"]
    if not remote_pseudo_terminal:
        ssh_command += ["-T"]  # disable pseudo-terminal allocation for non-interactive sessions
    ssh_command += ["-i", str(SSH_KEYFILE), remote_ip, remote_command]

    result = subprocess.run(
        ssh_command, check=check, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout, result.stderr


def command_exists(cmd):
    """Check if a command exists in the system."""
    return subprocess.call(f"command -v {cmd} > /dev/null 2>&1", shell=True) == 0


def remove_motd() -> None:
    """Remove the insights-client motd if it exists."""
    motd_path = Path("/etc/motd.d/insights-client")
    if motd_path.exists():
        logger.info("Removing insights-client motd")
        motd_path.unlink()


def update_local_host() -> None:
    """Update the local host with the required packages and repositories."""
    requirements = ROOT / "requirements.txt"
    requirements_dev = ROOT / "requirements-dev.txt"

    logger.info("Installing/updating required distro packages...")
    run_command("sudo yum update -y")
    run_command(["sudo", "yum", "groupinstall", "-y", "Development Tools"])
    run_command("sudo yum install -y " + " ".join(YUM_PACKAGES))

    logger.info("Installing/updating Python packages...")
    run_command("python3 -m pip install --upgrade pip")
    run_command(f"pip install -r {requirements}")
    run_command(f"pip install -r {requirements_dev}")

    logger.info("Checking for ssh keyfile")
    if not SSH_KEYFILE.is_file():
        logger.error("Missing SSH keyfile: '%s' (this must be manually copied to the server)", SSH_KEYFILE)
        sys.exit(1)
    try:
        Path(SSH_KEYFILE).chmod(0o600)
    except PermissionError:
        logger.error("Failed to set permissions on %s", SSH_KEYFILE)
        sys.exit(1)

    logger.info("Checking for required binaries...")
    buildable_files: list[Path] = [config.VALKEY_CLI, config.VALKEY_BENCHMARK]
    if not all(file.is_file() for file in buildable_files):
        logger.info("something was missing - retrieving and building needed binaries")
        valkey = ROOT / "valkey"
        if not valkey.is_dir():
            run_command(f"git clone https://github.com/SoftlyRaining/valkey.git {valkey}")

        run_command("git fetch", cwd=valkey)
        run_command("git reset --hard origin/benchmark-multi-replace", cwd=valkey)
        run_command("make distclean", cwd=valkey)
        run_command("make -j", cwd=valkey)

        for file in buildable_files:
            run_command(f"cp {valkey/'src'/file.name} {file}")


def ensure_server_git_repo(server_ip, repo_url, target_dir):
    """Clone git repo if it doesn't exist on server"""
    logger.info("%s: Ensuring repo %s...", server_ip, target_dir)
    remote_commands = f"""
        if [ ! -d "{target_dir}" ]; then
            git clone "{repo_url}" "{target_dir}"
        fi
    """
    run_command([remote_commands], remote_ip=server_ip, remote_pseudo_terminal=False)


def update_server(server_ip):
    """Update a server with the required packages and repositories."""
    logger.info("%s: ensure server is in known-hosts", server_ip)
    std, _ = run_command(f"ssh-keygen -F {server_ip}", check=False)
    if not std:
        logger.warning("%s: Adding new fingerprint to known_hosts...", server_ip)
        Path.home().joinpath(".ssh").mkdir(parents=True, exist_ok=True)
        run_command(f"ssh-keyscan -H {server_ip} -T 10 >> ~/.ssh/known_hosts 2>/dev/null")

    std, _ = run_command("exit", remote_ip=server_ip, check=False)
    if std:
        logger.error("Error: Cannot connect to %s", server_ip)
        sys.exit(1)

    logger.info("%s: Setting up packages...", server_ip)
    remote_commands = f"""
    set -e
    sudo yum update -y
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y {" ".join(YUM_PACKAGES)}
    """
    run_command([remote_commands], remote_ip=server_ip)

    ensure_server_git_repo(server_ip, "https://github.com/brendangregg/FlameGraph.git", "FlameGraph")

    logger.info("%s: Ensuring repos cloned...", server_ip)
    for repo_url, target_dir in REPOSITORIES:
        ensure_server_git_repo(server_ip, repo_url, target_dir)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("⊹˚₊‧───Starting update/setup───‧₊˚⊹")
    remove_motd()
    update_local_host()

    # update servers in parallel
    logger.info("Updating %d servers in parallel...", len(SERVERS))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(update_server, SERVERS)

    logger.info("Update/setup complete!")
