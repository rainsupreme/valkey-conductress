"""Updates/installs all packages and dependencies and sets up servers for use."""

import concurrent.futures
import logging
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from . import config

ROOT = config.PROJECT_ROOT
DEV = True

logger = logging.getLogger(__name__)

# Get config values
SERVERS = config.SERVERS
SSH_KEYFILE = config.SSH_KEYFILE
REPOSITORIES = config.REPOSITORIES


def load_requirements(name: str) -> list[str]:
    with open(f"requirements/{name}.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # strip comments
    lines = [line.split("#")[0].strip() for line in lines]

    return lines


def get_linux_distro() -> str:
    with open("/etc/os-release", "r", encoding="utf-8") as f:
        data = f.readlines()
    data = [line for line in data if line.startswith("NAME")]
    assert len(data) == 1
    distro = data[0].split('"')[1]
    return distro


def subprocess_command(command: str) -> None:
    cmd_list = command.split()
    result = subprocess.run(
        cmd_list, check=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.stderr:
        print(repr(result.stderr))


# ======== ensure fabric installed and imported ========
try:
    from fabric import Connection
    from invoke import run
except ImportError:
    subprocess_command("sudo dnf install -y python3-pip")
    subprocess_command("python3 -m pip install --upgrade pip")
    subprocess_command("pip install fabric invoke")
    try:
        from fabric import Connection
        from invoke import run
    except ImportError:
        logger.error("Fabric is not installed even after attempting installation.")
        sys.exit(1)


def ensure_ssh_key() -> None:
    """Ensure ssh keyfile is present"""
    logger.info("Checking for ssh keyfile")
    if not SSH_KEYFILE.is_file():
        logger.error("Missing SSH keyfile: '%s' (this must be manually copied to the server)", SSH_KEYFILE)
        sys.exit(1)
    try:
        Path(SSH_KEYFILE).chmod(0o600)
    except PermissionError:
        logger.error("Failed to set permissions on %s", SSH_KEYFILE)
        sys.exit(1)


def ensure_server_ssh_fingerprints() -> None:
    """Ensure all servers are in known_hosts"""
    logger.info("Ensuring all servers known")
    for server_ip in SERVERS:
        result = run(f"ssh-keygen -F {server_ip}", hide=True)
        if not result or not result.stdout:
            logger.warning("%s: Adding new fingerprint to known_hosts...", server_ip)
            Path.home().joinpath(".ssh").mkdir(parents=True, exist_ok=True)
            run(f"ssh-keyscan -H {server_ip} -T 10 >> ~/.ssh/known_hosts 2>/dev/null", hide=True)


def path_exists(conn: Connection, path: Union[str, Path], expected_type: Optional[str] = None) -> bool:
    """Check if a path exists and get its type"""
    try:
        path_stat = conn.sftp().stat(str(path))

        if expected_type:
            if stat.S_ISDIR(path_stat.st_mode):
                assert expected_type == "directory"
            elif stat.S_ISREG(path_stat.st_mode):
                assert expected_type == "file"
            else:
                assert expected_type == "other"
        return True
    except FileNotFoundError:
        return False


def remove_motd(conn: Connection) -> None:
    """Remove the insights-client motd if it exists."""
    motd_path = "/etc/motd.d/insights-client"
    if path_exists(conn, motd_path, expected_type="file"):
        logger.info("%s: Removing insights-client motd", conn.host)
        conn.sudo(f"rm {motd_path}")


def update_pip_packages(conn: Connection):
    packages = load_requirements("pip-requirements")
    if DEV:
        packages += load_requirements("pip-requirements-dev")
    conn.run("python3 -m pip install --upgrade pip", hide=True)
    conn.run(f"pip install {' '.join(packages)}", hide=True)


def update_dnf_host(conn: Connection):
    packages = load_requirements("rhel-requirements")
    logger.info("%s: Updating os packages")
    conn.sudo("dnf update -y")
    conn.sudo('dnf groupinstall -y "Development Tools"')
    conn.sudo(f"dnf install -y {' '.join(packages)}")


def ensure_git_repo_cloned(conn: Connection, repo_url, target_dir):
    logger.info("%s: Ensuring repo %s...", conn.host, target_dir)
    if not path_exists(conn, target_dir, expected_type="directory"):
        result = conn.run(f'git clone "{repo_url}" "{target_dir}"', hide=True)
        assert result


def ensure_conductress(conn: Connection, pull=False):
    conductress_path = Path("conductress")

    if not path_exists(conn, conductress_path, expected_type="directory"):
        ensure_git_repo_cloned(
            conn, "https://github.com/SoftlyRaining/valkey-conductress.git", conductress_path
        )
    if pull:
        logger.info("%s: pulling conductress", conn.host)
        with conn.cd(conductress_path):
            conn.run("git pull")

    if not path_exists(conn, conductress_path / config.VALKEY_CLI, "file") or not path_exists(
        conn, conductress_path / config.VALKEY_BENCHMARK, "file"
    ):
        logger.info("%s: retrieving and building needed binaries", conn.host)
        valkey_path = conductress_path / "valkey"
        ensure_git_repo_cloned(conn, "https://github.com/valkey-io/valkey.git", valkey_path)

        with conn.cd(valkey_path):
            conn.run("git fetch")
            conn.run("git reset --hard origin/unstable", hide=True)
            conn.run("make distclean", hide=True)
            conn.run("make -j", hide=True)

        conn.run(f"cp {valkey_path / 'src/valkey-cli'} {conductress_path / config.VALKEY_CLI}", hide=True)
        conn.run(
            f"cp {valkey_path / 'src/valkey-benchmark'} {conductress_path / config.VALKEY_BENCHMARK}",
            hide=True,
        )


def update_host(conn: Connection):
    """Perform all updates on host at specified connection"""
    update_pip_packages(conn)
    update_dnf_host(conn)
    ensure_conductress(conn)
    ensure_git_repo_cloned(conn, "https://github.com/brendangregg/FlameGraph.git", "FlameGraph")

    logger.info("%s: Ensuring config repos cloned...", conn.host)
    for repo_url, target_dir in REPOSITORIES:
        ensure_git_repo_cloned(conn, repo_url, target_dir)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("⊹˚₊‧───Starting update/setup───‧₊˚⊹")

    ensure_ssh_key()
    ensure_server_ssh_fingerprints()

    logger.info("Connecting to all servers")
    connections = [Connection("localhost", connect_kwargs={"key_filename": str(config.SSH_KEYFILE)})]
    connections += [
        Connection(host, connect_kwargs={"key_filename": str(config.SSH_KEYFILE)}) for host in SERVERS
    ]

    logger.info("Updating all servers in parallel")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(update_host, connections)
    # for c in connections:
    #     logger.info("Updating %s", c.host)
    #     update_host(c)

    logger.info("Update/setup complete!")
