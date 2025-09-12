"""Updates/installs all packages and dependencies and sets up servers for use."""

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from asyncssh import SSHClientConnection

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
    import asyncssh
except ImportError:
    subprocess_command("sudo dnf install -y python3-pip")
    subprocess_command("python3 -m pip install --upgrade pip")
    subprocess_command("pip install asyncssh")
    try:
        import asyncssh
    except ImportError:
        logger.error(
            "asyncssh is not available even after installation. Try again - python may need to be restarted."
        )
        sys.exit(1)
from src.utility import async_run


@dataclass
class Host:
    name: str
    conn: asyncssh.SSHClientConnection

    async def run(self, command: str) -> str:
        result = await self.conn.run(command)
        assert (
            result.exit_status == 0
        ), f"Command failed: {command} on {self.name} with exit status {result.exit_status}"
        out = result.stdout
        if not out:
            return ""
        if isinstance(out, memoryview):
            return bytes(out).decode()
        if isinstance(out, bytes) or isinstance(out, bytearray):
            return out.decode()
        return out

    @classmethod
    async def from_name(cls, name: str) -> "Host":
        """Create a Host instance from a host name."""
        if name == "localhost":
            conn: SSHClientConnection = await asyncssh.connect(
                name,
                client_keys=[str(SSH_KEYFILE)],
                known_hosts=None,  # Disable known hosts check for localhost
            )
        else:
            conn: SSHClientConnection = await asyncssh.connect(
                name,
                client_keys=[str(SSH_KEYFILE)],
            )
        return cls(name=name, conn=conn)


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


async def ensure_server_known(server_ip: str):
    stdout, _ = await async_run(f"ssh-keygen -F {server_ip}")
    logger.info("%s: ensuring known", server_ip)
    if not stdout:
        logger.warning("%s: Adding new fingerprint to known_hosts...", server_ip)
        Path.home().joinpath(".ssh").mkdir(parents=True, exist_ok=True)
        await async_run(f"ssh-keyscan -H {server_ip} -T 10 >> ~/.ssh/known_hosts 2>/dev/null")


async def ensure_server_ssh_fingerprints() -> None:
    """Ensure all servers are in known_hosts"""
    await asyncio.gather(*(ensure_server_known(server_ip) for server_ip in SERVERS))


async def path_exists(host: Host, path: Union[str, Path], expected_type: Optional[str] = None) -> bool:
    """Check if a path exists and get its type"""

    commands = [f'test -{arg} "{path}"; echo $?' for arg in "efdL"]
    result = await host.run(" && ".join(commands))
    result = [int(x) == 0 for x in result.strip().split("\n")]  # return code 0 means test evaluated to true
    if not result[0]:
        return False
    if expected_type:
        if expected_type == "file":
            assert result[1], f"Expected {path} to be a file. ({result})"
        elif expected_type == "directory":
            assert result[2], f"Expected {path} to be a directory. ({result})"
        elif expected_type == "symlink":
            assert result[3], f"Expected {path} to be a symlink. ({result})"
        else:
            raise ValueError(f"Unknown expected_type: {expected_type}")
    return True


async def remove_motd(host: Host) -> None:
    """Remove the insights-client motd if it exists."""
    motd_path = "/etc/motd.d/insights-client"
    if await path_exists(host, motd_path, expected_type="file"):
        logger.info("%s: Removing insights-client motd", host.name)
        await host.run(f"sudo rm {motd_path}")


async def update_pip_packages(host: Host):
    packages = load_requirements("pip-requirements")
    if DEV:
        packages += load_requirements("pip-requirements-dev")
    await host.run("python3 -m pip install --upgrade pip")
    await host.run(f"pip install {' '.join(packages)}")


async def update_dnf_packages(host: Host):
    packages = load_requirements("rhel-requirements")
    logger.info("%s: Updating os packages", host.name)
    await host.run("sudo dnf update -y")
    devtools = host.run('sudo dnf groupinstall -y "Development Tools"')
    packages = host.run(f"sudo dnf install -y {' '.join(packages)}")
    await asyncio.gather(devtools, packages)


async def ensure_git_repo_cloned(host: Host, repo_url, target_dir):
    logger.info("%s: Ensuring repo %s...", host.name, target_dir)
    if not await path_exists(host, target_dir, expected_type="directory"):
        logger.info("%s: Cloning repo %s", host.name, repo_url)
        await host.run(f'git clone "{repo_url}" "{target_dir}"')


async def ensure_conductress(host: Host, pull=False):
    conductress_path = Path("conductress")

    if not await path_exists(host, conductress_path, expected_type="directory"):
        await ensure_git_repo_cloned(
            host, "https://github.com/SoftlyRaining/valkey-conductress.git", conductress_path
        )
    if pull:
        logger.info("%s: pulling conductress", host.name)
        await host.run(f"cd {conductress_path} && git pull")

    if not all(
        await asyncio.gather(
            path_exists(host, conductress_path / config.VALKEY_CLI, "file"),
            path_exists(host, conductress_path / config.VALKEY_BENCHMARK, "file"),
        )
    ):
        logger.info("%s: retrieving and building needed binaries", host.name)
        valkey_path = conductress_path / "valkey"
        await ensure_git_repo_cloned(host, "https://github.com/valkey-io/valkey.git", valkey_path)

        await host.run(
            f"cd {valkey_path} && git fetch && git reset --hard origin/unstable && make distclean && make -j"
        )

        await asyncio.gather(
            host.run(f"cp {valkey_path / 'src/valkey-cli'} {conductress_path / config.VALKEY_CLI}"),
            host.run(
                f"cp {valkey_path / 'src/valkey-benchmark'} {conductress_path / config.VALKEY_BENCHMARK}"
            ),
        )


async def update_host(name: str):
    """Perform all updates on host at specified connection"""
    host = await Host.from_name(name)
    await update_dnf_packages(host)
    await update_pip_packages(host)
    await ensure_conductress(host)
    await ensure_git_repo_cloned(host, "https://github.com/brendangregg/FlameGraph.git", "FlameGraph")

    logger.info("%s: Ensuring config repos cloned...", host.name)
    await asyncio.gather(
        *(ensure_git_repo_cloned(host, repo_url, target_dir) for repo_url, target_dir in REPOSITORIES)
    )


async def update_host_list(names: list[str]) -> None:
    await asyncio.gather(*(update_host(name) for name in names))


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.info("⊹˚₊‧───Starting update/setup───‧₊˚⊹")

    ensure_ssh_key()
    asyncio.run(ensure_server_ssh_fingerprints())

    asyncio.run(update_host_list(["localhost"] + SERVERS))

    logger.info("Update/setup complete!")
