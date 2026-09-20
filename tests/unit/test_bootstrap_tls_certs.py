"""bootstrap.ensure_tls_test_certs: idempotent CA + server cert generation.

Drives ``ensure_tls_test_certs`` against a fake Host that executes each shell
command locally in a temp directory, then asserts ``openssl verify`` accepts
the produced chain and that a second call is a no-op (idempotent). Skipped when
openssl is not installed.
"""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from conductress import bootstrap

_OPENSSL = shutil.which("openssl")
requires_openssl = pytest.mark.skipif(_OPENSSL is None, reason="openssl not installed")


class _LocalHost:
    """A bootstrap.Host stand-in that runs commands locally under a home dir."""

    def __init__(self, home: Path):
        self._home = home
        self.commands: list = []

    def get_home_path(self) -> Path:
        return self._home

    def log_info_msg(self, message: str) -> None:  # pragma: no cover - logging only
        pass

    def log_warn_msg(self, message: str) -> None:  # pragma: no cover - logging only
        pass

    async def run(self, command: str, check: bool = True) -> str:
        self.commands.append(command)
        result = subprocess.run(command, shell=True, capture_output=True, encoding="utf-8")
        if check and result.returncode != 0:
            raise RuntimeError(f"command failed: {command}\n{result.stderr}")
        return result.stdout or ""


async def _path_exists(host, path, expected_type=None):
    """Local re-implementation of bootstrap.path_exists for the fake host."""
    p = Path(str(path))
    if not p.exists():
        return False
    if expected_type == "file" and not p.is_file():
        raise RuntimeError(f"{path} not a file")
    if expected_type == "directory" and not p.is_dir():
        raise RuntimeError(f"{path} not a directory")
    return True


@requires_openssl
def test_ensure_tls_test_certs_generates_a_verifiable_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "path_exists", _path_exists)
    home = tmp_path / "home"
    (home / "conductress").mkdir(parents=True)
    host = _LocalHost(home)

    asyncio.run(bootstrap.ensure_tls_test_certs(host))

    tls_dir = home / "conductress" / "tls"
    ca = tls_dir / "ca.crt"
    server_crt = tls_dir / "server.crt"
    server_key = tls_dir / "server.key"
    assert ca.is_file() and server_crt.is_file() and server_key.is_file()

    # openssl verify accepts the server cert against the CA.
    verify = subprocess.run(
        [_OPENSSL, "verify", "-CAfile", str(ca), str(server_crt)],
        capture_output=True,
        encoding="utf-8",
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


@requires_openssl
def test_ensure_tls_test_certs_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "path_exists", _path_exists)
    home = tmp_path / "home"
    (home / "conductress").mkdir(parents=True)
    host = _LocalHost(home)

    asyncio.run(bootstrap.ensure_tls_test_certs(host))
    first_count = len(host.commands)
    # A second call sees the four files present and runs no openssl commands.
    host.commands.clear()
    asyncio.run(bootstrap.ensure_tls_test_certs(host))
    assert host.commands == []
    assert first_count > 0
