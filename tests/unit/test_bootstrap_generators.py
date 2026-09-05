"""Tests for generator provisioning in bootstrap: io_uring enablement and cachecannon.

Why these exist:

* io_uring: the generator fleet must run ONE I/O engine everywhere. cachecannon
  silently falls back to ringline's mio engine where io_uring is disabled, which
  is a measurement confound, not a failure. RHEL 9 disables io_uring by default.
* cachecannon: the v3 epoch's canonical generator must exist at the pinned
  commit on every runner; version drift across the fleet silently breaks
  comparability.
* LimitNOFILE: cachecannon at 400 connections x 8 threads can require ~128k fds
  (observed failing fast on a kernel-5.14 RHEL 9 host with 65536).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductress import config
from conductress.bootstrap import (
    IO_URING_SYSCTL_DROPIN,
    SYSTEMD_SERVICE_TEMPLATE,
    Host,
    ensure_cachecannon,
    ensure_io_uring_enabled,
)


def _host(responses):
    """Host mock whose run() answers by first matching substring in `responses`.

    `responses` is a list of (substring, reply) pairs; first match wins.
    Unmatched commands return "". All commands are recorded on host.commands.
    """
    host = MagicMock(spec=Host)
    host.log_info_msg = MagicMock()
    host.log_warn_msg = MagicMock()
    host.get_home_path = MagicMock(return_value=__import__("pathlib").Path("/home/ec2-user"))
    host.commands = []

    async def fake_run(command, *args, **kwargs):
        host.commands.append(command)
        for needle, reply in responses:
            if needle in command:
                return reply
        return ""

    host.run = AsyncMock(side_effect=fake_run)
    return host


def _ran(host, needle):
    return any(needle in c for c in host.commands)


class TestEnsureIoUringEnabled:
    @pytest.mark.asyncio
    async def test_disabled_host_gets_enabled_and_persisted(self):
        """The RHEL 9 case: io_uring_disabled=2 must be flipped and persisted."""
        host = _host(
            [
                ("/proc/sys/kernel/io_uring_disabled", "2\n"),
                (IO_URING_SYSCTL_DROPIN, ""),  # no existing drop-in
            ]
        )
        await ensure_io_uring_enabled(host)
        assert _ran(host, "sysctl -w kernel.io_uring_disabled=0")
        assert _ran(host, f"sudo tee {IO_URING_SYSCTL_DROPIN}")

    @pytest.mark.asyncio
    async def test_enabled_host_not_rewritten_but_still_persisted(self):
        """Runtime value 0 with no drop-in: no sysctl -w, but persist anyway --
        a reboot would otherwise restore the distro default."""
        host = _host(
            [
                ("/proc/sys/kernel/io_uring_disabled", "0\n"),
                (IO_URING_SYSCTL_DROPIN, ""),
            ]
        )
        await ensure_io_uring_enabled(host)
        assert not _ran(host, "sysctl -w")
        assert _ran(host, f"sudo tee {IO_URING_SYSCTL_DROPIN}")

    @pytest.mark.asyncio
    async def test_absent_sysctl_is_left_alone(self):
        """Kernels predating the sysctl have io_uring unconditionally enabled;
        writing the sysctl there would break sysctl --system on boot."""
        host = _host(
            [
                ("/proc/sys/kernel/io_uring_disabled", "absent\n"),
            ]
        )
        await ensure_io_uring_enabled(host)
        assert not _ran(host, "sysctl -w")
        assert not _ran(host, "sudo tee")

    @pytest.mark.asyncio
    async def test_existing_correct_dropin_not_rewritten(self):
        from conductress.bootstrap import IO_URING_SYSCTL_CONTENT

        host = _host(
            [
                ("/proc/sys/kernel/io_uring_disabled", "0\n"),
                (IO_URING_SYSCTL_DROPIN, IO_URING_SYSCTL_CONTENT),
            ]
        )
        await ensure_io_uring_enabled(host)
        assert not _ran(host, "sudo tee")


class TestEnsureCachecannon:
    PIN = config.SWEEP_V3_CACHECANNON_COMMIT

    @pytest.mark.asyncio
    async def test_skips_when_pinned_and_built(self):
        host = _host([("git rev-parse HEAD", self.PIN + "\n")])
        with patch("conductress.bootstrap.path_exists", new=AsyncMock(return_value=True)):
            await ensure_cachecannon(host)
        assert not _ran(host, "cargo build")
        assert not _ran(host, "git clone")

    @pytest.mark.asyncio
    async def test_wrong_commit_triggers_rebuild(self):
        """Version drift is a silent comparability break -- must rebuild."""
        host = _host(
            [
                ("git rev-parse HEAD", "deadbeef" * 5 + "\n"),
                ("command -v cargo", "yes\n"),
            ]
        )
        with patch("conductress.bootstrap.path_exists", new=AsyncMock(return_value=True)):
            await ensure_cachecannon(host)
        assert _ran(host, f"git checkout {self.PIN}")
        assert _ran(host, "cargo build --release")

    @pytest.mark.asyncio
    async def test_fresh_host_clones_installs_toolchain_and_builds(self):
        host = _host(
            [
                ("git rev-parse HEAD", "\n"),
                ("command -v cargo", "no\n"),
            ]
        )
        # Call order: (1) repo dir -> absent, (2) at-pin binary check is skipped
        # (not at pin), so next is (2) the post-build binary check -> present.
        replies = iter([False, True])

        async def path_exists_seq(*args, **kwargs):
            return next(replies, True)

        with patch("conductress.bootstrap.path_exists", new=AsyncMock(side_effect=path_exists_seq)):
            await ensure_cachecannon(host)
        assert _ran(host, "git clone https://github.com/cachecannon/cachecannon.git")
        assert _ran(host, "dnf install -y cargo rust")
        assert _ran(host, "cargo build --release")

    @pytest.mark.asyncio
    async def test_missing_binary_after_build_raises(self):
        """A 'successful' build with no binary must fail loudly, not record a
        provisioned host that will emit opaque exit-1 generator failures."""
        host = _host(
            [
                ("git rev-parse HEAD", "deadbeef" * 5 + "\n"),
                ("command -v cargo", "yes\n"),
            ]
        )
        # Repo dir exists; binary never appears.
        seq = iter([True, False])

        async def path_exists_seq(host_arg, path, *args, **kwargs):
            try:
                return next(seq)
            except StopIteration:
                return False

        with patch("conductress.bootstrap.path_exists", new=AsyncMock(side_effect=path_exists_seq)):
            with pytest.raises(RuntimeError, match="binary missing"):
                await ensure_cachecannon(host)

    def test_binary_path_matches_task_default(self):
        """Bootstrap must build where the cachecannon task expects the binary."""
        from conductress.tasks.task_cachecannon import DEFAULT_CACHECANNON_BINARY

        assert DEFAULT_CACHECANNON_BINARY == "/home/ec2-user/cachecannon/target/release/cachecannon"


class TestServiceNofileLimit:
    def test_service_template_nofile_covers_cachecannon(self):
        """400c x 8t on io_uring can need ~128,128 fds (kernel-5.14 observation);
        the service template must clear that with margin."""
        for line in SYSTEMD_SERVICE_TEMPLATE.splitlines():
            if line.startswith("LimitNOFILE="):
                assert int(line.split("=")[1]) >= 128128 * 2
                return
        raise AssertionError("SYSTEMD_SERVICE_TEMPLATE has no LimitNOFILE line")
