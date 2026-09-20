"""Tests for host disk reporting in the status export payload."""

from unittest.mock import patch

from conductress.status_export import DISK_WATCH_PATHS, _get_disk_info


def _usage(total: int, free: int):
    return type("U", (), {"total": total, "used": total - free, "free": free})()


def _stat(dev: int):
    return type("S", (), {"st_dev": dev})()


class TestGetDiskInfo:
    """_get_disk_info feeds the per-host disk alarm on the status dashboard."""

    def test_watches_project_root_temp_dir_and_tmp(self):
        """The project filesystem, the resolved temp dir and /tmp itself are all in the watch list."""
        import tempfile

        from conductress.config import PROJECT_ROOT

        assert str(PROJECT_ROOT) in DISK_WATCH_PATHS
        assert tempfile.gettempdir() in DISK_WATCH_PATHS
        assert "/tmp" in DISK_WATCH_PATHS

    def test_single_filesystem_keeps_legacy_shape(self):
        """Same device under both paths: one entry, top-level fields as before."""
        # 256 GB total, 92% free
        with (
            patch("conductress.status_export.shutil.disk_usage", return_value=_usage(256_000_000_000, 236_000_000_000)),
            patch("conductress.status_export.os.stat", return_value=_stat(1)),
        ):
            info = _get_disk_info(("/root", "/tmp"))
        assert info["path"] == "/root"
        assert info["size_bytes"] == 256_000_000_000
        assert info["avail_bytes"] == 236_000_000_000
        assert info["free_pct"] == 92
        assert len(info["filesystems"]) == 1
        assert "device" not in info and "device" not in info["filesystems"][0]

    def test_tightest_filesystem_drives_the_alarm(self):
        """A full tmpfs shows up in free_pct even when the root disk is 70% free."""
        usages = {"/root": _usage(512_000_000_000, 365_000_000_000), "/tmp": _usage(189_000_000_000, 0)}
        devices = {"/root": _stat(1), "/tmp": _stat(2)}
        with (
            patch("conductress.status_export.shutil.disk_usage", side_effect=lambda p: usages[p]),
            patch("conductress.status_export.os.stat", side_effect=lambda p: devices[p]),
        ):
            info = _get_disk_info(("/root", "/tmp"))
        assert info["path"] == "/tmp"
        assert info["free_pct"] == 0
        assert info["size_bytes"] == 189_000_000_000
        assert [fs["path"] for fs in info["filesystems"]] == ["/root", "/tmp"]
        assert [fs["free_pct"] for fs in info["filesystems"]] == [71, 0]

    def test_low_disk_free_pct(self):
        with (
            patch("conductress.status_export.shutil.disk_usage", return_value=_usage(100, 3)),
            patch("conductress.status_export.os.stat", return_value=_stat(1)),
        ):
            assert _get_disk_info(("/only",))["free_pct"] == 3

    def test_unreadable_path_is_skipped_not_fatal(self):
        def usage(path):
            if path == "/gone":
                raise OSError("boom")
            return _usage(100, 50)

        with (
            patch("conductress.status_export.shutil.disk_usage", side_effect=usage),
            patch("conductress.status_export.os.stat", return_value=_stat(1)),
        ):
            info = _get_disk_info(("/gone", "/ok"))
        assert info["path"] == "/ok"
        assert [fs["path"] for fs in info["filesystems"]] == ["/ok"]

    def test_returns_empty_when_nothing_readable(self):
        with patch("conductress.status_export.shutil.disk_usage", side_effect=OSError("boom")):
            assert _get_disk_info() == {}
