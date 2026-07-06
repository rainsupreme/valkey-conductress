"""Tests for host disk reporting in the status export payload."""

from unittest.mock import patch

from conductress.status_export import _get_disk_info


class TestGetDiskInfo:
    """_get_disk_info feeds the per-host disk alarm on the status dashboard."""

    def test_reports_free_pct_and_bytes(self):
        # 256 GB total, 92% free
        fake = type("U", (), {"total": 256_000_000_000, "used": 20_000_000_000, "free": 236_000_000_000})()
        with patch("conductress.status_export.shutil.disk_usage", return_value=fake):
            info = _get_disk_info()
        assert info["size_bytes"] == 256_000_000_000
        assert info["avail_bytes"] == 236_000_000_000
        assert info["free_pct"] == 92

    def test_low_disk_free_pct(self):
        fake = type("U", (), {"total": 100, "used": 97, "free": 3})()
        with patch("conductress.status_export.shutil.disk_usage", return_value=fake):
            assert _get_disk_info()["free_pct"] == 3

    def test_returns_empty_on_oserror(self):
        with patch("conductress.status_export.shutil.disk_usage", side_effect=OSError("boom")):
            assert _get_disk_info() == {}
