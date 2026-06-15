"""Unit tests for platform detection."""

from unittest.mock import mock_open, patch

from conductress.platform import get_local_platform_tag


@patch("platform.machine", return_value="aarch64")
def test_arm64_detection(mock_machine):
    assert get_local_platform_tag() == "arm64"


@patch("platform.machine", return_value="x86_64")
def test_intel_detection(mock_machine):
    cpuinfo = "vendor_id\t: GenuineIntel\nmodel name\t: Intel Xeon\n"
    with patch("builtins.open", mock_open(read_data=cpuinfo)):
        assert get_local_platform_tag() == "intel"


@patch("platform.machine", return_value="x86_64")
def test_amd_detection(mock_machine):
    cpuinfo = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC\n"
    with patch("builtins.open", mock_open(read_data=cpuinfo)):
        assert get_local_platform_tag() == "amd64"


@patch("platform.machine", return_value="x86_64")
def test_x86_fallback_to_amd(mock_machine):
    with patch("builtins.open", side_effect=OSError):
        assert get_local_platform_tag() == "amd64"
