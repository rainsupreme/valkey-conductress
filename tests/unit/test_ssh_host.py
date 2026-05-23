"""Unit tests for SshHost — SSH connection and output normalization."""

from conductress.ssh_host import SshHost


class TestEnsureStr:
    """Tests for SshHost._ensure_str() — output normalization."""

    def test_none_returns_empty(self):
        assert SshHost._ensure_str(None) == ""

    def test_empty_string_returns_empty(self):
        assert SshHost._ensure_str("") == ""

    def test_empty_bytes_returns_empty(self):
        assert SshHost._ensure_str(b"") == ""

    def test_string_passthrough(self):
        assert SshHost._ensure_str("hello") == "hello"

    def test_bytes_decoded(self):
        assert SshHost._ensure_str(b"hello") == "hello"

    def test_bytearray_decoded(self):
        assert SshHost._ensure_str(bytearray(b"hello")) == "hello"

    def test_memoryview_decoded(self):
        data = memoryview(b"hello world")
        assert SshHost._ensure_str(data) == "hello world"

    def test_utf8_bytes(self):
        assert SshHost._ensure_str("café".encode()) == "café"


class TestSshHostInit:
    """Tests for SshHost initialization."""

    def test_default_state(self):
        host = SshHost("10.0.0.1")
        assert host.ip == "10.0.0.1"
        assert host.username == ""
        assert host.ssh is None

    def test_with_username(self):
        host = SshHost("10.0.0.1", username="ec2-user")
        assert host.username == "ec2-user"
