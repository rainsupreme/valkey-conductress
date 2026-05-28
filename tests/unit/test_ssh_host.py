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


class TestSshConnectionPool:
    """Tests for class-level SSH connection pooling."""

    def setup_method(self):
        """Clear pool between tests."""
        SshHost._pool.clear()

    def test_pool_reuses_connection(self, monkeypatch):
        """Multiple SshHost instances for the same IP should share one connection."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False

        call_count = 0

        async def fake_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_conn

        monkeypatch.setattr("conductress.ssh_host.asyncssh.connect", fake_connect)

        async def run():
            host1 = SshHost("127.0.0.1")
            host2 = SshHost("127.0.0.1")
            await host1.ensure_ssh_connection()
            await host2.ensure_ssh_connection()
            assert host1.ssh is host2.ssh
            assert call_count == 1  # Only one real connection made

        asyncio.run(run())

    def test_pool_reconnects_on_closed(self, monkeypatch):
        """If pooled connection is closed, a new one is created."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        closed_conn = MagicMock()
        closed_conn.is_closed.return_value = True
        new_conn = MagicMock()
        new_conn.is_closed.return_value = False

        connections = iter([closed_conn, new_conn])

        async def fake_connect(*args, **kwargs):
            return next(connections)

        monkeypatch.setattr("conductress.ssh_host.asyncssh.connect", fake_connect)

        async def run():
            host1 = SshHost("127.0.0.1")
            await host1.ensure_ssh_connection()
            # Simulate connection dying
            SshHost._pool[("127.0.0.1", "")] = closed_conn
            host2 = SshHost("127.0.0.1")
            host2.ssh = closed_conn  # Simulate stale reference
            await host2.ensure_ssh_connection()
            assert host2.ssh is new_conn

        asyncio.run(run())

    def test_different_hosts_get_different_connections(self, monkeypatch):
        """Different IPs should get separate pooled connections."""
        import asyncio
        from unittest.mock import MagicMock

        conns = {}

        async def fake_connect(ip, **kwargs):
            conn = MagicMock()
            conn.is_closed.return_value = False
            conns[ip] = conn
            return conn

        monkeypatch.setattr("conductress.ssh_host.asyncssh.connect", fake_connect)

        async def run():
            h1 = SshHost("10.0.0.1")
            h2 = SshHost("10.0.0.2")
            await h1.ensure_ssh_connection()
            await h2.ensure_ssh_connection()
            assert h1.ssh is not h2.ssh
            assert len(conns) == 2

        asyncio.run(run())
