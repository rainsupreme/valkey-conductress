import json

import pytest

import src.config as config_module
from src.config import ServerInfo, get_servers, load_server_ips


class TestServerInfo:
    def test_disabled_default_false(self):
        s = ServerInfo(ip="1.2.3.4")
        assert s.disabled is False

    def test_disabled_true(self):
        s = ServerInfo(ip="1.2.3.4", disabled=True)
        assert s.disabled is True


class TestLoadServerIps:
    def test_filters_disabled_servers(self, tmp_path, monkeypatch):
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.1", "name": "active"},
                {"ip": "10.0.0.2", "name": "off", "disabled": True},
                {"ip": "10.0.0.3", "name": "also-active", "disabled": False},
            ]
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

        servers = load_server_ips()
        assert len(servers) == 2
        assert servers[0].ip == "10.0.0.1"
        assert servers[1].ip == "10.0.0.3"

    def test_all_disabled(self, tmp_path, monkeypatch):
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.1", "disabled": True},
            ]
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

        servers = load_server_ips()
        assert len(servers) == 0

    def test_none_disabled(self, tmp_path, monkeypatch):
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.1"},
                {"ip": "10.0.0.2"},
            ]
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

        servers = load_server_ips()
        assert len(servers) == 2


class TestGetServers:
    """Tests for the lazy-load get_servers() accessor."""

    def test_get_servers_returns_servers_when_config_exists(self, tmp_path, monkeypatch):
        """get_servers() should return the server list when servers.json exists."""
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.1", "name": "server1"},
                {"ip": "10.0.0.2", "name": "server2"},
            ]
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
        # Reset the cached _SERVERS so get_servers() re-loads
        monkeypatch.setattr(config_module, "_SERVERS", None)

        servers = get_servers()
        assert len(servers) == 2
        assert servers[0].ip == "10.0.0.1"
        assert servers[1].ip == "10.0.0.2"

    def test_get_servers_raises_when_no_config(self, tmp_path, monkeypatch):
        """get_servers() should raise FileNotFoundError when no config file exists."""
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(config_module, "_SERVERS", None)

        with pytest.raises(FileNotFoundError):
            get_servers()

    def test_get_servers_caches_result(self, tmp_path, monkeypatch):
        """get_servers() should cache the result after the first call."""
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.1", "name": "server1"},
            ]
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(config_module, "_SERVERS", None)

        first_call = get_servers()
        second_call = get_servers()
        # Should return the same object (cached)
        assert first_call is second_call

    def test_get_servers_falls_back_to_default(self, tmp_path, monkeypatch):
        """get_servers() should fall back to servers.default.json when servers.json is missing."""
        config = {
            "valkey_servers": [
                {"ip": "10.0.0.5", "name": "default-server"},
            ]
        }
        default_file = tmp_path / "servers.default.json"
        default_file.write_text(json.dumps(config))
        monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(config_module, "_SERVERS", None)

        servers = get_servers()
        assert len(servers) == 1
        assert servers[0].ip == "10.0.0.5"
