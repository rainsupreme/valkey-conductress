import json
import pytest
from src.config import ServerInfo, load_server_ips


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
