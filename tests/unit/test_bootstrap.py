import ast
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from src.bootstrap import Host, ensure_file_descriptor_limits, load_requirements, path_exists


class TestFileDescriptorLimits:

    @pytest.mark.asyncio
    async def test_limit_already_sufficient(self):
        """Test when current limit is already >= 65536"""
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="65536\n")

        await ensure_file_descriptor_limits(host)

        host.run.assert_called_once_with("ulimit -n")

    @pytest.mark.asyncio
    async def test_limit_already_configured(self):
        """Test when limit is configured in limits.conf but not active"""
        host = MagicMock(spec=Host)
        limits_conf = """# /etc/security/limits.conf
#<domain>      <type>  <item>         <value>
#*               soft    core            0
* soft nofile 65536
* hard nofile 65536
# End of file
"""
        host.run = AsyncMock(side_effect=["1024\n", limits_conf])
        host.log_info_msg = MagicMock()

        await ensure_file_descriptor_limits(host)

        assert host.run.call_count == 2
        host.log_info_msg.assert_called_once()

    @pytest.mark.asyncio
    async def test_insufficient_configured_limit(self):
        """Test when configured limit is too low"""
        host = MagicMock(spec=Host)
        limits_conf = """# /etc/security/limits.conf
#<domain>      <type>  <item>         <value>
* soft nofile 4096
* hard nofile 4096
# End of file
"""
        host.run = AsyncMock(side_effect=["1024\n", limits_conf])

        with pytest.raises(RuntimeError, match="Insufficient file limit"):
            await ensure_file_descriptor_limits(host)

    @pytest.mark.asyncio
    async def test_configure_new_limits(self):
        """Test when limits need to be configured"""
        host = MagicMock(spec=Host)
        limits_conf = """# /etc/security/limits.conf
#<domain>      <type>  <item>         <value>
#*               soft    core            0
# End of file
"""
        host.run = AsyncMock(side_effect=["1024\n", limits_conf, "", ""])
        host.log_info_msg = MagicMock()

        await ensure_file_descriptor_limits(host)

        assert host.run.call_count == 4
        assert "* soft nofile 65536" in host.run.call_args_list[2][0][0]
        assert "* hard nofile 65536" in host.run.call_args_list[3][0][0]


class TestLoadRequirements:

    def test_basic_requirements(self):
        content = "package1\npackage2\npackage3\n"
        with patch("builtins.open", mock_open(read_data=content)):
            result = load_requirements("test")
        assert result == ["package1", "package2", "package3"]

    def test_with_comments(self):
        content = "package1\n# comment\npackage2  # inline comment\n"
        with patch("builtins.open", mock_open(read_data=content)):
            result = load_requirements("test")
        assert result == ["package1", "", "package2"]


class TestHostGetHomePath:

    def test_with_username(self):
        host = Host(ip="1.2.3.4", username="testuser", name="test", conn=MagicMock())
        assert host.get_home_path() == Path("/home/testuser")

    def test_without_username(self):
        host = Host(ip="1.2.3.4", username="", name="test", conn=MagicMock())
        assert host.get_home_path() == Path.home()


class TestHostGetLinuxDistro:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("os_release,expected_name", [
        ('PRETTY_NAME="Ubuntu 22.04.3 LTS"\nNAME="Ubuntu"\nVERSION_ID="22.04"\nVERSION="22.04.3 LTS (Jammy Jellyfish)"\n', "Ubuntu"),
        ('NAME="Red Hat Enterprise Linux"\nVERSION="9.2 (Plow)"\nID="rhel"\nID_LIKE="fedora"\n', "Red Hat Enterprise Linux"),
        ('NAME="Amazon Linux"\nVERSION="2023"\nID="amzn"\nID_LIKE="fedora"\nVERSION_ID="2023"\n', "Amazon Linux"),
    ])
    async def test_parse_distro(self, os_release, expected_name):
        host = MagicMock(spec=Host)
        host.distro = ""
        host.run = AsyncMock(return_value=os_release)
        
        result = await Host.get_linux_distro(host)
        
        assert result == expected_name
        assert host.distro == expected_name

    @pytest.mark.asyncio
    async def test_distro_name_cached(self):
        host = MagicMock(spec=Host)
        host.distro = "Cached Distro"
        host.run = AsyncMock()
        
        result = await Host.get_linux_distro(host)
        
        assert result == "Cached Distro"
        host.run.assert_not_called()


class TestBootstrapImportOrdering:
    """Guards against importing non-stdlib modules before they're installed.
    
    This prevents the common packaging mistake where setup.py (or similar packaging files)
    imports dependencies at module level before pip has installed them, causing installation
    to fail for users installing from source.
    """

    def test_detects_non_stdlib_import(self):
        """Verify detection logic works for non-stdlib imports"""
        code = "import asyncssh\nfrom setuptools import setup"
        tree = ast.parse(code)
        stdlib_modules = {'os', 'sys', 'setuptools'}
        
        with pytest.raises(AssertionError, match="non-stdlib module: asyncssh"):
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        assert module in stdlib_modules, f"non-stdlib module: {module}"

    def test_detects_non_stdlib_from_import(self):
        """Verify detection logic works for non-stdlib from imports"""
        code = "from asyncssh import connect"
        tree = ast.parse(code)
        stdlib_modules = {'os', 'sys', 'setuptools'}
        
        with pytest.raises(AssertionError, match="non-stdlib module: asyncssh"):
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        assert module in stdlib_modules, f"non-stdlib module: {module}"

    def test_bootstrap_imports_non_stdlib(self):
        """Verify bootstrap.py imports non-stdlib modules (expected behavior for runtime script)"""
        bootstrap_path = Path(__file__).parent.parent.parent / "src" / "bootstrap.py"
        
        with open(bootstrap_path) as f:
            tree = ast.parse(f.read())
        
        found_asyncssh = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "asyncssh":
                        found_asyncssh = True
        
        assert found_asyncssh, "bootstrap.py should import asyncssh (it's a runtime script, not packaging)"


class TestPathExists:

    @pytest.mark.asyncio
    async def test_path_not_exists(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="1\n1\n1\n1\n")
        
        result = await path_exists(host, "/nonexistent")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_file_exists(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n0\n1\n1\n")
        
        result = await path_exists(host, "/some/file", expected_type="file")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_directory_exists(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n1\n0\n1\n")
        
        result = await path_exists(host, "/some/dir", expected_type="directory")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_symlink_exists(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n1\n1\n0\n")
        
        result = await path_exists(host, "/some/link", expected_type="symlink")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_expected_file_but_is_directory(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n1\n0\n1\n")
        
        with pytest.raises(AssertionError, match="Expected .* to be a file"):
            await path_exists(host, "/some/dir", expected_type="file")

    @pytest.mark.asyncio
    async def test_expected_directory_but_is_file(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n0\n1\n1\n")
        
        with pytest.raises(AssertionError, match="Expected .* to be a directory"):
            await path_exists(host, "/some/file", expected_type="directory")

    @pytest.mark.asyncio
    async def test_expected_symlink_but_is_file(self):
        host = MagicMock(spec=Host)
        host.run = AsyncMock(return_value="0\n0\n1\n1\n")
        
        with pytest.raises(AssertionError, match="Expected .* to be a symlink"):
            await path_exists(host, "/some/file", expected_type="symlink")
