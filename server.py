import logging
import os
import subprocess
from config import sshkeyfile
from utility import run_command, hash_local_file

logger = logging.getLogger(__name__)

valkey_binary = 'valkey-server'

class Server:
    manual_upload_source = 'manually_uploaded' # special value indicating a binary was uploaded

    def __init__(self, server_ip: str, binary_source: str, specifier: str, args: list) -> str:
        self.source = binary_source
        self.specifier = specifier
        self.args = args
        self.server_ip = server_ip

        self.__ensure_stopped()
        if self.source == Server.manual_upload_source:
            cached_binary_path = self.__ensure_binary_uploaded(self.specifier)
        else:
            cached_binary_path = self.__ensure_build_cached()
        command = [cached_binary_path, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
        self.run_command(command)

    def info(self, section: str) -> dict:
        result = run_command(f'./valkey-cli -h {self.server_ip} info {section}'.split())
        result = result.strip().split('\n')
        keypairs = {}
        for item in result:
            if ':' in item:
                (key, value) = item.strip().split(':')
                keypairs[key.strip()] = value.strip()
        return keypairs
    
    def used_memory(self) -> int:
        info = self.info('memory')
        return int(info['used_memory'])
    
    def count_items_expires(self) -> int:
        info = self.info('keyspace')
        keys = 0
        expires = 0
        for line in info.values():
            # 'keys=98331,expires=0,avg_ttl=0'
            (key, expire, avg_ttl) = [int(x.split('=')[1]) for x in line.split(',')]
            keys += key
            expires += expire
        return (keys, expires)
    
    def get_commit_hash(self):
        return self.hash

    def fill_keyspace(self, valsize: int, keyspace_size: int, test: str) -> None:
        load_type_for_test = {
            'set': 'set',
            'get': 'set',
            'sadd': 'sadd',
            'hset': 'hset',
            'zadd': 'zadd',
            'zrange': 'zadd',
        }
        test = load_type_for_test[test]
        run_command(f'./valkey-benchmark -h {self.server_ip} -d {valsize} --sequential -r {keyspace_size} -n {keyspace_size} -c 650 -P 4 --threads 50 -t {test} -q'.split())

    def expire_keyspace(self, keyspace_size: int) -> None:
        """Adds long expiry data to all keys in keyspace. No keys should actually expire in a test of any sane duration."""
        day = 60 * 60 * 24
        run_command(f'./valkey-benchmark -h {self.server_ip} --sequential -r {keyspace_size} -n {keyspace_size} -c 650 -P 4 --threads 50 -q EXPIRE key:__rand_int__ {7 * day}'.split())

    def check_file_exists(self, path: str) -> bool:
        command = f'[[ -f {path} ]] && echo 1 || echo 0;'.split()
        result = self.run_command(command)
        return result.strip() == '1'

    def run_command(self, command: list, check=True):
        return run_command(command, remote_ip=self.server_ip, check=check)

    def __get_cached_build_path(self) -> str:
        return os.path.join('~', 'build_cache', self.source, self.hash)

    def __get_source_binary_path(self) -> str:
        return os.path.join('~', self.source, 'src')
    
    def __ensure_stopped(self):
        self.run_command(['pkill', '-f', valkey_binary], check=False)

    def __is_binary_cached(self) -> bool:
        return self.check_file_exists(os.path.join(self.__get_cached_build_path(), valkey_binary))

    def __should_pull_after_checkout(self, specifier) -> bool:
        source_path = self.__get_source_binary_path(self.source)
        command= f'cd {source_path}; git rev-parse --symbolic-full-name {specifier} --'
        result = self.run_command(command.split()).strip()
        if result == '':
            raise ValueError(f"{specifier} is an invalid specifier in {self.source} (empty result)")
        if result == '--':
            return False # a specific commit by hash, unstable~2, etc.
        if result.startswith('refs/heads/'):
            return True # local branch
        if result.startswith('refs/remotes/'):
            return True # remote reference
        if result.startswith('refs/tags/'):
            return False # tag
        raise ValueError(f"{specifier} is an unhandled type of specifier in {self.source} (got {repr(result)})")

    def __ensure_build_cached(self) -> str:
        source_path = self.__get_source_binary_path(self.source)
        self.run_command(f'cd {source_path}; git reset --hard && git fetch && git checkout {self.specifier}'.split())
        if self.__should_pull_after_checkout(self.specifier):
            # ensure branch/etc is up to date with remote
            self.run_command(f'cd {source_path}; git pull'.split())
        self.hash = self.__get_current_commit_hash()
        
        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = os.path.join(cached_build_path, valkey_binary)

        if not self.__is_binary_cached():
            logger.info(f"building {self.specifier}... (no cached build)")

            self.run_command(f'cd {source_path}; make distclean && make -j USE_FAST_FLOAT=yes'.split())
            self.run_command(f'mkdir -p {cached_build_path}'.split())
            build_binary = os.path.join(source_path, valkey_binary)
            self.run_command(['cp', build_binary, cached_binary_path])

        return cached_binary_path

    def scp_file_from_server(self, serverSrc: str, localDest: str) -> None:
        command = ['scp', '-i', sshkeyfile, f'{self.server_ip}:{str(serverSrc)}', str(localDest)]
        subprocess.run(command, check=True, encoding='utf-8')

    def scp_file_to_server(self, localSrc: str, serverDest: str) -> None:
        command = ['scp', '-i', sshkeyfile, str(localSrc), f'{self.server_ip}:{str(serverDest)}']
        subprocess.run(command, check=True, encoding='utf-8')

    def __ensure_binary_uploaded(self, localPath) -> str:
        self.hash = hash_local_file(localPath)

        cached_build_path = self.__get_cached_build_path()
        cached_binary_path = os.path.join(cached_build_path, valkey_binary)

        if not self.__is_binary_cached():
            logger.info(f"copying {localPath} to server... (not cached)")

            self.run_command(f'mkdir -p {cached_build_path}'.split())
            self.scp_file_to_server(localPath, cached_binary_path)

        return cached_binary_path

    def __get_current_commit_hash(self) -> str:
        source_path = self.__get_source_binary_path(self.source)
        return self.run_command(f'cd {source_path}; git rev-parse HEAD'.split()).strip()