import logging
import os

from config import valkey_binary
from utility import run_command, run_server_command, check_server_file_exists

logger = logging.getLogger(__name__)

class Server:
    @classmethod
    def __ensure_stopped(cls):
        run_server_command(['pkill', '-f', valkey_binary])

    @classmethod
    def __get_cached_build_path(cls, repo: str, hash_id: str) -> str:
        return os.path.join('~', 'build_cache', repo, hash_id)

    @classmethod
    def get_repo_binary_path(cls, repo: str) -> str:
        return os.path.join('~', repo, 'src')

    @classmethod
    def is_build_cached(cls, repo: str, hash_id: str) -> bool:
        return check_server_file_exists(os.path.join(Server.__get_cached_build_path(repo, hash_id), valkey_binary))
    
    def __init__(self, server_ip: str, repo: str, commit_id: str, args: list) -> str:
        self.repo = repo
        self.commit_id = commit_id
        self.args = args
        self.server_ip = server_ip

        Server.__ensure_stopped()
        cached_binary_path = self.__ensure_build_cached()
        command = [cached_binary_path, '--save', '--protected-mode', 'no', '--daemonize', 'yes'] + args
        run_server_command(command)

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
        return self.commit_hash

    def fill_keyspace(self, valsize: int, keyspace_size: int, test: str) -> None:
        run_command(f'./valkey-benchmark -h {self.server_ip} -d {valsize} --sequential -r {keyspace_size} -n {keyspace_size} -c 650 -P 4 --threads 50 -t {test} -q'.split())

    def expire_keyspace(self, keyspace_size: int) -> None:
        """Adds long expiry data to all keys in keyspace. No keys should actually expire in a test of any sane duration."""
        day = 60 * 60 * 24
        run_command(f'./valkey-benchmark -h {self.server_ip} --sequential -r {keyspace_size} -n {keyspace_size} -c 650 -P 4 --threads 50 -q EXPIRE key:__rand_int__ {7 * day}'.split())

    def __ensure_build_cached(self) -> str:
        repo_path = Server.get_repo_binary_path(self.repo)
        run_server_command(f'cd {repo_path}; git reset --hard && git fetch && git switch {self.commit_id} && git pull'.split())
        self.commit_hash = self.__get_current_commit_hash()
        
        cached_build_path = Server.__get_cached_build_path(self.repo, self.commit_hash)
        cached_binary_path = os.path.join(cached_build_path, valkey_binary)

        if not Server.is_build_cached(self.repo, self.commit_hash):
            logger.info(f"building {self.commit_id}... (no cached build)")

            run_server_command(f'cd {repo_path}; make distclean && make -j USE_FAST_FLOAT=yes'.split())
            run_server_command(f'mkdir -p {cached_build_path}'.split())
            build_binary = os.path.join(repo_path, valkey_binary)
            run_server_command(['cp', build_binary, cached_binary_path])

        return cached_binary_path

    def __get_current_commit_hash(self) -> str:
        repo_path = Server.get_repo_binary_path(self.repo)
        return run_server_command(f'cd {repo_path}; git rev-parse HEAD'.split()).strip()