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
    
    def __init__(self, server_ip: str, repo: str, specifier: str, args: list) -> str:
        self.repo = repo
        self.specifier = specifier
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

    def should_pull_after_checkout(self, specifier) -> bool:
        repo_path = Server.get_repo_binary_path(self.repo)
        command= f'cd {repo_path}; git rev-parse --symbolic-full-name {specifier} --'
        result = run_server_command(command.split()).strip()
        if result == '':
            raise ValueError(f"{specifier} is an invalid specifier in {self.repo} (empty result)")
        if result == '--':
            return False # a specific commit by hash, unstable~2, etc.
        if result.startswith('refs/heads/'):
            return True # local branch
        if result.startswith('refs/remotes/'):
            return True # remote reference
        if result.startswith('refs/tags/'):
            return False # tag
        raise ValueError(f"{specifier} is an unhandled type of specifier in {self.repo} (got {repr(result)})")

    def __ensure_build_cached(self) -> str:
        repo_path = Server.get_repo_binary_path(self.repo)
        run_server_command(f'cd {repo_path}; git reset --hard && git fetch && git checkout {self.specifier}'.split())
        if self.should_pull_after_checkout(self.specifier):
            # ensure branch/etc is up to date with remote
            run_server_command(f'cd {repo_path}; git pull'.split())
        self.commit_hash = self.__get_current_commit_hash()
        
        cached_build_path = Server.__get_cached_build_path(self.repo, self.commit_hash)
        cached_binary_path = os.path.join(cached_build_path, valkey_binary)

        if not Server.is_build_cached(self.repo, self.commit_hash):
            logger.info(f"building {self.specifier}... (no cached build)")

            run_server_command(f'cd {repo_path}; make distclean && make -j USE_FAST_FLOAT=yes'.split())
            run_server_command(f'mkdir -p {cached_build_path}'.split())
            build_binary = os.path.join(repo_path, valkey_binary)
            run_server_command(['cp', build_binary, cached_binary_path])

        return cached_binary_path

    def __get_current_commit_hash(self) -> str:
        repo_path = Server.get_repo_binary_path(self.repo)
        return run_server_command(f'cd {repo_path}; git rev-parse HEAD'.split()).strip()