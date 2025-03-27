# memory efficiency

import datetime
import logging

from config import conductress_log, conductress_output
from server import Server
from utility import *

logger = logging.getLogger(__name__)


class MemBench:
    def __init__(self, server_ip: str, repo: str, commit_id: str, test: str, has_expire: bool):
        """Tests memory efficiency for 5 million keys of the specified type. Returns bytes of overhead per item."""
        self.title = f'{test} memory efficiency, {repo}:{commit_id}, has_expire={has_expire}'
        pretty_header(self.title)

        # settings
        self.server_ip = server_ip
        self.repo = repo
        self.commit_id = commit_id
        self.test = test
        self.has_expire = has_expire

    def test_single_size(self, valsize: int) -> float:
        args = ['--io-threads', '9']
        valkey = Server(self.server_ip, self.repo, self.commit_id, args)
        commit_hash = valkey.get_commit_hash()

        before_usage = valkey.used_memory()

        count = 5 * million
        print(f'loading {human(count)} {human_byte(valsize)} {self.test} elements')
        valkey.fill_keyspace(valsize, count, self.test)
        if self.has_expire:
            print(f'expiring elements')
            valkey.expire_keyspace(count)

        after_usage = valkey.used_memory()
        (item_count, expire_count) = valkey.count_items_expires()
        assert(item_count == count)
        print(item_count, expire_count)
        if self.has_expire:
            assert(expire_count == count)
        else:
            assert(expire_count == 0)

        # output result
        keysize = 16
        total_usage = after_usage - before_usage
        per_key = float(total_usage) / count
        per_key_overhead = per_key - valsize - keysize
        print(f'done testing {human_byte(valsize)} {self.test} elements: {per_key_overhead:.2f} overhead per key')

        result = {
            'method': 'mem',
            'repo': self.repo,
            'commit': self.commit_id,
            'commit_hash': commit_hash,
            'test': self.test,
            'count': count,
            'has_expire': self.has_expire,
            'endtime': datetime.datetime.now(),
            'size': valsize,
            'per_key_size': per_key,
            'per_key_overhead': per_key_overhead,
        }

        result_fields = 'method repo commit commit_hash test count has_expire endtime size per_key_size per_key_overhead'.split()
        result_string = [f'{field}:{result[field]}' for field in result_fields]
        result_string = '\t'.join(result_string) + '\n'
        with open(conductress_output,'a') as f:
            f.write(result_string)
        return per_key_overhead