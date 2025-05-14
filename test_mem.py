"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import datetime
import logging

from config import CONDUCTRESS_OUTPUT
from server import Server
from utility import MILLION, human, human_byte, print_pretty_header

logger = logging.getLogger(__name__)


class TestMem:
    """Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

    def __init__(self, server_ip: str, repo: str, specifier: str, test: str, has_expire: bool):
        self.title = f"{test} memory efficiency, {repo}:{specifier}, has_expire={has_expire}"
        print_pretty_header(self.title)

        # settings
        self.server_ip = server_ip
        self.repo = repo
        self.specifier = specifier
        self.test = test
        self.has_expire = has_expire

    def test_single_size(self, valsize: int) -> float:
        """Test memory efficiency for a single item size."""
        threads = 9
        valkey = Server.with_build(self.server_ip, self.repo, self.specifier, threads, [])
        commit_hash = valkey.get_build_hash()

        before_usage = valkey.used_memory()

        count = 5 * MILLION
        print(f"loading {human(count, 1)} {human_byte(valsize)} {self.test} elements")
        valkey.fill_keyspace(valsize, count, self.test)
        if self.has_expire:
            print("expiring elements")
            valkey.expire_keyspace(count)

        after_usage = valkey.used_memory()
        (item_count, expire_count) = valkey.count_items_expires()
        assert item_count == count
        print(item_count, expire_count)
        if self.has_expire:
            assert expire_count == count
        else:
            assert expire_count == 0

        # output result
        keysize = 16
        total_usage = after_usage - before_usage
        per_key = float(total_usage) / count
        per_key_overhead = per_key - valsize - keysize
        print(
            f"done testing {human_byte(valsize)} {self.test} elements: "
            f"{per_key_overhead:.2f} overhead per key"
        )

        result = {
            "method": "mem",
            "repo": self.repo,
            "specifier": self.specifier,
            "commit_hash": commit_hash,
            "test": self.test,
            "count": count,
            "has_expire": self.has_expire,
            "endtime": datetime.datetime.now(),
            "size": valsize,
            "per_key_size": per_key,
            "per_key_overhead": per_key_overhead,
        }

        result_fields = (
            "method repo specifier commit_hash test count has_expire "
            "endtime size per_key_size per_key_overhead"
        )
        result_list = [f"{field}:{result[field]}" for field in result_fields.split()]
        result_string = "\t".join(result_list) + "\n"
        with open(CONDUCTRESS_OUTPUT, "a", encoding="utf-8") as f:
            f.write(result_string)
        return per_key_overhead
