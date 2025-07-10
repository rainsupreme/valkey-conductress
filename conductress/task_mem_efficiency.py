"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import datetime
import logging

from .server import Server
from .utility import (
    MILLION,
    HumanByte,
    HumanNumber,
    print_pretty_header,
    record_task_result,
)

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

    def test_single_size(self, valsize: int):
        """Test memory efficiency for a single item size."""
        threads = 9
        valkey = Server.with_build(self.server_ip, self.repo, self.specifier, threads, [])
        commit_hash = valkey.get_build_hash()

        before_usage = valkey.used_memory()

        count = 5 * MILLION
        print(f"loading {HumanNumber.to_human(count)} {HumanByte.to_human(valsize)} {self.test} elements")
        valkey.run_valkey_command_over_keyspace(count, f"-d {valsize} -t {self.test}")
        if self.has_expire:
            print("expiring elements")
            valkey.run_valkey_command_over_keyspace(count, f"-t {self.test}")

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
            f"done testing {HumanByte.to_human(valsize)} {self.test} elements: "
            f"{per_key_overhead:.2f} overhead per key"
        )

        result = {
            "count": count,
            "has_expire": self.has_expire,
            "total_usage": total_usage,
            "key_size": keysize,
            "val_size": valsize,
            "per_key_size": per_key,
            "per_key_overhead": per_key_overhead,
        }
        record_task_result(
            f"mem-{self.test}",
            self.repo,
            self.specifier,
            commit_hash or "",
            per_key_overhead,
            datetime.datetime.now(),
            result,
        )
