"""Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

import datetime
import logging
from dataclasses import dataclass

from src.server import Server
from src.task_queue import BaseTaskData, BaseTaskRunner
from src.utility import (
    MILLION,
    HumanByte,
    HumanNumber,
    print_pretty_header,
    record_task_result,
)

logger = logging.getLogger(__name__)


@dataclass
class MemTaskData(BaseTaskData):
    """data class for memory efficiency task"""

    type: str
    val_size: int
    has_expire: bool

    def short_description(self) -> str:
        return f"{HumanByte.to_human(self.val_size)} {self.type}" + (
            " with expiration" if self.has_expire else ""
        )

    def prepare_task_runner(self, server_ips: list[str]) -> "TestMem":
        """Return the task runner for this task."""
        return TestMem(
            server_ips[0],
            self.source,
            self.specifier,
            self.type,
            self.val_size,
            self.has_expire,
        )


class TestMem(BaseTaskRunner):
    """Tests memory efficiency of the specified type. Result is bytes of overhead per item."""

    def __init__(self, server_ip: str, repo: str, specifier: str, test: str, val_size: int, has_expire: bool):
        self.title = f"{test} memory efficiency, {repo}:{specifier}, has_expire={has_expire}"
        print_pretty_header(self.title)

        # settings
        self.server_ip = server_ip
        self.repo = repo
        self.specifier = specifier
        self.test = test
        self.val_size = val_size
        self.has_expire = has_expire

    def run(self):
        """Test memory efficiency for a single item size."""
        threads = 9
        valkey = Server.with_build(self.server_ip, self.repo, self.specifier, threads, [])
        commit_hash = valkey.get_build_hash()

        before_usage = valkey.used_memory()

        count = 5 * MILLION
        print(
            f"loading {HumanNumber.to_human(count)} {HumanByte.to_human(self.val_size)} {self.test} elements"
        )

        valkey.run_valkey_command_over_keyspace(count, f"-d {self.val_size} -t {self.test}")
        if self.has_expire:
            if self.test != "set":
                logger.error("Expiration is only supported for sets, skipping expiration test.")
            else:
                valkey.run_valkey_command_over_keyspace(count, f"EXPIRE key:__rand_int__ {7*24*60*60}")

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
        per_key_overhead = per_key - self.val_size - keysize
        print(
            f"done testing {HumanByte.to_human(self.val_size)} {self.test} elements: "
            f"{per_key_overhead:.2f}B overhead per key"
        )

        result = {
            "count": count,
            "has_expire": self.has_expire,
            "total_usage": total_usage,
            "key_size": keysize,
            "val_size": self.val_size,
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
