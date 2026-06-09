"""Memory workload population with exact size control via valkey-py.

Replaces valkey-benchmark for memory tests. Gives precise control over
key, value, member, and field sizes regardless of command type.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import valkey

if TYPE_CHECKING:
    from conductress.sweep.memory_coordinator import MemoryWorkload

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000  # commands per pipeline flush


def _pad_to_size(prefix: str, index: int, target_bytes: int) -> str:
    """Generate a string of exactly target_bytes using prefix:index:padding.

    Raises ValueError if target_bytes is too small to encode a unique index.
    """
    base = f"{prefix}:{index:07d}:"
    if len(base) > target_bytes:
        raise ValueError(
            f"target_bytes={target_bytes} is too small for unique items "
            f"(need at least {len(base)} bytes for prefix='{prefix}', index={index})"
        )
    return base + "x" * (target_bytes - len(base))


def populate(host: str, port: int, workload: "MemoryWorkload") -> None:
    """Populate keyspace for a memory workload with exact item sizes.

    Uses pipelining for performance (~5M items in <5s on localhost).
    """
    client = valkey.Valkey(host=host, port=port)
    pipe = client.pipeline(transaction=False)

    count = workload.item_count
    logger.info("Populating %s: %d items on %s:%d", workload.label, count, host, port)

    if workload.command == "set":
        _populate_set(pipe, count, workload.key_size, workload.value_size)
    elif workload.command == "zadd":
        _populate_zadd(pipe, count, workload.value_size)
    elif workload.command == "sadd":
        _populate_sadd(pipe, count, workload.value_size)
    elif workload.command == "hset":
        _populate_hset(pipe, count, workload.field_size, workload.value_size)
    else:
        raise ValueError(f"Unsupported command: {workload.command}")

    if workload.has_expire:
        _apply_expire(pipe, client, count, workload)

    client.close()
    logger.info("Population complete: %s", workload.label)


def _populate_set(pipe: valkey.client.Pipeline, count: int, key_size: int, value_size: int) -> None:
    """SET key:<seq> <value> — one key per item."""
    value = "v" * value_size
    for i in range(count):
        key = _pad_to_size("key", i, key_size)
        pipe.set(key, value)
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()  # flush remainder


def _populate_zadd(pipe: valkey.client.Pipeline, count: int, member_size: int) -> None:
    """ZADD myzset <random_int_score> <member> — single sorted set, random scores."""
    rng = random.Random(42)  # seeded for reproducibility
    max_score = 2**53
    for i in range(count):
        member = _pad_to_size("m", i, member_size)
        score = rng.randint(0, max_score)
        pipe.zadd("myzset", {member: score})
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()


def _populate_sadd(pipe: valkey.client.Pipeline, count: int, member_size: int) -> None:
    """SADD myset <member> — single set."""
    for i in range(count):
        member = _pad_to_size("m", i, member_size)
        pipe.sadd("myset", member)
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()


def _populate_hset(pipe: valkey.client.Pipeline, count: int, field_size: int, value_size: int) -> None:
    """HSET myhash <field> <value> — single hash."""
    value = "v" * value_size
    for i in range(count):
        field = _pad_to_size("f", i, field_size)
        pipe.hset("myhash", field, value)
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()


def _apply_expire(pipe: valkey.client.Pipeline, client: valkey.Valkey, count: int, workload: "MemoryWorkload") -> None:
    """Apply EXPIRE to all keys (only meaningful for SET workloads)."""
    from conductress.config import MEM_TEST_EXPIRE_SECONDS

    if workload.command != "set":
        logger.error("Expiration only supported for SET workloads, skipping for %s", workload.command)
        return
    for i in range(count):
        key = _pad_to_size("key", i, workload.key_size)
        pipe.expire(key, MEM_TEST_EXPIRE_SECONDS)
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()
