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

MAX_SCORE = 2**53  # zadd scores drawn uniformly from [0, MAX_SCORE]
POPULATE_SEED = 42  # seeded for reproducibility
POPULATE_MODES = ("random", "sequential", "churn")


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


def _operations(command: str, count: int, key_size: int, value_size: int, field_size: int, mode: str):
    """Yield ('add', ...) / ('del', ...) op tuples to build a `count`-item structure
    of the given command type under the given populate mode.

    Pure and deterministic (seeded), so it is unit-testable without a server.

    Modes apply uniformly to every command type:
      sequential: items added in ascending index order. For zadd this means
                  ascending scores 0..count-1 -> dense B+tree packing (~100%).
      random:     items added in a shuffled order; for zadd, uniform-random
                  scores -> a fresh structure (~69% B+tree leaf occupancy).
      churn:      random fill to `count`, then `count` rounds of 50/50 random
                  add/delete -> a long-lived, churned structure. The size
                  random-walks and stays ~count (drift ~sqrt(count)); callers
                  must measure the actual cardinality, not assume `count`.

    For hashtable-backed types (set/sadd/hset) sequential vs random insertion
    order does not affect resident memory, but the modes are supported uniformly
    (they may matter for other structures/engines, and churn exercises the
    hashtable's shrink-on-delete behavior for all types).

    Add tuples carry the full payload; del tuples carry the item identifier:
      set:  ("add", key, value)      / ("del", key)
      sadd: ("add", member)          / ("del", member)
      hset: ("add", field, value)    / ("del", field)
      zadd: ("add", member, score)   / ("del", member)
    """
    if mode not in POPULATE_MODES:
        raise ValueError(f"Unknown populate_mode: {mode!r} (expected one of {POPULATE_MODES})")
    if command not in ("set", "sadd", "hset", "zadd"):
        raise ValueError(f"Unsupported command: {command}")

    rng = random.Random(POPULATE_SEED)
    value = "v" * value_size

    def add_op(idx: int):
        if command == "set":
            return ("add", _pad_to_size("key", idx, key_size), value)
        if command == "sadd":
            return ("add", _pad_to_size("m", idx, value_size))
        if command == "hset":
            return ("add", _pad_to_size("f", idx, field_size), value)
        # zadd: ascending scores in sequential mode, random otherwise
        score = idx if mode == "sequential" else rng.randint(0, MAX_SCORE)
        return ("add", _pad_to_size("m", idx, value_size), score)

    def del_op(idx: int):
        prefix, size = {
            "set": ("key", key_size),
            "sadd": ("m", value_size),
            "hset": ("f", field_size),
            "zadd": ("m", value_size),
        }[command]
        return ("del", _pad_to_size(prefix, idx, size))

    # Fresh fill.
    order = list(range(count))
    if mode != "sequential":
        rng.shuffle(order)
    for idx in order:
        yield add_op(idx)

    if mode != "churn":
        return

    # Churn: 50/50 add/delete random walk over `count` rounds. Deleting a live
    # item uses swap-remove for O(1). Size stays ~count.
    live = list(range(count))  # item indices currently present
    next_idx = count
    for _ in range(count):
        if live and rng.random() < 0.5:
            pos = rng.randrange(len(live))
            idx = live[pos]
            live[pos] = live[-1]
            live.pop()
            yield del_op(idx)
        else:
            yield add_op(next_idx)
            live.append(next_idx)
            next_idx += 1


def _execute_operations(pipe: valkey.client.Pipeline, command: str, ops) -> None:
    """Execute add/del op tuples (from _operations) against the pipeline, flushing
    every BATCH_SIZE commands. Collections use fixed keys (myset/myhash/myzset)."""
    pending = 0
    for op in ops:
        kind = op[0]
        if command == "set":
            if kind == "add":
                pipe.set(op[1], op[2])
            else:
                pipe.delete(op[1])
        elif command == "sadd":
            if kind == "add":
                pipe.sadd("myset", op[1])
            else:
                pipe.srem("myset", op[1])
        elif command == "hset":
            if kind == "add":
                pipe.hset("myhash", op[1], op[2])
            else:
                pipe.hdel("myhash", op[1])
        elif command == "zadd":
            if kind == "add":
                pipe.zadd("myzset", {op[1]: op[2]})
            else:
                pipe.zrem("myzset", op[1])
        else:
            raise ValueError(f"Unsupported command: {command}")
        pending += 1
        if pending % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()  # flush remainder


def populate(host: str, port: int, workload: "MemoryWorkload") -> None:
    """Populate keyspace for a memory workload with exact item sizes.

    Uses pipelining for performance (~5M items in <5s on localhost). The populate
    mode (random | sequential | churn) applies uniformly across all command types.
    """
    client = valkey.Valkey(host=host, port=port)
    pipe = client.pipeline(transaction=False)

    count = workload.item_count
    mode = workload.populate_mode
    logger.info(
        "Populating %s: %d items on %s:%d (mode=%s)",
        workload.label,
        count,
        host,
        port,
        mode,
    )

    # Churn changes which items exist, so the fixed range(count) EXPIRE pass
    # would no longer line up. Reject the combination rather than mis-report.
    if mode == "churn" and workload.has_expire:
        raise ValueError("populate_mode='churn' is incompatible with has_expire (churn changes which items exist)")

    ops = _operations(workload.command, count, workload.key_size, workload.value_size, workload.field_size, mode)
    _execute_operations(pipe, workload.command, ops)

    if workload.has_expire:
        _apply_expire(pipe, client, count, workload)

    client.close()
    logger.info("Population complete: %s", workload.label)


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
