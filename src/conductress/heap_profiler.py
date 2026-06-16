"""Parse jemalloc heap profiles to categorize memory allocations by struct/purpose.

Uses addr2line to resolve allocation stacks from jemalloc heap dumps,
then maps each allocation to a logical category based on function name patterns.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Protocol


class HasRunHostCommand(Protocol):
    """Protocol for objects that can run commands on a remote host."""

    async def run_host_command(self, cmd: str, check: bool = ...) -> tuple[str, str]: ...


logger = logging.getLogger(__name__)

# Categories ordered by specificity (most specific patterns first).
# Each allocation is assigned to the FIRST matching category found
# when walking the stack from leaf (allocator) toward root (main).
#
# ⚠️  ORDER MATTERS — first match wins. Ordering rationale:
#   1. Type-specific structs (skiplist, listpack, hash_entry) — most specific,
#      their patterns would otherwise match broader categories below.
#   2. robj_embval — "createEmbeddedString" would match robj's "createString" substring.
#   3. sds — common allocator, must come before hashtable/dict because SDS allocations
#      often appear in stacks that also contain hashtable/dict callers above them.
#   4. hashtable — "resize"/"bucket" are generic-sounding; hash_entry must come first
#      because entryConstruct calls hashtableInsertAtPosition (which contains "hashtable").
#   5. dict — legacy dict; "createEntry" is a substring hazard (hash_entry's
#      "entryCreate" is more specific and checked earlier).
#   6. robj — broad catch-all for object creation; checked late so specific paths above
#      (robj_embval, hash_entry) aren't shadowed.
#   7. server_infra — startup allocations, least likely to collide.
#
# Special handling for robj allocations (see _categorize_stack):
#   - robj_embkey: robj match + objectSetKey/objectSetKeyAndExpire in caller stack
CATEGORIES: list[tuple[str, list[str]]] = [
    # Type-specific data structures — most specific patterns
    ("skiplist", ["zslCreateNode", "zslInsert", "zslUpdateScore", "zslDelete", "zslCreate"]),
    ("listpack", ["lpNew", "lpAppend", "lpInsert", "lpPrepend", "listpack"]),
    # hash_entry before hashtable: entryConstruct/hashTypeCreateEntry call into hashtable funcs
    ("hash_entry", ["entryConstruct", "entryCreate", "entryWrite", "hashTypeCreateEntry"]),
    # robj_embval before robj: "createEmbeddedString" contains "createString"
    ("robj_embval", ["createEmbeddedString"]),
    # sds before hashtable/dict: SDS allocs appear in stacks containing dict/hashtable callers
    ("sds", ["sdsnewlen", "sdsdup", "sdsMakeRoom", "_sdsnewlen", "sdscatlen", "sdscat"]),
    # hashtable: bucket array infrastructure only (entries captured by hash_entry above)
    # NOTE: "rehash" removed — new hashtable rehash is caught by "bucket"/"resize",
    # and legacy dict rehash (rehashEntriesInBucketAtIndex) should fall to "dict".
    # NOTE: "kvstore" removed — kvstore is a wrapper layer above both dict and hashtable.
    # The only case where it was the first match (kvstoreCreate in initServer) is server_infra.
    (
        "hashtable",
        [
            "hashtable",
            "bucket",
            "resize",
            "Chained",
            "hashTypeEntry",
        ],
    ),
    # dict: legacy dict (pre-hashtable migration); "createEntry" shadowed by hash_entry above
    (
        "dict",
        [
            "dictCreate",
            "dictEntry",
            "createEntry",
            "createEmbeddedEntry",
            "dictExpand",
            "dictResize",
            "dictRehash",
            "dictAdd",
            "dictInsert",
            "dictSetVal",
            "dictSetKey",
        ],
    ),
    # robj: broad object creation — after specific paths (robj_embval, hash_entry)
    ("robj", ["createObject", "createString", "createRaw", "createZset", "createUnembedded"]),
    # server_infra: startup/init allocations — least collision risk, checked last
    (
        "server_infra",
        ["initServer", "aeCreate", "createShared", "hdr_init", "bioInit", "initServerConfig", "ACL", "clusterInit"],
    ),
]

# All category names in stacking order (bottom → top in stacked area chart).
# Stable baseline at bottom, volatile categories at top.
# Adjacent categories trade bytes (e.g. sds ↔ robj_emb*, dict ↔ hashtable).
CATEGORY_NAMES: list[str] = [
    "server_infra",
    "other",
    "robj",
    "robj_embkey",
    "robj_embval",
    "sds",
    "dict",
    "hashtable",
    "hash_entry",
    "listpack",
    "skiplist",
]

# Patterns indicating the robj allocation has an embedded key (checked across full stack)
_EMBEDDED_KEY_MARKERS = ["objectSetKey", "objectSetKeyAndExpire"]

# Function name patterns that indicate jemalloc/allocator internals (skip these frames)
_JEMALLOC_PATTERNS = [
    "je_",
    "prof_",
    "imalloc",
    "ifree",
    "arena_",
    "tcache_",
    "large_",
    "extent_",
    "base_",
    "rtree_",
    "mutex_",
    "atomic_",
    "spin_",
    "fls_",
    "malloc_",
    "calloc_",
    "realloc_",
    "sz_",
    "bitmap_",
    "bin_",
    "slab_",
    "pac_",
    "decay_",
    "hpa_",
    "sec_",
    "san_",
    "tsd_",
    "ctl_",
    "stats_",
    "prof_backtrace",
    "prof_alloc",
    "ztrymalloc_usable",
    "ztrycalloc_usable",
    "ztryrealloc_usable",
    "zmalloc",
    "zcalloc",
    "zrealloc",
    "zfree",
    "valkey_malloc",
    "valkey_calloc",
    "valkey_realloc",
]

_SKIP_FUNCS = {"??", "_start", "main", "__libc_start_call_main", "__libc_start_main_alias_2"}

# jemalloc env var for profiling (Valkey uses je_ prefix)
JEMALLOC_PROF_ENV = 'JE_MALLOC_CONF="prof:true,lg_prof_sample:0,prof_final:true,prof_prefix:/tmp/valkey-heap"'

# Make args to enable jemalloc profiling (Valkey-specific, includes USE_FAST_FLOAT)
JEMALLOC_PROF_MAKE_ARGS = 'USE_FAST_FLOAT=yes JEMALLOC_CONFIGURE_OPTS="--enable-prof"'

# Engine-agnostic jemalloc profiling flag (combine with engine.make_args)
JEMALLOC_PROF_CONFIGURE_OPTS = 'JEMALLOC_CONFIGURE_OPTS="--enable-prof"'

# Heap dump location prefix (matches prof_prefix above)
HEAP_DUMP_PREFIX = "/tmp/valkey-heap"


def _is_jemalloc_frame(func: str) -> bool:
    """Check if a function is a jemalloc/allocator internal to skip."""
    if func in _SKIP_FUNCS:
        return True
    return any(pat in func for pat in _JEMALLOC_PATTERNS)


def _categorize_stack(funcs: list[str]) -> str:
    """Walk resolved function names, skip allocator frames, return first matching category.

    Special case: if the first match is 'robj' but the full stack contains an
    embedded-key marker (objectSetKey/objectSetKeyAndExpire), reclassify as 'robj_embkey'.
    """
    first_match: Optional[str] = None
    for func in funcs:
        if _is_jemalloc_frame(func):
            continue
        if first_match is None:
            for cat_name, patterns in CATEGORIES:
                if any(pat in func for pat in patterns):
                    first_match = cat_name
                    break
            if first_match and first_match != "robj":
                return first_match
            # If robj matched, continue scanning for robj_embkey markers
        elif first_match == "robj":
            if any(marker in func for marker in _EMBEDDED_KEY_MARKERS):
                return "robj_embkey"
    return first_match or "other"


def _parse_stacks(heap_content: str) -> list[tuple[list[str], int]]:
    """Parse a jemalloc heap_v2 file into (address_list, byte_count) tuples."""
    stacks: list[tuple[list[str], int]] = []
    current_addrs: Optional[list[str]] = None

    for line in heap_content.split("\n"):
        if line.startswith("@"):
            current_addrs = line.split()[1:]
        elif line.strip().startswith("t*:") and current_addrs:
            m = re.match(r"\s*t\*:\s*(\d+):\s*(\d+)", line)
            if m:
                stacks.append((current_addrs, int(m.group(2))))
                current_addrs = None

    return stacks


def _resolve_addresses(sorted_addrs: list[str], addr2line_output: str) -> dict[str, str]:
    """Parse addr2line output (alternating function\\nfile:line) into {addr: func} map."""
    resolved: dict[str, str] = {}
    lines = addr2line_output.strip().split("\n")
    for i in range(0, min(len(lines) - 1, len(sorted_addrs) * 2), 2):
        if i // 2 < len(sorted_addrs):
            resolved[sorted_addrs[i // 2]] = lines[i]
    return resolved


def categorize_heap_dump(heap_content: str, addr2line_output: str) -> Optional[dict[str, int]]:
    """Parse a jemalloc heap dump and categorize allocations.

    Args:
        heap_content: Raw content of the .heap file.
        addr2line_output: Output of `addr2line -f -e binary addr1 addr2 ...`
                          for all unique addresses in the heap file.

    Returns:
        Dict mapping category name -> total bytes, or None on parse failure.
    """
    stacks = _parse_stacks(heap_content)
    if not stacks:
        logger.warning("No allocation stacks found in heap dump")
        return None

    # Collect and sort unique addresses
    all_addrs: set[str] = set()
    for addrs, _ in stacks:
        all_addrs.update(addrs)
    sorted_addrs = sorted(all_addrs)

    resolved = _resolve_addresses(sorted_addrs, addr2line_output)

    # Categorize each stack
    category_bytes: dict[str, int] = defaultdict(int)
    for addrs, bytes_val in stacks:
        funcs = [resolved.get(addr, "??") for addr in addrs]
        cat = _categorize_stack(funcs)
        category_bytes[cat] += bytes_val

    return {cat: category_bytes.get(cat, 0) for cat in CATEGORY_NAMES}


@dataclass
class HeapProfileResult:
    """Result of heap profiling: breakdown + retained stacks for re-categorization."""

    breakdown: dict[str, float]  # category -> bytes/key
    raw_stacks: list[list]  # [[func_names...], bytes] pairs for re-categorization

    def to_serializable_stacks(self) -> list[list]:
        """Return stacks in JSON-serializable format: [[funcs...], bytes]."""
        return self.raw_stacks


def recategorize_from_stacks(raw_stacks: list[list], num_keys: int) -> dict[str, float]:
    """Re-categorize previously saved stacks using current CATEGORIES patterns.

    Args:
        raw_stacks: List of [func_names_list, bytes] pairs from HeapProfileResult.
        num_keys: Number of keys for per-key normalization.

    Returns:
        Dict mapping category -> bytes/key with current category definitions.
    """
    category_bytes: dict[str, int] = defaultdict(int)
    for entry in raw_stacks:
        funcs, bytes_val = entry
        cat = _categorize_stack(funcs)
        category_bytes[cat] += bytes_val
    return {cat: round(category_bytes.get(cat, 0) / num_keys, 2) for cat in CATEGORY_NAMES}


async def collect_heap_profile(
    ssh_host: HasRunHostCommand, binary_path: str, num_keys: int
) -> Optional[HeapProfileResult]:
    """Collect and parse heap profile from a remote host after server shutdown.

    Finds the most recent heap dump, resolves addresses via addr2line,
    categorizes allocations, and returns breakdown + retained stacks.

    Args:
        ssh_host: Remote host with run_host_command(cmd) method.
        binary_path: Path to the valkey-server binary on the remote host.
        num_keys: Number of keys in the database (for per-key normalization).

    Returns:
        HeapProfileResult with breakdown and raw stacks, or None if profiling failed.
    """
    # Find the heap dump file
    heap_path, _ = await ssh_host.run_host_command(f"ls -t {HEAP_DUMP_PREFIX}*.heap 2>/dev/null | head -1", check=False)
    heap_path = heap_path.strip()
    if not heap_path:
        logger.warning("No heap dump found at %s*.heap", HEAP_DUMP_PREFIX)
        return None

    # Read the heap file
    heap_content, _ = await ssh_host.run_host_command(f"cat {heap_path}", check=False)
    if not heap_content or "heap_v2" not in heap_content:
        logger.warning("Invalid heap dump content")
        return None

    # Extract unique addresses from parsed stacks (single parse, no duplication)
    stacks = _parse_stacks(heap_content)
    if not stacks:
        logger.warning("No stacks in heap dump")
        return None

    all_addrs: set[str] = set()
    for addrs, _ in stacks:
        all_addrs.update(addrs)
    sorted_addrs = sorted(all_addrs)

    # Resolve addresses via addr2line on the remote host
    addr_str = " ".join(sorted_addrs)
    addr2line_out, _ = await ssh_host.run_host_command(f"addr2line -f -e {binary_path} {addr_str}", check=False)
    if not addr2line_out:
        logger.warning("addr2line produced no output")
        return None

    # Resolve addresses to function names
    resolved = _resolve_addresses(sorted_addrs, addr2line_out)

    # Build resolved stacks (for retention) and categorize in one pass
    resolved_stacks: list[list] = []
    category_bytes: dict[str, int] = defaultdict(int)
    for addrs, bytes_val in stacks:
        funcs = [resolved.get(addr, "??") for addr in addrs]
        resolved_stacks.append([funcs, bytes_val])
        cat = _categorize_stack(funcs)
        category_bytes[cat] += bytes_val

    raw_breakdown = {cat: category_bytes.get(cat, 0) for cat in CATEGORY_NAMES}

    # Convert to per-key values
    if num_keys <= 0:
        logger.warning("num_keys=%d, cannot compute per-key breakdown", num_keys)
        return None

    per_key = {cat: round(raw_bytes / num_keys, 2) for cat, raw_bytes in raw_breakdown.items()}

    total_bytes = sum(raw_breakdown.values())
    logger.info(
        "Heap profile: %d stacks, %d bytes, %d keys -> %.1f B/key total",
        len(stacks),
        total_bytes,
        num_keys,
        total_bytes / num_keys,
    )
    return HeapProfileResult(breakdown=per_key, raw_stacks=resolved_stacks)


async def cleanup_heap_dumps(ssh_host: HasRunHostCommand) -> None:
    """Remove heap dump files from the remote host."""
    await ssh_host.run_host_command(f"rm -f {HEAP_DUMP_PREFIX}*.heap", check=False)
