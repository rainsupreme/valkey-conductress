"""CPU profile categorization for flamegraph stacks.

Categorizes collapsed perf stacks by leaf function into functional categories,
similar to how heap_profiler.py categorizes memory allocations.
"""

import re

CPU_CATEGORIES_MAIN = [
    (
        "hash_lookup",
        r"hashtable|findBucket|siphash|dictHash|hashtableIncrementalFind|"
        r"hashtableObjectGetKey|dictSdsKeyCompare|addKeysToIncrFindBatch",
    ),
    (
        "command_parse",
        r"processInput|processCommand|getCommandFlags|lookupCommand|"
        r"processMulti|addCommandToBatch|processClientIO|handleParseResults|processClientsCommandsBatch",
    ),
    ("command_overhead", r"call$|ustime|zmalloc_used_memory|moduleFireCommand|hdr_record|afterCommand"),
    ("reply_build", r"addReply|_addReplyToBuffer|prepareClientToWrite|sdscat|commitDeferredReply"),
    ("string_ops", r"sds|memcpy|memmove|memcmp|strlen|stringObjectLen|objectGetVal"),
    ("networking_io", r"write|read|epoll|aeApi|connSocket|sendmsg|recvmsg|writev|__libc_write"),
    ("memory_alloc", r"zmalloc|zfree|je_|malloc|free|realloc"),
    ("commandlog", r"commandlog|slowlog"),
    ("key_access", r"getCommand|lookupKey|expireIfNeeded|dbFind"),
    ("cleanup", r"resetClient|freeClientArgv|tryOffload|commandProcessed"),
    ("acl_check", r"ACL"),
    ("synchronization", r"pthread_mutex|__aarch64_swp|__aarch64_cas|spmc|mutex"),
]

CPU_CATEGORIES_IO = [
    ("synchronization", r"pthread_mutex|__aarch64_swp|__aarch64_cas|spmc|mutex|IOThreadMain$"),
    ("networking_io", r"write|read|epoll|connSocket|sendmsg|writev|__libc_write|connSocketWrite"),
    ("memory_alloc", r"zmalloc|zfree|je_|malloc|free|realloc"),
    ("command_parse", r"processInput|parseCommand|processMulti|handleParseResults"),
    ("reply_build", r"addReply|_addReplyToBuffer|trackBufReferences|releaseBufReferences"),
]

# Stacks whose path goes through the event-loop poll wait or the IO-thread
# job-queue spin are idle/spin-wait, not useful work. They are matched against
# the FULL stack string (not just the leaf) before leaf categorization, since
# the leaf is usually a deep kernel frame. Representing idle as its own band is
# essential — IO threads spend most of their time here at low pipeline depth.
IDLE_PATTERN = re.compile(r"IOJobQueue_availableJobs|IOThreadPoll|aeApiPoll|epoll_pwait|epoll_wait")

# Ordered category names for dashboard metadata (idle + other always appended).
CPU_CATEGORY_NAMES_MAIN = [name for name, _ in CPU_CATEGORIES_MAIN] + ["idle", "other"]
CPU_CATEGORY_NAMES_IO = [name for name, _ in CPU_CATEGORIES_IO] + ["idle", "other"]


def categorize_cpu_stacks(
    collapsed_stacks: list[list],
    categories: list[tuple[str, str]],
) -> dict[str, float]:
    """Categorize collapsed stacks by leaf function into percentage breakdown.

    Args:
        collapsed_stacks: [[stack_string, sample_count], ...] pairs
        categories: [(name, regex_pattern), ...] ordered list, first match wins

    Returns:
        dict of category_name -> percentage (0-100), always sums to ~100.
        Unmatched stacks go to 'other'.
    """
    compiled = [(name, re.compile(pattern)) for name, pattern in categories]
    totals: dict[str, int] = {}
    grand_total = 0

    for entry in collapsed_stacks:
        stack_str, count = str(entry[0]), int(entry[1])
        grand_total += count

        # Idle/spin-wait is detected against the FULL stack (poll wait or
        # job-queue spin appears mid-stack; the leaf is a deep kernel frame).
        if IDLE_PATTERN.search(stack_str):
            totals["idle"] = totals.get("idle", 0) + count
            continue

        # Leaf function is the last semicolon-separated element
        leaf = stack_str.rsplit(";", 1)[-1] if ";" in stack_str else stack_str

        matched = False
        for name, pattern in compiled:
            if pattern.search(leaf):
                totals[name] = totals.get(name, 0) + count
                matched = True
                break
        if not matched:
            totals["other"] = totals.get("other", 0) + count

    if grand_total == 0:
        return {"other": 100.0}

    return {cat: (samples / grand_total) * 100.0 for cat, samples in totals.items()}
