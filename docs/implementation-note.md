# Mixed-client scaling: implementation notes

## Feature summary

Parameterized memtier_benchmark thread/connection counts and empirical
client-CPU measurement for `MixedTaskData`, enabling the authorized
400/1200/2400-connection client-count sweep to distinguish latency-bound
behavior from capacity ceilings.

## Design decisions

### 1. Taskset pinning for `--client-cpus`

When `benchmark_cpu_override` (CLI `--client-cpus`) is set, both the prefill
and measurement memtier commands are prefixed with `taskset -c <cpulist>`.
This makes the declared CPU allocation truthful: without `taskset`, memtier
threads float across all available cores and the result's CPU utilization
denominator would be meaningless.

**Composition with GNU time**: `/usr/bin/time -v` wraps the entire
`taskset -c ... memtier_benchmark ...` invocation.  time measures the process
tree including scheduling constraints, which is correct — it reports the
CPU-seconds that the pinned process actually consumed.

### 2. Capacity model: `capacity_cores` / `capacity_basis`

Two capacity models replace the previous `allocated_cores` field:

| Mode | `capacity_cores` | `capacity_basis` | Rationale |
|------|------------------|-------------------|-----------|
| **Unpinned** (no `--client-cpus`) | `memtier_threads` | `memtier_thread_count` | Each memtier worker thread runs one epoll loop consuming ≤1 core. Without pinning, the thread count is the practical throughput ceiling. |
| **Pinned** (`--client-cpus` set) | `min(memtier_threads, count(cpulist))` | `min(memtier_thread_count,taskset_cpulist)` | The lower of worker concurrency and the enforced cpuset is the hard CPU ceiling. |

`utilization = max(cores_busy_per_rep) / capacity_cores`.
`saturated = utilization >= 0.90` (shared `CLIENT_CPU_SATURATION_THRESHOLD`).

The term `allocated_cores` is deliberately avoided for the unpinned case:
memtier threads are OS threads freely scheduled across available cores, not
a core allocation.  `capacity_cores` is named to express "the maximum
throughput the client side can deliver" rather than implying physical pinning.

### 3. Empirical measurement via GNU `/usr/bin/time`

GNU time (`/usr/bin/time -v`) wraps each measurement-phase command and
reports user + system CPU-seconds vs wall-clock time.
`cores_busy = cpu_seconds / wall_seconds` gives the number of CPU cores
the memtier process tree kept busy during the measurement window.

- Probed once per run via `/usr/bin/time --version`.
- Graceful degradation: if absent, `measurement_method = "unavailable"`,
  `utilization`/`saturated` are omitted, and a `note` field explains.
- Prefill commands are NOT wrapped with time (prefill is not a measurement
  window and would pollute the signal).

### 4. Warmup is an active control

`warmup` is passed to memtier as `--warmup-period <seconds>` when greater
than zero. Memtier executes the same configured workload during this period
but excludes warmup operations from the reported benchmark statistics.

- The CLI exposes `--warmup` using the same human-duration syntax as
  `--duration` and defaults to the existing Conductress five-second warmup.
- `--warmup 0s` is the explicit no-warmup setting; in that case Conductress
  omits `--warmup-period` from the command entirely.
- Negative values are rejected in both CLI parsing and task validation.
- GNU time intentionally covers warmup plus measurement because client CPU
  saturation should be detected across the generator's full steady workload.

### 5. Upper bounds

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MAX_MEMTIER_THREADS` | 256 | memtier threads are OS threads with epoll loops; 256 is already extreme for hosts ≤192 vCPUs |
| `MAX_MEMTIER_CLIENTS` | 1000 | 256 × 1000 = 256K connections at max threads; beyond is operational hazard |
| `MIN_TOTAL_CONNECTIONS` | 1 | Structural guard; actual minimum is 400 (default 8×50) |
| Keyspace guard | `total_connections <= MIXED_KEYSPACE` | Prevents zero prefill requests per connection |

### 6. Backward compatibility

- `memtier_threads=0` and `memtier_clients=0` in task JSON mean "use legacy
  default (8 and 50 respectively)".  Old envelopes without these fields
  deserialize with 0 and produce identical behavior.
- `short_description()` only shows concurrency info when overridden, keeping
  the default case uncluttered.
- Golden fixture `MixedTaskData.json` was updated with the new fields at
  their default values; schema compat test passes.
