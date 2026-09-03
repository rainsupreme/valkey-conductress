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

### 7. Comparison dimensions and provenance

Mixed comparison groups include a visible
`w<seconds>-wa<applied>-t<threads>-c<clients>` variant. This prevents
400-, 1200-, and 2400-connection results from being merged into one
statistical sample bucket. `warmup_applied` also separates historical mixed
results that recorded `warmup=5` while the field was still a no-op from new
results where the warmup actually executes. Results preserve
`server_cpu_override` and `server_args` so server topology and experimental
gates remain reproducible.

### 8. Unsupported key-size control fails closed

Memtier's existing mixed path does not implement Conductress's exact-key-size
contract. Nonzero `key_size` values were previously recorded without changing
the generated workload. Both CLI and task deserialization now reject nonzero
values instead of presenting a fake control; zero remains backward compatible.

### 9. Measurement and scheduling details

- GNU time detection requires the parser-compatible `GNU time` signature;
  BusyBox variants degrade to unavailable telemetry rather than parse noise.
- Server perf counters and CPU profiles use delayed starts, exclude warmup,
  and record `perf_duration_seconds = duration` with
  `perf_warmup_included = false`. Client GNU-time CPU accounting remains
  warmup-inclusive for conservative generator-saturation detection.
- Prefill requests use ceiling division, avoiding partial keyspace coverage
  when connection count does not divide the three-million-key keyspace.
- Duration estimates model one request-bounded prefill and one
  warmup-plus-measurement phase per repetition.


### 10. Measurement-only perf counters and CPU profiling

Server perf counters (`perf stat`) and CPU flamegraph profiling (`perf record`)
now cover ONLY the scored `--test-time` interval for mixed tasks, matching
PerfTaskRunner semantics where perf starts after the warmup phase.

**Mechanism**: `ProfilingManager.perf_stat_start()` and `cpu_profile_start()`
accept a new `delay_seconds` parameter (default 0, backward compatible).
When non-zero:
- **perf stat**: the shell command is prefixed with `sleep <delay> && `,
  so the perf process sleeps through the warmup and only starts counting
  once the scored measurement begins. The sentinel file is created
  immediately (before the sleep), so `perf_stat_stop()` remains race-free.
- **CPU profile**: a cancellable `Event.wait(delay)` runs in the profiling
  thread before `perf record` launches. Failure cleanup can wake the thread
  immediately and prevent recording from starting.

**MixedTaskRunner changes**:
- `perf_stat_start(delay_seconds=float(self.warmup))` — armed immediately
  before the blocking memtier command; the delay covers the warmup phase.
- `cpu_profile_start(duration, delay_seconds=float(self.warmup))` — armed
  on the FINAL repetition only (same as PerfTaskRunner), with the same
  warmup delay.
- `perf_stat_stop()` called after memtier completes (scored phase is over).
- Per-rep perf stat collection and summing across reps (matching PerfTaskRunner).
- Result schema: `perf_duration_seconds = float(duration)` (scored only),
  `perf_warmup_included = False`, `perf_rep_count = <actual count>`.
- CPU profile stacks persisted as `cpu_stacks_main` and `cpu_stacks_io`
  (same schema keys as PerfTaskRunner).

**warmup=0 handling**: delay is `float(0)`, which produces no sleep prefix
for perf stat and no delayed Event wait for CPU profile — perf starts immediately,
identical to a zero-warmup PerfTaskRunner.

**Failure cleanup**: every exit path after collector arming reaches one
per-repetition `finally` block. Perf is stopped and joined according to its
lifecycle state; delayed or active CPU profiling is cancelled through the
exact process handle. No collector can outlive a failed repetition.

**Client GNU-time CPU measurement**: intentionally remains warmup-inclusive
(covers warmup + measurement). This is conservative for saturation detection:
if the client is saturated at all during the run, we want to know.
The distinction is documented in the result schema.

**Backward compatibility**: `delay_seconds=0` (the default) is a no-op at
all API layers: `ProfilingManager`, `Server`, and PerfTaskRunner callers
are completely unaffected. The `Server.perf_stat_start()` and
`Server.cpu_profile_start()` signatures add `delay_seconds` as a keyword
argument with default 0.


### 11. Lifecycle correctness and cancellation (round 5)

The per-repetition collector lifecycle was refactored so that ANY code path
after perf stat or CPU profile arming reaches a single `finally` block.
Previously, cleanup only ran when `run_host_command` raised (the memtier
invocation); exceptions during GNU-time parsing, RPS parsing, perf stop,
perf report, metric writing, or result handling could leave background
perf threads and perf-record subprocesses racing server shutdown.

**try/finally per-rep block**: Each repetition in `MixedTaskRunner.run()` now
has an inner `try/finally` that:
- Stops perf stat exactly once (idempotent; `perf_stat_stop` is safe to
  call when already stopped).
- Joins perf stat exactly once (`perf_stat_wait`).
- Cancels or joins the CPU profile exactly once.

On success, the `try` body stops, joins, reports, and marks perf consumed;
the `finally` block observes that state and does nothing. CPU profile collection
likewise marks the profile consumed. On failure anywhere inside the try body,
`finally` performs only the missing stop/join/cancel operations.

**`ProfilingManager.cpu_profile_cancel()` API**: New backward-compatible
cancellation method using `threading.Event`:
- The delay sleep was replaced with `Event.wait(timeout=delay_seconds)`.
  When `cpu_profile_cancel()` sets the event, the thread wakes from the
  wait and returns without launching `perf record`.
- If `perf record` has already started, `cancel()` terminates the exact
  `Popen` subprocess (no broad `pkill`), waits for termination, and joins the
  profile thread to completion before clearing lifecycle state.
- `cpu_profile_cancel()` is a no-op when no profile is active.
- `Server.cpu_profile_cancel()` delegates to `ProfilingManager`.

**Backward compatibility**: `cpu_profile_start(delay_seconds=0)` still
works identically — with no delay, `Event.wait` is never called, and
`Popen` replaces `subprocess.run` (both block until perf record finishes).
All existing callers (`PerfTaskRunner`, `BoundedInsertionTaskRunner`) are
unaffected; they never call `cpu_profile_cancel()`.

**Perf lifecycle state**: `perf_armed` and `perf_stopped` distinguish an
active collector, a stopped collector awaiting join/report, and a fully
consumed collector. Successful repetitions call stop and wait once; failure
cleanup retries stop only if the initial stop did not complete.
