# Dragonfly as a comparator engine: integration plan

Status: plan, not started. Written 2026-09-11 from a code read of `main` at `7740769`
and an engine survey (sources at the end), and revised when the engine model it
depends on landed (the two axes in `SweepEngine`, the `release-and-tip` scope,
and the Redis v3 series that use it). Intended to be picked up in a fresh
session; everything a fresh session needs is here. `config.py` and `--help` are
the source of truth for what exists today; this document records the design.

## Goal

Publish Dragonfly throughput, latency and memory series on the dashboard next to
Valkey and Redis, measured by the same Conductress cells on the same hosts, with
the parameter table published alongside. Valkey stays the subject; Dragonfly is a
reference line, as Redis is today.

## Non-goals

- Bisecting Dragonfly history. It is a tracked release, measured as each release ships, not
  a swept ref.
- Profiling Dragonfly internals. Same posture as Redis: aggregate numbers only.
- Cluster mode, persistence, replication, or any workload beyond the GET/SET/mixed
  cells the dashboard already runs.
- Making every engine-specific assumption in Conductress generic. Only the ones
  Dragonfly actually hits (listed below).

## Why Dragonfly and not the others

Dragonfly is shared-nothing thread-per-core with fibers, io_uring, mimalloc and
non-forking snapshots: the architecture the io-threads and thread-owned-clients
work is arguing about, shipped. Its only rigorous public numbers are its own
(2.4x Valkey at 16 vCPU, 4.5x at 48 vCPU, GCP, March 2025); no independent,
documented benchmark exists. Garnet (Microsoft, MIT, .NET) is the second most
informative and reuses most of this plan. KeyDB is unmaintained since 2023.
Redict is Redis 7.2.4. Kvrocks is RocksDB-on-disk and its maintainers say a
direct comparison "makes no sense".

Licence: BSL 1.1 with an additional-use grant that forbids competing in-memory
store products and hosted offerings. Nothing in the licence or the Terms of Use
restricts running or publishing benchmarks.

**Project rule.** Conductress is BSD-licensed and stays clear of BSL-licensed
source: no Dragonfly source is ever cloned, vendored, patched or compiled on a
runner or in this repository. Only the official prebuilt release binaries are
executed, and only their public output (throughput, latency percentiles, total
memory from `INFO`) is recorded. `profile_internals=False` applies as it does
for Redis, so no flamegraph, allocation breakdown or symbol data is collected.
If a release has no asset for a platform, that platform has no Dragonfly point
for that release; building from source is not a fallback.

## What Conductress assumes today, and where Dragonfly breaks it

Every engine is a Redis-lineage git tree. Three places encode that:

1. **Build.** `SweepEngine` (`src/conductress/config.py`) describes an engine on
   two axes: `provisioning` (how a binary comes to exist) and `scope` (how much
   history is measured). The only implemented provisioning is `built-from-git`:
   `binary_manager.py` checks out a commit from `source` (a `REPOSITORIES`
   entry) and runs `make distclean && cd src && make -j`, expecting
   `src/<binary_name>`. `prebuilt-release` is declared on the axis and raises
   `NotImplementedError` until the download path exists. Dragonfly is CMake,
   C++20, with vendored dependencies (helio, abseil, boost), and under the
   project rule above it is never built here at all.
2. **Launch.** `server.py` builds a Valkey command line: `--io-threads N`,
   `--server-cpulist`, `--bio-cpulist`, `--aof-rewrite-cpulist`,
   `--bgsave-cpulist`, `--save ""`, `--protected-mode no`, `--daemonize yes`,
   `--logfile`. Dragonfly accepts none of the cpulist flags, has no `--daemonize`,
   and names its thread count `--proactor_threads`.
3. **Series.** `sweep/coordinator.py` populates `merge_commits` from the ref;
   with `scope="history"` it bisects them, with `scope="release-and-tip"` it
   measures the latest release landmark and the tip at most once per
   `tip_interval_hours` (the Redis mode). `sweep/exporter.py` emits
   `series-{platform}-{workload}-*.json` keyed by commit with `metadata.engine`,
   `metadata.scope` and a per-point `sample` (`release`, `tip`, `history`);
   commit links come from the engine's `REPOSITORIES` entry
   (`engine_repo_slug`). A tracked release engine's "commits" are release tags
   that arrive monthly, each measured once and then re-measured over time; the
   re-measure is the one planner behaviour the current mode lacks, since it
   measures each revision once.

Two smaller ones:

4. `memory_coordinator.py` and the exporter read jemalloc statistics for the
   memory decomposition. Dragonfly uses mimalloc. `profile_internals=False`
   already skips the flamegraph and jemalloc breakdown for Redis; confirm nothing
   else parses `MEMORY STATS` or `INFO memory` fields that mimalloc lacks.
5. `Server.get_num_cpus` allocates `io_threads + 2` CPUs (main + io + two
   background). Dragonfly's model is N proactor threads and nothing else, so the
   mapping is `threads -> proactor_threads` with `threads` CPUs pinned via
   `taskset`, not `threads + 2`.

## Design

### Engine axes

`SweepEngine` already carries the two axes. Dragonfly fills the second value of
the provisioning axis and adds the fields that value needs, plus a launch
profile, rather than special cases inline:

```python
@dataclass
class SweepEngine:
    source: str
    ref: str                              # built-from-git: git ref to sweep
    binary_name: str
    floor_tag: Optional[str] = None
    make_args: str = DEFAULT_MAKE_ARGS
    heap_alloc_funcs: list[str] = field(default_factory=list)
    profile_internals: bool = True
    provisioning: str = "built-from-git"  # or "prebuilt-release"   (exists)
    scope: str = "history"                # or "release-and-tip"    (exists)
    tip_interval_hours: float = 24.0      #                          (exists)
    release_repo: Optional[str] = None    # prebuilt-release: GitHub owner/repo whose releases are tracked
    release_channel: str = "latest"       # prebuilt-release: "latest" (non-prerelease) or an exact tag to hold
    release_asset: Optional[str] = None   # prebuilt-release: asset name pattern with {arch}
    launch: str = "valkey"                # launch profile name (see below)
    thread_flag: str = "--io-threads"
    cpu_overhead: int = 2                 # extra CPUs beyond threads
```

Existing entries keep working with defaults. Dragonfly:

```python
SweepEngine(
    source="dragonfly", ref="", binary_name="dragonfly",
    provisioning="prebuilt-release", scope="release-and-tip",
    release_repo="dragonflydb/dragonfly", release_channel="latest",
    release_asset="dragonfly-{arch}.tar.gz",   # confirm exact names on the release page
    launch="dragonfly", thread_flag="--proactor_threads", cpu_overhead=0,
    profile_internals=False,
)
```

For a prebuilt-release engine the tip *is* the latest release, so
`release-and-tip` reads as: the release step once on appearance, and a drift
re-measure of the current release at most once per `tip_interval_hours`.

`release_channel="latest"` is the normal mode: the engine follows Dragonfly's
releases automatically (see "Release tracking"). Setting it to an exact tag holds
the pin, for a bisection by hand or to freeze a comparator during a report.

### Provisioning (binary_manager)

Add a `prebuilt-release` path beside the existing build path: resolve
`release_asset` for the host arch (`x86_64` / `aarch64`), download from
`https://github.com/dragonflydb/dragonfly/releases/download/{release}/...`,
verify the published checksum, unpack into the existing build cache keyed by
`(source, release, arch)` so the cache and `ensure_binary_cached` logic are
unchanged. The "commit hash" for a prebuilt release is the tag string. Remove
the `NotImplementedError` guard in `SweepEngine.__post_init__` in the same
change; the guard exists so the axis value cannot be declared before the path
that honours it.

There is no build-from-source path for this engine, by the project rule above.
A platform without an asset has no point for that release.

### Launch profiles (server.py)

Factor the Valkey command-line construction into a `valkey` profile and add a
`dragonfly` profile. Both receive `(binary, port, threads, cpus, logfile,
extra_args)` and return a shell command. Dragonfly, to be confirmed against
`dragonfly --helpfull` on the current release:

```
taskset -c {cpus} {binary} --port {port} --bind 0.0.0.0
  --proactor_threads {threads} --dbfilename "" --snapshot_cron ""
  --logtostderr=false --log_dir {logdir}
  --maxmemory {mem} --conn_use_incoming_cpu=false
```

Launched with `nohup ... &` since there is no daemonize. Readiness is the
existing `PING` probe. Shutdown is the existing `SHUTDOWN NOSAVE`, which
Dragonfly supports. Check whether `--ulimit memlock` matters bare-metal; the
official Docker examples pass `--ulimit memlock=-1`.

Thread pinning: Dragonfly has `--proactor_affinity_mode`; with `taskset` on the
whole process and `proactor_threads == len(cpus)` the placement is equivalent to
the Valkey cpulist flags for benchmarking purposes.

### Release tracking (auto-bump)

Dragonfly ships a tagged release with prebuilt binaries roughly monthly (nine in
the first eight months of 2026; feature releases every six to eight weeks, patch
releases between, sometimes two days apart). The engine follows them without a
human in the loop:

- On each scheduling round the coordinator resolves `release_channel`. For
  `"latest"` it calls `GET /repos/{release_repo}/releases/latest` (unauthenticated
  is fine at this rate; the runner's existing GitHub token if one is configured),
  which excludes pre-releases and drafts by definition. The result is cached for
  an hour so the fleet does not poll GitHub per cell.
- A new tag becomes a new entry in the tracked engine's commit list, exactly as a
  new merge commit does for Valkey. The binary is provisioned into the build cache
  under the new tag on first use; the old tag's binary stays cached so a hold-back
  or a manual comparison needs no re-download.
- Every release is measured once per platform when it appears, then re-measured
  on the drift-canary cadence while it is current. The series therefore has two
  kinds of points: a release step and a drift sample, tagged in `metadata` so the
  dashboard can draw the step and the band differently.
- Hold-back: `release_channel="v1.40.2"` freezes the engine on that tag. Use it
  when a release breaks a launch flag or the parser, and file the follow-up. The
  runner logs a warning on every round while a newer release exists so a hold
  does not silently become permanent.
- Failure mode to design for: a release whose assets are missing for one arch
  (arm64 has historically lagged). Resolution fails for that platform only; that
  platform keeps measuring the previous tag and the status page shows the
  mismatch. Never let one platform's asset gap block the others.

This means a Dragonfly regression shows up as a step down that persists until
their next release. That is what a user downloading Dragonfly would experience,
and the methodology text should say so: the line is Dragonfly as shipped, not
Dragonfly's `main`.

### Series semantics

A tracked-release engine's commit list is its release tags, newest last, so the
planner's `merge_commits`, `commit_index` and landmark machinery work unchanged
with tags in place of hashes; every tag is a release landmark. The
`release-and-tip` scope already gives it no bisection and no backfill. The one
planner extension it needs is the drift re-measure: today `tip_due` treats a
revision that has a point as done, and a drift sample re-measures the current
tag, so the sample needs its own identity (tag plus measurement date) or the
point needs a history of values. The exporter already emits `metadata.engine`,
`metadata.scope` and a per-point `sample`; a drift sample adds `"drift"` to that
vocabulary next to `release`, `tip` and `history`. The dashboard draws a
tracked-release series as a stepped band across the date axis with a label at
each step; the same rendering suits a Redis series, whose points are all
release or tip samples.

Commit links: `engine_repo_slug` reads the engine's `REPOSITORIES` entry, so a
`dragonflydb/dragonfly` entry there is enough; link release pages, not commits,
for tracked-release engines.

### Dashboard

- Engine selector gains Dragonfly. `compare.html` reads the engine list and the
  comparable workloads from the epoch manifest instead of a hardcoded pair, so
  it becomes three-way without a code change there.
- Tracked-release series render as a stepped band with a label at each release; drift samples within a release draw as the band, release steps as markers.
- The methodology text gains a paragraph: Dragonfly version, flags, thread
  count, the note that `MGET` across shards pays a multi-shard transaction
  Valkey does not, and that cluster-mode emulation is off.
- Publish the parameter table (threads, connections, pipelining, key/value
  sizes, key count, warmup, duration) next to any chart with a comparator.
  Dragonfly's own benchmark post does this; match it.

## Fleet rollout

1. **Asset check** (blocking). On the `v1.40.2` release page, confirm an
   aarch64 asset exists and note exact asset names and checksum file. If none,
   the arm64 platforms have no Dragonfly series until assets appear; the x86
   platforms proceed.
2. **Kernel and ulimit.** All four runners are 5.10+ (armbench is RHEL 9 with
   the io_uring backport). Check `ulimit -l` on each; raise via the existing
   systemd drop-in if Dragonfly complains.
3. **Smoke on g4bench.** Provision, launch with `--proactor_threads 7`, run one
   cachecannon GET P10/400c cell through the normal task path, confirm INFO
   fields the memory task reads (`used_memory`) are present and sane.
4. **One-cell A/B.** Same cell against Valkey unstable head on the same host,
   same hour. Sanity-check the ratio against Dragonfly's published 16-vCPU
   figure; a wildly different number means a launch-flag mistake, not a finding.
5. **Fleet enable** as a v3-epoch coordinator at low urgency, one platform at a
   time, via the existing sweep pause and epoch precedence levers.
6. **Dashboard** PR after the first series lands, so the methodology paragraph
   describes what actually ran.

## Acceptance

- `SWEEP_ENGINES` gains Dragonfly with no change to Valkey or Redis behaviour;
  existing tests pass.
- No Dragonfly source exists anywhere on a runner or in this repository at any
  point of the rollout; the only Dragonfly bytes are release assets in the
  binary cache.
- Dragonfly GET/SET/mixed series on all four platforms, with the parameter
  table published.
- A new Dragonfly release is measured on every platform within one scheduling
  day of appearing on GitHub, with no human action; a held-back channel logs a
  warning each round; an arch with missing assets keeps its previous tag and
  is visible on the status page.
- No flamegraph, jemalloc or symbol data collected for Dragonfly (same as Redis).
- The guide's brand rule holds: the comparator is presented as a reference line,
  the dashboard title and identity are unchanged.

## Open questions

- arm64 prebuilt availability (docs only link x86-64).
- Kernel floor: Dragonfly's docs say 4.14, 4.19 and 5.11 in three places.
  5.10+ is assumed sufficient.
- Whether `SHUTDOWN NOSAVE` and `INFO` field names match closely enough for the
  existing parsers; `INFO` is Redis-compatible by design but has extra sections.
- Drift re-measure cadence within a release: daily like the drift canary, or weekly.
- Whether to also track Valkey releases as a stepped band, so Dragonfly is compared to a
  release as well as to unstable head.

## Effort

Two to three days to the first published series: one day for provisioning and
launch profiles, half a day for the drift re-measure and GitHub polling (the
release-and-tip scope itself already exists), half a day for the dashboard, the
rest for the fleet steps and a rollout that waits on real cells.
Garnet afterwards: one more day (reuses kinds, profiles and series mode; adds a
.NET runtime install and its own launch profile; `MSET` is non-atomic by
default, irrelevant to the GET/SET cells).

## Sources

- Dragonfly licence: https://github.com/dragonflydb/dragonfly/blob/main/LICENSE.md
- Dragonfly vs Valkey benchmark (their methodology): https://www.dragonflydb.io/blog/dragonfly-vs-valkey-benchmark-on-google-cloud
- Dragonfly releases (v1.40.2, 2026-09-03; 1.40.1 fixed pipelined connection state): https://github.com/dragonflydb/dragonfly/releases
- Dragonfly shared-nothing design: https://github.com/dragonflydb/dragonfly/blob/main/docs/df-share-nothing.md
- Dragonfly cluster-mode emulation: https://www.dragonflydb.io/docs/managing-dragonfly/cluster-mode
- Garnet: https://github.com/microsoft/garnet (MIT; v2.1.7, 2026-09-09)
- KeyDB status: https://github.com/Snapchat/KeyDB/issues/895 (creator's departure, recommends Valkey)
- Kvrocks on direct comparison: https://github.com/apache/kvrocks/discussions/389
