# Configuration

Conductress reads its settings from three places, in order of specificity:

1. **`config.py`** — the built-in defaults, committed to the repository. These
   are the values a runner uses when nothing overrides them.
2. **Environment variables** — override a subset of those defaults without
   editing or redeploying source. Listed per setting below.
3. **Local JSON files** in the project root — per-host configuration that is
   never committed (each is gitignored, with a committed `*.default.json`
   example alongside it).

Nothing here changes benchmark behavior on its own: the defaults are the
effective values unless an override is set.

## Local files

These live in the project root next to `config.py`. Each is gitignored so a host
keeps its own copy; a committed `*.default.json` shows the expected shape and is
used as a fallback when the real file is absent.

| File | Purpose | Example file |
| --- | --- | --- |
| `servers.json` | Benchmark server list (`ip`, `username`, `name`, optional `disabled`). | `servers.default.json` |
| `runner.json` | This runner's stable identity (`runner_id`, `display_name`). | `runner.default.json` |
| `repositories.json` | Optional. Extends or replaces the built-in repository list (see below). | `repositories.default.json` |

## Paths

The load generators run on the remote benchmark host over SSH, so their paths
are resolved on that host, not on the machine running the CLI.

| Setting | Default | Environment override | Notes |
| --- | --- | --- | --- |
| `REMOTE_MEMTIER_BENCHMARK` | `~/conductress/memtier_benchmark` | — | Used in the benchmark command strings. `~` expands on the **remote** host, so this stays a `~/...` string. |
| `MEMTIER_BENCHMARK` | `PROJECT_ROOT/memtier_benchmark` | — | Local-path form for code that resolves the binary under this checkout. |
| `CACHECANNON_BINARY` | `/home/ec2-user/cachecannon/target/release/cachecannon` | `CONDUCTRESS_CACHECANNON_BINARY` | Absolute path the cachecannon generator builds to on the benchmark host. |

Both binaries are installed on each host by the bootstrap step; you rarely
need to override these. cachecannon drives the throughput, mixed and latency
tasks and the sweeps; memtier_benchmark drives the scenario tasks, which need
its finite request-count mode and per-second timeseries output.

## Benchmark defaults

The throughput and latency defaults (connection counts, thread counts,
pipelining, warmup and scored durations, value sizes, adaptive-repetition
targets) are grouped near the top of `config.py` under the `Benchmark defaults`
and `Sweep configuration` headings. Change them in source; they are the single
source of truth shared by the CLI and the runner. The cachecannon v3 sweep
values are locked to a measured workload identity — changing any of them defines
a new workload, not a continuation of the series.

## Sweep environment toggles

The sweep measures each new commit under a fixed combination of load generator
and benchmark parameter set; one such combination is a *sweep epoch*, and each
epoch carries an identifier (`v1`, `v3`, ...). Results are comparable only
within an epoch, so the `v1`/`v3` shorthand used here and elsewhere names which
generator-and-parameter identity produced a point. `v1` identifies the sweep
driven by `valkey-benchmark`, Valkey's stock load generator; `v3` identifies the
cachecannon-driven sweep.

The v3 Valkey roster is six series, all at the v3 identity (400 connections, 8
client threads, 7 server io-threads, 3M uniform keys, 16-byte keys and values
unless the series says otherwise):

- **GET throughput at P10** (`get-k16-v16-t7-p10`) — the pipelined-read ceiling.
- **mixed GET/SET throughput at P10** (`mixed-s20-k16-v16-t7-p10`) — the
  canonical 80:20 read/write mix.
- **SET throughput at P10** (`set-k16-v16-t7-p10`) — the write counterpart of
  the GET series.
- **GET throughput at P1** (`get-k16-v16-t7-p1`) — the unpipelined read path.
  It runs a **16-client-thread** budget (`SWEEP_V3_P1_CLIENT_THREADS`), not the
  8 the other throughput series use: at P1 cachecannon issues one syscall per
  request instead of one per ten, so it spends 4–5x more CPU per request and 8
  client threads saturate the client below the server's ceiling — on Graviton3
  the 8-thread P1 backfill hit client utilization 0.941, meaning it was
  measuring the client rather than the server. 16 threads (25 connections per
  thread at 400c) restores the headroom. The client budget is part of the
  series identity and never changes within the series, so the P1 series history
  was cleared and restarted at 16 threads rather than mixing two client shapes
  on one line. Its point is scored on the **median** of the per-rep series
  rather than the mean, because P1 throughput can land in two distinct modes
  across server restarts; the median reports the mode most reps reached, where a
  mean would report a value no rep produced. The coefficient of variation and
  the published score bounds still show the full between-restart spread, so a
  mode split is visible rather than hidden.
- **GET throughput at P10 with 1024-byte values** (`get-k16-v1024-t7-p10`) —
  the raw-encoded, copy-dominated reply path. A 16-byte value is embstr-encoded
  and its reply is dispatch-bound; a 1024-byte value is a separate encoding
  whose reply cost is the buffer copy, so it moves on reply-path changes the
  16-byte series cannot see (and does not echo embstr-path changes that series
  does). This series starts at `SWEEP_V3_LARGE_VALUE_FLOOR_TAG` (`9.0.0`)
  rather than the fork point: it guards the reply path going forward, and a
  full backfill would slow the other five series for history nobody has asked
  for. A series floor overrides the engine floor for that one series.
- **GET latency at P1** (`get-k16-v16-t7-p1-r100k`) — p99 at a fixed request
  rate (`SWEEP_V3_LATENCY_*`), lower is better.

A cachecannon task chooses how the recorded score aggregates the per-rep series
through `score_aggregate` (`mean`, the default, or `median`), and always records
the minimum and maximum of that series; the v3 export publishes those bounds
beside the score, its coefficient of variation and its repetition count. The
export also records each series' client budget — `connections` and
`client_threads` — in the series `metadata`, so a reader can tell the 16-thread
P1 line apart from the 8-thread P10 lines.

Two series rules follow from the epoch definition:

- A series that needs no load generator belongs to every epoch. Memory
  overhead is read from the server's own `INFO` after Conductress fills it
  through its populator, so one memory series is published under each epoch in
  `SWEEP_GENERATOR_INDEPENDENT_EPOCHS` from one state file; the first entry is
  the epoch it schedules and pauses under.
- An epoch-1 series that a newer epoch has replaced is *retired*
  (`SWEEP_V1_RETIRED_SERIES`, keyed `metric:workload`): it keeps its state and
  keeps publishing the history it holds, but never queues another task, so the
  replacement is the only series still measuring that workload. Every v1 Valkey
  throughput series — the default GET series and every entry in
  `SWEEP_THROUGHPUT_WORKLOADS` (the 64- and 128-byte GET series, the SET series,
  the four P1 variants, and the four platform-optimal series) — is retired, as
  is the v1 GET latency series; the set is derived from that same roster data
  rather than hand-listed, so it cannot drift from it. Memory series (which
  belong to every epoch) and a comparison engine's own series follow the
  engine's scope and are not retired here. The `conductress sweep pause`
  selectors are the runtime lever for everything else; retirement is the
  permanent one.

### Resetting a series

A series' client budget (its connection count and client-thread count) is part
of its identity, so a change to it — like the P1 GET series moving from 8 to 16
client threads — must clear the old points rather than mix two client shapes on
one chart line. `conductress sweep reset-series` clears one series' history so
the next boundary publish restarts it fresh:

```
conductress sweep reset-series --epoch v3 --workload get-k16-v16-t7-p1
```

It backs up the coordinator's state file to `<file>.bak-<UTC timestamp>` and
deletes it, then relocates (never deletes) any queued task files whose note
names the series into a `benchmark_queue/reset-<timestamp>/` folder, so a cell
queued at the old budget cannot complete after the deploy and land a stale point
on the fresh line. Pass `--engine redis` to reset a comparison engine's
prefixed series (e.g. `redis-get-k16-v16-t7-p1`), `--dry-run` to see what it
would do without touching anything, and `--force` to override its refusal to
run while `conductress.service` is active (a running runner may be mid-task on
the series). The coordinator's completion-time identity guard (below) is the
backstop: a queued 8-thread cell that slips past the reset is refused at
completion rather than recorded.

The deploy-time procedure is: stop the runner service, reset both the Valkey and
the Redis P1 series, then start it again —

```
sudo systemctl stop conductress.service
conductress sweep reset-series --epoch v3 --workload get-k16-v16-t7-p1
conductress sweep reset-series --epoch v3 --workload get-k16-v16-t7-p1 --engine redis
sudo systemctl start conductress.service
```

The completion-time **identity guard** in the v3 coordinator refuses to record a
completed cell whose `threads` (or any other identity field) does not match the
series, logging a WARNING and skipping the point rather than crashing — so a
stale 8-thread cell that completes after the deploy never contaminates the
16-thread line. Published per-commit perf and CPU-stack files for the old points
remain on the data server (the publish `rsync` has no `--delete`); they are
harmless orphans the dashboard no longer references once the series file is
rewritten.

### Engines

The sweep can measure more than one server (`SWEEP_ENGINES` in `config.py`).
An engine is described on two independent axes:

- **Provisioning** says how a binary for a revision comes to exist.
  `built-from-git` clones the engine's repository, checks out the revision and
  runs `make` with the engine's `make_args`. `prebuilt-release` is reserved for an engine whose source licence the
  project does not accept on its runners: only the official release asset for
  the host architecture would be downloaded and run, and no source would ever
  be checked out. Declaring an engine with it is a startup error until the
  download path exists.
- **Scope** says how much of the engine's history the sweep measures. `history`
  is the full treatment: release landmarks, bisection of significant deltas,
  backfill of gaps, and every new tip. `release-and-tip` measures the latest
  release once and the current tip at most once per `tip_interval_hours`, with
  no bisection and no backfill.

Valkey is `history`. Redis is `release-and-tip` at 24 hours: it exists to power
the engine comparison, which reads one point at each engine's latest release
and the recent tip and nothing else, so measuring its history would spend
runner time on data nobody reads. A comparison engine's series carry the
engine name as a prefix (`redis-get-k16-v16-t7-p10`), are exported with
`metadata.engine` and `metadata.scope`, and tag each point with why it was
measured (`sample`: `release`, `tip` or `history`). Every epoch-1 throughput
series of a comparison engine is retired; its v3 series are the ones that
measure. Memory series follow the engine's scope like any other series.

An engine that opts out of internal profiling (`profile_internals=False`)
records aggregate results only: throughput, latency and total memory, with no
CPU flamegraph and no allocation breakdown, so nothing from its binary's symbol
table is published.

These variables control which epochs run. Each toggle enables or disables one
epoch's coordinators, and the precedence variable sets which epoch measures a
new commit first. They fail startup on an unrecognized value rather than
silently selecting the wrong epoch.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `CONDUCTRESS_SWEEP_V3_ENABLED` | `false` | Enable the v3 (cachecannon) sweep coordinators. |
| `CONDUCTRESS_SWEEP_EPOCH_PRECEDENCE` | `v3,v1` | Comma-separated scheduling precedence, highest priority first. The first-listed epoch measures each new commit first. `v1,v3` restores v1-first. |

Booleans accept `1/true/yes/on` and `0/false/no/off` (case-insensitive); any
other value, including an empty string, is a startup error.

## Repositories

`REPOSITORIES` in `config.py` is the built-in list of source repositories made
available for benchmarking. It is authoritative when no `repositories.json` is
present.

An optional `repositories.json` in the project root adjusts that list:

- **Extend** (default): its entries are appended to the built-in list. A
  directory name already present in the built-in list is skipped.
- **Replace**: set `"replace": true` and only the file's entries are used.

Each entry is `{"url": <git url>, "name": <directory name>}`. A malformed file
(bad JSON, missing `url`/`name`, or `replace: true` with no entries) is a clear
error, not a silent fallback.

Example — upstream-only (this is exactly `repositories.default.json`):

```json
{
    "replace": true,
    "repositories": [
        {"url": "https://github.com/valkey-io/valkey.git", "name": "valkey"}
    ]
}
```

Example — keep the defaults and add one more:

```json
{
    "repositories": [
        {"url": "https://github.com/example/valkey.git", "name": "example"}
    ]
}
```

## Publish and control endpoints

The dashboard data server and the fleet control-plane URL both have a default
that points at the project's default endpoint and an environment override.

| Setting | Default | Environment override | Notes |
| --- | --- | --- | --- |
| `PUBLISH_TARGET` | `ec2-user@data.conductress.rainsupreme.net:/var/www/data` | `CONDUCTRESS_PUBLISH_TARGET` | rsync target for `--publish`. |
| control URL | `https://data.conductress.rainsupreme.net/api/v1` | `CONDUCTRESS_CONTROL_URL` | Base URL for the fleet control API. Plain HTTP is rejected except for `localhost`/`127.0.0.1`. |

The fleet client reads several more control-plane variables — see the
[Fleet-aware CLI](fleet-cli.md) guide for `CONDUCTRESS_CONTROL_TIMEOUT`,
`CONDUCTRESS_CONTROL_CA_BUNDLE`, and the operator-token variables.

## Operator token

The control CLI needs an operator bearer token.

| Environment variable | Meaning |
| --- | --- |
| `CONDUCTRESS_OPERATOR_TOKEN` | The token value. |
| `CONDUCTRESS_OPERATOR_TOKEN_FILE` | Path to an owner-only file holding the token (default `~/.config/conductress/operator.token`). Group/world-readable files are rejected. |

See the [Fleet-aware CLI](fleet-cli.md) guide for full details on secure client
configuration.
