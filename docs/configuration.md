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
  Its point is scored on the **median** of the per-rep series rather than the
  mean, because P1 throughput can land in two distinct modes across server
  restarts; the median reports the mode most reps reached, where a mean would
  report a value no rep produced. The coefficient of variation and the
  published score bounds still show the full between-restart spread, so a mode
  split is visible rather than hidden.
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
beside the score, its coefficient of variation and its repetition count.

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
