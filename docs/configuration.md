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
cachecannon-driven sweep, whose roster is GET throughput, mixed GET/SET
throughput, and GET latency at a fixed request rate (`SWEEP_V3_LATENCY_*`).

Two series rules follow from that definition:

- A series that needs no load generator belongs to every epoch. Memory
  overhead is read from the server's own `INFO` after Conductress fills it
  through its populator, so one memory series is published under each epoch in
  `SWEEP_GENERATOR_INDEPENDENT_EPOCHS` from one state file; the first entry is
  the epoch it schedules and pauses under.
- An epoch-1 series that a newer epoch has replaced is *retired*
  (`SWEEP_V1_RETIRED_SERIES`, keyed `metric:workload`): it keeps its state and
  keeps publishing the history it holds, but never queues another task, so the
  replacement is the only series still measuring that workload. The
  `conductress sweep pause` selectors are the runtime lever for everything
  else; retirement is the permanent one.

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
