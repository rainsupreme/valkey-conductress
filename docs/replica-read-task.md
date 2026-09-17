# Replica-read task

`conductress queue add-replica-read` measures read throughput served by a
**replica** while its primary ingests writes at a fixed rate over a real
replication link. Score = mean reader throughput (GET/s at the first replica).

This is the topology the concurrent-reads work (door-2 feature 1, speculative
GET on I/O threads) targets: on a replica the only writer is the replication
stream, executed on the main thread, with no client write/read interleaving.

## Topology

All instances run on one benchmark host over loopback, each with its own
port, working directory (`--dir`, so full-sync RDB files do not collide),
logfile and CPU allocation from the topology-aware allocator.

| instance | port          | io-threads             | generator                          |
|----------|---------------|------------------------|------------------------------------|
| primary  | `--base-port` | `--primary-io-threads` | writer: cachecannon SET, `rate_limit` = `--write-rate` |
| replica  | base + 1      | `--io-threads`         | reader: cachecannon GET, closed loop, `--connections`/`--threads`/`--pipelining` |
| replica  | base + 2 ...  | `--io-threads`         | none (extra `--replicas` add fan-out load only) |

Server arguments: `--server-args` applies to every instance; `--primary-args`
and `--replica-args` are appended after it for their role, so a role-specific
flag overrides a shared one (valkey applies later command-line config over
earlier). `--replica-args` is the feature lever for an A/B.

## Phases per repetition

1. Kill stray servers, start the topology, `REPLICAOF`, wait for link up.
2. Preload: cachecannon prefill of `--keyspace` keys at the primary, then wait
   until every replica's offset equals the primary's and both key counts match.
3. Measure: start the fixed-rate writer at the primary, one second later the
   reader at the first replica (`--warmup` + `--duration`). `INFO replication
   stats` is sampled from every instance every `--sample-interval` seconds
   with a single remote shell per sample, so primary/replica offset skew is a
   few milliseconds. `--info-fields` adds arbitrary INFO fields to the sample
   (e.g. door-2 counters); absent fields are omitted, not errors.
4. Guards (fail loudly): reader 0% errors and >= 99% hit rate; writer 0%
   errors and achieved rate >= 90% of `--write-rate`. An under-delivering
   writer would silently turn the cell into a lower-write-rate cell.

## Result row

`method = "replica-read"`, `score` = mean reader rps over reps, `cv`/`reps`
as for other tasks. `data` carries: the topology spec, per-port server CPUs,
writer target and achieved rate, replication lag (primary minus replica
offset, bytes: max/mean, per-rep p99), per-rep reader/writer latency, the full
per-instance INFO series, and both generated TOMLs.

Lag resolution is bounded by sample skew times replication bytes per second;
lags below that are noise and are clamped at zero.

## A/B recipe

Two tasks that differ only in `--specifier` (or `--replica-args`):

```bash
conductress queue add-replica-read --source valkey --specifier <upstream-sha> \
  --io-threads 8 --write-rate 50000 --runner g4bench --note "replica-read base w50k io8"
conductress queue add-replica-read --source valkey-rainfall --specifier <feature-sha> \
  --io-threads 8 --write-rate 50000 --runner g4bench --note "replica-read feature w50k io8"
```

Sweep `--write-rate` (10k / 50k / 200k) and `--io-threads` (4 / 8 / 16) for the
headline grid. Match by note and specifier in `results/output.jsonl`.

## Not yet

- Cluster mode: `TopologySpec` can describe `cluster_enabled` per instance
  (bus port = port + 10000) but the bootstrap (ADDSLOTS / MEET / REPLICATE) is
  not implemented; `TopologyGroup.start` refuses such a spec.
- Event phases (full sync at t, failover, slot migration) are the natural next
  step on the same runner: a phase list on the task plus per-instance actions.
- Sweep coordinator integration: this is a manual-queue task type only.
