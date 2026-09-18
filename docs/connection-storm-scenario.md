# connection-storm scenario

`connection-storm` is a scenario in the pathological-workload framework
(`queue add-scenario`). Like every scenario it runs a steady **background GET
load** (memtier, same prefill and keyspace as the other scenarios) and lays a
**pathology overlay** on top; here the overlay is a burst of clients that all
try to connect at once, each issues one command, and any client whose attempt
times out or is refused closes and reconnects after a back-off delay. Partway
through, a **stall injector** can make the server's main thread unavailable for
a bounded window. The scenario measures two things at once: how badly the storm
disturbs the background throughput (the standard scenario dip/recovery metrics)
and how the storm itself behaves (the `storm.*` metrics).

This answers a different question from a throughput benchmark. A throughput run
asks "how many ops/sec in steady state?"; connection-storm asks "when the
server briefly cannot serve new connections, how much extra connection load
does that create, how far does concurrent throughput dip, and how long does
each recover?".

## Queuing a run

```
conductress queue add-scenario --scenario connection-storm \
    --source valkey --specifier unstable \
    --io-threads 1 --duration 20 \
    --storm-clients 2000 --storm-burst-ms 200 \
    --storm-connect-timeout-ms 1000 --storm-reply-timeout-ms 500 \
    --storm-policy fixed:200 \
    --storm-stall debug-sleep:2 --storm-burst-after-stall-ms 200
```

The scenario is configured with `--storm-*` flags, which are serialized into a
single `overlay_spec` JSON field on the task. They are **only** valid with
`--scenario connection-storm` (any other scenario rejects them). All are
optional; each falls back to a default.

- `--storm-clients` -- the client population (the storm's size). Default 2000.
- `--storm-burst-ms` -- the window over which first attempts are spread. A
  smaller window is a sharper burst. Default 200.
- `--storm-connect-timeout-ms` / `--storm-reply-timeout-ms` -- how patient each
  client is before giving up an attempt and reconnecting. Short timeouts
  amplify the storm. Defaults 1000 / 500.
- `--storm-policy` -- reconnect back-off: `immediate`, `fixed:<ms>`, or
  `exp:<base_ms>:<max_ms>[:jitter]`. Default `fixed:200`.
- `--storm-handshake` -- repeatable; commands each client sends right after
  connecting. Default a single `HELLO 3`.
- `--storm-first-command` -- the command each client issues after the
  handshake. Default `GET storm:key`.
- `--storm-stall` -- `none` (a baseline/control run) or `debug-sleep:<seconds>`
  (`DEBUG SLEEP`, which blocks the main thread and is portable across Redis and
  Valkey). See "Auto-added `--enable-debug-command local`" below.
- **Ordering** -- by default the storm is *stall-first*: the stall command is
  issued, then the burst starts `--storm-burst-after-stall-ms` (default 200)
  later. This is the realistic case: a stall already in progress when
  reconnecting clients arrive. Pass `--storm-burst-first` for the legacy
  ordering (burst first, stall injected part-way through).
- `--storm-workers` -- worker processes the clients fan out across. Default
  `0` = auto (`min(8, cpu_count)`): a single event loop opens only a few
  thousand connections per second, so a one-process "200 ms" burst of a few
  thousand clients actually spans far longer. The achieved span is reported as
  `storm.burst_actual_ms`.
- `--storm-prewarm-connections` -- throwaway connections opened and closed
  before the storm's baseline (default `= --storm-clients`; `0` disables). See
  "Cold-server first-contact cost".
- `--storm-bind-addrs` -- `,`-separated loopback source addresses to spread the
  clients' ephemeral ports across (see "Ephemeral-port exhaustion").

## The three storm mechanisms

A connection storm is costly for three distinct reasons, and the `storm.*`
metrics record evidence for each.

1. **Kernel listen-queue overflow.** A listening socket has a bounded accept
   queue (`somaxconn` / the server's `tcp-backlog`). While the main thread is
   stalled it stops calling `accept()`, so connections that finished their TCP
   handshake sit in that queue; once it is full the kernel drops further
   completions. Read as `storm.listen_overflow_delta` (`ListenOverflows` /
   `ListenDrops` from `/proc/net/netstat`, before/after the measured storm).

2. **The first reply needs the main thread.** Even with multiple I/O threads, a
   newly accepted connection's first command is processed on the main thread.
   While the main thread is stalled, every fresh connection blocks waiting for
   its first reply and eventually times out, closes, and reconnects. The result
   is **amplification** (`storm.amplification`): connect attempts divided by the
   client population. 1.0 means everyone connected first try; a stall drives it
   above 1.0.

3. **The post-recovery tail.** When the stall clears, the server does not return
   to normal instantly. Clients that already gave up leave half-open
   connections and a full accept queue the server must still drain, so connect
   latency and success recover over a measurable window
   (`storm.quiescence_from_stall_end_s`).

## Two recovery numbers -- they answer different questions

The scenario carries **two** recovery figures, and they are not the same thing:

- **`recovery_seconds`** (top-level, unchanged from every other scenario) is the
  **background** load's recovery: seconds for the memtier GET throughput to
  climb back to 90% of its own baseline after the dip, derived from the
  interval RPS timeseries by `compute_dip_metrics`. It is the *server-side
  collateral* on unrelated traffic.
- **`storm.quiescence_from_stall_end_s`** is the **client-side** recovery:
  seconds from the stall clearing until every storm client has reconnected and
  no attempt is in flight. It is about the reconnecting herd, not the
  background load.

A run can have a small background dip but a long client quiescence, or vice
versa; report both. The scenario's primary `score` stays what every scenario
uses (mean background RPS) -- connection-storm does not change the score
convention.

## The `storm.*` metrics

Under the per-rep scenario metrics (and aggregated in results):

- `storm.quiescence_from_stall_end_s` -- client-side recovery (above); `null`
  when there was no stall or the storm never quiesced within the run.
- `storm.quiescence_from_start_s` -- client quiescence measured from the storm
  start instead of the stall end.
- `storm.amplification` -- attempts / clients (mechanism 2).
- `storm.attempts`, `storm.connected_clients` -- the raw totals.
- `storm.outcomes` -- attempt counts per outcome (`connected`,
  `connect_timeout`, `reply_timeout`, `refused`, `reset`, `error`).
- `storm.attempt_latency` -- p50/p90/p99/max per-attempt duration.
- `storm.listen_overflow_delta` -- mechanism-1 counters for the measured storm.
- `storm.prewarm_listen_overflow_delta` -- the same counters for the prewarm
  phase alone (see below).
- `storm.burst_actual_ms` -- the span the burst actually took (last
  first-attempt start minus first); compare against `--storm-burst-ms`.
- `storm.stall` -- the stall record (kind, wall-clock start/end).
- `storm.timeline` -- per-bucket `started`/`connected`/`timeouts` counts.
- `storm.generator_config`, `storm.schema_version` -- provenance.

The scenario also records `overlay_start_offset_s`: how long after the
background measurement began the overlay started (the 1 s connect-establish
delay, matching how bgsave's ~40%-of-duration offset lets a reader align the
memtier dip with the overlay event). Use it to line up the background RPS dip
with the stall/storm timeline.

## Auto-added `--enable-debug-command local`

`DEBUG SLEEP` is rejected unless the server was started with
`enable-debug-command`. For `connection-storm` with a `debug-sleep` stall the
scenario appends `--enable-debug-command local` to the server arguments
automatically, logs it at INFO, and records the effective arguments in results
as `server_args_effective`. This happens for **no other scenario** and for no
other stall kind.

## Cold-server first-contact cost

A freshly started server overflows its listen queue on the **first** large
connection burst even with no stall injected, and takes zero overflows on an
identical second burst. Observed on this setup: a 2000-connection burst against
a fresh server (default `tcp-backlog` 511) drove roughly 1000-1250
`ListenOverflows`/`ListenDrops`, reproduced across several fresh servers; an
identical immediate second burst against the same server took zero.

The cost is **per-connection**, not per-byte or allocator-related: prewarming
with 2000 throwaway connections removed it, while prewarming with tens of MB of
key data, or exercising a specific allocator size class, did not. The
underlying mechanism is **not established here** -- recorded as an observed
cold-server effect, not attributed to any specific cause.

So the storm overlay runs a **prewarm phase** by default: it opens
`--storm-prewarm-connections` connections (default `= --storm-clients`) with the
same handshake, closes them, waits briefly, and only then takes the netstat
baseline. The prewarm phase's own overflow delta is recorded separately as
`storm.prewarm_listen_overflow_delta`, so the cold-start effect stays visible.
Set `--storm-prewarm-connections 0` to disable prewarm and measure the
cold-start effect itself.

**Prewarm reduces this overflow but does not always eliminate it, because two
separate causes overlap.** One is the cold-server per-connection cost above,
which prewarm addresses. The other is purely the **burst rate**: when the burst
opens connections faster than the server drains its `tcp-backlog`-bounded accept
queue, the queue overflows no matter how warm the server is. Compare a warmed
run's `storm.listen_overflow_delta` against `storm.prewarm_listen_overflow_delta`
and `storm.burst_actual_ms`. A high worker count (the default) makes the burst
sharp enough that the rate cause dominates; a single-worker, slower burst leaves
prewarm more of the cold-start cost to remove. Neither mechanism is attributed
here beyond "connections arrived faster than the server accepted them".

## Observed effect of `max-new-connections-per-cycle`

A separate, measured server-config effect worth knowing when reading overflow
numbers: with the server's `max-new-connections-per-cycle` at its default of
**10**, a 2000-connection burst whose clients each send `HELLO`+`GET` on connect
dropped **445-1587 SYNs** on a *warm, idle* server (no stall, no background
load). Raising it to **1000** dropped **zero**. A burst of **bare** SYNs (no
data sent) never dropped at either setting. This is an observed measurement on
this setup; it is reported neutrally, with no mechanism claimed beyond the
numbers. It means an overflow delta is sensitive to this server setting, so
record it alongside the result when it is non-default.

## Cross-engine and environment caveats

- **I/O-thread semantics differ between engines.** The number and role of I/O
  threads is not identical across Redis and Valkey, and neither changes the fact
  that a connection's *first* command is served on the main thread. Do not read
  a storm result as an I/O-thread scaling result.
- **Loopback vs a real network.** Generator and server on one host over loopback
  removes NIC, driver, and switch effects and gives the server an unrealistically
  fast client. Good for comparing two builds or configs on one host; not a model
  of a production network path.
- **Generator ramp rate.** Even fanned across processes, the generator opens a
  finite number of connections per second, so `storm.burst_actual_ms` can exceed
  the nominal `--storm-burst-ms` for a large, short burst. Read the achieved
  span.
- **Ephemeral-port exhaustion.** Every client connection consumes an ephemeral
  port in the `(source_ip, dest_ip:port)` tuple space, capping near ~28k ports
  for a single source address. A large storm can exhaust the client's own ports
  before it stresses the server, turning a server test into a client-side
  limitation. Spread source addresses with `--storm-bind-addrs`
  (`127.0.0.2,127.0.0.3,...`, all local loopback) and watch the `error`/`reset`
  outcome counts.

## Extending

The generator's concerns each live in their own module
(`src/conductress/stormgen/`), so variants are additive:

- **New stall injectors.** Add a `StallInjector` subclass in `stall.py` (e.g. a
  busy Lua script or a large key-space scan) and one line in `parse_stall`.
- **New reconnect policies.** Add a `ReconnectPolicy` subclass in `policy.py`
  and one line in `parse_policy`.
- **New overlays generally.** The scenario runner drives overlays through a
  small `Overlay` abstraction (`start` / `finish` / `abort`, returning an
  `OverlayResult(rate, metrics)`). `CommandOverlay` wraps the shell-command
  scenarios; `StormOverlay` runs the generator. A new overlay is a new subclass,
  not a change to the runner.
