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
- `--storm-stall` -- `none` (a baseline/control run), `debug-sleep:<seconds>`
  (`DEBUG SLEEP`, which blocks the main thread for the whole window -- a hard
  stall -- and is portable across Redis and Valkey), or
  `slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s>` (a repeating partial
  stall: the main thread keeps progressing but is blocked `block_ms` out of
  every `period_ms` for `duration_s`, a duty cycle of `block_ms / period_ms`).
  `kind` is `debug-sleep` (exact, dataset-independent) or `lua` (a
  self-calibrated busy `EVAL` loop; `block_ms` capped at 1000 to stay under the
  server's `busy-reply-threshold`). See "Two stall regimes" and "Auto-added
  `--enable-debug-command local`" below.
- **Ordering** -- by default the storm is *stall-first*: the stall command is
  issued, then the burst starts `--storm-burst-after-stall-ms` (default 200)
  later. This is the realistic case: a stall already in progress when
  reconnecting clients arrive. Pass `--storm-burst-first` for the legacy
  ordering (burst first, stall injected part-way through).
- `--storm-start-delay-s` -- launch the generator this many seconds after the
  overlay starts (default 5). The background measurement begins about one
  second after the overlay starts, so the stall lands roughly `delay - 1`
  seconds into the memtier series and the series carries an undisturbed
  baseline before it. The storm's own duration is the measurement window minus
  this delay minus a 2 s margin (never below 5 s), so it ends before the
  background measurement does. Set `0` to start the storm together with the
  measurement.
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
- `--server-sample-ms` -- poll the server's INFO counters every N ms during the
  measurement (`0` = off, else `>= 20`). connection-storm defaults to 100 when
  the flag is absent (see "The server-side sampler").

## Two stall regimes

The stall injector comes in two regimes, which stress different server limits:

- **Hard stall** (`debug-sleep:<seconds>`) -- the main thread is blocked for the
  entire window and does nothing else: no `accept()`, no command processing. The
  accept queue fills and overflows, and the amplification is driven by every
  fresh connection waiting on a main thread that never runs.
- **Slow loop** (`slow-loop:<kind>:<block_ms>:<period_ms>:<duration_s>`) -- the
  main thread keeps progressing but with long event-loop iterations: it is
  blocked `block_ms`, then free for the rest of `period_ms`, repeating for
  `duration_s`. The duty cycle is `block_ms / period_ms`. This is the realistic
  "expensive command" or "busy Lua" regime: the server is degraded, not frozen,
  so between blocks it does drain some of the accept queue and answer some first
  commands. A prediction this regime lets you test on the fleet: under a partial
  (e.g. 50%) duty slow loop the server's accept throttle
  (`max-new-connections-per-cycle`, default 10 accepted per event-loop
  iteration) governs how many connect timeouts occur, whereas under a hard stall
  the throttle is irrelevant because the loop never iterates during the stall.

The `lua` slow-loop block is a pure busy `EVAL` (no keys, so it is
dataset-independent), self-calibrated at loop start (two rounds at a fixed
iteration count, then scaled to `block_ms`) and gently re-scaled from each
measured round trip. It is kept below 1 s per block so it never trips the
server's `busy-reply-threshold` (5 s). The `debug-sleep` block is exact.

A slow loop records, in `storm.stall`, how the repeated blocking actually
behaved:

- `blocks_issued` -- how many blocking commands ran.
- `achieved_block_ms_p50` / `achieved_block_ms_max` -- measured block durations.
- `effective_duty` -- summed measured block time over the loop's wall time (the
  realized `block / period`).

For a slow loop, `storm.stall.started` is the first block and `storm.stall.ended`
is the loop end, so the whole degraded window is the shaded band on the plot. A
slow loop whose `duration_s` (plus the burst offset) would outlast the storm's
own duration is clamped to the storm window minus one second, with a warning;
`StormOverlay._duration_s()` already sizes the storm to the measurement window,
so a very long loop is bounded by it.

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
- `storm.stall` -- the stall record: `kind`, wall-clock `started`/`ended`, and
  (for a `slow-loop` stall) `blocks_issued`, `achieved_block_ms_p50`,
  `achieved_block_ms_max`, and `effective_duty`. The achieved-block fields are
  `0`/`null` for a hard stall or no stall. Present from generator
  `schema_version` 3 onward.
- `storm.timeline` -- per-bucket `started`/`connected`/`timeouts` counts.
- `storm.origin_wall` -- the generator's monotonic origin in wall-clock
  (`time.time()`) seconds, so the storm timeline can be placed on the shared
  axis (see "Time axis"). Present from generator `schema_version` 2 onward.
- `storm.launched_wall` -- when the runner launched the generator process
  (wall-clock), `--storm-start-delay-s` after the overlay started; the
  generator's prewarm runs between this and `storm.origin_wall`.
- `storm.generator_config`, `storm.schema_version` -- provenance.

The scenario also records `overlay_start_offset_s`: how long after the
background measurement began the overlay started (the 1 s connect-establish
delay, matching how bgsave's ~40%-of-duration offset lets a reader align the
memtier dip with the overlay event). Use it to line up the background RPS dip
with the stall/storm timeline.

## The server-side sampler

The scenario also records the **server's own view over time**. When
`server_sample_ms > 0` a sampler opens one persistent connection to the server
under test and, every `server_sample_ms` milliseconds, sends `INFO clients`
and `INFO stats` in one write and reads both replies. `connection-storm`
enables it at 100 ms by default (`--server-sample-ms` overrides; `0` turns it
off); other scenarios leave it off unless the flag is given. Validation refuses
a cadence below 20 ms so the row count stays bounded (100 ms over a 60 s
measurement is 600 rows).

Each row is `{t_sent, t_reply, connected_clients, total_connections_received,
rejected_connections, total_commands_processed, listen_overflows,
listen_drops}`. `t_sent`/`t_reply` are wall-clock (`time.time()`). The
`connected_clients` ramp after the stall clears is the mechanism made visible:
the main thread resumes accepting and answering, and the count climbs back to
the client population.

- **Cost.** `INFO clients` / `INFO stats` are O(1), and the sampler is ONE extra
  client -- so `connected_clients` reads one higher than the storm's own
  population while the sampler is connected.
- **Gaps are data, never retried.** While the main thread is stalled, `INFO`
  does not return; the sampler does not time out and retry inside a tick -- the
  reply simply arrives late and the real `t_reply` is recorded. A consumer
  detects a gap as `t_reply - t_sent > 5 x tick` (helper `sampler_gaps(rows,
  tick_ms)`); the plot uses it to break the line and draw a grey marker.
- **Kernel counters are local-only.** When the runner and the server share a
  host (loopback, the fleet today) each row carries `ListenOverflows` /
  `ListenDrops` as deltas from the first tick, read from `/proc/net/netstat`
  (reusing `conductress.stormgen.netstat`). When the server is remote those two
  fields are `null` (logged once) -- the counters would be the runner host's,
  not the server's.

Storage: `scenario_metrics["server_timeline"]` (the rows) and
`scenario_metrics["server_timeline_tick_ms"]`, per repetition.

## Time axis

Every series can be placed on one wall-clock axis. The runner records, per
repetition, `measure_start_wall` (just before the overlay starts),
`background_start_wall` (just before the memtier command) and
`background_end_wall`. The generator records `storm.origin_wall` (its monotonic
origin in wall-clock). The stall record's `started`/`ended` are already
wall-clock.

Plot convention: **`t = 0` is the storm event: `storm.stall.started` when a
stall exists, otherwise `storm.origin_wall` (the generator's burst origin,
after prewarm).** The default window opens at the background measurement's
start (`background_start_wall`, a negative axis position) and closes 3 s after
the last storm bucket or the stall end. Each series maps by subtracting `t0`:
the background buckets from `background_start_wall + second`, the sampler rows
from their own `t_reply`, and the storm timeline from `storm.origin_wall +
t_ms/1000`. This replaces the older ~1 s inference via `overlay_start_offset_s`.

## Plotting

With the `plots` extra installed (`pip install 'conductress[plots]'`):

```
conductress plot connection-storm <task-id> [<task-id> <task-id>] \
    --out fig.png [--rep N] [--xrange=-3:8]
```

One column per task id (at most three), four rows sharing the x axis:

1. background throughput (memtier `interval_rps`, step plot at 1 s)
2. `connected_clients` (server sampler, line broken across gaps)
3. listen-queue overflows (cumulative, its own row and scale -- not a twin
   axis; an empty row with a note when the server was remote)
4. storm clients: `timeouts` per bucket as bars, cumulative `connected` as a
   light filled area behind them

Every row carries the stall band, a dash-dot quiescence marker, and grey
vertical lines where the sampler had gaps. Repetitions draw as thin lines with
the median rep bold; `--rep N` draws one. Colour is per column (before/after
side-by-side, one hue each); series within a column are told apart by row, not
colour. The default figure is 1500x1100 px at 100 dpi.

Example (from local calibration, two backlog configurations side by side):
each column is one task; row 1 shows background GET/s dipping to near zero
inside the stall band and recovering after; row 2 shows `connected_clients`
flat during the stall then ramping back to the full client population once the
main thread resumes accepting; row 3 shows the cumulative listen-queue
overflow climbing only while the accept queue is starved; row 4 shows the
timeout bars concentrated around the stall with the cumulative-connected fill
rising to the quiescence marker. (No image is committed here: the repository's
`.gitignore` excludes `*.png` outside `brand/`, so a rendered figure lives with
the run, produced by the `conductress plot` command above.)

## Auto-added `--enable-debug-command local`

`DEBUG SLEEP` is rejected unless the server was started with
`enable-debug-command`. For `connection-storm` with any stall that issues
`DEBUG SLEEP` -- a `debug-sleep` hard stall **or** a `slow-loop:debug-sleep:...`
partial stall -- the scenario appends `--enable-debug-command local` to the
server arguments automatically, logs it at INFO, and records the effective
arguments in results as `server_args_effective`. This happens for **no other
scenario**; a `slow-loop:lua` stall does not need it (its block is an ordinary
`EVAL`).

## Storm size

A large storm runs up against several ceilings that are not the server code
under test. Know them so a storm result is not silently a limit test:

- **maxclients.** Valkey's default `maxclients` is 10,000. A storm of, say,
  10,000 clients plus the background memtier load (400 connections) plus the
  sampler would be rejected with "max number of clients reached", turning a
  storm test into a maxclients test. The scenario guards against this: when the
  storm client count plus headroom (4096, covering the background load and the
  sampler) would exceed 10,000, `StormOverlay.extra_server_args()` also appends
  `--maxclients <clients + 4096>` (logged at INFO, recorded in
  `server_args_effective`). This happens for **no other scenario**.
- **Ephemeral ports.** Every client connection consumes an ephemeral port in the
  `(source_ip, dest_ip:port)` tuple space. On the runners `ip_local_port_range`
  is 32768-60999, about 28K ports per source address; a storm larger than that
  from one source address exhausts the client's own ports before it stresses the
  server. `tcp_tw_reuse=2` lets the kernel reuse ports still in `TIME_WAIT` on
  loopback, which softens but does not remove the cap. Spread the load across
  loopback source aliases with `--storm-bind-addrs 127.0.0.2,127.0.0.3,...` to
  multiply the available port space, and watch the `error`/`reset` outcome
  counts for exhaustion.
- **File descriptors.** Each connection is a file descriptor on the server. The
  runner shell's `ulimit -n 65536` is the server's fd ceiling; a storm plus the
  background load must fit under it (raise `maxclients` and the storm stays
  bounded by this hard limit).
- **Generator delivery rate.** Even fanned across workers (8 workers at roughly
  5-10K connects/s each), the generator opens a finite number of connections per
  second. `storm.burst_actual_ms` shows the span the burst actually took; when
  it exceeds the nominal `--storm-burst-ms` the generator could not deliver the
  requested burst at that size, so read the achieved span rather than the
  nominal one.

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
