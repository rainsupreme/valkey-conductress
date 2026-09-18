# Connection-storm task

The storm task measures how a server behaves under a **connection storm**: a
large burst of clients all try to connect at once, each issues one command, and
any client whose attempt times out or is refused closes and reconnects after a
back-off delay. Partway through the run a **stall injector** makes the server's
main thread unavailable for a bounded window. The stall is what turns an
orderly burst into a storm, and the task quantifies how bad the storm gets and
how long the server takes to recover once the stall clears.

This is a different question from throughput. A throughput task asks "how many
operations per second can the server sustain in steady state?"; the storm task
asks "when the server briefly cannot serve new connections, how much extra
connection load does that create, and how long does the aftermath last?".

## Queuing a run

```
conductress queue add-storm \
    --source valkey --specifier unstable \
    --clients 2000 --burst-ms 200 \
    --connect-timeout-ms 1000 --reply-timeout-ms 500 \
    --policy fixed:200 \
    --stall debug-sleep:2 --stall-after 5 \
    --duration 20 --io-threads 1
```

Key levers:

- `--clients` -- the client population (the storm's size).
- `--burst-ms` -- the window over which the clients' first attempts are spread.
  A smaller window is a sharper burst.
- `--connect-timeout-ms` / `--reply-timeout-ms` -- how patient each client is
  before it gives up an attempt and reconnects. Short timeouts amplify the
  storm because clients abandon and retry sooner.
- `--policy` -- the reconnect back-off: `immediate`, `fixed:<ms>`, or
  `exp:<base_ms>:<max_ms>[:jitter]` (exponential back-off, optionally with full
  jitter to de-synchronize the reconnecting herd).
- `--handshake` -- `;`-separated commands each client sends right after
  connecting (default `HELLO 3`; pass `''` for none).
- `--first-command` -- the command each client issues after the handshake
  (default `GET stormkey`; the runner prefills the key `stormkey`).
- `--stall` -- the stall injector: `none` (a baseline/control run) or
  `debug-sleep:<seconds>` (`DEBUG SLEEP`, which blocks the main thread and is
  portable across Redis and Valkey).
- `--stall-after` -- seconds into the run at which the stall fires. It must
  fire and clear inside the run window.
- `--workers` -- worker processes the clients fan out across, for populations
  too large for one process's event loop.
- `--bind-addrs` -- `,`-separated loopback source addresses to spread the
  clients' ephemeral ports across (see "Ephemeral-port exhaustion" below).
- `--io-threads` -- server I/O threads. Note this does **not** change which
  thread serves a new connection's first command (see mechanism 2).

## The three mechanisms

A connection storm is costly for three distinct reasons, and the task records
evidence for each.

1. **Kernel listen-queue overflow.** A listening socket has a bounded accept
   queue (`somaxconn` / the server's `tcp-backlog`). While the main thread is
   stalled it stops calling `accept()`, so connections that finished their TCP
   handshake sit in that queue; once it is full the kernel drops further
   completions. The task reads `ListenOverflows` and `ListenDrops` from
   `/proc/net/netstat` before and after the run and reports the delta in
   `listen_overflow_delta`. A non-zero delta is direct evidence the accept
   backlog overflowed.

2. **The first reply needs the main thread.** Even with multiple I/O threads, a
   newly accepted connection's first command is processed on the main thread.
   While the main thread is stalled, every fresh connection blocks waiting for
   its first reply and eventually hits `--reply-timeout-ms`, closes, and
   reconnects. The result is **amplification**: the number of connect attempts
   divided by the client population. 1.0 means everyone connected first try; a
   stall drives it well above 1.0.

3. **The post-recovery tail.** When the stall clears, the server does not
   return to normal instantly. Clients that already gave up leave half-open
   connections and a full accept queue the server must still drain, so connect
   latency and success rate recover over a measurable window. That window is
   the primary score.

## Reading the metrics

The result row's `score` is **`recovery_seconds`** and **lower is better**
(the task records the raw number and documents the direction here; it does not
touch any sweep coordinator, and storm results are their own series, never
sweep-comparable with throughput history). The `data` block carries:

- `recovery_seconds` -- wall time from the stall clearing until the storm is
  quiescent: every client connected and no attempt still in flight. `null` when
  the run never reached quiescence (some client never connected within the run)
  or when there was no stall.
- `recovery_lower_is_better` -- always `true`; a self-describing marker so a
  reader of the row does not have to know the convention out of band.
- `amplification`, `attempts`, `connected_clients` -- the totals from
  mechanism 2.
- `outcomes` -- attempt counts per outcome (`connected`, `connect_timeout`,
  `reply_timeout`, `refused`, `reset`, `error`).
- `attempt_latency` -- p50/p90/p99/max of per-attempt duration.
- `time_to_quiescence` -- quiescence measured both `from_start` (storm start)
  and `from_stall_end` (the score).
- `timeline` -- per-bucket counts (`started`, `connected`, `timeouts`) at the
  `--tick-ms` width, so the storm's shape over time can be plotted.
- `listen_overflow_delta` -- the mechanism-1 counters.
- `info_timeline` -- per-tick server INFO samples (`connected_clients`,
  `total_connections_received`, `rejected_connections`) from a persistent
  connection, and `info_sampler_gaps` -- ticks where the sampler's own
  connection was blocked behind the stalled main thread, so "server quiet" is
  never confused with "could not observe".
- `generator_config`, `stall_record`, `schema_version` -- provenance for
  reproducing the run.

## Cross-engine and environment caveats

- **I/O-thread semantics differ between engines.** The number and role of I/O
  threads is not identical across Redis and Valkey, and neither changes the
  fact that a connection's *first* command is served on the main thread. Do not
  read a storm result as an I/O-thread scaling result.
- **Loopback vs a real network.** Running the generator and server on one host
  over loopback removes NIC, driver, and switch effects and gives the server an
  unrealistically fast client. It is excellent for comparing two builds or two
  configurations on the same host; it is not a model of a production network
  path.
- **Ephemeral-port exhaustion.** Every client connection consumes an ephemeral
  port in the `(source_ip, dest_ip:port)` tuple space, which caps near ~28k
  ports for a single source address. A large storm can exhaust the client's own
  ports before it stresses the server, silently turning a server test into a
  client-side limitation. Spread the source addresses with `--bind-addrs`
  (e.g. `127.0.0.2,127.0.0.3,...`, which are all local loopback) to multiply
  the available port space, and watch the `error`/`reset` outcome counts for
  client-side exhaustion.

## Extending

Each concern in the generator is its own module
(`src/conductress/stormgen/`), so variants are additive:

- **New stall injectors.** Add a `StallInjector` subclass in `stall.py` (e.g. a
  busy Lua script or a large key-space scan) and one line in `parse_stall`.
  Everything else -- the task, the metrics, the CLI -- is unchanged.
- **New reconnect policies.** Add a `ReconnectPolicy` subclass in `policy.py`
  and one line in `parse_policy`.
- **Background load.** The task carries a `background_load` field (currently
  only `none`) so a closed-loop steady-state load can be layered under the
  storm later without changing the task's serialized shape.
- **Cluster mode.** The single-instance topology could be generalized the same
  way the replica-read task describes, as a follow-up.
