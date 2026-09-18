"""Connection-storm load generator.

A standalone generator that opens a burst of client connections against a
Redis or Valkey server, holds each open after its first successful command,
and reconnects on failure according to a configurable policy. Its purpose is
to reproduce and measure the connection-storm failure mode: when a server's
main thread stalls, new connections queue in the kernel accept backlog, time
out, and reconnect, so the offered connection load amplifies far above the
client population and the server needs time to drain the backlog once the
stall clears.

The package is usable without the rest of the benchmark harness via
``python -m conductress.stormgen``; it emits one JSON result document with a
``schema_version`` field. Each concern lives in its own module so a new
reconnect policy, stall injector, or metric can be added without touching the
others:

* :mod:`policy` -- reconnect back-off policies
* :mod:`stall` -- server-side stall injectors
* :mod:`resp` -- minimal RESP encoder and incremental reply parser
* :mod:`client` -- one storm client's connect/command/retry state machine
* :mod:`runner` -- fans clients across worker processes and aggregates events
* :mod:`metrics` -- pure reductions from events to reported numbers
* :mod:`netstat` -- kernel listen-queue overflow counters
"""

SCHEMA_VERSION = 1
