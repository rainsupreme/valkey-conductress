"""Standalone CLI for the connection-storm generator.

Run without the rest of the harness::

    python -m conductress.stormgen --host 127.0.0.1 --port 6379 \\
        --clients 2000 --burst-ms 200 --reply-timeout-ms 500 \\
        --policy fixed:200 --stall debug-sleep:2 --duration-s 10 --json out.json

Emits one JSON result document (to ``--json`` and/or stdout) with a
``schema_version`` field, the reduced metrics, the stall record, and the
kernel listen-overflow deltas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from . import SCHEMA_VERSION, metrics
from .policy import parse_policy
from .runner import StormConfig, StormResult, run_storm
from .stall import parse_stall


def build_parser() -> argparse.ArgumentParser:
    """Build the stormgen argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m conductress.stormgen",
        description="Connection-storm load generator for Redis/Valkey servers.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6379, help="Server port (default: 6379)")
    parser.add_argument("--clients", type=int, default=1000, help="Number of storm clients (default: 1000)")
    parser.add_argument(
        "--burst-ms",
        type=int,
        default=200,
        help="Window over which client first-attempts are spread uniformly (default: 200)",
    )
    parser.add_argument(
        "--connect-timeout-ms", type=float, default=1000.0, help="Per-connect timeout in ms (default: 1000)"
    )
    parser.add_argument(
        "--reply-timeout-ms", type=float, default=1000.0, help="Per-reply timeout in ms (default: 1000)"
    )
    parser.add_argument(
        "--policy",
        default="fixed:200",
        help="Reconnect policy: immediate, fixed:<ms>, or exp:<base_ms>:<max_ms>[:jitter] (default: fixed:200)",
    )
    parser.add_argument(
        "--handshake",
        action="append",
        default=None,
        help="Handshake command sent after connect, repeatable (default: a single 'HELLO 3'; "
        "pass --handshake '' once for no handshake)",
    )
    parser.add_argument(
        "--first-command",
        default="GET stormkey",
        help="First command each client issues after the handshake (default: 'GET stormkey')",
    )
    parser.add_argument("--duration-s", type=float, default=10.0, help="Total run duration in seconds (default: 10)")
    parser.add_argument(
        "--stall",
        default="none",
        help="Server stall injector: none or debug-sleep:<seconds> (default: none)",
    )
    parser.add_argument(
        "--stall-after-s",
        type=float,
        default=1.0,
        help="Seconds into the run at which the stall is injected (default: 1.0)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Worker processes across which clients fan out (default: 1)"
    )
    parser.add_argument(
        "--bind-addr",
        action="append",
        default=None,
        help="Source address a client may bind (repeatable) to spread ephemeral ports across loopback aliases",
    )
    parser.add_argument(
        "--bucket-ms", type=int, default=metrics.DEFAULT_BUCKET_MS, help="Timeline bucket width in ms (default: 100)"
    )
    parser.add_argument("--json", default=None, help="Write the result document to this path (also printed to stdout)")
    return parser


def _resolve_handshake(raw: Optional[List[str]]) -> List[str]:
    """Turn the repeatable --handshake flag into the handshake command list.

    ``None`` (flag absent) means the default single ``HELLO 3``. An explicit
    empty string (``--handshake ''``) means no handshake at all.
    """
    if raw is None:
        return ["HELLO 3"]
    return [cmd for cmd in raw if cmd.strip()]


def config_from_args(args: argparse.Namespace) -> StormConfig:
    """Build a validated :class:`StormConfig` from parsed args.

    Policy and stall specs are parsed here so an invalid spec fails fast with
    a clear message before any connection is opened.
    """
    parse_policy(args.policy)  # validate early
    parse_stall(args.stall)  # validate early
    if args.clients < 1:
        raise ValueError(f"--clients must be >= 1, got {args.clients}")
    if args.duration_s <= 0:
        raise ValueError(f"--duration-s must be > 0, got {args.duration_s}")
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    return StormConfig(
        host=args.host,
        port=args.port,
        clients=args.clients,
        burst_ms=args.burst_ms,
        duration_s=args.duration_s,
        connect_timeout_ms=args.connect_timeout_ms,
        reply_timeout_ms=args.reply_timeout_ms,
        policy_spec=args.policy,
        stall_spec=args.stall,
        handshake=_resolve_handshake(args.handshake),
        first_command=args.first_command,
        workers=args.workers,
        bind_addrs=args.bind_addr,
        stall_after_s=args.stall_after_s,
        bucket_ms=args.bucket_ms,
    )


def build_document(result: StormResult) -> dict:
    """Reduce a raw :class:`StormResult` to the emitted JSON document."""
    config = result.config
    reduced = metrics.summarize(
        result.events,
        config.clients,
        storm_start=0.0,
        stall_end=result.stall_end_relative(),
        bucket_ms=config.bucket_ms,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "host": config.host,
            "port": config.port,
            "clients": config.clients,
            "burst_ms": config.burst_ms,
            "duration_s": config.duration_s,
            "connect_timeout_ms": config.connect_timeout_ms,
            "reply_timeout_ms": config.reply_timeout_ms,
            "policy": config.policy().describe(),
            "handshake": list(config.handshake),
            "first_command": config.first_command,
            "workers": config.workers,
            "stall": config.stall().describe(),
            "stall_after_s": config.stall_after_s,
            "bucket_ms": config.bucket_ms,
            "bind_addrs": list(config.bind_addrs) if config.bind_addrs else [],
        },
        "stall": {
            "kind": result.stall.kind,
            "started": result.stall.started,
            "ended": result.stall.ended,
            "stall_end_relative": result.stall_end_relative(),
        },
        "listen_overflow_delta": result.listen_delta,
        "metrics": reduced,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args, run one storm, emit the result document. Returns an exit code."""
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = asyncio.run(run_storm(config))
    document = build_document(result)
    payload = json.dumps(document, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
