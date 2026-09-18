"""Server topologies: one or more valkey-server instances on one or more hosts.

A ``TopologySpec`` describes a layout declaratively: each ``InstanceSpec`` has a
role, port, io-thread count, role-specific server arguments, and the host it
runs on (``None`` means the host the group is given, normally the runner
itself). ``TopologyGroup`` brings the layout up, wires replication, samples
``INFO`` from every instance on a fixed cadence and tears it down.

Two layouts matter today. ``TopologySpec.standalone`` is one instance on the
default port with the legacy working directory and logfile, and is what every
single-instance task runs. ``TopologySpec.replica_read`` is a primary plus
replicas on consecutive ports of the same host, each with its own working
directory. Instances on distinct hosts are the same spec with a ``host`` per
entry. ``ReplicationGroup`` is the older one-instance-per-host abstraction;
its behaviour is preserved here so callers can migrate to ``TopologyGroup``.

Cluster mode is deliberately representable (``cluster_enabled`` per instance,
bus port = port + 10000) but not yet driven: the bootstrap step for a cluster
(ADDSLOTS / MEET / REPLICATE) is a follow-up. ``TopologyGroup.start`` refuses a
spec with ``cluster_enabled`` set so nobody mistakes the placeholder for a
working lever.
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from conductress import config, cpu_sampling
from conductress.config import ServerInfo
from conductress.server import Server

logger = logging.getLogger(__name__)

PRIMARY_ROLE = "primary"
REPLICA_ROLE = "replica"
VALID_ROLES = (PRIMARY_ROLE, REPLICA_ROLE)

# Base port for topology instances. Cluster bus port is port + 10000, so any
# lane of instance ports must stay below 55536.
DEFAULT_BASE_PORT = 6379
CLUSTER_BUS_PORT_OFFSET = 10000
MAX_INSTANCE_PORT = 65535 - CLUSTER_BUS_PORT_OFFSET

# Per-instance working directory root on the benchmark host (see
# Server.instance_dir for why every instance needs its own).
INSTANCE_DIR_ROOT = Path("~") / "conductress-instances"


@dataclass
class InstanceSpec:
    """One valkey-server process in a topology."""

    role: str
    port: int
    io_threads: int = 1
    server_args: str = ""
    cpu_override: str = ""
    cluster_enabled: bool = False
    # ``None`` = the host the TopologyGroup is given (the runner itself).
    host: Optional[ServerInfo] = None
    # A dedicated working directory keeps two instances on one host from
    # clobbering each other's RDB. ``False`` is the legacy layout (no --dir,
    # cwd of the launching shell) that the single-instance tasks have always
    # used; the standalone factory sets it so nothing on the runners moves.
    own_dir: bool = True

    def __post_init__(self):
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {self.role!r}")
        if not 1024 <= self.port <= MAX_INSTANCE_PORT:
            raise ValueError(
                f"port must be in 1024..{MAX_INSTANCE_PORT} (cluster bus needs port+10000), got {self.port}"
            )
        if self.io_threads < 1:
            raise ValueError(f"io_threads must be >= 1, got {self.io_threads}")

    @property
    def instance_dir(self) -> Optional[Path]:
        """Working directory (``--dir``) for this instance on the host, or ``None`` for the legacy layout."""
        return INSTANCE_DIR_ROOT / str(self.port) if self.own_dir else None

    def resolved_host(self, default: ServerInfo) -> ServerInfo:
        """The host this instance runs on, given the group's default."""
        return self.host if self.host is not None else default

    def to_dict(self) -> dict:
        """JSON-able form for the result row."""
        return asdict(self)


@dataclass
class TopologySpec:
    """A set of instances. Exactly one primary; replicas follow it."""

    instances: list = field(default_factory=list)

    def __post_init__(self):
        primaries = [i for i in self.instances if i.role == PRIMARY_ROLE]
        if len(primaries) != 1:
            raise ValueError(f"topology needs exactly one primary, got {len(primaries)}")
        # A port may repeat across hosts (the multi-host layout uses the
        # default port everywhere) but not within one.
        endpoints = [(i.host.ip if i.host else None, i.port) for i in self.instances]
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(f"instance ports must be unique per host, got {[i.port for i in self.instances]}")

    @property
    def primary(self) -> InstanceSpec:
        """The single primary instance."""
        return next(i for i in self.instances if i.role == PRIMARY_ROLE)

    @property
    def replicas(self) -> list:
        """Replica instances in spec order."""
        return [i for i in self.instances if i.role == REPLICA_ROLE]

    @property
    def cluster_enabled(self) -> bool:
        """True when any instance declares cluster mode."""
        return any(i.cluster_enabled for i in self.instances)

    def hosts(self, default: ServerInfo) -> list:
        """Distinct hosts the layout touches, in first-appearance order."""
        seen: dict = {}
        for inst in self.instances:
            host = inst.resolved_host(default)
            seen.setdefault(host.ip, host)
        return list(seen.values())

    def to_dict(self) -> dict:
        """JSON-able form for the result row."""
        return {"instances": [i.to_dict() for i in self.instances]}

    @classmethod
    def standalone(
        cls,
        io_threads: int,
        *,
        server_args: str = "",
        cpu_override: str = "",
        port: int = DEFAULT_BASE_PORT,
        host: Optional[ServerInfo] = None,
    ) -> "TopologySpec":
        """One primary, no replicas, legacy working directory and logfile.

        This is the layout every single-instance task has always run; on the
        default port it produces the same ``valkey-server`` command line and
        the same files on the host as ``ReplicationGroup`` with one server.
        """
        return cls(
            instances=[
                InstanceSpec(
                    role=PRIMARY_ROLE,
                    port=port,
                    io_threads=io_threads,
                    server_args=server_args,
                    cpu_override=cpu_override,
                    host=host,
                    own_dir=False,
                )
            ]
        )

    @classmethod
    def replica_read(
        cls,
        *,
        replicas: int,
        replica_io_threads: int,
        primary_io_threads: int = 1,
        server_args: str = "",
        primary_args: str = "",
        replica_args: str = "",
        base_port: int = DEFAULT_BASE_PORT,
    ) -> "TopologySpec":
        """Primary on ``base_port`` plus ``replicas`` replicas on the following ports.

        ``server_args`` applies to every instance; ``primary_args`` /
        ``replica_args`` are appended after it for their role, so a role-specific
        flag (e.g. enabling speculation only on the replica) overrides a shared
        one -- valkey applies later command-line config over earlier.
        """
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")
        instances = [
            InstanceSpec(
                role=PRIMARY_ROLE,
                port=base_port,
                io_threads=primary_io_threads,
                server_args=" ".join(a for a in (server_args, primary_args) if a),
            )
        ]
        for n in range(replicas):
            instances.append(
                InstanceSpec(
                    role=REPLICA_ROLE,
                    port=base_port + 1 + n,
                    io_threads=replica_io_threads,
                    server_args=" ".join(a for a in (server_args, replica_args) if a),
                )
            )
        return cls(instances=instances)


class TopologyGroup:
    """Bring up a ``TopologySpec``, wire replication, sample, tear down.

    ``host`` is where instances without an explicit ``host`` run (normally the
    runner itself). Instances start one at a time: the build cache is keyed by
    source/specifier/make_args and the build runs in the host's shared source
    tree, so the first instance performs the build and the rest hit the cache.
    """

    def __init__(
        self,
        host: ServerInfo,
        spec: TopologySpec,
        binary_source: str,
        specifier: str,
        make_args: str = "",
    ) -> None:
        self.host = host
        self.spec = spec
        self.binary_source = binary_source
        self.specifier = specifier
        self.make_args = make_args

        self.servers: list[Server] = []
        self.primary: Optional[Server] = None
        self.replicas: list[Server] = []
        # spec instance -> running server, in spec order
        self.by_port: dict[int, Server] = {}

    def hosts(self) -> list:
        """Distinct hosts this topology touches."""
        return self.spec.hosts(self.host)

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Build (once, cached) and start every instance; leave each a primary.

        Every instance is started as a standalone primary (``REPLICAOF NO ONE``)
        and any replica still attached from an earlier run is detached, so
        ``begin_replication`` always starts from a known shape.
        """
        if self.spec.cluster_enabled:
            raise NotImplementedError(
                "cluster_enabled is declared in the spec but the cluster bootstrap "
                "(ADDSLOTS/MEET/REPLICATE) is not implemented yet; run non-cluster"
            )
        # ``self.servers`` is filled incrementally so a failure partway through
        # still lets ``stop_all_servers`` tear down what did start.
        self.servers = []
        self.primary = await self._start_instance(self.spec.primary)
        self.replicas = []
        for spec in self.spec.replicas:
            self.replicas.append(await self._start_instance(spec))
        self.by_port = {s.port: s for s in self.servers}
        await self._detach_unknown_replicas()

    async def _start_instance(self, inst: InstanceSpec) -> Server:
        host = inst.resolved_host(self.host)
        server = Server(host.ip, inst.port, host.username, instance_dir=inst.instance_dir)
        cached_binary_path = await server.ensure_binary_cached(self.binary_source, self.specifier, self.make_args)
        self.servers.append(server)  # registered before start so a failed start is still stopped
        await server.start(
            cached_binary_path,
            inst.io_threads,
            server_cpu_override=inst.cpu_override,
            server_args=inst.server_args,
        )
        await server.replicate(None)
        await server.wait_until_ready()
        return server

    async def _detach_unknown_replicas(self) -> None:
        """Detach replicas attached to our instances that are not part of this topology.

        A previous run's replica on another host can still be pointed at a
        primary we just restarted. It would receive our writes and skew the
        primary's replication work, so it is told ``REPLICAOF NO ONE``.
        """
        expected_ips = {s.ip for s in self.servers}
        unknown = await self._unknown_replica_ips(expected_ips)
        if not unknown:
            return
        logger.warning("Unexpected replicas found: %s", unknown)
        for ip in unknown:
            await Server(ip).replicate(None)
        await asyncio.sleep(0.5)  # let the primaries notice the disconnect
        unknown = await self._unknown_replica_ips(expected_ips)
        if unknown:
            raise RuntimeError(f"Unexpected replicas remain after cleanup: {unknown}")

    async def _unknown_replica_ips(self, expected_ips: set) -> list:
        unknown: list = []
        for server in self.servers:
            unknown += [ip for ip in await server.get_replicas() if ip not in expected_ips]
        return unknown

    async def begin_replication(self) -> None:
        """Point every replica at the primary (REPLICAOF host port)."""
        if self.primary is None:
            raise RuntimeError("topology not started")
        primary = self.primary
        await asyncio.gather(*[r.replicate(primary.ip, str(primary.port)) for r in self.replicas])

    async def end_replication(self) -> None:
        """Make every instance a standalone primary again."""
        await asyncio.gather(*[s.replicate(None) for s in self.servers])

    async def wait_for_repl_sync(self, timeout_s: float = 600.0) -> None:
        """Block until every replica reports link up and no sync in progress."""
        deadline = time.monotonic() + timeout_s
        for replica in self.replicas:
            while True:
                info = await replica.info("replication")
                if info.get("master_link_status") == "up" and info.get("master_sync_in_progress") == "0":
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"replica {replica.ip}:{replica.port} did not sync within {timeout_s}s")
                await asyncio.sleep(1)

    async def wait_for_offsets_caught_up(self, timeout_s: float = 300.0) -> None:
        """Block until every replica's processed offset equals the primary's.

        Used after a preload so the measurement starts from a fully replicated
        dataset (a reader would otherwise see misses on keys still in flight).
        """
        if self.primary is None:
            raise RuntimeError("topology not started")
        deadline = time.monotonic() + timeout_s
        while True:
            primary_offset = int((await self.primary.info("replication")).get("master_repl_offset", "0"))
            lagging = []
            for replica in self.replicas:
                replica_offset = int((await replica.info("replication")).get("master_repl_offset", "0"))
                if replica_offset < primary_offset:
                    lagging.append((replica.port, primary_offset - replica_offset))
            if not lagging:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"replicas still behind primary after {timeout_s}s: {lagging}")
            await asyncio.sleep(0.5)

    async def kill_all_valkey_instances(self) -> None:
        """Kill every valkey-server on every host the topology touches."""
        await asyncio.gather(
            *[Server(h.ip, username=h.username).kill_all_valkey_instances_on_host() for h in self.hosts()]
        )
        await asyncio.sleep(1)

    async def stop_all_servers(self) -> None:
        """Stop every instance and release its CPUs."""
        await asyncio.gather(*[s.stop() for s in self.servers])

    # ------------------------------------------------------------------ sampling

    async def sample(self, extra_fields: Optional[list] = None, cpu: bool = True) -> dict:
        """One ``INFO replication stats`` snapshot per instance.

        All instances are queried from ONE remote shell invocation, back to
        back, so the skew between the primary's and a replica's offset reading
        is a few milliseconds of local valkey-cli connect time rather than one
        SSH round trip per instance. Lag resolution is therefore about
        ``skew_seconds * replication_bytes_per_second``; treat lags below that
        as zero.

        Returns ``{"t": monotonic_seconds, "instances": {port: {...}}}`` with the
        replication offsets, ops/sec, and any ``extra_fields`` present (missing
        fields are silently omitted so a stock build and a feature build sample
        identically).

        With ``cpu=True`` the same shell also emits the host's per-core
        ``/proc/stat`` counters and every instance's per-thread CPU ticks, so
        the sample gains ``"cores"`` (``{cpu: {busy, idle, softirq, total}}``)
        and ``"threads"`` (``{port: {tid: {comm, ticks}}}``). These are
        cumulative; ``cpu_sampling`` turns consecutive samples into utilisation.
        """
        if self.primary is None:
            raise RuntimeError("topology not started")
        if len(self.hosts()) > 1:
            raise NotImplementedError(
                "sample() reads every instance from one shell; multi-host sampling is not implemented"
            )
        extra_fields = extra_fields or []
        cli = str(config.PROJECT_ROOT / config.VALKEY_CLI)
        parts = [
            f"echo '{_SAMPLE_SEPARATOR}{s.port}'; {cli} -h {s.ip} -p {s.port} info replication stats"
            for s in self.servers
        ]
        if cpu:
            parts.append(cpu_sampling.cpu_sample_command({s.port: s.valkey_pid for s in self.servers}))
        t = time.monotonic()
        out, _ = await self.primary.run_host_command("; ".join(parts), check=False)
        if not cpu:
            return {"t": t, "instances": parse_multi_info(out, extra_fields)}
        info_lines, procstat_lines, thread_lines = cpu_sampling.split_sections(out, _SAMPLE_SEPARATOR)
        return {
            "t": t,
            "instances": parse_multi_info("\n".join(info_lines), extra_fields),
            "cores": cpu_sampling.parse_proc_stat(procstat_lines),
            "threads": {port: cpu_sampling.parse_thread_stats(lines) for port, lines in thread_lines.items()},
        }

    def pids(self) -> dict:
        """Main pid per instance port (``-1`` for an instance that is not running)."""
        return {s.port: s.valkey_pid for s in self.servers}


_SAMPLE_SEPARATOR = "=== conductress-instance "


def parse_multi_info(output: str, extra_fields: Optional[list] = None) -> dict:
    """Parse the concatenated per-instance INFO output produced by ``sample``."""
    extra_fields = extra_fields or []
    instances: dict = {}
    port: Optional[int] = None
    info: dict = {}

    def flush():
        if port is None:
            return
        row = {
            "role": info.get("role"),
            "master_repl_offset": _to_int(info.get("master_repl_offset")),
            "instantaneous_ops_per_sec": _to_int(info.get("instantaneous_ops_per_sec")),
            "total_commands_processed": _to_int(info.get("total_commands_processed")),
            "total_reads_processed": _to_int(info.get("total_reads_processed")),
            # Cumulative main-thread event-loop time (usec) and iteration count.
            # Unlike CPU time these are not inflated by spin-waiting, so they
            # are what cpu_sampling reports as the server's duty.
            "eventloop_cycles": _to_int(info.get("eventloop_cycles")),
            "eventloop_duration_sum": _to_int(info.get("eventloop_duration_sum")),
            "eventloop_duration_cmd_sum": _to_int(info.get("eventloop_duration_cmd_sum")),
        }
        if info.get("role") == "slave":
            row["master_link_status"] = info.get("master_link_status")
            row["master_last_io_seconds_ago"] = _to_int(info.get("master_last_io_seconds_ago"))
        for name in extra_fields:
            if name in info:
                row[name] = _to_number(info[name])
        instances[port] = row

    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith(_SAMPLE_SEPARATOR):
            flush()
            port = int(line[len(_SAMPLE_SEPARATOR) :])
            info = {}
        elif ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    flush()
    return instances


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_number(value: str):
    """INFO values are strings; keep ints as ints, floats as floats, else raw."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def replication_lag_stats(samples: list, primary_port: int, replica_port: int) -> dict:
    """Summarise primary-minus-replica offset deltas (bytes) across samples."""
    lags = []
    for sample in samples:
        inst = sample["instances"]
        p = inst.get(primary_port, {}).get("master_repl_offset")
        r = inst.get(replica_port, {}).get("master_repl_offset")
        if p is None or r is None:
            continue
        lags.append(max(0, p - r))
    if not lags:
        return {"samples": 0}
    lags_sorted = sorted(lags)
    return {
        "samples": len(lags),
        "mean_bytes": sum(lags) / len(lags),
        "max_bytes": lags_sorted[-1],
        "p99_bytes": lags_sorted[min(len(lags_sorted) - 1, int(round(0.99 * (len(lags_sorted) - 1))))],
    }
