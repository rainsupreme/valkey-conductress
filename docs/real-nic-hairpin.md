# Real-NIC hairpin topology (`--client-netns`)

By default, local benchmarks run the load generator and server on the same
host over loopback. Loopback is the most stable harness for **relative,
server-CPU-side** comparisons (its low noise floor is what makes tight A/Bs
possible) — but it does not exercise the real network path: no NIC driver, no
hardware interrupts, no NAPI/softirq, different latency profile.

The **dual-ENI hairpin** keeps everything on one host while sending traffic
through the real ENA path in both directions:

```
  [loadgen netns]  ENI-B ──► VPC fabric ──► ENI-A  [default netns]
   valkey-benchmark                          valkey-server (+ SSH)
```

- The generator runs inside the `loadgen` network namespace, which owns the
  secondary ENI. Namespace separation is load-bearing: without it, Linux sees
  both private IPs as local and short-circuits via the `local` routing table.
- The generator targets the **primary ENI's private IP** (conductress rewrites
  a localhost target automatically when `--client-netns` is set).
- Both directions traverse real driver/IRQ/NAPI machinery. Measured on a
  Graviton 4 host: ~99µs fabric RTT (vs ~15µs loopback) and 400K+ hardware
  interrupts on the secondary ENI during a short burst.

## When to use which path

| Question | Path |
|---|---|
| Server-CPU A/Bs, bisects, per-core gates | loopback (default) — lowest noise |
| Kernel-path work: IRQ steering, busy-poll, epoll composition, absolute ceilings | `--client-netns loadgen` |

Topology is part of the workload definition: never compare a hairpin cell
against a loopback cell. Sweep history is loopback; any default change would
be a deliberate step-change with overlap cells.

## One-time host setup

1. Create a secondary ENI in the instance's subnet and attach it (device
   index 1). The ENI needs security groups that admit intra-SG traffic — on
   these instances that means BOTH `default + web server` AND `default` (the latter
   carries the self-referencing allow-all rule; without it the hairpin
   silently drops).

   ```
   aws ec2 create-network-interface --subnet-id <subnet> \
       --groups <sg-webserver> <sg-default> \
       --description "<host> netns load-generator ENI"
   aws ec2 attach-network-interface --device-index 1 \
       --instance-id <instance> --network-interface-id <eni>
   ```

2. Run `scripts/setup-loadgen-netns.sh` on the host (idempotent):

   ```
   sudo ./setup-loadgen-netns.sh <ENI_MAC> <ENI_IP>/20 <SUBNET_GATEWAY>
   ```

   It finds the interface by MAC (kernel names differ per host), moves it into
   the `loadgen` namespace, assigns the IP, and sets the default route.

3. **Reboot note: network namespaces do NOT persist** (the ENI stays attached;
   only the namespace config is lost). Two options:
   - Install the systemd unit for automatic recreation at boot — see the
     install steps in `scripts/loadgen-netns.service` (uses
     `/etc/loadgen-netns.conf` for this host's MAC/IP/gateway).
   - Or re-run `setup-loadgen-netns.sh` manually after reboots.

   Either way, a hairpin task on a host whose namespace is missing fails
   immediately at preflight with a pointer here — not with a cryptic
   benchmark error mid-run.

Each host's ENI ID, private IP, MAC and subnet gateway are deployment details.
Keep them in that host's `/etc/loadgen-netns.conf` (which the systemd unit
already reads) or in local notes, not in this document.

## Usage

```
conductress queue add --tests get --sizes 16 --io-threads 8 \
    --pipelining 10 --client-netns loadgen --note '[myexp:hairpin]'
```

Notes:
- Preload/population still runs over loopback (correctness only — the timed
  measurement is what traverses the ENI path).
- `sudo ip netns exec` wraps the numactl invocation; CPU pinning behaves
  identically inside a namespace.
- Validation that traffic really takes the NIC: watch `/proc/interrupts`
  deltas for the secondary ENI's vectors during a run.
- Why not the public IP? It also traverses the NIC (IGW NAT hairpin) but is
  billed as regional data transfer (~$0.01/GB/direction — tens of $/hour at
  benchmark volumes) and adds IGW variability.
