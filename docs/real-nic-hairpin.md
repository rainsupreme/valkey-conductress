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
- Both directions traverse real driver/IRQ/NAPI machinery. Validated on
  g4bench: ~99µs fabric RTT (vs ~15µs loopback) and 400K+ hardware interrupts
  on the secondary ENI during a short burst.

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
   our hosts that means BOTH `default + web server` AND `default` (the latter
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

3. **Reboot note: network namespaces do NOT persist.** Re-run the script after
   any host reboot (the ENI stays attached; only the namespace config is lost).

## Provisioned hosts (Aug 2026)

| Host | ENI | netns IP | MAC | Gateway |
|---|---|---|---|---|
| g4bench | eni-06b9b580692985dbe | 172.31.45.237 | 0e:5c:94:9d:0e:73 | 172.31.32.1 |
| bench | eni-0a2b2cb2324b7f8a1 | 172.31.40.160 | 0e:9d:8f:cd:cf:e1 | 172.31.32.1 |
| armbench | eni-02dbc99b6f9e561e9 | 172.31.65.103 | 16:ff:ee:8c:9e:69 | 172.31.64.1 |
| intelbench | eni-092c9b0568abf0145 | 172.31.69.61 | 16:ff:dc:03:32:b1 | 172.31.64.1 |

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
