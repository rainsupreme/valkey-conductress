# Binary layout noise in per-commit throughput series

Rebuilding Valkey at a different commit changes where the linker places functions in the binary. On Graviton 3 that alone moves GET throughput by several percent, and by 12% in the worst case we measured. Most real per-commit changes are smaller than that. A throughput swing on a stock build is therefore not evidence of a regression, or of an improvement, until the layout has been ruled out. This page is a record of one measurement campaign, in September 2026, and of what it implies for reading a per-commit series.

## What we measured

Sep 8-18 2026, GET 16-byte keys and values, io-threads 7, pipeline 10, stock `-O3 -flto` builds, one per-commit series per platform. A delta is the change in mean throughput from one measured commit to the next measured commit on that platform's series; usually the two are one commit apart, and where the sweep skipped commits the span is longer.

| Platform | Median CV | Deltas in window | Deltas above 3% | Largest |
|---|---|---|---|---|
| Graviton 3 | 0.90% | 44 | 9 | 12.1% |
| Graviton 4 | 0.45% | 41 | 4 | 4.0% |
| AMD (Zen 4) | 1.90% | 39 | 0 | 2.2% |
| Intel (Sapphire Rapids) | 2.31% | 37 | 0 | 2.5% |

Twelve distinct commits swung by more than 3% on some platform. Each one, on every platform that measured it (dash: not measured there):

| Commit or PR | Graviton 3 | Graviton 4 | AMD | Intel | Reproduced? |
|---|---|---|---|---|---|
| 4257 afterCommand gate | +12.1% | +3.7% | - | +2.5% | yes, at about 3% |
| 3940 nested prefetching, hash and zset | -8.9% | -1.0% | +0.7% | -1.6% | no; Graviton 4 shows a real 1% |
| 3438 cluster bus IO offload | +7.0% | -0.9% | -1.0% | - | no |
| 3212 listpack hash-field expiration | +4.9% | +1.2% | -1.4% | +1.0% | no |
| 4be324e8 fullsync streaming compression | -4.7% | - | - | - | not measured elsewhere |
| 4487 module defrag callback | -4.7% | -1.6% | +2.0% | - | no; fixed-layout pair +0.3% |
| 4076 QoS for cluster events | +4.5% | -0.9% | -0.3% | +2.2% | no |
| 4702 LZ4 context accounting | -4.5% | -1.0% | -1.4% | -0.2% | no; fixed-layout pair -1.4% |
| 4670 CLUSTER SYNCSLOTS guard | -3.8% | - | - | - | not measured elsewhere |
| 3967 ACL role support | +0.3% | +4.0% | -1.6% | +0.3% | no |
| 2972 SISMEMBER XX option | +0.3% | -3.2% | +0.9% | -0.2% | no |
| 4019 EXEC IFEQ, IFNE, XX, NX | - | +3.1% | - | - | not measured elsewhere |

Reproduced means the same sign, at least a third of the size, and significant on at least one other platform. Nine of the twelve can be checked; one reproduces, and that one (PR 4257) is a deliberate hot-path change measured at a quarter of its Graviton 3 size everywhere else. The eight that do not reproduce attach to changes in cluster, module, hash, set, ACL and memory-accounting code. Two of them may carry a small real cost under the swing: PR 3940 touches the prefetch machinery GET uses, and PR 4702 accounts memory per client. Neither is a 4-9% change.

Re-measuring parent and child with the code layout held fixed confirmed the picture on Graviton 3: PR 4487 went from -4.7% to +0.3%, PR 4702 from -4.5% to -1.4%, and PR 4257 from +12% to +3.3%. Hardware counters give the mechanism. Instructions per request were constant to within 2% across every cell; cycles per request tracked the frontend-stall fraction; backend stall did not move. Same work, different instruction-fetch cost.

## What it means for reading a series

- A swing on one platform, attached to a PR that cannot touch the hot loop, is layout until shown otherwise. Check the other platforms first. That costs nothing, and it settled nine of the twelve swings above.
- On Graviton 3 GET, a stock-build delta below about 6% is unconfirmed; on Graviton 4, below about 4%. The x86 hosts showed no such floor at this workload, but at 2% CV they cannot confirm a real 3% effect either.
- A real change can have its size misreported by three to four times. PR 4257 is +3% on three independent measurements and +12% on the one stock Graviton 3 point.
- Instructions per request and frontend-stall fraction, recorded alongside throughput, classify most cases without a re-run. Flat instructions with moving frontend stall is layout. Moving instructions is work.

## Controlling for it

Both controls are a `--make-args` string on any cell, no source change; the build cache keys on the flags, so stock and controlled builds of one commit coexist.

- **Name-sorted sections**: `OPTIMIZATION="-O3 -flto=auto -ffunction-sections -fdata-sections" LDFLAGS="-Wl,--sort-section=name"`. Works with GNU ld.bfd.
- **Profile-guided layout**: the same flags plus `-fprofile-use=DIR -fprofile-prefix-path=TREE`, with one profile trained on the parent and applied to both builds.

Neither is a clean control. Name-sort is sensitive to insertions: on AMD, a change that adds functions read -3% under it while the stock pair read 0%. A parent-trained profile is biased against changes to hot functions: PR 4257 read -2.5% under it. Use name-sort when the change adds no functions to hot translation units, the profile when it touches no hot function, and report the stock delta alongside either. The stock delta is what a default build will see; the controlled delta is the change's own cost; when they disagree, the disagreement is the finding.

Two approaches that avoid both failure modes have not been tried here: relaying out both binaries post-link from one profile (BOLT), and measuring each commit under several randomized layouts and averaging (the Stabilizer method).

A longer write-up with the per-cell task IDs, the full counter tables and the cross-platform PR matrix is kept outside the repository and is available on request.
