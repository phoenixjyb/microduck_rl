# Genuine CPU scheduled diagonal diagnostic: passed, not admission

Exact source `1ae0a2956e10626679cabcac3a18c5c40ec383c3`, branch
`feat/athletics-obstacle-curriculum`, on the frozen 100.98 WSL environment.
This closes the [predeclared CPU diagnostic](2026-10-03-progressive-recovery-preparation.md)
after the [five-direction CUDA baseline](2026-10-03-frozen-five-direction-recovery-baseline.md).
It is two genuine fresh CPU first attempts with the unchanged policy, not PPO
training or a full held-out recovery qualification.

## Exact-source qualification and execution

The verified Git bundle SHA256 was
`c5237fbfe00b6500c357f65a31bcb19e9d8686a45cf6ef2edc74bf1f91cdffb6`.
The detached WSL source snapshot imported its own new module and remained clean.
The combined default-CPU suite passed **596 tests in 60.27 s** under the
150-second/6-GiB/200%-CPU/Nice-10/control-group cap; service invocation
`77369e0e177443bfa9868a628d74789a`, runtime **72.808 s**.

Six fresh portable-CPU services, each capped at 120 seconds/2 GiB, passed
**193 tests**: parent 24, lesson plan 68, schedule 39, runtime 17, trace 25,
probe 20. Their invocation IDs were respectively:

- `e242215b6ef84ff9ae3d6bf2e7c24b81`
- `c737c16334d54572a1283cd83729ec39`
- `9a9a2ca907c34dfd920fc8dff89e065e`
- `4fc23de3774546519b378764fa6b2db9`
- `510af531f6fd47f0a8aa11a30c9bd629`
- `e7648979d33b437fba1f317cd52ce70c`

All actual portable services started with `CUDA_VISIBLE_DEVICES` empty,
`ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`, `OMP_NUM_THREADS=1`,
and `MICRODUCK_STANCE_PROFILE=wsl-10098-20260930`. Torch 2.9.1 reported
CPU capability DEFAULT and one CPU thread; actual libtorch_cpu SHA256
`7918fc09ff644c9b667921100b924e33ea432c85737c2982b56283691b32a4d7`.
The short test-service memory peaks are implausibly small; they are not used
as measurements of interpreter/RSS requirements.

The main worktree then fast-forwarded under the existing shared GPU lease,
after two idle samples: no compute PID, 0% GPU utilization, 663 MiB, 30 C.
Transition invocation `eb3b7a5fefd8435a8bd54e21efb5b6d4` finished successfully.
No dependency, driver, protected service or FilmBrain process was changed.

Probe service `microduck-cpu-scheduled-diagonal-1ae0a2956e10.service`,
invocation `3d613c2d64f840538cb86bdd640aafa3`, finished successfully before
07:16 Shanghai. Actual cap: **240 seconds / 2 GiB / 200% CPU / Nice 10 /
control-group**. Service runtime **90.880 s**, reported memory peak **1.4 GiB**;
runner elapsed **83.38213752303272 s**. No optimizer step or CUDA allocation.
Warp's startup message that no CUDA-capable device was detected is retained in
the journal of this deliberately CUDA-hidden CPU run; it is not a numeric
simulation failure or evidence of a GPU capture.

## First-attempt results

Each fresh runtime was seeded with CPU seed 671 before construction; no
counter/support hooks, reset or replacement attempt. Both retained **250 policy
ticks / 2,500 accepted physics steps / five simulated seconds**, including all
ten pulse-window pre-solve/Euler/unforced-post records and complete force arrays.
The zero control has its declared window at step 500. The diagonal pulse is
2 N world-horizontal trunk-COM force for 20 ms, starting at step 375 (0.75 s).

| Evidence | Zero control | Held-out diagonal ++ |
| --- | ---: | ---: |
| Measured collection wall seconds | 27.262182326056063 | 27.348949366947636 |
| Recorded collection seconds | 27.103472015820444 | 27.19310799310915 |
| Maximum displacement m | 0.002404326805844903 | 0.002404326805844903 |
| Final-second planar speed p95 m/s | 0.00016992714595323043 | 0.00045266606578253643 |
| Final-second tilt p95 rad | 0.01976817660033703 | 0.02202187478542328 |
| Final-second minimum root height m | 0.11728814989328384 | 0.11727173626422882 |
| Both-feet support fraction | 1.0 | 1.0 |
| Soft-limit exposure fraction | 0.0 | 0.0 |
| Maximum motor torque Nm | 0.11519977450370789 | 0.11519977450370789 |
| Maximum absolute joint power W | 0.037765782326459885 | 0.037765782326459885 |

All nine unchanged numerical gates passed for both attempts; actor replay error
was exactly zero. Both authenticated pre-force prefixes through step 375,
including the first motor proposal inside policy tick 37, had SHA256
`9234439bc712ec88f0f84f9e8b5930de507c45d5314787d8be5334e79d0d8273`.
Post-push reactive actions were not forced to equal the control's actions.

The new loader also qualified the real retained 256,368-byte parent: both
trainable CPU actor/critic groups restored exactly, combined state SHA256
`e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329`,
CPU RNG unchanged, actor outputs bit-equal to the separate frozen evaluation
loader on the declared synthetic shadow input. No optimizer, rollout storage,
simulator state or training RNG was restored; that shadow is not a physical test.

## Separate fresh CPU closeout

Fresh CPU service `microduck-cpu-scheduled-independent-1ae0a2956e10.service`,
invocation `f30805d20b174a6fa15ba602be1410cd`, independently rehashed every
artifact, reloaded the exact parent, rescored both complete raw traces and
matched the original receipts and prefix decision exactly. Cap 120 seconds/
2 GiB/200% CPU/Nice 10/control-group; runtime **33.162 s**, check elapsed
**26.232393242185935 s**. It also rechecked the closed prerequisite inventories,
actual CPU profile, clean exact source and unchanged FilmBrain/protected services.

Decision: **`cpu-scheduled-diagonal-probe-passed`**. All eight training/capability
admission flags remain false. This is recorded actor/plant/control/force/terminal
consistency replay, not full fresh trajectory physics re-simulation, independent
GPU attestation, BAM-output recomputation or thermal-model qualification. One
fixed held-out seed/cell does not establish randomized recovery, the full
held-out matrix, ball balancing, hopping retention or real hardware capability.

## Durable evidence

WSL directory:
`/home/yanbo/work/microduck_rl-stance-replication-20260930/artifacts/evaluations/stance-wsl-cpu-scheduled-diagonal-1ae0a2956e10`.

| Artifact | SHA256 |
| --- | --- |
| launch.json | `5f3d479858f41713dc2f6ebefceb490ed23644b57b775fb2511bfbee09ff609e` |
| report.json | `66ae93b3da60a27d32e0412d78724e4683289bb5ffdae8b12209847e2df93aea` |
| case-0.pt (37,296,584 bytes) | `a0020b84acacf67f00da5f2dfdc9c6b966675654dce1f109d79ab833093a08a2` |
| case-1.pt (37,305,992 bytes) | `8607a51c61498b329b1aa2fbc3c163433efcdb6903a1cebabd06925a7adf3142` |
| independent-closeout.json | `168738a55372d46734b360ce4ebe4a99c0660ebeb7f5ebe6d61c785644ccaac5` |

The closeout contains the complete original nine-file inventory, including
checkpoint bytes, case metadata and replay receipts. Keep the original records
immutable and retain the independent closeout separately. Next work remains the
bounded real-policy/fresh-optimizer preparation and a separately predeclared
frozen-policy timing/dose screen; do not train already-passing cells or infer
64-world GPU throughput from these one-world CPU attempts.

A complete byte-identical Mac mirror was verified: **10 files / 74,895,814
bytes**, at
`artifacts/retained/cpu-scheduled-diagonal-1ae0a2956e10.2Sccuu/stance-wsl-cpu-scheduled-diagonal-1ae0a2956e10`.
This mirror includes both full raw traces; its byte verification is not another
physics replay. The older five-direction CUDA raw traces still remain on WSL,
separate from their previously retained small Mac metadata mirror.
