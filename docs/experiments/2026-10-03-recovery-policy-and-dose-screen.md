# Fresh policy wiring and bounded frozen-parent dose screen

## Real-parent policy preparation: passed, not training

Source `d6a37650bfb5a4d0c4bc7392589809db432feeb1`, exact feature branch
`feat/athletics-obstacle-curriculum`, frozen 100.98 WSL dependencies.
This follows the [genuine scheduled diagonal diagnostic](2026-10-03-cpu-scheduled-diagonal-diagnostic.md).

The actual retained actor and critic were wired into fresh FinitePPO and
24-step rollout storage for seeds 653/659 and 2/64 environments. All four
combinations retained exact initial combined state SHA256
`e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329`,
the exact actor/critic parameter union in a new empty Adam optimizer, empty
storage, and unchanged global CPU RNG. The actor was bit-equal to the separate
frozen loader on the declared synthetic shadow input. The private learner RNG
is **not connected to a sampler**. No environment, rollout or optimizer step
was run; all eight admission flags remain false.

Service `microduck-real-recovery-policy-prepare-d6a37650bfb5.service`, invocation
`c3906171dc2744d39c641b33023f289d`, exited successfully under a 120-second /
2-GiB / 200%-CPU / Nice-10 / control-group cap. Runtime 8.818 s; check elapsed
1.8654629320371896 s; memory peak 526 MiB. FilmBrain was unchanged, both
protected AI Mission services remained inactive, and CUDA was hidden.

Durable WSL directory:
`/home/yanbo/work/microduck_rl-stance-replication-20260930/artifacts/evaluations/stance-wsl-recovery-policy-preparation-d6a37650bfb5`.
Three files, 280,726 bytes:

| Artifact | SHA256 |
| --- | --- |
| checkpoint.pt | `2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5` |
| launch.json | `2dac373a3a0eaefa4f997bc5035bdc5d21beddc10f5bf04a0454653b618f0ceb` |
| report.json | `a3543a7b42b6d2bf25c7c913f17179ca0bc957a1f5adc5251714acff50088ff5` |

The Mac combined suite passed 656 tests. A WSL-profile run retained a real
scope failure: 651 passed, one skipped, four failed (60.67 s; invocation
`ffd846508e004a01a22ff591e6274012`). Those four unchanged legacy PPO/lean
fixtures hard-code the historical 100.100 execution window and parent, rather
than the selected WSL profile. No caps, dependencies, fixtures or runtime
checks were relaxed. Separate fresh processes then passed 621 WSL-relevant
tests (61.77 s, invocation `016672db3315475b8a7f5c36c8c9b5af`), and the
historical-profile fixtures passed 34 with one skip (1.47 s, invocation
`8040faf507184470a1ee48ccb89938ee`). Fresh portable-CPU policy tests passed
25 (0.09 s, invocation `c1d6796d125741db87c568bcbd7c693a`). These are distinct
profile-scoped checks, not a claim that the failed combined WSL run passed.

## Predeclared next diagnostic, before source commit or execution

Protocol `football-b1d-cpu-dose-screen-v1` uses the unchanged frozen parent;
it cannot train or admit a capability. Three separately constructed, freshly
seeded CPU runtimes (seed 671), one environment each, five simulated seconds,
250 policy ticks / 2,500 Euler steps, no reset or replacement attempt:

1. Zero wrench, control window at step 500.
2. World +x trunk-COM 2 N for 20 physics steps (40 ms), onset step 250.
3. World +x trunk-COM 4 N for 10 physics steps (20 ms), onset step 250.

The two doses have the same 0.08 N-s impulse but different peak force and
duration; evaluate them separately. Compare all authenticated pre-push state
and actor prefixes through step 250, then permit reactive actions to differ.
The previously closed diagonal evidence and its exact ten-file inventory must
rehash first; its independent-closeout SHA256 is
`168738a55372d46734b360ce4ebe4a99c0660ebeb7f5ebe6d61c785644ccaac5`.
Use the unchanged nine first-attempt numerical gates and full force/phase
checks. No temperature/thermal claim or fresh whole-trajectory re-simulation
is implied by recorded CPU consistency replay.

The unique source-derived user service is capped at **180 seconds / 2 GiB /
200% CPU / Nice 10 / control-group**. Each collection has a strict 60-second
wall and recorded-time cap. The prior actual two-case service took 90.880 s;
three cases at 1.25 safety factor imply 170.4 s, rounded up to 180. A new case
requires elapsed time below 100 s, reserving 60 s collection and 20 s closeout;
the observed independent two-case rescore was 26.232 s, so a one-case replay
at 1.25 factor is 16.4 s, rounded to 20. Neither estimate qualifies GPU
throughput. Launch requires **strictly more than 360 seconds** before the
fixed **2026-10-03 08:00 Asia/Shanghai** cutoff, reserving the 180-second service,
120-second separate CPU closeout and 60-second margin. Do not extend a cap or
cutoff after failure.

Retain raw captures and replay receipts before judging elapsed/terminal gates.
A complete first-attempt physical failure may be followed by the next fresh
cell, never retried. Partial or over-budget collection aborts; earlier durable
evidence remains. A failed zero control cannot establish a dose deficit.
Different prefixes or incomplete force/phase delivery make the result
inconclusive. Otherwise report either `frozen-dose-screen-no-deficit` or
`frozen-dose-screen-measured-deficit`, with all admission flags false.

Before execution: focused source tests, fresh exact-source WSL default and
portable CPU tests, clean branch, and guarded source fast-forward under the
existing shared lease. Preserve FilmBrain and protected services. On completion:
independently rehash and rescore all three raw captures in a fresh capped CPU
process and record its deterministic decision. Do not train already-passing
cells or start a long GPU job, hopping promotion, ball balance, video, raw
perception, actor-observation expansion or physical motion in this diagnostic.
