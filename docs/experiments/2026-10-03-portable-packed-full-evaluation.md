# Portable packed stance replication evaluation

This is a predeclaration for one complete evaluation of the three retained
packed-replication trainings. It is not new training and has not run yet. The
owner authorized simulation, development and training until **2026-10-03 08:00
Asia/Shanghai**. This experiment answers the outstanding nominal replication
question before disturbance recovery or football work can advance.

Protocol: `football-b1n-wsl-portable-packed-full-evaluation-v1`. Trace namespace:
`football-b1n-portable-packed-full-evaluation-trace-v1`. The original one-case
probe namespace and its checkpoint-255/evaluation-541 restrictions stay unchanged.
The original [replication decision rule](2026-09-17-stance-lean-replication.md)
and every numerical gate remain unchanged.

## Evidence permitting this evaluation

The [R4 capture and independent replay](2026-10-02-wsl-portable-probe-r4.md)
completed on source `07466b58021c084dbc2f688effa936d9466ba456`. Its one nominal
case produced 128/128 completed numerical passes and exact actor replay. All
14 files / 621,439,568 bytes were delivered to native Linux, authenticated before
tensor loading and independently replayed with CUDA hidden. That result permits
this new evaluation; it does not itself establish replication or a learned skill.

Pinned R4 inputs:

| Input | SHA256 |
| --- | --- |
| Launch | `dc2955e29f9dff489b77b2e8b19ecb596c6a1105f85c2cd200bcdf2e9172376c` |
| Completed report | `c75bbc4d43b977d50478840b362982199d3797490c53e68be57d821cfe54facf` |
| Whole 14-file byte inventory | `1587ad5e90b63ccfe06ff5a588185dab55b19c89ad6a52da06b858a90e8f2b62` |
| Independent replay receipt | `eed529043cb4af240cc922138e73ca383beda9dd79f4805a17ce3fd962100406` |

The original archive remains untouched. The independently retained receipt is
copied separately to `artifacts/tools/r4-native-independent-replay-07466b58021c.json`
on WSL and checked against its exact digest. Preparation also reruns complete
R4 plan/bundle/score verification, not merely receipt inspection.

## Frozen matrix and historical authentication

Original training source: `be2d59661af293b0d67ae20d2e16db50514cce14`.

| Training seed | Exact completed training report SHA256 |
| --- | --- |
| 577 | `2a5a3ec503bcf39ed89d4f3a82cea8b2bd3974741313f3c7cd2ab8d97280b5de` |
| 587 | `202d9b31aa27295c18a7c78086505ec7d821bc69d0568224def3bbe754ae9f05` |
| 593 | `5ee3b60dab8c8e051d4fc4205b21302a1c85b12ebf25b4d4e57a6e00a6973d5c` |

For each training seed, evaluate **64, 128, 192, 255** in that order, each at
evaluation seeds **541, 547, 557**, with **128 worlds**: exactly **36 cases /
4,608 first attempts**. Resolve each training directory by its declared source,
purpose and seed, not a glob or latest-run heuristic. Bind every checkpoint to
its original launch/runtime identities and exact hash.

Authenticate all three entire original training archives under the original
AVX2 profile, including initializer reconstruction and every retained update,
before any capture and after completion. Each isolated CPU subprocess keeps the
existing **60-second / 1-MiB combined output** bounds. Three-seed authentication
has its own receipt namespace; the old seed-577 authentication is not widened.
Original AVX2 authentication is not portable actor replay or GPU attestation.

The evaluation changes no weights, PPO, reset, reward, observation, action,
delay, plant, motor, scorer or tolerance. Actor remains the frozen 44D/10-action
CPU policy; physics remains the R4 eager packed CUDA0 path with graphs disabled.
Use the exact portable CPU profile at process start: DEFAULT capability,
compatible MKL, one CPU thread and the pinned Torch CPU library. Each fresh world
has one nominal 5-second first attempt, capped at 250 policy ticks / 2,500 physics
steps. No auto-reset. Early failures close their first attempts and count as
failures; wall-budget prefixes never become complete cases. Evaluation seeds
here measure nominal repeatability, not randomized obstacle generalization.

## Measured budget and fixed execution window

Window: **2026-10-03 00:30–08:00 Asia/Shanghai**. The R4 measurements are the
same full-length case path, worlds and tick count, not an analogy to a different
lesson. Extend only the number of cases to 36. One sample is not a measured
worst-case bound; the hard watchdog and unchanged 1.25 safety factor carry that
uncertainty. Slowdown is reported as incomplete, not granted more time.

| Component | Seconds |
| --- | --- |
| R4 prelude | 21.741637519095093 |
| R4 environment construction | 2.2584270699881017 |
| R4 case work including immediate bundle replay | 193.14527830597945 |
| R4 supervisor replay | 9.91598303313367 |
| R4 unattributed overhead, conservatively repeated per case | 34.69382807682268 |
| Full repeating unit | 240.0135164859239 |
| Twelve bounded authentication subprocess reserves | 720 |
| Full prediction: prelude + 36 units + authentication reserves | 9382.228231012356 |
| Service: ceil(1.25 × prediction) | **11728** |
| Final parent replay reserve: ceil(1.25 × 36 × R4 replay) + 60 | 507 |
| Final parent authentication reserve | 180 |
| Child watchdog: service − 507 − 180 − 60 | **10981** |
| Post-service closeout reserve | 600 |
| Additional deadline margin | 60 |

The authentication reserve counts supervisor and child pre/post checks for all
three seeds. CPU preparation is separately capped at **360 seconds**, 6 GiB,
200% CPU, Nice 10, CUDA hidden; it must complete before GPU launch. Re-derive
the prediction and both caps from the exact R4 report on every plan and input
check; constants are checked transcriptions. The old probe caps are unchanged.

Preparation, supervisor and child entry refuse unless the entire **11,728-second
service + 600-second closeout + 60-second margin** fits before the cutoff.
Latest start is strictly before **04:33:32 Shanghai**, with repeated entry checks
allowed to refuse conservatively sooner. Systemd rendering independently checked
on WSL is `3h 15min 28s`; `KillMode=control-group`, 6 GiB, CPU quota 200%, Nice 10.
The inherited lease and separate watchdog bound the single child even if a native
call hangs. Stop only our owned process group on conflict; never other workloads.

## Host entry and retention

Only `gw98-direct`, clean exact `feat/athletics-obstacle-curriculum` in
`/home/yanbo/work/microduck_rl-stance-replication-20260930`, on the existing frozen
stack. Before installation, require hash-verified exact source, focused Mac and
Linux original/portable CPU suites, and source-qualified compiled plant evidence.
No dependency or driver change. Preserve 100.100's historical main checkout and
unresolved GPU mismatch; it remains CPU-only.

One sequential user service and one new source-unique output under
`artifacts/evaluations/stance-wsl-portable-full-evaluation-<source12>`. Require the
existing FilmBrain lease and two idle samples. Preserve FilmBrain service PIDs
and restart counts, protected AI-mission services inactive in both managers,
WSL temperature below 75°C, total GPU memory at most 5,120 MiB and at least
6,144 MiB free. Refuse another compute owner or frozen-host/source drift.

Retain every case's checkpoint, runtime, launch, control and raw trajectory
bytes, manifest, collection receipt and independently replayed score. Retain
ordered comparison, source/host/authentication/measurement hashes, child log,
ownership telemetry and final supervisor file hashes. Exclusive fsynced writes;
no overwrite, resume, new best-checkpoint search, automatic retry or cap extension.
Raw evidence is approximately 22.4 GB; WSL disk must have adequate free space.
Do not copy it onto the space-limited Mac.

The CPU-only supervisor replays all 36 bundles after the CUDA child and rederives
the complete matrix decision. Further retained verification must use the exact
source/launch/report/bytes and portable CPU profile. Same-host independent CPU
rederivation is not second-host replay or independent GPU attestation. A full
external-host copy is not assumed to fit the overnight network budget; do not
claim that delivery/replay tier until every byte and case has actually passed.

## Decision and next curriculum gate

Every attempt must pass the original no-failure, tilt, planar-speed,
displacement, height, supporting-feet and motor soft-limit gates. Tilt remains
5 degrees (approximately 0.0873 rad); speed 0.03 m/s, displacement 0.02 m,
final-second height at least 0.105 m, feet support at least 99% after 0.2 seconds,
and joint-time soft-limit exposure at most 1%. No relaxed motor or stability gate.

Each training seed passes only if at least one declared checkpoint has at least
122/128 passes on **every** held-out evaluation seed. Different training seeds
may have different passing checkpoints; no common-iteration gate is introduced.
All three pass: `lean-replication-passed`; some pass: `lean-replication-seed-dependent`;
none pass: `lean-replication-rejected`; missing, failed or unverifiable evidence:
`lean-replication-incomplete`. Never drop cases or seeds from the denominator.

Even a nominal replication pass admits no complete skill: `checkpoint_admitted`,
`learned_stance_accepted`, `football_balance_accepted`, `physical_motion_authorized`,
`independent_gpu_attestation` and binary-runtime-equivalence flags remain false.
Only after the applicable evidence gate may the smallest flat-ground disturbance
contract be designed, tested and separately predeclared. No raw perception,
actor-observation expansion, video, hopping promotion, rolling-ball training or
physical motion in this evaluation. At 08:00 start no new Duck work, confirm
owned processes finished safely and artifacts durable, and leave protected
services unchanged. Background continuation remains unconfirmed; durable user
services do not establish a registered Codex automation.

## Source verification before installation

The broader Mac CPU stance regression suite passed **1,276 tests in 93.22 s**,
CUDA hidden, including the original protocol, training authentication,
checkpoint, plant, controls, scorer, collection and bounded process contracts.
After the final test additions, the full-runner and original packed-probe
focused suite passed **149 tests in 12.44 s**. Those tests use the exact retained
R4 timing components, exercise 36 ordered fresh packed-runtime allocations with
mocked CUDA ownership, distinct winning iterations, missing/duplicate/reordered
cases, malformed ownership/timing receipts, watchdog and CLI refusal paths.
Mocked allocations are not GPU evidence. Diff, Python compilation and relative
documentation link checks passed. A read-only review found no definite matrix,
namespace, numerical-gate or lease defect. Linux qualification and actual
capture remain separate prerequisites; no full evaluation is claimed here.
