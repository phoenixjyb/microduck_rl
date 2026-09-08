# B1-N bounded CUDA integration probe

Protocol `football-b1n-cuda-integration-v1`. This predeclares a software integration
check of the [caller-owned stance runtime](2026-09-09-football-b1n-stance-lesson.md),
not a trained stance candidate, full hold, five-second evaluation, football
balance policy or permission to start PPO. Do not repeat the closed native hold.

## Cases and evidence

Use explicit `cuda:0`, seed527, two actual rigid full-collision Duck worlds with
the existing nominal flat-floor placement and unchanged BAM/control/solver
settings. Normal Entity/BAM initialization and Warp forward/Euler code are used.
No cameras, rendering, checkpoint loads, optimizer steps, pushes or robot motion.

1. Normal integration: two policy ticks, zero leg action then all-one normalized
   leg action. The existing limiter/delay applies. Both worlds must complete
   exactly20 physical substeps without termination. Retain21 continuous boundary
   snapshots, exact observations, physical metrics, soft-limit masks and rewards.
2. First-terminal isolation: a fresh two-world runtime, two policy ticks. After
   the first real Euler commit, inject a clearly labelled synthetic0.4-rad pitch
   into world0's owned state. This is not an action or a policy-performance test.
   Require executed counts(1,10) then(0,10), zero later reward for the closed world,
   unchanged first-terminal/contact record and bitwise unchanged physical,
   motor, FIFO and observation fields for world0 while world1 continues.
3. Selectively reset world0, returning its owned terminal evidence. Require
   counters(0,20) and bitwise unchanged sibling state. This is not a mounting,
   recovery or automatic-reset acceptance test.

Retain actual Torch and Warp device identity plus CUDA-initialization status;
CPU-produced results fail the CUDA validator. Explicit counter prefixes, finite
JSON and negative capability flags are checked again by the supervisor. The
schema validator is not an independent re-simulation or complete trajectory
acceptance evaluator. GPU timing here includes explicit synchronization and
evidence overhead; it cannot alone estimate the64-world PPO smoke throughput.

## Exact launch boundary

Work only on100.100, machine ID`0c79e415429b4933a400159bfa79a34d`, GPU UUID
`GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`, driver595.84. Require the exact clean
feature-branch commit, its frozen`.venv`, versions, robot XML/mesh hashes, motor
parameter hash, lockfile and project config. The committed
[dependency Python tree pins](2026-09-09-stance-cuda-probe-runtime.json) cover
189 mjlab,69 MuJoCo Warp and37 BAM Python files. Native binaries and the full
OS/driver stack are not claimed completely equivalent by these Python hashes.

`python -m mjlab_microduck.stance_cuda_probe prepare --source FULL_SHA` is CPU-only
and writes a new exclusive launch manifest under
`artifacts/evaluations/stance-cuda-probe-SOURCE12/launch.json`. It refuses an
existing output directory. Source identity comes from the committed implementation
that passed CPU tests, not from an unreviewed remote branch or arbitrary config.

After Linux CPU validation, launch `supervise --source FULL_SHA` as one new
retained user service named `microduck-stance-cuda-probe-SOURCE12.service`, using
Type=exec, RemainAfterExit=yes, RuntimeMaxSec=180s, TimeoutStopSec=10s,
KillMode=control-group and no restart. Use an allowlisted parent environment with
CUDA hidden; only its supervised child receives CUDA_VISIBLE_DEVICES=0.

The supervisor takes the existing exclusive `/home/converge/.local/state/microduck-gpu0.lock`
lease, requires two idle samples and inactive protected system services, then
runs exactly one child with a120-second watchdog. It repeatedly verifies source,
runtime and assets, rejects backend/numerical warnings in the retained log,
monitors GPU ownership/temperature/protected services, and kills only its owned
child process group on failure. Never start, stop or restore protected services
or any unrelated workload. Require the full180-second service budget plus600s
closeout before September10 07:30 Shanghai (September9 23:30 UTC).

Retained outputs are `launch.json`, `child.log`, `probe.json` if the child completes,
and `report.json` with source-bound hashes, telemetry and a deterministic
integration-only decision. All JSON writes are exclusive and fsynced; do not
overwrite or silently retry a failed attempt. Diagnose retained logs read-only.
Verify idle GPU after the child exits while retaining the lease. Only a successful
probe removes this narrow CUDA integration blocker; complete evaluation,
PPO/launcher assembly and the disposable smoke remain required before training.

## CPU validation before launch

168 focused CPU tests passed in8.74s, including32 new probe/identity/log/schema/
supervision tests and the prior stance and idle-gate tests. The exact short cases
ran on Warp CPU. Tests that synthesize CUDA-shaped metadata are explicitly schema
fixtures, never published CUDA evidence. Fault, drift, warning and overwrite
tests use synthetic inputs and do not change external services. Linux validation
and the real CUDA attempt are pending at this predeclaration.

### Preflight correction before any GPU attempt

Source `5c1bf5dcff56e5bdef7c3983d53d6b97e54d6168` passed557 Linux CPU tests
(two expected missing-checkpoint skips) in25.72s. CPU preparation succeeded,
but its startup output exposed a false positive: the existing exact informational
line `[mdp] Patches 1-2 active: NaN-safe reward/advantage` matched the NaN log guard.
No GPU child or service was launched from that source. Its exclusive launch
manifest remains retained, unused, under `stance-cuda-probe-5c1bf5dcff56`.

The guard now exempts only that exact complete source-bound startup line. Added
regression cases retain refusal for appended warnings and subsequent non-finite
observations. All other warning and numeric checks are unchanged. The corrected
source requires a new manifest, new CPU validation and its own unique service.

## Retained CUDA result

Corrected source `f42a21cf592184276b89a4be17620e87a7fe6941` passed146 focused
local CPU tests in9.31s and212 Linux tests in16.28s, including the shared process
supervisor. The earlier557-test broader Linux result belongs to the preceding
source, not this correction. No tests were skipped in the corrected selections.

The unique user service `microduck-stance-cuda-probe-f42a21cf5921.service`
completed successfully: MainPID0, active/exited (retained receipt), Resultsuccess,
ExecMainStatus0. Its child PID174499 returned0 after20.305345s within the120s
child and180s independent service bounds. Actual Torch and Warp devices were
`cuda:0`, Warp reported CUDA, and Torch CUDA initialization was true.

Both normal worlds completed20 physical steps with21 continuous boundary
snapshots. The explicit synthetic terminal produced steps(1,10), then(0,10);
closed physical/motor/FIFO/observation fields remained bitwise fixed and the
first terminal record remained unchanged. Selective reset returned that record,
reset only world0 and preserved sibling state at step20. The normal short-loop
elapsed1.685929s is not a full-rollout or PPO throughput estimate.

The deterministic decision is `cuda-integration-only-passed`. Optimizer steps0;
learned stance, football balance, complete trajectory evaluation and physical
motion flags all remain false. No checkpoint or video was created.

Live samples saw no compute PID except the owned child; protected system services
remained inactive. Sampled temperature range45–50 C. Both closeout idle samples
showed0% GPU,12 MiB,49 C and no compute PID. A subsequent readback showed47 C,
no compute PID and protected services still inactive. These are sampled readings,
not a continuous hardware-temperature maximum or exclusive host-wide lock.

Raw files are retained on100.100 and mirrored byte-for-byte under local
`artifacts/evaluations/stance-cuda-probe-f42a21cf5921/`. SHA-256:

| File | SHA-256 |
| --- | --- |
| launch.json | `35bf01ef07724e25d4e0119131403bfb9958f0a91ed404a9d3da93250ba54f84` |
| child.log | `e2e7390a3f58a8b292840f42a0758f6373be1ebc2e28f4b1827853b13fb05ce4` |
| probe.json | `0ef5145ace7531529bda34a35e9451a9d585a742555c79f2a6301007f828a08e` |
| report.json | `b76051d377e6f107e25116aeae826335409a31cb9cca0b3d04d54e0c05a88d1a` |

The mirrored report, every linked raw hash, source identity, CUDA schema,
continuous counters, negative capability flags, log guard and observed PID
ownership were revalidated locally without initializing CUDA. The unused
preceding source's launch manifest is also retained and mirrored unchanged:
`6ff597b7de72b3a2f85f8f35c0d418f4dfaf665c6daf711de1944dedb1c6e0b1`.
The first local verification command lacked `/usr/sbin` in its allowlist and
failed importing MuJoCo because `sysctl` was unavailable; restoring that standard
tool path allowed the read-only verification. It did not affect the remote run.

This completes only the short CUDA integration gate. The full source-bound
first-attempt evaluator, PPO assembly and disposable smoke remain unfinished.
Do not infer stance, push recovery, hopping, obstacle retention or rolling-ball
capability from this result; do not rerun closed probes or loosen their gates.
