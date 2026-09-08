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
