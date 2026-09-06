# Overnight recovery curriculum: September 6–7

User authorization: stepwise simulation coding, evaluation and gated training
until **2026-09-07 07:00 Asia/Shanghai** (September 6 23:00 UTC).
Parent source: `a1cd6a5ccab18dc1739e29a487b60f8723ed50a5`.
Worktree: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum` on
`feat/athletics-obstacle-curriculum`, host `converge@192.168.100.100` only.
No physical duck is available; all results are simulation evidence.

## Boundaries and live entry check

At 22:57 Shanghai September 6, the remote exact branch was clean at the parent,
GPU utilization was 0%, temperature 44 C, memory 12 MiB, with no compute process
or running Duck user unit. Both protected system services were inactive:
`recomo-ai-mission-vllm.service` and
`recomo-ai-mission-subject-model-worker.service`. Recheck before each launch.
Do not use 100.98, restore protected services, alter unrelated workloads, or
mutate source/configuration while its job runs.

Historical U4/U3/O3a rejection, seed-359 recorder uncertainty and seed-367
repeatability stop are unchanged. No retry of their cases for acceptance.
The new experiment is a recovery-control diagnostic, not resurrection of U4.

## Chunk A: measured recovery contract

The source observer `RecoveryMeasurement` runs only when the existing bounded
first-attempt motor audit is enabled, for **both uncapped and capped** runs.
It samples route speed immediately before each control step, after supervisor
phase updates. It consumes terminal flags afterward, permanently excluding
that environment's reset episodes. It cannot modify commands or rewards and
stores O(number of environments) state, not trajectories.

Starting at first observed recovery, measured speed must stay within
**+/-0.03 m/s of nominal for a sampled span of 0.50 s**, completed no later
than **2.0 s** after entry. At dt=0.02 this requires 26 consecutive samples,
not 25. Leaving recovery or leaving the speed band interrupts the streak.
Failure to observe recovery, or termination before enough observations, is
explicitly not success. Late stability cannot clear the two-second window.
These are diagnostic engineering criteria, not calibrated hardware limits.
Existing approach/recovery mean-speed gates remain; interaction has no speed
tracking target. Sampling does not include the terminal step's post-physics
speed, because post-return state may already be reset. Short attempts are
therefore conservatively censored. No GPU observer-equivalence claim yet.

## Chunk B: predeclared frozen recovery-cap A/B v1

Protocol: `recovery-cap-specialist-s379-v1`. No data-dependent seed selection.
Actor SHA-256:
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
Near HC4-R2 supervisor:
`c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7`.
Far HC4-LH supervisor:
`0b2608080671c5df85d8c9f900d68b6a6f298ec820eb1c6ba75afc948337505a`.
Resolve unique retained paths and hash them before launching; never substitute
U4 or compose a new supervisor. Freeze runtime versions and dependency hashes
from the seed-373 motor audit, plus exact clean tested launch source.

Four cells, in order: (.30 m/s, .90 m), (.40, .90), (.30, 1.15), (.40, 1.15).
All use centered lateral 0, seed **379**, **8 environments**, **700 steps**,
12-second attempt timeout, first attempts only, exact structured geometry,
zero sensor noise, no dataset, trajectory recorder or video. Near/far select
their respective retained specialist. For each cell launch sequentially:
baseline A1 (no cap), same-seed baseline A2 (no cap), candidate B (**.20 m/s²**
positive recovery acceleration cap). All three enable motor audit and the new
speed observer. This is 12 children maximum, 96 attempts total. Stop immediately
on runtime, identity, finite-value or numerical gate failure; do not run later
children to compensate. Save partial evidence and the deterministic stop.

Each completed baseline must have eight resolved clean attempts, no collision,
fall, timeout, hard/other terminal, nonfinite or rated-speed exceedance, and
unchanged legacy pooled torque-utilization p99 <= .60. A2 must also match A1
outcome counts, differ by <=.02 in legacy torque p99 and <=.02 m/s in each
approach/recovery mean speed. These are declared practical repeat tolerances,
not bitwise determinism or repair of the historical control. Both baselines
must have eight measured recoveries within the window; censoring closes this
diagnostic as inadequate evidence, not a lowered speed criterion.

B must satisfy the same absolute gates and recovery window. Per-cell approach
and recovery speed must be no more than .03 m/s below either baseline; pooled
sample-weighted phase speed must be no more than .01 m/s below either baseline
set. Keep the legacy torque gate unchanged. For the new pre-reset **recovery**
pooled utilization p99, require no per-cell increase against the smaller of
A1/A2, and at least 5% reduction in the arithmetic mean of the four cell p99s
against each baseline set. The mean of cell quantiles is explicitly NOT a
pooled percentile. Maximum named-joint recovery p99 must not increase against
either baseline in any cell. Save every joint's values and sample coverage.
No thresholds may be tuned after seeing v1 outcomes.

Even a pass means only `recovery-cap-diagnostic-supports-pilot`, not policy or
physical admission. A failure means stop this GPU experiment, diagnose retained
evidence read-only, and predeclare a distinct smallest hypothesis if justified.
Do not change the rate/seed/checkpoint and rerun until something passes.

Runner implementation and adversarial tests remain REQUIRED before launch:
strict report identities (including capped stage and original source stage),
exact case keys, finite metrics, accounting, observer coverage, duplicate/missing
report refusal, deterministic comparison, source/model/runtime hashes, unique
output directory, crash/timeout receipts and exclusive GPU preflight per child.
Use `artifacts/evaluations/recovery-cap-specialist-s379-v1` on 100.100.
Each child timeout <=180 s; total service ceiling <=2400 s. Start only when
the full ceiling plus five-minute closeout margin fits before the deadline.
No retry or overwrite of an existing output directory.

## Chunk C and subsequent curriculum (conditional, not yet predeclared training)

Only after B supports it, wire a recovery-speed-only supervisor adapter using
the pre-reset motor stream; freeze the locomotion actor and existing avoidance/
yaw behavior. Consume every low-level terminal sample across five-step action
windows. Test reset accounting, finite rewards, command limits, frozen weights,
checkpoint resume and held-out evaluator separation before training.

Proposed first pilot: 128 environments, 40 PPO updates, one fresh declared seed;
first run a separately bounded timing smoke. Freeze reward coefficients,
checkpoint cadence, seeds, data provenance, stopping rules and exact source in
a new experiment document BEFORE any pilot. No training is launched by this
document alone. Reserve checkpoint/save time and stop well before 07:00.

Evidence-led progression: centered .30 recovery -> centered .40 -> near/far
and +/- .08 lateral placements -> independent training/evaluation seeds ->
separate observation noise/dropout robustness. Each expansion needs a matched
baseline and unchanged collision, stability, speed and motor gates. Do not
advance hopping, Locked, H2, raw perception or video on obstacle evidence.
If a gate blocks GPU progress, continue bounded diagnostic/source/test work;
do not consume GPU time merely to keep it busy.

## Deadline and continuation handoff

Heartbeat `microduck-curriculum-until-sep-7-07-00` checks every 15 minutes,
reporting only material progress, failure or required action. Every remote
job must enforce its own conservative time bound; heartbeat timing alone is
not a deadline mechanism. At/after 07:00 start no new Duck work; confirm jobs
complete or safely stopped and checkpoint/output hashes durable, leave protected
services unchanged, delete the heartbeat and report retained state. The local
computer and desktop app must remain running for local scheduled continuation.

Current handoff: source observer and this predeclaration are implemented;
the A/B runner, GPU A/B, trainer adapter and pilot remain pending. Finish and
test the runner against synthetic adversarial reports, commit/push, fast-forward
the idle clean remote branch, then launch the unique bounded A/B service.

Final local validation: **465 focused CPU checks passed** (7.82 s, CUDA hidden),
including observer, cap, motor stream, audit, rollout, existing gate suites and
asynchronous-terminal/exact-deadline regressions. Two existing real-spec
actuator/site-pattern warnings remain. Diff whitespace checks passed. This is
source-level validation only; no GPU A/B or optimizer update has been run for
this chunk.

## Runner implementation handoff (before any A/B observation)

`python -m mjlab_microduck.recovery_ab --source <full tested commit>` implements
the exact ordered prefix above. Retained paths verified on 100.100:
`artifacts/checkpoints/hc4r2-bc-796634d-s42/supervisor.pt` (near) and
`artifacts/checkpoints/hc4lh-11002cc-center002/supervisor.pt` (far). The older
`center006` far file has a different hash and is not used.

The runner checks source/model identity and idle protected GPU state before
every child. It retains fsynced exclusive launch/runtime/decision receipts,
child logs, report hashes and numerical stops; no retry is implemented. Use
one retained user service with `RuntimeMaxSec=2400`, `TimeoutStopSec=15`,
`KillMode=control-group`, `Restart=no`. It refuses launch unless 45 minutes
remain, and separately checks the remaining service and absolute child budget.
Only a complete passing 12-report matrix supports predeclaring a pilot;
partial prefixes and legacy motor/recovery-window failures do not.

Pure `evaluate_paths(ordered_report_paths)` recomputes the deterministic
decision without simulation. Tests use real CPU observer/audit collectors for
synthetic reports and mocked subprocesses for orchestration. They exercise
missing/duplicate/swapped/extra-after-failure evidence, case and model identity,
reset/measurement coverage, cap configuration, baseline tolerance, per-cell
and pooled speed limits, load improvement, invalid statistics, timeout/crash,
changed/busy host, expired deadlines and preservation of existing directories.
These tests are not GPU repeatability or policy acceptance evidence.

Runner precommit validation: **523 focused CPU tests passed locally** (10.12 s,
CUDA hidden; two existing actuator/site-pattern warnings), including 58 new
A/B checks and the retained 465 regressions. Diff whitespace checks passed.
No A/B outcomes were available when the protocol, thresholds or runner were set.

## Retained v1 stop: first baseline, not a cap comparison

Launch source `bd2a20a6232f647e656a2dfe6788b037e29a06b1` passed **523 CPU tests
on 100.100** (7.06 s). Service
`microduck-rl-recovery-ab-s379-bd2a20a.service` ran September 6
23:33:55–23:34:19 Shanghai. Exactly one simulation child ran, exit 0,
16.3819944877 s; parent deliberately exited 2 for the numerical gate. The unit
is retained failed, not restarted/reset. No A2, capped B, later cell or PPO ran.

First cell: HC4-R2, 0.30 m/s, obstacle 0.90 m forward and centered, seed 379,
eight first attempts. **8/8 clean**, zero collisions, falls, timeouts, hard/other
terminals, nonfinite steps or rated-speed exceedance. Legacy torque-utilization
p99 **0.5060326457 <= 0.60**. All eight entered recovery and remained observable
for **2.94–3.78 s**, but **0/8** achieved the half-second speed band within two
seconds (or any later retained stable span). They are `window-missed`, not
censored or omitted. Deterministic decision: **`numerical-gate-stop`**, sole
failed criterion `recovery-window`; pilot support and all admission flags false.

Mean approach / interaction / recovery route speeds: **0.0326169506 /
0.1995471418 / 0.2034718841 m/s**, with 15 / 1825 / 1382 samples. The very short
approach phase includes startup and is not a steady-state speed estimate.
Mean passage time: 8.0549997687 s. Full-case commanded speed range:
0.2949104309–0.3119393289 m/s. Recovery pre-reset pooled utilization p99:
0.5006072929; maximum named-joint recovery p99: left knee 0.6498272851. These
pre-reset quantities do not replace the legacy gate or describe physical heat.

Read-only diagnostic observation: environment 0's existing 10 Hz representative
trace has 38 recovery samples, measured route speeds **0.0813181847–0.2612001896
m/s**. Its largest positive command increment in recovery is only
**0.0059571862 m/s**, below the proposed cap's 0.02 m/s per update. Recovery also
contains substantial yaw commands and lateral rejoining. Therefore acceleration
capping has no demonstrated remedy here; the capped arm was never run, and
these representative observations do not prove a cause across all environments.
The instantaneous-speed requirement may be sensitive to gait oscillation, but
the measured mean speed deficit cannot be dismissed as censoring or a single
spike. Do not loosen the band, deadline, safety gate or retry seed 379.

All eight files were copied byte-for-byte to local
`artifacts/diagnostics/recovery-cap-specialist-s379-v1`; originals remain in
remote `artifacts/evaluations/recovery-cap-specialist-s379-v1`. SHA-256 manifest:

| File | SHA-256 |
| --- | --- |
| `00-cell0-a1/hierarchical-teacher-evaluation.json` | `b6bf4a7baa4e16e47a1e672f382cca80e1e9b46dbe70b5bd9da3481062a73828` |
| `00.log` | `faa8e11d4f1fbbe98a79b488d4551ba272df918f3e9d1d799d5f932f2b6fd294` |
| `decision-00.json` and `decision.json` (each) | `8b34b5186a45dda26aad43cb71b095367c765a4b3ed090c997179d416422a0a2` |
| `launch-00.json` | `48740a3dea349e99e90fe3dd9e86c357b3b282f50499e4ef771070e56f57cb3c` |
| `launch.json` | `ebd0a47494534ff40d9cd227848e1dc3853c718dc64996cee813dd6b3f93a96f` |
| `runtime-00.json` | `fa28bd487ab07e1357a761dcae3410a43c31763e26776acda26fb8616a1d57a8` |
| `runtime.json` | `b415923019deb7e4d18d01939bc76a564e796cdb5c97b1af51bdf5e873bf4983` |

Local canonical-JSON recomputation exactly matches the retained decision.
At 23:36:03 Shanghai, GPU was idle (0%, 45 C, 12 MiB), no compute process or
running Duck unit, both protected services inactive, and source clean. Existing
actor/supervisor checkpoints were never modified.

**Superseding handoff:** v1 is closed. Keep its gates and evidence immutable.
Next continue bounded source/read-only diagnosis separating frozen-gait speed
response, route-heading/lateral recovery, and instantaneous versus sustained
speed measurement. Inspect the existing actor evaluator and command frames;
test any identified contract defect before changes. If a new simulation control
is justified, predeclare a genuinely distinct minimal no-obstacle/straight-line
speed-response diagnostic, fresh identity and finite runtime, with the same
actor and unchanged motor limits, before launching. It cannot admit v1 or PPO.
Do not silently enable the recovery-speed PPO pilot: its prerequisite failed.
Keep working within the overnight window on diagnosis/tests, not repeated
rejected GPU jobs. Deadline and service-preservation rules above remain active.

Evidence-closeout CPU validation: **524 tests passed locally** (8.50 s),
including a hash-bound canonical replay of the retained seed-379 stop. The
evidence replay skips explicitly on checkouts without the separately retained
artifacts; it executed here. Two pre-existing actuator/site warnings remain.
