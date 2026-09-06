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
