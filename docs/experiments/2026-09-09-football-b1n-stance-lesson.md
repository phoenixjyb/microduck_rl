# B1-N: nominal closed-loop stance pilot declaration

Protocol `football-b1n-nominal-stance-v1`. The fixed-target CPU hold aborted for
forward tilt; its source, report and unchanged stops are retained in the
[hold experiment](2026-09-09-football-flat-hold.md). This declaration creates a
separate specialist, not an alteration or promotion of any gait/hop actor.
**Reference components only: environment integration, complete evaluator and
launch manifest are not yet implemented or admitted. Do not launch training.**

## Plant, reset and control

Use the same rigid full-collision XML/meshes, HOME_FRAME, BAM XL330 m6 and
nominal startup values as the hold. Native diagnostic solver was Euler/Newton,
100 iterations, tolerance1e-8, gravity-9.81, step0.002 s. The GPU implementation
must explicitly match supported settings and record any backend differences;
CPU component tests alone do not establish Warp equivalence. Floor input
friction0.6 combines with the foot XML to give effective sliding friction1.0
in the retained trace; do not silently claim the effective coefficient is0.6.
Preserve condim3 and record effective parameters and collision filtering.

Reset each episode to the validated nonpenetrating flat stance with zero joint
and root velocity, identity root orientation, zero correction/delay state.
No pushes, randomized mechanics, sensor noise, falls/prone resets or movement
commands in this first pilot. Episode5 s; controller50 Hz (10 physics steps),
fixed3-physics-step delay. Terminate at the hold's existing numerical, torque,
speed, tilt, root-height, hard-limit, forbidden-contact and support-loss stops.
Time limit is separate from failure; no failure reset may hide an evaluation
attempt. Log and stop the job on NaN, optimizer nonfinite values or solver warnings.

Ten leg outputs are bounded target corrections, not motor torques: +/-0.2 rad,
at most1 rad/s target slew, intersected with each joint's central90% range.
Head/neck targets remain nominal. The reference action limiter returns realized
corrections for the next observation; never report unclipped actions as applied.
Do not increase firmware gains or remove motor limits to compensate for failure.

## Observation and optimizer identity

Actor44D: body projected gravity3; body angular velocity3 scaled by1/5;
14 joint offsets from nominal scaled by1/0.5; 14 joint velocities scaled by1/10;
10 previous realized target corrections scaled by1/0.2. Exact joint order and
CPU reference are in `stance_lesson_contract.py`. No camera, translation,
contact force, prior61D actor or inherited normalizer. Fixed scaling only.

Critic50D appends world root linear velocity3 (m/s), root-height error from
0.12 m scaled by1/0.1, and left/right supporting normal forces scaled by1/10 N.
These six values are privileged critic-only simulation information. Actor and
critic are fresh ELU MLPs (128,128,64); no running observation normalization.
Gaussian actor initial scalar std0.3; no checkpoint/optimizer restoration.

Seed521, 512 environments, 512 PPO iterations, 24 policy steps per update.
Adam3e-4 fixed learning rate; 5 epochs, 4 mini-batches, gamma0.99, GAE0.95,
clip0.2, value coefficient1 with clipped value loss, entropy0.005,
gradient norm1; normalize advantages across collected trajectories, not each
mini-batch. Save common iterations128,256,384 and final511; retain initialization
and source/config/asset/parameter hashes. Local TensorBoard only; no model upload.

Reward accumulates physics_dt times the following dimensionless sum at each
actually executed post-step physics boundary (up to10 per policy tick):
2 exp(-(tilt/0.10)^2) + exp(-(||v_xy||/0.03)^2)
+ exp(-((height-0.12)/0.015)^2) + 0.5 mean(foot_support >0.01 N)
- 0.2 mean((applied_motor_torque/0.36)^2)
- 0.05 mean((hinge_velocity/10)^2)
- 0.05 mean((realized_correction/0.2)^2)
- 0.1 mean((target_correction_change/0.02)^2).
Apply an additional failure cost-2 once per terminal transition, not scaled by
dt; never pay a timeout bonus. Correction and correction-change values are held
for that policy tick; all physical terms are evaluated at every executed physics
boundary. Sum only executed substeps on an early terminal. Check motor stops at
every substep, not only the last one. This is a pilot hypothesis, not a tuned recipe.

## Evaluation, promotion and launch gates

Evaluate all four declared checkpoints without selecting a favorite seed:
seeds541,547,557; 128 environments each; one5 s first attempt each from the
nominal reset. No auto-reset score averaging, smoothing or changed motor gates.
Per-attempt candidate pass requires no failure, final1 s tilt p95<=0.0873 rad,
root planar speed p95<=0.03 m/s, planar displacement<=0.02 m throughout,
final1 s height minimum>=0.105 m, supporting feet on>=99% of physics samples
after0.2 s, and soft-limit exposure<=1% of joint-time samples. All raw physics
samples must obey hold stops. Each evaluation seed needs>=95% attempts passing
at final511; earlier checkpoints are diagnostic, not substitutes for final.
Even a pass admits only a nominal-stance candidate: fresh-seed replication and
small-disturbance recovery must precede B1 acceptance or fixed-ball training.

Before launch: implement/test the exact runtime configuration, substep counters,
reward terms, observation ordering, action/delay/reset semantics and evaluator.
Require an exact clean source/lock/runtime manifest and successful Linux normal
task imports. Run a separately labeled disposable64-env/16-iteration integration
smoke at seed523; its model must never become the pilot parent. Inspect finite
metrics and measured runtime before declaring a30-minute maximum pilot service
safe within the authorization cutoff, leaving at least10 minutes for closeout.
Use the existing exclusive lease and independent timeout, preserve protected
services and100.98, and retain partial checkpoints on any bounded stop.

Walking/avoidance/hop policies and evidence remain immutable. No integrated
retention or football skill is claimed by the isolated nominal-stance pilot.

## Reconnect and reference validation

100.100 became reachable again at this update. Read-only checks found no GPU
compute process, 0% utilization /12 MiB /46 C, and both protected system services
inactive. Historical active/exited Duck units are not running training; the
completed map had MainPID0 and Result=success. No service was changed.

The remote worktree was clean at `24e061c` and was fast-forwarded to exact
`832b436f7f1f4027744e1128825f877dc53a7ffa`. The interrupted remote wrench report
does **not** exist; no file was overwritten or wrench/hold rollout repeated.
The separately completed Mac wrench evidence remains the retained result.
The previously absent B0-A input was copied to its declared remote diagnostics
path and verified against SHA256
`d15226a8eacb69a347f4ece725528ecb102e9fde05e69e9035cb861c6e5bebf6`.

The Linux regression run at832b436 passed422 tests with2 skips in21.80 s.
The real retained-pose test ran. Both skips were from
`test_foundation_command_session.py:80`, for separately retained checkpoint files
not present at the test paths; these skips are not actor-loading acceptance.
A focused rerun confirmed26 passed/2 skipped there and19 passed across the
checkpoint-audit and retained-pose module tests.

Normal `python -m mjlab.scripts.list_envs` exited0 and listed81 tasks without
the circular-registration warning. The console `list-envs` entrypoint returned81:
read-only source inspection explains that its main returns the task count and
the console wrapper passes that to sys.exit. This is not81 failed tasks or a
fixed dependency bug. Neither invocation initializes the GPU environment.

Retained remote files under `artifacts/evaluations/`, mirrored to Mac
`artifacts/diagnostics/` with identical hashes:

- `football-linux-reconnect-832b436.json`: SHA256
  `2cbadc928ed6c3a897090beeffdf50c3555469649bf5b157a35a1f34b173809e`.
- `football-linux-reconnect-832b436-cli-check.json`: SHA256
  `613265e2dd0d2104d2b82675c912d3c9dc7c11995f2f82514a63a97d47e79a26`.

Exact imported distributions: Torch2.9.1, Warp1.12.0, MuJoCo3.10.0,
MuJoCo Warp3.8.1, mjlab1.3.0, BAM1.0.1. CUDA remained uninitialized.
This removes the connectivity/normal-import blockers observed earlier, but does
not validate new B1 runtime integration, GPU parity or optimizer readiness.
The new CPU observation/action reference passed within76 football/stance tests
in5.29 s, including exact real-robot joint ordering and nominal soft-range checks.
The broader local CPU regression set then passed432 tests in18.59 s. No training
or rollout result is inferred from these source/reference checks.

## Substep transition and first-attempt scoring components

`stance_transition.py` implements the declared reward terms and hold stops in
batched Torch operations. A policy tick owns up to10 physics snapshots. It
counts only live worlds, includes the executed terminal substep's dense reward,
charges failure-2 once, and stops counting subsequent reward/episode time for
that world. Excessive proposed torque closes a world before a physics step.
NaN, invalid evidence and simulator warnings raise job-level errors. Failure
takes precedence over simultaneous timeout. No reward autograd graph is retained.

`stance_evaluation.py` implements numerical scoring of a continuous first-attempt
prefix. It refuses skipped/reset counters and any frames after the first failure.
Full evidence has boundaries0..2500; the final-second window is2000..2500
inclusive, and support is measured strictly after step100. Quantiles use linear
interpolation over raw samples; soft exposure excludes the initial unstepped
boundary. A successful numerical score explicitly does not admit a checkpoint:
source/hash, seed/environment identity, exact384-attempt per-checkpoint assembly
and final-checkpoint campaign checks still belong to the complete evaluator.

Runtime audit: mjlab's stock manager environment performs all decimation substeps
before its termination/reward managers. It also documents one-substep-old derived
quantities at those managers. Using that loop unchanged would violate this
lesson's refreshed per-substep stops and reward accounting. The next implementation
must bind the new transition component to fresh post-step state, check proposed
BAM torques before applying them, preserve first-terminal state, and physically
skip/freeze closed worlds until reset. An accounting live mask alone does not
prove that the simulator stopped evolving those worlds. Do not monkeypatch the
shared installed environment or claim the CPU component checks validate this
not-yet-written integration.

The current broad local CPU regression set passed466 tests in24.22 s. The new
scorer also read the exact retained native hold file and rejected its first tilt
failure at step474 without rerunning MuJoCo or altering the file. That2.3 MiB
trace is now backed up on100.100 at the same `artifacts/diagnostics/` relative
path and verified against its original SHA256
`6bd04008ac8161f55f7a3c77c45b10ccee58c37cba6d4e8de5c5107ca2ea768f`.
The real-trace test must run on Linux, not be counted as an optional-input skip.
