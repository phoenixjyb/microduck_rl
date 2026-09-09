# B1-N: nominal closed-loop stance pilot declaration

Protocol `football-b1n-nominal-stance-v1`. The fixed-target CPU hold aborted for
forward tilt; its source, report and unchanged stops are retained in the
[hold experiment](2026-09-09-football-flat-hold.md). This declaration creates a
separate specialist, not an alteration or promotion of any gait/hop actor.
**Short CUDA integration passed: the caller-owned Warp loop ran the source-bound
two-world normal/isolation/reset probe on100.100. Complete evaluation and the
PPO/launcher adapter remain unfinished; no optimizer or learned stance is
admitted. Do not launch training before the remaining smoke gates.**

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
lesson's refreshed per-substep stops and reward accounting. The GPU implementation
must bind the new transition component to fresh post-step state, check proposed
BAM torques before applying them, preserve first-terminal state, and physically
skip/freeze closed worlds until reset. An accounting live mask alone does not
prove that the simulator stopped evolving those worlds. Do not monkeypatch the
shared installed environment or claim the CPU component checks validate this
not-yet-written GPU integration.

The current broad local CPU regression set passed466 tests in24.22 s. The new
scorer also read the exact retained native hold file and rejected its first tilt
failure at step474 without rerunning MuJoCo or altering the file. That2.3 MiB
trace is now backed up on100.100 at the same `artifacts/diagnostics/` relative
path and verified against its original SHA256
`6bd04008ac8161f55f7a3c77c45b10ccee58c37cba6d4e8de5c5107ca2ea768f`.
The real-trace test must run on Linux, not be counted as an optional-input skip.

## Native CPU runtime integration

`stance_native_runtime.py` now connects the fixed-scale observations, rate/soft-
limited targets, three-physics-step delay, real BAM computation and substep
transition accounting to one owned native MuJoCo world. It refreshes derived
state after integration, retains every boundary, and checks an invalid existing
state or excessive proposed torque before another physics step. The first failure
breaks decimation immediately; calling step again on a closed world is refused.
Observations and trace reads do not advance the model. A normal episode needs an
explicit reset; a numerical/programming fault closes the audit instance and
cannot be cleared by episode reset. Reset clears native state/control, motor
friction/damping, previous motor torque, target correction and delay history.

This is deliberately not registered as a training task or presented as a GPU
backend. Short CPU integration tests cover a10-substep policy tick, target-delay
timing, an injected first-substep tilt, pre-step state/torque rejection, and an
independent live world advancing without changing a closed sibling. No full
hold rollout, optimizer, candidate evaluation, ball training or physical motion
was run. Synthetic failure injections are tests of control flow, not capability.

The correction-change bridge allows only1e-14 rad of subtraction roundoff at
the declared0.02-rad target-slew boundary, then represents the cost input within
that bound. This does not change physical torque, speed, tilt or historical
acceptance thresholds. GPU vector-world isolation, normal runtime initialization
and source-bound evaluator/launch assembly remain the next requirements.

Validation of this integration: the local CPU regression selection passed
474 tests in 23.68 s, including eight native-runtime tests and the unchanged
retained first-hold rejection test. These counts establish software behavior,
not learned stance. Read-only host checks at this update found a clean remote
`07770335cf3e172b1754560b45a1f9e7c3117414`, no GPU compute process,
0% utilization / 12 MiB / 46 C, and both protected system services inactive.
No optimizer job or service change was made.

The clean Linux worktree was then fast-forwarded to exact
`ae122b5177504075cfcb833ad43cef07a428aecd`. With CUDA hidden, all 76 focused
native-runtime, transition, evaluation, observation/action, flat-hold and BAM
tests passed in 6.36 s, with no skips. The retained first-hold input was present.
Post-test checks again showed an idle GPU at 46 C and both protected services
inactive. This completes Linux CPU validation of this chunk; GPU integration
and the declared disposable smoke remain unrun.

## Warp Euler candidate/commit integration boundary

`stance_warp_integrator.py` introduces a narrow integration-state boundary, not
the complete GPU environment. Read-only inspection of installed MuJoCo Warp
3.8.1 found no per-world pause argument in `forward.step` or mjlab's
`Simulation.step`. Changing timestep to zero is not the proposed solution: the
forward solver also consumes timestep. Shared installed modules remain unchanged.

The new adapter calls stock Euler with separate candidate arrays for `qpos`,
`qvel`, `time` and `qacc_warmstart`. It validates every candidate before committing
any field, then commits only live rows. Closed committed rows retain their exact
bits while live rows use the unchanged stock Euler result. All-closed batches
skip the integrator call. Nonfinite input/output faults the adapter and requires
job closeout. Explicit Torch/Warp synchronization favors correctness; this is
not CUDA-graph-ready or a throughput claim. The supported plant has no activation,
flex, tendon, mocap or equality state and uses Euler, matching the intended rigid
stance scope. This does not freeze motor/delay state by itself.

The private write-set dependency is guarded by the installed version and exact
source SHA256 pins; changing either file requires a new audit:

- `mujoco_warp/_src/forward.py`:
  `c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3`.
- `mujoco_warp/_src/smooth.py`:
  `63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f`.

Important boundary: the batch's forward/contact calculations can still run for
closed worlds. Candidate integration arithmetic can also run for them, but those
outputs never become committed physical state. This is not a whole-solver skip
or a frozen shared contact array. The full runtime must separately own the first
terminal observations/contacts, mask BAM torque history, friction/damping and
target-delay updates, and refresh live derived state after commit. An integration
mask alone still does not admit the stance environment or a training launch.

Eleven focused tests passed on Warp's **CPU** backend in 12.98 s. The component
fixture is a free sphere, not the Duck or football task. It checks exact agreement
with stock Euler for live rows, unchanged closed rows over four commits, zero
calls for an all-closed batch, no buffer aliasing, source/shape guards, and no
partial commit on nonfinite candidate output. Both the synthetic-acceleration
damping-disabled path and a real forward solve with implicit Euler damping were
tested. No CUDA, robot-contact or learned-balance result is inferred.
The broader local CPU regression selection passed 485 tests in 24.71 s;
Markdown rendering and relative-link checks also passed.

Linux confirmation at exact source `54311fe2256669be796c45b997ad9d329450185e`:
87 focused stance/Warp/native/BAM/hold tests passed in 14.20 s with CUDA hidden
and no skips. The installed private-source pins matched. After testing the
worktree was clean, no GPU compute process existed, telemetry was
0% / 12 MiB / 46 C, and both protected system services remained inactive.

Next: bind and test per-world BAM/delay/reset state and terminal capture to this
boundary, then verify actual CUDA behavior under a separately bounded integration
probe before the declared optimizer smoke. Full source-bound evaluator assembly
and all existing launch/retention gates remain required.

## Per-world motor and command-delay state

`stance_control_state.py` now supplies two separate components for the pending
GPU runtime. `StanceActionDelay` implements the declared ten-leg limiter in
batched Torch, preserves four nominal head/neck targets, and owns a per-world
three-substep FIFO. The caller sets targets once per policy tick, peeks before
computing motor proposals, and advances only accepted physical-step rows.
Closed rows retain target/correction/FIFO state; explicit selective resets clear
only selected rows. Invalid input faults the instance rather than permitting
an episode reset to erase the error.

`BamStateCommit` calls the actual stock BAM computation on a shallow actuator/model
copy with separately owned motor history, firmware actuator object and cloned
friction/damping model fields. Immutable motor parameters and freshly solved
input forces are read, not altered. Only live rows with finite proposals within
the existing 0.36 Nm gate commit torque history, effective voltage/gain and model
friction/damping. Invalid numerical output commits nothing and faults the job;
excessive finite torque returns a rejected-row mask for the transition layer.
The full caller must still write only accepted ctrl rows, shift their queues,
integrate them and capture refreshed state. This module does none of that physics
or contact capture itself. All-closed batches skip BAM computation.

The wrapper requires the exact repository FrictionDR adapter, XL330 m6 voltage-
controlled actuator, nominal 7.5 V / 0.1 drop gain / 6 V floor, unit gain/friction
scales, kp200 and dt0.002. No startup resampling or episode randomization is
admitted. Changed nominal tensors, replaced motor history, or unexpanded/replaced
model fields are rejected. Reset clears selected motor history and controlled
DOF friction/damping without changing unrelated DOFs or sibling worlds. Native
or Warp ctrl and simulator reset remain the full runtime's responsibility.

The BAM 1.0.1 write-set audit is guarded by these exact installed source hashes:

- `bam/mjlab.py`:
  `af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440`.
- `bam/actuator.py`:
  `6927d7b2341cbaca0e5d872fae4ddce6200056885310d9ee5acaba1f34e09c13`.
- `bam/dynamixel/actuator.py`:
  `8d3ed39d68758981204901dd18dcce11dcb107b842067e036d2b8ac7b1411cec`.

During local validation, the first type guard incorrectly expected the base
voltage-controlled class. Read-only inspection confirmed the actual
`bam.dynamixel.actuator.XL330Actuator` subclass, which inherits the audited
compute functions; the guard now requires that exact class. An independent
reference fixture replaced an invalid attempt to deepcopy native MjsActuator
handles. Neither adjustment changed the motor model or an installed library.

Tests use two owned CPU tensor worlds from actual native robot force snapshots,
not GPU worlds or a new hold rollout. They compare live rows across successive
calls with a separate direct-BAM reference, verify frozen rejected/inactive rows,
selective reset, finite-output atomicity, limiter agreement with NumPy, and delay
advancement on exactly the accepted motor rows. These components still do not
admit learned stance or GPU training.

Local validation: all 14 new control tests passed. The broader selection had
498 passes and one failure in the unchanged
`test_surviving_cpu_descendant_closes_the_cell_instead_of_advancing` test.
Its deliberately orphaned CPU child was detected as intended, but macOS returned
`PermissionError: [Errno 1] Operation not permitted` during the subsequent
SIGKILL group cleanup, masking the expected rejection. A focused rerun reproduced
that error; a later read-only-instrumented run passed without emitting a permission
error. The two failed-run process groups (38919 and 41178) were absent on later
inspection. This is retained as an intermittent cleanup issue, not a proven fix
or a fully green local regression result. No supervisor behavior was weakened or
changed; validate the unchanged process-group tests on Linux before GPU use.

Linux confirmation at exact source `ddd7a8c3f9e1846c92ca333eff8fa8784d44bf1b`:
the same broad regression selection passed 497 tests with two skips in 23.32 s.
All new motor/delay tests and the unchanged process-group supervision tests
passed. Both skips were the previously documented separately retained checkpoint
inputs at `test_foundation_command_session.py:80`; they are not model-loading
acceptance. CUDA was hidden. The post-test branch was clean and the GPU remained
idle at 0% / 12 MiB / 46 C with no compute process; both protected system services
remained inactive. The macOS cleanup issue is not claimed fixed by this Linux run.

Next unfinished integration: first-terminal physical/contact evidence capture,
full live-mask wiring with refreshed Warp state and ctrl/actuator/reset ordering,
then bounded actual CUDA validation and the declared disposable optimizer smoke.
Keep the source-bound evaluator and all numerical/launch gates unchanged.

## First-terminal contact evidence

`stance_contact_evidence.py` now reads the active prefix of Warp's shared contact
buffer into owned tensors and decodes contact-frame force/torque with the installed
upstream helper. It validates broadphase/contact counts, per-world constraint
capacity, world/rigid-geom IDs, finite active constraint forces, and included
constraint addresses before decoding. Pyramidal contiguous rows and elliptic
per-axis addresses are checked separately, including negative elliptic addresses.
Stale contact/constraint padding is not evidence. No installed decoder was edited.
The helper is pinned to MuJoCo Warp 3.8.1 and `support.py` SHA256
`3ac8475d5e41318d8289601ad41493245d95d2138023d47e0462f04b6c67b94a`.

`contact_summary` preserves the native hold's definitions: sum nonnegative
normal force for each floor/foot pair in either geom order; any other pair with
distance <= 0 is forbidden. Geometry IDs must remain bound to the compiled model
and asset manifest. The reader does **not** call forward or prove freshness; the
full runtime must own the required fresh forward solve and numerical diagnostics.
It is not a complete Warp warning/overflow detector or an integrated stepping loop.

`FirstTerminalContacts` retains one owned, immutable first-attempt record per
world, including qpos/qvel, the supplied physical snapshot, physics counter,
contact positions/frames/friction/forces, and an actual excessive torque proposal
when applicable. No reset/overwrite API exists. A new terminal flag must agree
with a physical/proposed-torque failure or an exact step-2500 timeout. Invalid
later rows cannot leave an earlier partial ledger update. Returned records are
copies, so subsequent solver-buffer reuse or consumer changes cannot rewrite
the first terminal. This small ledger does not retain a complete trajectory,
validate source/checkpoint identity, or replace the declared held-out evaluator.

Validation: 15 new contact/evidence tests passed within 116 focused local CPU
stance/BAM/hold tests in 7.27 s. Actual Warp CPU contact forces for a simple
two-sphere/floor fixture match native MuJoCo in both pyramidal and elliptic modes
within rtol1e-4 / atol1e-5. These are primitive contact fixtures, not a Duck hold
rollout or CUDA parity result. Ledger tests use explicitly synthetic rigid-Duck-
shaped states with those contacts, exercise ownership/first-terminal/timeout and
malformed-evidence behavior, and do not claim a trained policy. Read-only user
service inspection found only historical exited/failed Duck units, none running;
the GPU was idle and protected system services remained inactive.

The next integration step is the complete caller-owned loop: read fresh Warp
physical state, apply the common live mask to BAM/delay/ctrl/Euler accounting,
capture first terminals before any reset, and expose the exact actor/critic
observations. Then verify actual CUDA under a bounded retained probe, assemble
the source-bound evaluator, and only afterward consider the predeclared smoke.

Linux confirmation at exact source `59b75170da9ab74eaa3be3de5de12e1207163774`:
the broader stance, football, foundation and launch-guard regression selection
passed **512 tests with two skips in 32.45 s** on 100.100, with CUDA hidden and
an allowlisted environment. This includes all 15 new contact-evidence tests.
The two skips at `test_foundation_command_session.py:80` require separately
retained checkpoint inputs and do not establish actor-loading acceptance.
The remote worktree was clean after fast-forwarding to the exact published
feature-branch source. GPU checks during validation showed no compute process,
0% utilization, 12 MiB used and 46 C; both protected system services remained
inactive and unchanged. No optimizer or CUDA probe was launched. The existing
continuation task was verified active through September10 07:30 Asia/Shanghai;
the next work remains the complete stance runtime and source-bound evaluation,
not a repeated contact fixture or a claimed rolling-football capability.

## Caller-owned full-robot Warp loop

`stance_warp_runtime.py` connects normal mjlab Entity/BAM initialization, the
owned action limiter/FIFO, fresh forward solves, staged motor state, masked
Euler commits, substep rewards/stops and first-terminal contact retention.
It uses the actual full-collision Duck on a flat floor, not a primitive sphere
or a CPU tensor bridge posing as initialized Entity data. The runtime remains
unregistered with PPO and has no optimizer, checkpoint loader or service launcher.
CPU execution of this Warp path does not establish CUDA behavior or throughput.

The compiled plant matches the native hold's motor gear, force range, armature,
contact dimensions and friction arrays. The first implementation fixed voltage
before `edit_spec`, accidentally lowering the compiled force ceiling from about
1.06755 to 0.97642 Nm. An exact-spec regression caught that mismatch. The corrected
path compiles the historical motor specification first, then fixes the exclusively
owned initialization configuration at 7.5 V / 0.1 drop before normal initialize.
The independent 0.36 Nm pre-step gate is unchanged. Only the caller FIFO supplies
the three-step delay; stock delay is disabled so it is not applied twice. Exact
joint-name matching also avoids the irrelevant site-namespace warning.

The loop applies one common accepted-world mask to BAM history, model friction,
controls, FIFO and committed integration state. Each accepted step gets a
new-control solve, stock Euler candidate/commit and a refreshed post-step solve
before reward and failure accounting. The physical clock is checked against
exact accepted float32 timestep additions; integer counters govern episode gates.
Previously closed worlds are explicitly excluded from subsequent policy ticks,
including a world already at step2500. Their motor, physical and cached
observation fields remain unchanged while live siblings continue. First terminals
are owned records; an explicit selective reset returns the old records before
clearing selected rows, never auto-resets an evaluation attempt. The external
evaluator must retain the returned boundary prefixes and reset evidence.

Finite field/contact/constraint checks run before integration. The binding
requires the dense solver and a CCD pool equal to full shared contact capacity;
accepted broadphase/contact counts and per-world constraints must fit. Warp has
no native `warning.number` array: the false warning tensor is **not** proof that
all backend warnings were supervised. Actual GPU log/numerical supervision and
source/runtime/asset-bound evaluation remain required before any launch. A fault
closes the runtime and cannot be erased by reset; it does not promise rollback
of the entire control-and-solver transaction.

The nonzero angular-velocity test also caught a pre-existing native-audit frame
bug: `mjOBJ_BODY` selects the rotated inertial frame, whereas `mjOBJ_XBODY`
selects the regular body frame. This is confirmed by the installed MuJoCo3.10.0
read-only experiment and its [upstream implementation](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_core_util.c#L768).
`stance_native_runtime.py` now uses XBODY, matching the already-declared body-frame
observation and Warp Entity data. For the synthetic body angular velocity
(0.2,-0.3,0.4), the old inertial reading was approximately
(-0.29810,0.09879,0.43746). A new regression distinguishes the two frames; zero
velocity alone could not expose this bug. No retained hold trace, gait actor,
hop actor, trained normalizer or historical acceptance result was rewritten.

Local validation: **129 focused CPU tests passed in 6.44 s**, including twelve
new runtime/mask tests and one native-frame regression. Short real-robot Warp
tests exercise at most two controller ticks, delayed targets, finite reward,
normal initialization, and exact compiled plant fields. Explicit synthetic
tilt, excessive torque, near-timeout, nonfinite solve and wrong-clock injections
test refusal/isolation behavior, not policy skill or a completed five-second
attempt. Rotated nonzero-velocity tests check body-frame observation conventions.

Next: validate this exact chunk on Linux, then predeclare and run a separately
bounded actual CUDA integration probe with retained numerical/log diagnostics.
Finish the full source-bound evaluator and PPO/launcher adapter before the
declared disposable64-env/16-iteration smoke. No optimizer was launched here;
do not repeat the closed native hold or promote to football balancing.

Linux confirmation at exact source `cf0f5a0d2076c8a65e62272944b15f49dd1cf6ea`:
**525 tests passed with two skips in 31.37 s** in the broader stance, football,
foundation and launch-supervision selection, with CUDA hidden and an allowlisted
environment. All new full-robot Warp CPU and body-frame checks ran. The two
skips are the unchanged separately retained checkpoint inputs at
`test_foundation_command_session.py:80`, not stance failures or actor admission.
The worktree was fast-forwarded from clean exact `081e074`; post-test telemetry
showed no compute PID, 0% / 12 MiB / 46 C, and both protected system services
inactive. No CUDA probe, optimizer, full hold or ball rollout ran. Next work is
the source-bound CUDA validation/diagnostic entrypoint and complete evaluator;
use explicit `cuda:0` device identity and independently bounded launch guards.

The next [bounded CUDA integration probe](2026-09-09-stance-cuda-integration.md)
is now implemented and predeclared, with reviewed dependency Python tree pins
and168 focused local CPU tests passing. It has no optimizer or capability
admission. Complete Linux CPU validation and the exact leased/retained service
attempt before moving on to full evaluator and PPO adapter work.

CUDA confirmation: corrected source `f42a21cf592184276b89a4be17620e87a7fe6941`
passed the unique retained two-world probe in20.305s, after212 Linux checks.
The [probe evidence and exact hashes](2026-09-09-stance-cuda-integration.md)
record actual `cuda:0`, first-terminal isolation, selective reset, no foreign GPU
owner, peak sampled50 C, and idle closeout. Raw evidence is mirrored on the Mac.
Next unfinished chunk: bind complete first-attempt trajectory evidence to source,
runtime, actor/checkpoint and episode identity, then implement/test the declared
PPO adapter and disposable64-env/16-iteration smoke. Do not repeat this successful
integration probe or the closed hold merely to fill the authorized window.

The next [first-attempt trajectory recorder/replay component](2026-09-09-stance-attempt-trace.md)
now consumes actual runtime snapshots and reconciles per-world terminal prefixes,
pre-action observations, frozen duplicates and exact serialized trace hashes.
It is a tested evaluator component, not complete launch/restore/contact provenance
or a384-attempt held-out matrix. All policy-admission flags remain false. Finish
those bindings and the PPO adapter before the declared disposable smoke.

Strict fresh-model restoration and exclusive hash-bound file bundles are now
implemented in the [checkpoint/bundle component](2026-09-09-stance-checkpoint-bundle.md).

The next [compiled plant and recorded-state component](2026-09-09-stance-plant-evidence.md)
adds CPU reset, joint/observation mapping and terminal contact-summary checks to
the v2 bundle. It is not a new trained capability or authorization to start PPO.

The [control-path evidence component](2026-09-09-stance-control-evidence.md) follows
with opt-in correction/FIFO and motor-commit replay in explicit v3 bundles. It
does not independently recompute BAM solver outputs or admit a trained policy.

The [CPU PPO precursor](2026-09-09-stance-ppo-adapter.md) adds explicit terminal
bootstrap, selective reset, finite GAE/update checks and a learner-only fixture
codec. Synthetic optimizer continuation is not real simulation resume, the
declared smoke or pilot, or a new learned Duck capability.
It uses the installed RSL5.0.1 model API and preserves caller RNG state. Replayed
actions must agree with the restored deterministic actor; copied filenames or
positive receipts alone are insufficient. Full compiled-plant/reset/contact
provenance, exact held-out matrix and PPO/launcher assembly remain unfinished.
