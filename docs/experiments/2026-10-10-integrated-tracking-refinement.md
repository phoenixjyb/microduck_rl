# Integrated command-tracking continuation

## Declaration before execution

The [motor comparison](2026-10-10-integrated-motor-refinement.md) reduced load
but both arms failed the unchanged foundation gate. Its motor arm passed9/12
motor cases; all24 cases stayed upright. This next experiment repairs command
tracking, not admission to obstacles, hopping or football balance.

Reuse the bounded two-arm harness in `integrated_motor_refinement.py` through
`python -m mjlab_microduck.integrated_tracking_refinement <mode> --source
<exact-clean-commit> --run-id <unique-id>`. Both arms independently warm-load
native `artifacts/training/motor-refinement-4da92898-20261010/motor/model_999.pt`,
SHA256 `ee6d0f69e34fd73c34f3b1f307a2d4bde85f73d1ed81a53cba07ca5899091fc1`.
Require finite iteration999/common step24000; exactly restore actor, critic,
normalizers, distribution and Adam; synchronize Python LR with restored Adam;
reset both learning clocks tozero and start fresh episodes. No old-plant policy.

Preserve the source's consolidated posture/CoM/action-rate settings,61→14
contract, unfiltered actions, BAM plant, DR/noise, command sampling and learner.
Keep learned motor-load weight-1 from the first action in **both** arms; no
replay of the original motor ramp. The stock linear and angular tracking
Gaussians remain unchanged at weights2, stds√.1 and√.5. No upstream yaw-fix
toggle, actor expansion, world-frame command, new command bucket or action filter.

The sole treatment is `mdp.body_twist_tracking_cost` at constant weight-.5
versus0 in control. Positive raw cost is
`sum((cmd_xy-v_body_xy)^2) + (cmd_yaw-w_body_z)^2`, normalized by fixed1m/s
and1rad/s references, with no clipping. It is zero when moving exactly as
commanded, not a stillness tax. It ignores vertical and roll/pitch gait motion.
No weight escalation follows automatically. Existing tracking already includes
the command; this tests extra error shaping, not a missing-input repair.
Duplicate tracking pressure or rare large errors may trade away gait/uprightness;
record actual reward mass and preserve the strict downstream gates.

The pre-reset metrics hook retains raw cost even for the control's zero-weight
term. After every action, require raw finite/nonnegative cost and actual weighted
reward finite/nonpositive/equal to the same pre-reset cost times weight, including
terminal worlds. Raw NaN/Inf raises before reward sanitization can hide it.
Log mean/max raw cost and actual weighted reward each24-step update, alongside
falls, losses, motor torque/soft-limit/rated-speed/power/thermal proxies.

Matched training: seed881,256 worlds,1000 updates×24 steps per arm
(6,144,000 transitions each). Preflights for **both** arms: seed877,64 worlds/
5 updates with official baked-normalizer ONNX CPU61→14 finite inference;
seed879,256 worlds/10 updates timing the actual respective objective. Require
all four same-source/protocol/parent receipts and each benchmark×100×1.5 below
2040s before either full run. Save initial/every250/final999 atomically/fsync.

Retain native artifacts, one sequential user service, control→tracking→evaluate;
individual180s preflights,2100s arms,180s evaluation and4500s full-service cap,
15s kill grace, RAM8GiB/CPU200%/Tasks64/Nice10. Stop on any failed phase,
no retry or automatic next experiment. Hold the existing owned GPU lease;
exact GPU/DINO allowlist; aggregate≤6GiB,free≥8GiB,temperature<75C;
20% Torch allocator is not a Warp cap. Check every25 updates/between phases.
All four AI Mission user/system scopes stayinactive. Preserve unrelated work
and NX on100.98; no driver, venv or dependency changes.

## Unchanged decision and stop

Evaluate only each arm's final999, sequentially: seeds839/853/857,8 worlds,
first240 steps/4.8s, startup60 excluded only from tracking; commands0/0,
.30/0,.20/-.20,.20/+.20, neutral posture/no pushes/fixed commands/official
normalizer. Stop a case on its first terminal; no autoreset success laundering.
Every world must satisfy speed MAE≤.10 (idle≤.03), yaw MAE≤.10, torque
p99 utilization≤.60, zero rated-speed exceedance and zero terminals.
Retain both deterministic decisions and same-case `tracking_minus_control`
differences. No interim selection or weakened threshold. Even a passing matrix
is next-experiment admission, not policy publication or simulator/hardware
qualification. One training seed remains diagnostic, not robust causal proof.

Commit/push code/tests/declaration before native execution; focused CPU checks
and both GPU smokes precede full training. Stop after retaining this comparison.
No physical motion, raw perception, protected service restoration or new skill
job follows automatically.

## Source checks before native execution

The focused Mac suite passed69 tests in27.77s across integrated foundation,
motor/tracking refinement, motor-step stream and evaluation. One pre-existing
actuator/site selector warning remains. Tests cover exact cost values, command
versus absolute-motion semantics, ignored gait axes, NaN/Inf rejection, arm
parity, source-parent bindings, unchanged decision limits and cached pre-reset
reward verification despite a mocked terminal reset. Syntax, relative document
links and whitespace checks passed. Independent Luna read-only review found no
blocking issue and verified the installed mjlab1.3 reward→metrics→autoreset
ordering. CPU checks are not GPU-smoke or learned-skill acceptance.

## Native preflight and retained launch

Executed source `6dbc0613c7fd55f081bf4cebea292b9acec8a61e` was pushed before
execution. The clean native integration checkout fast-forwarded to it; the
historical research checkout and frozen venv were not changed.
`microduck-tracking-6dbc0613-preflight.service` completed with Resultsuccess,
ExecMainStatus0/MainPID0. Native focused checks passed74 tests in5.73s, with
the existing actuator/site warning. Both64-world5-update GPU smokes restored
the exact parent/normalizers/Adam, reset clocks tozero, and passed official
normalized ONNX CPU61→14 finite inference. Restored Python/Adam LR was1e-5.
The known initial-save logger warning remained; final smoke ONNX exports were
required and passed, not waived. Actual tracking weighted reward was negative
in treatment andzero in control, while both kept motor weight-1.

256-world10-update control/tracking benchmarks took8.202083/8.346592s;
conservative full-arm projections1230.31/1251.99s fit2040s. Four small receipt
copies on Mac match native SHA256, without duplicating models:

| Receipt | SHA256 |
| --- | --- |
| Smoke control | `a203609c469f4dfe24c315d3c13dc05633878759d0548410645e563c325fb77e` |
| Smoke tracking | `43ba758b428e31e3c7088fdd0561855d4fb5ab1874c46e506b7d2752aa13edef` |
| Benchmark control | `0e5d5cd190829c72d3b9cade797b052da63268602fab9fd09f551a6a6e0f09fb` |
| Benchmark tracking | `e9c34ec3d6bb811c1058353641dafdf3194471af5760d500133233cb9f2809b5` |

`microduck-tracking-6dbc0613-matched-eval.service` started2026-10-10 18:25:41
Asia/Shanghai. Live properties confirm4500s/RAM8GiB/CPU200%/Tasks64/Nice10/
KillModecontrol-group. It runs control then tracking then evaluation with
independent2100/2100/180s timeouts,15s kill grace and `set -e`. Artifacts remain
native under `artifacts/training/tracking-refinement-6dbc0613-20261010`.

At42 control updates/1008 steps, losses and raw command/motor costs werefinite,
latest6144 transitions hadzero falls/NaN terminations, and motor weight remained
-1. A durable initial checkpoint exists. Latest sampled GPU occupancy1710MiB/
53C included only Duck PID816869/746MiB and preserved DINO PID1592/946MiB;
all four protected service scopes remainedinactive. This is healthy launch
evidence, not a completed training result or improved tracking claim. Both
full arms and their unchanged held-out decisions remain pending.

## Completed comparison

The launch snapshot above is superseded: the retained service finished
2026-10-10 18:54:10 Asia/Shanghai, Resultsuccess/ExecMainStatus0/MainPID0.
Control/tracking completed1000 updates each in1003.007/588.700s, with
6,144,000 transitions per arm. All1000 logged update rows per arm werefinite,
withzero NaN terminations. Final999 payloads werefinite at iteration999/
common step24000, with17 Adam states at step70000. Training falls in the last
100 updates were153 control/114 tracking over614,400 transitions each;
training is stochastic/DR-enabled, not the deterministic held-out protocol.
Maximum retained GPU temperature was59C control/61C tracking, memory1710MiB.

Both held-out matrices are **foundation-not-ready**. All24 cases completed
240 steps upright withzero terminals. Control passed12/12 motor-envelope
cases and2/12 tracking cases; treatment passed6/12 motor and3/12 tracking.
All9 moving cases failed tracking in each arm. All three treatment idle cases
failed motor torque (p99 utilization1.06754), and all three positive-yaw cases
failed motor torque. No treatment case passed the full foundation conditions.
The control's two full-case passes do not admit a12-case matrix with10 failures.

Case means across three seeds/eight worlds are descriptive, not all-world gates:

| vx / yaw command | Control speed MAE | Tracking speed MAE | Control yaw MAE | Tracking yaw MAE | Control / tracking torque p99 utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | .00147 | .00187 | .02147 | .02168 | .49171 / 1.06754 |
| .30 / 0 | .10546 | .11267 | .21792 | .16311 | .57758 / .52046 |
| .20 / -.20 | .05575 | .05558 | .25407 | .19370 | .49104 / .56983 |
| .20 / +.20 | .06029 | .05809 | .26553 | .20107 | .54404 / .66399 |

Added error shaping reduced moving yaw MAE about24–25%, but straight speed
error worsened and idle motor load regressed. Last100-update mean raw command
cost was.658924 control/.405717 treatment; the actual treatment weighted term
was-.202859. Lower training cost does not override failed held-out gates or
demonstrate command-correct turning. Reject this treatment as the next parent.

Exact SHA256 bindings (models remain native, only small result JSONs mirrored):

| Artifact | SHA256 |
| --- | --- |
| Control model999 | `b7a6601c479ef6805d4fc68236deff1805e6997bea4286e5b0f69b7b3255deec` |
| Tracking model999 | `5d03c2241d5291ded5786ed87ec85cd70f8262fcd47954837114e1102dfe6fc9` |
| Control result | `798ec743caafb85f954775dbda2d1a4c233379483d5e1f191887eb3e7c4132ac` |
| Tracking result | `933e8626b2da6bd6181e28ae936fa5792fc804f60aa6ec04b2adbc19a9e23a2c` |
| Evaluation result | `e803f630a02aa6286b4b334334ea6696ada7c458849726f2237af645a29c6af3` |

CPU verification recomputed both decisions and every paired difference;
all24 standalone case records match aggregate entries, with exact declared
seed/world/update/parent/final-checkpoint bindings. CUDA-hidden native load
checked final model/Adam finiteness and clocks. At23:31:10 Asia/Shanghai the
GPU had15232MiB free/45C, DINO PID1592 only, no Duck process; all four protected
service scopes remainedinactive. No workload was stopped or restored.

## Response audit declaration before further learning

Run one bounded diagnostic-only replay of the **same two final999 checkpoints**
and original24 cases through `python -m mjlab_microduck.integrated_response_audit
--source <exact-clean-commit> --run-id <unique-id>`. Recheck clean source,
owned GPU lease, same frozen runtime, exact evaluation and checkpoint hashes
above and the existing DINO/occupancy/protected-service guards. One sequential
user service, child180s/outer240s, RAM8GiB/CPU200%/Tasks64/Nice10,15s kill
grace; no learning, MP4, new command distribution, physical motion or next job.

Opt-in evaluator diagnostics record per-world signed body vx/yaw means,
absolute yaw means and nonzero-command correct-sign fraction after startup60.
Per-actuator absolute force p99, signed mean force and sampled soft-limit
fraction include all steps. Joint names/entity-local IDs come from the existing
direct unit-gear hinge actuator mapping, not array width or raw qpos offsets.
Hard-stop proximity uses the actual hard joint bounds and fixed.05rad margin,
with hard-range violation recorded separately. Qpos is sampled pre-action;
force is the existing post-decimation pre-reset stream, explicitly different
clocks rather than a purported simultaneous force/position causal test.

Default acceptance rows/decision are unchanged. Replay must match every old
acceptance field (absolute floating tolerance1e-6; identities/booleans/counts
exact), and reproduce both rejected decisions. Any mismatch fails the audit;
do not overwrite or silently replace original evidence. Retain separate JSONs
and diagnostics with policy/physical acceptancefalse. This diagnoses weak
steering versus oscillation and the idle torque source before another reward
or curriculum change; do not promote the rejected tracking arm.

Audit source checks passed77 focused tests in6.58s (existing actuator/site
warning), plus an8-test audit rerun after tightening numeric replay type checks.
Syntax, links, whitespace and all retained result hashes passed. Independent
Luna review found no identity/acceptance confound and required the declared
external service wrapper before GPU execution: the Python module's guards
are not a replacement for cgroup/watchdog caps. The run must use that wrapper,
not an uncapped direct invocation; native CPU checks get a60s timeout inside
the240s overall service before the180s diagnostic child.

### Strict replay failure and bounded diagnosis declaration

The first audit at source246b2ce2 failed its first comparison with
`ValueError: diagnostic replay differs from original acceptance evidence`.
Native CPU checks had passed82 tests in5.59s; the GPU audit stopped, with no
case admitted and no Duck process left. Original training/evaluation artifacts
remain unchanged. The original implementation did not persist a mismatched
row before rejection, so it could not explain the numerical difference. Fix
that observability defect: retain raw case and field-level comparison before
the existing strict check, **without widening1e-6 or changing any gate**.

Installed mjlab1.3 `utils/random.py` seeds Python/NumPy/Warp/Torch but explicitly
documents that MuJoCo Warp is not fully deterministic. `evaluate_case` seeds
the environment; the optional joint-position read is a read-only qpos view,
whose CPU copy adds synchronization but not RNG or simulator-state writes.
This source evidence suggests replay variability is possible; it does not yet
establish the observed cause.

Before any full replay retry, run two fresh sequential diagnostic children
using `--case-diagnosis default` then `--case-diagnosis diagnostic`. Each is
bounded to the first held-out seed839's four commands for both final policies
(eight cases,8 worlds,240 steps), same checkpoints/runtime/guard/lease, with
original-field differences retained rather than asserted as exact-replay success.
Default mode collects no new telemetry; diagnostic mode adds it. Both explicitly
set replay_verifiedfalse and diagnostic-only-not-admission. Preserve original
rejected decisions; partial seed coverage cannot override them. No full-matrix
retry or new learning job follows. Use one240s service with60s CPU checks and
independent75s default/diagnostic child timeouts,15s grace and the same resource
caps. Retain failures and compare default-versus-diagnostic drift before inferring
a cause or choosing a curriculum change.

The follow-up capture/diff changes passed78 focused CPU tests in6.72s, with
the same existing selector warning. Strict replay tolerance remains1e-6;
the partial observational modes are explicitly not full replay verification
or acceptance and preserve the old rejected matrix.

### Completed observational diagnosis

Source `31a5fb6aebf746d354cbfc87b17bfc5db0b49034` was pushed before the
clean, idle native checkout fast-forwarded. The sequential retained service
`microduck-response-31a5fb6a-diagnosis.service` ran from23:50:53 to23:52:19
Asia/Shanghai, Resultsuccess/ExecMainStatus0/MainPID0. Its live wrapper had
240s/RAM8GiB/CPU200%/Tasks64/Nice10/KillModecontrol-group; native CPU checks
passed83 tests in5.54s with the existing selector warning. Both fresh child
processes completed their eight declared cases, each8 worlds/240 steps, with
zero terminals and zero rated-speed exceedance. These are partial descriptive
replays, **not an accepted policy or a full strict-replay pass**.

Small retained JSONs were mirrored without copying models. Native and Mac
launch/result SHA256 values agree; every standalone case equals its aggregate
entry, with source, original evaluation and two checkpoint identities intact.

| Receipt | SHA256 |
| --- | --- |
| Default launch | `20217939bb370a70e189d7124b4782ed7a80019f41ef18b4ed8c094d35201a45` |
| Default result | `e97379fa76b945e4e08d9a74fecb59dc9d20bc6118e2293bf980f653163a5c04` |
| Diagnostic launch | `15b5d836798b50d77cf1ba879ef1b1c1a090297583f6932f8771a874b9b08b32` |
| Diagnostic result | `9f94a08c6ccc642ef34f5f2e718166c595d1ae8d58ecb3382fcf74cfcac94657` |

Artifacts remain under native `artifacts/evaluations/` in
`tracking-replay-default-31a5fb6a-20261010` and
`tracking-replay-diagnostic-31a5fb6a-20261010`; the Mac has matching small copies.
The original failed strict audit is retained separately and not overwritten.

Default/diagnostic replays had154/156 original scalar fields outside1e-6,
respectively, all numerical: no differing identities, types, completion,
terminal lists, or rated-speed exceedance. Maximum absolute yaw-MAE deviations
from original were.0268083/.0184502rad/s. Default-versus-diagnostic also differs
in every case. Thus added qpos telemetry is **not necessary** for observed
drift. This is consistent with the installed simulator's nondeterminism warning,
not a proof of the precise mechanism or permission to widen replay tolerance.
Original foundation-not-ready decisions remain authoritative; both new results
explicitly retain replay_verifiedfalse and policy_acceptancefalse.

Instrumented seed839 means across eight worlds, after60 startup steps:

| Command vx/yaw | Control signed vx | Tracking signed vx | Control signed yaw / absolute yaw | Tracking signed yaw / absolute yaw |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | .000218 | .000024 | .00582 / .01506 | .00677 / .01548 |
| .30 / 0 | .19712 | .19125 | -.01140 / .20467 | .04203 / .16413 |
| .20 / -.20 | .14542 | .14676 | -.06423 / .20692 | -.05190 / .13002 |
| .20 / +.20 | .14063 | .14319 | .05309 / .23201 | .12852 / .20751 |

Treatment negative-turn correct-sign fraction was.63611 versus control.61181;
positive-turn fraction.72569 versus.61181. The negative-turn error improvement
mostly accompanies reduced angular oscillation, not stronger mean commanded
turning; positive-turn signed response improved, but motor and tracking gates
still failed. One seed/four commands cannot establish generalization.

The idle load regression localizes to `right_hip_yaw`: treatment per-joint
absolute torque p99=.640524Nm, signed mean=.605716Nm, soft-limit exposure
95.8854%, versus control.349149Nm/.227990Nm/.1042%. Treatment idle pooled
torque utilization remains1.067539, reproducing the original motor failure.
Both arms spend substantial samples beyond that joint's configured range:
control idle96.875%, treatment idle97.5%; maximum excursion beyond the nearest
bound.048876/.059993rad. It is the only joint with sampled range violations
in these eight cases. Qpos and force have different sampling clocks; these
statistics localize a problem, not a synchronous force/constraint causal proof.

Read-only native CPU entity/compiled-model inspection verified all14 unit-gear
actuator targets against entity joint identities. `right_hip_yaw` is entity
joint9, compiled joint10, qpos address16; range[-.523598775598297,
.43633231299858416]rad, limited1. Installed EntityData reads qpos by its joint
address mapping, and Entity initializes limits from corresponding compiled
joint ranges. The action configuration has scale1/default-position offset and
no explicit action clip. This rules out a simple column-name swap but does not
establish why the solver/policy exceeds the range. MuJoCo constraints are not
physical hard-stop qualification; do not claim physical safety from this audit.

Next bounded diagnostic should capture the named joint's requested target,
actual position, configured limits and constraint response before changing the
policy/plant. If persistent out-of-range targets are confirmed, predeclare a
small matched joint-target-safety continuation from the **control** parent,
with existing motor/tracking gates plus explicit range-exposure checks. Do not
promote the rejected treatment, silently clamp the evaluator, or expand the
speed/obstacle curriculum until the joint issue is understood. No further GPU
job was launched in this diagnosis.

Peak retained default/diagnostic GPU occupancy1692/1860MiB, temperature55/58C.
After completion the GPU returned to961MiB used/15232MiB free/50C with only
preserved DINO PID1592. Both protected services remainedinactive in user and
system scopes. No unrelated workload, dependency, driver, physical robot or
100.98 NX workload was changed.

### October11 target/constraint capture declaration

Before changing the recipe, add opt-in `--case-diagnosis targets` to the existing
bounded audit. Replay only seed839's four commands for both exact final999
policies above, eight cases,8 worlds/240 steps each. Retain unchanged original
gate evidence, replay_verifiedfalse, no policy admission or learning. Capture
actual entity joint-position target buffers at the next pre-action boundary
(previous applied target, including encoder-bias correction), the named joint
position, configured limits, and generalized constraint torque via entity
joint DoF addresses. Generalized constraints include contacts and limits and
have solver integration lag: do not label them isolated hard-stop forces or
claim synchronous physical causality. No extra forward(), action processing,
target clipping, RNG draw, reward or plant change is permitted.

Use the same regular owned lease/DINO identity/protected-service guards and
frozen runtime. Run one sequential retained user service with240s outer cap,
60s CUDA-hidden focused tests and150s GPU child timeout,15s kill grace,
RAM8GiB/CPU200%/Tasks64/Nice10/KillModecontrol-group. Inspect the target and
constraint evidence before selecting any learning continuation. Repository
guidance and independent read-only Luna review favor the existing
`joint_pos_limit_proximity` penalty on the offending named joint rather than
target clipping, which would change intentional low-kp overshoot behavior.

### Completed target capture and matched qpos repair predeclaration

The target service completed00:28:30 Asia/Shanghai, Resultsuccess/MainPID0/
ExecMainStatus0; native focused checks passed56 tests in5.51s. All eight
seed839 cases completed240 steps without terminals. Receipt copies and case
aggregates agree, with the original exact checkpoint/source/evaluation bindings.
Launch SHA256 `7f400046877e0e57326858cccd0d5f20b1a2f2e50d776939bc5fdf8a659327aa`;
result SHA256 `8d2d18ae78ec17519051589c9c7e5bf2b4429320eeb2334fc32c39a00dce3a93`.
Native and Mac small copies are in
`artifacts/evaluations/tracking-target-2f2a910d-20261011`.

Right-hip-yaw idle means: control actual position.429428rad / previous applied
target.867787rad / generalized constraint torque-.225185Nm; tracking actual
.434857rad / target1.636049rad / constraint-.602251Nm. Upper configured range
is.436332rad. Targets were beyond configured range99.2188%/99.5833% of samples;
max actual positions.484981/.496325rad. This supports persistent commanded
overshoot and opposing generalized constraints, not an isolated hard-stop
force measurement or a complete causal account. Both checkpoints stay rejected.
Target capture peaked1860MiB/55C, then returned to the preserved DINO only;
protected services remainedinactive. No plant/action/dependency was changed.

Predeclare `integrated-qpos-limit-continuation-v1`, run by
`python -m mjlab_microduck.integrated_joint_limit_refinement`. Both arms start
from **control final999**, SHA256
`b7a6601c479ef6805d4fc68236deff1805e6997bea4286e5b0f69b7b3255deec`,
iteration999/common step24000, not the rejected high-load tracking checkpoint.
Strict warm load retains actor/critic/normalizers/Adam and synchronizes Python
LR with restored Adam; resets iteration/common clock and starts fresh episodes.
This is an unaccepted exploratory continuation of the new integrated plant,
not transplantation from an older model or physical qualification.

Matched full arms: seed907,256 worlds,1000 updates,24 steps/update,
6,144,000 transitions per arm, save cadence250, final999 only for evaluation.
All inherited commands, randomization, reward weights and frozen source
curricula stay as in the tracking control (motor cost-1, extra command-error
weight0). The **sole intervention** is existing `joint_pos_limit_proximity`
scoped by exact entity name `^right_hip_yaw$`, with.15rad margin. Control weight0;
treatment-1 initially, -2 after common step6000, -4 after12000 (strict existing
step boundary). No target clipping, filter, solver or motor parameter change.
Both arms log the same raw pre-reset position cost, stop proximity and maximum
overrun, plus existing motor/tracking/NaN/fall metrics. Verify live negative
weighted cost equals raw cost times actual manager weight, including terminals.

Run64-world5-update smokes at seed887 for **both** arms before training,
requiring official normalized ONNX CPU61-to14 finite inference. Treatment
smoke compresses its ramp to updates1/2; run256-world10-update benchmarks at
seed889 (treatment full-4) and require conservative measured projections fit
2100s/arm. Preflight service caps360s/RAM8GiB/CPU200%/Tasks64/Nice10 with
60s CPU checks and independent60s per smoke/benchmark,15s kill grace.

Only after both preflights pass, launch one sequential retained service:
control then limit then held-out evaluation, no overlap with other Duck jobs.
Full caps4500s overall,2100s each training child/240s evaluation child,
RAM8GiB/CPU200%/Tasks64/Nice10/KillModecontrol-group; regular owned GPU lease,
existing DINO-only shared-host guards and protected-services-inactive checks.
Repository must be clean at the exact pushed source. Do not change active jobs.

Evaluate the original seeds839/853/857 and four vx/yaw cases,8 worlds/240
steps,60-step tracking startup exclusion. Keep every original tracking,
terminal, motor-torque and rated-speed gate; additionally require all14 joints'
maximum configured-range excursion<=.005rad and right-hip-yaw exposure within
.05rad of a bound<=1% of all-step samples per case. Bind diagnostic protocol,
margin, named coverage and step counts. No threshold tuning after seeing results.
Report original decisions separately alongside stricter combined decisions.
Qpos improvement alone cannot admit a policy that still fails tracking.
Stop after retaining this matched result; no automatically chained curriculum,
obstacles, videos, raw perception, policy publication or physical motion.

Initial focused suite passed95 tests in6.34s with the existing selector warning;
the one earlier cfg-equality test failure was a test-only integer0 versus
float0.0 repr mismatch, corrected without changing recipe behavior. Independent
Luna review found no preflight blocker; its margin-identity hardening suggestion
was incorporated before execution.

### Qpos repair preflight and retained training launch

Exact recipe source `1689e874e73def563a20eed13011f788f30d5449` was tested
and pushed before the clean native checkout fast-forwarded. The final Mac
focused suite passed95 tests in6.72s; native CUDA-hidden checks passed95 in5.97s,
with only the existing actuator/site selector warnings. Syntax, documentation
links and whitespace checks passed. The independent review performed no edits
or GPU operations; owner inspected and incorporated its margin identity guard.

`microduck-limit-1689e874-preflight.service` completed Resultsuccess/MainPID0/
ExecMainStatus0. Both64-world5-update smokes restored the exact control parent,
all models/normalizers/Adam, clocks0 and Python/Adam LR1e-5, and passed official
normalized ONNX CPU61-to14 finite inference. Treatment exercised -1/-2/-4
live weights (boundary updates can contain two stages); control remained0.
All logged preflight reward/loss/motor/qpos fields werefinite withzero NaN
terminations; actual negative weighted qpos reward equaled the pre-reset raw
cost times weight. Benchmark/full mode preserves motor-1 and command-error0.

Ten-update256-world benchmarks took7.665340s control/7.681333s treatment.
Conservative1000-update projections1149.80/1152.20s fit2040s, leaving60s inside
each2100s child cap. These are timing/readiness receipts, not evidence that
the new policy passed numerical gates: training DR still produces falls,
nonzero torque exposure and occasional rated-speed exceedance in benchmarks.

Small result receipt copies at Mac
`artifacts/evaluations/qpos-limit-1689e874-20261011-preflight` agree with native
SHA256; no checkpoint/venv was copied:

| Receipt | SHA256 |
| --- | --- |
| Smoke control | `5ed617e69d9330772a0c94aa8d1193e002a2bf521a27063a4e4fc22621f480bb` |
| Smoke limit | `a71adbfefcc159328c7312ab22d3cf64e57f59c3045fc534733a9ce44af34496` |
| Benchmark control | `ad018e3f8f63666e2073a8a783d7bd8707f0745fdd23a7580a51f7a79bb3f910` |
| Benchmark limit | `9fe9c133e429016340ec1562a32849f2b01f28e7ab35bbefbe227910aba91cc3` |

`microduck-limit-1689e874-matched-eval.service` started2026-10-11 00:34:43
Asia/Shanghai. Live properties confirmed4500s/RAM8GiB/CPU200%/Tasks64/Nice10/
KillModecontrol-group. Its `set -e` wrapper runs control then limit then the
stricter held-out evaluation with2100/2100/240s child caps and15s kill grace.
Native run root is `artifacts/training/qpos-limit-1689e874-20261011` in
`/home/converge/work/microduck_rl-upstream-20261010`. Stop after this result;
do not automatically chain another experiment or promote a failed matrix.

At40 control updates/960 steps, all logged metrics werefinite withzero NaN
terminations and52 training falls across245,760 transitions; motor weight-1,
command/error and qpos control weights0. Durable model0 exists. Sampled GPU
occupancy1710MiB/free14484MiB/56C included only Duck PID1078078 and preserved
DINO PID1592; both protected services remainedinactive in both scopes.
This is a healthy active-job snapshot, not a completed result or a joint-repair
claim. Do not alter this checkout or active job while its later phases still
require source1689e874. Documentation-only follow-up commits may be pushed on
Mac without advancing the native running checkout. Both full arms and their
unchanged-plus-range held-out decisions remain pending.

### Completed qpos repair comparison

The launch snapshot is superseded. The retained matched service completed
2026-10-11 00:56:16 Asia/Shanghai, Resultsuccess/ExecMainStatus0/MainPID0.
Both arms finished1000 updates/24000 common steps/6,144,000 transitions,
in591.924902s control/585.985689s limit. All1000 logged rows per arm werefinite
withzero NaN terminations. CUDA-hidden native verification checked exact source,
seed/world/update/parent bindings, durable final999 bytes, finite model/Adam,
iteration999/common step24000 and17 optimizer states at Adam step90000.
Retained training health peaked1710MiB/60C for each arm.

Recomputed stricter and original decisions exactly match the retained aggregate;
all24 standalone case records equal aggregate entries and every paired delta
recomputes. Both overall decisions remain **foundation-not-ready**. All24 cases
completed240 steps withzero terminals andzero rated-speed exceedance. Control
passed9/12 motor cases and0/12 joint-range cases; limit passed12/12 motor and
12/12 joint-range cases. The only limit-arm failures are tracking in all nine
moving cases. Its three idle cases pass all combined numerical gates, which
does not admit the twelve-case matrix or qualify physical motion.

| Command vx/yaw | Control / limit speed MAE | Control / limit yaw MAE | Control / limit torque p99 utilization | Control / limit right-hip stop fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | .001060 / .001268 | .011682 / .009030 | .871514 / .233140 | .975000 / 0 |
| .30 / 0 | .111252 / .116308 | .208235 / .334012 | .501551 / .491571 | .970660 / 0 |
| .20 / -.20 | .060630 / .063556 | .207591 / .296008 | .486962 / .503170 | .973090 / 0 |
| .20 / +.20 | .062183 / .066582 | .243385 / .321521 | .505547 / .493308 | .973090 / 0 |

Means above are descriptive across three seeds/eight worlds, not replacements
for per-world tracking gates. The treatment removes the measured stop-pushing
behavior in this held-out matrix and reduces idle torque utilization73.25%,
but moving yaw error worsens. Across all twelve cases, mean soft-limit exposure
is.009818 control/.000747 limit, mechanical-power proxy1.126896/1.008303W,
and thermal-load proxy.033940/.022179. These are sampled simulator proxies,
not physical temperatures or power guarantees. During randomized training,
last100-update falls were124/169, so do not generalize the upright held-out
result into improved stability under every disturbance.

Last100-update mean right-hip stop exposure was.886340 control/.003483 limit;
raw qpos proximity cost.138297/.001863 and motor cost.075128/.066108.
Original checkpoints/results remain native; only small JSON evidence was
mirrored to Mac `artifacts/evaluations/qpos-limit-1689e874-20261011`.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `2be067c8e2fdcad4bab72b9fa8df5aaac088e0afd8f5ac473219e6dd89eef1f8` |
| Limit final999 | `212a567984014f90054cbd7137e7560d8c76a00496a3ca15958407b9de746e6b` |
| Control result | `a530b6af3df55246806f39636e7624090e17420f3205b9b153c0aefea2fd4c3b` |
| Limit result | `2526605628932202107f5100899dfbd88f7afc540b3663d3c0f8e196b675b9b8` |
| Held-out result | `c54468f4fc18c8cd1a72af4c44554f7e8927dab41ef710f3c41d74995c61772b` |

### Range-retaining tracking continuation predeclaration

Prepare `integrated-range-retaining-tracking-continuation-v1` in
`integrated_safe_tracking_refinement.py`, with no GPU execution yet. Both arms
warm-load the **limit final999** bytes above, actor/critic/normalizers/Adam
retained, Python LR synchronized to Adam, fresh episodes and reset clocks0.
This parent is allowed only as an exploratory range/motor-repaired baseline;
its full foundation gate is still rejected. No policy promotion is implied.

Both arms keep joint-limit weight-4/.15rad margin, motor weight-1, unchanged
action/plant/commands/DR and frozen source curricula. Sole treatment difference:
extra body-twist tracking cost weight-.125 initially, -.25 after6000 steps,
-.5 after12000 steps (existing strict stage boundary); control0 throughout.
Matched seed919/256 worlds/1000 updates/24 steps/update, save cadence250;
evaluate final999 only. Small tracking stages test whether error pressure can
recover moving command response without reintroducing the measured stop hack.
Do not broaden speed/obstacles or alter the numerical thresholds.

After GPU recovery, require fresh64-world5-update smokes for both arms at
seed911, treatment stages compressed to updates1/2, official normalized ONNX
CPU61-to14 finite inference;256-world10-update benchmarks at seed913 with
treatment full-.5 and conservative measured projections inside2100s/arm.
Same360s preflight wrapper/60s CPU and individual GPU children,8GiB RAM,
CPU200%/Tasks64/Nice10/KillModecontrol-group/15s grace. Only after preflight
passes use one4500s retained sequential service, control/track2100s each and
evaluation240s, with exact clean pushed source and owned GPU lease.

Evaluation/decision reuse the **stricter qpos repair recipe** unchanged:
seeds839/853/857, four command cases,8 worlds/240 steps, startup60 for tracking,
all original motor/tracking/terminal gates plus5mrad maximum range excursion
for all14 joints and<=1% right-hip exposure within.05rad of configured bounds.
Both arms capture/validate pre-reset raw and actual weighted motor/qpos/command
costs. Stop after retaining that matrix; no automatically chained skill,
raw perception, publication, video or physical motion.

Independent read-only Luna review found no recipe/inheritance/gate blocker.
Its additional live-observer boundary-test suggestion was incorporated for
both control and tracking at steps0/6000/6001/12000/12001 before execution.

### New native GPU driver blocker, diagnosed read-only

At07:14 Asia/Shanghai, `nvidia-smi` fails with
`Failed to initialize NVML: Driver/library version mismatch` / library595.99.
Kernel journal gives exact client595.99.02 versus loaded module595.91.07.
`/proc/driver/nvidia/version` and `/sys/module/nvidia/version` agree on595.91.07;
NVML symlink points to libnvidia-ml.so.595.99.02. `modinfo` reports replacement
module595.99.02, and DKMS reports it installed for running kernel6.8.0-138.

APT history records `/usr/bin/unattended-upgrade` at06:14:56–06:15:35 upgrading
the NVIDIA driver/libraries/DKMS from595.91.07 to595.99.02. Host last booted
October9 19:35:05. Thus the automatic update changed disk components after
the Duck job completed, while the old module remains loaded. This is separate
from training/evaluation failure: those results finished over five hours earlier.

No Duck training/evaluation process remains. DINO PID1592 is still running;
both protected services are currentlyinactive, but the system units areenabled
and could restart on reboot. Current GPU occupancy cannot be certified because
NVML fails. Do not bypass shared-host checks, launch CUDA, unload modules,
stop DINO, change packages or reboot under this diagnosis. A coordinated reboot
requires explicit confirmation of its impact on unrelated workloads; after it,
recheck module/library agreement, protected-service state and actual occupancy
before native preflight. 100.98/NX remains out of scope. The next recipe is
CPU-prepared only; no replacement GPU job was launched.

Range-retaining tracking source `5715a15636728e7b0a179df556a3ffb2cda4df7e`
passed102 focused Mac tests in6.68s, syntax/link/hash/whitespace checks, and was
pushed before the clean, inactive native checkout fast-forwarded. The bounded
CUDA-hidden native service `microduck-safe-track-5715a156-cpu.service`
completed07:19:43 Asia/Shanghai, Resultsuccess/ExecMainStatus0/MainPID0,
with102 tests passing in6.02s. Only the existing actuator/site selector warnings
remained. No GPU smoke/benchmark/training/export of this new stage has run;
CPU success does not clear the NVML mismatch or qualify a new learned policy.
Driver/library recovery, workload coordination and fresh GPU preflights are
the remaining launch gates. No reboot or service/driver mutation was performed.

### Owner reboot and explicit shared-worker rebinding

The preceding blocker snapshot is superseded. The owner rebooted100.100;
live SSH reports boot2026-10-11 07:46:44 Asia/Shanghai. Loaded NVIDIA module,
on-disk module and NVML now agree on595.99.02. No package/driver modification
was performed by this task. The protected user vLLM service restarted and
allocated9692MiB. Under the existing Duck reservation authorization, only the
two named user mission services were stopped; both user/system scopes now
reportinactive. DINO was preserved, not restarted or stopped.

The retained DINO worker now has PID2021, confirmed by NVML's exact executable
name, `/proc/2021/exe` and UID1000. Preflight must explicitly bind
`MICRODUCK_RETAINED_DINO_PID=2021`. The default remains the original dated PID;
there is no runtime process-name discovery or permissive competitor whitelist.
Every health check verifies canonical PID, exact executable and same owner,
exact NVML worker count, no GPU process other than that worker/own PID, GPU UUID,
6GiB aggregate cap,8GiB minimum free,<75C, and all four protected scopesinactive.
A worker restart fails closed and needs a new verified binding. PID is an
operational binding only; recipe, parent bytes, policy/plant and gates are unchanged.

The post-stop NVML check reports961MiB used/15232MiB free/45C, DINO946MiB only.
Retained limit parent and held-out aggregate hashes match the completed results
above. CPU validation and independent guard review precede clean pushed source,
native checks, fresh GPU smokes/benchmarks and any bounded matched continuation.
This recovery is not learned-policy acceptance or physical qualification.

Independent read-only Luna review found no fail-open path. Its wrong-NVML-name
and self-PID coverage suggestions were incorporated. The focused Mac suite
passed123 tests in5.50s; whitespace checks passed. Native CPU and GPU preflight
evidence remains pending for this exact updated source.

### Recovered native preflight and bounded continuation launch

Exact reviewed source `be487638129a9894d6cc77d87b4d0cdfc06c446c` was pushed,
then fast-forwarded into the clean inactive native checkout. Its read-only host
guard passed with explicit DINO PID2021. The retained preflight service
`microduck-safe-track-be487638-preflight.service` completed07:52:18,
Resultsuccess/ExecMainStatus0/MainPID0. Native CUDA-hidden tests passed123 in5.51s.

Both64-world5-update smokes and256-world10-update benchmarks verified exact
repaired parent bytes, restored models/normalizers/Adam, fresh clocks0,
Python/Adam LR1e-5, finite saved state and exact update/common-step counters.
All30 update rows werefinite withzero NaN terminations. Actual pre-reset
motor/qpos/command weighted costs were checked live; motor stayed-1 and
qpos-4, while treatment smoke exercised-.125/-.25/-.5 and benchmark-.5.
Both official normalized ONNX exports passed CPU finite61-to14 inference.

Benchmarks took7.465244s control/7.472858s treatment; conservative1000-update
projections1119.79/1120.93s are inside2040s, leaving60s in each2100s cap.
Health receipts peaked1710MiB aggregate/57C. Randomized benchmark falls10/8
and nonzero motor exposure are training observations, not held-out acceptance.
Small JSON copies under Mac
`artifacts/evaluations/safe-tracking-be487638-20261011-preflight` match native
hashes; no checkpoint/venv duplication was needed.

| Receipt | SHA256 |
| --- | --- |
| Smoke control | `777184819a6d1458f28c6844f1e35769ceaf12d4f175d96c01af50ee670c0fbe` |
| Smoke tracking | `743c0165f69c4531aa5bb293f734ec7bded2ecf444e35070fa2f4bd2dea80a19` |
| Benchmark control | `0fff4de0b1dd46216a65d574578f4833de96b3e7bc5ac4206c663cf6098b01b1` |
| Benchmark tracking | `10b865cce942021ab9bf5bea601063db42bc414c28991706ac1d3cff931852db` |

`microduck-safe-track-be487638-matched-eval.service` started07:53:14
Asia/Shanghai. Live properties confirm4500s/RAM8GiB/CPU200%/Tasks64/Nice10/
KillModecontrol-group. Its fail-on-error wrapper runs control then tracking
then the stricter held-out matrix, with2100/2100/240s child caps and15s grace.
Run root: `artifacts/training/safe-tracking-be487638-20261011` in native
`/home/converge/work/microduck_rl-upstream-20261010`. Both arms retain seed919,
256 worlds and1000 updates; only treatment tracking pressure differs.

Do not advance or alter the running native checkout. A documentation-only
follow-up on Mac does not change its pinned source. Full arm completion,
durable final999 hashes and the24-case held-out comparison are stillpending.
No automatically chained new skill, video, publication or physical motion.

### Completed range-retaining tracking comparison

The preceding launch snapshot is superseded. The service completed2026-10-11
08:14:21 Asia/Shanghai, Resultsuccess/ExecMainStatus0/MainPID0. Control took
574.331751s and tracking579.717843s; each completed1000 updates/24000 common
steps/6,144,000 transitions at seed919/256 worlds. CUDA-hidden verification
checked finite model/Adam, final999 iteration/common-step counters, exact parent
and result/checkpoint hashes, all1000 finite update rows per arm andzero NaNs.
Every standalone held-out case equals its aggregate record, both deterministic
decisions recompute exactly, and all paired differences recompute exactly.

Both arms passed12/12 motor and12/12 range checks. All24 cases wereupright,
240 steps, zero terminals andzero rated-speed exceedance; right-hip hard-stop
proximity exposure waszero in every case. Both still fail every moving case's
combined tracking gate; only their three idle cases pass the full criteria.
Thus neither checkpoint is an accepted locomotion foundation.

| Command vx/yaw | Control / tracking speed MAE | Control / tracking yaw MAE | Control / tracking torque p99 utilization |
| --- | ---: | ---: | ---: |
| 0 / 0 | .001253 / .001077 | .010984 / .011521 | .260426 / .218449 |
| .30 / 0 | .090479 / .114211 | .249249 / .182633 | .496784 / .475410 |
| .20 / -.20 | .054618 / .071267 | .252181 / .215522 | .468694 / .421626 |
| .20 / +.20 | .053086 / .069341 | .267843 / .208965 | .469650 / .426845 |

These are descriptive means, not replacements for per-world gates. The treatment
improves yaw MAE in each moving command group while increasing forward-speed
MAE in each group. Seed839's straight-case mean bodyvx is roughly.215 versus
.189m/s. Reduced turning error alongside slower forward response is consistent
with a gait trade-off; it does not prove why the policy slowed. The live
unclipped twist cost combines forward/lateral/yaw squared residuals with1m/s
and1rad/s references. Its last100 training mean decreased.626766→.447897,
so optimizing this aggregate does not establish balanced command tracking.

Training last100 falls were120/128; health peaks1710MiB/59C for both. Across
twelve held-out cases mean mechanical-power proxy1.093368/1.077399W and
thermal-load proxy.022703/.019899, not physical power/temperature guarantees.
Only small JSON records were mirrored to Mac
`artifacts/evaluations/safe-tracking-be487638-20261011`; models remain native.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `b867d289423836c838998a6b1e0dae660a0b8c8af1c8e2e2be1d7e13bd0d1646` |
| Tracking final999 | `503fb02f0bbddbd3fc886453b20a2c878a91a8c1c985eeddfeec768020c53b39` |
| Control result | `8fbbe6acebe04f25d9062c53ec290d1b70950e7ae7abe73a26da8d57ebcb4267` |
| Tracking result | `90b64285278ce6084a26aec2ae0780e5e24eace8b7224217c54b019c42d1459c` |
| Held-out aggregate | `8ea8d8373c4f419e5a0ef1e47e32ff55d13a8b1a7a29c1c9fc1531d345eb12a7` |

### Forward-specific dose-escalation predeclaration

Under the owner's renewed continuation request, test
`integrated-forward-retaining-turn-continuation-v1` in
`integrated_forward_tracking_refinement.py`. This is a bounded next experiment,
not automatic policy promotion or a new obstacle/hop skill. Both arms load the
exact tracking final999 above, preserving actor/critic/normalizers/Adam and
Python/Adam LR, fresh episodes and warm clocks0. The parent is exploratory only:
motor/range subsets passed, but its full foundation decision is rejected.

Both arms keep qpos-4/.15rad, motor-1, and body-twist-error-.5 fixed. Same plant,
commands, DR, frozen source curricula, unfiltered actions,61D actor/14 actions
and learner. Add the same positive forward-only cost term in both configs:
`(command_vx - root_link_lin_vel_b[:,0])**2`, referenced to1m/s, no clipping or
absolute-speed tax. It ignores lateral/yaw/vertical/roll/pitch. Raw nonfinite
residuals fail closed. Control weight0; treatment-.25 initially, -.75 strictly
after6000 common control steps, -1.5 strictly after12000. Each iteration collects
24 steps per world; these boundaries are not aggregate world transitions.

Matched seed937/256 worlds/1000 updates, save250, evaluate final999 only.
This isolates adding one escalation schedule, not individual dose efficacy.
Do not choose a best intermediate checkpoint or weaken per-world tracking,
motor, terminal or range/stop gates. The full unchanged evaluator remains
seeds839/853/857×four commands×eight worlds×240 steps, tracking startup60.
Check measured yaw/motor/range retention too; fixed shaping is not proof that
those capabilities will survive. No promotion if the full gate fails.

Before long training require both64-world5-update seed929 smokes, treatment
stages compressed to updates1/2, official normalized ONNX finite CPU61-to14,
and both256-world10-update seed931 benchmarks at full treatmentweight-1.5.
Use the existing360s preflight wrapper/60s CPU and GPU child caps and measured
1000-update projections below2040s. Only after exact clean pushed source and
successful receipts may one4500s sequential user service run control then
forward then evaluation with2100/2100/240s caps and15s grace. Keep GPU lease,
explicit verified DINO PID2021,6GiB aggregate/8GiBfree/<75C and all protected
scopesinactive. Preserve unrelated workloads and100.98/NX.

Every rollout captures raw forward residual after reward/before reset, even
for control's zero-weight term, and verifies actual weighted reward. Log fixed
motor/qpos/twist costs plus forward cost/weight extrema and exact step counts
at each weight. The focused Mac suite passed138 tests in7.34s, including raw
zero/match/under/overshoot/reverse, ignored axes, nonfinite failure, exact
parent/config parity, strict boundary behavior, and pre-reset weighted reward
checks after simulated state reset. Native/GPU readiness is stillpending.

Independent read-only Luna review found no harness or measurement blocker.
Its advancing-clock test suggestion was incorporated: a24-step update beginning
at6000 records1 old/23 new weight exposures; the12000 boundary similarly splits.
This fixes the schedule interpretation without changing it or selecting doses.
The final focused Mac suite passed146 tests in6.10s; whitespace checks passed.

### Forward-error preflight and retained launch

Exact recipe source `a09a9c0a065ed4c94b99b34322c4fa9fe8cc8896` was pushed
before the clean inactive native checkout fast-forwarded. Retained service
`microduck-forward-track-a09a9c0a-preflight.service` completed with
Resultsuccess/ExecMainStatus0/MainPID0. Native CUDA-hidden tests passed146 in8.78s.
Both smokes and benchmarks verified exact parent bytes, restored finite
model/normalizers/Adam, clocks0, exact saved iteration/common-step counts,
fixed motor-1/qpos-4/twist-.5, and live pre-reset forward weighted-cost equality.
All30 update rows werefinite withzero NaNs. Both official normalized ONNX
exports passed finite CPU61-to14 inference; treatment smoke's120 control steps
were25 at-.25,24 at-.75 and71 at-1.5, exactly matching the declared strict clock.

Ten-update256-world benchmarks took9.446374s control/9.503735s forward,
giving conservative1000-update projections1416.96/1425.56s below2040s.
Sampled total GPU health peaked1717MiB/60C. Randomized benchmark falls5/4
are not numerical admission. Receipt copies in Mac
`artifacts/evaluations/forward-tracking-a09a9c0a-20261011-preflight` agree
with native hashes; trained weights and full logs remain native.

| Receipt | SHA256 |
| --- | --- |
| Smoke control | `7c5822153f53836bce37b60ffc4d0824fcde065dfbd52fe1ad1b5c202bbdc036` |
| Smoke forward | `e20eb56135b3b44e23b0469581c155c48ed06347da2631aeee47e7071f7fa504` |
| Benchmark control | `69cf42d1b6440595f34482609c52c85722fab883d0db24e873cf49d2aa27144a` |
| Benchmark forward | `189e63abd69fdde19820a597caaffdb3e65ac3fbcb80785e1bfc30ac2c9d8e32` |

`microduck-forward-track-a09a9c0a-matched-eval.service` started2026-10-11
13:50:26 Asia/Shanghai. Live properties confirm4500s/RAM8GiB/CPU200%/Tasks64/
Nice10/KillModecontrol-group. It sequentially runs control then forward then
the unchanged24-case held-out comparison, with2100/2100/240s children and15s
grace. Native run root is `artifacts/training/forward-tracking-a09a9c0a-20261011`
in `/home/converge/work/microduck_rl-upstream-20261010`. Keep this running
checkout at sourcea09a9c0a; documentation-only follow-ups on Mac must not
advance it. Completion, final999 hashes and numerical comparison arepending.
Stop after retaining this matrix; no automatically chained new skill or motion.

### Completed forward-pressure comparison

The preceding pending snapshot is superseded. The retained service completed
2026-10-11 14:18:50 Beijing time, inactive/dead, Resultsuccess,
ExecMainStatus0/MainPID0. Control/forward took768.434917/799.505250s; each
completed1000 updates/24000 common steps/6,144,000 transitions at seed937,
256 worlds. CUDA-hidden verification checked exact source/parent bindings,
finite final model/Adam, final counters and checkpoint bytes, all1000 finite
update rows, zero NaNs, fixed shaping and exact staged forward-weight exposure.
All24 standalone case records match the aggregate, deterministic decisions and
paired differences recompute exactly. Models/logs remain native; small JSON
copies are retained on Mac at
`artifacts/evaluations/forward-tracking-a09a9c0a-20261011`.

Both arms passed12/12 motor and12/12 joint-range checks. Every case completed
240 steps upright, with zero terminals, rated-speed exceedance and right-hip
hard-stop proximity exposure. Both failed all nine moving combined-tracking
cases; only three idle cases passed. Neither checkpoint is an accepted policy.

| Command vx/yaw | Control / forward speed MAE | Control / forward yaw MAE | Control / forward torque p99 utilization |
| --- | ---: | ---: | ---: |
| 0 / 0 | .000826 / .000701 | .005513 / .006060 | .175904 / .228581 |
| .30 / 0 | .106726 / .101061 | .161155 / .179796 | .487335 / .490011 |
| .20 / -.20 | .061256 / .056040 | .181813 / .208943 | .467975 / .478754 |
| .20 / +.20 | .055769 / .053954 | .168967 / .208301 | .481793 / .490883 |

These descriptive means do not replace per-world gates. Forward-specific
pressure slightly reduced speed error but increased yaw error in every moving
group; do not promote the treatment. Signed mean-yaw bias is smaller than
instantaneous yaw MAE, consistent with variable response, but this is not a
frequency analysis or proof of a particular oscillation mechanism. Last100
training falls120/98 and forward cost.022081/.019292 are training diagnostics,
not stability acceptance. GPU peaks1717MiB/66C. Held-out mechanical-power
proxy means1.076198/1.045611W and thermal proxies.021024/.021405 are simulation
metrics, not hardware power or temperature guarantees.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `604700b0d50ee5f11558cf826a1c50fc4283bae801a80f7d6a8f90dff1a14132` |
| Forward final999 | `07263b054cc0b69667654fbd389d35f081d701f7aabb818c86b44d302e35ac37` |
| Control result | `ee1c1868b9ee992c5ff19b9a9298620f00dc5242e3ec5903a631cabc7b779b76` |
| Forward result | `b40511408a8c634620637cd0ad4a909e77e65918b034cec0eab32677c8e74c71` |
| Held-out aggregate | `8ff8a273854b42d66cc6707b598264e2286cd8cb309adf797947e1f41f32d4cb` |

### Gentle turning-command curriculum predeclaration

Under the owner's renewed continuation, test
`integrated-gentle-turn-range-curriculum-v1` in
`integrated_turn_range_curriculum.py`. Choose the preceding control final999
as the exact research parent because its moving yaw errors are lower; this
does not qualify it as an accepted locomotion foundation. Both arms restore
that exact model/normalizers/Adam and learning rate, fresh episodes/clocks0.

Only yaw-command sampling range differs. Control keeps±.50rad/s throughout;
treatment begins±.05, expands to±.10 strictly after6000 common control steps,
and±.20 strictly after12000. With24 steps/update and1000 updates, treatment
exposure is6001/6000/11999 action steps respectively. Mutate the actual command
manager's sampling cfg before action, not merely the source env cfg. Expansion
does not force resampling or rewrite current commands; ordinary3–5s resampling
continues. Initial range is set before env construction/reset. Log live range,
actual yaw-command maxima, per-update and cumulative stage counts, and fail
closed on unexpected range, reverse stage or nonfinite/out-of-bound commands.

Both arms retain motor-1, qpos-4/margin.15rad, twist-cost-.5, forward-cost0,
vx range(.2,.4), existing idle/straight proportions, source frozen curricula,
DR, plant, learner, unfiltered actions and61-to14 policy contract. Matched
seed953/256 worlds/1000 updates, save250, final999 only. This compares the
whole easier-to-harder curriculum, not independent stage efficacy; final±.20
does not establish capability across±.50.

The held-out protocol and gates remain unchanged: seeds839/853/857, commands
0/0, .30/0 and .20/±.20, eight worlds,240 steps, first60 omitted for tracking
only. Every world must satisfy speed MAE≤.10 (idle≤.03) and yaw MAE≤.10;
torque p99 utilization≤.60, zero terminals/rated-speed exceedance, all14 joint
range overruns≤.005rad, and right-yaw .05rad stop-proximity exposure≤1%.
No best-intermediate selection or gate relaxation.

Before long training require both64-world5-update seed947 smokes (curriculum
compressed to strict boundaries24/48, exposure25/24/71), official normalized
finite CPU ONNX61-to14 checks, and both256-world10-update seed949 benchmarks
(curriculum at final±.20). Require measured conservative1000-update projections
below2040s and exact clean pushed source. Use the retained360s preflight,
60s children; then only one4500s sequential user service with2100/2100/240s
control/curriculum/evaluation children and15s grace. Keep the owned GPU lease,
verified retained DINO PID2021, aggregate≤6GiB/free≥8GiB/<75C and all four
protected mission-service scopes inactive. Preserve100.98/NX and unrelated
workloads. Stop after retaining this matrix; no new skill/video/publication,
raw perception, physical motion or automatic promotion.

Focused Mac tests passed154 in12.32s (three existing actuator-selector
warnings), including matched config, exact parent, live-manager mutation,
unchanged current commands/rewards, strict boundary exposure and fail-closed
range/nonfinite checks. Whitespace checks passed. GPU preflight is pending.

Independent read-only Luna review approved the design for bounded preflight,
not policy acceptance. Its artifact-location concern was resolved without
duplicating models: the declared parent is intentionally absent from Mac but
exists in the native training checkout, with the exact604700b0… digest above
verified again. Native source is clean, previous service inactive/success,
GPU DINO-only969MiB/51C/free15225MiB, all protected scopes inactive.

### Turning-range preflight and retained launch

Exact source `2b75ab0272e6728af6b4124b8e8942b614d57cb5` was committed/pushed
before the clean inactive native checkout fast-forwarded. Retained service
`microduck-turn-range-2b75ab02-preflight.service` completed inactive/dead,
Resultsuccess/ExecMainStatus0/MainPID0. Native CUDA-hidden focused tests
passed154 in14.99s. Both smokes/benchmarks verified exact parent/checkpoint
bytes, finite restored models/normalizers/Adam, warm clocks0, final iteration
and common-step counters, all30 finite updates with zero NaNs, actual fixed
weighted rewards, manager-owned monotonic yaw range, observed commands within
live bounds, and exact per-update/cumulative stage exposure. Smoke exposure
was control120 at±.50; curriculum25/24/71 at±.05/±.10/±.20. Both official
normalized ONNX exports passed finite CPU61-to14 inference.

Ten-update256-world benchmarks took9.010299/8.973171s, projecting conservatively
to1351.54/1345.98s per1000 updates, below2040s. Total GPU sampled peaks were
1717MiB/60C. Benchmark training falls6/2 are not held-out acceptance. Small
receipt copies at Mac
`artifacts/evaluations/turn-range-2b75ab02-20261011-preflight` agree with
native hashes; no duplicate weights or environment were copied.

| Receipt | Verified SHA256 |
| --- | --- |
| Smoke control | `0cf3e5cc4a2c881866a97cd31287b2604af95ff667c8ca7e0e8c4811fd7bb2d2` |
| Smoke curriculum | `847a0419c3a3d5f2d3d4d7c355eeb6fe0e0f27d91228d8a6e44cd6e9c3df9987` |
| Benchmark control | `3ec64ad5d437e0c3caeec0d54528beb3c170fdbb7881a578e295f10744f53725` |
| Benchmark curriculum | `95c7fe349432dfbeedd7750a51cd0d7cca7827e826c97333b208bc49ffb8694b` |

`microduck-turn-range-2b75ab02-matched-eval.service` started2026-10-11
14:58:22 Beijing time. Live properties confirm4500s/RAM8GiB/CPU200%/Tasks64/
Nice10/KillModecontrol-group, sequential control then curriculum then unchanged
held-out evaluation, with2100/2100/240s children and15s grace. Native retained
root: `artifacts/training/turn-range-2b75ab02-20261011` in
`/home/converge/work/microduck_rl-upstream-20261010`.
Launch sampling confirms only DINO2021 and the owned Duck Python process,
1649MiB total/55C, protected services inactive in all four scopes, clean exact
source2b75ab02. Keep the active native checkout pinned; this documentation-only
follow-up does not advance it. Full training, final999 hashes and numerical
comparison are pending. No policy promotion or chained new skill/motion.

### Completed turning-range comparison

The preceding pending snapshot is superseded. The sequential service completed
2026-10-11 15:24:59 Beijing time, inactive/dead, Resultsuccess/ExecMainStatus0/
MainPID0. Control/curriculum completed1000 updates each in722.412282/748.999181s,
seed953/256 worlds/24000 common steps/6,144,000 transitions per arm. Verification
checked exact source/parent binding, finite models/normalizers/Adam and final
iteration999/common-step24000, durable0/250/500/750/999 checkpoints, all2000
finite update rows with zero NaNs, fixed shaping and command-range schedules.
Treatment exposure was exactly6001/6000/11999 steps at±.05/±.10/±.20; control
24000 at±.50. Standalone case JSON agrees with the aggregate, deterministic
decisions and all paired differences recompute exactly.

Both final checkpoints pass12/12 motor and12/12 range checks, complete all
240-step cases with zero terminals/rated-speed exceedance, and fail all nine
moving combined-tracking cases. No policy is promoted. Every moving world's
yaw MAE exceeds.10; only10/24 control and4/24 curriculum straight worlds pass
speed MAE≤.10. Every arc world passes the speed subset. Idle24/24 worlds pass
both tracking components per arm. These are world counts, not case counts.

| Command vx/yaw | Control / curriculum speed MAE | Control / curriculum yaw MAE | Control / curriculum torque p99 utilization |
| --- | ---: | ---: | ---: |
| 0 / 0 | .000914 / .000802 | .006869 / .007148 | .176927 / .227674 |
| .30 / 0 | .103473 / .110957 | .149547 / .174242 | .508729 / .466225 |
| .20 / -.20 | .049636 / .062895 | .174856 / .216164 | .492962 / .444585 |
| .20 / +.20 | .052796 / .053704 | .184389 / .177000 | .485657 / .461515 |

Descriptive group means are not relaxed acceptance gates. Curriculum lowers
motor load but worsens speed error in every moving group, yaw error in straight
and negative arcs, and only slightly improves positive-arc yaw error. Last100
training falls62/89 and command costs.316284/.336610 are not held-out stability.
Health peaks1717MiB/66C, and live post-completion GPU is DINO-only969MiB/48C;
all four protected service scopes remain inactive. Small JSONs retained on Mac
at `artifacts/evaluations/turn-range-2b75ab02-20261011`; weights/logs stay native.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `11c20ccc4d0682ab1403013dd5ca3308913ea2fd61a86e1c5bc3fb5408c2b8c2` |
| Curriculum final999 | `54c1a33894cd0e705d1d23d5e92fb06da015daf604c5d3887a6d68167574eb2f` |
| Control result | `f063545cecb9547f8312f79296f03ffe422184dd26c5e296c580acc4d1dbfff9` |
| Curriculum result | `525c3041ddd716dd82aff38330448537576059c6d27a254aa05f80cde3e790d4` |
| Held-out aggregate | `71bcfbe46ad8919123222067e2a26f9737967e187950838811efff8b08d5bf94` |

### Full-angular reward diagnosis predeclaration

The installed mjlab stock `track_angular_velocity` computes
`exp(-((command_yaw-body_wz)**2 + body_wx**2 + body_wy**2)/std**2)`.
It is not yaw-only. Std remains√.5, weight2, and the frozen action-rate weight
is-.8 (the-.1→-1 schedule's last stage strictly before36000). Tightening the
Gaussian would also tighten roll/pitch motion pressure, so do not do that
before measuring its existing response. Independent read-only Luna review
agrees that an eight-case diagnostic is preferable to another blind std change.

Retained pooled signed yaw responses control/curriculum are-.153149/-.057446
for-.20, and+.109768/+.200668 for+.20. Per-world mean absolute yaw bias is
.068489/.142554 negative and.091746/.047540 positive, below the corresponding
instantaneous MAEs. This indicates asymmetric under-response and within-window
variation; it does not establish oscillation frequency or a causal remedy.

Predeclare `integrated-angular-reward-audit-v1`: both exact rejected final999
checkpoints above, seed839 only, four unchanged commands,8 worlds/240 steps,
startup60. Retain the original full matrix and decisions unchanged. Add only
an opt-in diagnostic metric to the existing evaluator; do not modify rewards,
plant, actions, actor61D/14D contract, command timing or gate definitions.
Capture raw body angular XYZ, current commanded yaw and the actual unscaled
weighted angular reward after reward computation and before auto-reset; verify
that reward equals the known full XYZ Gaussian on every step, fail closed on
nonfinite states/mismatches/stale unconsumed snapshots. This diagnostic timing
is distinct from admission's pre-action response timing.

After startup summarize per-world RMS roll/pitch, yaw RMS/MAE/bias, actual full
reward, a same-state yaw-only counterfactual, roll/pitch multiplicative factor,
full-Gaussian<.1 fraction, yaw-only>.8/full<.1 fraction, and absolute yaw-gradient
magnitudes for full versus yaw-only forms. Counterfactuals describe these states,
not the behavior of a newly trained policy. No frequency or physical claim.
Preserve replay numerical differences at the existing1e-6 tolerance; a changed
replay cannot silently replace retained acceptance evidence or widen tolerance.

Require exact clean pushed audit source, original aggregate/checkpoint hashes
and identical frozen runtime, inactive prior service, owned regular GPU lease
and existing DINO2021/protected-service/memory/temperature guards. Run one
retained sequential diagnostic user service with240s hard cap/15s grace,
RAM8GiB/CPU200%/Tasks64/Nice10/KillModecontrol-group. No learning, automatic
next training, video, perception, publication or physical motion in this audit.
Retain output hashes and diagnosis before selecting a further experiment.

Focused Mac checks passed167 tests in6.55s, three existing actuator-selector
warnings; whitespace checks passed. Tests include actual upstream reward
callable parity with its omitted-asset robot default, raw nonfinite/mismatch
failure, stale-snapshot rejection, reset-independent capture, exact counterfactual
decomposition/gradient, short-terminal handling and fixed checkpoint bindings.
No GPU diagnostic outcome is claimed before the retained replay completes.
Independent code review found no diagnostic logic blocker. Run only on the
native checkout holding the exact artifacts; enforce240s wall-clock cap via
the user-service wrapper (240 simulation steps per case is not that wall cap).

### Completed full-angular reward diagnostic

Native CUDA-hidden focused suite passed167 in15.96s before launch at exact
source `1958523b27047288a4d3b1e1c9c65a2557c84759`. Retained service
`microduck-angular-reward-1958523b-audit.service` enforced240s wall cap,
RAM8GiB/CPU200%/Tasks64/Nice10/KillModecontrol-group and completed successfully
with MainPID0/ExecMainStatus0/inactive state at2026-10-11 16:03:41 Beijing
time. All eight seed839 cases completed
240 steps with finite pre-reset raw state and actual full XYZ weighted reward
verified every step. Health peaks1867MiB/60C. GPU returned to DINO-only969MiB,
all four protected service scopes inactive. Native/Mac small retained files:
`artifacts/evaluations/angular-reward-1958523b-20261011`.

| Command vx/yaw | Full angular reward control / curriculum | Same-state yaw-only reward control / curriculum | Mean roll/pitch factor control / curriculum |
| --- | ---: | ---: | ---: |
| 0 / 0 | 1.998969 / 1.999143 | 1.999629 / 1.999630 | .999670 / .999757 |
| .30 / 0 | 1.361723 / 1.357010 | 1.867330 / 1.834556 | .713189 / .708596 |
| .20 / -.20 | 1.334371 / 1.251998 | 1.840764 / 1.755389 | .709948 / .689130 |
| .20 / +.20 | 1.314581 / 1.357373 | 1.831684 / 1.835909 | .703706 / .718711 |

These are means of eight per-world180-step post-startup summaries, not paired
causal evidence. RMS roll/pitch rates were roughly.463–.524rad/s when moving.
Roll/pitch attenuate the reward by a material amount but do not make it dead:
full Gaussian<.1 occurs in2.57–7.22% of moving samples; the yaw-only>.8 yet
full<.1 fraction is only.07–.56%. Analytic absolute reward derivative with
respect to body yaw rate is roughly61–67% of its same-state yaw-only value.
This derivative is not a PPO/actor parameter gradient, learning signal proof,
or prediction that removing roll/pitch shaping improves a trained policy.
Do not conclude complete Gaussian saturation, or tighten all angular axes.

Replay is explicitly `replay_verified=false`:702 fields exceed the existing
1e-6 absolute comparison tolerance (547 response diagnostics,64 speed errors,
64 yaw errors,8 torque p99,8 thermal,7 power,4 soft-limit fractions). Maximum
absolute differences: speed MAE.005253m/s, yaw MAE.025454rad/s, torque p99
utilization.010144. All eight cases keep the same pass/fail outcomes: motor and
range pass, idle tracking passes, every moving case fails tracking. Checkpoint
bytes, runtime and original aggregate identity are exact. No retained decision
was overwritten and no tolerance changed. Source inspection confirms the
added metric only reads state/clones snapshots/returns zeros, without reward,
commands or RNG writes; this does not establish the cause of replay variation.
Treat this as a separate diagnostic trajectory, not an exact replay or new
qualification. The original full matrix remains authoritative.

| Artifact | Verified SHA256 |
| --- | --- |
| Audit launch | `5f66d07e67b7ae3ea5b7e76aabda09beff5c8fcb4521bb69b1d0e44d9c7dc3ae` |
| Audit result | `7bc55fd015aad0d597495e40b43e71ffbfdeb4d4f931f19370e0a4cba8961f17` |

Recommended next bounded experiment: separate yaw precision from roll/pitch
precision in the existing positive angular tracking Gaussian, leaving its
roll/pitch kernel, weight, plant, actions, motor/qpos shaping, source command
distribution and all admission gates fixed. Use a matched unchanged control
and one predeclared yaw-width schedule; verify exact control equivalence and
live pre-reset reward activity, then smoke/benchmark before any new long job.
This is a candidate design, not an implemented recipe or automatic launch.
Neither rejected final is accepted for a new skill or physical deployment.
Follow-up adds only an explicit derivative-kind metadata label, not numeric or
gate changes;46 focused tests passed6.51s and whitespace checks passed. The
retained1958523b replay and its hashes remain unchanged, not retroactively
claimed to have run the later metadata label.

### Predeclared independent yaw-precision continuation

Implement `integrated-independent-yaw-precision-v1` in
`mjlab_microduck.integrated_yaw_precision_refinement`. This replaces the
candidate-only proposal above with a bounded matched experiment, not a policy
promotion. Both arms warm-start the rejected turn-range control final999,
SHA256 `11c20ccc4d0682ab1403013dd5ca3308913ea2fd61a86e1c5bc3fb5408c2b8c2`.
Restore exact actor, critic, normalizers, Adam and learning rate; reset learning
iteration/common steps to zero and start fresh simulator episodes.

Both arms use positive angular reward
`2 * exp(-yaw_error_squared / yaw_variance - roll_pitch_rate_squared / .5)`.
The control keeps yaw variance `.5`, equivalent to the original full XYZ
Gaussian. Treatment changes only yaw variance at the following strict action
clock boundaries; roll/pitch variance `.5` and angular weight `2` stay fixed.

| Treatment common action steps | Yaw variance | Standard deviation rad/s | Exposure steps |
| --- | ---: | ---: | ---: |
| 0 through 6000 | .5 | .707107 | 6001 |
| 6001 through 12000 | .25 | .500000 | 6000 |
| 12001 through 23999 | .1 | .316228 | 11999 |

This tests one complete schedule, not the isolated efficacy of each dose.
Each arm uses seed977,256 environments,1000 updates of24 action steps,
checkpoint cadence250 and final999 only. Preserve the source full +/- .5rad/s
yaw command distribution, plant/domain randomization, unclipped actions,
61D actor/14D action contract, learner and fixed motor weight-1, qpos weight-4
with margin.15, twist cost-.5, forward cost0 and action-rate weight-.8.
No observation expansion, all-axis width tightening or roll/pitch removal.

Evaluation is the unchanged held-out matrix: seeds839/853/857, commands
vx/yaw 0/0,.3/0,.2/-.2,.2/+.2,8 environments,240 steps, startup60 omitted
only from tracking statistics. Every world must satisfy speed MAE<=.1m/s
(idle<=.03), yaw MAE<=.1rad/s; pooled torque p99 utilization<=.6,
zero rated-speed exceedance/terminals, all14 joint-range overruns<=.005rad
and right-hip-yaw near-limit exposure<=1% at .05rad margin. Retain all cases
and the deterministic unchanged decision; do not select intermediate winners,
relax gates or silently replace earlier evidence. A passing simulation matrix
would permit only the next separately reviewed simulation step, not deployment.

Mac focused CPU suite passed195 tests in10.30s with three existing
actuator-selector warnings; whitespace checks passed. Independent read-only
review found no blocker for preflight. Tests cover direct installed-upstream
control parity on1000 random vectors, raw NaN/Inf/overflow rejection, unchanged
configuration/learner, exact strict curriculum timing, and actual positive
weighted reward captured post-reward/pre-reset. Live observation must verify
every reward sample, consume snapshots, and retain per-update and cumulative
yaw-width exposure plus raw yaw/roll-pitch errors; missing/stale/mismatched
snapshots or unexpected live parameters fail closed.

Before either full arm: require exact clean pushed source and parent bytes,
native CUDA-hidden CPU tests, then paired64-world/5-update smokes seed971
with compressed treatment stage exposures25/24/71 and normalized ONNX
61-to14 checks. Paired256-world/10-update benchmarks seed973 use the final
width (.5 control/.1 treatment); each measured elapsed time times150 must
fit2040s. Retain all four preflight receipts and hashes. One preflight user
service enforces360s wall cap, each child60s plus15s termination grace.

Only after successful preflight launch one sequential retained user service:
control then precision then unchanged evaluation, child caps2100/2100/240s,
service wall cap4500s and15s grace. Use RAM8GiB/CPU200%/Tasks64/Nice10 and
KillMode control-group. Require the existing owned regular GPU lease, exact
GPU UUID and retained DINO PID2021/executable, GPU usage<=6144MiB,
free>=8192MiB, temperature<75C, and both protected AI Mission services
inactive in user and system scopes. Preserve DINO and unrelated workloads;
100.98/NX remains out of scope. No automatic new skill, video, perception,
policy publication, driver/package changes or physical motion. GPU preflight
and this round's training results remain pending at this predeclaration.

### Yaw-precision preflight passed; matched training launched

At exact source `4447fe64954675a7f38cdff38cfd288d4a93c0ae`, native
CUDA-hidden focused tests passed195 in13.87s (three existing warnings).
Additional Mac NaN-guard/observation/inference checks passed15 in6.70s.
The sequential preflight completed successfully and returned to DINO-only
969MiB. All four receipts have exact parent/source/runtime bindings, finite
models/Adam and update losses, correct step/transition budgets, zero NaN
terminations, actual live reward validation and exact yaw-stage exposure.
Both normalized ONNX CPU smokes passed61-to14 shape/finiteness/hash checks;
the existing automatic logger-export warning was not used as export evidence.
The separate retained normalized exports and inference receipts are verified.

| Preflight | Elapsed seconds | Conservative full-arm projection seconds | Verified result SHA256 |
| --- | ---: | ---: | --- |
| smoke-control | 5.134571 | not applicable | `305f08a76075a5c799fedf4e215b6e62fa9dbc18c839e8d90fc02aa528a564d0` |
| smoke-precision | 4.721343 | not applicable | `cdc082930721e89e84a4e777eab0635977f0af9196fa76f90257340e5453f997` |
| benchmark-control | 8.024094 | 1203.614088 | `0317726d21d8d66e5fa0a680741383a660c38592000f78c4f6259d0c5f0e038f` |
| benchmark-precision | 8.040501 | 1206.075142 | `a37fc673da1ea976f0c51eb496846a5e295167cb9f208aa2ad1d9c9c39d4da3e` |

Both projections fit2040s. Treatment smoke exposed widths.5/.25/.1 for
25/24/71 steps as predeclared. Smokes had zero falls; benchmarks had4/8
control/treatment falls, which are exploratory training observations, not
stability acceptance. Peak preflight health1717MiB/57C. The frozen native
runtime exactly matches the parent run: torch2.9.1, Warp1.12.0, MuJoCo3.10.0,
MuJoCo Warp3.8.1, mjlab1.3.0 and better-actuator-models1.0.1. Native models
stay in place; only four small JSON receipts were copied to Mac at
`artifacts/evaluations/yaw-precision-4447fe64-20261011-preflight`, with matching
hashes, avoiding duplicate checkpoint storage.

Launched `microduck-yaw-precision-4447fe64-matched-eval.service` at
2026-10-11 21:47:05 Beijing time. While active, verified MainPID755510,
RuntimeMaxSec4500, TimeoutStopSec15, RAM8GiB, CPU200%, Tasks64, Nice10 and
KillMode control-group. It runs control1000 then precision1000 then the
unchanged final999 held-out matrix sequentially; do not modify the running
native checkout. Retained run root is
`artifacts/training/yaw-precision-4447fe64-20261011` on100.100.

Initial live control snapshot reached update12/1000 with finite reward/losses,
zero cumulative NaN terminations,12 training falls, durable initial/model0
files, variance.5, unchanged motor/qpos/twist/forward weights and positive
actual angular reward. GPU1717MiB/55C with only Duck and retained DINO2021;
all four protected service scopes were inactive at launch. The exploratory
update12 right-hip range overrun was.013072rad: this is not a passing held-out
range result and must not be described as policy qualification. Both final
training receipts, full stage exposures and held-out decisions remain pending.
This follow-up is documentation only; training stays pinned to4447fe64.

### Control completed; precision treatment active

On2026-10-11, the pinned4447fe64 control finished1000 updates/24000 action
steps/6,144,000 transitions in605.665524s. All1000 update records are finite,
with zero NaN terminations, fixed motor/qpos/twist/forward/roll-pitch shaping
and exactly24000 steps at yaw variance.5. The last100 updates recorded64
training falls; this is not a held-out stability pass. CUDA-hidden checkpoint
inspection verified finite model/Adam states, exact saved iteration/common
steps and durable bytes for every predeclared checkpoint below.

| Control checkpoint | Verified SHA256 |
| --- | --- |
| 0 | `65f25a2186f94f15799f887f2e942402405d1a6a44c609bbb7a3ebf196fc7b99` |
| 250 | `46f605cf6c294caddfd18447dd603124eb8426115095508b23370cc827933fa2` |
| 500 | `8141be3907a7bf33b6f5f3aa4d78860cd17cfb52f08eee3dbc39257f36e7216c` |
| 750 | `6793e0c738319d2e1f0bd279083f108fbf5f95d4cbfbd6343c166edf370fc955` |
| 999 | `3e0b01bce31f8b73a535fa44bbfe92ce522908c03916269ad6ae203432b6249a` |

Control result SHA256:
`2357ab458a83d0ef86b625c426c6fcb5e5b6c467bd228a4e1d136cc7ba9400bc`.
Only its small receipt was copied to Mac at
`artifacts/evaluations/yaw-precision-4447fe64-20261011/control.json`, with the
same hash; checkpoints remain on100.100. Existing sequential service started
precision next, with no overlapping Duck worker or checkout/source changes.
Its first46 updates were finite/NaN-free with28 training falls, correct initial
variance.5 and durable model0. Health1717MiB/61C, retained DINO2021 preserved,
all four protected service scopes inactive. Treatment stage transitions,
final999 and both held-out decisions remain pending. Do not infer tracking
improvement or policy acceptance from successful training completion alone.

### Completed independent yaw-precision round: rejected

The retained4447fe64 sequential job completed both1000-update arms and the
unchanged24-case held-out matrix successfully. Verified every predeclared
checkpoint's finite model/Adam state, exact iteration/common steps, final
hashes, all2000 finite update records, source/runtime/parent identity and
deterministic decisions plus paired differences. Precision consumed exactly
6001/6000/11999 steps at variances.5/.25/.1; control consumed24000 at.5.
No NaN terminations. Both arms completed all12 first-episode cases (240 steps
each), with12/12 motor and12/12 range passes, but all9 moving tracking cases
failed;3 idle cases passed. Neither final is accepted or promoted.

| Command vx/yaw | Speed MAE control / precision | Yaw MAE control / precision | Speed-passing worlds control / precision | Yaw-passing worlds control / precision |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | .000661 / .000992 | .005764 / .008282 | 24 / 24 | 24 / 24 |
| .30 / 0 | .103182 / .088993 | .144889 / .175695 | 8 / 17 | 1 / 0 |
| .20 / -.20 | .047972 / .045664 | .170735 / .203119 | 24 / 24 | 0 / 0 |
| .20 / +.20 | .047543 / .039449 | .154009 / .166543 | 24 / 24 | 1 / 0 |

Means describe24 worlds across three seeds; admission still requires every
world to pass, not a group mean. Precision improves some speed measures but
worsens yaw MAE in every command group. Straight mean absolute per-world yaw
bias is.052144/.076836rad/s, negative-arc.097264/.128602, positive-arc
.051473/.049878. Mean bias is not instantaneous MAE: the remaining error also
contains within-window variation, without establishing oscillation frequency
or physical impossibility. Last100 training falls64/74; training mean yaw
squared error.260109/.273815 and roll/pitch squared rate.480017/.629578.
Do not repeat width tightening or attribute the difference to a proven cause.

Control/precision elapsed605.665524/609.786538s; health peaks1717MiB/61C
during training and2859MiB/59C during evaluation. Host returned to DINO-only
969MiB/47C; protected services remain inactive. Only small retained JSONs were
copied to Mac; models remain in the native run directory.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `3e0b01bce31f8b73a535fa44bbfe92ce522908c03916269ad6ae203432b6249a` |
| Precision final999 | `34cce22e654be94145f97c5a12dfc6e6a374be78c375b3d1d5c68e651d7952dd` |
| Control receipt | `2357ab458a83d0ef86b625c426c6fcb5e5b6c467bd228a4e1d136cc7ba9400bc` |
| Precision receipt | `fafc207916b6376e2354050f31d6742719495a27fc43c2f028e36c93593f6e6b` |
| Evaluation launch | `1ac127fb4b8809343bce432ef45d4ae866ed3a7124563a093a0260a326d3c96c` |
| Evaluation aggregate | `b273231f8a57734a5ea020e379bf0d8c6e977b720ce272cdefe2fdb89e8f98f2` |

Read-only optimizer diagnosis: each retained TensorBoard arm has1000 ordered
end-of-update LR values. Control/precision end at1e-5, minimum1e-5,
maximum7.59375e-5, means1.618125e-5/1.631812e-5, with723/726 floor-valued
updates. Mean action standard deviations.171065/.169839. These are update
endpoints, not every minibatch's LR. The inherited floor is not permanent:
installed rsl_rl_lib5.0.1 adapts at each minibatch, reducing LR when KL>.02
and increasing it when0<KL<.005 under desired_kl.01. No retained KL trace
exists, so neither inherited LR nor batch size is established as the cause.
Independent source review confirms the adaptive controller and warm-restore
path. Avoid a one-time LR reset that may immediately be undone by adaptation.

### Predeclared matched adaptive-KL-budget continuation

Implement `integrated-adaptive-kl-budget-v1` in
`mjlab_microduck.integrated_kl_budget_refinement`. Test one factor:
desired_kl.01 control versus.02 treatment (`klbudget`); retain adaptive
schedule, its1e-5 floor/1e-2 ceiling, five epochs/four minibatches (20 optimizer
steps/update), restored Adam moments and identical inherited starting LR.
This asks whether a less restrictive adaptive controller improves measured
tracking; it is not a proven diagnosis or permission to force larger updates.

Both arms warm-start the exact rejected yaw-control final999 SHA above,
restoring actor/critic/normalizers/Adam and restarting counters/episodes at0.
Keep yaw variance.5, roll/pitch variance.5 and angular weight2 fixed, together
with motor-1, right-hip qpos-4/margin.15, twist-.5, forward0, action-rate-.8,
commands, plant/DR, unfiltered/unclipped actions and61D/14D contract. Full
arms use seed1009,256 environments,1000 updates, cadence250/final999 only.
All held-out seeds, commands, timing, tracking/motor/range gates and first-
episode rules above remain unchanged. Do not promote an unqualified final.

Instrument the actor instance's existing bound KL function in both arms,
calling it once and returning its exact original tensor. Retain detached raw
min/mean/max and pre-adaptation LR without RNG, optimizer, reward or global
PPO-code writes; transfer summaries only after the update. Require reviewed
rsl_rl_lib5.0.1, exactly20 calls/update, fixed live target/schedule/epoch/batch
configuration, finite raw telemetry and matching algorithm/Adam group LR.
Retain per-minibatch mean KL and pre-LR arrays, threshold counts, raw extrema
and post-update LR; consume each update's samples and reject stale/missing
coverage. This is observation, not a new KL clamp or early-stop mechanism.
Starting LR is explicitly checked against the known parent's1e-5. Pre-LR
means before that minibatch's KL-controller branch, not its optimizer step;
applied LR is labeled inferred from the next pre-LR or final post-update LR
under the reviewed single-GPU PPO implementation. Independent review found
no blocker and prompted this starting-value/timing hardening.

Focused Mac suite passed214 tests in7.78s with three existing warnings;
whitespace checks passed. Tests cover unchanged matched configs, exact parent
binding, return-tensor/gradient/RNG preservation, strict threshold counts,
complete rollout/update integration, bad runtime/live config/Adam LR and
missing/extra/nonfinite telemetry rejection. Before training require a clean
reviewed pushed source, native CUDA-hidden suite, both64-world/5-update smokes
seed991 with normalized ONNX61-to14 inference, then both256-world/10-update
benchmarks seed997; elapsed times150 must each fit2040s. Reuse360s preflight
and4500s sequential training/evaluation service caps,2100/2100/240s child
caps,15s grace and all existing host/lease/DINO/protected-service guards.

Campaign authority ends2026-10-12 07:00 Beijing time. Check the absolute
deadline before launch and every sequential child; require its full remaining
cap plus grace to fit. At cutoff, start no new work and preserve durable
checkpoints; stop only a still-running campaign-owned Duck service if needed,
never DINO or unrelated jobs. Do not restore protected services without an
explicit request. No driver/package changes, new raw perception, publication
or physical motion. GPU preflight and KL-budget outcomes are pending.
After review hardening, the complete focused suite passed215 tests in8.12s,
with the same three existing warnings, and whitespace checks passed again.

### KL-budget native preflight and bounded launch — 2026-10-11 23:07 Beijing time

Source `a9a70f0e977037c706c7c108aaf868dbe8b4bca0` passed the same215-test
CUDA-hidden native suite in9.96s (three existing warnings). All four GPU
preflights completed; normalized ONNX CPU inference was finite61-to14 for
both smokes. Read-only verification checked source/parent hashes, restored
models/normalizers/Adam and reset counters, finite checkpoint payloads,
complete finite updates, fixed rewards and yaw variance `.5`, all20 KL
minibatches per update and every strict-threshold adaptive LR transition.

| Preflight | Elapsed s | Benchmark elapsed x150 s | Falls / NaN | Peak GPU MiB / C |
| --- | ---: | ---: | ---: | ---: |
| smoke-control | 4.738157 | — | 0 / 0 | 1511 / 53 |
| smoke-klbudget | 4.703245 | — | 0 / 0 | 1511 / 54 |
| benchmark-control | 7.718395 | 1157.759198 | 1 / 0 | 1717 / 57 |
| benchmark-klbudget | 7.957605 | 1193.640706 | 2 / 0 | 1717 / 57 |

Both projections fit2040s. These preflight falls/torques are diagnostics,
not held-out acceptance. Runtime pins match the declaration; the live
optimizer capture additionally requires `rsl-rl-lib==5.0.1`.
Small receipt copies in
`artifacts/evaluations/kl-budget-a9a70f0e-20261011-preflight/` match native
bytes (models remain native-only):

- smoke-control: `42aba9002a21ad5b1edeb8a8ba2f4e44b0066957f2d5a4ea4d659f2b9594fbca`
- smoke-klbudget: `e6672eeedc3ef6bef1a29fd512b8fee7dddb9f4a093bfe0d4e8c429eb106f137`
- benchmark-control: `3700c01cb611e7198c8f906463c9cd4ab9d23bed8c171e4b69849ae39af3f245`
- benchmark-klbudget: `ec41872bc71fa3688feadd872620e3cec8c9b3543540ba9be6374e82bca3d312`

The first wrapper attempt exited125 before Python or any full-arm output:
`timeout: invalid time interval 's'`. Read-only journals identified systemd
environment expansion of the shell's `${cap}` argument. Retry replaced
shell-variable-dependent child arguments and PYTHONPATH with literals; no
recipe, parent, runtime or output was changed/overwritten. The native
checkout is intentionally clean detached HEAD at the exact pushed source,
not a local branch; an earlier overstrict branch-name precheck was corrected
after read-only inspection, without switching that checkout.

`microduck-kl-budget-a9a70f0e-matched-eval.service` then launched successfully
at2026-10-11 23:06:31+08, MainPID863519: control1000 → klbudget1000 → unchanged
24-case final999 evaluation, retained under
`artifacts/training/kl-budget-a9a70f0e-20261011/`. Live service properties
confirmed4500s runtime,15s stop grace,8GiB RAM,200% CPU,64 tasks,Nice10 and
control-group kill mode. Absolute deadline1791759600 is checked before
launch and each child against the full remaining cap/grace budget; child
caps2100/2100/240s plus15s grace remain unchanged. Keep this native source
pinned while active; documentation advances on the Mac only.

Before launch, exact DINO PID2021 owned946MiB (aggregate969MiB, free15225MiB,
48C); the protected services were inactive in both user/system scopes.
Control update58 retained finite losses/rewards, zero NaN terminations and
`model_0.pt`; its KL target `.01` and20 optimizer minibatches were active.
Completion and unchanged held-out gates remain pending. No policy acceptance,
publication, physical motion or new skill completion is claimed.

### Retained CPU-only minibatch telemetry audit

`python -m mjlab_microduck.integrated_kl_budget_audit <updates.jsonl>
--expected-updates <N> --desired-kl <.01|.02>` reads retained bytes only,
prints a deterministic SHA-bound JSON report, and loads neither simulator nor
checkpoint. It verifies20 finite KL/pre-LR/inferred-applied-LR values per
ordered update, exact restored initial LR1e-5 and cross-update continuity,
strict adaptive transitions, true integer threshold counts and summary
consistency. It distinguishes applied-minibatch floor counts from
end-of-update floor counts; neither is policy acceptance. Float32 aggregate
mean reductions may round one ULP outside constant-array extrema, so only
logging summaries use an explicit numeric tolerance. Controller branches and
held-out gates retain exact original thresholds.

An independent bounded implementation review and owner inspection found this
float32 edge case; two owner regression tests cover it and ensure the logging
tolerance does not relax controller thresholds. The audit suite passed98
tests; the full focused suite passed313 tests in8.92s with the same three
existing warnings. Whitespace checks passed. CLI and API both audited all600
native preflight optimizer minibatches from hash-matched small JSONL copies:

| Preflight | Applied minibatches at LR floor | Updates ending at LR floor | High-KL branches |
| --- | ---: | ---: | ---: |
| smoke-control | 42 / 100 | 5 / 5 | 55 / 100 |
| smoke-klbudget | 24 / 100 | 3 / 5 | 30 / 100 |
| benchmark-control | 38 / 200 | 4 / 10 | 51 / 200 |
| benchmark-klbudget | 0 / 200 | 0 / 10 | 15 / 200 |

Preflight input SHA256 (native/Mac bytes equal):

- smoke-control: `7ca6ba281ce40f552f1d28427914c086914f467fc4123c5e5955ae63efc8d84f`
- smoke-klbudget: `27e3ee6223886cf170491a57cf5a9b558c7c2155b78b3752400b9ca0597125e1`
- benchmark-control: `258fe138778ee07a266a111b0b5479ce39f8ed93f30f93824ba2283cd0f9702c`
- benchmark-klbudget: `c98e26caf9c976a3902da97cff652761110e987fe30b9b52dc71b5abc61057b3`

This proves measured controller activity in short preflights, not improved
held-out tracking. Native training remains pinned at `a9a70f0e`; this offline
audit addition does not modify or reload that running service.

### Completed KL-budget round: rejected — 2026-10-11 23:30 Beijing time

Both1000-update arms and all24 held-out cases completed. Read-only checks
verified all ten finite model/Adam checkpoints at0/250/500/750/999, exact
source/runtime/parent/warm counters, all2000 finite update records and40,000
minibatch KL/LR transitions, retained case hashes, deterministic decisions
and paired differences. Both arms completed12/12 first-episode cases at240
steps with no terminals or rated-speed exceedance,12/12 motor and12/12 range
passes. Both fail all9 moving tracking cases;3 idle cases pass. Neither final
is accepted/promoted. Do not increase the KL budget further or select another
checkpoint after seeing this matrix.

| Command vx/yaw | Speed MAE control / klbudget | Yaw MAE control / klbudget | Speed-passing worlds control / klbudget | Yaw-passing worlds control / klbudget |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | .000597 / .000640 | .004987 / .006426 | 24 / 24 | 24 / 24 |
| .30 / 0 | .098683 / .110138 | .149003 / .184245 | 14 / 5 | 0 / 0 |
| .20 / -.20 | .046320 / .049198 | .168188 / .211608 | 24 / 24 | 0 / 0 |
| .20 / +.20 | .038612 / .045034 | .165566 / .194422 | 24 / 24 | 0 / 0 |

Each group summarizes24 worlds over three held-out seeds; group means are
not admission. Max pooled torque-p99 utilization across cases is.490732
control/.518420 treatment (gate.60). Larger KL changes the optimizer as
intended, but worsens yaw and speed MAE in every group in this matched round.

| Optimizer / training diagnostic | Control | klbudget |
| --- | ---: | ---: |
| Applied optimizer minibatches at LR floor | 5966 / 20000 (29.83%) | 612 / 20000 (3.06%) |
| Updates ending at LR floor | 641 / 1000 | 81 / 1000 |
| High-KL branches | 6023 / 20000 | 3306 / 20000 |
| Mean minibatch KL | .014930 | .027890 |
| Max applied LR | .000170859375 | .000576650391 |
| Last100 training falls | 67 | 171 |
| Last100 mean yaw squared error | .252722 | .488374 |
| Last100 mean roll/pitch squared rate | .458997 | .821942 |
| Last100 mean motor cost | .053406 | .073254 |
| Last100 mean power W | 2.351799 | 2.876324 |
| Last100 mean thermal proxy | .052698 | .071441 |
| Last100 soft-limit fraction | .006937 | .015429 |

This rejects the proposed larger budget, not all optimizer explanations or
the physical feasibility of turning. Control training was not permanently
stuck at the LR floor. Preserve desired_kl.01 for the next matched test.
Elapsed605.254399/619.383518s; training health1717MiB/61C, evaluation2789MiB/
59C. Service ended at23:28:53 (manager consumption journal retained), is
inactive/MainPID0, and all expected successful receipts exist. Host returned
to DINO-only969MiB/50C; all protected scopes remain inactive. Small hash-
matched JSON/JSONL copies are in
`artifacts/evaluations/kl-budget-a9a70f0e-20261011/`; no models copied to Mac.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `3ee173730a78fae605e561a3d40024b8e66d3b0a0be09262a6322fa819fe3df4` |
| klbudget final999 | `9bfff0f2dcf520b6266c07fd68d7b80beb5b91482ed993cf47236c852d6a6cdd` |
| Control receipt | `8ac3f6f9c382f10d019c48cb0e18ceb1b5adeceda5d6e7fd5c1ba0150aa981cb` |
| klbudget receipt | `d021623863ee1d7003e7eefb3964c961ccdc217d2b519c06de6c51cf7a3204f7` |
| Control update stream | `a99fb74b3acea7b1e4d76e6163f47d98cefe626b908ee3bebd728d93abcd5985` |
| klbudget update stream | `1172470161c07f919e6951fbf324b6f879dc0ee08d7692aac837f7e39a8eff38` |
| Evaluation launch | `b968e1f5d476c7ab79ac45f377c292c367d9e79094403219ceaa66d5dce34025` |
| Evaluation aggregate | `5b4f755e6be5eead82f4ce0290d59a13f0a8f246e3618d85ea7f0112c6fd2dad` |

Next bounded hypothesis, subject to implementation/tests/predeclaration:
compare the current angular reward `2ab` against `2(a+b-1)`, where
`a=exp(-yaw_sq/.5)` and `b=exp(-roll_pitch_sq/.5)`. Both equal2 at zero error
and exactly match when either error group is zero. The treatment removes
cross-axis attenuation of each analytic derivative but also deliberately
penalizes simultaneous errors more: treatment minus control is
`-2(1-a)(1-b)` and the treatment can be negative (range[-2,2]). This is one
reward-form change, not a pure gradient-only intervention or proof of
Gaussian saturation; preserve both kernel widths, weight, optimizer and all
other contracts/gates. Independent read-only review confirms these semantics
and requires treatment-aware pre-reset actual reward checks, not clipping
negative values or reusing the old positive-only observer unchanged.

### Predeclared matched angular-decoupling continuation

Implement `integrated-angular-decoupling-v1` in
`mjlab_microduck.integrated_angular_decoupling_refinement`, with
`mdp.decoupled_body_angular_tracking` as the treatment-only callable.
Control keeps the current product Gaussian `2ab`; treatment is
`2(a+b-1)` with the explicit signed semantics above. Both keep yaw and
roll/pitch variances.5 and term weight2; zero-error maximum and axis-error
curves at the other axis's zero error match exactly. Do not silently replace
the treatment with a normalized sum (which halves each axis's derivative),
remove roll/pitch shaping, clip its negatives, or call this a gradient-only
intervention. Reward-manager source inspection confirms signed weighted
values are summed rather than positively clipped; the term rejects invalid
raw state/exponents before the manager's NaN sanitization. Keep the same
finite combined-exponent validity domain as control.

Both arms use the exact unaccepted current KL-control final999 as initialization:
`artifacts/training/kl-budget-a9a70f0e-20261011/control/model_999.pt`, SHA
`3ee173730a78fae605e561a3d40024b8e66d3b0a0be09262a6322fa819fe3df4`.
Restore actor/critic/normalizers/Adam exactly, including initial LR1e-5;
restart counters/episodes at0. Using it as a shared parent is not policy
promotion. Preserve adaptive desired_kl.01, four minibatches/five epochs,
existing optimizer capture/audit, motor-1, named right-yaw qpos-4/margin.15,
twist-.5, forward0, action-rate-.8, plant/DR/commands and unfiltered/unclipped
61D actor/14D actions. Only the angular reward callable differs between arms.

Predeclared modes: paired64-world/5-update smokes seed1013; paired256-world/
10-update benchmarks seed1017; paired256-world/1000-update training seed1019,
cadence250/final999 only. Evaluate final999 from both arms on the same24
held-out cases (seeds839/853/857; four commands;8 worlds;240 steps; startup60),
with the unchanged every-world tracking, motor, named range and first-episode
gates. No post-hoc checkpoint selection, gate widening, skill/physical
acceptance, MP4 or publication from an unqualified outcome.

The observer must validate the actual pre-reset weighted angular reward
against the appropriate product/signed formula every step, including terminal
worlds; retain raw yaw and roll/pitch squared errors, signed reward mean,
negative-sample fraction, fixed kernel/weight identity and all20 optimizer
minibatches/update. Stale/missing/nonfinite/mismatched snapshots or learner
state fail closed. Positive-only checks from the old precision treatment
are deliberately not applied to the signed arm. Tests must establish matched
config/optimizer parity, formula and boundary/axis equivalence, raw failure
guards, real signed negative samples and24-step auto-reset-safe reward capture.

Before GPU work require reviewed tested/pushed clean source and the native
CUDA-hidden focused suite, both normalized ONNX61-to14 smokes and both measured
benchmarks; benchmark elapsed x150 must fit2040s. Retain360s preflight and
4500s sequential service caps;2100/2100/240s training/evaluation children,
15s grace, one owned GPU lease, literal arguments/PYTHONPATH (no systemd shell
variable expansion pitfall), exact DINO2021/GPU identity and all protected
service/occupancy/temperature guards. Require the full remaining caps/grace
budget before launch/every child to fit2026-10-12 07:00 Beijing time
(epoch1791759600). Preserve unrelated workloads. No raw perception, runtime/
driver/package change, policy publication or physical motion. Stop initiating
work at the cutoff; do not restore protected services without explicit request.
Implementation review and the Mac CUDA-hidden focused suite passed:342 tests,
including29 new reward-form/config/observer checks, with three existing actuator
selector warnings. No acceptance gate changed. Native preflight and the new
matched outcome are pending.

Native preflight completed at exact training source
`ea24a14a23e2ea74560ee7c4b69a4308d057af0a`:342 focused tests passed in14.45s
(three existing selector warnings). All four receipts, checkpoint hashes,
finite models/Adam, exact parent restore, actual pre-reset signed formulas,
fixed reward configuration and600 optimizer minibatches verified. Both
normalized ONNX exports are finite61-to14. Benchmarks took7.699s(control)
and7.734s(decoupled), giving conservative x150 projections1154.82/1160.10s
within2040s. Treatment negative-sample fractions were17.109%(smoke) and
17.762%(benchmark); control had none. Smokes had no falls, benchmarks0/1;
all had zero NaN terminations. These short training diagnostics are not
held-out motor/tracking acceptance. Peak host use1717MiB/57C; post-preflight
DINO-only969MiB/50C and protected services inactive.

| Preflight receipt | SHA256 |
| --- | --- |
| smoke-control | `ded2b2989a3be605df35e3ca5510250a5c45ecfbbfc6c63b569d7c5bac62157b` |
| smoke-decoupled | `47fceb5daa59505d8fdae4c1e71f6dcf620bc9ba7c3e5ab969d44840461729bf` |
| benchmark-control | `503a2dfc067ce806494dc0014bffe3c7256dfc36f2d871b3feefb101a4db7d7a` |
| benchmark-decoupled | `ada37c7e5609982aa850a1321adbad840d0e4761c9f402740c8b4e3895e1b6da` |

Launched one sequential retained service
`microduck-angular-decoupling-ea24a14a-matched-eval.service`, initialMainPID922489,
under `artifacts/training/angular-decoupling-ea24a14a-20261011`. Live properties
verified4500s runtime/15s stop/control-group kill/8GiB RAM/200% CPU/64 tasks/
nice10; exact DINO2021 retained. Control1000 then decoupled1000 then unchanged
24-case evaluation, with per-child deadline checks and2100/2100/240s caps.
Small preflight receipts/update streams copied under Mac
`artifacts/evaluations/angular-decoupling-ea24a14a-20261011-preflight`; models
remain native only. Full training/evaluation and numerical decision pending.
