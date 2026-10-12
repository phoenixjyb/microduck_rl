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

Read-only response review while training runs found no source-level body/world
frame or global yaw-sign mismatch: fixed body commands are checked against the
actor command slice and measured body root-link rates. Previous KL-control
turn means were-.1455/+.1390rad/s for commands-.2/+.2, with correct-sign
fractions.780/.777, not a global sign reversal. At straight.3m/s, pooled
mean yaw was-.002706rad/s, mean per-world absolute bias.029950rad/s, yet
mean yaw MAE.149003rad/s. The resulting mean Jensen gap.119053rad/s shows
substantial absolute residual beyond constant mean bias. It does not establish
oscillation frequency, tilt, contact or motor causality from aggregate data.
Gross motor-envelope saturation is not apparent in retained summaries;
body-z rate is not identical to world-vertical heading rate during tilt.
Do not silently substitute world-frame yaw or relax the existing body-frame
gate. A future diagnostic replay would need aligned orientation/rate/action/
target/contact trajectories to isolate those mechanisms; none launched here.

Added CPU-only `mjlab_microduck.body_response_audit`: validates complete
8-world/240-step retained cases, exact60/180 diagnostic window and body-frame/
pre-action timing, named joint layouts, finite non-boolean numeric summaries,
sign fractions and Jensen/triangle consistency. Reports per-world signed speed/
yaw bias, absolute-bias lower bound, original MAE and exact residual Jensen
gap, preserving tiny negative roundoff rather than clipping it. The1e-6
summary arithmetic tolerance is not an acceptance threshold. Hash-bound CLI
prints deterministic JSON without loading torch/mjlab or writing files;
diagnostic-only, no policy/physical claim. Owner review tightened sampling/
acceptance identity and actual tolerance-boundary tests. All24 retained
KL-budget held-out case files audited successfully. Focused suite now391
passed (49 new auditor tests), with the same three existing selector warnings.
Training source remains frozen at `ea24a14a`; these CPU-only source/doc changes
are not deployed into the active worktree.
Additional owner review added the complementary triangle bound
`MAE<=abs(command)+mean(abs(rate))`; with the existing reverse bound, zero-yaw
cases must reproduce `MAE=mean(abs(rate))` within descriptive roundoff. Three
new zero/negative/positive-command regression cases passed; suite394 passed,
and all24 actual retained cases passed the stronger audit. Wording clarifies
that summaries cannot reconstruct temporal ordering or oscillation frequency,
rather than denying that nonzero residual variation exists.

### Completed angular-decoupling comparison

The retained service finished2026-10-12 00:07:24 Beijing time, inactive/dead,
with both training receipts and all24 held-out cases complete. Training
source remained exact clean `ea24a14a`; control/decoupled took603.794740/
605.228847s, each1000 updates/24000 steps/6,144,000 transitions. Verification
checked all10 cadence/final checkpoints, finite model/Adam/counters, both
exact parent restores, all2000 finite update rows, zero NaN terminations,
actual signed reward/cumulative stage identities and40000 optimizer
minibatch transitions. Standalone cases, deterministic decisions and paired
differences recompute exactly. All34 small copied receipts/update streams/
case/aggregate files match native SHA256s under Mac
`artifacts/evaluations/angular-decoupling-ea24a14a-20261011`; no model copies.

Both arms pass12/12 motor and12/12 named joint-range checks. All24 cases
complete240 steps with no terminal worlds or rated-speed exceedance. Both
still fail all nine moving combined-tracking cases; only idle passes all
worlds. Neither policy is accepted. Decoupling improves moving mean yaw
errors and yields15/72 moving worlds meeting yaw MAE<=.1, versus0/72 control,
but this is not15 accepted cases or a substitute for every-world admission.

| Command vx/yaw | Control / decoupled speed MAE | Control / decoupled yaw MAE | Speed-passing worlds /24 | Yaw-passing worlds /24 |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | .000719 / .000918 | .005751 / .007679 | 24 /24 | 24 /24 |
| .30 /0 | .090774 / .095470 | .148610 / .117411 | 18 /19 | 0 /7 |
| .20 /-.20 | .037294 / .042796 | .162892 / .129828 | 24 /24 | 0 /3 |
| .20 /+.20 | .040516 / .042282 | .148766 / .129019 | 24 /24 | 0 /5 |

Means are descriptive only. Moving speed error slightly worsens in treatment,
while every turning group retains all24 speed-passing worlds. Maximum
case-level pooled torque p99 utilization.490184/.502690 remains below.60;
these fields are already dimensionless utilization, not Nm to divide again.
Body-response descriptive audit passes all24 cases. Straight-group yaw Jensen
gap decreases.099877->.080508rad/s; negative-turn gap.089321->.058972 and
positive.100769->.080638. This supports further yaw precision investigation,
not a proven oscillation/contact mechanism or physical acceptance.

Last100 training falls60/38, yaw-square means.228060/.176203, roll/pitch-rate
square means.423012/.327317, motor costs.051482/.049857, power proxies
2.297877/2.298346W, thermal proxies.050854/.049307 and soft-limit fractions
.006364/.005710. Signed reward negatives cover14.5026% of treatment samples.
Learning-rate floor utilization is6310/20000(31.55%) versus5837/20000(29.185%);
no optimizer reset or budget change occurred. Final Adam LRs are1e-5/
5.062500000000001e-05. Training peaks1717MiB/61C; evaluation2859MiB/59C;
post-round DINO-only969MiB/52C and all protected services inactive.

| Artifact | SHA256 |
| --- | --- |
| Control final999 | `0a7846baf444eab78c73f7024bfc1a0d4c62f758b4cfe7b0491b1983b2805b2c` |
| Decoupled final999 | `c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae` |
| Control receipt | `f29d5f573df0a516f231643a5f16118578f25f3c0af8ae689f8a083deb13e716` |
| Decoupled receipt | `958c158b7134d4582cacdfdec5b324e791e24bcbf25e8c2ae37f9ba5d0699584` |
| Control updates | `87bc0865d88a5c4e7f7931e4aed170ac3e6b9c235bbfa93423f110fa3c2b0ef3` |
| Decoupled updates | `20e300dd96d1cb4abcfa1421d9ecef7ffb867cd3e21eb86f3356b3e8c5438bfa` |
| Held-out aggregate | `265559a1f2769c911efc56e2e6c1a74e17be9f2521978448a03c6cd73e29e478` |
| Evaluation launch | `a65c6cb174e9fc78efe02cc1d62a9adf640e89979aaa5518fd51b02382ab90dc` |

### Predeclared independent-kernel yaw-precision continuation

Under the owner's continuation through2026-10-12 07:00 Beijing time, test
`integrated-decoupled-yaw-precision-v1` in
`mjlab_microduck.integrated_decoupled_yaw_precision_refinement`. Initialize
both arms from the preceding unaccepted decoupled final999/hash above, chosen
for lower moving yaw MAEs while retaining motor/range eligibility. This is a
shared research initialization, not policy promotion. Restore actor/critic/
normalizers/Adam exactly, including LR5.062500000000001e-05; restart episode/
iteration/common-step counters0. Do not silently reset LR to1e-5. The existing
capture/offline auditor will gain an explicit checked initial-LR argument,
default1e-5 for all older experiments; only this new parent supplies its exact
nonfloor value. PPO controller bounds, thresholds and targets stay unchanged.

Both arms use the signed decoupled angular reward, roll/pitch variance.5 and
weight2. Only yaw variance changes: control.5 throughout; precision.5 until
common step6000 inclusive, .25 from6001 through12000, and.1 from12001 onward.
Full exposure6001/6000/11999 actions. Smokes compress strict boundaries to
steps24/48 (exposures25/24/71); benchmarks use treatment final variance.1.
This tests the whole precision curriculum, not independent stage efficacy.
No product-kernel cross-axis attenuation is reintroduced, no negatives clipped,
no actor expansion or live command override. Preserve motor-1, right-yaw
qpos-4/margin.15, twist-.5, forward0, action-rate-.8, plant/DR/commands,
adaptive desired_kl.01/four minibatches/five epochs and61-to14 contract.

Paired smokes64 worlds/5 updates seed1021, benchmarks256/10 seed1023,
full256/1000 seed1027, cadence250/final999 only. Retain the same24 held-out
cases, startup window and every-world tracking/motor/named-range gates.
Before-action guards validate exact live callable/weight/previous variance,
monotone predeclared transitions and optimizer identity. Pre-reset snapshots
must match actual signed weighted formula at their current variance, including
terminal worlds; retain raw errors, negative counts/fraction and exact per-
update/cumulative variance exposures plus all20 optimizer minibatches.
Stale/missing/nonfinite/mismatched state fails closed.

Require reviewed tested/pushed clean source, native CUDA-hidden focused suite,
both normalized ONNX smokes and measured benchmarks before full training.
Preserve360s preflight/4500s sequential service caps,2100/2100/240s children,
15s grace, GPU lease, exact DINO2021 identity, protected-service inactivity
and occupancy/temperature guards. Verify each full remaining budget fits the
same07:00 cutoff before launch/every child. No publication, video, raw
perception, driver/package changes or physical motion. Owner review verified
the implementation and strengthened live schedule-identity/reverse-stage
guards plus120-action compressed signed pre-reset checks. Focused suite428
passed with three existing selector warnings, including explicit inherited-LR
capture/audit/CLI regressions and unchanged older default guards. Native
preflight and the bounded matched round are recorded below.

### Independent-kernel precision preflight and launch

Exact pushed source `ecdf2de4e08f1f67013ada69b8831c0c8bf651ad` passed the
native preflight: both five-update smokes and ten-update benchmarks completed,
with zero NaN terminations, finite model/Adam state, exact inherited parent LR,
all600 optimizer minibatch transitions and actual signed reward identities
verified. Both normalized ONNX smokes retain finite61-to14 outputs. Precision
smoke exposures are25/24/71 at yaw variances.5/.25/.1; benchmark precision
uses.1 for all240 actions. Control remains.5. These are execution checks,
not held-out motor/tracking admission: preflight pooled torque p99 exceeds.60.

The initial preflight wrapper selected347 CPU tests because a generation-time
substring replacement altered the intended test-list expansion. Read-only
journal/script inspection identified this selection issue. A separate exact-
source CUDA-hidden `microduck-decoupled-yaw-ecdf2de4-cpu-full.service` completed
all428 required tests, three existing selector warnings, in15.04s before full
training was allowed. No package or active-job changes were made.

| Mode | Measured seconds | Conservative full estimate seconds | Falls | Peak MiB / C | Receipt SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke-control | 4.698 | n/a | 0 | 1511 /53 | `e31424500ad260c98387f00b0a13c88c21ebf8f0db20666e5a4386214a5bf0e1` |
| smoke-precision | 4.681 | n/a | 0 | 1511 /54 | `445f0cb3b0d95a0a7a71c02e9e4f94a42d176659b58b18210d88ab16ef462095` |
| benchmark-control | 7.749 | 1162.296 | 1 | 1717 /56 | `90206da0d6ccd5d47c66d633c03fb41aef48d496390ccf5c529045db2043d002` |
| benchmark-precision | 7.791 | 1168.708 | 1 | 1717 /57 | `cd59e71606a26b4e76ded382e21a1465b5e413a4029e84ce57ab7ae1af856ae0` |

After exact clean-source/no-output/idle-host checks, launched retained
`microduck-decoupled-yaw-ecdf2de4-matched-eval.service`, initialMainPID983993,
under `artifacts/training/decoupled-yaw-ecdf2de4-20261012`. Verified4500s
runtime/15s grace/control-group kill/8GiB RAM/200% CPU/64 tasks/nice10 and
literal2100/2100/240s child caps with remaining-budget checks against07:00
Beijing. DINO2021 alone before launch:969MiB/48C, all protected user/system
services inactive. Early live control updates are finite; full numerical
decision remains pending. Native source stays frozen while this service runs.
All eight small preflight receipt/update files were copied under Mac
`artifacts/evaluations/decoupled-yaw-ecdf2de4-20261012-preflight`; every SHA256
matches native. Models remain native only. Documentation commits during the
run do not change its frozen training/evaluator source.

Prepared an optional diagnostic-only `body_yaw_trace` path without enabling it
in the active evaluator. It captures quaternion(wxyz), body/world XYZ angular
rates and both command representations at each pre-action boundary, cloning
live tensors before policy inference/step. A full mocked evaluator regression
requires all original case/response fields to remain exactly identical with
the option enabled. Default evaluation and policy/actor targets are unchanged.
CPU-only `mjlab_microduck.body_yaw_trace` checks the full240x8 trajectory,
held-out case/seed/hash, exact actor command, original1e-6 fixed-command check,
unit quaternions and rotation identity(1e-5), and reconstructs original settled
body-yaw MAE within1e-6 descriptive arithmetic tolerance. It reports bias,
standard deviation, residual sign transitions/lag1 covariance, world-vertical
rate and the instantaneous projected-body-x heading rate. Near-vertical
projection is explicitly undefined, never dropped from the body-yaw gate.
No frequency or contact/motor causal claim is inferred. CLI is read-only and
hash-bound; no tensor/simulator import. Independent bounded review found no
static frame/sign mismatch and recommended this trace if the current precision
round remains rejected, rather than guessing another damping coefficient.
Owner-reviewed integration and focused suite491 passed, with the same three
existing actuator-selector warnings. This is source/test evidence only; no
new diagnostic replay or admission claim has occurred yet.

Read-only pre-transition comparison identified a reproducibility limitation:
both initial actor/critic(including normalizers)/Adam trees compare exactly,
both restore the pinned parent/LR and use seed1027, but rollout summaries and
KL branches already differ while both yaw variances are.5. First-update raw
yaw-square means.156460208/.156718458 and roll/pitch-square means.253633749/
.254249032 are close, whereas adaptive LR ends5.0625e-5/2.25e-5. Thus
matched seed does not imply identical stochastic trajectories or isolate
observed final deltas from all execution variability. The frozen mjlab RNG
source explicitly notes MuJoCo Warp nondeterminism; [official MJWarp
FAQ](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html#frequently-asked-questions)
confirms GPU atomic-order/numerical variability. This is consistent with the
observed divergence, not proof of its unique cause. Preserve the completed
experiment and every-world gates, do not change drivers/runtime/determinism
mid-run, and regard single-pair reward deltas as preliminary descriptive
evidence requiring independent training-seed confirmation before promotion.
The capture integration regression additionally feeds a complete traced mock
case through the CPU auditor and proves that an episode ending at step61
retains61 raw samples but is rejected as incomplete, rather than hiding a
terminal or reconstructing a full trace. The focused suite remains491 passed.

### Completed independent-kernel precision comparison

The sequential service completed successfully, inactive/dead/MainPID0,
ExecMainStatus0, by2026-10-12 00:47 Beijing time. Exact clean native source
remained `ecdf2de4`; control/precision took598.570225/601.782753s, each1000
updates/256 worlds/seed1027/24000 steps/6,144,000 transitions. Verified all10
cadence/final checkpoints, finite model/Adam/counter state, exact parent/LR
restores, all2000 finite update rows with zero NaN terminations, all40000
optimizer minibatch transitions, signed reward and full variance exposures
(control24000 at.5; precision6001/6000/11999 at.5/.25/.1). Recomputed every
case/decision/paired difference. All34 small copied receipt/update/case files
match native SHA256s under Mac
`artifacts/evaluations/decoupled-yaw-ecdf2de4-20261012`; no model copies.

Both arms pass all12 motor and12 named-range checks, with every case240
steps/no terminal worlds/no rated-speed exceedance. Both still fail all nine
moving combined-tracking cases; only idle passes all worlds. Precision does
not improve moving behavior uniformly and is not accepted or selected for
promotion. Moving yaw-passing worlds14/72 control versus13/72 precision.

| Command vx/yaw | Control / precision speed MAE | Control / precision yaw MAE | Speed-passing worlds /24 | Yaw-passing worlds /24 |
| --- | ---: | ---: | ---: | ---: |
| 0 /0 | .001049 / .000796 | .008208 / .007783 | 24 /24 | 24 /24 |
| .30 /0 | .099392 / .104653 | .115942 / .122219 | 14 /6 | 7 /3 |
| .20 /-.20 | .041286 / .041189 | .132032 / .142140 | 24 /24 | 1 /1 |
| .20 /+.20 | .043022 / .045378 | .126160 / .116104 | 24 /24 | 6 /9 |

Means/world-pass counts are descriptive, not case admission. Maximum pooled
torque p99 utilization.540153/.507228 remains below.60. All24 body-response
audits pass. Straight yaw Jensen gap.084780->.057720 accompanies greater
absolute mean bias.031162->.064499rad/s; negative-turn gap.063316->.037239
likewise accompanies bias.068716->.104901. Reduced residual gap alone is
not better tracking. Positive-turn gap.096619->.076139 improves while bias
increases.029541->.039965. A retained time trace is needed before attributing
these effects to frame projection, temporal variation or actuator behavior.

Last100 training falls39/35, yaw-square.158123/.146701, roll/pitch-square
.291268/.284844, motor cost.048018/.047104, power proxy2.200325/2.091289W,
thermal proxy.047544/.046661 and soft-limit fraction.005112/.004784.
Negative signed reward samples10.6547%/23.0388%. Applied LR floors6502/20000
(32.51%)/6206/20000(31.03%); unchanged adaptive controller, both finalLR1e-5.
Training peaks1717MiB/62C; evaluation2859MiB/59C. Post-round DINO2021 only,
969MiB/51C; protected user/system services inactive. No new training selected
until the bounded orientation/rate diagnosis is reviewed.

| Artifact | SHA256 |
| --- | --- |
| Control final999 | `04665a21cd131364a188d965e719ccde4bfdd643f745b8fef8b8828982104899` |
| Precision final999 | `52845be16d58461df2b751ae7ff25bc497a341e4a6fd07064a88dc1805fb51a9` |
| Control receipt | `ff20bacdb1d5993a603f1fce4feff2ccf970b1a625497ca2c8a48d28fd8005bc` |
| Precision receipt | `b7ce808bc3eb8de4554573ce391e8e218371e8f3de3884a7ca7fce9db9398c36` |
| Control updates | `7905cc2a3f9a291b127d05c3df8642c05b3072f4a024d20383234b775148c8ed` |
| Precision updates | `798cf4839eb897cac663220afa64184cf45863df8db00a33ede6c24d4ad81056` |
| Held-out aggregate | `2e9b919a94275b791f52d64ccf7d3849f51655e124f195731664ff21a886cbad` |
| Evaluation launch | `9fdebf754be63eace50ab52724001355f28fa16cb4ba17a1997f713a7c4d391b` |

### Predeclared eight-case orientation/rate replay

Only after the completed round above, reviewed/pushed clean trace source,
CUDA-hidden native focused tests and an idle shared GPU, run one bounded
diagnostic service against exactly the retained aggregate/hash and both
final999 hashes above. Declare seed839 and all four original held-out
commands for each arm: eight cases, not a new full admission matrix. Capture
the optional read-only full240x8 trace, audit quaternion/frame/command/MAE
identity, and retain raw bytes before any summary or replay comparison.
Compare all original case fields with the retained case under the existing
1e-6 replay comparison; preserve every discrepancy and set replay_verified
false if any. Never overwrite old gate evidence or turn an approximate/new
trajectory into an exact-replay claim. Preserve runtime/parent/receipt/source
identities, owned GPU lease, exact DINO2021/protected-service guards,20%
allocator cap and240s runtime/15s grace, checked against07:00 Beijing. No
learning, actor change, video, raw perception, driver/package change or
physical motion. Outcome remains diagnostic-only and no policy promotion.

The replay implementation pins the full aggregate and both final checkpoint
hashes above, revalidates training/evaluation source and runtime receipts,
and recomputes the original decisions. Owner review added exact per-arm,
seed/command/checkpoint identity checks after raw retention and compares the
whole observed case rather than projecting away unexpected result fields.
Focused CPU checks pass523 tests (three existing selector warnings); the
copied actual aggregate/receipts/decisions also validate without simulation.
Native CUDA-hidden tests and the bounded live replay remain required.

### Completed eight-case orientation/rate replay

Native CUDA-hidden tests passed523/three existing warnings in10.46s. The
four-minute retained service `microduck-yaw-trace-4430bd67.service` completed
successfully, inactive/dead/MainPID0/ExecMainStatus0 at clean frozen source
`4430bd679ed065fca785d011a1bc2f9c0e57c6b8`. Retained output:
`artifacts/evaluations/yaw-trace-4430bd67-20261012`. All eight cases complete
240 steps without terminal worlds or rated-speed exceedance. All frame,
quaternion, actor-command and body-MAE arithmetic audits pass; original
aggregate/checkpoint hashes and deterministic rejected decisions unchanged.
Raw and summary hashes rechecked; summaries recompute exactly on native.
Copied16 raw/summary hashes also match. Mac arithmetic recomputation differs
in one float by1.39e-17 (not byte corruption or a changed admission threshold).

All eight replays differ from the original trajectories, with73/87/93/93
control and74/84/95/95 precision differing fields under the existing1e-6
comparison. Accordingly `replay_verified=false`; this seed839 diagnostic is
not an exact replay or a replacement12-case admission matrix.

| Command vx/yaw | Control / precision body yaw MAE | Signed bias | Residual standard deviation | Mean nonzero residual sign transitions /180 samples |
| --- | ---: | ---: | ---: | ---: |
| 0 /0 | .007088 / .007567 | -.000717 / .002803 | .009627 / .010582 | 86.50 /83.25 |
| .30 /0 | .111041 / .117648 | .017123 / .063165 | .149811 / .144085 | 54.88 /56.38 |
| .20 /-.20 | .131814 / .136990 | .074931 / .109117 | .147883 / .132172 | 40.38 /34.00 |
| .20 /+.20 | .121432 / .108532 | -.018198 / .022625 | .156683 / .130571 | 53.38 /53.13 |

These are means of eight per-world descriptive quantities. All heading
samples are defined. Largest per-world body-versus-heading MAE difference
is.001097rad/s; frame substitution does not explain the roughly.11-.14
moving body-yaw MAEs. Both temporal variation and signed bias remain.
Sign transitions are not a frequency-spectrum or actuator-causality claim.
Tighter Gaussian width has not uniformly improved tracking; do not tighten
again, widen gates or change coordinate frames based on this trace.

Maximum pooled torque utilization.526521 control/.496713 precision, below.60.
All18 health samples satisfy lease/occupancy/protected-service guards; peak
2003MiB/57C, post-run DINO2021 only969MiB/49C. No learning or policy promotion.

| Artifact | SHA256 |
| --- | --- |
| Trace launch | `b2785efba94e9d20bc740ce2d7e315c24c7a9f1ee00a4293491ad5fceb0f634a` |
| Trace result | `e4686f8e1229dbe8235f97554557eed927887353a23a4bc4bec1f6d81051c175` |
| Trace health | `bc0559ee0f00c658d062fe516aa6bbbdaa5b06dad9a18a19937a1a28b9a8916c` |

### Predeclared persistent yaw-bias continuation

The trace does not justify a frame correction or a motion-wide smoothness
increase. Test one reward-only hypothesis instead: persistent signed yaw-rate
bias is an escapable part of the remaining moving error, and modest pressure
on its one-second average may reduce it without taxing every gait fluctuation.
This is not a proven cause or a promised gate pass. Existing head-pose bias
shaping provides the repository pattern, not evidence of a yaw benefit.

Protocol `integrated-body-yaw-bias-continuation-v1`, arms control/bias. Both
fork the exact decoupled parent `c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae`
at `angular-decoupling-ea24a14a-20261011/decoupled/model_999.pt`, restoring
actor/critic/normalizers/Adam and initial LR5.062500000000001e-5 exactly.
Keep signed additive yaw/roll-pitch Gaussians at variance.5/.5 and weight2;
do not retain the rejected precision schedule. Keep command-error-.5,
forward-error0, motor-1, named right-hip-yaw qpos-limit-4/margin.15 and
action-rate-.8, all plant/DR/command distributions, adaptive KL.01/four
minibatches/five epochs,61D actor and14D unclipped/unfiltered actions.

Only treatment adds weight-.1 on positive `abs(EMA(command_wz - body_wz))`,
referenced to1rad/s; tau1s, alpha=min(1,step_dt/tau)=.02 at50Hz. Control has
weight0 and computes the same diagnostic EMA in the post-reward metric.
No actor/observation/action filtering. Clear each world's accumulated EMA
on fresh episodes or changed yaw commands; clear all on warm counter reset.
Repeated reward/metric calls at the same control step advance once, not twice.
Raw finite-state checks and cloned pre-reset snapshots include terminal worlds;
validate actual nonpositive weighted activity before reward sanitization can
hide errors. Retain per-update EMA absolute cost, signed EMA mean, raw residual
MAE and actual reward mass alongside unchanged angular/optimizer/motor telemetry.
At sustained0.1rad/s bias the asymptotic unscaled reward rate is-.01, compared
with-.005 for the current command-error yaw-square component. This small
coefficient deliberately tests bias pressure, not a new instantaneous penalty
or dose escalation. No widening of the instantaneous body-yaw gate.

Same-source smoke64 worlds/five updates/seed1031 in both arms with official
normalized61->14 ONNX CPU checks, then256-world/ten-update/seed1033 benchmarks.
Verify both completed finite preflights, exact warm-state/LR, same-step EMA
identity, actual reward activity and measured budget before full training.
Full matched256 worlds/1000 updates/seed1039 per arm,24 steps/update,
6,144,000 transitions, checkpoint0/250/500/750/999. Control then bias then
unchanged final999-only12-case held-out evaluator for each: seeds839/853/857,
eight worlds,240 steps/startup60, commands0/0,.30/0,.20/-.20,.20/+.20.
All prior tracking/motor/named-range gates remain exact. Compare all24 cases
and descriptive signed response; a bias reduction alone is not acceptance,
and one matched seed pair is not statistical or bitwise causal proof.

Only reviewed/tested/pushed clean source may run. One retained sequential user
service, existing lease/DINO2021/protected-service guards,20% allocator;
2100s cap per training arm,240s evaluation,4500s service/15s grace and explicit
07:00 Beijing budget checks. Stop on a true execution/finite-state failure,
diagnose read-only before changes. No rejected-policy video, obstacle/physical
promotion, driver/package changes or unrelated workload modification.

### Yaw-bias preflight and retained matched launch

Reviewed source `2f5ffa1ff8f19b1f733a0e07399daec938ba70a9` pushed and fast-forwarded
onto the idle clean native detached checkout. CPU checks pass570/three existing
selector warnings locally and on native (12.12s), followed by all four GPU
preflights in `microduck-yaw-bias-2f5ffa1f-preflight.service`, inactive/dead/
MainPID0/ExecMainStatus0/Resultsuccess. Verified exact parent bytes, all four
initial model/normalizer/Adam trees, restored LR/counters, cadence/final finite
checkpoints, receipt/source/runtime identities and all600 minibatch adaptive
KL transitions. Both normalized ONNX CPU checks remain61->14. No NaN ends.

| Mode | Seconds | Falls across updates | Mean EMA cost | Actual weighted bias reward rate | Mean signed angular reward rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke-control | 4.588428 | 0 | .049113 | 0 | .978067 |
| smoke-bias | 4.594369 | 0 | .049049 | -.004905 | .975073 |
| benchmark-control | 7.878100 | 4 | .055730 | 0 | .978805 |
| benchmark-bias | 7.972032 | 2 | .055577 | -.005558 | .981206 |

All live weighted-bias snapshots agree with-.1*positive cost or exact0 in
control; no same-step double EMA update. This small reward is active, not a
dead configured term. Benchmarks yield conservative150x estimates1181.715/
1195.805s, both below the2040s budget test. Preflight motor peaks around.65
are execution/gross-abort checks, not a held-out motor-gate pass. Peak1717MiB/
58C; post-run DINO2021 only969MiB/49C, protected user/system services inactive.
Sixteen small copied launch/result/warm-start/update files match native hashes
under `artifacts/evaluations/yaw-bias-2f5ffa1f-20261012-preflight`; models retained
only on native.

Matched sequential service `microduck-yaw-bias-2f5ffa1f-matched-eval.service`
launched by2026-10-12 01:10 Beijing time, MainPID1059703, exact clean source
`2f5ffa1f`, unique `artifacts/training/yaw-bias-2f5ffa1f-20261012`. Frozen source
stays unchanged while active. Literal2100/2100/240s child caps,4500s service,
15s grace/control-group kill,8GiB memory/CPU200%/tasks64/nice10; explicit07:00
deadline budget checked before each phase. Control1000 then bias1000 then the
unchanged24-case final999 evaluation. Training progress is not acceptance.

| Preflight receipt | SHA256 |
| --- | --- |
| smoke-control | `c1b99407efc2afa69611302d57cc877b8e27278eb54bf50ba8b85b0db7003a93` |
| smoke-bias | `0442d9edbdd71fc9d51022cbd5d13799e9fe7e8784631b1741abfe3a5f545c73` |
| benchmark-control | `e99960e2d0a2c2f8d9b7bcf216ea311f6834171f3374c08cc3bf3515e7adf3db` |
| benchmark-bias | `b70229e8a12b40ff3ee3b07916e790e49f330b959ff1367aef6dd9f83e59584f` |

### Completed persistent yaw-bias comparison: rejected

The retained sequential service ended successfully at2026-10-12 01:32:41
Beijing time, inactive/dead/MainPID0/ExecMainStatus0. Exact clean native source
remained `2f5ffa1ff8f19b1f733a0e07399daec938ba70a9`. Both1000-update arms
completed24,000 control steps/6,144,000 transitions, all five declared finite
cadence/final checkpoints and20,000 audited adaptive-KL minibatches each.
Verified identical initial model/normalizer/Adam trees, parent hash, restored
LR and fresh counters, source/runtime/receipt identities, all per-step reward
identities and full24-case result recomputation. No NaN terminations.

| Training diagnostic | Control | Bias |
| --- | ---: | ---: |
| Elapsed seconds | 618.730932 | 627.531285 |
| Falls across1000 updates | 402 | 430 |
| Mean EMA absolute yaw cost | .059251 | .059178 |
| Actual weighted bias reward rate | 0 | -.005918 |
| Last100 mean yaw squared error | .166026 | .165967 |
| Last100 mean raw yaw MAE | .296762 | .301534 |
| Last100 mean motor cost | .048960 | .048819 |
| Last100 thermal proxy | .048407 | .048262 |
| Last100 soft-limit fraction | .005542 | .005672 |
| Last100 right-yaw stop exposure | .000397 | .001025 |

The new reward was active and its observed value matched-.1*positive EMA
cost; this is not a dead-term explanation. It did not materially reduce its
training cost, and the final held-out yaw MAE increased in every moving group.
No dose escalation is justified by this one pair. GPU nondeterminism and a
single training seed limit causal conclusions; retain both outcomes, not a
best-seed or best-intermediate selection.

| Command vx/yaw | Speed MAE control / bias | Body-yaw MAE control / bias | Mean per-world absolute yaw bias control / bias | Speed-passing worlds control / bias | Yaw-passing worlds control / bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 /0 | .000841 / .001163 | .006914 / .009406 | .002582 / .002433 | 24 /24 | 24 /24 |
| .30 /0 | .097642 / .090163 | .119641 / .137756 | .031260 / .039137 | 15 /20 | 3 /0 |
| .20 /-.20 | .040053 / .037032 | .128165 / .139130 | .063148 / .052437 | 24 /24 | 2 /0 |
| .20 /+.20 | .045068 / .036966 | .133985 / .137649 | .034196 / .029022 | 24 /24 | 1 /0 |

Each group summarizes24 worlds from three held-out seeds; group means do not
replace every-world admission. Both arms fail all nine moving combined
tracking cases; three idle cases pass. Moving yaw-passing worlds are6/72
control and0/72 bias. Both pass12/12 motor and12/12 named-range checks, with
maximum pooled torque-p99 utilization.508255/.534881 (gate.60). All24 cases
complete240 steps without terminals or rated-speed exceedance. Neither final
is accepted, simulator-qualified, promoted, or authorized for physical motion.
Smaller turning signed-bias magnitudes do not compensate for worse
instantaneous yaw MAE; no filtered/heading-rate gate substitution.

Training peaks1717MiB/61C(control)/62C(bias); evaluation2859MiB/59C.
All retained health samples satisfy the existing occupancy/lease/service
guards. Post-run DINO2021 alone969MiB/49C, all four protected service scopes
inactive. Thirty-four small JSON/JSONL copies, preserving control/bias/evaluate
subdirectories, match native SHA256 values under
`artifacts/evaluations/yaw-bias-2f5ffa1f-20261012`; copied manifest retained.
Checkpoints remain durable on native only. Duplicate temporary flat copies
were removed after verifying all structured copies; no unique evidence lost.

| Artifact | Verified SHA256 |
| --- | --- |
| Control final999 | `c00276990c424086cd176dec9fb0337d57b3a2355a6a65811f1229c10ba534c0` |
| Bias final999 | `5c64d1398d7d27d886ecffbea50b6f069e6de5b9773c26f590fdf9050a67af79` |
| Control receipt | `7211a893dca0f3f7070445950fb3c9e75c09f9602a4c7a10e99d27aef9050ffc` |
| Bias receipt | `d18a9c7fc21fc77f9d4ab9683712538631c8c8e0e5e3db2c4f6e13b3b2016ffc` |
| Evaluation launch | `562e17721540105a19dafdf5b1f954f10025410c652c16b88fb858e45bf05790` |
| Evaluation aggregate | `c8f54974f24e1a4e15a9ffda5943cd253fda907bcdca18e1164577b1905ded38` |

Next development must address an evidence-backed distinct lever, with a new
predeclaration and focused tests before GPU work. Current effective training
yaw range is±.5, not the wider athletics template default; a sampling-density
hypothesis requires actual command-exposure evidence, and the previously
rejected gentle-range curriculum must not be overlooked.

### Predeclared fixed-recipe training-duration comparison

After failed width, KL, command-range and EMA-bias changes, compare added
training duration rather than another coefficient. Protocol
`integrated-training-duration-continuation-v1` in
`integrated_training_budget_refinement`: control1000 versus budget4000
additional updates from the same exact c81b decoupled parent. This parent
already descends through earlier trained warm-start continuations; neither
arm trains from scratch or represents the model's total lifetime updates.
The hypothesis is narrowly whether another3000 updates under the fixed
current recipe can improve held-out tracking without motor/range regression.
The canonical4096-world gait budget is not volume-equivalent: here256 worlds
means6.144M versus24.576M transitions, not393.216M at4000×4096×24.

Restore the same `angular-decoupling-ea24a14a-20261011/decoupled/model_999.pt`
SHA `c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae`,
all actor/critic/normalizer/Adam state and LR5.062500000000001e-5, fresh
episodes/counters0. Both use the unbiased control cfg of the preceding
experiment: yaw-bias weight0 with the same reward-state diagnostic, signed
additive angular weight2 and variances.5/.5, motor-1, qpos-4/margin.15,
twist-.5, forward0, action-rate-.8. Keep61D actor/14D unclipped/unfiltered
actions, adaptiveKL.01/four minibatches/five epochs and all frozen curricula,
plant/DR/head/pose commands. Only max_iterations differs1000/4000. Longer
normalizer/Adam/optimizer evolution is part of this duration intervention,
not a hidden reset or a promised causal/convergence proof.

Effective twist sampler remains vx(.2,.4), vy0, yaw±.5,3–5s resampling,
standing15%, forward-only25%, no turn-in-place. Prior latest control counts
932570 idle/1378625 straight/1917248 negative/1915557 positive action-world
steps already show balanced broad yaw exposure; do not call it untrained.
Add identical read-only actor-command dwell bins to both arms: exact idle,
exact straight, and signed magnitudes(0,.15],(.15,.25],(.25,.5]. Capture
before action and consume once after its control step, including worlds that
reset; validate finite support, exclusive counts and per-update/cumulative
world-step totals. No command/RNG mutation, band reward, new sampling mixture,
filter or new observation. This is descriptive telemetry, not a new gate.

Preflight paired64-world/five-update smokes seed1049 with normalized finite
61->14 ONNX CPU checks, then paired256-world/ten-update benchmarks seed1051.
Full paired256 worlds/seed1061/24 steps per update, control1000 then budget4000.
Retain cadence250 plus final999/3999 respectively; evaluate those declared
finals exclusively, never choose an intermediate from the observed metrics.
The shared harness must verify each arm's declared update count/final path,
not assume every recipe ends at999. Legacy1000 recipes retain their existing
final999 behavior and caps. Test malformed contracts and nonfinite timing.

Evaluate the unchanged12 held-out cases per arm: seeds839/853/857,8 worlds,
240 steps/startup60, commands0/0,.3/0,.2/-.2,.2/+.2. Preserve every-world
speed MAE≤.1/idle≤.03, body-yaw MAE≤.1, pooled torque-p99 utilization≤.60,
zero terminals/rated-speed exceedance, all14 hard-range overruns≤.005rad and
named right-hip-yaw proximity exposure≤1% at.05rad. All24 cases and both
decisions retained even if the longer final is worse. One paired seed and
GPU nondeterminism cannot establish robust convergence or physical readiness.

Require reviewed/tested/pushed clean exact source, native CUDA-hidden checks,
all four completed preflights, exact finite parent restores/checkpoints,
reward/optimizer/exposure identities and measured conservative projections
before full training. Benchmark seconds×150 must fit2040s for control;
×600 must fit7140s for budget. Per-arm cap2100/7200s, evaluation240s;
one sequential retained9600s user service/15s grace/control-group kill,
8GiB memory/CPU200%/tasks64/nice10,20% allocator and unchanged lease,
DINO2021/GPU/occupancy/temperature/protected-service guards. Literal caps
and paths; deadline checks require remaining9615/7455/255s before respective
phases. All work must safely finish before2026-10-12 07:00 Beijing time
(epoch1791759600); no launch if the remaining phase budgets do not fit.
No concurrent Duck GPU job, unrelated workload change, runtime/driver/package
change, perception/actor expansion, rejected-policy video or physical motion.
Do not restore protected services without explicit owner request.

Implementation verification:629 focused CPU tests pass locally with three
existing actuator/site-selector warnings, including legacy1000-recipe
compatibility, declared3999 selection, malformed/nonfinite budget guards,
exact receipt mode/seed/world-step identities, exclusive telemetry bins,
terminal-reset timing and inherited reward/optimizer checks. Whitespace
check passes. An initial synthetic telemetry fixture incorrectly retained
post-reset vx0 when reintroducing a nonzero yaw; the support guard correctly
rejected it. Correcting only the fixture to restore its declared vx made
the complete suite pass; no runtime bound or numerical gate was relaxed.
Native preflight/full training remain pending.

Final independent review caught a preflight identity gap in the shared
harness: same-source receipts were not checking declared mode/seed and full
step/transition counts. Closed it for smoke-before-benchmark and both arms'
preflights before full training, including exact final preflight weight hash.
No receipt can authorize a duration projection merely by occupying the
expected directory. Nineteen additional regression cases cover wrong
preflight identities/declarations; final focused suite648 passed in18.00s,
three existing warnings. Read-only review confirms the blocker is closed.

### Duration preflight verified and retained full launch

Exact reviewed/pushed native source
`7a9c37da8a11695180572c982ebcdf1358d84d30`, clean detached worktree.
Preflight service `microduck-training-duration-7a9c37da-preflight.service`
finished01:49:09 Beijing time, inactive/dead/MainPID0/Resultsuccess/exit0.
Its initial transported test list collected574 tests; before any full launch,
ran the corrected complete648-test list CUDA-hidden at the same clean source:
648 passed/three existing selector warnings in12.54s. No GPU configuration,
weight, source or numerical threshold changed to correct that invocation.

All four same-source preflight receipts, exact initial parent model/normalizer/
Adam restoration, LR/counters, finite cadence/final weights,600 captured
adaptive-KL minibatches and actual zero-weight bias diagnostics verified.
Every telemetry row has exclusive complete per-update/cumulative command
dwell counts. Both official normalized ONNX artifacts pass finite61->14
CPU inference and byte hashes. The runner's optional automatic ONNX hook
still logs its known logger_type warning; the separately retained normalized
export/inference check is the successful evidence, not that warning.

| Mode | Seconds | Falls | Mean diagnostic EMA cost | Actual bias reward | Mean signed angular reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke-control | 4.574443 | 0 | .053346 | 0 | .963731 |
| smoke-budget | 4.604462 | 0 | .053544 | 0 | .968039 |
| benchmark-control | 7.730003 | 0 | .051253 | 0 | 1.020039 |
| benchmark-budget | 7.792399 | 0 | .051137 | 0 | 1.017405 |

Conservative timing projections1159.500s(control×150)/4675.439s(budget×600)
fit2040/7140s. Paired benchmark dwell counts are identical (seed1051), with
5147 negative-target and3993 positive-target action-world steps out of61440;
these short dwell statistics do not establish rarity, tracking or admission.
No preflight NaNs. Peak1717MiB/58C; post-preflight DINO2021 only969MiB/50C
and all protected service scopes inactive. Twenty small launch/result/warm/
update/health files hash-match native under
`artifacts/evaluations/training-duration-7a9c37da-20261012-preflight`, plus
copied manifest. Models retained only on native.

Full service `microduck-training-duration-7a9c37da-matched-eval.service`
launched by01:51 Beijing time, MainPID1137467, unique retained run
`artifacts/training/training-duration-7a9c37da-20261012`. Verified9600s
runtime/15s grace/control-group kill/8GiB RAM/CPU200%/tasks64/nice10,
literal2100/7200/240s child caps and declared9615/7455/255s remaining-budget
checks against07:00. Native source remains frozen7a9c37da while active.
Sequential control1000 then budget4000 then unchanged24-case final999/3999
evaluation. Training progress is not acceptance.

| Preflight receipt | SHA256 |
| --- | --- |
| smoke-control | `f51e60d945de3987a1cfb8b1803e9f024dd390155c81b32e5fb2589e91906e9c` |
| smoke-budget | `dbe58b272a5b2225225eb97dfb2bf80636f76b8b0d018e0e1318fa5f58f0bb65` |
| benchmark-control | `b7d34588e8d7766e23861c4a3c8725e583324a805c60627b76367a9b84035a64` |
| benchmark-budget | `df9845b8069c39397116ce4ff761c2604290c45a838c117616cdade28a8df5c3` |

### Read-only aligned gait/head diagnostics prepared locally

While the duration service runs at frozen7a9c37da, add optional
`gait_trace=False` capture to the existing evaluator. This does not deploy or
change the active source. Retain pre-action commands, body/world angular rates
and quaternion, integer episode counter and batched physics time, named
left/right contact counts and world-frame net force, cached air/contact timers,
named foot-site world velocities, four named head joint offsets/velocities and
the exact actor/manager head commands. Read sensor data without calling sensor
update/reset; clone live views before policy/step. No RNG, reward, action,
command, plant, normalizer, filtering or observation change.

The stdlib-only offline summary validates complete8-world/240-step traces,
fixed commands/seed/checkpoint/timing, nonnegative integral contact counts
(the real sensor stores floats), reset-free clocks and the existing unit-quat/
body-world rotation/MAE arithmetic. Exclude the same60 startup steps; report
per-world no/one/two-contact fractions, sampled double-flight and timer
disagreement fractions, contact-masked foot-site horizontal-speed proxies,
contact-conditioned yaw MAE and within-world head-qdot/yaw residual correlation.
Pooled statistics remain explicitly labelled: they can include between-world
covariance and do not replace within-world evidence. Foot-site speed is not
exact contact-patch slip; sampled flight is not substep-resolved flight duration.
Correlations do not prove that head motion or flight causes yaw errors.

Partial terminal traces remain raw failure evidence and cannot be summarized
as completed cases. Tests compare original evaluator evidence with capture on
and off, verify pre-action clocks0..239, detach/nonaliasing/RNG preservation,
real cached-sensor timer immutability, named interleaved selectors, malformed
channels and an across-world correlation-confounding example. Independent
read-only API/XML review finds no mismatch.696 focused CPU tests pass locally
in47.30s (three existing selector warnings); whitespace check passes. Native
diagnostic execution is not yet authorized by a complete predeclared input
bundle: wait for both duration finals and all24 unchanged held-out cases,
verify their bytes/decisions and idle GPU, then pin a bounded diagnostic if
the numerical evidence warrants one. No new reward intervention is inferred
from training logs or these synthetic tests.

### Predeclared bounded duration gait/head replay

If the completed duration matrix retains a tracking failure, run one
diagnostic-only8-case replay, not another reward search. Select both declared
finals (control999/budget3999), seed839, and all four commands in the existing
order before seeing the duration result. No intermediate checkpoint, best
world, accepted-only subset or seed selection. Input run is fixed to
`training-duration-7a9c37da-20261012`, training source is full7a9c37da above;
parent c81b,256 worlds/seed1061 and the declared1000/4000 update counts must
match exact launch/receipt/weight bytes. The full24-case result and original
deterministic decisions must be verified before any replay.

`integrated_gait_yaw_audit` requires an explicit lowercase SHA256 of the
completed aggregate over raw bytes. Record that hash and both final hashes
below after completion, then pass the literal aggregate hash at invocation.
It independently checks source/protocol/runtime, exact resolved final paths,
all ordered24 case identities, selected final iterations and recomputed
decisions. The audit source must be a pushed/tested exact clean revision;
never advance the native worktree while duration training/evaluation is active.

Use the same eight-world240-step/startup60 evaluator with optional pre-action
body and gait capture. Write both raw traces plus original/observed case
evidence before validators can reject a partial episode or clock mismatch.
Hash-bind both summaries to the raw file and each result row to its summary.
Retain every replay difference rather than overwrite the original result or
claim GPU determinism. Recomputed MAE/frame/contact/head statistics remain
diagnostic-only, with all acceptance/qualification/physical flags false.
None of these eight cases can replace the full24-case admission matrix.

One retained user service with literal300s cap/15s grace/control-group kill,
8GiB RAM/CPU200%/tasks64/nice10,20% allocator, the existing GPU lease and
DINO2021/occupancy/temperature/protected-service guards. Start only after
the duration service is successfully inactive and GPU contains DINO only,
and only with at least315s remaining before07:00 Beijing. No learning,
new reward, raw perception, plant/runtime/package change or video. After
retaining and reviewing this evidence, any distinct training intervention
requires a separate tested predeclaration and smoke/benchmark before launch.

Preparation verification:716 focused CPU tests passed locally in25.34s,
three existing selector warnings. This includes strict input-bundle/hash/
receipt/path/order guards and a mocked eight-case entrypoint proving the
raw-before-both-summaries retention sequence. No simulator or GPU diagnostic
has run yet, and the active native duration source remains unchanged.

Root integration review added an exact cross-trace identity check for all
five shared pre-action state channels: matching means are insufficient if
the body and gait traces are offset or mixed. Mismatches retain raw evidence
before rejection. Five additional regression cases plus the real optional
capture fixture cover this alignment.721 focused CPU tests pass in19.38s,
three existing warnings; no admission threshold or active job changed.

### Duration continuation completed: no accepted tracking foundation

`microduck-training-duration-7a9c37da-matched-eval.service` completed before
02:47 Beijing time on2026-10-12: inactive/dead/MainPID0/Resultsuccess/exit0.
Native clean source remained frozen7a9c37da throughout. Independently verified
both exact parent actor/critic/normalizer/Adam restores, reset counters/LR,
every declared cadence weight and final999/3999, all1000/4000 finite update
rows,20,000/80,000 adaptive-KL minibatches, exclusive complete command dwell,
and all24 held-out case files against the final aggregate and decisions.
Actual yaw-bias reward remained zero throughout; duration was the treatment.
No NaN terminations. Training elapsed626.402/2557.679s, transitions6.144M/
24.576M. Training falls417/1481 are aggregate counts over different exposure;
tail100 mean falls/update.50/.24, not a matched held-out stability improvement.

All24 evaluation cases completed240 steps with zero terminals or rated-speed
exceedance. Both arms pass12/12 motor cases and12/12 named/all-joint range
cases; maximum pooled torque utilization.506797/.478202 remains below.60.
Idle passes all24 worlds per arm. Both arms still fail every one of the nine
moving-case combined tracking gates: `foundation-not-ready`, no acceptance.
Moving yaw-passing worlds28/72(control) versus10/72(budget), despite lower
tail100 training raw-yaw MAE.303053/.279155 and motor-cost.048434/.045713.
This single-seed continuation does not support automatic duration escalation
or a convergence claim. Canonical4096-world training is not equivalent to
this256-world experiment.

| Command vx/yaw | Speed MAE control/budget (m/s) | Body-yaw MAE control/budget (rad/s) | Yaw passing worlds control/budget |
| --- | ---: | ---: | ---: |
| 0/0 | .001279/.001377 | .010520/.010290 | 24/24,24/24 |
| .30/0 | .088456/.095778 | .105627/.119064 | 12/24,6/24 |
| .20/-.20 | .033051/.036601 | .110472/.136906 | 10/24,1/24 |
| .20/+.20 | .039538/.039031 | .124510/.122025 | 6/24,3/24 |

Means describe24 worlds across the same three held-out seeds, not relaxed
per-world gates. Positive/negative training command exposure was substantial:
target-band action-world counts396274/384974(control negative/positive),
1569237/1583546(budget); no evidence of missing signed-command training.
Train peaks1717MiB/62C; evaluation2859MiB/59C. Post-job DINO2021 only969MiB/
50C,15,225MiB free, and all user/system protected service scopes inactive.
No policy video, physical motion, raw perception or service restoration.

Retain37 structured small JSON/JSONL files under Mac
`artifacts/evaluations/training-duration-7a9c37da-20261012`, with a copied
manifest and native byte-hash verification. Weight files remain native only.
The predeclared tracking-failure condition for the eight-case gait/head
diagnostic is met. Pin the following exact full aggregate and finals before
replay; do not replace the original24-case matrix with replay results.

| Retained artifact | SHA256 |
| --- | --- |
| Control final999 | `f9b6ea96b3eebe4f51a24cf5fb6c44523ba8fe9ea608dbbb6078016c41892df3` |
| Budget final3999 | `381731d397098f41c7a1fae1017d5648dfa7c0de5e51054e934c2ba7ade74437` |
| Control receipt | `db4cffc44e80993735610d73d17184cfb2bf8e115a484e15b379f255379f7fbc` |
| Budget receipt | `a18e8b88c74837b468e71f4f3130d266ce1c68fefadbff375e005d5ca622e84e` |
| Evaluation launch | `3e870b50891b5881f2afcccd14e8b3b769a5af4e7303aeba3da37894f0ee0cc9` |
| Full24-case aggregate | `5a92030e62a4e3c42a2dba29901b699edb68122df70b0e0939a2e634dced784f` |

### First gait audit stopped: read-only Warp clock conversion correction

Native fast-forwarded only after duration completion/idle verification to
clean pushed751ba2c4f8d1e0e9b968c036f4b9ace8c47a2734; all721 CUDA-hidden
focused tests passed in14.51s with the three existing selector warnings.
The bounded `microduck-gait-yaw-751ba2c4-audit.service` stopped02:48:53
Beijing time with exit1 and retained `failure.json`: ValueError,
`finite pre-action gait channel layouts`. No completed diagnostic case or
new training result. Original duration input bytes/decisions remain intact.

Before editing, ran one bounded60-second, eight-world, pre-action-only
read-only shape probe under the same lease/allocator/host protections.
`microduck-gait-yaw-751ba2c4-shape-diagnosis.service` finished02:50:19,
inactive/Resultsuccess/exit0. All15 other channels are finite tensors in the
declared layouts. The physics clock alone is mjlab `TorchArray`, shape[8],
not a torch.Tensor subclass. Installed `mjlab.sim.sim_data.TorchArray`
implements read indexing by returning its backing tensor; the synthetic
test fixture had hidden this distinction. No sensor update/reset, policy
action, physics step, reward or input mutation was used to diagnose it.
Post-probe GPU is DINO2021 only.

Correct capture to read `robot.data.data.time[:]`, then use the same strict
layout/finite/clock checks and detached CPU clone. No time synthesis or
threshold change. A real CPU Warp TorchArray regression verifies successful
capture, clock immutability and retained-sample nonaliasing. Keep the failed
launch/health/failure receipts under
`artifacts/evaluations/gait-yaw-751ba2c4-20261012`. Retry the same predeclared
eight cases only after tested/pushed clean correction and native CPU tests,
with a new unique run directory/unit and the same literal original aggregate
hash. Never erase the failed attempt or interpret it as a policy failure.

### Corrected gait replay verified: no reward-mechanism conclusion

Clock correction source618d2305c4de8ea4d0705759eb1059a02518616e was clean,
pushed and tested:722 focused CPU tests locally12.75s/native14.44s, three
existing warnings. New `microduck-gait-yaw-618d2305-audit.service` completed
before02:54 Beijing, inactive/dead/MainPID0/Resultsuccess/exit0. Every one
of the eight declared cases completes240 steps/8 worlds with zero terminals.
Verified raw/summary/result SHA bindings, shared body/gait samples exactly,
fixed commands/head-command parity, reset-free episode/physics clocks,
quaternion/angular frames and recomputed yaw MAE. Both offline summaries
recompute exactly. All eight replay results differ from original retained
cases; retain those differences and do not replace the original decisions.

Across all11,520 settled world-samples, no sampled no-contact/double-flight
frames or air-timer disagreement. Idle has two contacts throughout. Moving
cases have one contact95.76–96.74% and two contacts3.26–4.24%; named foot-site
contact-masked horizontal-speed means.0107–.0161m/s. Two-contact yaw MAE
is larger(.1355–.2573 versus.0977–.1279 in one-contact samples), but it covers
only a small gait-phase subset. These are sampled contact/site-speed proxies,
not evidence of precise flight duration, contact-patch slip or causality.

All head commands are zero. Moving mean absolute head-yaw velocity1.34–1.77
rad/s co-occurs with the gait, but within-world residual/qdot correlations
are mostly modest/sign-inconsistent: world0 is negative in each moving
case, many others modestly positive, with mixed signs in some worlds.
This selected-seed replay provides no uniform head-motion causal finding.
Do not freeze the head, tighten tracking, change airtime/slip rewards or
substitute filtered yaw based on this diagnostic.

19 raw/summary/launch/result/health files hash-match native under Mac
`artifacts/evaluations/gait-yaw-618d2305-20261012`, plus copied manifest.
Diagnostic peaks1699MiB/57C, protected scopes inactive; GPU returns to DINO
only. Failed first attempt's three files also hash-match native and remain
retained separately. No policy acceptance, simulation qualification or motion.

| Corrected diagnostic artifact | SHA256 |
| --- | --- |
| Launch | `f4f0988f462e8bfcf1889a5819a02ba0d3329192a72ecacc934fcfb2efd561bd` |
| Eight-case result | `ecd91c3501e9ffab21d95574170ac744ca1e682c1501ecbc0f373b5d2a713c18` |

### Predeclared fixed-recipe rollout-batch comparison

Next candidate is `integrated-rollout-batch-continuation-v1`, not another
reward coefficient search. Two matched arms: control256 worlds and batch1024
worlds, each1000 additional updates from the same c81b parent above, full
actor/critic/normalizer/Adam restoration at LR5.0625e-05 and reset counters.
Seed1087 both arms,24 steps/update, five PPO epochs/four minibatches,
adaptive KL.01, cadence250 plus final999. Exactly6.144M/24.576M additional
transitions and20,000 optimizer minibatches each. Select both final999 only;
retain the same ordered24-case admission matrix and all per-world speed,
body-yaw, idle, motor, range, stability and terminal gates. No replay result,
intermediate checkpoint or relaxed threshold replaces admission.

Only rollout world count differs. Signed additive angular weight2/widths.5,
zero-weight EMA diagnostics, motor−1, named right-yaw proximity−4/.15margin,
twist−.5/forward0/action-rate−.8, head/body objectives, sampler, plant/DR,
collapsed curricula,61 actor/14 unclipped/unfiltered actions remain unchanged.
World count changes batch/minibatch size, environment/RNG topology and
normalizer/Adam trajectories, so this is not a pure causal more-data test.
The1024×1000 arm matches transition count of rejected256×4000, not optimizer
updates or trajectories. Neither capacity nor better tracking is presumed.

Declare exact `TRAINING_WORLDS={control:256,batch:1024}` opt-in; legacy
recipes remain256-only. Both five-update smokes use64 worlds/seed1073;
each ten-update benchmark uses its actual full arm world count/seed1079.
Before any full launch verify all four same-source exact receipts, restored
initial/checkpoint/Adam state, finite updates/motor/KL/dwell, both normalized
ONNX61->14 CPU checks and health. Projection remains benchmark elapsed×150;
must fit control2040s and batch4440s. A256-world benchmark cannot authorize
1024-world training. No projection from free VRAM or canonical4096 runs.

Keep existing20% allocator,6144MiB-total/8192MiB-free/<75C and DINO2021-only
shared-host lease/ownership guards,8GiB RAM/CPU200%/tasks64/nice10. Preflight
one retained sequential service≤600s,15s grace/control-group kill. Full run
one retained sequential service≤6900s, childcaps2100/4500/240s with15s grace;
require6915s remaining before07:00 at launch and per-phase remaining-budget
checks. Control precedes batch; evaluation runs only after both complete.
Do not change source while active. If measured capacity, finite/motor guards,
receipt identity or deadline fails, retain failure and diagnose read-only;
do not raise resource limits, alter driver/runtime or overlap GPU jobs.
No video, new skill promotion, raw perception, physical motion or protected
service restoration is authorized by this exploratory comparison.

Preparation verification:748 focused CPU tests pass locally in13.91s with
the same three existing selector warnings; whitespace check passes. Config
parity checks normalize only declared world counts/run tags, and shared
harness regressions reject absent/ambiguous/reversed/unbounded world plans,
wrong full-world counts and a256-world treatment benchmark. Independent
read-only integration review finds no blocker. Native capacity and full
training remain gated, not established by CPU tests.

### First rollout-batch preflight stopped at a legacy telemetry bound

Clean pushed b3d837fd92d3514c225c94514086f2b08df04895 passed all748 native
CUDA-hidden focused tests in14.55s, three existing warnings. Sequential
`microduck-rollout-batch-b3d837fd-preflight.service` completed both64-world
smokes and256-world benchmark, then failed02:59:35 Beijing with exit1.
Retained1024-world failure is ValueError,
`invalid bounded motor stream layout/configuration`, before any optimizer
update/full training. Read-only journal/source diagnosis identifies the
MotorStepStream constructor's256-world ceiling; it is not an OOM or measured
1024-world timing/capacity result. GPU returned to DINO2021 only969MiB/48C.

Preparation correction: keep MotorStepStream's default256 bound and add an
explicit integer `max_envs=1024` option, with1024 absolute ceiling. The shared
trainer opts in only after validating the actual full/preflight mode's exact
world contract;64-world smokes and all legacy modes retain256. Forward that
capacity through foundation.training to from_robot/constructor. Preserve
one-step bounded retention, pre-reset capture/generation lifecycle, cost
formula and all motor acceptance/resource guards. This raises only the
explicit telemetry layout capacity required by the already-declared arm;
20% allocator,6144MiB totalGPU/8192MiB free/<75C and8GiB hostRAM stay fixed.
CPU regressions exercise1024-world identical costs/repeated resets and reject
implicit1024,1025/4096, noninteger/boolean/unbounded capacity requests.
Preserve the failed source/receipts separately. Retry all four preflights at
one newly tested/pushed source in a new retained run; never combine old/new
receipts to authorize full training. Numerical recipe/seeds/gates unchanged.

### Corrected rollout-batch preflight verified and full launch

Sourceb0dedcb341fa9703ce944bf410918f10099bdf7f is clean/pushed;763 focused
CPU tests pass locally15.23s and native CUDA-hidden14.54s, three existing
warnings. Read-only integration review confirms bounded capacity forwarding
and no other fixed256 stream/training bound. The corrected sequential
`microduck-rollout-batch-b0dedcb3-preflight.service` completed successfully
before03:08 Beijing, inactive/dead/MainPID0/Resultsuccess/exit0.

All four exact same-source launch/receipt/runtime/parent identities, full
actor/critic/normalizer/Adam initial restoration and reset LR/counters,
finite cadence/final weights,600 captured optimizer minibatches and complete
per-update/cumulative actual command dwell verified. Actual yaw-bias reward
is zero. Both official normalized ONNX61->14 artifacts have finite CPU
inference receipts and matching byte hashes. No NaN terminations.

| Mode | Worlds/updates | Seconds | Training falls | Peak GPU MiB/C |
| --- | --- | ---: | ---: | --- |
| smoke-control | 64/5 | 4.556316 | 0 | 1511/54 |
| smoke-batch | 64/5 | 4.573328 | 0 | 1511/55 |
| benchmark-control | 256/10 | 7.736982 | 1 | 1717/56 |
| benchmark-batch | 1024/10 | 9.331889 | 13 | 2515/60 |

These falls are exploratory training counts over different exposure, not
held-out policy admission. Conservative projections1160.547/1399.783s
fit2040/4440s. Measured1024-world memory/time supports this bounded job,
not4096-world capacity or better tracking. Post-preflight DINO2021 only
969MiB/48C,15,225MiB free; all four protected service scopes inactive.
20 small receipts/launch/warm/update/health files hash-match native under
Mac `artifacts/evaluations/rollout-batch-b0dedcb3-20261012-preflight`, plus
copied manifest; weights/ONNX remain native only. The failed-source18 small
files also hash-match under the separate b3d837fd preflight evidence folder.

Full `microduck-rollout-batch-b0dedcb3-matched-eval.service` launched before
03:09 Beijing, MainPID1247506, unique retained native run
`artifacts/training/rollout-batch-b0dedcb3-20261012`. Verified6900s service
cap/15s grace/control-group kill/8GiB RAM/CPU200%/tasks64/nice10, literal
2100/4500/240s childcaps and6915/4795/255s remaining-budget checks before
07:00. Control256×1000 precedes batch1024×1000, then unchanged24-case
final999-only evaluation. Native source remains frozenb0dedcb3 while active.

| Corrected preflight receipt | SHA256 |
| --- | --- |
| smoke-control | `bff42713457ec80d42b4fdd04ab317dd4ac4c783de49ad17e4d5bb339340263c` |
| smoke-batch | `51739deeecaa74aec9b5cc0b7db30709a73cb2b9d3b3569aebf59eb352d9408d` |
| benchmark-control | `2ce151c728dcf8c55f613b8b2ea4d0d2e68b38d8f432609362329ab360f179a9` |
| benchmark-batch | `1637dd79376e8ae0c5d3ab5335002655d79f075edc48b61c4f94fe23a1ca0109` |

### Completed rollout-batch comparison: mixed, no accepted foundation

The retained service completed before03:35 Beijing, inactive/dead/MainPID0,
Resultsuccess/exit0, at exact clean sourceb0dedcb341fa9703ce944bf410918f10099bdf7f.
Both1000-update arms, all ten cadence/final checkpoints, exact parent
actor/critic/normalizer/Adam restoration and reset counters verified. All2000
update rows are finite with zero NaN terminations;40,000 optimizer minibatch
KL/LR transitions, actual signed angular/zero-weight bias rewards and complete
command dwell pass the read-only audit. Control/batch elapsed625.514331/
794.647376s and6.144M/24.576M additional transitions. Neither duration nor
the optimizer update count changed.

All24 held-out cases complete240 steps/8 worlds without terminal worlds or
rated-speed exceedance. Both arms pass12/12 motor and12/12 named joint-range
gates, but fail all nine moving combined-tracking cases; only idle passes.
Deterministic decisions and paired differences recompute exactly. No accepted
policy, simulation qualification, obstacle/hop admission or motion.

| Command vx/yaw | Control / batch speed MAE | Control / batch body-yaw MAE | Speed-passing worlds /24 | Yaw-passing worlds /24 |
| --- | ---: | ---: | ---: | ---: |
| 0 /0 | .001185 / .001199 | .009355 / .009226 | 24 /24 | 24 /24 |
| .30 /0 | .110628 / .092580 | .109742 / .108152 | 3 /20 | 5 /6 |
| .20 /-.20 | .043366 / .035963 | .121730 / .131600 | 24 /24 | 1 /0 |
| .20 /+.20 | .050295 / .039489 | .119316 / .110876 | 24 /24 | 2 /6 |

These group means are descriptive, not admission. Larger batch improves
moving speed in each group and moving yaw-passing world count8/72->12/72,
but negative-turn yaw worsens and still has no passing worlds. Its mean
signed negative-turn residual grows.050725->.082144rad/s; all24 worlds
under-track that negative command on average. No broad yaw improvement is
established and no automatic duration/batch escalation is justified.
World count also changes minibatch size, RNG topology and normalizer/Adam
trajectories; this one-seed result is not a pure sample-count causal finding.

Maximum case-level pooled torque p99 utilization.480619/.522800 remains
below.60; the field is already dimensionless. Last100 update means:
falls.63/1.00, raw training yaw MAE.302254/.275958, yaw squared error
.172009/.141083, roll/pitch squared rate.330927/.260384, motor cost
.047072/.046697, thermal proxy.046622/.046215, soft-limit fraction
.004619/.004900, right-yaw stop exposure.001032/.000717. Aggregate training
falls447/1254 cover different transition exposure; do not compare raw counts
as a held-out stability decision. Lower training error did not yield admission.
Both signed target-band command bins were substantially exercised.

Training peaks1717MiB/61C and2515MiB/68C; evaluation2859MiB/62C. Post-run
DINO2021 only969MiB/51C,15,225MiB free; all four protected service scopes
inactive.37 structured files match native SHA256 and sizes under Mac
`artifacts/evaluations/rollout-batch-b0dedcb3-20261012`, plus copied manifest;
weights remain native only. One SFTP control-copy timeout left a partial
update log/missing warm receipt. Read-only reachability, service and GPU
checks confirmed the completed job and reachable host; retry only those two
local regenerable copies, then reverify all37 hashes. No source, service,
driver, checkpoint or unrelated workload changed for that copy recovery.

The763 focused source tests had passed before launch. Additional Mac
CPU consumer regression checks pass87 tests in27.71s, and upstream/athletics
task-separation checks pass5 in5.92s; these do not establish policy behavior.
Next is the bounded action/joint-response diagnostic, not a guessed reward
coefficient, seed selection or higher-duration training.

| Artifact | SHA256 |
| --- | --- |
| Control final999 | `bdd2f66a64f86fdab6622ec6f945322688d92516fccd759f1b89d8db5378d4ca` |
| Batch final999 | `26dd8ace189b2ff0c15e252093672986587107fb935e2795ca90f8002373f1b7` |
| Control receipt | `9408a05802651a33183be0c4a8db3be97fa0784c59c92c145f4d17d8b173cea5` |
| Batch receipt | `46fa7c2fb584a019c92217cd42f8734a80761c3cba6abd0cc6e11613a92ce149` |
| Evaluation launch | `55aa097ca53603fcce1f8a24e9e578713b50a36eea83156df89c1f0072231926` |
| Held-out aggregate | `9a3ab60c7a4f7613bfd371015ce9ab9a59eb15921d5bc56a04d0ba2ea97067e4` |

### Predeclared bounded action/joint-response replay

The mixed batch result does not justify another guessed coefficient or a
longer/bigger-batch run. Next is an eight-case diagnostic-only replay of the
two declared final999 checkpoints above, seed839 and all four unchanged
commands, eight worlds,240 steps,60-step startup exclusion,20ms control dt.
Input is the literal native `rollout-batch-b0dedcb3-20261012` bundle at
b0dedcb341fa9703ce944bf410918f10099bdf7f. The runner pins the exact aggregate
and both checkpoint hashes, checks full24-case ordering, training receipts,
runtime/parent identities, deterministic decisions and paired differences
before selecting the declared eight cases. No intermediate checkpoints,
new admission criteria, actor slots, policy filtering or raw perception.

The optional evaluator hook clones pre-policy raw previous actions,
actor previous-action slots, actual named14-joint positions/velocities,
previous applied targets, cloned action offsets/live encoder bias, configured
hard limits, lagged generalized constraint forces and reset/common/physics
clocks. It attaches the current raw action after the existing single policy
call and before the existing single environment step. The default evaluator
does not invoke the helper. Runtime action layout, disabled wrapper/action
clipping, exact named action/motor columns, history alignment, reset-free
clocks and next-target affine mapping are checked. Real CPU Warp-backed
adapter tests cover constraint and clock slicing; no wrapper-as-Tensor
assumption. Samples are detached clones, not policy inputs.

Each full raw observed case and aligned body/action traces are retained before
offline descriptive summaries. Actual body-yaw gates remain unchanged;
body/action shared samples must agree exactly. Full observed-case differences
are retained, while the explicitly added `joint_target_diagnostics` field
alone is excluded from a separate original-field replay comparison. Other
unexpected fields or metric drift cannot disappear from the comparison.
The original24-case decisions are retained and never replaced by these eight.

Summaries describe settled per-world/per-joint raw-output variation, actual
joint motion, previous-target error, configured-range target overshoot,
actual position margin/stop proximity and lagged constraint exposure, plus
within-world yaw-residual correlations. Raw output beyond magnitude1 is not
saturation; BAM targets may intentionally exceed joint ranges. The .15rad
position-margin costs across all joints are hypothetical diagnostics, not
claims that the selected right-yaw reward was applied to every joint.
Constraint force includes contact and limit contributions and solver lag;
it is not an isolated simultaneous hard-stop force measurement. Correlations
are co-occurrence, not causal grounds to freeze head motion or clip targets.

Launch only after focused CPU checks, exact tested/pushed source, clean native
checkout and fresh DINO2021-only GPU/protected-service proof. One retained
user service,240s child cap plus15s grace,8GiB RAM,CPU200%,tasks64,nice10,
owned GPU lease,20% allocator and existing shared-host guards. The255s
remaining-budget check must fit before2026-10-12 07:00 Beijing. Source stays
frozen while active. No package/driver changes, new training, video, policy
promotion, simulator qualification or physical motion is authorized by this
diagnostic; any next training factor requires a separate evidence-led
predeclaration and tests within the remaining window.

Owner integration checks:963 focused Mac CPU tests pass in40.71s (34 known
actuator/site-match warnings), including new capture/offline/runner tests,
single-policy/single-step optional-hook spies, existing foundation/command
session consumers, and upstream/athletics task-separation tests. The source
whitespace checks pass, including the new untracked files. This is source
and CPU-contract evidence only; native replay has not yet run at predeclaration.

### First action-response audit stopped before simulation: reader mismatch

Native CUDA-hidden preflight passes960 tests/3 skips in29.79s,34 known
warnings. Three retained-artifact consumer checks skip because those separate
Mac artifacts are not distributed in Git; the new capture/offline/runner/hook
checks run. Exact clean native sourcef08338bec76f0091cb4ba6a81ff79384c294d705
and DINO2021-only969MiB/49C/all protected scopes inactive verified before
launching `microduck-action-response-f08338be-audit.service` with the declared
resource caps. It exits1/MainPID0 before simulation, retaining `failure.json`:
`ValueError: evaluation must select declared final checkpoints only`.

Read-only diagnosis confirms the input aggregate still has SHA256
9a3ab60c7a4f7613bfd371015ce9ab9a59eb15921d5bc56a04d0ba2ea97067e4.
Its exact batch schema uses `both final999 only; unchanged held-out protocol`
and omits `final_iterations`, as `motor.evaluation_plan` deliberately does
when both arms have1000 updates. The new reader copied the unequal-duration
schema, and synthetic fixtures repeated that assumption. Correct only the
reader/fixture to this pinned legacy schema; derive final999 from exact
per-arm contracts and retain literal checkpoint/aggregate pins and complete
receipts. Add a test against the actual recipe's evaluation plan, not only
synthetic fixtures. No input, checkpoint, reward, action pipeline or gate
changes. Keep the failed service/output immutable and retry under a new
tested source/unit/directory after fresh idle/shared-host proof.

Corrected reader and adjacent capture/body/gait/batch/receipt regression suite:
187 Mac CPU tests pass in27.41s,31 known warnings; whitespace check passes.
The copied first failure receipt matches native SHA256
452e338e17c41fb37b5bbe90e00d68c83ce1dba8a8d953602f8d28c60d50020d.
The retry preflight must validate the actual native full hashed input bundle
with CUDA hidden, in addition to synthetic fixture/source-contract tests,
before starting any simulation. Both final999 contracts, unchanged24-case
decisions and paired differences remain mandatory.

### Action/joint diagnostic completed: retained, not an exact replay

Corrected source82350d329632a8eba3d2945862bbd6d509098664 passes187 native
CUDA-hidden tests in20.78s and validates the actual full input bundle before
simulation: exact aggregate/two checkpoint hashes, contracts999,24 ordered
cases, unchanged deterministic decisions and paired differences. The new
`microduck-action-response-82350d32-audit.service` completes inactive/dead,
Resultsuccess/exit0/MainPID0 within its240s cap, preserving the first failed
unit/directory. Eight declared cases all complete240 steps without resets.
Raw/summary hashes, shared body/action channels, action-history/target/clock
alignment, all offline body/action summaries and full/retained-field replay
differences recompute exactly. The original24-case rejection is unchanged.

All eight original-field replays differ numerically beyond the1e-6 comparison
tolerance:75–96 retained metric differences per case. Maximum per-world
yaw-MAE drift.015407rad/s and speed-MAE drift.002120m/s. Do not call the
diagnostic an exact replay or substitute its sampled tracking values for the
original acceptance matrix. This records separate hash-bound observations
with verified within-run timing/frame arithmetic; it is not proof that adding
read-only capture establishes simulator determinism.

| Seed839 command vx/yaw | Diagnostic control / batch speed MAE | Diagnostic control / batch yaw MAE |
| --- | ---: | ---: |
| 0 /0 | .001136 / .000994 | .008772 / .008008 |
| .30 /0 | .110615 / .092605 | .099881 / .100804 |
| .20 /-.20 | .043162 / .036054 | .115021 / .125519 |
| .20 /+.20 | .050347 / .039231 | .115910 / .107076 |

These means are descriptive only. Settled right-hip-yaw .15rad penalty-margin
exposure is zero in every sampled world/case; its minimum actual range margin
across moving cases is.184362rad. Do not weaken its qpos cost on this evidence.
Right-yaw raw-action std is.0416–.0735rad versus left.0860–.1243rad in moving
case world-means; right-yaw mean absolute measured speed.1015–.2887rad/s
versus left.4054–.6361rad/s. Negative-turn raw-action means remain positive
on both yaw joints:control left.01476/right.13740, batch left.02385/right.16698.
This motivates checking existing mirrored-learning support, but observations
from separate episodes do not establish a causal motor limitation or guarantee
bilateral symmetry of the randomized plant. No head freeze, target clipping,
new margin dose or raw-perception/actor-input change follows from correlations.

Head-yaw raw std.1944–.2639rad and measured mean absolute qvel1.1611–1.7687
rad/s persist in moving cases with sign-mixed within-world correlations.
Generalized constraint signals retain their contact/limit/solver-lag caveats.
A separate CPU-only hash-bound checkpoint read finds finite14D learned
Gaussian std parameters:parent.1223–.2098, control.1206–.2042,
batch.1116–.2027rad. These are distribution parameters, not deterministic
output saturation or a measured failure of exploration.

Health peak1631MiB/57C; post-run DINO2021 only969MiB/49C/all four protected
scopes inactive. All19 structured raw/summary/launch/result/health files,
102,484,271 bytes, match native SHA256/sizes on Mac under
`artifacts/evaluations/action-response-82350d32-20261012`, with a copied
manifest. Transfer is compressed in flight; no checkpoint weights duplicated.

| Diagnostic artifact | SHA256 |
| --- | --- |
| Launch | `153e731809f347bb10f5ab6dd9ac90ffc90d1a76ff6913b81ec89053d6d53ffd` |
| Result | `3047fe46fbcdf96f81abe3214423d267dfbed53894f4df5617721a35e155861c` |
| Health | `509f47b7a262351df2355fd6e7ace89aea06517a6f59cd17615bbd9ad50a0f21` |

### Predeclared asymmetric-compatible instantaneous yaw-L1 comparison

Mirror-loss review finds the existing61D/14D transformation and RSL-RL5.0.1
API coherent, but repository guidance says never enable symmetry for an
asymmetric task. This recipe retains a right-yaw-only safety reward and
right-side `bearing_roll` randomization. Do not enable mirror loss, change
those protections or infer a bilaterally symmetric plant from the audit.

Next bounded candidate is `integrated-body-yaw-l1-continuation-v1`:
control extra yaw-L1 weight0 versus yawl1 weight−.5 from the first action,
with the same instantaneous positive raw cost in both configs:
`abs(command_wz - root_link_ang_vel_b[:,2])`, referenced to1rad/s.
This prices command error, including idle, not absolute turning or action
motion. It introduces no EMA, low-pass, new actor input or plant modification.
It is an exploratory objective change, not an established cause of failure.
The existing twist residual is squared, not L1: at yaw error.1 it contributes
−.005, while the extra term contributes−.05. The unchanged signed angular
Gaussian contributes about−.0396 on its yaw component relative to zero error.
Thus the fixed new coefficient is a bounded near-zero error-pressure test,
not another variance tightening, EMA-dose search or longer/bigger rollout.
It may worsen oscillation/stability/head compensation and must pass the same
numerical gates before any use beyond this experiment.

Both arms restore the exact c81b parent, actor/critic/normalizers/Adam and
initial LR5.0625e-05, then use fresh simulated states and reset counters.
Both256 worlds×1000 updates, seed1097,24steps/update, five PPO epochs/four
minibatches/adaptive KL.01, cadence250 plus final999:6.144M transitions and
20,000 optimizer minibatches each. Smokes64×5/seed1091 and benchmarks256×10/
seed1093 are matched. New reward and pre-reset metric exist in both arms;
only its weight differs. Configuration tests must normalize only that
weight and run tag. Actual weighted reward must equal declared weight×raw
cost before reset, be nonpositive and finite; every24step update retains
instantaneous MAE/signed error plus existing motor, range, Gaussian, KL,
command-dwell, falls and NaN telemetry. No observed fall is hidden by reset.
Signed telemetry is body-yaw minus command-yaw. Training metric is measured
post-reward/pre-reset, while held-out traces sample pre-action; derived-state
substep timing is retained, not claimed as exact sample-by-sample alignment.

Keep motor−1, right-yaw proximity−4/.15rad, twist−.5/forward0/action-rate−.8,
signed additive angular2/variances.5/.5, EMAweight0, head/body objectives,
sampler, collapsed curricula, plant/DR and61actor/14raw unfiltered/unclipped
outputs unchanged. Symmetry remainsNone. Select both final999 only and run
the original ordered24cases:seeds839/853/857, commands0/0,.3/0,.2/−.2,.2/+.2,
8worlds×240steps/startup60, same per-world speed/yaw, idle, motor/range,
stability and no-terminal gates. Do not cherry-pick intermediates or admit
the checkpoint merely because yaw improves or the training reward rises.

One sequential preflight service≤600s, then, only if same-source smokes,
normalized ONNX exports, benchmarks, source/runtime/parent receipts and
finite telemetry pass, one sequential matched-training/evaluation service
≤4500s (control2100/yawl12100/evaluation240),15s grace/control-group kill.
Each measured benchmark×150 must be<2040s. Require4515s remaining before
07:00 Beijing time on2026-10-12 at full launch, and recheck remaining budget
before every child. Keep DINO2021, exactGPU/shared lease,20%allocator,
6144MiB-total/8192MiB-free/<75C guards,8GiB RAM/CPU200%/tasks64/nice10,
four protected AI scopes inactive. No source changes while a GPU job is
active. Retain/diagnose failure read-only before changes. No video, skill
promotion, raw perception, physical motion or protected-service restoration.
Tests/native capacity and launch evidence follow below.

Preparation verification:27 new focused tests pass, plus991 local focused
regressions in25.72s with34 existing actuator/site-selector warnings;
whitespace check passes. Tests include actual pre-reset reward after a
simulated terminal reset, zero-weight control, complete24steps/20KL
minibatches/dwell and config parity. They caught an inherited telemetry-list
name collision before GPU use; child lists now use a unique `yaw_l1_` prefix
and the full-parent-update regression passes. Independent read-only review
finds no source/harness blocker. Native CPU and simulation remain gated.

### Instantaneous yaw-L1 result: rejected, no promotion

Source `259092c9269fac432fb38fdbd659b0fed3609006`, native exact clean checkout;
988 native focused tests passed/3 existing artifact-consumer skips,34 existing
warnings. Both preflight smokes completed with finite normalized ONNX CPU
exports. Matched256×10 benchmarks took8.1389/8.1334s (×150 projections
1220.83/1220.01s), below the2040s budget. Preflight service exited0.

`microduck-yaw-l1-259092c9-matched-eval.service` finished successfully at
04:42 Beijing time on2026-10-12:1000 updates per arm and the unchanged24-case
final999 matrix. Training took637.677/642.343s; all2030 updates including
preflight were finite, with complete24-step command dwell and20 raw-KL
minibatches per update. Actual yaw-L1 reward equalled weight×absolute residual
including terminal snapshots. Control/yawl1 last100 training means: raw
yaw-L1.289321/.293698rad/s, weighted reward0/−.146849; squared yaw error
.155139/.159935; motor cost.047039/.049920; thermal proxy.046587/.049332;
mechanical power proxy2.140953/2.347165. Falls401/431 overall,32/37 last100;
last100 right-yaw stop exposure.05355%/.17643%. Maximum stochastic-training
right-yaw overruns.03484/.04329rad are not held-out range acceptance.

Each arm passed12/12 stability,12/12 motor and12/12 range cases, but only
3/12 combined tracking cases (all idle); every one of the nine moving cases
failed. Moving-world yaw passes worsened19/72→12/72 despite better speed.
Means across three seeds×eight worlds below are descriptive only; admission
still requires every world within each case to pass the original limits.

| command vx/wz | speed MAE control/yawl1 (m/s) | yaw MAE control/yawl1 (rad/s) | speed passes /24 | yaw passes /24 |
|---|---|---|---|---|
|0/0|.001364/.000910|.010786/.007756|24/24|24/24|
|.3/0|.102991/.094000|.107415/.113639|9/19|13/4|
|.2/−.2|.042974/.036736|.120724/.118817|24/24|4/0|
|.2/+.2|.045282/.038512|.129035/.116663|24/24|2/8|

Signed body-minus-command yaw means: idle−.003200/−.001630, straight
−.006370/−.014063, negative turn+.023406/+.045823, positive turn
−.074120/−.050000rad/s. Positive-turn improvement is insufficient; do not
cherry-pick seeds/checkpoints, increase this coefficient or create a video.
This one training seed does not establish a robust causal effect.

All numerical decisions were recomputed independently from retained per-case
world errors, motor/range flags and paired differences. Structured evidence:
[57 byte-verified JSON/JSONL files and manifest](../../artifacts/evaluations/yaw-l1-259092c9-20261012/verified-manifest.json)
(9,178,859 original bytes). Weights/ONNX remain in the native run directory
`artifacts/training/yaw-l1-259092c9-20261012`; no duplicate local weights.
Final999 SHA256: control
`fe5175d9612a0680541c58a65485b8856bfac0be794c9e68c5b86348cab48954`,
yawl1 `43a03f90f6e567aa28ea5c46ea0d66f32f5e6c37f30b8a66407b87071a7af59f`.
Aggregate evaluation SHA256
`6d3c8ef493625597ccbae8095f7a537ef5c05a2603ce21ce83378b4b6a1aa91d`;
launch `04a21e66cd05e91691006a21a120ca425b08b5b2fe7317345aac97f5666e6135`;
health `26b7179785bd5eaea874c0ec27697f168af576603b464084d99cf34a3d513ca8`.

Training peak1717MiB/61C; evaluation peak2859MiB/59C. Service inactive/dead,
MainPID0,Resultsuccess,exit0; post-run DINO2021 alone946MiB (GPU total969MiB,
free15225MiB,51C). Four protected AI scopes remain inactive. No physical
motion, raw perception, observation expansion, service restoration or skill
promotion. Next bounded hypothesis is explicit neutral-head command exposure,
not a posture-prior relaxation: read-only audit found no distinct turn-induced
hip-yaw posture penalty sufficient to justify weakening that prior.

### Predeclared explicit neutral-head exposure comparison

`integrated-neutral-head-exposure-continuation-v1` changes only
`head_pose.zero_command_prob`:control0 versus neutral.25. Exact four-axis
zero means HOME-relative command, not a missing offset or a frozen head.
Both head tracking and EMA bias rewards continue to operate on the measured
servo view minus HOME minus command, with weights2/2 and tracking std.5.
Keep head ranges±(.39,.39,.49,.11)rad and resampling2–5s; the nonzero
head tasks still occupy75% of draws. Keep body-pose sampler, all objectives,
motor−1/right-yaw proximity−4/.15rad, plant/DR,61actor/14raw outputs,
unfiltered/unclipped BAM, signed angular2/.5/.5, optimizer and original
held-out gates unchanged. Old EMA yaw-cost weight0; no yaw-L1 term.

Hypothesis: explicit neutral-head exposure better covers the exact HOME-head
command used in the held-out body tracking protocol. Continuous support
already includes neighborhoods of HOME; this is not proof of an out-of-
distribution failure or of causal head interference. Adding the zero atom
also adds RNG draws per resampling and changes the mixture's second moment,
normalization, reward opportunities and Adam trajectory. Equal seeds do not
give identical simulated trajectories. Do not interpret an improvement as
an isolated mechanical head effect or remove the head task as a shortcut.

Both arms restore the original c81b parent, not the rejected yaw-L1 policy:
`c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae`,
initial Adam LR5.0625e-05, fresh episodes/counters0. Smokes64×5 seed1103;
benchmarks256×10 seed1109; full256×1000 seed1117 for both arms. Identical
24steps/update,5epochs×4minibatches/adaptive KL.01, cadence250+final999;
6.144M transitions/20,000 optimizer minibatches per full arm.

Same observer in both arms reads already-cached raw61D actor command slots
48:51/51:55/55:61 and verifies them against twist/head/body commands before
each action. It must not recompute noisy observations, commands or rewards,
consume RNG, or advance EMA state. Record exact-zero/nonzero head dwell crossed
with each existing twist bin; count new sampled head segments with the public
command counter plus episode reset (including zero→zero resamples). Consume
pre-action counts after the action even when termination/reset/resampling
changes the next command. Require complete24-step/cumulative counts; actual
finite preflight must show zero control HOME segments and both zero/nonzero
treatment segments. Do not demand exactly25% in a finite sample.

Compare both final999 only in the original ordered24cases:seeds839/853/857,
commands0/0,.3/0,.2/±.2,8worlds×240steps,startup60; unchanged per-world
speed/yaw, no-terminal, motor and range criteria. Retain all command tasks
in training; the existing evaluation fixes HOME head/body in both arms.
No intermediate selection or changed thresholds after viewing results.

Reuse one sequential preflight≤600s with four children≤120s, then only
after receipt/ONNX/finite/capacity checks one sequential full service≤4500s,
children2100/2100/240s,15s grace/control-group stop. Both measured benchmarks
×150 must be<2040s; require4515s until07:00 Beijing time on2026-10-12 at
full launch and recheck each child. Preserve DINO2021, shared exactGPU lease,
20%allocator,≤6144MiB total/≥8192MiB free/<75C guard, RAM8G/CPU200%/tasks64/
nice10, four protected AI scopes inactive. Freeze source during GPU work;
diagnose failures read-only. No skill promotion/video/physical motion/raw
perception/actor expansion/protected-service restoration. If this fails,
retain it as a rejected sampler diagnostic; no automatic dose search.

Preparation: main regression run1000 passed/34 existing warnings in46.07s;
the completed cache/counter/reset supplement separately passed19 focused tests
in22.85s. It checks unchanged actor/commands/RNG, two complete24-step updates,
zero→zero resampling/reset segment counts, missing/duplicate/skipped snapshots,
and the actual inherited reward/20-KL update path. Whitespace check passes.
Read-only review confirmed pinned mjlab1.3's cache/counter APIs and command
layout, without a launch blocker. Native tests, smoke/export and capacity
remain mandatory before full training.

### Neutral-head exposure result: rejected, no promotion

Exact source `1afb163d7e355ec0ca6eaca8b29f65354147c4af`; the native checkout stayed clean
and frozen. Native CPU regressions passed 1007 tests, with three existing
artifact-dependent skips and 34 existing warnings. An initial shell invocation
mistakenly attempted to execute one test filename after the test command;
the corrected complete suite exited 0 before any GPU job was launched.

Both 64-world smokes passed finite normalized ONNX export checks. Benchmarks
took 7.9022/8.1720 seconds (control/neutral); conservative full-arm projections
were 1185.33/1225.80 seconds, both below 2040 seconds. The sequential preflight
service exited successfully.

`microduck-neutral-head-1afb163d-matched-eval.service` completed successfully
at approximately 05:21 Beijing time on 2026-10-12. Full arms took
642.917/655.662 seconds. All 2030 updates including preflight were finite, with
complete 24-step dwell and 20 audited KL minibatches per update; each full arm
retained checkpoints 0, 250, 500, 750 and 999. Treatment head exposure was
24.6492% exact-zero dwell, with 9491 zero and 28930 nonzero sampled segments;
control had no exact-zero head segments. Cached actor/manager command identity
and counter/reset segment accounting were verified without recomputing noisy
observations or advancing command/reward state.

Both arms passed 12/12 stability, motor and range cases, but only 3/12 combined
tracking cases, all idle. All nine moving cases failed the unchanged per-world
speed-or-yaw gate. Neutral-head exposure improved speed but did not solve yaw:
moving-world speed passes rose 51/72 to 67/72, while yaw passes were 17/72 versus
20/72. Group means below are descriptive, not replacement acceptance criteria.

| Command vx/wz | Speed MAE control/neutral (m/s) | Yaw MAE control/neutral (rad/s) | Speed passes /24 | Yaw passes /24 |
|---|---|---|---|---|
| 0.0/0.0 | 0.000985/0.001086 | 0.008536/0.008685 | 24/24 | 24/24 |
| 0.3/0.0 | 0.112936/0.091025 | 0.105059/0.114779 | 3/19 | 10/10 |
| 0.2/-0.2 | 0.050786/0.035201 | 0.117977/0.120768 | 24/24 | 2/5 |
| 0.2/0.2 | 0.057968/0.037736 | 0.116963/0.126014 | 24/24 | 5/5 |

Training falls were 464/450 overall and 41/44 over the last 100 updates.
Last-100 motor cost was .049247/.049533; thermal proxy .048720/.048998;
mechanical-power proxy 2.205872/2.258577; squared yaw error .169459/.175341.
Last-100 right-yaw stop exposure was .27409%/.07682%; maximum stochastic
training overrun .04509/.03593 rad. These training quantities do not replace
the independently passed held-out motor/range gates. Equal seeds do not imply
identical RNG trajectories, and one training seed does not prove robustness.

All 24 case files, numerical decisions and paired differences were independently
recomputed. Retained evidence:
[57 byte-verified JSON/JSONL files and verification manifest](../../artifacts/evaluations/neutral-head-1afb163d-20261012/verified-manifest.json),
11,222,471 original bytes. Checkpoints and ONNX remain in the native
`artifacts/training/neutral-head-1afb163d-20261012` directory, avoiding duplicate
local weight storage. Final-999 SHA256s: control
`477db693c22e700aa737a4e47f06c9cb858d9412184b69fc0ba3f839bc8da9a1`, neutral
`b1231819f3d33edae3c6d3ccfca446346e63213c1054751c7b65addbc43c5e45`. Aggregate evaluation SHA256:
`4e875a3fe0d2c8c325055240ddac5a54c055430a0faa8b9cc08d75b32506ca22`; launch
`239b164e27d1d4a7a46a05ca8eb8d385700015916c6303b666c8ade192968c45`; health
`1ab5bdaef167609c76bc44e53d5ccaf3e938249f4174b41ecdc580366b59389d`.

Training peaked at 1717 MiB/63C; evaluation peaked at 2859 MiB/60C.
Service inactive/dead, MainPID 0, result success, exit 0; GPU returned to the
retained DINO PID 2021 alone. Four protected AI service scopes remained inactive.
No promotion, video, raw perception, actor expansion, physical motion or
protected-service restoration. Do not automatically adopt this sampler or
search a larger zero-command probability. Next bounded hypothesis is matched
entropy-pressure consolidation from the original parent, not either rejected
checkpoint.

### Predeclared entropy-pressure consolidation comparison

Protocol `integrated-entropy-consolidation-continuation-v1` changes only
PPO `entropy_coef`: control .01 versus consolidate 0. Both arms restore the
original c81b parent
`c81bfae742f9caa815a2896cd894d38ee4d069f3237c000086df2ba0c458ecae`,
not the rejected neutral-head or yaw-L1 checkpoint. Preserve actor, critic,
observation normalizers, learned 14D standard deviation, and Adam moments.
Start with inherited Adam LR 5.0625e-05 and fresh simulation/iteration counters.

Hypothesis: removing the explicit entropy bonus during a fixed consolidation
phase may improve deterministic mean-policy tracking. This is not evidence
that exploration caused the previous failure. The restored raw-action standard
deviations are approximately .122–.210 (geometric mean .151), not a random-policy
std reset. The pinned scalar Gaussian parameter is one learned value per named
action dimension; it is direct std, not log-std. Negative differential entropy
at these values is valid. A scalar entropy loss contribution cannot be compared
with policy-gradient magnitude without gradient/update evidence.

Zero entropy coefficient does not freeze variance: PPO likelihood gradients
still update it. Do not reset, clamp or filter std or actions. Both arms retain
the original continuous head sampler with zero-command probability 0, all
head/body tasks, rewards, motor−1, right-yaw proximity−4/.15rad, twist−.5,
forward 0, action-rate−.8, additive signed angular2/.5/.5 and yaw-bias weight 0.
Plant, DR, curriculum collapse, 61D actor and 14 raw unfiltered/unclipped
outputs, symmetry None and all existing numerical gates remain unchanged.

Fixed smokes: 64 worlds ×5 updates, seed 1123 for both arms. Benchmarks:
256×10, seed 1129. Full comparison: 256×1000, seed 1151 for both arms,
24 steps/update, 5 epochs×4 minibatches, adaptive KL .01, checkpoint cadence
0/250/500/750/final999. Each full arm has 6.144M transitions and 20,000
optimizer minibatches. Select final999 only; no intermediate selection or
post-result coefficient/threshold tuning.

Identical read-only std telemetry in both arms validates the live PPO
coefficient and actual scalar Gaussian Parameter before each action and after
each optimizer update. Resolve the 14 std columns through the actual named
joint-position action term and motor stream; no hardcoded physical joint IDs.
Record the first pre-rollout std vector and post-update vector, require variance
unchanged during the complete 24-step collection, and retain analytic entropy
as descriptive diagnostics. Do not call actor forward, distribution update,
sample, commands, observations or rewards again. Finite strictly positive std
is a valid-distribution operational check, not a new skill acceptance gate.

Run the original ordered 24-case matrix for both final checkpoints: seeds
839/853/857, commands 0/0, .3/0, .2/±.2, eight worlds, 240 steps, startup 60.
All original per-world speed/yaw, idle, motor, joint-range and first-episode
no-terminal gates remain in force. Equal initial seeds do not imply identical
subsequent random trajectories after the optimization paths diverge.

Preflight is one sequential retained service ≤600s, four children ≤120s.
Only after exact source/runtime/parent receipts, CPU tests, normalized ONNX,
finite positive std and benchmark checks pass, launch one sequential full
service ≤4500s (control2100/consolidate2100/evaluation240), 15s grace and
control-group termination. Each benchmark×150 must be below 2040s. Require
4515s remaining before 07:00 Beijing time on 2026-10-12 at full launch and
recheck before every child; otherwise retain preparation without launching.
Keep DINO2021, exact GPU lease, 20% allocator, ≤6144 MiB total/≥8192 MiB free/
<75C guards, RAM8G/CPU200%/tasks64/nice10 and all four protected AI scopes
inactive. Freeze source during GPU work. Diagnose failure read-only first.
No video, promotion, raw perception, observation expansion, physical motion
or protected-service restoration. Stop new Duck work at the 07:00 deadline.

Preparation checks: 22 new focused tests passed in 12.24s after owner review,
with 11 expected actuator/site-selector warnings; 1010 inherited focused
regressions passed in 46.43s with 34 existing warnings. Actual scalar Gaussian
and named action-manager fixtures, immutable 24-step std snapshots, drift and
nonpositive/frozen-variance rejection, no distribution/forward/sample calls,
unchanged RNG and the inherited reward/dwell/20-KL path are covered.
Whitespace checks pass. Native CPU tests and preflight remain mandatory.

### Completed entropy-pressure comparison

Exact source `f5f112bf43b8e09f274e291249a5f66626b0137f` was committed and pushed before
native CPU checks and launch. Native combined tests passed 1029, with three
existing artifact-dependent skips and 45 existing actuator/site warnings.
Both preflight smokes and benchmarks authenticated the c81b parent, restored
models/normalizers/Adam, positive named 14D std, counters, normalized finite
CPU ONNX and the unchanged optimizer/reward/dwell contracts.
Benchmark projections were 1163.67/1169.90 seconds, below 2040.

The full sequential service `microduck-entropy-f5f112bf-matched-eval.service`
completed successfully by 05:52 Beijing on 2026-10-12: inactive/dead, MainPID 0,
exit 0. Control/consolidate completed 1000 updates each in
629.954585/623.044368 seconds.
Each has 24,000 common steps, 6.144M transitions, 20,000 optimizer minibatches,
and durable 0/250/500/750/final999 checkpoints. All 2030 preflight/full update
rows are finite with zero NaN terminations; coefficient, immutable rollout std,
analytic entropy, 20-minibatch adaptive-KL transitions and complete exclusive
command-dwell counts independently verify. Final std remains strictly positive:
control .117387–.210215; consolidate .047066–.112305. Final differential entropy
is -7.260562/-17.828798; valid negative values, not an operational error.

Both arms complete 12/12 first-episode cases and pass 12/12 motor and 12/12
joint-range gates. Both pass only the three idle tracking cases: **all nine
moving combined-tracking cases still fail**. No checkpoint is promoted.
Moving-world speed subset passes improve 55/72 to 66/72, while yaw subset
passes decline 18/72 to 16/72. Pooled means do not replace per-world gates.

| Command vx/yaw | Control / consolidate speed MAE | Control / consolidate yaw MAE | Control / consolidate speed worlds | Control / consolidate yaw worlds |
| --- | ---: | ---: | ---: | ---: |
| 0.0/0.0 | 0.001437 / 0.000623 | 0.011225 / 0.005878 | 24/24 / 24/24 | 24/24 / 24/24 |
| 0.3/0.0 | 0.105764 / 0.093179 | 0.109200 / 0.107861 | 7/24 / 18/24 | 12/24 / 10/24 |
| 0.2/-0.2 | 0.044879 / 0.036584 | 0.123607 / 0.124545 | 24/24 / 24/24 | 2/24 / 3/24 |
| 0.2/0.2 | 0.053235 / 0.037489 | 0.130920 / 0.121257 | 24/24 / 24/24 | 4/24 / 3/24 |

Training falls total 413/299, last100 29/24. Last100 motor cost
.048968/.033280, thermal proxy .048477/.033050, mechanical power
2.202329/1.765577 W and yaw-error square .156918/.071964 improve descriptively.
Training right-yaw maximum overrun .048409/.035065 rad is not held-out joint
acceptance. Removing entropy pressure changes exploration and normalization
trajectories; this single-seed comparison does not establish the cause of
remaining deterministic yaw error. Do not automatically adopt zero entropy
or launch a coefficient sweep.

Independent verification recomputed the ordered 24-case matrix, original
decisions, all-world tracking/motor/range flags, paired differences and hashes.
Retained Mac evidence:
[57 byte-verified JSON/JSONL files and manifest](../../artifacts/evaluations/entropy-f5f112bf-20261012/verified-manifest.json),
11,020,717 original bytes. Weights and ONNX remain native at
`artifacts/training/entropy-f5f112bf-20261012`, without duplicate local storage.

| Artifact | SHA256 |
| --- | --- |
| Control final999 | `b1f006e450e26d82fb45312c4b0ba71bbfec075e8a803c98f26b9f2ed7c9a858` |
| Consolidate final999 | `93904aacd34596dbff9c73823f1510591ebe27452b2837eab83d2745da47bd42` |
| Evaluation aggregate | `c6bc3567f4b67a68cfb3edea9b2664bb88100a60c6d4c8d06e5c1630562f6f85` |
| Evaluation launch | `932bc155beac5a62fa4716d11e7b8d461db4665a15a7681707e6ebcc17f3d34a` |
| Evaluation health | `671c3adbe7819289de23a69ddd23e3281749f9aa5c3ef948f6244fba8df2ab17` |

Training peak 1717 MiB/62C; evaluation peak 2859 MiB/59C. Post-completion GPU
is retained DINO2021 alone, 969 MiB/51C; four protected service scopes remain
inactive. No video, promotion, raw perception, actor expansion, physical motion
or protected-service restoration. Next bounded step is a descriptive nonzero
head-yaw probe of these exact final weights, not policy admission.

## Bounded head-yaw command diagnostic (2026-10-12, predeclared)

After the entropy comparison is complete, verified, and retained, probe the two
final-999 policies without retraining or promoting either. This is a descriptive
policy-response diagnostic, not another acceptance matrix or a head-mechanics
causal claim. Four cases are fixed: both arms, seed 839, eight worlds, forward
command 0.3 m/s and body yaw zero, with head yaw -0.2 and +0.2 rad.
Neck pitch, head pitch, head roll, and all body-pose commands remain zero.
The complete original 24-case entropy matrix, original decisions, source,
parent hash, per-case JSON and final weights must authenticate before probing.
The original zero-head 0.3/0 row is retained alongside each observation, but is
**not an RNG-identical replay** and has no previously captured head-error trace.

The optional evaluator hook sets the head command before environment creation
and checks all 61 raw actor command slots before each unchanged policy call.
It requires plain named servo joints without passive-output joints. The
physical head view is named servo qpos minus HOME, in the established head
joint order; errors are measured against the actor-visible float32 command,
after the unchanged 60-step startup. There is one policy call and one step per
sample, no observation recomputation, sensor update, EMA update, sampling,
action filtering or clipping. The default evaluator path remains unchanged.
Existing aligned gait/body-yaw trace validation precedes any new summary.
First-episode terminals retain partial raw evidence; they are never reset,
discarded or relabelled as successful complete cases.

Predeclare a single sequential native user service with a 600-second cap,
480-second child timeout, 15-second stop grace, and a deadline check reserving
615 seconds before 07:00 Beijing. It uses the same GPU lease, 20% allocator,
host health guards and six runtime pins plus rsl-rl-lib 5.0.1. DINO and the four
inactive protected scopes are preserved. No video, policy promotion, raw
perception, actor expansion or physical motion follows from these summaries.
A failed original tracking gate remains failed. No new head-error threshold
is introduced post hoc.

Preparation checks: 34 new focused tests pass (17 evaluator-hook and 17
evidence/summary tests). Broader Mac regressions pass 1066 in 35.83s with
45 existing actuator/site warnings. Owner-reviewed changes preserve the
default evaluator path, retain partial terminals, authenticate the complete
input matrix and compare shared gait/body trace channels exactly. Whitespace
checks pass; native CPU checks and the fresh idle-GPU gate precede launch.

### Completed nonzero-head-yaw diagnostic

Exact pushed source `a4a7582f4dd61fd354699b97b0ca31186acd2d81`; service
`microduck-head-yaw-a4a7582f-diagnostic.service` completed successfully by06:00 Beijing,
inactive/dead, MainPID0, exit0. All four cases completed240 first-episode steps
with no terminals or rated-speed exceedance. This is descriptive response
evidence only; the entropy matrix's nine failed moving tracking cases remain
failed. Nonzero-head policy competence is **not** admitted.

| Arm / requested head yaw | Head-yaw MAE rad | Neck-pitch MAE rad | Speed MAE m/s | Body-yaw MAE rad/s |
| --- | ---: | ---: | ---: | ---: |
| control / -0.2 | 0.136686 | 0.135549 | 0.107809 | 0.100347 |
| control / 0.2 | 0.145764 | 0.144500 | 0.101758 | 0.100491 |
| consolidate / -0.2 | 0.120365 | 0.072750 | 0.095293 | 0.098950 |
| consolidate / 0.2 | 0.105496 | 0.080012 | 0.094020 | 0.101426 |

The commanded head turn is not tracked closely: pooled head-yaw absolute
error .1055–.1458rad is substantial relative to the .2rad request. No new
head-error threshold was introduced. Body-yaw errors remain near .10rad/s,
and head commands do not demonstrate that mechanical head interference
caused the locomotion failure. Original zero-head baselines are retained but
are not RNG-identical replays. All named HOME-relative head measurements,
float32 commands,61D actor slots,shared gait/body channels and descriptive
summaries independently validate without policy/sensor/reward recomputation.

Corrected native CPU regression invocation passed1063 tests,three existing
artifact-dependent skips,45 warnings in32.65s. The initial command mistakenly
placed new test paths on a new shell line: the inherited1029 tests passed,
then bash reported `tests/test_head_yaw_evaluator_hook.py: Permission denied`
(exit126). Read-only inspection identified the newline; no package/driver
change was made, and the corrected complete command exited0 before GPU launch.

Retained evidence:
[11 byte-verified JSON/JSONL files and manifest](../../artifacts/evaluations/head-yaw-a4a7582f-20261012/verified-manifest.json),
19,670,113 original bytes.
Diagnostic aggregate SHA256 `6efb3d03683e42bb51834d8d44fa9eb29575c1dfb0e7dde3cc402bd9aa4eaf26`; bound original entropy
aggregate `c6bc3567f4b67a68cfb3edea9b2664bb88100a60c6d4c8d06e5c1630562f6f85`.
Peak1631MiB/55C; post-GPU DINO2021 alone969MiB/50C and four protected scopes
inactive. No video, promotion, physical motion or protected-service restoration.

### Predeclared fixed low-yaw rehearsal comparison

Proceed only after the entropy comparison and four-case head diagnostic are
finished, verified and retained. Protocol
`integrated-low-yaw-rehearsal-continuation-v1` tests one curriculum change:
training twist yaw support control [-.5,+.5] versus rehearsal [-.25,+.25] rad/s.
Both start from the original c81b parent, not either rejected entropy/neutral/
yaw-L1 checkpoint. Preserve all restored models, normalizers, Adam moments and
initial LR 5.0625e-05, fresh simulator counters, entropy coefficient .01,
continuous head sampler with zero-command probability 0, all head/body tasks,
plant, DR, rewards, motor−1, right-yaw−4/.15rad, twist−.5, forward0,
action-rate−.8, additive angular2/.5/.5, yaw-bias0, symmetryNone and raw61/14.

Negative prior: the earlier staged ±.05→.10→.20 turning-range trial was rejected
and worsened straight/negative-arc yaw. That used the earlier 604700b0 parent
and full-angular shaping, whereas this fixed band uses the later c81b decoupled
parent; that distinction justifies at most this small test, not confidence
that narrowing ranges works. The ±.25 support includes both held-out ±.2
targets from the beginning and does not stage away from the positive/negative
target band. Conditional uniform turning probability of (.15,.25] magnitude
increases from 20% to 40%, but realized dwell is measured rather than assumed.
There is no training exposure above .25 in treatment, so it cannot establish
preserved high-turn capability. Identical seeds do not make trajectories
RNG-identical after commands diverge.

Equal full budgets: 256 worlds ×500 updates, seed1181, 24 steps/update,
5 epochs×4 minibatches, adaptive KL .01. Each has 3.072M transitions and
10,000 optimizer minibatches. Checkpoint cadence 0/250/final499.
Smokes64×5 seed1163; benchmarks256×10 seed1171. All phase settings match except
the yaw support. A read-only observer verifies the live manager ranges and
cached actor command slots48:51, measures original complete exclusive dwell
bins, and requires zero positive/negative high-bin exposure in rehearsal.
No extra command sampling, sensor/observation/reward computation or RNG use.

Evaluate both final499 checkpoints only in the original ordered 24-case
protocol, seeds839/853/857, commands0/0,.3/0,.2/±.2,8 worlds×240 steps,
startup60. Every original per-world speed/yaw, idle, motor, range and
first-episode terminal gate remains unchanged. No intermediate selection,
threshold relaxation, automatic promotion, video or additional skill launch.

Preflight: one retained600s service, four120s children; normalized finite CPU
ONNX and exact source/runtime/parent/receipt checks mandatory. Benchmark
projection to500 updates with1.5 safety factor must be below840s per arm.
Full: one sequential2100s user service, control900/rehearsal900/evaluation240s
children and15s grace, requiring2115s before07:00 Beijing at launch and a
fresh per-child deadline check. If this cannot fit, retain preparation only.
Keep native GPU lease,20% allocator,aggregate≤6144MiB/free≥8192MiB/<75C,
DINO2021 unchanged, RAM8G/CPU200%/Tasks64/Nice10 and all four protected scopes
inactive. Freeze source during GPU work. No raw perception, actor expansion,
action filters/clipping, physical motion or service restoration.

Preparation checks: 24 new focused tests pass; owner-reviewed adjacent
recipe, training-budget, checkpoint, original motor/foundation and prior
turning-range regressions pass155 in24.21s. Exact single-factor config parity
also passes against the inherited c81b recipe, including original entropy.01
and head probability0. Live entropy/sampler/cache drift, zero high-bin
treatment exposure, raw61D slots and unchanged RNG are covered. Whitespace
checks pass; native CUDA-hidden tests and bounded preflight remain mandatory.

### Low-yaw rehearsal result — rejected, final499 retained (2026-10-12 ~06:21 Beijing)

Exact source `aee0a31e394ad6c81e614f0d2dda4863439b47c0`; native
`microduck-low-yaw-aee0a31e-matched-eval.service` completed successfully,
inactive/dead/MainPID0/exit0. The pinned preparation suite passed1087 native
tests (three old-artifact skips,45 existing warnings), then all30 preflight
updates and1000 full updates were finite with zero NaN terminations. Both
500-update arms preserved full Adam/normalizers/initial LR and raw61D→14D
contracts, and retained0/250/499 checkpoints. Full timings307.053/309.498s.

The sampler change happened: target-band turning dwell20.2598%→40.0348%
(control400,185/1,975,271; rehearsal773,928/1,933,140). Rehearsal has zero
high-turn samples; control has993,570. This demonstrates exposure, not skill.

Both arms pass12/12 stability, motor and joint-range cases, but only3/12
tracking cases (idle): **all nine moving cases fail**. Moving-world speed
passes56/72→62/72 and yaw passes4/72→15/72. No accepted policy or promotion.

| Command vx/wz | Speed MAE control/rehearsal | Yaw MAE control/rehearsal | Yaw-passing worlds control/rehearsal |
| --- | --- | --- | --- |
| 0/0 | .001059/.001312 | .007719/.010869 | 24/24 |
| .3/0 | .105514/.099785 | .120888/.113905 | 3/8 |
| .2/−.2 | .044604/.041496 | .140310/.148122 | 0/0 |
| .2/+.2 | .044382/.044046 | .138729/.122269 | 1/7 |

Values pool three seeds descriptively, not in place of per-world gates.
Negative-turn signed yaw bias worsens+.070526→+.092516 rad/s; positive-turn
bias−.014590→−.027333. More modest-turn exposure alone did not solve tracking.

Training falls203/184, last10039/44. Last100 motor cost.049407/.049295,
thermal proxy.048872/.048695, absolute mechanical power2.251681/2.284089W,
yaw error squared.173017/.167411. Maximum training right-hip range overrun
.027176/.025748rad is separate from the passed held-out joint-range gates.
Peak train1717MiB/61C; evaluation2859MiB/59C. Post-job969MiB/51C,
DINO2021 alone; all four protected user/system service scopes remain inactive.

| Retained object | SHA256 |
| --- | --- |
| Control final499 | `a075cc15c226ff2e86711190f4a0bfc291e43becc0bc40b1d5a4bca6a9c23023` |
| Rehearsal final499 | `ad79e55a70bef5333bfcfdeee61a8baf985422308e8abd9bbb59566c518eac09` |
| Evaluation aggregate | `1ee0abfedfebf315b86257eaaccb7ff27d570937217335b5ed20b1ce87eec404` |
| Evaluation launch | `ecab4ee1a8d75f55248ebb21d15f42012df41df8f8ea887dca920aff8f3d67cb` |
| Evaluation health | `0737474bedb37942465921a141edfcbdd33cc37969ed6be39a22d24e0747f0dd` |

Retained native run `artifacts/training/low-yaw-aee0a31e-20261012`.
All57 JSON/JSONL files (4,720,725bytes) copied and independently hash/size
verified on Mac at `artifacts/evaluations/low-yaw-aee0a31e-20261012`, with
`verified-manifest.json` binding runtime/checkpoints/receipts/full decisions.
Weights/normalized ONNX remain native only to avoid duplicate storage.
These ignored assets are retained locally/native, not claimed uploaded in Git.

Do not adopt this narrowed support, stack it with prior rejected treatments,
or launch another training-factor search before a timing/observation audit.
No high-yaw preservation claim, video, new skill admission or physical motion.

### Existing head probe: bias versus variation, not simple under-response

Read-only recomputation from all11 hash-verified head-probe JSON/JSONL files
separates mean signed error from per-world standard deviation:
`sqrt(RMS² - signed_bias²)`, then averages eight worlds after startup60.

| Arm / requested head yaw | Actual mean rad | Signed bias rad | Mean per-world residual std rad |
| --- | --- | --- | --- |
| Control / −.2 | −.162506 | +.037494 | .148153 |
| Control / +.2 | +.158329 | −.041671 | .159095 |
| Consolidate / −.2 | −.131324 | +.068676 | .120358 |
| Consolidate / +.2 | +.179948 | −.020052 | .117212 |

Thus large instantaneous head MAE cannot be interpreted as only a DC tracking
bias. Descriptive mean response gain across requests is.802088/.778178; this
is not an RNG-identical trajectory or mechanical-causality comparison.
No filtered measurement replaces the original instantaneous body-yaw gate.
The previously retained head-probe aggregate/hash identity is unchanged.

### Predeclare bounded actor-observation timing diagnostic (before simulation)

After the low-yaw matrix rejection, run no additional reward/sampler training
factor before07:00 Beijing. Measure the remaining observation-view/timing
question without changing the61D actor,14D raw actions, rewards, commands,
plant, sensors, domain randomization, policy or acceptance thresholds.

Use both exact final499 low-yaw checkpoints, seed839, all four original
commands0/0,.3/0,.2/−.2,.2/+.2, eight worlds×240 steps, startup60: eight
diagnostic cases, not a new held-out acceptance matrix. Authenticate the
complete original24-case aggregate SHA256
`1ee0abfedfebf315b86257eaaccb7ff27d570937217335b5ed20b1ce87eec404`,
sourceaee0a31e, exact per-case JSON, final499 receipts/weights/runtime,
recomputed decisions and paired differences before touching the GPU.

The existing play=True factory leaves actor corruption enabled. Its effective
base_ang_vel term rotates body-frame root angular velocity by an existing
cached per-run IMU mounting quaternion (up to6deg), adds uniform±.03rad/s
noise, then serves lag0–1 from a delay buffer refreshed every64 steps.
It is not a separately sampled head-mounted gyro. Capture already-computed
raw manager actor slots0:3 before checkpoint observation normalization;
verify their exact equality to the actual indexed selected history value.
Read selected/newest history, sampled/effective lag, history length/counter,
cached mounting quaternion, true body angular XYZ, twist and episode/physics/
common/sim clocks. Never call observation functions, noise.apply(), delay
compute(), another policy inference or another physics step in the capture.

Keep post-reward/pre-reset angular reward/body XYZ and clocks via a metrics
hook. The framework computes these before its final forward(); derived
angular state may be one5ms physics substep stale. The next actor observation
is post-forward and possibly post-reset. Align consecutive nonterminal rows
using actual20ms physics and common/sim clocks; retain the terminal reward
and prior observation if a world ends, and stop rather than pairing a reset
observation or stitching another episode. No array-index-only phase claim.

Summarize per-world yaw differences attributable arithmetically to selected
versus newest history, mounting rotation, fresh bounded noise, and reward
versus next post-forward body state. Retain raw/partial observations before
summaries, original acceptance decisions, and whether original common case
fields replay exactly. This decomposition is descriptive, not causal proof
that delay/noise caused the tracking failure; it creates no new gate and
does not excuse or filter away any failed instantaneous body-yaw gate.

Run CPU regressions for real DelayBuffer indexed reads and RNG/buffer/counter
immutability, effective-layout/config drift, clock/terminal/startup handling,
bundle identity and one policy/step per frame. Run reviewed native CPU tests
with CUDA hidden before launching one retained600s service,480s child with15s
grace and615s deadline reserve. Require fresh idle-Duck GPU, clean exact source,
native lease,20%allocator, shared-host memory/temperature constraints,
DINO2021 unchanged and four protected scopes inactive. Freeze source while
active. No training, promotion, video, physical motion or service restoration.

Preparation: focused capture/bundle tests46 pass; optional evaluator hook plus
adjacent regressions38 pass. After actual factory-layout and strengthened
clock/history checks, the combined owner-reviewed suite passes1142 tests
in29.28s (45 existing actuator-selector warnings); whitespace checks pass.
Tests caught and corrected the summary math import to the pinned
`mjlab.utils.lab_api.math` before simulation. A separate read-only review
found and confirmed correction of actual head_command/body_command term
names; no actor layout is changed. Native CUDA-hidden validation and an
idle-GPU check are still required before launch.
### Cached angular-observation timing result: descriptive, not admission

Source `dbb5df4f24c891f2a5b6459336bc36b127988087`, service
`microduck-obs-timing-dbb5df4f-diagnostic.service`, completed successfully at
approximately06:32 Beijing October12: inactive/dead, MainPID0, exit0.
Native CUDA-hidden regression passed1139 tests with3 historical-artifact skips
and45 existing actuator-selector warnings in32.81s before GPU initialization.
All eight declared cases completed240 steps without a terminal. The complete
original24-case decisions and both final499 hashes were authenticated and kept;
this eight-case diagnostic does not replace that matrix or admit either arm.

Mean of eight per-world settled MAEs; rates are rad/s. These component MAEs
are not additive and must not be converted into causal contribution percentages.

| Arm / vx,wz | Actor minus current body yaw MAE | Selected minus newest delayed-view MAE | Mounting-view MAE | Fresh-noise MAE | Reward-phase versus next body-view MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control /0,0 | .016893 | .011230 | .000224 | .015213 | .006615 |
| Control /.3,0 | .067316 | .059709 | .004098 | .015213 | .043673 |
| Control /.2,-.2 | .067091 | .058285 | .004767 | .015213 | .043745 |
| Control /.2,+.2 | .067252 | .059165 | .004623 | .015213 | .045807 |
| Rehearsal /0,0 | .018311 | .012536 | .000231 | .015213 | .010397 |
| Rehearsal /.3,0 | .063403 | .055679 | .004552 | .015213 | .042704 |
| Rehearsal /.2,-.2 | .063055 | .055369 | .005344 | .015213 | .041799 |
| Rehearsal /.2,+.2 | .063708 | .056114 | .005043 | .015213 | .041715 |

Lag1 occurs49.0972% of settled samples in every case. Actual indexed history,
bounded noise, unchanged mounting quaternion,20ms physics/common clocks,
pre-reset reward samples and first-episode identity independently recompute
from the retained raw trace. The reward-phase difference describes two views
of body state at the same control boundary; it is not a measured reward error.
Moving body-yaw MAE remains .108–.148rad/s. No delay, noise, actor-normalizer,
filter, target or reward was changed; these observations do not excuse a failure.

All eight `original_fields_replay_exact` values are false. Read-only comparison
shows identical seed/command/worlds/steps/completion/terminal/checkpoint and
motor-stream provenance, but different numerical dynamics/load summaries.
Across eight cases, maximum per-world MAE shift is .001587m/s for speed and
.010634rad/s for yaw; torque-p99 shift at most .00441. Source inspection and
CPU fixtures show no deliberate RNG consumption or delay advancement in the
capture, but GPU-to-CPU copies/reductions add synchronization. Ordinary replay
spread versus an instrumentation-associated effect is unresolved. Do not claim
an RNG-identical physical replay or causal explanation from this comparison.

Peak1935MiB/57C; post969MiB/50C, DINO2021 alone, all four protected scopes
inactive. All19 JSON/JSONL files (21,315,338bytes) were independently hash/size
verified on native and Mac under
`artifacts/evaluations/obs-timing-dbb5df4f-20261012`; Mac
`verified-manifest.json` binds exact source, weights, summaries and host evidence.
Ignored assets are retained, not uploaded in Git; no duplicate weights were copied.

| Retained object | SHA256 |
| --- | --- |
| Result | `921d0edda0595878e9f52cd91eefd562161df8a982ea0a6f3e2eff5eff706fa1` |
| Launch | `522fea006e0915c8512315ca47f9810cd32a50a5c3849f96e429dc6749a21dce` |
| Health | `93805a43e44dfb052d48a46b88523cbfef786e63bc9fffb41b1d7569aa73db0d` |

### Predeclare one bounded repeated observation replay probe

Before07:00 Beijing, allow one four-case diagnostic only, not another reward,
sampler or training-factor search. Use the exact control499 checkpoint, seed839,
command vx=.3/wz=0, eight worlds×240 steps/startup60 in fixed order:
**baseline, trace, trace, baseline**. Enable the existing angular-reward capture
in every run; toggle only the optional observation trace. Baseline here means
angular-diagnostic-only, not the historical uninstrumented evaluator. Preserve
all existing policy/config/sensor/RNG/step semantics and configure the same
Torch backend as the earlier runs; do not introduce a deterministic-mode change.

Authenticate the full original24-case matrix and training receipts as before,
plus the exact completed timing aggregate above. Retain each raw case before
summaries, including a partial terminal; stop before another case on failure.
Compare the baseline repeat, trace repeat and both nearest trace/baseline pairs
using identical identity/provenance and signed scalar/per-world metric deltas.
No new tolerance, causal test or policy gate: four repeats cannot establish a
replay distribution, mechanical cause or learned benefit. Preserve the original
rejected decisions in the result. Do not alter gates or cherry-pick a replay.

Require source-frozen exact clean pushed commit, native CUDA-hidden focused
regressions, idle-Duck lease/shared-host health, DINO2021 preserved and all four
protected scopes inactive. One600s service,480s child plus15s grace and615s
launch deadline reserve; same8GiB/CPU200%/Tasks64/Nice10/20%Torch allocator.
No overlap, training, policy promotion, video, physical motion or restoration.

Preparation: owner-reviewed new replay-probe tests and adjacent timing fixtures
pass65 tests in7.78s; the new probe plus evaluator-hook checks pass25 in6.63s.
The existing1142-test combined regression passes again in28.75s with45 existing
warnings. These overlapping runs are not summed as a new unique-test count.
Whitespace checks pass. The independent read-only review found no launch
blocker; the owner added explicit prior timing-launch runtime/hash and all eight
raw-case hash checks before committing. Native CPU qualification remains a
separate required launch gate.
### Four repeated observation replays: ordinary numerical spread exists

Exact source `a6dc9380a53652e3af9d576f54d88236334eb79d`, service
`microduck-obs-replay-a6dc9380-diagnostic.service`, completed at approximately
06:45:48 Beijing October12, inactive/dead, MainPID0, Result=success, exit0.
Before launch, the exact native source passed1158 CPU-hidden tests with3
historical-artifact skips and45 existing warnings in33.37s; the complete Mac
selection passed1161 with45 existing warnings in27.77s. Neither initializes
CUDA in the separate bundle qualification. All four declared runs completed
240 steps without terminals and retained the exact control499 hash above.

| Comparison, right minus left | Max absolute per-world speed-MAE difference m/s | Max absolute per-world yaw-MAE difference rad/s |
| --- | ---: | ---: |
| Baseline repeat (run3 minus run0) | .001118 | .015081 |
| Trace repeat (run2 minus run1) | .001073 | .010376 |
| First trace minus baseline (run1 minus run0) | .000770 | .011787 |
| Second trace minus baseline (run2 minus run3) | .001535 | .012812 |

All common-field comparisons are non-exact; both full traces also differ.
The baseline repeat's yaw-MAE spread is larger than either adjacent trace/
baseline difference. Ordinary replay variation therefore exists even without
observation tracing. The eight-case timing discrepancy cannot be attributed
to the trace hook alone; neither can this small four-run sample establish that
instrumentation has no effect, a repeat distribution, simulator cause or a
learned improvement. The original full matrix rejection remains authoritative.
Do not add deterministic-mode changes, tolerances or policy admission to this
result. It does not justify filtering or relaxing instantaneous yaw tracking.

Native CPU-only verification recomputed all four comparisons and both raw
observation/reward summaries, authenticated inputs/weights and verified all
case hashes/order and shared-host bounds. Peak1699MiB/55C; post969MiB/49C,
DINO2021 alone and all protected scopes inactive. All7 JSON/JSONL files
(5,285,016bytes) are independently hash/size verified on native and Mac at
`artifacts/evaluations/obs-replay-a6dc9380-20261012`, with a Mac
`verified-manifest.json`. No checkpoint duplication or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Result | `5d3283917e21a197418fc89152fdedbab04df1f6a686a0097dc966560a433efa` |
| Launch | `6e1a0e5a49a2edb4ab29da2ad4c3cfca2914e4cd931c9dfb3372fa3a294d8858` |
| Health | `b3570611b033644646bc0f547b31e2cc3308e6da50cf037082ede50c3af21b5c` |

### Predeclare next-window angular-latency learning comparison (not launched)

Prepare one evidence-informed hypothesis rather than stack rejected reward,
head, entropy or command-support changes. This is **source/test preparation
only** in the current window; fresh execution authority and a sufficient
measured budget are required after07:00 Beijing. No additional training
factor runs before that cutoff.

Use the original c81 parent and unchanged model/normalizer/Adam/LR restoration.
Both arms inherit the original control recipe; only fresh-arm actor
`base_ang_vel.delay_max_lag` changes1→0 (minimum already0). Keep the body-rate
function,6deg mounting rotation, uniform±.03 noise, corruption enabled,
update period64/per-env/per-env-phase/hold flags, all other observation terms
and delays,61D raw actor/14D raw actions, existing signed decoupled angular
objectives, head sampler, command support, asymmetric plant, motor and joint
constraints unchanged. No filtering, clipping, sensor recalibration or raw
camera perception. This changes a training distribution, not a deployed sensor.

Pinned mjlab1.3.0 instantiates and executes a DelayBuffer **only if max_lag>0**.
The fresh arm therefore bypasses that angular buffer entirely, not a sampled
fixed-zero buffer. Removing its phase/lag draws can change later RNG streams.
This is a matched-recipe learning comparison, not an RNG-identical isolated
sensor counterfactual. It may teach reliance on fresher rates and worsen
robustness under the unchanged delayed evaluator; reject it if so.

Matched predeclared runs:64×5 smoke seed1193,256×10 benchmark seed1201,
then256×1000 seed1213 per arm,24 steps/update,5 PPO epochs/4 minibatches,
6.144M transitions/20,000 optimizer minibatches each. Retain common0/250/500/
750/999 checkpoints; only final999 is eligible for the original three-seed×
four-command/eight-world×240-step matrix. Evaluate **both arms using the stock
corrupted0/1-delay factory**, not their training observation configs. Every
original stability, speed, instantaneous yaw, motor and named joint-range gate
stays unchanged. No post-hoc checkpoint choice or automatic policy promotion.

Prepared config parity and live telemetry must reject unrelated drift. Guard
actual actor config/layout/cache and command slots48:51; control's real delay
bounds/history/flags and selected cache; fresh's absent angular buffer and
logical effective lag0. Verify projected-gravity0/1 and joint-velocity fixed1
remain intact. Read effective lag without calling compute(), observation/noise
functions, policy or another physics step. Retain complete zero/one-lag dwell
per24-step update and cumulative common-step accounting; telemetry is not a
tracking gate. A future same-source smoke must verify the actual buffer-bypass
path, finite optimization, motor safety telemetry and durable checkpoints.

Before any full run, measured same-world benchmarks for both arms must fit
unchanged2100s child caps with margin and the then-authorized deadline. Use one
sequential retained service/lease, no GPU overlap, exact pushed clean source,
pinned native runtime,20%Torch allocator, memory≤6144/free≥8192MiB/<75C,
DINO and unrelated work preserved, protected scopes inactive. No long job is
admitted by source tests or these timing diagnostics. Hopping, obstacles,
football, video, raw perception and physical motion remain unadvanced.

Preparation implemented in `integrated_angular_latency_refinement.py` with24
focused tests. New and adjacent duration-recipe checks pass44 in7.60s;
the complete owner-reviewed combined selection passes1185 in29.27s with45
existing actuator-selector warnings. Review caught and fixed an inherited
observer-mode collision: parent `mode` must remain `control` for its angular
schedule; the arm uses separate `latency_mode`. Actual prepared sensor funcs,
params/noise/scale/delay flags were checked against the factory. Config parity,
real CPU DelayBuffer immutability, complete census and malformed drift checks
pass. Native CPU qualification and a future GPU smoke remain separate; no
fresh-arm learner, checkpoint, performance or physical result exists yet.

Native CPU qualification subsequently passed at source
`c1771ed67c92329e713097d9d71b0522876d42b5`:1182 passed/3 historical-artifact
skips/45 existing warnings in32.76s, CUDA hidden. The original c81 parent bytes,
both prepared run configs and preserved inherited observer mode were checked;
the separate qualification asserted CUDA uninitialized. At06:56 Beijing,
no Duck GPU process/service is running, DINO2021 remains alone at969MiB/48C,
and all four protected scopes remain inactive. No new job is started for the
remaining window. This qualifies preparation only: the fresh-buffer-bypass
GPU smoke, measured paired benchmarks and full held-out evaluation remain
future execution gates under fresh authority. All previous checkpoints and
hashed rejected matrices are retained unchanged.

### October12 next-window execution: resolved-selector smoke repair

The owner authorized another bounded development/training round after the
07:00 closeout. At07:28 Beijing, native100.100 was clean at
`2d443a09653ffbe90da999fd7dcda2a7242d7c17`; no Duck job was running,
DINO2021 was the sole GPU process (969MiB/48C), and all four protected
service scopes remained inactive. No service restoration, package/driver
change, raw perception or physical motion is authorized by this round.

Same-source native and Mac focused CPU selections each passed88 tests
(7.38s and9.33s). The sequential preflight service
`microduck-angular-latency-2d443a09-preflight.service` then stopped at
07:29:23, exit1, before its first optimizer update. Its retained failure is
`ValueError: unchanged joint_vel effective delay config`. No benchmark,
fresh arm or full learner started. Retain the failed directory unchanged at
`artifacts/training/angular-latency-2d443a09-20261012/smoke-control`.

Read-only diagnosis: mjlab1.3.0's manager resolves `SceneEntityCfg` regex
selectors to ordered names and IDs (or `slice(None)` when all joints match)
before the observer runs. The telemetry incorrectly compared these live
names to the pre-resolution regex. Correct the assertion to check the actual
named14-servo order and matching resolved IDs; do not mutate the observation
term, buffers, actions or RNG and do not relax any policy gate. Add CPU tests
that exercise actual selector resolution rather than only unresolved mocks.

The initial-save ONNX warning about `Logger.logger_type` is not this failure.
The registered `MicroduckOnPolicyRunner` inherits `VelocityOnPolicyRunner.save`,
which already exported `smoke-control.onnx`; its metadata warning occurred
before logging initialization. There is no evidence here of an absent export
path. Fresh successful smoke receipts must still prove final exported ONNX
bytes and the finite61-to14 CPU inference gate; the failed partial export is
not a successful smoke or training result.

| Failed-run object | SHA256 |
| --- | --- |
| Failure | `ae94534fa1d0bf8882e61bec56507015396cefd5dc500e3f481e029b0f877b17` |
| Launch | `53f3a7104cae0331db12364ef84d8549f9ccea74d589a413bba31f2c0865978b` |
| Warm-start | `58665c85b787f57c8071880440591a9d6dea97707cfb2ebd07c15e76feeacb9d` |

After the tested repair is committed/pushed, use a new unique run directory
and exact clean source. Repeat both64x5 smokes and256x10 benchmarks; only if
both same-arm receipts and measured1.5x projections fit the existing2100s
per-arm caps may the declared256x1000 control/fresh comparison start.
Keep the original c81 parent, all recipe/seed/budget settings and unchanged
stock delayed held-out matrix above. Run sequentially in retained user
services with the same shared-GPU lease/bounds and protected-workload checks.
The repaired assertion changes telemetry validation only, not the training
factor. No admission or new skill follows from passing preparation.

Repair preparation:32 focused selector/latency tests pass in4.93s;
the owner-reviewed integrated/motor-stream/observation-timing regression
selection passes759 tests with14 existing actuator-selector warnings in
12.87s, CUDA hidden on Mac. Whitespace checks pass. The failed launch,
warm-start and failure JSON bytes are copied and independently hash verified
on Mac; initial checkpoint and ONNX remain native only. Native same-source
CPU qualification and GPU smoke are still separate required gates.

### Angular-latency matched learning result: rejected, no promotion

Exact source `ac6a9880717d3d602f82b20e2fdf1442bcd61226`.
The repaired native same-source CPU selection passed759 tests with14 existing
warnings in13.66s; both64x5 GPU smokes passed finite raw observations/actions,
optimizer checks, effective-lag census, durable checkpoints and normalized
61-to14 ONNX CPU inference. Matched256x10 benchmarks took8.067083/8.087391s,
projecting1210.062/1213.109s with the declared1.5x margin, below the2040s
admission threshold for each2100s cap. Preflight service completed successfully
at07:34:14 Beijing October12; idle-Duck shared-host health was verified before
the separate sequential training/evaluation service started.

`microduck-angular-latency-ac6a9880-matched-eval.service` completed successfully
at approximately07:58 Beijing: inactive/dead, MainPID0, exit0. Control/fresh each
ran256x1000 updates at seed1213,24 steps/update,20,000 optimizer minibatches
and6.144M transitions; elapsed631.513908/626.374500s. All2030 combined smoke,
benchmark and full updates have finite telemetry and zero NaN terminations.
Every declared0/250/500/750/999 checkpoint is finite, durable and hashed on
native; final999 alone was evaluated. Warm-start JSON matches embedded
receipts and preserves exact c81 parent/model/normalizer/Adam/LR restoration
with zero initial clocks. Original c81 parent bytes remain unchanged.

Full training falls were395/287; last100-update falls46/27. Control's effective
angular lag census was3,166,114 lag0 and2,977,886 lag1 action-world steps;
fresh's was6,144,000 lag0 and0 lag1 with no angular DelayBuffer. Telemetry
protocol/mechanism labels, actual command dwell, staged reward weights and
all20 optimizer minibatches/update are verified. The missing buffer changes
RNG consumption: these are matched-recipe results, not isolated causal proof
about latency.

All12 held-out cases per arm completed their first240-step episode without
falls; all12 retained motor and named joint-range gates passed. Both idle
matrices pass, but all9 moving cases per arm fail the original speed/yaw
tracking gate. Recomputed deterministic decision for both:
**foundation-not-ready**. Moving speed world passes improved61/72→69/72 and
yaw passes4/72→8/72; neither aggregate nor average improvements satisfy the
every-world case gate. Do not widen it or replace instantaneous error with an
EMA, and do not adopt/promote the fresh recipe on these partial gains.

| Command vx m/s, yaw rad/s | Mean speed MAE control / fresh m/s | Mean yaw MAE control / fresh rad/s | Speed world passes /24 control / fresh | Yaw world passes /24 control / fresh |
| --- | ---: | ---: | ---: | ---: |
| 0.0, 0.0 | 0.001023 / 0.000868 | 0.009360 / 0.007488 | 24 / 24 | 24 / 24 |
| 0.3, 0.0 | 0.098931 / 0.086095 | 0.116947 / 0.118307 | 13 / 21 | 3 / 4 |
| 0.2, -0.2 | 0.037390 / 0.028629 | 0.122073 / 0.130818 | 24 / 24 | 1 / 2 |
| 0.2, 0.2 | 0.042810 / 0.038949 | 0.133824 / 0.132247 | 24 / 24 | 0 / 2 |

Fresh also uses more moving mechanical power and thermal-load proxy. At
straight .3m/s, mean mechanical power is1.772031/1.891619W and mean thermal
proxy.029590/.032866; max held-out torque utilization p99 is.495725/.540645.
Across moving commands fresh's soft-limit exposure and torque-utilization
p99 are higher, though all existing held-out motor gates still pass. These
are simulated mechanical/load proxies, not measured motor temperatures or
hardware acceptance. Training is not globally joint-range-clean: maximum
right-hip overrun is.032450/.036308rad, including exploration/fall steps;
that evidence is retained separately from the passing held-out range gates.
Do not clip low-kp/BAM action targets to conceal it.

Native CPU-only verification authenticated all checkpoint payloads/iterations/
common counters and receipts, recomputed every case decision and paired
difference, and checked complete lag/dwell census and shared-host logs.
Mac independently verified all59 retained JSON/JSONL payloads
(9,463,421bytes), plus the exact retained manifest, and recomputed decisions/
paired differences from copied cases. Retained under
`artifacts/evaluations/angular-latency-ac6a9880-20261012` on Mac, with a
`verified-manifest.json`; native originals/checkpoints are under the
corresponding `artifacts/training` run directory. Weights/ONNX remain native
only; no duplicate weight storage or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Control final999 | `6735b1c6869528972121ea31c010b3a67831a9f77176a69bc8f6d343e7501c3e` |
| Fresh final999 | `232e450641a19cb19fcc2ef0406e18cf85cf79e9e26669f074190330e2486301` |
| Held-out aggregate | `60103e80ad7ec3e1944f6f03c0931ffa319cee13715d16c51741f4a4ab77fda8` |
| Native verification | `ac3184ee78420d376c1ac0d8fa4225bb63d224371bee742329f7dfb49000b47f` |
| Retained manifest | `ea2f1719fd94c5f0e953a24f3d5fb1c9d83c17d0fb7e7ab2bf15593b9f6d81d5` |
| Preflight verification | `f506d337ea9577a1396c0f3b0f817238a2390ec15002025f14149045704d6c18` |
| Evaluation launch | `950eaeb0a4e1d0b20b8d6b71edfde22cc6dc0cc0549484cc2173892e9f588160` |
| Evaluation health | `8fe2171c461c41fc92b0acc9be2cab1387469167f7007cc153d1a3b8c094c20e` |

Peak recorded GPU occupancy2859MiB and62C; post-verification969MiB/52C,
DINO2021 alone and all four protected scopes inactive. No Duck GPU job remains.
No accepted new turn, hop, obstacle or football skill, MP4, raw perception,
physical motion or protected-service restoration follows from this round.

Next curriculum direction (recommendation, not a launched/predeclared job):
separate straight-gait consolidation from progressive signed-turn lessons,
then combine them with retention checks. Do not run another fresh-sensor
factor sweep or jump to obstacles from a rejected endpoint. Before a further
learner, predeclare the smallest matched progressive-command experiment,
preserve the stock delayed evaluator and existing gates, and test actual
stage boundaries, zero-command coverage, both turn signs and old motor/range
constraints. The original c81 parent and all rejected comparison evidence
remain available; no current rejected arm is an accepted teacher.

### Predeclare progressive straight-to-signed-turn command curriculum

The owner authorized another bounded round after the rejected angular-latency
comparison. Native100.100 was verified at08:12 Beijing October12: clean
`6a9eb284f1439691af26ce964fd090453a86c8b5`, no running Duck service/process,
DINO2021 alone at969MiB/48C and all four protected service scopes inactive.
This round does not authorize a driver/package change, protected-service
restoration, raw perception, physical motion, video or policy publication.

Protocol: `integrated-progressive-command-continuation-v1`, implemented in
`integrated_progressive_command_refinement.py`. Both arms inherit the original
c81 parent and unchanged control recipe from
`integrated_training_budget_refinement`: exact model/normalizer/Adam/LR
restoration with fresh zero clocks. Do not inherit the rejected fresh-sensor
checkpoint or bypass its angular delay. Both arms retain stock angular0/1
delay/noise/IMU rotation, head/body sampling, signed additive decoupled angular
rewards, motor-1, named right-hip qpos-4, command-cost-.5, forward-cost0,
yaw-bias0, entropy.01/adaptive-KL.01,61D actor/14D unfiltered/unclipped actions,
plant and all other curriculum/DR settings.

The sole experimental factor is the whole command-distribution curriculum:
progressively change yaw sampling support and forward-only sampling probability.
Control is constant yaw±.50rad/s and forward probability.25. Treatment:

| Start update, inclusive | Pre-action common-step start | Yaw support rad/s | Forward-only probability |
| ---: | ---: | ---: | ---: |
| 0 | 0 | ±.05 | .65 |
| 250 | 6000 | ±.10 | .50 |
| 500 | 12000 | ±.25 | .35 |
| 750 | 18000 | ±.50 | .25 |

Boundaries are inclusive and aligned to the first action of each declared
250-update stage, not the prior range experiment's strict-after boundary.
This is a predeclared fixed-time learning hypothesis, not proof of mastery at
each boundary or permission to tune pacing during a running job. Each full
stage contains6000 control steps; its actual action-world command dwell must
be retained, including natural carryover of earlier commands.

Keep forward speed support(.2,.4), lateral0, exact-idle probability.15,
turn-in-place probability0, and natural3–5s resampling unchanged. Pinned
`UniformVelocityCommand` already clamps forward-only samples to vx≥.3,
vy/yaw0; increasing its probability therefore also changes the speed mixture
within the original support. Idle commands still override all components.
Neither nominal straight nor sign probabilities imply exact sample quotas.

Apply stages only to the live command manager's sampling config before action.
Do not force resampling, reset episodes, rewrite current commands/cached actor,
recompute observations, advance delay histories, add policy calls or draw RNG
in the stage observer. Existing commands remain valid within expanding yaw
bounds; changing the forward probability affects future ordinary resamples.
It can change later RNG streams. This is a whole-recipe comparison, not an
isolated yaw-versus-speed causal experiment.

Telemetry must check the current expected live range/mix and unchanged sampler
fields, finite raw actor/manager command equality at slots48:51, consecutive
pre-action/post-step clocks, monotone stages and complete24-step per-update
and cumulative stage census. Retain existing actual idle/straight/both-sign
dwell and motor/range/KL/reward telemetry. Fail closed on drift; do not use
nominal sampling fractions or average errors to relax existing gates.

Matched preflight:64x5 smokes at seed1231 per arm; treatment stages compressed
to inclusive updates0/1/2/3. Because2.4s smoke is shorter than the3–5s
resampling dwell, initial commands may persist throughout: smoke proves
live stage transitions and inherited actual signed-command coverage, not
resampling in every stage. Initial±.05, rather than±.01, keeps meaningful
signed samples above the inherited coverage threshold|yaw|>.01 without
weakening it. Matched256x10 benchmarks at seed1237 use the final±.50/.25
distribution for treatment; throughput is not evidence of stage learning.

Full matched runs:256x1000, seed1249 each,24 steps/update,5 PPO epochs and
4 minibatches. Retain0/250/500/750/999 checkpoints; final999 only is eligible
for the unchanged stock-delayed three-seed/four-command/eight-world/
240-step held-out matrix. Retain every original speed/yaw, first-episode
stability, motor and named joint-range gate. Do not pick a better intermediate
checkpoint, stack rejected treatments, filter/clamp actions or promote a
partially successful arm.

Before a full learner, require exact clean pushed source, native CUDA-hidden
focused/integrated regressions, both same-source finite GPU/normalized ONNX
smoke receipts and both measured benchmarks projecting below2040s with the
existing1.5x margin for2100s per-arm caps. Use sequential retained user
services only: GPU lease, preserved verified DINO2021,20%Torch allocator,
aggregate memory≤6144/free≥8192MiB/<75C,8GiB RAM/CPU200%/Tasks64/Nice10,
15s stop grace and all four protected scopes inactive. Full service cap5300s,
children2100/2100/900s for control/progressive/evaluation. Diagnose any failure
read-only before a tested fix, retaining the failed unique run directory.
Stop this bounded round after retaining/recomputing its matrix. No obstacle,
hop, football or physical acceptance follows from source or training progress.

Preparation: the new lesson and inherited budget-focused selection pass33
tests in4.78s. The owner-reviewed integrated/motor-stream/observation-timing
selection passes772 tests with14 existing actuator-selector warnings in12.59s
on Mac, CUDA hidden. Real CPU DelayBuffer state/identity, commands, cached
actor and RNG remain unchanged at compressed stage boundaries; compute,
append and resample spies fail on an extra call. Tests cover exact idle and
both yaw signs, real pinned sampler forward clamping/idle override, config
parity, inclusive boundaries and complete stage/dwell census. Review found
and corrected idle rejection and inappropriate delay-object deep-copy
comparison before any GPU launch. Production performs cfg-only stage writes,
not costly copies of buffers or RNG state. Stage schedules must start at zero
and stay equal to the declaration; only the live manager is changed, not the
source env config. Whitespace checks pass. Native CPU/GPU preflight remains
a separate execution gate; preparation alone establishes no learned result.

### Progressive-command initial smoke: inherited-schedule collision repaired

The first native source was `903d6c0756a492425b6ce4ae3e4de750042a29ad`.
Its CPU qualification passed; `microduck-progressive-command-903d6c07-preflight.service`
then failed at08:20:43 Beijing, exit1, in smoke-control before any PPO update:
`ValueError: too many values to unpack (expected 2)`. No benchmark, treatment
or full learner started. The initial-save ONNX metadata warning was separate.
Failed artifacts are retained unchanged under native
`artifacts/training/progressive-command-903d6c07-20261012/smoke-control`;
launch/failure/warm-start JSON bytes are copied and hash-verified on Mac under
`artifacts/evaluations/progressive-command-903d6c07-20261012-failed`.

Read-only diagnosis confirmed that the new command triples overwrote inherited
`MotorObserver.schedule`, whose motor callback expects pairs. Rename the new
field to `command_schedule` throughout; preserve inherited `mode='control'`
and `schedule=((0,-1.),)`. No command-stage, recipe, seed, learner, plant or
acceptance setting changes. Add an unmocked motor callback test at every
boundary clock in every mode; mocked reward callbacks had concealed this
collision in the initial CPU fixture. Independent review confirms the fields
remain distinct. Retry only from a tested/pushed exact clean source and a new
unique retained run directory; do not overwrite the failed attempt.

| Failed object | SHA256 |
| --- | --- |
| Failure | `ea4524ac03e1b201c5f5782d17f8a16b40a89345cab746f00e844ceba5a4c50f` |
| Launch | `94968dbc7d42686e32e44e748d3d93f8ddb882468ca9313ffc46f232fc34d886` |
| Warm-start | `58665c85b787f57c8071880440591a9d6dea97707cfb2ebd07c15e76feeacb9d` |

Repaired focused tests pass39 in9.43s; the broader owner-reviewed selection
passes778 with14 existing actuator-selector warnings in14.90s, CUDA hidden
on Mac. Whitespace checks pass. DINO remains the sole GPU process at969MiB/
49C after the failure. Native CPU and successful GPU smoke/benchmark receipts
remain required before a long run; this repair is not a training result.

### Progressive-command matched learning result: yaw/speed trade-off, rejected

Exact source `75272a123540ac025667dccbf20914968fb81d00`.
Same-source native CUDA-hidden regressions passed778 tests with14 existing
warnings in13.26s; Mac passed778 in14.90s. Both64x5 GPU smokes completed with
finite raw actor/actions/optimizer telemetry, zero NaN terminations/falls,
durable final4 weights and official normalized finite61-to14 ONNX inference.
Progressive smoke stage action-world counts were1536/1536/1536/3072; actual
dwell retained720 idle,4680 straight,1440 negative-low and840 positive-low.
No target/high-yaw sample was fabricated to make the short smoke appear to
resample every stage.

Matched256x10 final-distribution benchmarks took7.814852/7.799017s; conservative
1000-update projections1172.228/1169.853s passed the2040s admission threshold.
The sequential preflight service completed08:25:14 Beijing October12.
Before the separate full service, native CPU verification authenticated
all four receipts/checkpoints/normalized ONNX hashes, exact stage/dwell
census and current idle-Duck shared-host bounds with DINO2021 preserved.

`microduck-progressive-command-75272a12-matched-eval.service` then completed
successfully at09:16:26 Beijing, inactive/dead, MainPID0, exit0.
Each arm ran256x1000 updates, seed1249,24 steps/update,6.144M transitions and
20,000 optimizer minibatches. Elapsed control/progressive630.067506/627.607202s.
All2030 combined smoke/benchmark/full update rows are finite with zero NaN
terminations. Original c81 bytes, exact zero-clock model/normalizer/Adam/LR
restoration, every durable0/250/500/750/999 checkpoint and all payload iteration/
common counters are authenticated. Only final999 was evaluated.

Control stage census is6,144,000 action-world steps at±.50/.25; treatment is
exactly1,536,000 in each declared stage. Actual straight dwell rose
1,320,347→2,303,904, while idle stayed914,898→930,006 and signed command bins
remained active. Counts represent consumed cached commands, not nominal
quotas. Natural carryover, inclusive boundaries, live sampler settings and
per-update/cumulative24-step census all verify. No source/config change was
made during the job. Full training falls451/395 and last100 falls41/40 are
not held-out stability acceptance.

Both arms complete all12 first-episode held-out cases without terminals and
pass all12 motor and named joint-range checks. Idle passes in all24 worlds
per arm. Every one of the9 moving cases still fails combined tracking:
recomputed deterministic decision **foundation-not-ready** for both.
Moving yaw world passes6/72→20/72, but speed passes62/72→48/72. In straight
.3m/s, speed passes14/24→0/24 despite the greater training straight exposure.
This is a speed/yaw trade-off, not an overall locomotion improvement; do not
promote/adopt the clocked curriculum, select an intermediate model or relax
the original every-world instantaneous gates.

| Command vx m/s, yaw rad/s | Mean speed MAE control / progressive m/s | Mean yaw MAE control / progressive rad/s | Speed world passes /24 control / progressive | Yaw world passes /24 control / progressive |
| --- | ---: | ---: | ---: | ---: |
| 0.0, 0.0 | 0.001225 / 0.000743 | 0.009542 / 0.006641 | 24 / 24 | 24 / 24 |
| 0.3, 0.0 | 0.097450 / 0.129426 | 0.120577 / 0.103853 | 14 / 0 | 4 / 13 |
| 0.2, -0.2 | 0.038523 / 0.063703 | 0.130951 / 0.136989 | 24 / 24 | 1 / 0 |
| 0.2, 0.2 | 0.044517 / 0.068356 | 0.136061 / 0.114986 | 24 / 24 | 1 / 7 |

Read-only diagnosis from the retained settled pre-action body-frame response
confirms underspeed, not merely a noisy error statistic: at commanded .3m/s,
mean achieved vx falls.202568→.170574m/s. Negative-turn mean body yaw response
also weakens-.174141→-.130186rad/s; positive-turn response is.172633→.178792.
These are descriptive signed body-frame rates, not integrated heading/path or
physical-robot acceptance. They do not replace the unchanged instantaneous gate.

| Command vx m/s, yaw rad/s | Mean achieved body vx control / progressive m/s | Mean signed body yaw rate control / progressive rad/s |
| --- | ---: | ---: |
| 0.3, 0.0 | 0.202568 / 0.170574 | 0.003283 / 0.005235 |
| 0.2, -0.2 | 0.164356 / 0.136323 | -0.174141 / -0.130186 |
| 0.2, 0.2 | 0.157542 / 0.131687 | 0.172633 / 0.178792 |

The progressive arm has lower mean simulated mechanical power/thermal proxy
in every command group; at straight .3m/s they are1.756944/1.613319W and
.031708/.027490. This does not establish greater efficiency at equal achieved
motion, because speed tracking degraded. Group torque-utilization p99 stays
within existing motor gates. No measured motor-temperature/hardware claim.
Training still includes right-hip range overruns up to.033335/.035333rad and
maximum per-update stop exposure.010417/.007650; these exploration/fall
metrics remain separate from passing held-out range gates. Low-kp/BAM action
targets remain unclipped.

Native CPU-only final verification recomputes each decision/case/paired
difference, exact stage and actual command dwell, all20 minibatches/update,
fixed reward/KL/parent provenance and shared-host logs. Mac independently
verifies59 JSON/JSONL payloads (9,829,339bytes) plus the exact retained manifest
and recomputes the decisions/differences. Retained on Mac under
`artifacts/evaluations/progressive-command-75272a12-20261012`, with
`verified-manifest.json`; native originals/checkpoints remain in the
corresponding `artifacts/training` directory. Weights/ONNX are native only;
no duplicate model storage or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Control final999 | `1667928669172c1588258ceeab1344d795fe8ef633a6785a24a75b5e29ff5be7` |
| Progressive final999 | `e5b63b790d15072cf695a02dd12c5935d4e0edd9965307acfbab579e3751a16d` |
| Held-out aggregate | `ace8c2347ebe942bc7a0ac43110d110277fcceb27489d2a84cd458f9efee174a` |
| Native verification | `3eb3d2818fc3fd848764cd6f7ba287fc745f2de0d03eb02f309147354d24a353` |
| Retained manifest | `6bf689120045567e801520066ec25b818d0baaa7695253880b3e360a2851809a` |
| Preflight verification | `8cd3eff109bb962246f2ee74b70283f415b060eb9a3cfe34ace82531833f2808` |
| Evaluation launch | `b174f559a727feac1f9eb58bdb1972c8c884616e48ceb0a77d985884bbd85aab` |
| Evaluation health | `1941d027f4189f180acc198a261b9c2dcaa996a944635fdd550a2274564c4dde` |

Recorded peak GPU2859MiB/62C; post-verification969MiB/49C, DINO2021 alone
and all four protected scopes inactive. No Duck GPU job remains. This round
adds tested curriculum infrastructure and retained learning evidence, not
an accepted new turning, obstacle, hop or football skill. No MP4, physical
motion, raw perception or protected-service restoration was performed.

Next direction (recommendation only, no new job launched): do not repeat a
clocked multi-stage sweep or promote a yaw subset while hiding the straight
speed regression. Define a separately measured straight-gait lesson and its
stage-exit/retention criteria before introducing turns; diagnose the actual
achieved forward velocity and gait/load profile that accompany lower yaw
error. Keep the stock delayed evaluator and current gates unchanged. Any
new learner requires its own predeclaration, focused tests and measured GPU
smoke/budget gate; the original parent and all rejected endpoints remain
retained, not accepted teachers.

### Fixed straight/idle lesson: predeclared matched diagnostic

The owner authorized the next bounded slice after the progressive round.
This experiment asks whether concentrating the *unchanged* learning recipe
on exact standing and .30m/s straight commands can consolidate straight
behavior without degrading the original held-out arcs. Greater straight
exposure is not assumed beneficial: the preceding progressive round already
increased exposure while reducing achieved speed. This is a whole-sampler
support contrast, not an isolated reward/gait causal proof.

Protocol `integrated-straight-lesson-continuation-v1`, arms `control/straight`.
Both restore original c81 parent bytes, model, normalizer, Adam and original
Python LR with fresh episodes and zero learning/environment clocks; neither
uses a rejected endpoint as an accepted teacher. Control retains vx(.2,.4),
yaw(-.5,.5). Straight treatment changes only those supports to vx(.3,.3),
yaw(0,0). Lateral zero, standing.15, forward-only.25, turn-in-place0 and
ordinary3-5s resampling remain unchanged. Ordinary and forward-only moving
buckets both collapse to(.3,0,0); standing overrides to exact zero. Actual
actor-consumed commands, not nominal fractions, are the exposure evidence.
There is no clocked stage change, forced resample or direct cached-command
write. The 61D actor/14 raw action interface, signed additive angular reward,
all reward weights, plant, BAM, head/body sampling, stock observation delays,
randomization, optimizer and full held-out numerical gates remain unchanged.

The shared training coverage API keeps four command bins by default for all
legacy recipes. Only this explicitly declared treatment requires idle and
straight; it also checks exactly zero yaw on every action and zero signed-yaw
counts. Control still requires all four bins. This is a sampler lesson
declaration, not a relaxed evaluation requirement. Observer inheritance keeps
the original motor/reward schedules and actual pre-action command census.

| Phase | Worlds x updates per arm | Matched seed | Cap |
| --- | ---: | ---: | ---: |
| CUDA-hidden focused/broad regressions | no GPU | n/a | before smoke |
| GPU smoke + normalized finite61-to14 ONNX |64x5|1259|600s each|
| Same-world benchmark |256x10|1277|600s each|
| Matched continuation |256x1000|1279|2100s each|
| Final999 full held-out matrix |8x240, seeds839/853/857, original4commands|fixed|900s total|

Each full arm has6.144M transitions,24 steps/update and20,000 optimizer
minibatches. Only final999 is evaluated; durable0/250/500/750/999 remain
retained, not candidates for hindsight selection. Admission requires exact
clean pushed source, native regressions, same-source successful smokes/ONNX
and measured conservative projection below2040s per arm. One retained user
service executes control, straight and full evaluation sequentially, with
the existing GPU lease and shared-host bounds, DINO2021 preserved and both
protected services inactive in user/system scopes. No automatic next job.

Three distinct numerical outcomes must be retained:

- Full foundation decision: recompute the original12-case gate for each arm,
  unchanged, including named joint-range exposure.
- Straight stage exit: all6 treatment idle/straight cases must pass original
  every-world speed/yaw, first-episode stability, motor and named range gates.
- Arc retention: all6 treatment arc cases must pass original stability,
  motor and range gates; every exact paired world must have speed **and** yaw
  MAE no greater than control. Retention is not arc acceptance: original
  absolute tracking gates still determine full foundation readiness.

`straight_lesson_gate` combines the stage-exit and retention results, records
every failure and authenticates ordered cases/checkpoint hashes. A failed
result leaves turning progression closed. A pass only permits preparing a
separately declared turn lesson, not automatic training, policy promotion,
simulator qualification, hopping, obstacle/football admission or physical
motion. One training seed is exploratory even with three held-out seeds.

Straight cases additionally retain read-only pre-action body-yaw and named
foot-contact/force/foot-velocity/head traces; the existing offline gait
summary supplies descriptive contact/slip/yaw co-occurrence, not causal or
acceptance gates. Report achieved vx, per-world tracking, falls, motor/load,
joint-range exposure and gait evidence together. No raw perception, actor
expansion, filtering, clipping, driver changes or MP4 is authorized by this
diagnostic. Retain JSON evidence locally and weights only on the native host.

Local integration verification:852 tests passed with14 existing actuator
selector warnings in215.91s, covering the integrated recipe/harness, new
stage-exit scorer, offline gait summaries and motor/observation contracts.
The focused recipe run passed23 tests. These are source/CPU checks only;
native smoke, timing admission, learning and held-out results are separate.

### Fixed straight lesson result: concentration did not clear the frontier

Exact source `41808a064dd4449fd75e2a679c347a865c749511`.
Native852 CUDA-hidden regressions passed with14 existing warnings in29.52s;
Mac852 passed in215.91s. Both64x5 smokes had zero falls/NaNs, durable
weights and normalized finite61-to14 ONNX inference. Benchmarks took
8.016729/7.988376s; conservative full projections1202.509/1198.256s cleared
the2040s admission bound. CPU-hidden verification authenticated all four
preflight receipts, parent restoration, exact actual coverage and weights.

`microduck-straight-lesson-41808a06-matched-eval.service` completed
October12 at10:10:47 Beijing, inactive/dead/MainPID0/Resultsuccess/exit0.
Each matched arm ran256x1000 updates, seed1279,24 steps/update,6.144M
transitions and20,000 optimizer minibatches. Control/straight elapsed
621.585026/624.041645s. Both original c81 parent restorations, original
Python LR/model/normalizer/Adam, zero clocks and all0/250/500/750/999
checkpoint payloads are authenticated. All2030 preflight/full update rows
are finite with zero NaN terminations. Only final999 was evaluated.

Actual straight dwell increased1,306,292→5,235,568 action-world steps;
idle908,152/908,432. All treatment signed-yaw bins are exactly zero; control
retains both signs and its original complete coverage. Training falls418/411,
tail100 falls44/46; these exploration counts do not establish held-out
stability or an improvement.

All24 held-out cases complete their first240-step episode without terminals.
Both arms pass every motor and named/all-joint range case; idle tracking passes
all24 worlds. Both nevertheless fail all9 moving-case combined tracking gates:
**foundation-not-ready**. Moving speed passes57/72→52/72, yaw15/72→24/72.
The concentrated lesson trades some yaw improvement for worse speed, not a
new accepted locomotion skill.

| Command vx/yaw | Speed MAE control / straight (m/s) | Body-yaw MAE control / straight (rad/s) | Speed passes /24 control / straight | Yaw passes /24 control / straight |
| --- | ---: | ---: | ---: | ---: |
| (0.0, 0.0) | 0.001177 / 0.000806 | 0.010324 / 0.007570 | 24 / 24 | 24 / 24 |
| (0.3, 0.0) | 0.101639 / 0.107956 | 0.115002 / 0.104106 | 9 / 4 | 9 / 16 |
| (0.2, -0.2) | 0.037661 / 0.051690 | 0.124016 / 0.119701 | 24 / 24 | 3 / 4 |
| (0.2, 0.2) | 0.041269 / 0.054635 | 0.123041 / 0.128077 | 24 / 24 | 3 / 4 |

At straight .3m/s, settled achieved vx .198366→.192048m/s confirms actual
underspeed, not simply noisy error statistics. Straight signed body-yaw mean
.019616→-.004017rad/s, while absolute body-yaw MAE remains.115002/.104106.
These are body-frame rates, not integrated heading or physical acceptance.
Arc achieved vx also falls: negative.164048→.148501, positive.159732→.145608m/s.

The separately recomputed stage scorer rejects:28 per-world straight-stage
failures, zero arc safety failures and70 paired per-world arc-retention
regressions. Stage exit false, arc safety true, arc retention false.
Turning progression stays closed. No subset/intermediate model selection,
gate relaxation or promotion.

The six retained straight gait traces describe one-contact fractions
96.7593%/96.6435%, two-contact3.1944%/3.3102%, no-contact.0463% in both.
Contact-masked foot-site XY speed means left.013947/.013458 and
right.011796/.012229m/s are similar. These are sampled site/contact proxies,
not contact-patch slip, continuous flight duration or causal evidence for
changing slip/head/contact rewards.

Saved actor normalizers have parent count64,512,000 and both finals70,656,000,
the expected+6.144M. Installed rsl-rl5.0.1 uses epsilon.01 and no count cap.
Yaw-command std parent/control/straight.231978/.231918/.221664; normalized
+.2rad/s command.829737/.829680/.866388. This is a modest input-scaling shift,
not collapse. It is part of the whole-sampler contrast, not proof of causal
action effects; do not freeze normalization based on these statistics alone.

A descriptive pre-action angular-component calculation on4320 settled samples
per arm gives mean XY-rate magnitude.242785/.204687rad/s. Under the existing
training kernel, XY component score is1.736073/1.798482 and yaw component
1.909824/1.924371 (each component maximum2; signed combined reward is their
sum minus2). XY deficit from the stationary maximum.263927/.201518 is larger
than yaw deficit.090176/.075629. This is arithmetic on retained pre-action
states, not the post-reward timing or proof that an alternate gait is feasible.
It motivates a separately predeclared motion-allowance contrast, not relaxing
any evaluation/stability gate or repeating exposure-only/duration sweeps.

Native final CPU verification recomputes complete receipts/checkpoint hashes,
actual dwell, rewards/KL/20 minibatches, standalone cases/aggregate decisions,
paired differences, stage scorer, gait summaries and shared-host bounds.
Mac independently verifies62 payloads (69,604,727bytes) plus the sealed manifest,
recomputes cases/decisions/gates/gait/normalizer arithmetic, and retains
`verified-manifest.json` under
`artifacts/evaluations/straight-lesson-41808a06-20261012`.
The larger retained JSON includes raw gait traces and their aggregate bindings;
weights/ONNX remain native only. No model copy or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Control final999 | `e1547073a05edde109e980b8d8307150ede1980292e2f10657c15106caf6baa5` |
| Straight final999 | `fe5c24f466c1bc3708db75ffb4f961713fe26ec9a58a7d5a4499b51d68df14f0` |
| Full aggregate | `27c13f5c8b8cd7a0e895ccf2077414ffe64c0cff0414acf71b8700b0f4060b2a` |
| Stage-exit decision | `a8b783049ee76109b24083a2f485cf4c214ba50c910891d358c59c8de43eb96e` |
| Gait summary | `f9236bc4e639165ee7e328c5ec07bd61baa19b059c813ad2ec5e16f483b1a794` |
| Saved-normalizer audit | `afe9670dafe8d9c8d68d53b6fc0c05ee502898ca28048428a0bf78c9e1761e80` |
| Native verification | `dd27da260f58a419d49793869b285159dc7bc68763d55db4c3a8aca04ad4abb5` |
| Sealed manifest | `6cd25176e3c5ab15c37589965bc96ea779138a9a95033e3129c935613ef8b112` |

Peak recorded GPU2381MiB/61C; after completion DINO2021 alone969MiB/49C,
all four protected service scopes inactive. Training still exposes right-hip
overruns up to.043416/.030554rad; held-out range passing does not erase those
exploration states. BAM targets remain unclipped. Brief SSH connection timeouts
recovered without restarting/changing jobs, and did not interrupt the retained
service. No MP4, raw perception, physical motion or protected-service restoration.

### Roll/pitch motion allowance: predeclared single-factor foundation trial

The rejected fixed lesson does not justify adopting its sampler or advancing
turning/obstacle/hop curricula. Retained pre-action rates instead motivate a
bounded question: is the existing roll/pitch angular-motion reward discouraging
forward gait? The measured component deficit is a reason to test, not a causal
finding; neither gait feasibility nor improved speed is assumed.

Protocol `integrated-roll-pitch-allowance-continuation-v1`, arms
`control/allowance`. Both restore original c81 parent/model/normalizer/Adam/LR,
fresh episodes and zero clocks, and use the original mixed command sampler.
The **only** treatment is `track_angular_velocity`'s
`roll_pitch_variance=.5→1.0`; yaw_variance stays.5 and weight2. The signed
additive function remains exp(-yaw_sq/.5)+exp(-xy_sq/variance)-1, without
clipping negative rewards. This softens a training motion penalty only;
upright, falls, motor limits, named qpos limits and every held-out acceptance
threshold are unchanged. Do not confuse reward width with a safety gate.

Keep all other rewards (motor-1, named qpos-4, twist-.5, forward0, yaw-bias0),
head/body sampling, DR, plant/BAM,61D actor/14 raw unfiltered actions, stock
observation delays, live normalization, adaptive KL.01/entropy.01 and original
Python LR unchanged. Neither the rejected straight endpoint nor an intermediate
model is an accepted teacher. No frozen normalization, sensor retiming or pose
reward change. Actual command dwell must include idle/straight/both yaw signs.

| Phase | Worlds x updates per arm | Matched seed | Cap |
| --- | ---: | ---: | ---: |
| CUDA-hidden integrated/adjacent checks | no GPU | n/a | before smoke |
| GPU smoke + normalized finite61-to14 ONNX |64x5|1283|600s each|
| Same-world benchmark |256x10|1289|600s each|
| Matched continuation |256x1000|1291|2100s each|
| Final999 unchanged full evaluation |8x240, seeds839/853/857, original4commands|fixed|900s total|

Each arm has6.144M transitions/20,000 minibatches; retain0/250/500/750/999,
evaluate only final999. The original full12-case decision is the advancement
gate, with no stage-only pass replacing it. Preserve straight-case read-only
gait traces for descriptive comparison. Both live reward parameters and
pre-reset signed rewards must match the declared variance, and all inherited
motor/range/twist/forward-zero/yaw-bias-zero/KL/actual-command checks remain
active. The shared observer defaults to.5 for every legacy recipe; only the
new explicitly declared trial selects1.0.

Admission requires clean exact tested/pushed source, same-source native checks,
two successful64x5 smokes with actual coverage/normalized ONNX, and conservative
benchmark projections below2040s per arm. One retained lease-protected user
service runs control, allowance and evaluation sequentially; never overlap
another Duck GPU process. Keep DINO2021 and unrelated workloads untouched and
both protected services inactive in user/system scopes. No automatic next job,
MP4, raw perception, actor expansion, physical motion or policy promotion.

Prelaunch owner review retains the complete training-budget observer chain,
including zero yaw-bias checks and actual command census; it does not use an
angular-only shortcut. Focused new/legacy checks passed31 tests. The owner-run
integrated/adjacent suite passed865 tests with14 existing selector warnings
in27.25s. Native smoke, measured admission and full results remain separate.

### Roll/pitch allowance: completed matched result, not promoted

Exact source `9d2de049584905ac6363fb1eecb5e69658286ff9`, native retained root
`artifacts/training/roll-pitch-9d2de049-20261012`. The preflight service finished
October12 10:35:18 Beijing and the sequential matched/evaluation service finished
11:00:27; both inactive, Result success, exit0, MainPID0. No job was modified
while active. Native CUDA-hidden865-test regression, both64x5 normalized finite
ONNX smokes and256x10 benchmarks passed. Conservative projections1163.994/
1163.282s were below2040s. This is admission, not policy acceptance.

Both arms completed1000 updates,24,000 steps,6.144M transitions and20,000
minibatches, with finite metrics and zero NaN terminations. Control/allowance
elapsed627.624/625.924s; exploration falls450/467, final100 falls34/47. Both
restored the exact original c81 parent/model/normalizers/Adam/Python LR and
started fresh episodes at zero clocks. Retained0/250/500/750/999 checkpoints
are finite and clock/hash-verified. Actual command census includes all eight
idle/straight/signed-yaw bins; straight dwell1,302,945/1,310,597, idle915,433/
927,053. The sole live reward change was roll/pitch variance.5 to1.0.

All24 held-out cases complete240 steps without terminals and pass every motor
and named/all-joint range case. Idle tracking passes all24 worlds. Both arms
nevertheless fail all nine moving-case combined tracking gates:
**foundation-not-ready**. Moving speed passes61/72 to55/72; yaw8/72 to19/72.
This trades negative-turn yaw improvement for straight underspeed, not a new
skill or permission to advance turning/obstacles/hopping/football.

| Command vx/yaw | Speed MAE control / allowance (m/s) | Body-yaw MAE control / allowance (rad/s) | Speed passes /24 control / allowance | Yaw passes /24 control / allowance |
| --- | ---: | ---: | ---: | ---: |
| (0.0, 0.0) |0.001331 /0.000847|0.012449 /0.007063|24 /24|24 /24|
| (0.3, 0.0) |0.099427 /0.105928|0.109245 /0.112361|13 /7|5 /8|
| (0.2, -0.2) |0.042354 /0.046394|0.122171 /0.114592|24 /24|1 /9|
| (0.2, 0.2) |0.047070 /0.045739|0.130982 /0.133458|24 /24|2 /2|

Straight achieved vx.200797 to.194078m/s at .30m/s target; negative-arc vx
.159931 to.153995 and positive.155395 to.155043. Signed body-yaw means are
straight.001391/.007021, negative-.159196/-.171666, positive.134571/.144944;
these are body-frame rates, not integrated heading or physical outcomes.
Straight power1.815261/1.721537W and thermal proxy.031076/.029557 accompany
the lower speed, not proof of improved efficiency at matched achieved speed.

Exploration max torque utilization p99.676274/.687188, soft-limit fraction
.008429/.009068 and thermal proxy.053852/.054290 remain separately recorded.
Right-hip stop exposure reaches.009928/.011556 and overrun.036379/.035983rad;
held-out range passing does not erase these training states. Targets remain
raw/unclipped. All inherited motor/range/twist/forward-zero/bias-zero/KL/actual
command checks remained active, not only the angular observer.

Native final CPU verification authenticates receipts, exact source/runtime,
all checkpoints and clocks, actual census/rewards/minibatches, full standalone
cases and decisions, paired differences, straight gait summaries and host
bounds. Independent Mac verification hashes60 JSON/JSONL payloads
(69,580,228bytes) plus their sealed manifest, recomputes all cases/decisions/
differences/gait summaries, and saves `verified-manifest.json` in
`artifacts/evaluations/roll-pitch-9d2de049-20261012`. Checkpoints and ONNX remain
native only; no duplicate model storage or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Control final999 | `6291dc401b3910cb890b05ea79a766de4f298dfcfe8438aa334727d3e0d9c17a` |
| Allowance final999 | `d222ea25254f3b1c2b25fec2201482f70672233e1ebeea5738544cd6ade06f20` |
| Full aggregate | `0f2664594677cd8aed11b411f67c0167d7dea8724a31e9148a9ac171d24372a8` |
| Gait summary | `8e85528b92a66ae35839e642e8f8a0d9878417c73d05dbc8e0e120fb2d83c198` |
| Native verification | `b43b3643f40285cc9fd3442416fb5495cc89460862ccd7bbd4aeb55d76c52d31` |
| Sealed manifest | `f83fe14063487bb4d1e071d3f2d474190cbb7c6affbbaf692740b801ed3e7d25` |

Peak recorded GPU2381MiB/61C; post-job DINO2021 alone969MiB/48C, all four
protected service scopes inactive. No alternate-checkpoint selection, widened
acceptance gate, MP4, raw perception, actor expansion, physical motion, policy
promotion or protected-service restoration. Restore the original.5 reward for
the next independent initialization diagnostic; do not stack this treatment.

### Adam-state reset: predeclared two-arm initialization diagnostic

After the rejected reward/exposure contrasts, test one training-initialization
factor rather than add another coefficient. The c81 research parent has been
continued repeatedly with its inherited actor, critic, empirical normalizers,
Adam moments/parameter step counters and Python LR. Exact restoration is now
verified; that makes an optimizer-state-only contrast possible. It does **not**
establish that inherited moments caused underspeed, nor that a1000-update
continuation can qualify a fresh upstream recipe. No random-policy arm, LR
reset, normalization freeze, wider roll/pitch reward or sampler change is added.

Protocol `integrated-adam-reset-continuation-v1`, arms `control/reset`.
Both first restore the exact c81 parent actor/critic/normalizers/Adam, reset
episode/curriculum clocks, and keep original Python/group LR
`5.062500000000001e-05`. The reset arm then clears **only the same Adam
object's state mapping**, dropping moments and per-parameter step counters for
both actor and critic. Optimizer/group/parameter identities, options/defaults,
learned parameters and all normalization buffers remain byte-identical; torch
CPU/CUDA RNG remains unchanged by this intervention. The control retains all
restored slots. The first actual PPO updates must repopulate reset Adam, not
merely produce a completion receipt.

Strict initialization rejects missing/partial/nonfinite/wrong-shaped slots,
non-Adam optimizers, wrong model parameter coverage, invalid step counters or
incorrect inherited LR. Mode-aware dispatch is explicit and optional; all
legacy recipes keep their original two-argument warm loader. Metadata separates
verified parent Adam restoration from the reset arm's subsequently empty Adam;
it must not falsely claim active optimizer-state preservation in that arm.
Initial checkpoints independently authenticate model/normalizer equality and
control-populated/reset-empty slots; every final must have finite populated Adam.

Both arms use the original mixed command support/resampling/head/body/DR,
stock delayed61D observations,14 raw unclipped/unfiltered BAM actions, current
installed plant/runtime, motor-1, named qpos-4, twist-.5, forward0, bias0,
signed additive angular weight2/yaw variance.5/roll-pitch variance.5,
adaptive KL.01/entropy.01 and5epochs x4minibatches. No parameter, observation,
reward, acceptance or action-processing change is hidden in initialization.
The observer retains the complete training-budget reward/KL/actual-dwell chain.

| Phase | Worlds x updates per arm | Matched seed | Cap |
| --- | ---: | ---: | ---: |
| CUDA-hidden integrated/adjacent regressions | no GPU | n/a | before smoke |
| Smoke + official normalized finite61-to14 ONNX |64x5|1301|600s each|
| Same-world benchmark |256x10|1303|600s each|
| Matched continuation |256x1000|1307|2100s each|
| Final999 full held-out evaluation |8x240, seeds839/853/857, original4commands|fixed|900s total|

Retain0/250/500/750/999, evaluate only final999,6.144M transitions and20,000
minibatches per arm. The unchanged complete12-case foundation decision governs
advancement; paired speed/yaw, terminals, torque, power, thermal and named/all-
joint range metrics explain differences but cannot replace it. Read-only
straight gait traces remain descriptive. No intermediate selection or seed
sweep rescue. One paired seed is diagnostic, not multi-seed skill qualification.

Admission requires clean exact tested/pushed source, same-source native CPU
regressions, both64x5 smokes with actual command coverage and normalized ONNX,
authenticated initial/final optimizer semantics and both benchmark projections
below2040s. One retained lease-protected sequential user service runs both arms
then evaluation, overall5300s; preflight overall2500s. Preserve DINO2021 and
unrelated work, keep both protected services inactive in user/system scopes.
No automatic follow-on training, MP4, raw perception, actor expansion, policy
publication, physical motion or protected-service restoration. A failed matrix
must be retained and diagnosed before any separate larger fresh-foundation
experiment; it does not prove upstream training infeasible.

Owner review corrected the loader to the installed rsl-rl5 separate actor and
critic API before native execution. Focused CPU tests26/26 and integrated/
adjacent regressions895/895 passed (14 existing selector warnings,42.07s).
The read-only native parent inspection confirms17 populated parameter states,
each step210,000, canonical step/exp_avg/exp_avg_sq and original group LR.
Native regression/smoke/optimizer-checkpoint admission still precedes training.

The first native preflight at source2263c36f passed895 CPU tests, then failed
smoke-control before its first PPO update with `Adam step on canonical device`.
The failed root `artifacts/training/adam-reset-2263c36f-20261012` and service are
retained. Read-only installed loader inspection explains the overly strict
validator: mjlab loads every checkpoint tensor with map_location=cuda:0, and
torch2.9 noncapturable/nonfused Adam load_state_dict preserves the supplied step
tensor rather than recasting it to CPU. Thus restored CUDA step counters are
valid. No optimizer/model/driver change is needed. The corrected read-only
check accepts CPU or the parameter's own device in this mode, requires the
parameter device for capturable/fused Adam, and rejects another GPU's device.
It never moves/restamps the original counter. Reset still changes only the same
state mapping, letting torch recreate its normal fresh slots after real PPO.
A new exact-source unique run must repeat all admission checks; the failed
preflight is not reused or overwritten and admits no learner or skill.

Corrected owner integrated/adjacent regression:900 tests passed,14 existing
selector warnings,30.46s, CUDA hidden. CPU device-policy tests cover mapped
noncapturable counters and the installed torch loader's unchanged-tensor rule.
GPU smoke still has to exercise the actual CUDA-restored state.

### Adam-state reset: completed result and initialization hypothesis not adopted

Exact source `23ad07207eea55198839f56de7a91437fadd6760`, retained native root
`artifacts/training/adam-reset-23ad0720-20261012`. Both services finished
successfully, inactive, MainPID0, exit0. Corrected900-test native CPU preflight,
both64x5 official normalized finite ONNX smokes and256x10 benchmarks passed;
conservative projections1185.939/1191.246s were below2040s.

Both arms completed1000 updates,24,000 steps,6.144M transitions and20,000
minibatches, with finite metrics and zero NaN terminations. Control/reset elapsed
625.030/624.633s; training falls460/481, final100 falls64/34. These exploration
counts are not the held-out stability gate or proof of better convergence.
Actual census covers all eight bins: idle950,593/925,290 and straight1,313,225/
1,303,651, with both signed-yaw bands exercised throughout the unchanged sampler.

Native CPU authentication verifies the exact original c81 models/normalizers
in both initial snapshots, zero episode/learning clocks, original LR and
parameter groups. Control initial Adam has17 populated parameter states;
reset initial Adam is empty. At every0/250/500/750/999 checkpoint both have17
finite states with the expected per-parameter steps. Final control230,000,
reset20,000: the intended intervention happened and real PPO repopulated it.
No parent model, normalization buffer, parameter/group identity, option or RNG
was modified by the reset itself. Fresh-device counter validation preserves
the installed loader's semantics; it does not recast restored control slots.

All24 held-out cases complete their first240-step episode with no terminals,
and both arms pass every motor and named/all-joint range case. Idle passes all
24 worlds. Both still fail all nine moving-case combined tracking gates:
**foundation-not-ready**. Moving speed passes69/72 to52/72; yaw5/72 to16/72.
The reset trades some yaw improvement for substantial straight underspeed.
Do not adopt the treatment, select another checkpoint or weaken any gate.
This one-seed diagnostic does not establish a general optimizer-state cause.

| Command vx/yaw | Speed MAE control / reset (m/s) | Body-yaw MAE control / reset (rad/s) | Speed passes /24 control / reset | Yaw passes /24 control / reset |
| --- | ---: | ---: | ---: | ---: |
| (0.0, 0.0) |0.000846 /0.000842|0.009043 /0.007026|24 /24|24 /24|
| (0.3, 0.0) |0.085672 /0.108217|0.124844 /0.105011|21 /4|3 /11|
| (0.2, -0.2) |0.030813 /0.044734|0.148046 /0.129870|24 /24|0 /1|
| (0.2, 0.2) |0.033037 /0.048152|0.122626 /0.118844|24 /24|2 /4|

Straight power1.938401/1.675156W and thermal proxy.032913/.027522 accompany
the worse speed, not matched-speed efficiency gains. Training max torque
utilization p99.673939/.672019, soft-limit exposure.008254/.008150 and thermal
proxy.054119/.051573 are retained separately. Right-hip stop exposure reaches
.013835/.011068 and overrun.031955/.034724rad; held-out passing does not erase
those exploration states. BAM targets remain raw, unfiltered and unclipped.

Native final CPU verification authenticates all preflight/full receipts,
initial and durable checkpoint semantics, source/runtime, all2030 finite
update rows, actual dwell/rewards/KL, full cases/decisions/differences and gait
summaries. Mac independently hashes60 JSON/JSONL payloads (69,584,105bytes) plus
the sealed manifest, recomputes all cases/decisions/differences/gait summaries,
and saves `verified-manifest.json` under
`artifacts/evaluations/adam-reset-23ad0720-20261012`. Weights and ONNX remain
native only; no duplicate model storage or Git asset upload.

| Retained object | SHA256 |
| --- | --- |
| Control final999 | `391ffb64c89eb14870381584147f91ba235efc6bf1a3d95cc5b5b44d93cc49d3` |
| Reset final999 | `2528bb1f8aa142904b671ae9b0f2039f937361123bafc9d36dbf1b63cde682ea` |
| Full aggregate | `51947d814ead2d33cc7b2acde06e0309ae191e073876e62c3073787086f60835` |
| Gait summary | `8fd62c3235daaece1e46fbd40e85b8a395fb9f1842ab2d680b78f8adf5745a1b` |
| Native verification | `3329a2ab08e78a19cd9cf4a6ac8c015e720c703a468d159d6aefb4a1be3415e3` |
| Sealed manifest | `c14cc475f085c39b0db7a4fcb3ca72ca54040d132dca2ac48d9d43800cf8b397` |

Peak recorded GPU2381MiB/62C; completion DINO2021 alone969MiB/48C, protected
services inactive in all four scopes. No promotion, MP4, raw perception, actor
expansion, physical motion or protected-service restoration. Stop coefficient/
reset stacking on the same unaccepted parent. The next independently declared
fresh foundation is a new stock-curriculum baseline, not a causal comparison
with these collapsed-curriculum warm continuations; see its separate declaration.
