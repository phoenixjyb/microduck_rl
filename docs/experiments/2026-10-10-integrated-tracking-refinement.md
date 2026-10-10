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
