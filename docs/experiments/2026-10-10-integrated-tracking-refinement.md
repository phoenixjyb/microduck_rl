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
