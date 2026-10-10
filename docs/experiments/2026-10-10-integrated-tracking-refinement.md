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
