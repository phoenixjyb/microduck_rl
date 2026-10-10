# Integrated motor-aware continuation

## Declaration before execution

The fresh upstream-plant foundation completed but did not pass its unchanged
tracking/motor gates. This is a repair experiment on that same locomotion task,
not admission to obstacles, hopping or football balance. No raw perception,
physical motion, policy publication, driver/dependency changes, or protected
service stop/restart. Preserve the exact GroundingDINO worker and unrelated work.

Continue only parent `model_1499.pt` from
`artifacts/training/turn-foundation-791f27c4-20261010/foundation`, SHA256
`b4c51e072b4b21c865ab4108c477ed4c27f26a129577e7ad17a14b1e6250dd90`.
Strictly restore actor, critic, both observation normalizers, distribution and
Adam state; verify value equality against the parent. Synchronize PPO's Python
learning rate with the restored Adam rate. Reset iteration and common step to0;
fresh episodes, no simulator-state continuation. Both arms independently load
the parent, never the other arm's checkpoint.

Pin the source's last consolidated stage below36000 steps before constructing
the managers: action-rate weight-.8, idle fraction.15, head-bias weight2,
head ranges±(.39,.39,.49,.11), body ranges±(.005,.005,.005,.05,.05,.05),
trunk/head CoM DR±.01m. Clear inherited schedules. In particular do not advance
the unconsolidated1500-update source endpoint on the first warm-start reset.
Keep twist sampling, rewards, architecture, BAM plant, DR/noise, learner and
61→14 contract otherwise unchanged. No disabled upstream yaw fix is enabled.

Matched arms: control versus motor-cost treatment, identical seed863,
256 worlds,1000 additional PPO updates,24 steps/update (6,144,000 transitions
per arm). Both contain the same existing `mdp.motor_torque_load_cost` term:
mean over14 actuators of u²+4·relu(u-.70)², u=|force|/.60Nm clipped at2.
Control weight0 throughout. Treatment starts0, then weight-.25/-.50/-1
strictly after100/300/600×24 control steps. Apply the staged weight to the
live RewardManager before each action, including steps without episode resets.
This one-factor contrast tests motor cost, not a new yaw objective. More
practice may help turning in either arm; no learning outcome is promised.

Sequential preflights for both arms:64 worlds/5 updates/seed859;256 worlds/
10 updates/seed861. The motor smoke compresses penalty transitions to1/2/3
updates to exercise every weight; the motor benchmark holds weight-1 to time
the active cost. These are separate disposable learned states, not parents.
Require same-source successes, strict parent restoration and official normalized
ONNX CPU61→14 finite inference. Each10-update benchmark×100×1.5 must fit2040s
before either1000-update arm. External caps:180s/preflight,2100s/arm,
180s/final evaluation. One retained sequential service for control, motor, then
evaluation; overall4500s cap, RAM8GiB, CPU200%, TasksMax64, Nice10,
KillMode=control-group. No overlapping Duck learners or other GPU workload
beyond the retained DINO allowlist. No automatic retries or budget extension.

Entry point: `python -m mjlab_microduck.integrated_motor_refinement <mode>
--source <exact-clean-commit> --run-id <unique-id>`. Modes:smoke-control,
smoke-motor,benchmark-control,benchmark-motor,control,motor,evaluate.
Retain results under the native integration worktree's `artifacts/training`.
Commit and push declaration/code/tests before any GPU execution; no venv sync.

Use the existing owned cooperative GPU lock and exact GPU UUID/worker checks.
All four AI Mission user/system scopes must remain inactive. Preserve the
existing≤6GiB aggregate/≥8GiB free/<75C bounds;20% Torch allocator fraction is
not a Warp cap. Recheck every25 updates and before/after phases. Keep existing
finite observation/action/loss/model/Adam, command-consumption, NaN, gross
motor and durable fsync/atomic checkpoint checks. Save initial, every250th
iteration and final999, including the official normalized ONNX sidecar.

Log per-update falls, both command signs and exact idle, torque p99, soft-limit,
rated-speed exposure, power/thermal proxies plus actual motor weight range,
pre-reset unbounded stream cost and weighted reward. Verify the weighted term
is finite/nonpositive and equals the existing clipped cost from the same
pre-reset forces. These control-step quantities are not substep peaks or
hardware thermal certification; reward-manager sanitization is not raw-term
finiteness proof.

## Fixed decision and stop

Evaluate only final999 for each arm: unchanged foundation protocol, seeds
839/853/857,8 worlds,first240 steps (4.8s), startup60 excluded only from
tracking. Idle0/0,straight.30/0,moving.20/-.20 and.20/+.20. Neutral posture,
no push, fixed commands, official normalizer, stop case at first terminal.
That is24 cases and at most46,080 held-out transitions. Reuse the existing
deterministic decision without relaxing thresholds: every world speed MAE
≤.10 (idle≤.03), yaw MAE≤.10, torque p99 utilization≤.60, zero rated-speed
exceedance and zero terminals. No best-checkpoint fishing or interim promotion.
Report both decisions and same-case differences; single training seed is a
diagnostic, not statistically robust causal proof. A stationary policy cannot
pass the moving-speed goal. Stop after retaining the comparison; no new skill
job follows automatically. Policy/simulator/physical acceptance remainsfalse.

## Tested source, preflights and retained launch

Executed source `4da928983af61f9b509a579a553cc7c1904abe28` was committed and
pushed before GPU execution. Native integration checkout is clean/detached at
that exact source; historical research checkout remains clean on
`feat/athletics-obstacle-curriculum` at`3b32468fd4d1bfc417dd4aa048834c2bddd98ef7`.
Independent read-only review identified the unpersisted PPO Python LR hazard;
the loader now synchronizes it with the exactly restored Adam state. Review
also prompted explicit smoke-before-benchmark/control-before-motor guards and
retained same-case deltas. No worker edited the implementation.

Focused Mac tests passed63 checks before the final paired-report addition;
the final refinement-only rerun passed12 tests. Native exact-source checks
passed64 tests in5.44s, with one existing actuator/site selector warning.
Syntax, whitespace and relative experiment-link checks passed. No dependencies
were installed or changed. HTTPS push failed authentication; the already
authenticated GitHub SSH route successfully pushed the exact integration
branch, without changing credentials or configured remotes.

Retained preflight unit `microduck-motor-4da92898-preflight.service` finished
Resultsuccess/ExecMainStatus0/MainPID0 (active/exited, not a running job).
Both64-world smokes completed5 updates and exact-parent restoration. Both
official normalized ONNX exports passed CPU61→14 finite inference. A warning
on the *initial pre-learning* save concerned the logger not yet having
`logger_type`; final trained exports existed and passed the mandatory check.
It did not silently waive an export gate. The compressed motor smoke observed
every declared weight and negative weighted reward when active, including
approximately-.17317 at weight-1. Both parent Adam rates were2.25e-5 and
both restored learning clocks werezero.

256-world control/motor benchmarks completed10 updates in7.374844/7.356516s.
Conservative projections are1106.23/1103.48s, below2040s. Models/Adam and
losses were finite, with zero NaN terminations. These are admission smokes,
not motor-envelope or learned-skill acceptance. All four preflight result
receipts were copied to Mac; hashes match native bytes:

| Receipt | SHA256 |
| --- | --- |
| Smoke control | `f764eca1e07c3def886e5289db53ed39a1eb5fc0a8c86dc33b5bcbd1c40ec884` |
| Smoke motor | `bdc8a004d9758318daa218289b003f9558f19171813dc07e66068687d659c935` |
| Benchmark control | `2bc7ffd82d86566eb6e4b5fd8233b9591081d2592f6bd26e7296a816f30025e9` |
| Benchmark motor | `cbfd0603e4cdfd2a1fc566171163e901d0c61385d0f86db455006e614f5c5f65` |

Matched service `microduck-motor-4da92898-matched-eval.service` started
2026-10-10 12:45:49 Asia/Shanghai. Live ExecStart verified sequential
control→motor→evaluate with `set -e`, independent GNU timeouts2100/2100/180s
and15s kill grace. Live service properties confirm overall4500s, RAM8GiB,
CPU200%, TasksMax64, Nice10 and KillModecontrol-group. No subsequent job
follows evaluation. Data stay under native
`artifacts/training/motor-refinement-4da92898-20261010`.

At46 control updates/1104 steps, losses remainedfinite, no NaN termination,
zero falls in the latest6144 transitions, torque utilization p99≈1.06754,
soft-limit exposure≈.0850. Latest retained health showed GPU1710MiB/54C,
the current Duck process746MiB and unchanged DINO1592/946MiB. All four
protected service scopes remainedinactive. Training and both held-out decisions
are pending; control is not penalized for motor load by design. Do not infer
motor-aware improvement before its own arm and the fixed evaluations finish.

## Completed comparison and recovery

The pending launch snapshot above is superseded by the completed receipts.
After SSH access returned, the retained service reported Resultsuccess,
ExecMainStatus0, MainPID0 and NRestarts0, finishing at2026-10-10 13:06:46
Asia/Shanghai. Active/exited is the retained service state, not a running
learner. Both arms completed1000 updates/24000 steps/6,144,000 transitions
each (12,288,000 total), in572.618s control and569.611s motor. No new training
or GPU evaluation was launched during recovery.

Both final999 checkpoints match their result receipts, contain finite payloads
and17 Adam states at step50000, and retain iteration999/common step24000.
All1000 update rows per arm are finite, with zero NaN terminations. Training
falls were not zero: the last100 updates recorded151 control and145 motor
falls over614,400 transitions per arm. Maximum retained GPU temperature was
60C control/59C motor, with aggregate memory no higher than2852MiB.

The unchanged deterministic decision is **foundation-not-ready for both arms**.
All24 held-out cases completed240 steps with zero terminals and zero rated-speed
exceedance. Motor-envelope cases passed0/12 control versus9/12 treatment;
all three treatment failures were straight.30m/s cases (torque p99 utilization
.62247–.64967, above.60). Tracking failed11/12 control and12/12 treatment.
The control's one tracking pass was seed857 idle, which still failed motors.
No case admits the next skill stage.

Descriptive means below aggregate the three evaluation seeds/eight worlds;
acceptance uses every world, not these means. Control torque p99 utilization
was1.06754 in every case. Startup exclusion applies only to tracking.

| Command vx / yaw | Control speed MAE | Motor speed MAE | Control yaw MAE | Motor yaw MAE | Motor torque p99 utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 | .00524 | .00866 | .03756 | .07584 | .54855 |
| .30 / 0 | .10754 | .10651 | .24885 | .33083 | .63615 |
| .20 / -.20 | .04916 | .06371 | .28636 | .27345 | .56163 |
| .20 / +.20 | .06063 | .06171 | .27879 | .34107 | .55832 |

Across the12 matched cases, mean squared-torque thermal proxy decreased
from.101950 to.033041 (67.6%), and sampled soft-limit exposure from.071081
to.001916 (97.3%). These are control-step simulation proxies, not measured
hardware temperature or substep peaks. Power did not uniformly improve, and
yaw tracking worsened for idle, straight and positive-yaw case means. This
supports motor-load reduction in this single-seed diagnostic, not successful
turn learning or robust multi-seed causal proof.

### Durable evidence

Native checkpoints and exports remain under
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/motor-refinement-4da92898-20261010`.
Only224KiB of receipts/evaluation JSONs were mirrored on Mac under
`artifacts/evaluations/motor-refinement-4da92898-20261010`; no large checkpoint
was duplicated. Exact SHA256 bindings:

| Artifact | SHA256 |
| --- | --- |
| Control model999 | `a233d4b95bfc786c9e28a161e353a646d339868bc13718e898c71460c90e83a6` |
| Motor model999 | `ee6d0f69e34fd73c34f3b1f307a2d4bde85f73d1ed81a53cba07ca5899091fc1` |
| Control result | `e3132ab1c3938276d8183ed9b5c5362d912464693849ff087d4f2fd452cfc41e` |
| Motor result | `6c3caf5b2809b0bea12bc6233e0ceec1274d3e1f6795c09ef0816be1600cb589` |
| Evaluation result | `49a9662ba01a8b3b56477b62070225a98e0d3cde48452b4b636babeeed961a65` |

Recovery checks bound both arms to exact executed source4da92898, parent
SHA256, seed863 and declared counts. Every evaluation row binds to its arm's
final checkpoint. All24 standalone case JSONs equal their aggregate entries;
CPU recomputation of both decisions and all paired differences exactly matches
the retained report. Mac receipt hashes match native bytes. CUDA-hidden native
checkpoint inspection verified saved model/Adam finiteness. Documentation-only
closeout does not require retraining or a dependency change.

At2026-10-10 17:55:31 Asia/Shanghai,100.100 was reachable, with no Duck GPU
process,47C GPU temperature and15232MiB free. Only the preserved GroundingDINO
worker PID1592 used946MiB (961MiB aggregate). All four protected AI Mission
service scopes remainedinactive. The access outage did not lose the experiment;
no outage cause or reboot is inferred.100.98 and the team's NX workload remain
untouched.

Stop here as declared. The next recommendation is a separately predeclared,
matched commanded-turning/speed-tracking refinement that retains motor pressure
and these gates, with focused tests and bounded smoke before another full run.
No new skill job follows automatically; obstacles, hopping, football balance,
video admission, raw perception and physical motion remain outside this result.
