# Locomotion before large head motion: matched lesson, predeclared

Branch `integration/upstream-2026-10-10`; protocol
`integrated-motor-range-head-lesson-v1`; implementation
`src/mjlab_microduck/integrated_motor_range_head_lesson.py`.

## Why this lesson

The [linear-priority contrast](2026-10-13-motor-range-linear-boost.md) is sealed
and rejected: moving speed55→58/72, moving yaw0/72 in both arms, both turning
speed means worse. The [action release](2026-10-13-motor-range-action-release.md)
and yaw contrasts also failed. Stop the scalar sweeps; do not seed new training
from their rejected endpoints or infer motor-safe gait from a favorable mean.

Try one curriculum-complexity factor: small secondary head goals before large
head motion while preserving **all** locomotion command coverage. Large head
targets may compete with learning gait, but existing results do not establish
this cause. This is a diagnostic lesson, not a proven explanation or skill.

## One factor and unchanged teacher state

Both arms warm-load original unaccepted936a:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA256 `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Verify original iter999/env24000 and actual models, normalizers43008000,
17 Adam states140000 and LR restored identically; restart learner/env clocks0.
Materialize all seven source curricula at final conditions first, then change
only `head_pose.ranges` before live managers build their configuration copies.

|Actor head slot, HOME-frame radians|Control final range|Quiet-head lesson|
|---|---:|---:|
|Neck pitch|±1.1|±.05|
|Head pitch|±1.1|±.05|
|Head yaw|±1.4|±.07|
|Head roll|±.31|±.015|

Quiet ranges are the exact existing velocity-recipe beginner ranges. Four
nonzero command slots remain alive; zero-command probability0 and resampling
2–5s are unchanged. No zero padding, command slot deletion, sampled-value clamp
or manual resampling. Hold each arm's range fixed for this whole bounded run:
no calendar widening, no extra RNG, no inherited curriculum replay. Evaluation
already commands head/body zero in both arms; keep its full protocol unchanged.

Linear tracking returns to stock XYZ+2/stdsqrt(.1), not rejected boost2.5;
action-rate-1, stock XYZ angular+2/stdsqrt(.5), trunk-angular-.05, whole-body
momentum-.02, motor-1 (.60Nm/.70soft/overload4), bilateral named hip-yaw
qpos-4/.15rad, forward-error weight0 remain identical. Head tracking+2/std.5
and self-negating DC bias+3/tau1s stay identical too; no sign change or reward
removal. Standing share.25, eight-bin locomotion command sampler/dwell, body
goals, CoM/head-CoM DR, observation noise/delays, BAM plant and PPO are fixed.
Actor raw61→raw14 remains unfiltered/unclipped; normalization stays on and only
the official baked-normalizer export is used. No perception or actor expansion.

## Direct lesson evidence and fixed interpretation

Before each action, verify the actual live head-command configuration/callable
reward parameters and weights, finite world×4 actor command and componentwise
range bounds. Capture min/max/absolute means and nonzero counts before stepping,
resampling or autoreset; consume only after exactly one env step. Retain24-step
and world counts, same live ranges in curriculum profile, and finite summaries.
Read the completed-step weighted head-tracking and head-bias reward matrix;
**never recompute the stateful EMA** or pretend weighted/weight is independent
raw-cost evidence. Keep both arm snapshots equally read-only. Runtime, native
closed-output and independent portable validators check bounds/counts/signs.
Actual linear/action, angular, optimizer, motor, thermal, range and fall metrics
remain in the same rows. These distinguish lesson exposure and reward mass from
behavior; they do not establish causality or transfer.

|Phase per arm|Worlds × updates|Matched seed|Child cap|
|---|---:|---:|---:|
|Normalized61→14 ONNX smoke|64×5|1553|600s|
|Measured benchmark|256×10|1559|600s|
|Control / quiethead continuation|256×1000|1567|2100s|
|Final999 evaluation matrices|8×240 per case|839/853/857|900s combined|

Full native CPU checks and both smokes/benchmarks precede either learner.
Per-arm projection `benchmark_elapsed*100*1.5<2040s`, no cap widening.
Each learner6144000 transitions/20000 minibatches; final counts49152000 and
Adam160000. Retain initial plus0/250/500/750/999; evaluate **only final999**.
Run control→quiethead→both matrices sequentially with the retained user services.

All twelve cases per arm and original per-world gates are unchanged:
seeds839/853/857 ×idle/(.3,0)/(.2,-.2)/(.2,.2), eight first-episode worlds240
steps; omit60 startup steps from tracking only. SpeedMAE≤.03 idle/≤.10 moving,
yawMAE≤.10; no terminal or rated-speed exceedance; pooled per-case torque p99≤.60;
all14 hard margins≥-.005rad, right-hip stop fraction≤1%. Keep denominators and
partial failures; do not average out a failing world or change the combined gate.

Report all groups' speed/yaw passes/MAEs, achieved vx/signed yaw, idle retention,
falls total/first100/last100, torque/power/thermal and range/stop, plus actual head
exposure/reward mass. Describe a moving-tracking benefit only if all three moving
speed MAEs fall without yaw-MAE regression and no paired held-out motor/range/
stability regression. This descriptive rule waives no gate. A numerical gate
pass still requires independent-seed confirmation and transfer/replay checks;
diagnostic policy/simulator/physical acceptance remains false.

## Curriculum roadmap, not automatic promotion

1. This small-head lesson tests idle/straight/mirrored-turn locomotion together.
2. Only after all unchanged motion/safety gates and independent-seed confirmation,
   predeclare expanding the existing head ranges gradually while retaining gait.
3. Qualify stop/start and moving-turn transitions with explicit retained traces.
4. Add compact externally supplied obstacle range/bearing/size/closing rate;
   slowdown is allowed inside avoidance, speed tracking before/after remains.
5. Retain stabilization/hop transfer checks before combining skills; rolling-ball
   support follows its independent plant-feasibility curriculum.

Only this matched lesson is admitted for execution by this document. No automatic
range expansion, harder skill, rejected teacher, longer run, video, raw camera,
physical robot motion or publication.

## Shared-host safety and acceptance levels

100.100 native checkout `/home/converge/work/microduck_rl-upstream-20261010`,
exact clean tested/pushed source. Fresh UUID/DINO PID2029/exe/uid and original
service context must be checked. Pause/restore only originally active user
`recomo-ai-mission-vllm.service`; other three protected scopes inactive. Preserve
DINO/unrelated workloads and100.98/NX/FilmBrain. Require owned lease, used≤6144MiB,
free≥8192MiB/temp<75°C, Torch20%, CPUthreads1, memory8GiB/CPU200%. Nonenabled
preflight45min/matched95min/outer150min retained services, childgrace15s. Restore
only after no Duck child/GPU process, leasefree and DINO-only proof, then verify
GPU cgroup ownership; unexpected entrants cause refusal, not overlap or stops.
No driver/package change. Source/tests, native admission, learners, held-out
motion and physical outcomes are separate. Native source stays frozen while active.

Source preparation, native admission and results will be recorded below; none
is established by this predeclaration alone.

## Source preparation evidence

CUDA-hidden single-thread119 focused checks passed in8.40s, covering this lesson
and inherited linear/action recipes: paired configuration differs only in head
ranges, parent/contracts/gates stay fixed, live configuration/reward drift is
refused, pre-action commands survive reset-side mutation, actual signed head
reward rows and24-step/component counters reconcile. Independent read-only
source/declaration/operations-builder review cleared; owner reviewed the test
diff and corrected copied fixture assumptions before executing it. Installed
mjlab1.3 primary source confirms `_step_reward` survives reward-manager reset.

Portable stdlib arithmetic checks2 passed including range/count/sign/nonfinite
fault cases; all12 generated-operation templates compile in a no-write dry run.
Document links and whitespace checks pass. Native complete scoped CPU regression,
smokes, projections and actual learner/evaluation evidence remain mandatory and
pending; source success alone admits no motion capability.

## Native admission passed; matched training active

Exact clean tested/pushed native source
`4de0017d59658ea00d5fe32ab3a352d542029f32`;1171 native scoped regression
checks passed36.70s with16 existing warnings. Both normalized61→14 ONNX smokes
passed4.707/4.614s with1/0 training falls, zero NaNs. Benchmarks8.107/8.089s,
5/3 falls, projections1216.109/1213.351s below2040s. Actual identical initial
SHA256 `d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`
confirms original936a models/normalizers/Adam/LR restored in all four phases.
Closed-phase validators passed all30 action/linear/head rows; all four head
slots had nonzero actual dwell in both arms. Both retained linear2/action-1.

At final benchmark row, actual control head absolute commands were
[.565076,.564077,.731109,.155352]rad versus
[.025353,.025863,.036927,.007600]rad for quiethead. Weighted head tracking
1.543259→1.648124 and bias-.546710→-.413717 verify lesson/reward exposure, not
locomotion improvement. Zero RNG/reward/filter changes are introduced by capture.

Admission proof `a9526d3aeac047729ad0025177e9e4e4a6443f0b8cea5eba4624c93bbe86b6f7`;
original service context `bf50b5e734ee06b30aab2965ebf11928f7ef65be86b272f4fe523e4185b98f96`.
Closed JSON-only Mac packet
`artifacts/evaluations/motor-range-headlesson-4de0017d-20261013-preflight`;
owner authenticated both hashes and repeated all30 source row validators.
Source/ops reviews, shell/Python/systemd validation,3 mock restoration tests
including7 refusal cases and2 portable arithmetic tests passed before launch.

Run `artifacts/training/motor-range-headlesson-4de0017d-20261013`, services
`microduck-headlesson-4de0017d-matched-eval.service` and
`recomo-duck-headlesson-4de0017d-handoff.service`: control→quiethead→both final
matrices. First retained audit: control141/1000,201 training falls, zero NaNs,
finite rewards/optimizer, proper full ranges; GPU1710MiB/63°C, DINO2029 retained,
all four protected scopes inactive during the authorized user-vLLM pause.
Native source4de0017d is frozen while active. Learner outcome, all final
checkpoints, numerical evaluation/seal and original-service restoration remain
pending. Do not infer capability or promote harder skills from this admission.
