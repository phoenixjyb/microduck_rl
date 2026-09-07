# F1-Y: predeclared continuation yaw-command support

Status: **implemented and locally tested; not launched**. Two of three paired
pilot slots have been consumed. This is the final possible pilot this window;
do not launch it simply because time remains or rerun a failed service.

## Evidence and hypothesis

F1-N's complete closed evidence and exact hashes are in
[its ledger](2026-09-08-f1n-neck-smoothing.md). Its original22-label rejection
remains final. Its actual neck regularizer was live, but improved sampled
action smoothness did not satisfy motor or speed gates.

The read-only command audit found that continuation used exactly zero yaw
while the deployed held-out heading controller consumed nonzero yaw in every
settled sample. Hypothesis: preserving yaw-response command support during
continuation may improve performance under the unchanged heading controller.
This is not established root cause, an assertion of full-parent out-of-domain
behavior, or permission to modify the evaluator.

## One changed axis, fixed controls

Start both arms independently from unaccepted narrow8498 SHA
`7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`,
restoring all learned/Adam/learning-rate state and common step204000.
Do not continue from either rejected F1-M or F1-N final model.

Use F1-N control settings in both arms: motor weight/live stage-4,
neck-action-rate-0.1, lateral cost-0.5, global action-rate-1, linear tracking
weight4.5/variance0.05, identical robot/dynamics/observations/head-body commands.
The **only configuration contrast** is twist `ranges.ang_vel_z`:

- Control: `[0,0]`.
- Yaw-support: `[-0.35,+0.35]` rad/s, the existing evaluation controller's
  declared command cap, not an interval fitted to favorable rollout samples.

Use the existing uniform body-command sampler unchanged. Keep forward0.3m/s,
lateral0, heading/world/forward-only/turn-in-place modes disabled, resampling
interval1e6s and existing episode-reset sampling. Commands are therefore
sampled per episode, not a new closed-loop training controller. This is marginal
command support, not a claim to match the controller's temporal distribution.
No extra mixture probability, yaw reward, reward-weight/physics changes,
head freeze, observation expansion or episode-length extension.

## Fixed protocol and stop policy

Intended runner `mjlab_microduck.foundation_yaw_experiment`, new output
`artifacts/experiments/f1y-yaw-support-paired-s499-v1`.
Reuse development seeds491 for paired10-update smokes and499 for paired
500-update pilots,256env,24steps/update, checkpoint cadence50. Fixed final8998,
common step216000; all initial learned snapshots identical. Child-start hash
equals training/evaluation seed; matched backend fingerprints. No seed hunt,
best-checkpoint selection, extra arm, retry or longer optimization.

Require fresh absolute-safe parent references503/509/521 first. After both
smokes and pilots finish, evaluate control then yaw-support fixed8998 at503,
509,521, stopping at the first failed pair. Use the exact current400-step,
8env,100startup/300settled, heading-hold, command delivery, raw motor/route
retention and **all** original speed/window/lateral/heading/per-joint/pooled
torque/squared-load/power gates. No special forgiveness for the control or
for a treatment that learns turning at the expense of straight motion.

## Required implementation and live evidence

Before launch, focused tests must prove the config differs only in the yaw
range; the original sampler produces both yaw signs in treatment and exact
zero in control; actor command observations receive the sampled command;
curricula/reset do not silently overwrite bounds or enable heading/forward
sampling; and the body/head commands and61D/14D interfaces remain unchanged.
Adapt the training command guard only through an optional explicit validator;
default fixed-command behavior remains unchanged for historical callers.
Retain per-update actual command range/sign counts, declared versus live
reward weights and the existing read-only neck reward evidence. Do not
accept configuration-only proof that yaw actually reached the actor.

Preserve fixed restoration/cadence, finite reward/gradient/optimizer guards,
fall/motor/thermal gross guards, first-failure sequencing, immutable output
and hashes. CPU tests, clean pushed exact branch, frozen runtime, old sentinel
hashes, idle GPU and inactive protected SYSTEM services are independent gates.

One sequential retained user service on100.100, hard cap3600s, stop timeout30s;
children180s per smoke,900s per pilot,120s per evaluation,60s internal reserve.
Entire cap plus90s must fit before September8 07:00 Shanghai; no GPU launch
after06:00. Preserve100.98/FilmBrain and unrelated workloads. After this slot,
continue CPU evidence and curriculum/retention work only. No harder obstacles,
H2, MP4, raw perception, physical motion or claim of accepted hopping.

## Implementation gate evidence

`foundation_yaw_experiment.py` retains the F1-N campaign's fixed sequence,
same-parent restoration, full checkpoint/counter validation and original
numerical gates, with the single command-range contrast above. It also pins
the inspected upstream uniform sampler and PPO implementation and verifies
the complete closed F1-N manifest before each child. Historical runners and
closed evidence are not rewritten or rerun.

The optional `YawCommandObserver` validates live sampler modes/ranges and
reward weights, compares raw actor input against the current sampled command
before PPO acts, checks the original PPO transition afterward, then verifies
physics entry uses that same command. It does not refresh observations,
sample commands/noise, change actions or alter RNG state. After a step, it
requires commands to remain identical in every non-reset environment. Each
24-step window retains actual yaw min/max/sign counts and reset counts in
`command-activity.jsonl`, with exact aggregate reconciliation before evaluation.
Both signs must actually reach the treatment actor; control must remain zero.

Local regression890 passed,3 existing actuator/site warnings. Focused tests
exercise the original command sampler and installed observation manager on
CPU (synthetic sensor data, no physics), including episode reset sampling,
same-step actor/transition delivery, unchanged default fixed-command guard,
one-sided/stale/misconfigured input rejection, and first-failure sequencing.
The actual GPU smoke remains a separate gate. Remote CPU tests, pinned
runtime/history, pushed clean exact source and idle-host checks are still
required before the one authorized slot3 launch.
