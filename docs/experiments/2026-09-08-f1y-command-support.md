# F1-Y: predeclared continuation yaw-command support

Status: **closed, numerical rejection at first pair503**. All three paired
pilot slots have been consumed. No further training or GPU experiment is
permitted this window; continue bounded CPU retention/compatibility work only.

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

## Retained closeout, September8 02:06 Shanghai

Source `2614d09a994223a025d7a0d17d5b6879c7b513dd`; service
`microduck-rl-f1y-2614d09-s499.service` ran01:42:03–01:55:08 Shanghai,
normal exit0. Exit0 means the campaign produced a valid decision, not acceptance.
The retained unit is active/exited with MainPID0. At02:05:48 the GPU had no
compute PID,0% utilization,45C and12MiB; both protected SYSTEM services remained
inactive. No service was changed or restarted during closeout.

Both10-update smokes and500-update pilots finished. Pilot elapsed times were
306.108s control and307.387s yaw; each retained11 post-update checkpoints through
8998/common step216000. All26 post-update and four initial checkpoints were
verified for exact cadence, counters, finite tensors and restoration evidence
on both hosts. Both pilots' maximum sampled fall fraction was0.00390625,
maximum temperature58C and maximum sampled training torque p99 was0.720211/
0.729346. These are gross training guard observations, not evaluation acceptance.

The live command audit reconciles12,000 steps per pilot: control consumed
3,072,000 exact-zero yaw samples; yaw consumed1,528,537 positive and1,543,463
negative samples, no zero samples. Actor/transition/physics-entry equality and
unchanged non-reset commands passed. The original neck reward remained live
at-0.1 in both arms. This validates delivery, not the causal hypothesis.

Exactly five original evaluation reports exist: parent503/509/521 and
control503/yaw503. All are absolute-safe400-step rollouts. The first paired
decision is `numerical-gate-stop`; candidate509/521 were not run.

| Settled503 metric | Parent | Control | Yaw support |
| --- | ---: | ---: | ---: |
| Body forward m/s |0.266276|0.264220|0.259175|
| Route forward m/s |0.264949|0.263178|0.258836|
| Absolute lateral m/s |0.076845|0.060435|0.056216|
| Maximum heading rad |0.220906|0.135438|0.107919|
| Pooled pre-reset torque p99 |0.554892|0.562075|0.556649|
| Squared utilization mean |0.039622|0.041084|0.038988|
| Absolute mechanical power W/motor |0.136729|0.151850|0.142821|
| Soft-limit fraction |0.001280|0.001756|0.001012|

The exact16 failed labels were independently reconstructed from original raw
motor/route/command evidence on both hosts:

```text
straight-body-mean-outside-band
cross-route-motion
body_forward_per_env_mean-nonregression
route_forward_per_env_mean-nonregression
joint-torque-nonregression:left_hip_yaw
joint-torque-nonregression:head_yaw
joint-torque-nonregression:right_hip_yaw
matched-control:body_forward_per_env_mean-nonregression
matched-control:route_forward_per_env_mean-nonregression
matched-control:joint-torque-nonregression:left_hip_yaw
matched-control:joint-torque-nonregression:left_ankle
matched-control:joint-torque-nonregression:right_hip_roll
matched-control:joint-torque-nonregression:right_hip_pitch
body_mean_in_band_all_envs-failed
route_mean_in_band_all_envs-failed
stable_route_window_all_envs-failed
```

Sampled heading/lateral/load improved versus control, but speed worsened and
several named joints regressed. No averaging away these failures, causal or
generalization claim, new confirmation seed, retry, or policy promotion.

Remote evidence: `artifacts/experiments/f1y-yaw-support-paired-s499-v1`;
Mac mirror: `artifacts/diagnostics/f1y-yaw-support-paired-s499-v1`.
All105 payloads (179,459,039 bytes), plus manifest, verified on both hosts.

| Artifact | SHA256 |
| --- | --- |
| Manifest |`63a9c8c5cd8a909a1f7cc1490deff60978368c22bbfd2e314985b667febfbe41`|
| Decision |`142723cf3b0211920e22d7c3d3a6f65478c7dd1cf103c099ff39b74554441495`|
| All four initial learned states |`72d232b192e363d602c08108d643cce1a639f8d5885e0ff393b51ce05010756b`|
| Control fixed8998 |`a1828f89db7fa87380fdeaa8c0e5155fbc3c5711e77241eb483f8be5932b2266`|
| Yaw fixed8998 |`1dbc54bc9780c71f6c2ff4e49ccdc7d680a9b5269f7681bab02b04a3ae2f26fc`|
| Parent503 report |`58080a39854614ab7c1fe7f0dee202cf9b79db426662ccc0c1fb24c3c742bb26`|
| Parent509 report |`430c31d32fdb2f7621d863a85a63a497a295ae80a37ecfdc7c12d8ea2d0995cd`|
| Parent521 report |`2661c5bf8c9dc64670e5c13b530c68bb82f4d9a8028b307510a2bd8d47f9cd51`|
| Control503 report |`97d5563b2bf845ee3dd7a74b9ba9e755bbc60858b516ad1a1de64a1b9a1da9cf`|
| Yaw503 report |`b68023cd2d949b3033c94194fd9eff0d84058e8820cd2e58faf68c3357d8853b`|

Prelaunch remote focused tests131 passed; local regression890 passed. Original
reports and decisions remain immutable. Next: the CPU-only static skill
compatibility prerequisite, then separately evidenced behavioral retention.
