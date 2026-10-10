# Exploratory gentle-turn learning: priority reset

The owner requested actual duck training after several source-only diagnostic
rounds. This new experiment deliberately does not promote the rejected native
packet/MJB admission, corner supervisor, hopping, or physical acceptance. Those
results remain unchanged. Their forensic closeout is no longer a prerequisite
for this separately authorized exploratory simulation learner. A numerical
improvement here is provisional evidence under the installed simulator, not
proof that that simulator is physically correct.

## Predeclared scope

Use the retained motor-aware parent model_7998.pt, SHA256
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
Strictly restore actor, critic, normalizers and Adam; start fresh rollout episodes.
Both smoke and candidate start independently from this parent, not from each other.
Keep the 61-dimensional actor, 14 motor actions, plant, reward terms, completed
motor curricula and PPO settings. Change only the command distribution: forward
0.15–0.30 m/s, zero lateral command, yaw -0.20–0.20 rad/s, resample every 3–5 s.
This is a foundation for later moving turns, not corner/obstacle training itself.
Head/body commands stay neutral; no in-place pivot or standing distribution.

Run 256 worlds, 24 steps/update. Smoke: seed 701, 10 updates, 360 s hard process
cap; candidate: seed 709, 128 updates, 900 s hard process cap, only after a complete
same-source smoke and measured 1.5x timing budget. Checkpoint every 10 updates;
atomic fsync-backed saves. No automatic further run or multi-seed promotion.

Use native100.100 frozen environment without dependency/driver changes. Share
only with the existing GroundingDINO worker (observed PID1592), preserving it and
all CPU workloads. A restart/identity change closes this run rather than silently
replacing the workload whitelist.
Hold the existing cooperative GPU lock for the process lifetime. Reject another
compute process, protected AI Mission services in either namespace, GPU >=75 C,
total GPU usage >6 GiB or free GPU memory <8 GiB. Torch allocator capped at 20%;
Warp is not covered by that allocator cap, so observe aggregate GPU occupancy.
Externally enforce user-unit limits: CPU 200%, RAM 8 GiB, low priority, 64 tasks, stop all child processes
on a hard timeout. Never stop unrelated services to fit the run.

Enable installed NaN guard and reuse finite actor/action/raw manager output,
return, gradient, optimizer-loss and learned-parameter checks. Adam moments are
not independently finite-checked after each update. Existing reward manager
may sanitize individual terms: this is not a proof of every raw reward term.
Reuse pre-reset control-step motor stream and gross training stop gates: fall
fraction >0.50 per update, torque utilization p99 >0.85, rated-speed exceedance
>0.01. These are abort guards, not policy acceptance. No substep peak or hardware
thermal certification is claimed. Log soft-limit exposure, load and power.

## Fixed evaluation after a complete candidate

Evaluate parent then candidate sequentially, no overlapping Duck GPU processes.
Seed 719, eight worlds, 240 control steps (20 ms), first 60 startup steps excluded
only from tracking summaries, never from terminal/motor gates. Cases: forward
0.30/yaw0, forward0.20/yaw-0.20, forward0.20/yaw+0.20. Stop the whole case on first
terminal; do not continue through reset episodes. Measure actual body-forward
speed and body yaw rate, not initial-route projection during turns.

Exploratory improvement requires all cases complete, torque p99 <=0.60, zero
rated-speed exceedance, every world's speed MAE no worse than parent+0.01 m/s,
yaw MAE no worse than parent+0.03 rad/s, and pooled speed MAE improvement >=0.005
m/s in at least one case. Retain failures as well as improvements. This tiny
single-seed comparison does not accept a skill or lift old simulator gates.

No raw perception, hopping revision, obstacle supervisor, video, football,
physical motion, protected-service restoration or long follow-on job in scope.

Entry point: `.venv/bin/python -m mjlab_microduck.exploratory_turn_pilot
{smoke,pilot,evaluate} --source <exact-clean-commit> --run-id <unique-id>`.
Retained artifacts: `artifacts/training/<unique-id>/{smoke,pilot,evaluate}`.

## Completed run: real learning, candidate not promoted

Source `373e62c363abdc1b535ad16c840eb487e2f3b1b9`, retained run
`turn-explore-373e62c3-20261010` on native100.100. Focused regression passed62
tests on both Mac (13.79s) and native (5.39s), with CUDA hidden for the tests.
The independent read-only review confirmed parent/checkpoint numbering, command
delivery and finite/motor guards; its worker-identity and optimizer-wording
findings were corrected before either training service launched.

Smoke completed10 PPO updates in6.981s of training, with zero falls and no gross
guard failure. The independent candidate completed128 updates in80.429s:
786432 environment transitions, common step195072. All eight actor and eight
critic weight/bias tensors changed from the candidate's initial checkpoint;
all17 Adam state entries advanced2560 steps (128 ×5 epochs ×4 minibatches).
The full final checkpoint, including Adam state, was finite at closeout. That
post-run check does not retroactively claim per-update Adam-moment checks.

The largest training-update fall fraction was1/256, torque utilization p99
0.705904, and rated-speed exceedance fraction0.000127883; no declared training
abort guard fired. Final soft-limit fraction was0.004127 versus0.010440 at the
first update. This stochastic training comparison is descriptive, not a causal
motor-efficiency or thermal certification. Sampled GPU usage peaked1712MiB
including DINO, temperature60C. All four protected service scopes stayed inactive
at the retained health checks, not continuously instrumented between samples.

All six held-out cases completed their first240 steps without terminals; torque
p99 stayed below0.60 and rated-speed exceedance was zero. Per-world gates still
rejected the candidate. Below are eight-world means, not replacements for those
per-world gates.

| Forward / yaw command | Speed MAE, parent → candidate (m/s) | Yaw MAE, parent → candidate (rad/s) |
| --- | --- | --- |
| 0.30 / 0.00 | 0.08310 → 0.08045 | 0.21736 → 0.18684 |
| 0.20 / -0.20 | 0.05447 → 0.05209 | 0.16888 → 0.18539 |
| 0.20 / +0.20 | 0.05598 → 0.05440 | 0.17168 → 0.15220 |

Deterministic decision: `no-demonstrated-improvement`, with
`settled_yaw_mae_per_world-regression`. Left-turn yaw error regressed beyond
the0.03rad/s allowance in a world; no case reached the0.005m/s pooled speed
improvement threshold either. Preserve the original policy as baseline. This
run establishes actual optimizer activity and operational simulation feasibility,
not a new accepted turning, obstacle, hopping, stabilization or physical skill.

Candidate `pilot/model_8126.pt`, SHA256
`7de0a632c2edaa76fc22b94ae6c914b6d6bcd9ef5ef1cc429d46a6a351cb17fc`.
The44-file retained payload manifest SHA256 is
`7f43785faff47328144f7c6b5b8db48910f103ef4b3e8ba4a02e6712f832a6e0`;
`closeout.json` SHA256 is
`43e53bc510d7c10e00b8a3eba21c1b6b935e5f0012aa413c2877186942291e58`.
Checkpoint/YAML/TensorBoard/health/metrics files remain on native100.100; small
JSON evidence and candidate rollout metrics are mirrored on Mac under
`artifacts/evaluations/turn-explore-373e62c3-20261010`, hash-checked against that
manifest. No checkpoint copies or large duplicate training storage on Mac.

Three sequential `microduck-rl-turn-explore-373e62c3-{smoke,pilot,evaluate}-20261010`
user services completed. Their transient definitions were garbage-collected:
post-GC `systemctl show` default properties do not prove the former resource caps.
The launch argv and smoke's live property check retain that distinction;
closeout records terminal journals, no remaining main PID/cgroup and exact
checkpoint/optimizer evidence. At closeout only DINO1592/946MiB remained on the
GPU (44C), and the four protected service scopes were inactive. Nothing restored
or stopped an unrelated workload.

Next learning-focused slice: separately predeclare a longer mirrored-turn budget
contrast from the original parent under the same held-out walking/turning gates,
before changing rewards, actor inputs or adding obstacle geometry. The tiny128
update run is not evidence that more iterations will fix the deficit. Do not
restart packet archaeology as a prerequisite for that exploratory experiment;
do not weaken the old simulator or physical acceptance gates either. No follow-on
GPU job, policy promotion or video was launched by this closeout.
