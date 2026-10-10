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
only with the existing GroundingDINO worker, preserving it and all CPU workloads.
Hold the existing cooperative GPU lock for the process lifetime. Reject another
compute process, protected AI Mission services in either namespace, GPU >=75 C,
total GPU usage >6 GiB or free GPU memory <8 GiB. Torch allocator capped at 20%;
Warp is not covered by that allocator cap, so observe aggregate GPU occupancy.
Unit limits: CPU 200%, RAM 8 GiB, low priority, 64 tasks, stop all child processes
on a hard timeout. Never stop unrelated services to fit the run.

Enable installed NaN guard and reuse finite actor/action/raw manager output,
return, gradient, optimizer and learned-parameter checks. Existing reward manager
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
