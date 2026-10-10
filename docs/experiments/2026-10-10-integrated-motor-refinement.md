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
