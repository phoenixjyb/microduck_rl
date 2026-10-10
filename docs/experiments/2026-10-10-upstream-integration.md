# Controlled upstream integration

The owner approved the October 10 sync recommendation. Integrate source in
`integration/upstream-2026-10-10`, without moving the default branches or the
research branch `feat/athletics-obstacle-curriculum`.

## Exact inputs and preservation

- Research baseline: `3b32468fd4d1bfc417dd4aa048834c2bddd98ef7`.
- RL upstream: `ceca01706143d729305baa94cbd8ab990979e325`.
- Runtime upstream: `dec725c67ffbbdde7f1de30f8c936b1bf9c1d70e`.
- Framework source upstream: `033ae22a2c7a30a25a6fa77b16c113ed88dd1b55`.

Runtime/framework integration branches are source-only fast-forwards. Do not
install sibling mjlab main into the RL environment: upstream RL still requires
mjlab1.3.0, Warp1.12.0 and Torch2.9.1. No driver or dependency sync in this step.

The six Git conflicts combine both custom task/MDP additions and upstream
features. The inference reset moves from R to X because upstream uses R for
roulade; camera follow and incremental steering remain available. Upstream's
removed Velocity2 registrations must not survive without their deleted factory.

Upstream walking removed the widening speed schedule, standing-height reward
and exact-zero stillness term. Those removals are intentional upstream recipe
changes, not terms to silently delete from the historical hop experiments.
`athletics_velocity_env_cfg.py` retains the baseline configuration for custom
Run, motor-aware Run, obstacle, sprung, hop and hostile-terrain tasks. Standard
walking and VelStand use the new upstream recipe. Hop drops the new turn-bucket
minimum-fraction field only because its cyclic phase command never executes
the velocity sampler; unknown dropped fields still fail closed.

This separates **configuration**, not whole simulator versions. Robot assets,
MDP functions and contact families changed upstream. Old checkpoint acceptance
and paired evaluations must continue using the retained baseline tree and
frozen environment. Do not rerun old policy comparisons under the new source
and call them equivalent merely because the actor/action dimensions remain
61/14. The rejected gentle-turn candidate remains rejected.

Upstream's VelStand notes document useful policy lineage and yaw-command-remap
hypotheses. Both experimental yaw-reward fixes were left disabled after losing
turning. Keep them off; this sync is not evidence of a learned turning fix.

## Bounded native smoke predeclaration

After CPU contracts pass and the exact source commit is pushed, prepare a
separate native integration worktree. Reuse the existing baseline venv via
explicit PYTHONPATH; leave its source checkout, artifacts and dependencies alone.
Run `scripts/smoke_upstream_integration.py` as one sequential user service:

- Upstream walking, then custom motor-aware Run; each64 worlds ×5 PPO updates
  ×24 steps. Seeds811 and812. Fresh random policies, no inherited checkpoint,
  no W&B/Hub downloads, no expert BC run and no video.
- Literal CUDA_VISIBLE_DEVICES=0; existing cooperative GPU0 lock. Preserve
  GroundingDINO PID1592 and all unrelated CPU workloads. Refuse any other GPU
  process or active protected AI Mission service in user/system scopes.
- Observed aggregate GPU memory≤6GiB, free memory≥8GiB and temperature<75C.
  Torch allocator capped20%; this is not a Warp allocator cap. Check host
  health before initialization, every24 rollout steps and after both tasks.
- User service hard cap600s, RAM8GiB, CPU200%, low scheduling priority and64
  tasks. No retry/continuation of a failed job without read-only diagnosis.
- Require finite64×61 actor observations,64×14 actions and step rewards,
 120 complete rollout steps per task, durable final model_4.pt, finite saved
  actor/critic/Adam state and nonempty optimizer state. Count terminals; random
  policy falls are not evidence of a walking skill. The installed reward NaN
  sanitization does not prove every raw reward term was finite.

This is source-integration and actual-optimizer execution evidence only. It
does not accept an upstream policy, a new skill, motor thermal capability,
simulator fidelity or physical motion. Do not launch a longer training campaign
automatically from this smoke.

## Source validation before native execution

On Mac, with CUDA hidden and the retained venv on PATH:

- Combined integration/athletics/upstream configuration suite:192 passed,
 1 skipped (architecture-specific installed-GPU check),59.26s.
- Newly imported feature suite (head bias, NaN observations, pickup, publishing,
 BAM inference, sit/stand fallback, ground-pick, roller standup, wheel glide):
 121 passed,45.23s; two upstream ONNX deprecation warnings.
- Dedicated integration tests after adding the nested model/Adam finite check:
 5 passed,30.51s. These overlap the first suite; do not sum them as unique tests.
- Full repository collection completed with10,719 tests, no collection errors.
 Full execution of the historical forensic suites was not run for this sync.
- `uv lock --check --offline` passed (157 packages), without syncing the venv.
- Syntax compilation and integration-only whitespace checks passed. A full
 staged merge whitespace check also reports inherited upstream XML/JSON
 trailing spaces and pre-existing research-document/test whitespace; those
 untouched input bytes are retained. Existing duplicate
 `pose_target_match` definitions are present in both input branches; not a
 newly introduced merge duplication.

The runtime and framework source integrations are clean fast-forwards pushed
to their forks' integration branches; no Rust build, framework installation,
daemon deployment or physical verification was performed.
