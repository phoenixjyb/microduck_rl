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

## Native completion

Exact merged source: `0b368ceff28d32f1f486761b730c102fe8f8e909`, clean
detached worktree `/home/converge/work/microduck_rl-upstream-20261010`.
The original native research checkout stayed clean at `3b32468f`; no checkpoint
or dependency was overwritten.

The first capped CPU service reached its180s limit. Read-only inspection before
retry showed64 threads (the TasksMax ceiling), only12.270s accumulated CPU and
no other GPU workload beyond the preserved DINO worker. Thread exhaustion was
suspected, not established from a stack trace. The exact same source and test
selection, with OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS all1,
passed196 tests with1 architecture skip in7.94s. Retain the initial timeout
journal as an operational failure, not a simulator or test assertion failure.
Subsequent smoke launch also used those three thread limits.

The GPU service `microduck-upstream-0b368cef-smoke.service` completed successfully
(Result=success, ExecMainStatus=0, MainPID=0). RemainAfterExit retains the unit
as active/exited; it is **not an active training process**. Live retained
properties confirm RuntimeMaxSec600, MemoryMax8589934592, CPUQuota200%,
TasksMax64, Nice10 and KillMode=control-group.

| Task | PPO updates | Worlds × control steps | Wall time including setup | Terminations |
| --- | ---: | ---: | ---: | ---: |
| Upstream walking | 5 | 64 ×120 | 5.506s | 193 |
| Motor-aware Run | 5 | 64 ×120 | 4.245s | 195 |

Together these are15,360 environment transitions. Both final model_4.pt files
and saved optimizer states were finite and nonempty; rollout actor/action
shapes were61/14 and rewards finite. The frequent terminations are expected
from fresh random policies and **do not establish stable locomotion**. The
motor-aware journal also reports torque utilization p99≈1.0675 and soft-limit
exposure≈0.3392 on the displayed final metric, so these disposable checkpoints
are not accepted motor-safe policies.

Runtime remained Torch2.9.1, Warp1.12.0, MuJoCo3.10.0,
MuJoCo-Warp3.8.1, mjlab1.3.0 and BAM1.0.1. No environment or driver sync.
Initial host sample:961MiB,43C. Last in-process sample:1608MiB,50C,
including644MiB attributed to the smoke process. Post-exit sample:961MiB,45C,
DINO1592/946MiB only. These are samples, not continuous peak telemetry.
All four protected user/system service scopes remained inactive at the sampled
checks. No unrelated service was stopped or restarted.

Full checkpoints, TensorBoard logs and JSON remain in the native worktree's
`artifacts/training/upstream-0b368cef-smoke`. Only small result/JUnit evidence is
mirrored on Mac in `artifacts/evaluations/upstream-0b368cef`; hashes match:

| Artifact | SHA256 |
| --- | --- |
| result.json | `0d4051262531e7e64c66d41f532f1b28451f3f9ff334389f908901f74797675b` |
| Walking model_4.pt | `70ab2361780c84617e1e12c4b475293fb2e1a55181bce78d5f169ccaf99facc0` |
| Motor-aware model_4.pt | `32324c38ed20aa53507c62f9433517597e257570339675bbdad355393c9649e9` |
| Native CPU JUnit | `21e18885840ef2b50f827adaabe605a3ab54bbe187024441d9ea132a955daae7` |

Playback reset regressions passed3 tests (7 with the wheel-glide suite).
Additional obstacle reward/hop revision checks passed21 tests; obstacle
observation/entity/scene-adapter/hop-gate checks passed38 (overlapping suites,
not additional unique totals). The final finite-tree integration test also
checks actual TensorDict observations, not just plain dictionaries.

Disposition: upstream source integration is ready for review and short learning
experiments. No new learned skill is promoted. The next substantive training
experiment should use the upstream yaw/dead-zone findings with paired straight
and left/right-turn evaluation, retaining the baseline and motor/stability
nonregression gates. It requires a separate declaration; no longer run was
started from this smoke.
