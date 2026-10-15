# MicroDuck walking and turning handoff

Snapshot verified on **2026-10-15 at 09:42 Asia/Shanghai**. The next Codex
session should build a bounded waypoint controller over the existing frozen
walking policy, not restart a reward-tuning campaign. Walking and both-sign
walking turns have numerical and sampled visual evidence in flat simulation.
Turn-in-place, very slow walking, obstacle avoidance and hardware deployment
are not qualified by this checkpoint.

## Read first

- [Project working agreements](../../AGENTS.md), especially delivery first and
  the observation, actuator and normalization invariants.
- [Current training and calibration evidence](2026-10-14-stock-walking-baseline.md).
  It is chronological: the later command-response results supersede the earlier
  inference that steering was generally absent, not the recorded gentle-turn failures.
- [Upstream reproduction boundary](2026-10-14-upstream-v8-reproduction.md).

The owner wants visible motion capabilities and useful training exposure.
Time-box diagnosis; do not spend another session building diagnostic machinery
or repeating small scalar/seed sweeps. Keep simulation demos, robust navigation
and physical acceptance distinct. The owner previously said no real Duck was
available; no physical robot has been validated in this work.

## Workspace and source identity

| Item | Verified value |
| --- | --- |
| Mac workspace | `/Users/yanbo/Projects/microduckPlayground/microduck_rl` |
| Current working branch | `integration/upstream-2026-10-10` |
| Code and evidence tip before this handoff | `9725c15c1371ce7c2b1090a59f76ad7f37e74afd` |
| Fork | `https://github.com/phoenixjyb/microduck_rl` |
| Native SSH | `converge@192.168.100.100` |
| Native checkout | `/home/converge/work/microduck_rl-upstream-20261010` |
| Native HEAD | `199fef5142613ae3ea0c80c7a8d7dfdeca1d5c5d`, detached and clean |
| Reused native Python | `/home/converge/work/microduck_rl-athletics-obstacle-curriculum/.venv/bin/python` |

Mac was clean at the pre-handoff tip, which the fork ref matched through HTTPS.
The native checkout deliberately lacks the later documentation-only closeout.
This handoff adds documentation after that tip; use `git log` and current refs
rather than expecting HEAD to remain equal to a snapshot SHA.

The old `feat/athletics-obstacle-curriculum` line and dated September/October 6
handoffs are historical, not the current training branch. The venv's older
directory name is not the active source location. Keep source and runtime separate;
do not switch back to an older checkout just to reuse its venv.

Learning source was `23b35c1d0ee6bf92aa884dfb86a413c08e485435`.
The 36-case calibration used `22ea6bdfc3082587f324dda8bbb6eae0f1fe3369`;
the two-turn visual review used native HEAD `199fef51…`.
The last two stages evaluated the same weights; they did not retrain the policy.

## Authority and live host state

Both authorized GPU windows have ended: overnight training through 07:00,
then a 30-minute calibration/demo window starting 07:51:09 and closed early
after vLLM restoration at 08:01:30. **No Duck job is currently running.**
The present user request is to write this handoff, not open a new GPU window.
CPU implementation and tests can proceed when requested; obtain a renewed
window before pausing vLLM or launching another GPU job.

At the 09:42 read-only check:

- GPU: RTX 4090 **Laptop**, 16,376 MiB reported total, UUID
  `GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`; 10,656 MiB used, 5,538 MiB free, 51°C.
- DINO PID **2029**, 946 MiB, executable
  `/home/converge/Tonghao/VLM/grounding_dino_cpp_dev/build_worker/grounding_dino_cpp_worker`.
- USER `recomo-ai-mission-vllm.service` active, GPU PID **2754957**, 9,692 MiB.
- USER subject-model worker and both SYSTEM AI-mission services inactive.
- Both calibration/demo Duck services inactive with `Result=success`.

PIDs, occupancy and permissions are point-in-time facts. Recheck them; do not
whitelist a new PID merely by process name after a reboot. Preserve DINO and
all unrelated workloads. Do not use 100.98 as an automatic fallback: NX/team
and FilmBrain workloads there were explicitly placed out of scope.

The user approved an **80°C software cutoff** for the prospective Duck runs.
Existing helper defaults remain 75°C; 80°C requires an explicit argument.
This did not authorize driver, power, clocks, fan, OS delegation or motor-limit
changes. No physical motion, model publication or production deployment is authorized.

## Frozen policy and retained artifacts

Task: **`Mjlab-Velocity-Flat-Backlash-MicroDuck`**, stock walking recipe.
Actor interface stays **61 observations to 14 actions**, BAM/backlash encoder
plant, 50 Hz control and baked observation normalization in the exported ONNX.

Native candidate checkpoint:

```text
/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/stock-walk-extension1-s1019-20261015/resume-later/model_9998.pt
SHA256 acaeec42cec3640df4500feed3013bcab93ca7910c67dda2ac6120906ff6e2f1
```

Sibling normalized export: `resume-later.onnx`, SHA256
`465e23c5535b3b1b15c7880b4fed06cc7a514b3fffd06bca5c57253c6dd3355a`.
Use standard runner loading and export, never a hand-converted actor without its normalizer.

This candidate has 10,001 cumulative PPO updates and **245,784,576 transitions**:
501 parent updates plus 5,500 foundation and 4,000 continuation updates, each
with 1,024 worlds and 24 steps. A separate five-update smoke is not in that
lineage. rsl_rl resume reuses an iteration label, so final label 9998 is not
the actual update count. Saved `common_step_counter=240024` is correct.
Resume restored weights, optimizer, normalizers and curriculum counters;
RNG/simulator trajectory continuity was not claimed.

The foundation checkpoint is also retained:
`artifacts/training/stock-walk-resume-foundation-s1019-20261015/resume-main/model_5999.pt`,
SHA256 `638820e797fe137a55a2ffa7f1f6c6434f56d75e91f165a585d579d7b6df982c`.
All checkpoints and exports remain on native; do not duplicate the checkpoint
history on the capacity-limited Mac or commit large binaries to Git.

Under `artifacts/evaluations/`, these directories exist on native and Mac:

- `stock-walk-resume-foundation-s1019-20261015-review`
- `stock-walk-extension1-s1019-20261015-review`
- `stock-walk-command-response-s1019-20261015`
- `stock-walk-moving-turn-demo-s1019-20261015`

They contain results, per-case metrics and, for the reviews, videos. Small
training launch/result receipts accompany the two training reviews on Mac.
Artifacts are ignored, not included in an ordinary fork clone. Retrieve missing
evidence from native instead of rerunning an experiment just to recreate it.

## Actual motion evidence

The calibration used three seeds 42/211/223, eight worlds per case, 500 scored
steps plus 50 warmup steps. All **36 cases** had zero falls, NaNs or nonfinite
steps; these are short flat-simulation trials, not navigation qualification.

| Requested forward m/s and yaw rad/s | Observed forward m/s and yaw rad/s | Interpretation |
| --- | --- | --- |
| 0 and 0 | approximately 0 and 0 | stable idle |
| 0.2 and 0 | 0.0047 and 0.0060 | mostly stationary |
| 0.3 and 0 | 0.1774 and 0.0346 | forward walking |
| 0.4 and 0 | 0.2325 and 0.0165 | faster forward walking |
| 0 and ±0.5 | approximately 0 and ±0.0032 | turn-in-place ineffective |
| 0.3 and +0.5 | 0.1783 and +0.4282 | left walking arc |
| 0.3 and −0.5 | 0.1695 and −0.3241 | right walking arc |
| 0.3 and +0.8 | 0.1786 and +0.6689 | stronger left walking arc |
| 0.3 and −0.8 | 0.1645 and −0.5487 | stronger right walking arc |

All three seeds turned in the requested direction for the moving ±0.5 and
±0.8 commands. Small ±0.2 commands at forward 0.2 remain ineffective. Do not
report supported command bounds as achieved speeds, or say that the policy
cannot steer at all. The earlier gentle-command failures remain unchanged.

The visual demo repeated six cases without falls/NaNs, then recorded seed42:

- [Left walking turn](../../artifacts/evaluations/stock-walk-moving-turn-demo-s1019-20261015/record/left/microduck-run-0.30mps-yaw+0.50radps-step-0.mp4)
- [Right walking turn](../../artifacts/evaluations/stock-walk-moving-turn-demo-s1019-20261015/record/right/microduck-run-0.30mps-yaw-0.50radps-step-0.mp4)

Both are 10 seconds, 500 frames, 50 fps, 960×540. Ten 1-fps frames per clip
were inspected: upright stepping and opposite body rotation are visible.
Full playback remains unreviewed. Preserve the original receipt flags;
`policy_acceptance=false` is not changed by sampled visual inspection.

## Next bounded task

Build a **frozen-policy waypoint/path-control trial without obstacles first**.
No waypoint result has yet been collected for this new checkpoint.

1. Inspect and reuse existing components before creating another framework:
   [hierarchical controller contract](../hierarchical_obstacle_controller.md),
   `src/mjlab_microduck/hierarchical_obstacle.py`,
   `hierarchical_obstacle_rollout.py`, `corner_navigation.py`,
   `gait_path_qualification.py` and `body_yaw_trace.py`.
   Older controllers bind different policies and envelopes; do not silently
   load their parent checkpoints or reuse their acceptance claims.
2. Keep the learned actor frozen. Start with stop `(0,0)`, forward `(0.3,0)`
   and left/right walking arcs `(0.3,±0.5)`. Use the actual measured speed
   and asymmetric turn response. Do not automatically raise a low requested
   speed to 0.3, rely on spin-in-place, or borrow old 0.8 m/s speed bounds.
3. Predeclare a few mirrored left/right waypoint routes, deterministic seeds,
   timeouts, completion tolerance and motor/stability checks. Collect world XY,
   projected-body-x heading and contact/gait traces. Judge actual path and
   heading over gait cycles; retain instantaneous yaw error as a diagnostic.
   Existing trace helpers have fixed shapes/protocols: adapt explicitly and
   test rather than feeding incompatible new traces to old scorers.
4. Write CPU tests for command bounds, stop behavior, goal/timeout handling,
   heading wrap and route symmetry. Obtain a fresh GPU window, run a bounded
   smoke/evaluation under the existing lease and restoration pattern, then
   record a route clip. Do not queue another long training run first.
5. After path control works, add a single visible obstacle and external compact
   geometry, allowing slower motion during interaction while preserving the
   nominal tracking objective before/after it. The stock walk model uses
   feet-only collision geometry: full-body clearance and collision coverage
   need explicit validation before claiming obstacle avoidance.

Perception stays separate. The existing
[obstacle observation contract](../obstacle_observation_contract.md) describes
an older obstacle-actor interface. For this frozen 61-D locomotion actor,
put range/bearing/width/height/closing-rate/validity at the **supervisor**;
do not append those channels to the locomotion actor without a separate design.
Do not inherit old HC speed-recovery thresholds as passes for this checkpoint.
Hop/jump and rolling-football balance remain separate specialist tracks.

## Tools and runtime to reuse

- `scripts/train_stock_walking.py`: exact-source guarded stock training and
  standard resume. Approved resume update choices are 5/5500/4000, 1,024 worlds,
  seed1019, 0.5-second idle pause after each PPO update; save every100 updates.
  `MICRODUCK_WARM_START` must be unset for same-task resume.
- `scripts/review_stock_walking.py`: guarded `gentle`, `command-response` and
  `moving-turn-demo` protocols. `--evaluation-only` skips clips. Outputs must
  be new directories; every invocation binds source and checkpoint SHA.
- `scripts/evaluate_checkpoint.py` and `scripts/record_checkpoint.py`: fixed
  forward/yaw commands through `evaluation.fix_velocity_commands`. Direct
  invocation alone does **not** provide the host lease/watchdog.
- `scripts/stock_walking_overnight.py`: finite completed workflow, not a
  scheduler. Its old date/deadline/run IDs must not be replayed.

Native pins: torch2.9.1, Warp1.12.0, MuJoCo3.10.0, MuJoCo-Warp3.8.1,
mjlab1.3.0, better-actuator-models1.0.1, rsl-rl-lib5.0.1.
Do not `uv sync`, upgrade the driver/runtime or move to newer mjlab incidentally.

The last code suite passed **104 tests on Mac**. Native passed the 61 affected
review/recording/host-guard tests at `199fef51`; the earlier full native suite
passed 103 before the added demo-protocol test. No new build is needed for this
handoff. To rerun the focused suite on Mac:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_train_stock_walking.py tests/test_stock_walking_overnight.py \
  tests/test_review_stock_walking.py tests/test_exploratory_turn_pilot.py \
  tests/test_stock_walking_recording.py tests/test_upstream_integration.py \
  tests/test_infer_policy_bam.py
```

For native CPU checks, use the reused Python above, `CUDA_VISIBLE_DEVICES=''`,
`PYTHONPATH=src`, and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## GPU execution rules for a later approved window

Use `/home/converge/.local/state/microduck-gpu0.lock` and one retained user
service/one GPU child at a time. Recheck clean exact source, original service
states, GPU UUID, every GPU PID and DINO executable/owner before pausing only
the originally active USER vLLM. Fail closed on another workload; do not stop it.

Retain the current limits unless separately reviewed: Torch allocator20%,
total GPU use≤6144MiB/free≥8192MiB, explicit temperature cutoff, requested service
`MemoryMax=8GiB`, child timeout and outer service cutoff with restoration margin.
Check cgroup enforcement rather than assuming a property enforces its limit. Freeze the
native checkout while a Duck job is active. The user-service CPUQuota property
was not actually enforced on this host; the explicit0.5s idle pause provided
real pacing. Training observed at most76°C after the 80°C approval.

Reference scoped handoff files are on native under
`/home/converge/.local/state/<run>-context.py`, `<run>-handoff.sh`,
`<run>-restore.py`, and `<run>-handoff/` for:

- `stock-walk-overnight-s1019-20261015`
- `stock-walk-command-response-s1019-20261015`
- `stock-walk-moving-turn-demo-s1019-20261015`

They preserve original states and restore only USER vLLM after proven GPU exit,
retaining `context.json`, `restore-idle-proof.json` and `restored.json`.
Treat them as references, not reusable launch scripts: source SHAs, dates,
PIDs, output paths and expired authority are pinned. Never run vLLM concurrently
with Duck to work around the lease. Leave failed historical units/results intact.

Codex heartbeat creation never returned a confirmed registered task. Do not
assume unattended agent continuation, an active scheduler, or any renewable
GPU permission exists. The completed native services were finite and independent.

## Background that should not block the next motion demo

The stock walking fallback is **not** a faithful upstream VelStand v8 BC run.
The restored upstream recipe still needs stand PPO teacher
`69u48n8l/model_9750.pt` and walk teacher `441tzs6d/model_3750.pt`; public ONNX
assets are not those checkpoints. Exact teacher access remains unresolved.
Do not silently disable BC or relabel an ONNX warm start as reproduction.

The generic CPU deployment rehearsal lacks the full passive-backlash training
plant. Its moving failures are not the task-native walking-turn result above.
Keep historical rejected experiments intact, but do not resume those diagnostic
loops or treat a past host/package mismatch as a current driver failure.
