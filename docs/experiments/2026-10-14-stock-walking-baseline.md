# Stock walking baseline, from scratch

First milestone: visible upright forward walking and both-sign turning. This
is a foundational gait, not VelStand recovery, obstacle negotiation, hopping,
football balance or hardware acceptance. Exact VelStand teachers are unresolved;
this run needs none. Close the custom v8 reward-tweak line rather than extend it.

## Frozen learning recipe

Task `Mjlab-Velocity-Flat-Backlash-MicroDuck`; fresh random actor/critic/Adam and
normalizers, no resume, ONNX import, BC or W&B network. The velocity recipe and
backlash wrapper match source `3750776761ecfaa5671b36972087cafba45b8527`.
Retain its BAM walk/backlash plant,61D observations,14 raw actions,50Hz control,
24 steps/update, PPO5 epochs/4 minibatches/adaptive LR.001, normalization and
all existing DR/rewards/curricula. Operational logging is TensorBoard, local
checkpoints every250 updates and final checkpoint, with no video during training.

This is not reproduction of the published VelStand v8 learning run. Flat stock
walking is deliberately simpler and sufficient for the first gait milestone.
Use the standard training function; do not silently narrow commands, disable
head goals or insert a new smoothing/reward/observation wrapper.

## Admission and learning budget

Learning seed1019 is fixed before runs. First smoke64×5, then sequential fresh
profiles1024/2048/4096×20 with the same seed. Each short child≤300s; one retained
profile service≤1800s including exit/restoration. Select the largest safely
fitting batch, using measured steady iteration time and GPU/RAM occupancy;
the profile checkpoint is not a warm start or a skill verdict.

Target first training slice:6000 updates with the selected batch, unchanged
stock curricula. At4096 worlds this is589,824,000 transitions (96× a previous
256×1000 diagnostic). Do not assume4096 fits until measured. Predeclare the
selected batch/source, a conservative runtime projection and a bounded retained
training service before launching. If the projected run exceeds10800s, do not
quietly shorten learning into another tiny diagnostic: choose a separately
declared continuation plan or report the capacity constraint.

## Shared-host ownership

100.100 only, exact clean source, GPU
`GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`, existing owned lease. Preserve verified
DINO2029 and all unrelated services. Reuse scoped original USER vLLM pause and
post-exit restoration; all other protected scopes remain inactive. Never train
alongside vLLM or on100.98/NX/FilmBrain. Existing resource bounds remain: Torch
20% GPU allocator, total GPU≤6144MiB/free≥8192MiB/temp<75C, serviceMemoryMax8GiB
andCPU200%. A safety monitor checks live ownership/temperature during each run;
any violation terminates only its own Duck process. Outer service bounds and
ExecStopPost retain restoration independently of this chat. No packages,
drivers, reboot, credential setup, publication or physical motion.

## Behavior check after learning

Keep historical failures unchanged. Inspect the final checkpoint in its own
flat/backlash training plant with `scripts/evaluate_checkpoint.py` and
`scripts/record_checkpoint.py`; both load the checkpoint normalizer. First report forward speed, both turn signs,
upright survival and idle drift alongside a visual rollout. A useful simulation
demo need not satisfy every prior diagnostic precision gate, but neither a
finite checkpoint nor a demo means robust navigation or hardware qualification.
Only then proceed to stop/recovery and an obstacle supervisor over the working
gait; hopping/football remain specialist tracks.

The existing generic CPU `infer_policy.py` reads direct servo joint positions
and lacks the training plant's passive-backlash encoder feedback. Its rehearsal
remains useful as a distinct transfer diagnostic, not an exact-plant walking
verdict. Use the native task evaluator/recorder for the first gait milestone.

## First native admissions and thermal stop

Source `275ef098dac8fad5c9812baf6f17be8d559a81fd` passed24 focused checks on
Mac/native (6.56/5.25s). Seed1019 smoke64×5 completed6.13s; final checkpoint and
61D→14D ONNX were finite, NaN termination maximum0, GPU peak1542MiB/temp57C.
The1024×20 profile completed23.68s, finite checkpoints/exports/scalars, zeroNaNs,
23,584 final transitions/s, peak2688MiB and67C.

The2048×20 profile was stopped at about iteration16 by the unchanged shared GPU
budget/temperature guard (exit143). Last successful samples were4060MiB used /
12134MiB free, temperature72→73→74C. The memory evidence points to the75C thermal
threshold, although the failed sample itself was not retained.4096 was not run.
No failed profile is accepted; preserve its partial artifacts. Scoped restoration
completed with vLLM owning the GPU again and DINO2029 preserved.

Next bounded admission is **thermal-1024:1024×120**, seed1019, fresh stock recipe,
CPUQuota100% instead of200%, child≤600s and service≤1000s. This is a sustained
resource test, not a promoted gait or warm start. It retains the75C cutoff and
all other bounds. Choose a full training batch only after stable temperature
and measured runtime; do not relax cooling limits to get4096. The native profile
assets remain at `artifacts/training/stock-walk-profile-s1019-20261014`.

## Selected full training slice (predeclared)

The CPU100% thermal admission completed120 updates in124.81s, with finite final
checkpoint/ONNX and all scalar logs, zeroNaNs, peak2690MiB and71C (last70C).
Last60 updates averaged1.002s, p95=1.02045s and maximum1.02592s. The conservative
projection `180 + 1.25 × p95 × 6000` is **7833.4s**, below the10800s child cap.
Its checkpoint is retained but is not loaded into the full run.

Select **1024 environments ×6000 updates**, seed1019, fresh stock initialization:
**147,456,000 transitions**,24× the earlier6.144M-transition diagnostics. One
retained user service, CPUQuota100%, MemoryMax8GiB, RuntimeMax11200s, child cap
10800s plus hard timeout10815s and scoped post-exit vLLM restoration. Existing
GPU/motor/NaN limits remain unchanged; no larger batch or recipe change is
admitted. Run ID `stock-walk-foundation-s1019-20261014`; task-native checkpoints,
TensorBoard logs and exports live in `artifacts/training/<run-id>/train`.
The exact committed learning source is bound in launch/context JSON before pause.

Retain every250-update checkpoint and final `model_5999.pt`; assess the final
checkpoint without cherry-picking a favorable intermediate. If the safety or
runtime guard stops the job, retain the latest durable checkpoint and report
the incomplete learning budget; do not call it completed or auto-reroll.
After completion, use the existing task evaluator for idle/straight/both turns
and the bounded recorder with `--yaw-rate` for corresponding visual evidence.
The recorder now pins the actual command without the forward-only sampler's
hidden speed clamp/yaw suppression. Rendering quality remains unverified until
the actual MP4s are made and inspected. Robust navigation remains a later tier.

## Execution handoff

Full learning source `ac8c0d869104df6c00977027ebac3bad9a9dfbf8` passed82 focused
tests on Mac/native (6.37/5.19s). Service
`recomo-duck-stock-walk-foundation-s1019-20261014.service` started2026-10-14
23:20:06 Shanghai. At the first check it was active at update20/6000, with durable
`model_0.pt`, zeroNaN terminations, GPU2690MiB used/13504MiB free and63C.
Only leased Duck2310583 and retained DINO2029 owned GPU memory; all protected
scopes were inactive. Original USER vLLM restoration is independently retained
in `/home/converge/.local/state/stock-walk-foundation-s1019-20261014-handoff`.
Do not advance or modify the active native checkout. Training completion,
final-checkpoint qualification and visual rollouts are still pending.

## Sustained-runtime correction after the first full attempt

The first full attempt stopped at update192 on the same resource guard; the last
successful sample was2690MiB/74C and all optimizer/reward metrics were finite,
with zeroNaNs. Only `model_0.pt` had reached the original250-update cadence, so
the192-update learning budget is incomplete and its live progress was not
durable. vLLM was restored (PID2314029) with DINO2029 preserved. Keep the failed
run unchanged; the120-update thermal admission did not prove steady state.

Read-only device telemetry was51C with36C T.Limit margin, maximum-operating
T.Limit offset0 and target87C. NVIDIA documents T.Limit as the margin to maximum
operating temperature ([reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).
The75C cutoff was conservative, not a hardware-overheat/driver failure. It remains
unchanged for this work; no power limit, clocks, fan settings or driver is altered.

Next prospective run `stock-walk-paced-foundation-s1019-20261014`: same fresh
seed1019,1024 worlds and6000 updates, but **CPUQuota75%** and **save every100**.
This changes wall-clock duty cycle/checkpoint cadence only, not rewards,
commands, physics timestep, normalization, PPO or curricula. Save more frequently
because the previous guard fired before the first regular checkpoint. Model_5999
remains the final evaluation target; extra checkpoints are not a selection sweep.

Projected cap admission uses the measured CPU100% p95 scaled by100/75:
`180 + 1.25 × 1.0204525 × (100/75) × 6000` = **10384.5s**, still below10800s.
This is a conservative estimate, not a proven throttle-speed law. Keep the same
10800s child/11200s service caps,8GiB memory and scoped restoration. Monitor past
the previous192-update stop and verify a durable100/200 checkpoint before
handing off. If a guard stops it again, retain and report the actual partial
state rather than quietly declaring this a full training success.

## Enforced wall-clock cooling

The CPU75% attempt also stopped at update216. `model_100.pt` and `model_200.pt`
are durable; finite metrics and zeroNaNs do not make this a6000-update result.
vLLM was restored with DINO preserved. Read-only inspection found no `cpu.max`
file in the user-service cgroup: the requested CPUQuota property did not impose
the expected pacing. Do not change host controller delegation or raise the75C
guard to work around this.

Next run `stock-walk-cooled-foundation-s1019-20261014` uses a **0.5s wall-clock
pause after each completed PPO update**. The existing logger is called unchanged,
CUDA is synchronized, then the process sleeps; the temporary process-local
logging hook is restored in `finally`. No simulated step, reward, action,
normalizer update, RNG draw or curriculum clock is added or removed. Standard
`run_train` still owns learning. TensorBoard Perf timing excludes this pause,
so runtime admission explicitly adds3000s for6000 pauses.

Projection is `180 + 1.25 × 1.0204525 × 6000 + .5 × 6000` = **10833.4s**.
Predeclare child cap12600s (3.5h), hard timeout12615s, serviceRuntimeMax13000s;
all other limits and scoped restoration remain. First run a paced64×5 smoke,
then the fresh1024×6000 run in the same sequential service if smoke succeeds.
No new reward lesson or seed search. The earlier partial checkpoint200 is
retained as fallback evidence, not silently treated as a finished baseline.

## October 15 overnight continuation (predeclared)

The cooled run exited143 at iteration576 on `ValueError: shared GPU
budget/temperature`. Its last successful sample was2690MiB used,13504MiB free,
71C; the failing sample was not logged, so do not invent an exact peak. Metrics
remained finite with zeroNaN terminations. The durable resume parent is
`stock-walk-cooled-foundation-s1019-20261014/train/model_500.pt`, SHA256
`083b45b695e9bc4c1c45662a1dbba9458c154534a9b116f4fdb43b85e4cd1dbb`, iteration500,
common_step_counter12024. Original USER vLLM2343998 was restored, DINO2029
preserved, other protected scopes inactive. Preserve this failed run unchanged.

The user explicitly approved **80C software cutoff for the next run**, and
training through **2026-10-15 07:00 Asia/Shanghai** (2026-10-14T23:00:00Z).
Existing callers remain75C by default; only prospective commands pass80C.
Driver, power/clocks/fans, host controller delegation and robot motor limits
remain unchanged. The new guard logs actual failing resource values.

Finite service `recomo-duck-stock-walk-overnight-s1019-20261015.service` runs
the exact committed source in one sequential process group:

1. Same-checkpoint stock resume smoke:1024 worlds ×5 additional PPO updates.
2. Task-native parent evaluation: idle(0,0), straight(.3,0), left(.2,.2),
   right(.2,-.2), seeds42/211/223,8 worlds,500 scored+50 warmup steps each.
   Four seed42 clips,500 frames each. These are partial-policy diagnostics.
3. Standard stock resume from model500:1024×5500 additional updates,
   cap11000s, final label5999. Prior501+5500=6001 actual PPO updates;
  147480576 actual transitions including the parent. Label reuse is standard
   rsl_rl behavior, not an extra simulation/optimizer step.
4. Repeat the same task-native evaluation and four clips.
5. At most two4000-update same-recipe continuations, each **only if** its
   measured wall-clock projection plus1800s review+600s exit margin fits before
  07:00. Cap = ceil(180+1.5×last-run elapsed/update×4000), maximum12600s.
   Do not admit a projection above that maximum. Each admitted round retains
   its final checkpoint and repeats the same review. No reward/seed search.

Resume restores actor/critic, optimizer, normalizers and curriculum counter;
it does not restore simulator state or RNG trajectory bit-for-bit.
`MICRODUCK_WARM_START` must be unset. Same task, stock recipe, seed1019,
1024 worlds,24 steps/update and0.5s idle pause/update remain unchanged.
Save every100 updates; provenance and all local/cumulative budgets are retained.

All GPU children share the existing lease, exact DINO identity,20% allocator,
≤6144MiB used/≥8192MiB free, explicit<80C cutoff, finite checks and outer
wall-clock bounds. One GPU workload at a time. The parent service alone owns
the original USER vLLM pause; ExecStopPost restores it only after proven Duck
exit and DINO preservation. The absolute service cutoff leaves ten minutes for
restoration before07:00. Unrelated workloads and other protected scopes remain
untouched. No physical motion, policy publication or curriculum expansion.

The Codex heartbeat creation request has not returned or registered; no agent
continuation is claimed. This is a finite native training/review service, not
a cron/heartbeat workaround. Generated clips remain visually uninspected until
an actual frame review; training completion is not walking qualification.

### Live admission and early visual evidence

Exact learning/review source `23b35c1d0ee6bf92aa884dfb86a413c08e485435` passed102
focused tests on Mac/native (6.88/5.68s), with clean checkouts. The finite unit
started **2026-10-15 00:16:46 Shanghai**, RuntimeMax6h34m17s (hard fallback
around06:51), TimeoutStop300s; stage admission separately reserves600s before
07:00. Do not advance the native checkout while this service is active.

The standard resume smoke completed5 updates in10.01s, savedmodel504 SHA256
`219a0c4db85cf32b6cbf8d0c13bc6afe198eaa18feb5376bad755479978fa01a`, with finite
checkpoint/optimizer state, restored cumulative curriculum counter and zeroNaNs.
This is506 actual cumulative updates, not505 despite the final label504.

The model500 task-native review completed in149.08s, retained at
`artifacts/evaluations/stock-walk-overnight-s1019-20261015-partial-review` on
native and as a small3.4MiB Mac copy. All12 cases had zero falls/NaNs/nonfinite
steps across960 scored world-seconds. Means below are over three held-out seeds:

| Command vx / yaw | Observed vx (m/s) | Observed yaw (rad/s) |
| --- | --- | --- |
| idle0 /0 |0.044 | -0.246 |
| straight0.3 /0 |0.161 | -0.226 |
| left0.2 /+0.2 |0.123 | -0.266 |
| right0.2 /-0.2 |0.120 | -0.264 |

The partial policy is upright and moving slowly, **not command-qualified**:
substantial rightward yaw bias remains, including on a left-turn command.
Four MP4s decode as960×540,50fps,500 frames,10s each. Ten1fps sampled frames
per clip were actually inspected: render is clear, the duck remains upright,
and stepping/turning is visible. This is sampled visual evidence, not complete
playback, robust navigation or physical acceptance. Preserve the receipt's
`visual_inspection_pending=true`; the inspection is separately documented here.

The5500-update main continuation is active in
`stock-walk-resume-foundation-s1019-20261015/resume-main`. At iteration564 it
had finite metrics, zeroNaNs, GPU2690MiB used/13504MiB free,67C, only leased
Duck2366546 and DINO2029, all four protected scopes inactive. Later completion
and final clips remain pending. No model was promoted and no new reward lesson
was introduced. This documentation commit does not change frozen learning source.

## Overnight result: completed, steady standing, steering not demonstrated

Checked live on2026-10-15 at07:37 Shanghai. The finite service exited successfully,
and original USER vLLM was restored at04:07:19 with GPU PID2561248. DINO2029
remains present; other three protected scopes are inactive. No Duck GPU process
remains. The second extension was correctly **not started** under the predeclared
conservative deadline admission. This does not authorize a new GPU window.

The main run completed5500 updates in7699.48s; extension1 completed4000 in5587.34s.
Thus9500 additional updates/233472000 transitions were actually learned, not just
smoked. Including the501-update parent, final exposure is10001 updates and
**245784576 transitions**. Pipeline stages, including smoke and reviews, took
3h50m02.5s. Exact source remains `23b35c1d0ee6bf92aa884dfb86a413c08e485435`.

| Run | Final file | SHA256 | Saved curriculum counter |
| --- | --- | --- | --- |
| foundation | `resume-main/model_5999.pt` | `638820e797fe137a55a2ffa7f1f6c6434f56d75e91f165a585d579d7b6df982c` |144024 |
| extension1 | `resume-later/model_9998.pt` | `acaeec42cec3640df4500feed3013bcab93ca7910c67dda2ac6120906ff6e2f1` |240024 |

Both actual checkpoints were CPU-loaded and checked finite; saved counters,
launch/result provenance, ONNX exports and all video/evaluation checkpoint hashes
match. There are2631 training health samples (1524+1107), peak76C, peak2690MiB
GPU use, minimum13504MiB free. DINO and inactive protected scopes were retained
in every sample. All46 TensorBoard tags were finite, with zeroNaNs and no
positive penalty-like Episode_Reward terms. The80C approval removed the previous
thermal blocker without driver/power/fan changes.

Both full checkpoint reviews had zero falls/NaNs/nonfinite steps across their
12 short cases. This is stability evidence on flat simulation, **not steering
or obstacle acceptance**:

| Command vx / yaw | Foundation observed vx / yaw | Extension1 observed vx / yaw |
| --- | --- | --- |
| idle0 /0 | -0.00009 /+0.00168 | -0.00008 /-0.00019 |
| straight0.3 /0 |0.15020 /+0.04537 |0.17730 /+0.03733 |
| left0.2 /+0.2 | -0.00003 /+0.00192 |0.00474 /+0.01429 |
| right0.2 /-0.2 |0.00158 /-0.00241 |0.00462 /+0.00003 |

Ten1fps frames from each of the six moving clips were inspected. Straight
walking shows upright steps; gentle left/right clips mostly show standing, in
agreement with metrics. Full video playback remains unreviewed. Final clips and
small training receipts are copied to Mac under
`artifacts/evaluations/stock-walk-extension1-s1019-20261015-review` and the matching
foundation-review directory; models remain durable on native without duplicate
Mac checkpoint storage. No policy was promoted. Further exposure improved
straight speed but did not resolve gentle steering; do not blindly queue another
unchanged long run or claim a good obstacle avoider.

### Next bounded action: command-response calibration, not a new reward lesson

Prepared `review_stock_walking.py --protocol command-response --evaluation-only`.
It preserves the original four-point gentle protocol and adds straight0.2/0.4,
turn-in-place±0.5, moving0.3/±0.5 and moving0.3/±0.8. These all lie within the
unchanged stock training ranges vx[-0.4,0.4],yaw[-1,1]; stock turn-in-place training
already includes angular magnitudes0.4–1.0. Test whether stronger commands reveal
a useful turning response before concluding all steering is absent or altering
reward widths. That is a hypothesis, not a validated capability.

Fixed candidate: extension1model9998, exact hash above; same task/plant,
normalizers, seeds42/211/223,8 worlds,500 scored+50 warmup steps,36 cases,
1800s cap, existing lease/occupancy guard and explicitly approved80C cutoff.
No training, checkpoint selection, reward edits, raw perception or motion.
Numerical receipt remains `policy_acceptance=false`. Stronger-turn clips and any
next learning budget are chosen only after this sweep, not pre-queued blindly.

The07:00 training window has elapsed and vLLM is back. The sweep is prepared and
CPU-tested but **not launched**; a renewed GPU window and scoped vLLM pause are
the remaining execution gate.
