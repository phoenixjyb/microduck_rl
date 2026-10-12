# Fresh integrated curriculum foundation: bounded declaration

This is a new **single fresh-policy baseline**, not another c81 continuation or
a causal one-factor contrast with the rejected collapsed-curriculum studies.
The prior [Adam-state comparison](2026-10-10-integrated-tracking-refinement.md)
failed both full matrices. Its reset checkpoint is not an accepted teacher.

The original [fresh foundation](2026-10-10-integrated-turn-foundation.md) stopped
at1500 updates. Standing and head-pose stages end at2000; action smoothing,
head-pose bias and trunk CoM stages end at1500. That run did not give the final
stages consolidation. This run allows6000 updates at256 worlds, including about
4000 updates after the last configured stage. This tests a fresh recipe with a
larger budget, not a claim that duration is the sole cause of previous failures.
It is36,864,000 transitions, **not** a4096-world upstream-volume equivalent.

## Frozen recipe and boundaries

Protocol `integrated-fresh-curriculum-foundation-v1`. Factory calls the original
`integrated_turn_foundation.prepare_config("foundation")`, preserving every
stock reward, event, observation and curriculum, changing only declared world
count, update budget, seed and logging labels. Its existing scoped commands stay
vx[.2,.4], vy0, yaw[-.5,.5], resampling3–5s, forward bucket.25, turn-in-place0;
stock standing starts.02 and rises to.25 at2000 updates. Head/body/CoM curricula
are not collapsed, accelerated, widened or replaced. In particular this is not
the stock unconstrained [-.4,.4] vx sampler or the later warm-start recipe.

Fresh actor/critic, empirical normalizers and Adam start from the constructor;
no checkpoint is loaded, no c81 weights/moments or inherited LR are used.
Read-only initialization checks zero clocks, zero normalizer counts, empty
Adam and original stock Python/group LR.001, then installs the existing KL
telemetry capture without changing learner behavior. Adaptive KL.01, stock
entropy and5epochs x4minibatches remain as configured. Saved initial/final
checkpoints authenticate fresh initialization and real finite Adam updates.

Keep61D actor observations (raw twist slots48:51),14 BAM outputs with no clip
or filter, stock observation noise/delays, current installed plant/runtime and
50Hz control. Do not add the c81 motor/qpos/command-error/angular reward stack
to this fresh baseline. Motor torque/speed/power/thermal and joint ranges are
still measured, with the existing gross training abort guards and unchanged
strict held-out envelope/range gates. This is simulation-only exploration;
random-policy falls are expected and are not waived in held-out evaluation.

## Budgets and admission

| Phase | Worlds x updates | Seed | Process cap |
| --- | ---: | ---: | ---: |
| Exact-source integrated/adjacent CPU regressions | no GPU | n/a | before smoke |
| Official normalized finite61-to14 ONNX smoke |64x5|821|600s|
| Same-world fresh-recipe benchmark |256x10|823|600s|
| Fresh foundation |256x6000|1327|9000s|
| Final5999 held-out matrix |8x240 per case|839/853/857|900s total|

Smoke821 and benchmark823 reuse the earlier base factory's proven coverage
seeds, not its weights or acceptance. Each must again show actual nonzero idle,
straight and both signed-yaw action counts. No forced idle, synthesized counts,
resampling shortcut, relaxed coverage or bypass is allowed. A failure stops
admission and remains in its unique directory.

Both preflights must be successful under the exact clean pushed source, frozen
pins and host identity, with authenticated finite checkpoint hashes and smoke
ONNX inference. The ten-update benchmark conservatively projects to6000 as
elapsed x600 x1.5; require strictly less than8940s before the9000s learner.
The9000s cap is predeclared because the earlier fresh benchmark9.698s projects
to8728.2s and would not fit a7200s cap. Do not widen a cap after observing a
failed admission. Early throughput does not prove late curriculum performance.

One retained lease-protected user service runs the learner then evaluation
sequentially, overall10100s, per-child timeout and15s kill grace. Preflight is
separate, overall1300s. No concurrent Duck GPU workload. Preserve DINO2021,
unrelated processes, current drivers/packages and both protected AI services
inactive in user/system scopes. GPU used<=6144MiB/free>=8192MiB/temp<75C and
20% per-process torch allocation remain admission/runtime bounds.

## Evidence and numerical gate

Retain initial.pt, every250-update checkpoint and final5999; evaluate **only
final5999**, no intermediate selection or retrospective rescue sweep. Total
144,000 environment steps and120,000 optimizer minibatches. Log actual command
dwell, finite losses/rewards/KL, falls, motor costs/thermal/soft-limit exposure,
and read-only live curriculum weights/ranges. Manager updates may occur on
reset; do not mutate them merely to make a logged boundary exact.

Use the same held-out factory and full decision as the latest integrated
studies: play=True, stock delays/noise, fixed idle,(.3,0),(.2,-.2),(.2,.2),
seeds839/853/857,8 worlds each,240 first-episode steps,60-step tracking startup
exclusion, no reset laundering. All worlds must satisfy speed MAE<=.03 idle/
<=.10 moving and yaw MAE<=.10; zero terminals and rated-speed exceedance;
torque utilization p99<=.60; all14 servo min range distance>=-.005rad and
right-hip .05rad stop exposure<=1%. The named/all-joint checks are retained,
not replaced by the weaker original foundation-only decision.

Read-only straight gait traces explain body-yaw/contact/foot-site behavior;
they do not change commands/observations or authorize a video. A full numerical
pass permits considering the next separately reviewed experiment, not policy
publication, simulator qualification or physical acceptance. No obstacle/hop/
football promotion, MP4, raw perception, actor expansion, physical motion or
protected-service restoration follows from launch or training completion.
Retain/hash/recompute the full result on both hosts before any advancement.

## Source review before native admission

The bounded setup review preserved the stock schedules and selected proven
coverage seeds rather than forcing idle or skipping a bucket. Owner review
corrected duplicate evaluation kwargs, opaque selector objects in JSON telemetry
and noninteger normalizer-count acceptance before GPU execution. Initialization
now verifies canonical Adam owns exactly the unique actor/critic parameters;
receipts bind the actual initial.pt hash and fresh-state JSON to exact source,
budget, command census and final checkpoint. Evaluation reuses that full receipt
validation before its unchanged numerical gate.

Focused CPU tests11/11 passed; owner integrated/adjacent regressions911/911
passed with14 existing selector warnings in233.95s, CUDA hidden. These are
source checks only. Same-source native tests, smoke, measured admission and
the actual final held-out result remain separate gates.

## Completed native run and retained decision

The run and final-only evaluation completed on October12 at approximately15:02
Beijing. Both retained user services exited with `Result=success` and
`ExecMainStatus=0`; process completion is distinct from policy acceptance.
The exact training/evaluation source remained
`ca7c719796ba6308bd137a856e6b8fea230e3074`, clean and detached on100.100.
The declaration is not rewritten in response to the rejected outcome.

Same-source native CPU regressions911/911 passed. Smoke64x5 completed in5.410s
with finite normalized61-to14 ONNX inference; benchmark256x10 completed
in9.839s and projected8855.124s, below the predeclared8940s admission limit.
Fresh initial tensors, zero normalizer counts and empty Adam were verified in
each phase. The foundation completed6000 updates at256 worlds in3684.157s:
144,000 environment steps,36,864,000 transitions and120,000 optimizer
minibatches. All25 scheduled checkpoints (0,250,...5750,5999) were authenticated
and finite. Final actor/critic normalizer counts were36,864,000; each of17 Adam
parameter states had step120,000. All6000 foundation update records were finite
with zero NaN terminations. Actual dwell covered idle, straight and both signed
yaw buckets; final live telemetry confirms the stock curricula reached their
declared endpoints, including standing.25 and action-rate weight-1.

Training falls decreased from15,099 in the first100 updates to131 in the last100
(614,400 transitions per interval); total72,994. This is descriptive learning
progress under changing curricula, not a controlled causal result or acceptance
of balance. Maximum training torque utilization p99 was1.067539, soft-limit
fraction.337646 and thermal-load proxy.408267. The gross abort guard did not
replace or waive the stricter held-out gates.

### Final5999 held-out matrix: rejected

All12 seed/command cases completed240 first-episode steps across96 worlds,
without a terminal, fall or rated-speed exceedance. **All12 cases failed motor
envelope and joint-range/stop gates; all9 moving cases failed combined speed/yaw
tracking.** The deterministic decision is `foundation-not-ready`:33 failure
flags across12 unique cases, not33 evaluated cases. No intermediate checkpoint
was selected or evaluated as a retrospective rescue.

The following averages pool three seeds and eight worlds (24 worlds per row).
MAE is in m/s for speed and rad/s for yaw; signed response means are descriptive,
not substitutes for the per-world acceptance gate.

| Command vx / yaw | Speed MAE | Yaw MAE | Speed passes /24 | Yaw passes /24 | Achieved vx | Signed achieved yaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|0 /0|.001150|.008952|24|24|.000180|.001519|
|.30 /0|.129703|.217597|0|0|.170297|.016879|
|.20 /-.20|.080758|.255557|23|0|.119242|-.013998|
|.20 /+.20|.084033|.289013|21|0|.115972|.010678|

Moving speed passes total44/72 and moving yaw passes0/72. Idle tracking passes
24/24 but cannot rescue its motor/range failures. Maximum torque utilization
p99 is1.067539 in every command group, versus the unchanged.60 limit. Group
mean power/thermal-load proxies are.150325W/.170991 idle,
1.505598W/.181906 straight,1.060346W/.178844 negative arc and
1.029255W/.178820 positive arc. These are modeled mechanical-power and load
proxies, not measured motor temperatures or hardware safety certificates.

### Read-only rejection diagnosis

Both `left_hip_yaw` and `right_hip_yaw` spend about97–98% of sampled steps
within.05rad of their configured limits. Both have modeled absolute force p99
.640524Nm, above the.36Nm equivalent envelope gate. Their worst range distances
are-.089953rad and-.087741rad, respectively, versus the permitted-.005rad.
Right-hip stop exposure reaches.979167 versus the.01 limit. The right knee also
exceeds the torque gate in moving arcs (p99 about.388–.392Nm), despite staying
within its position range.

This supports a co-occurring limit-seeking/stall-like simulated posture, not a
proof that one joint alone causes underspeed or failed yaw. Position snapshots
are pre-action and force summaries follow decimation, so they are not a
synchronous causal pairing. Retained straight-gait traces and summaries remain
descriptive; simulator qualification and physical capability are still false.
No BAM target clipping, plant change or reward fix was implemented during
monitoring or rejection diagnosis.

### Retention, independent verification and next boundary

Native root:
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/fresh-foundation-ca7c7197-20261012`.
It retains weights, initial states, all scheduled checkpoints, health/update
records,12 standalone held-out cases, aggregate, straight-gait summaries and
native verification. Mac root:
`artifacts/evaluations/fresh-foundation-ca7c7197-20261012`.
Only the33 sealed JSON/JSONL payloads (47,957,777 bytes) and their manifest were
copied to Mac; weights/ONNX were not duplicated. A stdlib-only independent Mac
verifier checked every payload hash/size, reimplemented the complete numerical
gate, and recomputed all three retained straight-gait summaries. Its local
`verified-manifest.json` is a derived receipt outside the original seal.

| Retained object | SHA-256 |
| --- | --- |
|Final5999 checkpoint|`245784a7b0b7ba9120211c98c1ed93998fc655bf2cf0d6d858cc705edf1fba4e`|
|Evaluation aggregate|`3d97a4bec6aaafe8b279104aab4548027432f63634ee81f7d3d197f40268cd01`|
|Straight-gait summary|`1c43f7dee6aaf682f1b1c77505b1e04e307f974ecd3ce1c077acf7987f4161cb`|
|Native verification|`763b64ab2c01e9bda3dc9ec0d850e1fdfcd935b27d7dc981087fb856065beb70`|
|Sealed33-payload manifest|`1ffc7287ee4937990901f03ea83da9bb7c3503de9b5c1e7d3ee7bbd08dc618e2`|

Observed GPU peak was2073MiB and62C; closeout was969MiB and49C, only preserved
DINO PID2021 occupying the GPU. Both Duck services were inactive, and the two
protected AI services remained inactive in user and system scopes. No service
restoration, follow-on learner, video, raw perception or physical motion was
performed.

This completes the authorized training run, **not** the foundation capability.
The next candidate requires a separate bounded predeclaration addressing motor
load and both named hip-yaw range objectives under the same evaluator. Do not
adopt this endpoint as a teacher, simply extend its duration, weaken gates or
promote obstacle/hop/football lessons from its no-fall short evaluation.
