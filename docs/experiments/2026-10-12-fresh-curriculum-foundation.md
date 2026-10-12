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
