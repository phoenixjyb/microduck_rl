# Bounded yaw-price contrast after rejected consolidation

Protocol `v8-yaw-gain-contrast-v1`; unique run
`v8-yaw-gain-<executed-source8>-20261014`. This is a new exploratory one-factor
lesson, not a relabeling of rejected results or a policy promotion.

## Question and immutable parent

The [budget comparison](2026-10-14-v8-steering-consolidation-budget.md) gained
route components but lost positive-turn accuracy and a control walking-reference
gate. Do not keep extending that recipe or seed from either new actor.
The [cycle diagnosis](2026-10-14-cycle-heading-diagnosis.md) separates endpoint
bias from transient excursions, but does not prove a phase or head-motion cause.

Test whether reducing the existing body-yaw EMA L1 price permits better gait
coordination while retaining persistent-turn accuracy: **control−1.0 versus
reduced−0.5**, both tau.5s. This hypothesis may fail by increasing drift; the
unchanged held-out geometry and retention gates must expose that failure.
No new MDP term, phase tax, EMA horizon, actor input, action filter or motor
penalty is introduced. The rejected selected-motor weight remains exactly0.

Authenticate the completed rejected budget source
`c340ff00d8d532fa4a664a298589b7fa401d7467`, summary
`bfba1397b4a6657b2411f72eac25afb2f1bbb401b11f1d5443df6bc7f4d58987`
and manifest
`55bfe531793362c938c99eabc3650d6f42426b424c098ca85cf4dcfaa5b8e436`.
Both independently load the earlier motor-headroom **control** parent:
`artifacts/training/v8-motor-headroom-06c2b709-20261014/control/foundation/model_999.pt`,
SHA256 `dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a`.
Its original sealed summary/manifest and both original walking references must
also authenticate. This is a provisional research parent, not an accepted teacher.

Preserve all actor/critic/normalizer/exploration/Adam state with fresh local
runner/env clocks0. Historical parent iter999/env24000,17 Adam entries60000,
critic count18432000, actor affine/count0, and actual saved adaptive LR
.00011390625000000005 remain verified. No simulator/RNG recovery or bitwise
trajectory-resume claim. Both final999 states require env24000/Adam80000/
critic24576000 and unchanged actor affine/count0.

## Matched budgets and admission

Each arm: smoke64×5(seed947), benchmark256×10(seed953), foundation256×1000
(seed967),24 steps/update. Both short admissions must precede either foundation;
every stage independently loads the same original parent. Save0/250/500/750/999,
evaluate only final999 per arm, with no checkpoint selection. First-update mean
KL≤.05 and max minibatch mean≤.10; retain aborted state on refusal. Keep the
actual saved LR in optimizer groups and algorithm scalar. No failure-triggered
automatic price, LR, seed, horizon or budget retuning.

All seven source curricula remain the same frozen mixed step0 manager profile:
action-rate−.4, head-bias1, standing fraction.10, head ranges±.39/±.39/±.49/±.11,
static body-pose ranges and trunk/head CoM reset distributions±.005m. Actor61D,
14 unfiltered BAM outputs and exact runtime pins remain unchanged. The shared
read-only metric checks actual pre-reset weighted reward equals the exact
arm price×cost; observer and receipt independently bind that price.
Full config hashes must differ, matching only after the declared yaw-price mask.
Historical scalar-weight protocols must retain their old−1 behavior.

## Numerical gates and interpretation

Held-out seeds839/853/857,8 worlds,240 steps,.02s,startup60; exact12-case
idle/.30straight/.20with±.20yaw matrix. Save and hash actual pose/contact/root
path/motor traces before reductions. Keep legacy instantaneous-yaw diagnostic,
all complete-cycle AND full-trajectory route gates, motor/range/idle limits,
BOTH original walking references and the original sealed control walking
reference (moving speed passes≥69/72; per-command mean MAE growth≤.02m/s).

Diagnostic usefulness requires reduced to gain≥6 complete moving route worlds
against equally trained control, no moving-speed-pass-count decrease, no
per-command heading endpoint growth>.01rad/s or XY endpoint growth>.02m/s,
and no safety or any of the three walking-reference regressions in either arm.
`gain-lesson-useful-not-qualified` is incremental evidence only; otherwise
`gain-lesson-rejected`. Both original qualification decisions remain retained.
No obstacle, hop, ball, multi-seed qualification or physical skill follows
automatically from a one-seed diagnostic contrast.

## Scoped execution

Only100.100 and one nonenabled sequential retained user service; both smokes,
both benchmarks, both foundations, both final evaluations, then CUDA-hidden
deterministic comparison. Stage caps300/300/2100/600s per arm, comparison300s,
outer7200s,8GiB RAM/200% CPU/Torch20% GPU. Benchmark projections must fit2070s
per foundation. Native exact-source tests and CPU preflight must precede launch.
Local code/tests alone do not establish native admission or training progress.

Recheck clean exact source, runtime/parent/recipes, lease/UUID and installed
operation hashes plus Bash/unit verification. Retain fresh original protected
service context. Preserve DINO2029 and unrelated workloads; pause ONLY originally
active user vLLM and restore it after proven Duck GPU exit. Other user/system
scopes remain as found. Preserve100.98/NX/FilmBrain. No install, driver change,
reboot, raw perception, video, policy publication or physical motion.

Preparation, native admission and subsequent results will be recorded separately
below; neither preparation nor an active learner is capability acceptance.

Preparation: **103 focused CPU tests pass** across the gain lesson, explicit
weight hooks, old gait-scale/headroom/budget recipes, cycle diagnosis and route
contracts. Tests bind the actual−.5 reward and KL receipt, whole-map rejection of
invalid prices, same parent optimizer/normalizer state, historical−1 defaults and
exact one-factor config hashes. Parallel bounded implementation was owner-reviewed;
integration corrected native evidence paths, retained the historical observer
profile, and kept the foundation cap2100s. Diff-whitespace checks pass. Native
exact-source checks, CPU preflight and both GPU admissions are still separate.
