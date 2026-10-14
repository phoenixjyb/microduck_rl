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

## Native admission and active run

Executed source `7acacf87313c08973d682e2d701c7ef7b8cabe35`, unique run
`v8-yaw-gain-7acacf87-20261014`. After the previous service completed and its
restoration was independently verified, the clean native detached worktree was
fast-forwarded to this exact tested fork commit. Corrected12-file native CPU
suite: **165 passed**,25.56s. CPU-hidden preflight authenticates the original
parent, completed rejection seals, all PPO state/runtime pins and three honest
recipe pairs. No install or runtime change.

Native operation hashes match preparation; Bash syntax and user-unit validation
pass. Fresh original service context is retained before one start of the
nonenabled sequential service `recomo-duck-v8-yaw-gain-7acacf87-20261014.service`.

| Installed operation | SHA-256 |
| --- | --- |
| Context recorder | `c50c7a75bfe71cb6f2577cf259fdee2bcfd8529e2e6d90f947e3e78203bd8d87` |
| Guarded restorer | `f887e77783b196cbca509573d30ef0453990878d331c5368b951e8b558ec1f79` |
| Sequential handoff | `abcf1b1896b07f437cc1c0021a561b4ebf92e7db1ebb6efced1f0bf580087a9f` |
| User unit | `d478b715d6952bdab43cab6c47287badd7c8a093bce8d10c31aa88d0a00904ca` |

All four short admissions complete and independently revalidate. Smoke
control/reduced4.323/4.338s; first mean/max KL.018067/.032062 and.018207/.030564.
Benchmark7.610/7.860s; KL.016362/.021430 and.016458/.022366. Both1000-update
projections fit the2070s admission bound. Reduced actual weighted reward and KL
metadata bind−.5; control binds−1. Both horizons.5s; rejected motor penalty0.
All four short initial PTs and the control foundation initial share SHA256
`c6b09dc25d4693074e086faa5d5745ce04f5aa30cc12a03e9531f65e6a7e9562`;
actor/critic/Adam tensors and saved LR exactly match the original parent.

A CUDA-hidden read-only admission proof observes control312/1000 updates,
durable0/250 checkpoints, finite reward/optimizer/motor metrics,166 training
falls and0 NaN terminations. First full-update KL.016126/.021966 passes.
These are training diagnostics, not held-out stability or skill acceptance.
The proof independently binds GPU PID(s) to the Duck user-service cgroup and
confirms all four protected user/system scopes inactive, with DINO2029 preserved.
Sampled GPU1710MiB/67°C; other workloads and100.98 remain untouched.

Retained proof: native
`/home/converge/.local/state/v8-yaw-gain-7acacf87-20261014-handoff/admission-proof.json`,
17,310 bytes, SHA256
`262b951f1a809d8fc597c0ed6b73dee23ceaa2f9295d6d7b7c00336fff567699`.
Only this small matching proof is copied to
`artifacts/training/v8-yaw-gain-7acacf87-20261014-admission/admission-proof.json`.
Raw models/logs remain native. Keep the active native checkout frozen at7acacf87;
later documentation commits do not advance it while running.

Both full budgets, final evaluations and deterministic comparison remain
**pending**. The retained service continues sequentially and restores only
original user vLLM after proven Duck exit. No automatic subsequent campaign,
checkpoint promotion, perception task, video or physical motion.
