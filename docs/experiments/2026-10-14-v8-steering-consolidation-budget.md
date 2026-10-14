# Steering consolidation: matched additional optimization exposure

After the [selected motor headroom closeout](2026-10-14-v8-selected-motor-headroom.md),
retain its unchanged control as a provisional research starting point. The
−.25 knee/ankle penalty was rejected; it must remain inactive. Both original
walking/idle references and all route/safety gates remain unchanged.

## Evidence and one-factor question

The control retains walking, all first episodes and motor/range safety. Across
72 moving worlds it passes67 full-settled heading endpoint AND maximum checks,
but only12 complete route components. Of479 complete moving cycles,73 fail
heading endpoints and160 additional cycles fail their within-cycle heading
maximum;18 XY failures are straight only. Durations/support timing do not show
a unique causal phase. Repeated optimization under the same.5s yaw objective
improved positive-turn full heading from.100447 to.036039rad/s, although this
parent-to-child change is not a one-factor causal estimate.

Test whether **more optimization exposure** improves complete-cycle steering
under an otherwise identical recipe, before adding a phase-specific tax that
could suppress the natural gait. Protocol `v8-steering-consolidation-budget-v1`,
unique run `v8-steering-budget-<source8>-20261014`.

## Exact continuation and immutable recipe

Parent `artifacts/training/v8-motor-headroom-06c2b709-20261014/control/foundation/model_999.pt`,
SHA256 `dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a`.
Authenticate its sealed100,685-byte summary
`61391992bf577974883c454155f01367d8faccad916734719aef06a30e574671`
and manifest
`f5e7d7a15b6796172d57af354c022c21ec6f7a7281ce40ac9049b8d304341bbf`.
This is **not a route-qualified policy or an accepted teacher**.

Both arms independently load all actor/critic/normalizer/exploration/Adam
tensors into fresh local parameters. Parent iter999/env24000,17 Adam states
at60000, critic count18432000 and actor affine/count0 are verified. Inherit
actual saved LR **.00011390625000000005** into optimizer groups AND algorithm
scalar; do not replace it with the constructor rate. Local clocks0/fresh
episodes are deliberate, with no simulator/RNG recovery or bitwise-resume claim.

The only paired configuration difference is `agent.max_iterations` in the
foundation: **control1000**, **extended2000**, each256 worlds/seed967 and24
rollout steps/update. Both have the same body-yaw EMA.5s/weight−1, unfiltered
61→14 policy, BAM dynamics, full signed-command mix and previously sealed
seven frozen step0 curricula. Keep the inert selected-load reward slot at0 and
its read-only pre-reset instrumentation in both; do not stack the rejected
penalty. Smoke64×5(seed947) and benchmark256×10(seed953) are identical for both
arms, so their actual complete config hashes must match. Foundation complete
hashes differ, matching only after the declared iteration-budget mask. No
artificial arm-labelled config hash is needed.

All six starts use the same parent, not prior smoke/benchmark checkpoints.
Both short admissions must pass before either full run. First mean KL≤.05 /
maximum minibatch mean≤.10; retain aborted weights on refusal. No automatic
LR/weight/seed/threshold retuning after a numerical rejection.

Control final999: env24000/Adam80000/critic24576000. Extended final1999:
env48000/Adam100000/critic30720000. Actor affine/count0 remains identical.
Save every250 updates and each arm's final; evaluate **only999 control and1999
extended**, with no checkpoint sweep or later selection. Intermediate checkpoints
are recovery artifacts, not alternative accepted endpoints.

## Numerical qualification and a separate diagnostic comparison

Keep seeds839/853/857,8 worlds/240 steps/.02s, startup60 and the same12-case
idle/straight/±turn matrix. Retain and hash raw pose/contact/path/motor traces
before reductions; recompute legacy instantaneous-yaw and complete-cycle PLUS
full-trajectory path/heading gates. Both arms must retain BOTH original walking
references and the newly sealed provisional control walking baseline. Nothing
in this experiment relaxes the route gate or grants physical acceptance.

The deterministic CUDA-hidden `compare` command authenticates all stage and
evaluation receipts, then reports diagnostic usefulness only if extended gains
at least **six additional complete moving route-component worlds out of72**
against the equally initialized1000-update control, with no paired moving
speed-pass-count decrease, no per-command full-heading endpoint error growth
above.01rad/s, no per-command full-XY endpoint error growth above.02m/s, and no
motor/range/idle or sealed walking regression. The provisional baseline also
requires moving-speed passes≥69/72 and per-command mean MAE growth≤.02m/s.
All existing route failures remain reported even if this incremental lesson
is useful. One training seed is exploratory, not multi-seed qualification.

## Host and retained execution boundary

Only100.100, one retained nonenabled sequential user service: both smokes,
both benchmarks, control1000, extended2000, both final evaluations and CPU
comparison. Caps300/300/3900/600s per arm, comparison300s, outer10800s;
8GiB RAM/200% CPU/Torch20% GPU. Project each benchmark to its own declared
budget with elapsed×updates/10×1.5+60; each must fit below3870s. Actual preceding
1000-update throughput was about625s, so this is bounded rather than open-ended.

Before launch verify exact clean tested native source, installed runtime and
parent/recipe/lease/UUID, operation hashes, Bash syntax, user-unit configuration,
unique output paths and original protected-service context. Preserve DINO2029;
pause ONLY original active user vLLM and restore it after proven Duck exit.
Other protected scopes stay as found. Preserve100.98/NX/FilmBrain and unrelated
work. No GPU overlap, installs/drivers/reboots, raw perception, actor expansion,
filters/clamps, physical motion, video, publication or obstacle/hop/ball promotion.

Preparation:67 focused CPU tests pass, including old contrast/headroom behavior,
explicit budget-matrix refusal, honest identical short-stage config hashes,
parent optimizer/normalizer ownership and the six-world diagnostic comparison.
Owner integration corrected recipe-hook dispatch and kept config-budget checks
read-only. Independent read-only review found no blocking defect. Native
exact-source tests, CPU preflight and GPU admissions remain separate gates.
