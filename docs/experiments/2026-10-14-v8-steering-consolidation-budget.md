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

## Admitted native execution, final results pending

Executed source `c340ff00d8d532fa4a664a298589b7fa401d7467`, run
`v8-steering-budget-c340ff00-20261014`. Native11-file affected suite passes216
tests (three existing warnings,23.58s). CPU-hidden preflight authenticates the
sealed control parent, all optimizer/normalizer state, original references,
runtime and honest complete/masked recipe hashes. Transferred operation hashes,
Bash syntax and unit verification match local preparation; retain original
service context before one start of the nonenabled sequential service.

Both complete smokes and benchmarks pass. Control/extended smoke4.319/4.342s,
first mean/max KL.018543/.032189 and.017454/.032386. Benchmarks7.792/7.846s,
KL.016163/.021844 and.016242/.022250; both per-budget projections fit the
foundation cap. All four short initial PTs and the control foundation initial
share SHA256 `c6b09dc25d4693074e086faa5d5745ce04f5aa30cc12a03e9531f65e6a7e9562`.
Actor/critic/Adam fingerprints and inherited LR match the exact parent; the
rejected penalty is actually0, and the existing yaw penalty stays negative.

A read-only snapshot authenticates all four admitted receipts and observes
control109/1000 updates, finite rows,0 NaN terminations and85 training falls
so far. First foundation KL.016327/.023938 passes; these are training
diagnostics, not held-out acceptance. Latest sampled GPU1710MiB/63C,
DINO2029 retained, Duck PID1853745; protected user vLLM/subject worker are
inactive during the lease. Small local proof:
`artifacts/training/v8-steering-budget-c340ff00-20261014-admission/partial-report.json`.
Models/logs remain native. Keep native source frozen at the executed SHA while
the sequential job continues; later documentation commits do not advance it.

Both full budgets and final evaluations/comparison are **pending**, not
claimed complete. The service continues to control999, extended1999, the
unchanged held-out matrix and deterministic three-reference retention checks,
then restores only original user vLLM after proven Duck GPU exit. No new
campaign or skill promotion follows automatically.

## Completed matched exposure comparison: rejected

Both full budgets and final held-out evaluations completed successfully at the
executed source `c340ff00`. Control1000 took632.350s; extended2000 took1276.559s.
The full pair collected **18.432M transitions**, with613/1391 training falls and
zero NaN terminations. All update/reward/optimizer/motor fields are finite.
All six initial PTs share the admitted `c6b09dc2...` hash. Final local clocks
are999/env24000 and1999/env48000; Adam80000/100000 and critic24576000/30720000
are verified. Actor affine and updater count0 remain unchanged. Both first-update
guards pass (control mean/max.016327/.023938; extended.016588/.024003).

Both12-case held-out matrices are complete first episodes without falls or
terminals. Idle retains24/24 worlds each; motor/range safety and BOTH original
walking references pass. Legacy instantaneous-yaw and complete route gates
still reject all nine moving cases in both arms. Numerical gates are unchanged.

| Command vx / yaw | Complete route worlds:1000→2000 | Speed passes | Mean full heading endpoint/T rad/s | Mean full XY endpoint/T m/s |
| --- | --- | --- | --- | --- |
| .30 / 0 |2→9/24 |18→21/24 |.053328→.042922 |.082669→.071556 |
| .20 / −.20 |0→6/24 |24→24/24 |.072484→.046402 |.054286→.031901 |
| .20 / +.20 |5→0/24 |24→24/24 |.052882→.076419 |.047082→.043405 |

The extended arm gains8 complete moving route worlds (7→15/72) and moving
speed passes66→69/72, but **positive-turn heading error worsens44.5%**,
an increase.023537rad/s versus the allowed.01. The1000-update control also
fails the sealed provisional-parent walking reference (66 rather than69 speed
passes). Extended retains that reference. These are two explicit failures of
the predeclared comparison, yielding **`budget-lesson-rejected`**. Do not treat
either new checkpoint as a promoted teacher, extend this recipe again, or
reinterpret the partial gains as complete steering qualification.

Cycle counts479→480; endpoint failures120→93 and additional peak-only failures
143→91. Straight cycle XY failures36→23; turns have none. Positive-turn
endpoint failures rise33→50 while its complete route worlds fall5→0. This
directional trade-off, rather than missing gait cycles, blocks progression.
Held-out modeled raw force p99 maxima.321092→.333461Nm remain below.36Nm;
minimum named motor-hinge margins.160951/.197471rad. Training maxima are
separate: modeled utilization.836131/.839613, soft-limit exposure.021659/.023344,
mechanical power2.95844/3.03013W and thermal proxy.076957/.080334. Neither the
proxy nor these modeled limits establish physical servo safety.

### Durable assets and independent verification

The packet contains149 files /362,603,025 bytes, retained on100.100. Mac stores
only110,644-byte summary and23,628-byte manifest in
`artifacts/training/v8-steering-budget-c340ff00-20261014-closeout/` (ignored).
CUDA-hidden independent verification authenticated every receipt, initial and
final state, checkpoint cadence, raw evaluation hash and all three walking
references; it recomputed the complete deterministic decision without writing
over the original. `verified-evidence-not-policy-acceptance`, with both
authorization flags false.

| Artifact | SHA-256 |
| --- | --- |
| Control final999 PT | `4a29f18855349550625ff5df3170ffbd680a7046ed7d7fec5a4398062ecb9368` |
| Extended final1999 PT | `3e3924cf693c89c5289adfb996d4f6ac35c93bc54ac01ee4fd6e032b052e3742` |
| Control evaluation receipt | `7eb94b874b6fdaa10be42c9f1d5d0ed59538c8a80add76e07e46f504b7aef4f6` |
| Extended evaluation receipt | `2f680774fa17639a339b8711c27bb19b46f6faf31f5757ff7f17650e4ce147c6` |
| Decision | `770c5a59a10b546b3e1c2eef617fe14cf0e8d59950e18b46a74ddb668c6cca09` |
| Verified summary | `bfba1397b4a6657b2411f72eac25afb2f1bbb401b11f1d5443df6bc7f4d58987` |
| Inventory manifest | `55bfe531793362c938c99eabc3650d6f42426b424c098ca85cf4dcfaa5b8e436` |
| Independent verifier | `7d0ba6d42671fb1671d29ae0f6fddf3fed87d3c8bb1088062f886b79ce197ed5` |

Service exitsuccess/0, inactive. Guarded handback proves a DINO-only idle GPU
before restoring only original user vLLM, then the independent verifier confirms
live vLLM GPU PID1885812 belongs to its user-service cgroup. DINO2029 survives;
user subject worker and both system scopes remain inactive. No Duck GPU process
remains. Maximum sampled GPU1710MiB/67°C. Native executed checkout remains clean.
Closeout checks:55 focused CPU tests pass, including the retained diagnostic and
old continuation contracts; no install, perception, video or physical motion.

The next bounded lesson will change only the existing yaw-bias price, not the
EMA horizon or budget. It must start from the earlier sealed `dddd3757...`
control parent, not either rejected new actor. See the separately predeclared
[gain contrast](2026-10-14-v8-yaw-gain-contrast.md) for its unchanged gates.
