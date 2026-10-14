# Staged secondary head goals during steering practice

Status: **implemented and CPU-tested; native GPU admission and training pending**.
Protocol `v8-staged-head-goals-v1`, planned run
`v8-staged-head-goals-<tested-source8>-20261014`. Existing rejected results and
their gates remain unchanged. This is a curriculum-complexity hypothesis, not
an accepted steering, obstacle, hopping or physical skill.

## Why this slice, and what has already been tried

The completed [gain contrast](2026-10-14-v8-yaw-gain-contrast.md) establishes that
halving the existing yaw-bias price worsened heading in all three moving groups
and halved complete-route worlds, despite improved speed. The preceding
[extra-budget contrast](2026-10-14-v8-steering-consolidation-budget.md) also
failed its predeclared comparison. Do not continue a blind price/horizon/budget
sweep or adopt either rejected endpoint.

Secondary-task simplification is **not a new idea**: the
[older small-head lesson](2026-10-13-motor-range-head-lesson.md) improved group
tracking and motor metrics without passing its original motion gate; its
[yaw-focus](2026-10-13-quiet-head-yaw-focus.md) and
[gentle-turn](2026-10-13-quiet-head-gentle-turn-lesson.md) follow-ups also failed.
Those used the936a parent, full final head ranges in the control/reference and
beginner ranges in the quiet-head treatment, different smoothness/head-bias/
motor/range objectives and older instantaneous-yaw qualification. Do not pool
them with the current official-v8 seeded learner or claim they establish current
path qualification. Their rejection remains retained.

The stock velocity recipe already has a staged head-range curriculum. This
lesson tests a **specific temporal range history against the current fixed
midrange continuation**, not the novelty of staged training. The parent already
experienced midrange head goals; this is not first skill acquisition from scratch.
The hypothesis is that temporarily reducing secondary-goal complexity and then
restoring the same final goals could help steering coordination without leaving
a fixed-small-head endpoint. There is no causal proof that head goals caused the
measured drift.

## Exact paired range history

Keep control at its authenticated parent ranges throughout. Staged starts at the
existing beginner range, returns through the existing intermediate range, then
reaches the **same parent midrange** for250 final updates. Do not widen to the
stock full final±1.1/±1.4/±.31 ranges in this lesson.

| Local environment step (24 per update) | Control neck / pitch / yaw / roll half-range, rad | Staged half-range, rad |
| --- | --- | --- |
| 0 through11999 | .39 / .39 / .49 / .11 | .05 / .05 / .07 / .015 |
| 12000 through17999 | .39 / .39 / .49 / .11 | .17 / .17 / .21 / .047 |
| 18000 through23999 | .39 / .39 / .49 / .11 | .39 / .39 / .49 / .11 |

Thus updates1–500 practice smaller goals,501–750 intermediate,751–1000 the
original parent goals. All four61D head-command slots remain live and nonzero;
head resampling, zero-head probability, reward weights/parameters, actor affine
and observation noise remain unchanged. Only the head-command range history
differs; initial config ranges and its schedule must be declared in the canonical
mask. Both arms must have the same schedule callable and telemetry semantics.

## Parent and frozen non-head conditions

First authenticate the rejected gain source
`7acacf87313c08973d682e2d701c7ef7b8cabe35`, its verified summary
`c7f962464669c250cb8c2c4d62085d3614b10085747a9c95085d01591e3d4125`
and inventory
`4768d07efd6650ed47bc73f8ac33156a55e78ef1ed017259eacc6fa3d19ffda9`.
Both independently load the original motor-headroom control source06c2b709,
final999 PT `dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a`.
Neither gain endpoint is adopted. Authenticate that parent's original seals,
typed model and actual manager snapshot before use; it remains provisional.

Restore every actor/critic/normalizer/exploration/Adam tensor with fresh local
runner/env clocks0. Parent iter999/env24000,17 Adam60000, critic18432000,
actor count0 and saved LR.00011390625000000005 are unchanged. No simulator/RNG
recovery or bitwise-resume claim. Actor61→14 remains normalized, unfiltered and
unclipped with BAM dynamics, runtime pins, DR/noise and PPO unchanged.

Keep yaw-bias weight−1/tau.5s and selected-load weight0. Keep action-rate−.4,
head-bias1, standing fraction.10, original full signed twist sampler, static
body-pose goals and trunk/head CoM reset±.005m. Six non-head curricula remain
frozen exactly; only the seventh head-range history changes. No extra reward,
phase tax, command-yaw restriction, sensor or plant change.

## Implementation and temporal telemetry gate before launch

Before each action, both arms must apply the
same existing pose-range machinery to live command-manager config at the actual
local common step, then validate it; a reset-only callback cannot establish exact
boundary exposure. Do not write only the original env cfg after manager deepcopy.
Audit callback ordering, including reset-driven updates and an update straddling
the declared step. Distinguish
planned boundary, actual applied manager range and command sampling time.

Before each action retain the live range, actual actor/manager head command,
per-component bounds/nonzero counts and stage/counter. Snapshot before command
resampling/autoreset and reconcile exactly one completed step. Do not inject
resampling, mutate cached commands, consume extra RNG or re-evaluate a stateful
head-bias EMA. Widening contains old samples; it does not imply immediate draws
at the new range. Final-stage exposure must prove the parent ranges were actually
available for the declared250-update consolidation, not merely configured.

CPU tests must cover both boundary sides, first initialization, reset/time
rollback, exact head-slot order, manager-copy ownership, retained pre-reset
commands, honest sample bounds, no fabricated exposure, live reward invariance,
recipe masks and historical frozen-profile behavior. Source changes, focused/
affected tests and owner review must be committed/pushed before native preflight.

## Bounded learner and unchanged evaluation

Both arms independently start from the same parent for smoke64×5(seed947),
benchmark256×10(seed953) and foundation256×1000(seed967),24 steps/update.
Short admissions precede either full run and include a **separate bounded GPU
boundary smoke**:64 worlds,320 consecutive physics steps, seed971, one frozen
original-parent actor per arm, no optimizer/learning or counter jumps. The staged
probe uses the same three head ranges at actual steps0/80/160; control retains
the parent range at all three stages. All320 steps and all64 worlds are retained
as counts, with160 steps at the restored parent range. Use the same inference,
normalizer, manager application and capture machinery planned for the learner.
The accelerated schedule belongs only to this named probe, never the1000-update
learner; CPU tests must reject accidental interchange.

The boundary probe requires exact applied ranges and actor-command bounds,
finite actions/rewards/state and zero NaN terminations. Record every fall/reset;
the probe is not a zero-fall qualification test. Confirm each head component has
at least one actual nonzero sampled command outside the preceding smaller range
after each widening, without manufacturing resamples. All frozen actor tensors
and affine/count0 must be unchanged. A missing exposure or boundary is a failed
admission to diagnose, not permission to lengthen the probe after seeing results.
Standard5/10-update smokes alone do not test these transitions.

Save initial/0/250/500/750/999; evaluate only final999, no checkpoint search.
Verify final env24000/Adam80000/critic24576000 and frozen actor affine/count0.
First-update mean KL≤.05/max minibatch mean≤.10, retain refusal checkpoint. Stage
caps300/300/2100/600s per arm; benchmark projection must fit2070s. Boundary smoke
cap180s per arm; outer sequential-service cap7800s,8GiB/CPU200%/Torch20%GPU.
Sequence both standard smokes, both boundary probes, both benchmarks, both
foundations, both final evaluations, then CPU-only comparison(cap300s). All six
short admissions must pass before either full learner. No automatic
seed/LR/schedule/budget retry after failure.

Held-out seeds839/853/857 ×8worlds×240steps/.02s/startup60, idle/.30straight/
.20with±.20yaw. Preserve every raw quaternion/contact/root-path/motor trace,
first episodes without terminal/fall substitution, all complete-cycle AND
full-trajectory route gates and motor/range/idle limits. Retain BOTH original
walking references and the original sealed control walking baseline (≥69/72
moving speed passes and per-command mean MAE growth≤.02m/s).

Keep the same separate usefulness bar:≥6 additional complete moving route worlds
versus equally continued control; no moving-speed-count loss, heading endpoint
growth>.01rad/s or XY growth>.02m/s; no safety or any walking-reference loss in
either arm. Even useful evidence does not qualify a policy, teacher or later
skill. Numerical qualification still needs its distinct confirmation and transfer
gates. Source success or a healthy process is not a motion result.

Only100.100, one guarded nonenabled sequential service after exact tested native
source/runtime/parent/recipe/GPU-lease checks. Preserve DINO2029 and all unrelated
workloads; pause/restore only originally active user vLLM after proven Duck exit,
leave other scopes as found and preserve100.98/NX/FilmBrain. No raw perception,
MP4, policy publication, package/driver change, reboot or physical motion.
The design does not itself authorize a launch or imply an active next learner.

## Implemented admission machinery

The shared [schedule](../../src/mjlab_microduck/v8_head_goal_schedule.py),
[learner](../../src/mjlab_microduck/v8_staged_head_goals.py) and
[frozen-actor probe](../../src/mjlab_microduck/v8_head_goal_boundary_probe.py)
implement the declared histories. Historical observers retain their fixed head
ranges. The new observer reapplies live ranges before each action and before
the closed update snapshot, while retaining the actual pre-action commands
across resampling and autoreset. Existing head rewards are read from their
already-weighted columns, never recomputed to advance their EMA.

On the Mac, 124 affected CPU tests across the schedule, learner, probe and four
existing continuation suites passed. Compilation and `git diff --check` passed.
This includes both sides of every boundary, cached actor-slot equality,
non-fabricated widening exposure, reward/sampler invariance, the supported
motor-stream capacity bucket, fail-closed later-stage admission and unchanged
historical profiles. The exact tested source must still pass the same native
CPU suite and authenticated preflight before the guarded sequential service.
CPU/source evidence is not GPU admission, a motion result or policy acceptance.

### First native admission refusal (be6545df)

Exact source `be6545df8350611bc48b6c68c20faf9830251b2f` passed the same
124 native CPU tests in18.52s and authenticated preflight. Its nonenabled
sequential service started at2026-10-14 18:15:59 Shanghai. Control completed
the ordinary64-world,5-update smoke:120 steps/7680 transitions in4.377s,
final PT SHA `66182cec4d5bb447c043bc2235b4618f0c8fede706250ecbe2c906abf76f06f9`.
First-update KL mean.0181223/max minibatch mean.0322125 passed.

The post-run validator then refused
`source/runtime/parent/frozen-curriculum/recipe-bound training receipt`.
Read-only diagnosis showed the exact parent/runtime/live-profile/hash fields
matched, but expected recipe history retained Python tuples while saved JSON
contained lists. The expected recipe after a JSON roundtrip matched exactly.
This is a serialization defect, not an admitted smoke or a steering result.
No second arm, boundary probe, benchmark or full learner ran. Preserve this
failed run unchanged under `artifacts/training/v8-staged-head-goals-be6545df-20261014`.

The exit guard proved Duck GPU exit and restored only original user vLLM;
DINO2029 remained untouched. The restoration context/idle-proof/restored hashes
are respectively
`52568d330f8920fdf240f78c8942685f5364b675bfa2f34a1c98dec1648d91ac`,
`f764937956d619a0f10acebb9bd98d6ae016c97b76d847e8464cf65dae5f14b2`,
`90ed7fa0b17d552ee486d3610c9b4f2dcc76f512a7cc3f6d2fbfa222af06aab5`.
Any corrected run must use a new tested source/run identity, not overwrite or
promote this receipt. Training settings and the predeclared protocol stay fixed.

The correction normalizes only the range metadata returned to receipt
comparisons. Canonical recipe hashes, manager schedules and learner settings
are unchanged. Six mode-by-arm durable JSON-roundtrip tests were added; the
complete affected Mac suite now passes130 tests in58.75s, with compilation
and diff checks clean. Native revalidation and fresh admission remain required.
