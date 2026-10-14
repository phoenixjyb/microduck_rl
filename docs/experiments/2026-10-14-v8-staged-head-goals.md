# Staged secondary head goals during steering practice

Status: **training/evaluation complete and independently verified; lesson rejected**.
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

## Corrected native admission and active learner (bb223f66)

Executed source `bb223f668c556de2dc651ef40a381ac5fcd5a0ad` passed all130
native CPU tests in18.43s and authenticated source/runtime/parent/recipe
preflight. Read-only forensic validation of the first failed smoke now passes
with the JSON-shape correction; the failed service/run remains unchanged and
is not reused as admission evidence.

The new nonenabled user service
`recomo-duck-v8-staged-head-goals-bb223f66-20261014.service` started at
2026-10-14 18:22:27 Shanghai. Its retained root is
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/v8-staged-head-goals-bb223f66-20261014`.
All six fresh short admissions passed before the first foundation started:

| Admission | Control | Staged |
| --- | --- | --- |
| Ordinary64×5 smoke time | 4.320s | 4.337s |
| Smoke first KL mean / max minibatch mean | .018421 / .031920 | .020115 / .034277 |
| Frozen64×320 boundary probe falls / resets / NaNs | 0 / 0 / 0 | 0 / 0 / 0 |
| Actual stage action steps | 320 parent | 80 small / 80 intermediate / 160 parent |
| Benchmark256×10 time | 7.872s | 7.522s |
| Benchmark first KL mean / max minibatch mean | .016674 / .022343 | .015497 / .023192 |
| Conservative1000-update projection | 1240.84s | 1188.24s |

Both projections fit2070s. Both frozen probes authenticated the exact original
actor tensors before and after inference, count-zero affine, unchanged head
rewards and real cached actor/manager commands. The staged probe sampled beyond
the preceding smaller range in every component without forced resampling:
intermediate counts `[682,783,703,651]`, restored-parent counts
`[2068,2698,3107,2683]` (component order neck/pitch/yaw/roll).

Boundary receipt hashes control/staged are
`aab6afe934a82706285ab323e30d1ee8b8ca8fa66131964d22939858b3398d23` /
`a031a9d9f1e6413b3c21d0b6e4f34205751f49f2d0e5324f52dbb29cec36baf8`;
their stats hashes are
`2c600364045bc1fb86bed774cea7783e4086f7ba535913415ab49d625e5333ff` /
`50dc7119818451842cadf6b99ca51b90fd9aab0cbd34e604e772e499551d6af6`.
Ordinary smoke result hashes are
`7c0d7f6a257d9c7b97e74062645439c443c2aa88dcb49e30d8baceb342425d64` /
`b8ca4999ef027d7664914bd6b080582982f9cef45b0508e93dc38f33d90a7975`;
benchmark result hashes are
`3165cace4727860cba4284cde86c16fe81d9613a7afbc94a7482329e6b83b950` /
`dcda9c67400e3ee23de633e858f05253e53b8041bec777d75e950adbbe316457`.

At the first retained live check the control foundation reached update106,
all106 optimizer/reward/telemetry rows finite, zero NaN terminations, local
counter2544/Adam62120. First KL mean.014769/max.020183 passed. Durable initial
and iteration0 checkpoints exist; later cadence checkpoints are still pending.
GPU occupancy1710MiB/free14484MiB at59C included preserved DINO2029 and only
the leased Duck child. All four protected service scopes were inactive during
training. This snapshot is not a completed foundation or steering result.

The guarded sequence still owes two final999 checkpoints, both held-out
evaluations, the unchanged three-reference comparison, and restoration after
proven Duck exit. No new capability, teacher or physical policy is accepted.
The active native checkout stays clean at the executed source; later evidence
documentation commits must not change its code while the service is active.

## Completed paired result and closeout

The service finished successfully and exited before the independent CPU
closeout. Both arms completed1000 updates/6,144,000 transitions,12,288,000 total;
control637.122s, staged644.807s. Training falls638/609 and zero NaN terminations
in both arms; these training falls are distinct from held-out first episodes.
Both final checkpoints are iteration999/env24000,17 Adam clocks80000 and critic
normalizer count24576000. All six initialization checkpoints had identical
SHA `c6b09dc25d4693074e086faa5d5745ce04f5aa30cc12a03e9531f65e6a7e9562`.
Actor affine/count0 stayed unchanged; no simulator/RNG recovery is claimed.

Final PT hashes control/staged:
`22ee1f4c6f6336d3c1b824cde2a960698d594c850411b367e642520d371602a2` /
`812cbdbc5efe26ac6339cc0137f05656fd2647b63e4eb644ea26b2df29a28766`.
Final evaluation receipt hashes:
`06254e0697eef8b66dcf9b1613eb0e5eecab8c6709fb4bc6addc32f2d6b12bf4` /
`62741a045adaeab179b7c00c4c06f6afa796497f0327d4faa2e2c2394713a12e`.

### Motion evidence, not policy acceptance

| Held-out motion group | Complete route worlds, control → staged | Speed passes | Full heading endpoint/T, rad/s | Full XY endpoint/T, m/s |
| --- | --- | --- | --- | --- |
| Idle | 24 → 24 /24 | 24 → 24 | .001638 → .001568 | .000164 → .000141 |
| .30 straight | 3 → 7 /24 | 19 → 24 | .047402 → .040991 | .075506 → .060380 |
| .20 / −.20 turn | 1 → 1 /24 | 24 → 24 | .050838 → .055494 | .040523 → .034680 |
| .20 / +.20 turn | 3 → 5 /24 | 24 → 24 | .041588 → .032135 | .044525 → .033299 |

Moving totals improved7→13/72 complete-route worlds and67→72/72 speed passes.
The paired diagnostic gain reaches the declared+6-world bar with no speed
count loss or endpoint-regression violation. Both arms retain BOTH original
walking references and pass motor/range/idle safety. Held-out raw force p99
maxima.329938/.339729Nm remain below.36Nm, minimum hinge margins.154950/.171474rad.
All24 first-episode cases completed without substituted terminal/fall episodes.
Training peak force-utilization p99.819851/.816128, soft-limit fractions
.020206/.020136, mean mechanical power maxima2.897516/2.849746W, thermal proxies
.074667/.073754 are simulation diagnostics, not measured motor temperatures.

However, control fails the additional original-headroom walking baseline:
67 moving speed passes <69. Staged passes that baseline, but the protocol
requires retention in BOTH arms. The deterministic decision therefore stays
**`staged-head-goals-lesson-rejected`**, with the single failure
`[control, original_parent, walking-idle-retention-failure]`.
Do not retrospectively exempt control or relabel the pilot as accepted.
All nine moving route cases in each arm still fail full qualification;
instantaneous yaw remains0/72 diagnostic-only. No actor is promoted.

Negative-turn cycle heading endpoint failures worsen23→42; positive-turn
endpoint failures improve22→6 but peak-only failures increase47→66. Straight
cycle XY failures improve31→2. Thus the average improvement does not close
the gait-cycle envelope. Do not raise maximum-error limits to obtain acceptance.

### Temporal proof and retained assets

Independent closed-row reconciliation verifies control24000 parent-range action
steps, staged12000 small/6000 intermediate/6000 parent steps (500/250/250 updates).
Every head component has a nonzero sampled command in every world-step.
Staged intermediate outside-prior-range counts are
`[1070970,1064074,1029566,1025416]`; parent-stage counts
`[850509,855995,872678,881234]`. The complete parent consolidation is actual
exposure, not an initial/final-config inference.

The native independent closeout script
`/home/converge/.local/state/verify-v8-staged-head-goals-bb223f66-closeout.py`
SHA `a154385240d7a2286448ab3f323131f54b6bef62db44a9d03505ecad826566cf`
reauthenticated every receipt, checkpoint, raw held-out trace, decision,
three walking references and restoration ownership. The151-file inventory
contains373,251,071 bytes, retained only on100.100. Mac copies only the small
summary and inventory under
`artifacts/training/v8-staged-head-goals-bb223f66-20261014-closeout`.

Summary SHA `a81efd4ea694dd0d5147bd45c575544c273f910e7f5470de6c186a4bf1fc2b15`;
inventory SHA `dc420efcea50e0f355d715f29a88c77c005f1062ba515db794762ddaf4871f6b`;
decision SHA `967ef520149f8e731ea329b240e2380ef0fe71716c90fb800845c681c421b3be`.
Restoration context/idle-proof/restored hashes:
`b81b56c50ce83a75fbd0483c336ac93aa76f6543bfc2684ea7bd1b20b6d24aa3`,
`826aad5b7a892845237f5b9d1581e2cb61e3cf37cbea12502d54562ec8557632`,
`04c3bf658c271ff411c1441f31531155a0feec035786cd43eb41120bb67db5c2`.
Only original user vLLM was restored, GPU PID2107020 with DINO2029 preserved;
other protected scopes remain inactive. Peak GPU occupancy1754MiB/temp67C,
minimum free14440MiB. These ownership claims were rechecked live read-only.

### Next development gate

The new [CPU exposure audit](../../src/mjlab_microduck/v8_head_goal_exposure_audit.py)
streams exactly1000 closed updates, rejects counter/Adam/range/probe-kind/NaN
faults and fabricated widening, and binds its compact report to the input hash.
It produces exposure evidence only, never policy qualification. Its15 focused
tests and the combined90-test affected suite passed; compilation/diff checks
passed. Native use and results are separate from the executed learner source.

Next, [predeclare matched learning-seed replication](2026-10-14-staged-head-replication.md)
of the promising unchanged schedule, rather than adopting the pilot endpoint
or changing rewards after observing this result. The original rejection remains
immutable, all gates stay unchanged, and no new GPU job is admitted by this document.
