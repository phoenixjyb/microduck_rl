# Selected motor headroom before a stronger steering lesson

The owner authorized the next bounded rounds after the
[matched gait-scale continuation](2026-10-14-v8-gait-scale-steering-contrast.md).
Keep the faster provisional gait while testing one motor-load objective, not
simultaneously a new sampler, heading reward, learning rate or plant.

## Read-only evidence and fixed hypothesis

The prior sealed summary SHA256
`ac57a9922c9ee84f5386b24a281a49238b3f7e7d7710cdcb453d7e8029c523bf`
and manifest
`a83b604c48cbb1cbddb4bea27bc02149869eec7c0ed5aa14af2541ddc69d1c0e`
were authenticated again on100.100, followed by all24 raw-case hashes for
both arms before reduction. All raw data stays native. The gait-scale arm's
worst pooled motor-force p99 was.359080Nm against the existing.36Nm numerical
gate. Its straight-case individual left-ankle p99 reaches.446246Nm, left-knee
.416453Nm and right-knee.400547Nm. Negative-turn left-ankle/left-knee reach
.416266/.407040Nm; positive-turn left-knee/left-ankle.416139/.379193Nm.
These are maxima across three held-out cases, each calculated over its own
240-step/eight-world force samples, not per-world or substep peak torques.

The old gate pools all14 actuator columns; **it is not an individual-actuator
p99 gate**. Do not retrospectively alter its decisions. The proposed lesson
asks whether load can be reduced without losing walking or worsening steering.
The numerical.60Nm stall reference is momentary, not a safe continuous rating;
the squared utilization is a comparative current-duty proxy, not a calibrated
temperature model. Positive turns still under-steer; neither these load
associations nor the absence of a yaw hard stop proves the cause.

CPU compilation of the actual base model confirmed exact actuator names and
direct joint transmissions. Select `left_knee`, `left_ankle`, `right_knee`,
`right_ankle` by **actuator names**, never passive/interleaved joint indices.

## Predeclared matched experiment

Protocol `v8-selected-motor-headroom-v1`, unique run
`v8-motor-headroom-<source8>-20261014`. Both arms start independently from
`artifacts/training/v8-gait-scale-steering-1b223f93-20261014/gait-scale/foundation/model_999.pt`,
SHA256 `97fac488eae3b8100515dd310a3f956060f7364a55eaa55752602ef4a67ee000`.
This parent is **not route-qualified or an accepted teacher**. Authenticate
the sealed summary, manifest, exact PT and both original retention references.

Load all actor/critic/normalizer/exploration/Adam tensors directly into fresh
same-layout local parameter objects. Historical parent iter999/env24000,
17 Adam states at40000 and critic count12288000 remain recorded. Its saved
LR is **2.2500000000000008e-05**: inherit it into optimizer groups and adaptive
algorithm scalar. Actor affine/count0 stays frozen; local runner/env counters
start0 with fresh episodes. Simulator/RNG state is not recovered; no bitwise
resume or trajectory identity claim.

Both keep body-yaw EMA.5s, weight−1, the same raw61D observations/14D actions,
unfiltered BAM plant, command distribution and previously sealed frozen mixed
live curriculum. Explicit recipe injection reuses the matched driver without
mutating its historical globals or reclassifying old results. The **only**
paired config difference is `selected_motor_load.weight`:

- Control0: reward actually inactive; the metric still measures its cost.
- Headroom−.25: fixed throughout, no adaptation after viewing outcomes.
- Cost `mean_selected(u² + 4*relu(u−.70)²)`, `u=abs(force)/.60`.
- Strict exact-name resolution/finite inputs; no force/target clamp, no NaN
  sanitization or silent zero for missing data. Fail closed on invalid data.
- Capture actual weighted reward before autoreset; compare against the named
  pre-reset motor stream at every control step. Log each selected motor p99,
  global utilization, power, duty proxy, falls, first-update KL and curricula.

Each arm independently starts from this same parent for64×5 smoke(seed947),
256×10 benchmark(seed953) and256×1000 foundation(seed967),24 rollout steps/
update. Both smoke and benchmark receipts must pass before either full run.
Save0/250/500/750/999, evaluate final999ONLY for each arm, no checkpoint search.
Final local iter999/env24000, Adam60000 and critic18432000 are required.
First mean KL≤.05/max-minibatch mean≤.10; retain aborted weights before refusal.
All six initial model/critic/optimizer states must match the same parent.

## Unchanged qualification and a separate lesson decision

Retain held-out seeds839/853/857,8 worlds/240 steps/.02s, startup60 and commands
idle,.30 straight,.20 with±.20yaw. Raw traces are retained before reduction.
Recompute unchanged instantaneous-yaw, all-cycle AND full-trajectory path/
heading gates, motor/range/idle safety and BOTH sealed walking references.
No numerical gate is relaxed to admit the new actor.

The CPU `compare` command authenticates six training and two final-evaluation
receipts before writing a new source/checkpoint/result-hash-bound decision.
Predeclared **diagnostic lesson usefulness**, not policy acceptance, requires:

- Maximum selected-actuator case p99 over all nine moving cases reduced≥5%
  against the equally continued control.
- For every moving command and selected actuator, its maximum case p99 grows
  no more than5% relative to that command's continued control.
- Per-moving-command mean speed MAE grows≤.02m/s and total moving speed-pass
  count does not decrease against continued control.
- Per-moving-command mean absolute full-settled projected-heading endpoint
  error rate grows≤.01rad/s. This does not replace the stricter route gate.
- Both arms retain BOTH sealed walking/idle references, complete first episodes
  without terminals, and the unchanged motor/range safety gates.

Even a useful lesson is not a route-qualified policy. Report all existing
gate failures and one-seed uncertainty; no automatic next lesson/promotion.

## Shared-host limits and evidence

Use only100.100, one nonenabled sequential retained user service: paired
smokes, benchmarks, foundations, final evaluations, then CPU comparison.
Per-arm caps300/300/2100/600s; CPU comparison cap300s; outer7200s,
8GiB RAM/200% CPU, Torch20% GPU.
Conservative benchmark projection must fit below2070s. Verify clean exact
tested source/runtime/parent/recipes/lease/GPU UUID and durable unique output
paths before launch. Preserve DINO2029 and all unrelated workloads. Pause ONLY
originally active user vLLM, restoring it after verified Duck GPU exit; other
protected user/system scopes remain as found. Never overlap GPU workloads.

Do not touch100.98/NX/FilmBrain, packages/drivers/reboots, raw perception,
actor observation expansion, action filters, physical motion or publication.
Do not create video or promote obstacle/hop/ball skills before their applicable
qualification. Record source/tests, GPU admissions, completed learning,
numerical decisions and physical acceptance separately.

Preparation:54 focused CPU tests pass, covering historical-default preservation,
explicit new parent/Adam/critic clocks, strict actuator selection, real mjlab
zero-weight behavior, pre-reset reward accounting and deterministic motor/speed
comparison refusals. Compile/diff-whitespace checks pass; all68 local links in
this document and the curriculum resolve. Independent read-only review found
no blocking defect; owner inspection corrected mjlab's registered-column versus
zero-weight-callable distinction. Native exact-source tests, CPU preflight and
GPU admission remain separate launch gates.
