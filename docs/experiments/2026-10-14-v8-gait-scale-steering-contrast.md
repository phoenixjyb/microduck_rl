# Gait-scale steering: matched continuation after contact-phase diagnosis

The owner authorized the next rounds after the
[projected-rate closeout](2026-10-14-v8-projected-yaw-bias.md). First diagnose
the retained traces; then test one factor in a bounded paired continuation.
Previous rejected actors and acceptance decisions remain unchanged.

## Authenticated offline diagnosis

Diagnostic source `82d569d95dde5bc0a70798ff7582dfd50b755cff` ran CUDA-hidden
on100.100. It authenticates each sealed summary/manifest and each ordered
held-out raw-case hash immediately before reading that case's geometry. It analyzes all36
cases /288 worlds, without replaying a policy, initializing CUDA, or changing
vLLM/DINO. Native and local diagnostic suites each pass12 focused tests.

The measured interval is samples60 through239:179 intervals per world.
Unwrapped projected-body-X heading increments minus commanded yaw×actual dt
are partitioned among left-only, right-only, double-support, flight and
contact-transition. A stable label requires contact agreement at both interval
endpoints; changing labels become transition intervals. Every interval is
counted once. Each world's signed residual and duration reconcile to the
existing full-settled route result. Empty phase means are null, not fabricated
zeros. Group endpoint rates are equal-world means; phase rates are explicitly
duration-weighted. Force/contact timing may lag the kinematic interval, so
these are associations, not proof that a particular foot causes drift.

| Retained run | Straight signed / absolute endpoint rate | Negative turn signed / absolute endpoint rate | Positive turn signed / absolute endpoint rate |
| --- | --- | --- | --- |
| Steady body-yaw EMA | +.013255 / .074679 | +.036421 / .081787 | −.033309 / .095747 |
| Cumulative projected heading | −.011384 / .100865 | +.014686 / .083080 | −.033237 / .085890 |
| Projected-rate EMA | −.016780 / .112477 | −.009446 / .096306 | −.042665 / .097628 |

All rates are rad/s, measured from actual quaternion endpoint geometry. Small
signed means can mask substantial opposite per-world drift; they are not new
pass criteria. Drift is distributed across both single-foot support and
transitions. In projected-rate, about42–45% of moving interval duration is
left-only,42–43% right-only and12.5–13.9% transition, with no stable flight or
double-support intervals. This does not identify a single phase to punish.
Moving-cycle median durations are.44–.46s across the three runs; steady's
three command medians are all approximately.46s, with p10/p90 near.44/.48s.

The1,247,403-byte diagnosis remains native at
`/home/converge/.local/state/microduck-steering-phase-82d569d9-20261014.json`.
Only this compact reduction is copied to the Mac at
`artifacts/diagnostics/steering-phase-82d569d9-20261014/phase-heading-diagnosis.json`;
full PTs and raw trajectories remain native. Exact SHA256:

- Diagnosis: `fc7e1e48955b85ad564370d9989a599c66210e862bb3b096c698f6dcc5f59881`
- Core implementation: `ed08e90b8f095bb84e591da5e530ca477c0642dec06fc243e94b235008717c15`
- Reproduction script: `3f67afa049f46a51effa8c6c38b8408ed4e0f8d809933e087f60f263924367ea`

## Fixed paired learning question

Protocol `v8-gait-scale-steering-contrast-v1`, run
`v8-gait-scale-steering-<source8>-20261014`. Test whether a **.5s reward EMA**,
approximately one measured gait cycle, helps compared with the existing1.0s
EMA under the same additional optimization. This is a timescale hypothesis,
not a claim that the offline diagnosis proves the cause. A shorter EMA can
also overprice natural oscillation; speed/stability/motor retention must
expose that failure rather than be waived.

Both arms start from exact steady source `7f55ded7c615e0adda4f859e6e05dde40c447e66`,
final PT `d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f`.
Sealed summary `182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155`.
It is a provisional research starting state, not an accepted teacher.
Control keeps `body_yaw_bias_cost(tau_s=1.0)`; treatment uses `tau_s=.5`.
Both use the same `yaw_bias` key, weight−1.0, command/reset semantics and
actual pre-reset reward accounting. No projected-rate or cumulative target
is stacked onto it, and no contact-phase penalty is introduced.

The prior run ended at common counter24000, exactly a curriculum boundary.
Its live managers had head ranges35%, but reset-driven smoothing/head-bias/
standing/CoM settings still at their prior values. Freeze the **actual sealed
mixed snapshot** identically in both arms, rather than silently advance to
the next stage or invent a different plateau:

- Action-rate weight−.4; head-bias weight1.0; standing fraction.10.
- Head command ranges±.39/±.39/±.49/±.11rad; existing static body-pose ranges.
- Both trunk/head CoM reset distributions±.005m, with unchanged add semantics.
- Every known curriculum becomes one step0 stage; verify live manager values
  at every action/update, including crossing the local24000 boundary.

Load all saved actor/critic/normalizer/exploration/Adam tensors directly into
fresh same-layout parameter objects. Preserve parameter ownership and prove
state fingerprints against the parent. Historical parent iter999/env24000,
17 Adam entries at20000 minibatch steps and critic count6144000 remain recorded.
Read and retain the actual optimizer LR **7.593750000000004e−05**, setting both
optimizer groups and the algorithm's adaptive-LR scalar consistently. Do not
replace it with constructor1e−4 or fresh Adam. Keep actor affine frozen/count0.
Fresh episodes and explicit local runner/env counters0 are declared; simulator
and RNG state are not recovered, so this is not a bitwise continuation.

Each arm independently loads that same parent for smoke64×5(seed947),
benchmark256×10(seed953), foundation256×1000(seed967). Smoke and benchmark are
admission checks, not warm-up checkpoints. Both arms' complete admitted short
runs must precede either foundation. Save local0/250/500/750/999 and evaluate
**only final999 per arm**; no checkpoint search. Final Adam step40000, critic
count12288000, actor updater count0 and local env counter24000 are required.
First-update mean KL≤.05 /maximum minibatch mean≤.10 with20 minibatches;
write the guard and durable aborted state before refusing. No automatic
weight, tau, seed or budget retry after a numerical failure.

## Evaluation and operational boundaries

Keep the existing held-out seeds839/853/857,8 worlds,240 steps,.02s, startup60:
idle,.30 straight,.20 with±.20yaw. Capture raw body/root-XY/quaternion/contact
traces first and bind final PT/source hashes. Recompute unchanged instantaneous
yaw and complete gait-path gates (all cycles AND full endpoint/trajectory
maxima), plus BOTH sealed cautious and steady walking/idle references. Report
paired descriptive differences between the equally continued arms; changes
against the uncontinued parent are not a one-factor causal estimate.
Require complete first episodes, no falls/terminals, motor/range/idle safety
and existing speed retention before any further lesson is accepted. One
training seed is an exploratory comparison, not multi-seed qualification.

One nonenabled sequential user service on100.100: control/treatment smoke,
control/treatment benchmark, control/treatment foundation, then both final
evaluations. Stage caps300/300/2100/600s per arm, outer6900s,8GiB RAM/200% CPU,
Torch20% GPU; benchmark projection must remain below2070s per foundation.
No overlapping Duck/GPU jobs. Authenticate the exact clean tested native
source/runtime/parent/recipes/lease/UUID before pause, preserve DINO, and
pause/restore ONLY originally active user vLLM after proven Duck exit. Other
protected scopes stay as found; preserve unrelated workloads and100.98.

The operations builder only prepares local files. Before launch, explicitly
transfer the unique context/restore/handoff files to native `.local/state`,
install the unit under `.config/systemd/user`, compare file hashes, check Bash
syntax and `systemd-analyze verify`, and retain the original-service context.
Then recheck CPU preflight and source/unit absence before starting it once.
Preparing local files alone does not establish an installed or active service.

No raw perception, actor61D expansion, action14D filtering/clipping, new
physics/driver/package install, physical motion, MP4, policy publication,
obstacle/hop/ball promotion or automatic new campaign is authorized by this
experiment. Source/tests/admission, learner completion, numerical gates and
physical acceptance remain separate. Record the executed source and later
evidence-document commit separately once the paired run finishes.

Preparation:18 focused continuation tests pass; the affected74-file CPU suite
passes1613 tests (18 existing warnings,30.86s). Compile and diff-whitespace
checks pass, and all67 local links in the two affected documents resolve.
Independent read-only review checked the diagnosis, conditions and operations
plan; owner review corrected reused evaluator/manager API contracts before
launch. Native exact-source tests/preflight and GPU admissions are separate
gates, not claimed by these local results.
