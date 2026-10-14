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

## Completed matched rounds and independent closeout

Executed source **`1b223f93608e14ffcfbfb8be5e819846d032c3e0`** on100.100,
packet `artifacts/training/v8-gait-scale-steering-1b223f93-20261014`. Native
exact-source74-file tests passed1613 (18 existing warnings,30.36s). CPU-hidden
preflight authenticated parent tensors, all paired recipe hashes, both sealed
retention references and unchanged runtime; BAM Git commit remained
`62bd8ce12154340be97e06f7f41a0ca8f116d967`. Generated native file hashes, Bash
syntax and user-unit verification matched local preparation before one start.

Both smokes and benchmarks passed, with active actual negative reward and
mean/max-minibatch first-update KL below.05/.10. Control smoke/benchmark took
4.271/7.724s; treatment4.273/7.755s. Their benchmark projections admitted both
full budgets. All **six initial PT files have the same SHA256**:
`6558a206d2231da9234515cb5c7843ac67c66571cf392af96885af5b493fcbf8`.
Every actor, critic, affine, exploration and Adam tensor matches the parent;
only the declared local runner/environment clocks differ from its save.
Parent optimizer digest `8de593f27e69549f3aed9e95985ba8481a67bb839d93eeaa88f02faaf00ae805`
and inherited LR7.593750000000004e−05 match across all six starts.

Each full arm completed1000 additional updates /6.144M transitions, or
**12.288M additional transitions across the pair**. Control took618.456s,
treatment614.544s; all0/250/500/750/999 checkpoints and finite normalized61→14
exports are retained. Frozen live conditions remained exact throughout.
Both final checkpoints have local iter999/env24000,17 Adam states at40000,
critic count12288000 and unchanged actor affine/updater count0. Both first-update
KL guards pass: control mean.0147772/max.0330272; treatment.0148527/.0328993.
All reward/optimizer metrics are finite, with0 NaN terminations in either arm.
Training falls are843→764; reward activity24000 control steps each. Mean actual
cost.121553→.152235 and weighted reward−.121553→−.152235 reflect the different
filter, not an automatically comparable measure of skill quality.

### Decisions and the speed/steering trade-off

The two final-only evaluations took59.929/60.035s. All24 cases /192 worlds
completed240 first-episode steps, with no terminal, motor/range or idle failure.
Both sealed walking/idle references pass. Moving speed checks are **66/72→71/72**
(straight18→23/24, both turns24/24 in both arms); idle speed/yaw24/24 each.
Instantaneous moving yaw checks remain0/72. Full route qualification still
rejects all9 moving cases in **both** arms. Complete moving-world component
passes improve6/72→8/72, not a complete matrix pass. Idle route cases3/3 each.
Neither actor is an accepted foundation, teacher or newly integrated skill.

| Command vx / yaw | Speed MAE: control → .5s | Full XY endpoint/T: control → .5s | Full heading endpoint/T: control → .5s |
| --- | --- | --- | --- |
| .30 / 0 | .085600 → .050947 | .092285 → .053706 | .056950 → .057987 |
| .20 / −.20 | .046317 → .030143 | .052154 → .032101 | .077723 → .066015 |
| .20 / +.20 | .046380 → .031741 | .048227 → .034206 | .055074 → .100447 |

Speed/XY units m/s; heading rad/s. These are means of full-settled ENDPOINT
errors divided by interval duration, not maximum excursion or full route
qualification. Relative to the equally continued control, treatment speed MAE
improves40.5%/34.9%/31.6%, and XY endpoint error41.8%/38.4%/29.1%. Achieved mean
body vx is.256900/.186808/.181200, versus control.216616/.155992/.158277.
Negative-turn heading endpoint error improves15.1%; straight worsens1.8%,
and **positive-turn heading worsens82.4%**. Mean body yaw for the positive turn
is+.113569 at the+.20 command (control+.233119); that is a diagnostic, not
a replacement for the actual quaternion/path measurement. The useful walking
gain does not offset the asymmetric steering failure. One training seed and
nondeterministic GPU trajectories support an exploratory descriptive contrast,
not a statistical or physical generalization.

### Motor headroom and remaining failures

Held-out maximum modeled utilization p99 is.548056→.598467; corresponding raw
force p99.328834→.359080Nm against the unchanged.36Nm numerical cap. Treatment
passes but has only about **.00092Nm** remaining p99 margin, versus control
.03117Nm. Minimum named motor-hinge margin.139272→.085362rad. This is reduced
modeled headroom, not hardware robustness or measured servo temperature.
Training maxima are separate: utilization.889125/.885292, soft-limit fraction
.029402/.028843, rated-speed exceedance.000372/.000465, mechanical power
3.363760/3.350117W and thermal proxy.090533/.087372. The proxy is not temperature.

| Command | Full XY endpoint AND maximum passes: control → .5s | Full heading endpoint AND maximum passes | Cycles | Cycle heading endpoint failures | Additional peak-only heading failures |
| --- | --- | --- | --- | --- | --- |
| .30 / 0 |14→23/24 |19→19/24 |163→159 |35→42 |53→31 |
| .20 / −.20 |24→24/24 |15→17/24 |161→168 |65→55 |50→30 |
| .20 / +.20 |24→24/24 |22→11/24 |155→165 |28→79 |82→33 |

Every moving world has6–8 physical cycles in control and6–7 in treatment,
without cycle-duration or right-touchdown-count failure. Across479→492 cycles,
heading endpoint failures rise128→176 while additional peak-only failures fall
185→94. Straight cycle XY endpoint failures improve43→9; turning cycles have
none. Thus cycle coverage is present, but both endpoint drift and within-cycle
heading accuracy still block qualification. The positive-turn endpoint loss
and near-cap motor load are explicit next constraints, not reasons to relax
the evaluator or automatically extend/promote this faster actor. Predeclare
a bounded directional-steering/headroom lesson only after examining those
retained cases; preserve the walking gain as a research result and keep BOTH
retention references. No new GPU round is launched by this closeout.

### Retained assets, checks and workload restoration

Native evidence comprises144 sealed files /333360631 bytes. Full PTs and raw
traces remain native; Mac stores only54998-byte verified summary and23000-byte
inventory in `artifacts/training/v8-gait-scale-steering-1b223f93-20261014-closeout/`.
The CPU-hidden independent verifier authenticates all six training receipts,
both raw evaluation matrices, recipe/initial-state equality, Adam/normalizer
clocks, frozen affine and recomputed legacy/route/BOTH retention decisions.
Its status is `verified-evidence-not-policy-acceptance`; both authorization
flags remain false. The verifier is
`/home/converge/.local/state/verify-v8-gait-scale-1b223f93-closeout.py`.

Max sampled GPU use1754MiB, minimum free14440MiB, maximum temperature67°C.
The service finished Result=success/ExecMainStatus=0 and is inactive. DINO
PID2029 remained intact; only original user vLLM was paused/restored, active
with GPU engine PID1759724 at9692MiB after a separately retained DINO-only idle
proof. User subject worker and both system scopes remain inactive. The
independent closeout also rechecks the exact original/restored scope inventory,
binds the restore-handler hash and records live per-PID vLLM cgroup ownership.
No Duck GPU process remains; native executed source is clean at1b223f93.

Closeout checks:30 focused continuation/diagnosis tests pass (9.32s), all67
local document links resolve and diff-whitespace checks pass. Independent
read-only review confirms hashes, metrics, units, paired comparisons and the
unchanged rejections. Full restored service rows and GPU-memory readings stay
in separately hashed native receipts; the compact summary also records the
live cgroup ownership proof. The later documentation commit is not the
checkpoint's executed training source.

| Retained evidence | SHA256 |
| --- | --- |
| Control foundation result | `524b4f362a9329172303afcff55854bad6d8da75070b1f687998dacab0ec495b` |
| Treatment foundation result | `a0826aefb9ad12285b961b09a9607d949b7ae5cb603e7bce929ff75551641efa` |
| Control final999 | `97453e1f31e33d016585039f7d01cbcce9b419f97b0b25eaa284f9dc739530e7` |
| Treatment final999 | `97fac488eae3b8100515dd310a3f956060f7364a55eaa55752602ef4a67ee000` |
| Control normalized ONNX | `da4c65728d3b6caf043e9eb35bf75453a3300567f60cb9e750fd0460cf6755af` |
| Treatment normalized ONNX | `f221b74b66802ef2121d8c81d2b231574bcac17e23cba85ed760e843fc047433` |
| Control evaluation | `fdd98694d3e7d88eb8a4d1eacb2c7c9336745ca8c38d5ecf549801b72b9b0d75` |
| Treatment evaluation | `6e73ff025c6865db704d0fc98252a6c1f5205e0b93392af06529668f473d9b95` |
| Verified paired summary | `ac57a9922c9ee84f5386b24a281a49238b3f7e7d7710cdcb453d7e8029c523bf` |
| Full packet inventory | `a83b604c48cbb1cbddb4bea27bc02149869eec7c0ed5aa14af2541ddc69d1c0e` |
| Restoration context | `bf6e67583c37124c483ce5a21d7d50a32a1e459b848991a98a1ca64cad9fcb42` |
| Restoration idle proof | `84cc5b739026a7cc68522fa094b503ab62ed8deb1168ac2f0d80bebb662e8363` |
| Restoration receipt | `6cc5f37db5f8e1c0879f7d1a4a03dcd52bcd69ef33ff3cc64b1d7df89f96f058` |
| Independent verifier source | `7b82aacc7b0140fe42fdf8f1b67d8241106aa488abdcfa9ea6399ef608eb511e` |
