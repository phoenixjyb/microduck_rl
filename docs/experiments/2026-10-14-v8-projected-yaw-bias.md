# V8 projected yaw-rate bias — prospective coordinate-frame contrast

Follow the owner's approval after the
[mixed cumulative-heading round](2026-10-13-v8-projected-heading.md). That
round retained walking/idle, but worsening straight tracking and 2/72 complete
moving-world route passes did not qualify a foundation. It is not the teacher
for this experiment and its rejection remains unchanged.

## One fixed learning factor

Protocol `v8-projected-yaw-bias-v1`, run
`v8-projected-yaw-bias-<source8>-20261014`. Start from the previously declared
[steady-steering recipe](2026-10-13-v8-steady-steering.md). Keep reward key
`yaw_bias`, weight **−1.0**, command `twist`, tau **1.0s**, and replace ONLY
`body_yaw_bias_cost` with `projected_yaw_bias_cost`. The owned read-only metric
changes to identify the new measurement. No cumulative heading target,
pre-action priming, pose differencing, observation expansion or action filter.

Use the instantaneous analytic rate of body-X projected onto world XY:
`b=R(q)eX`, `omega_world=R(q)omega_body`, `db=omega_world×b`,
`heading_rate=(b.x*db.y−b.y*db.x)/(b.x²+b.y²)`.
Read the actual wxyz root quaternion and body angular velocity at the same
post-physics reward phase. Keep the old estimator exactly: residual is command
yaw minus measured rate, EMA alpha=min(1,step_dt/tau), positive cost=abs(EMA),
with episode-length≤1/yaw-command-change/rollback clearing history. A change
of forward speed alone does NOT reset it. Repeated same-step metric reads are
idempotent; skipped steps, mutated cached inputs or incompatible metadata refuse.

This geometry is invariant under global world-Z yaw. A level robot reproduces
the existing body-yaw EMA, while tilt may change the measured rate. The tests
must compare against an independent quaternion-integration finite-difference
oracle, including yaw-wrap cases, and preserve the old valid-pose filter/reset
semantics. No missing-substep displacement is integrated here: the prior
cumulative-target timing correction is not part of this instantaneous-rate cost.

For finite but undefined projections (XY norm²<1e−4), charge fixed cost1.0,
clear EMA, mark the sample invalid and restart the filter on valid recovery.
Do not fabricate good tracking, clip valid rates or pass an undefined pose in
qualification. Require unit quaternion tolerance1e−5 and finite compatible
input/derived tensors. Negative weight must produce actual reward=−cost before
autoreset, including terminal samples. Valid-only residual/rate/EMA means and
explicit invalid counts accompany all-sample cost/reward averages; eight signed
command buckets partition every transition with valid/invalid counts.

## Matched recipe, budget and unchanged gates

Prove the full remaining config matches steady after removing only each owned
reward/metric; record the different complete recipe hashes and bind the steady
full hash to its authenticated sealed summary before pausing any workload.
All other rewards/curricula, noise/delays, plant/BAM geometry, raw14D actions,
61D actor input, command sampler and optimizer settings stay fixed. The EMA is
reward state, never a runtime/action filter. No raw perception or physical motion.

Fresh original-v8 deterministic mean/frozen affine only, std.3, fresh critic/
empty Adam/clocks; match initial states to the same retained stock controls.
Seeds947/953/967; smoke64×5 → benchmark256×10 → fixed original-actor evaluation
→ foundation256×1000 → **final999 only** evaluation. LR1e−4/adaptive KL.01,
five epochs/four minibatches; save0/250/500/750/999. Before full training require
exact clean tested source/runtime/policy/manifest/control/baseline bytes,
complete finite small runs with normalized export and actual valid reward
activity, and benchmark projection<2070s. First update must have20 minibatches,
mean KL≤.05 and maximum minibatch mean≤.10; retain guard/refusal and aborted
state before stopping. No automatic parameter or budget retry on numerical failure.

Same held-out seeds839/853/857,8 worlds,240×.02s,startup60 and idle/.30 straight/
.20 with±.20yaw. Raw body/root XY/quaternion/contact traces saved first. Keep
legacy instantaneous-yaw decision and the full gait-aware route decision:
all cycles AND full settled endpoint/trajectory maxima, all12 cases, same
motor/hinge/rated-speed/idle/no-fall bars. Partial gains do not promote a skill.
Also retain BOTH original cautious and steady speed/idle references: moving
passes≥66/72, each command mean MAE≤its sealed reference+.02m/s, idle24/24
speed≤.03/yaw≤.10. The recent cumulative-heading result is a descriptive
comparison, not a replacement for either gate or a warm-start checkpoint.

Reference steady source `7f55ded7c615e0adda4f859e6e05dde40c447e66`, final PT
`d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f`, summary
`182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155`.
The nested cautious reference retains source `934bdff47a63d458f571189fe5db155fcd4fd8d0`
and summary `5c8542012dd31a42e6482300290de20ea1a90b68735a720f94f7c7b2f20f0b32`.

One nonenabled retained sequential user service on100.100 with stage caps
300/300/600/2100/600s, outer4500s,8GiB RAM/200% CPU/Torch20% GPU. Scope the
lease, UUID, memory, temperature and existing ownership as before. Preserve
DINO/unrelated workloads; pause/restore ONLY originally active user vLLM after
proven Duck exit, leaving other protected scopes as found. No100.98 changes,
package/driver installs, checkpoint search, videos, harder-skill promotion or
physical motion. Diagnose failures read-only before software-only repair.

Preparation:1624 affected local tests pass (18 existing warnings,46.80s),
including29 new focused checks and60 reward/retention checks. The actual MDP→
RewardManager→metric→terminal snapshot path is tested, alongside the independent
geometry oracle and old level-pose EMA equality. Compile, diff-whitespace and
local documentation links pass. Full steady reference hashes match the sealed
summary in all three modes; native exact-source tests and pre-pause preflight
remain separate. No learner had launched at predeclaration; the execution below
records its source independently from the later evidence-document revision.

## Completed execution and independent closeout

Executed source **`db9a77ae1baf27efa19a9fc9c16baacbc58f30dc`**, packet
`artifacts/training/v8-projected-yaw-bias-db9a77ae-20261014` on100.100. The
exact clean native source passed1624 affected CPU tests (18 existing warnings,
46.59s) and the pre-pause control/recipe/baseline preflight. Source implementation
and evidence packaging received separate read-only reviews; neither changed
the gates. The nonenabled sequential service completed with Result=success,
ExecMainStatus=0. This is **successful execution, not policy acceptance**.

Smoke64×5 took4.412s, benchmark256×10 took8.014s and original-actor evaluation
59.116s. Both small-run receipts independently verify finite state, normalized
export, active valid reward and guarded first-update KL. Foundation completed
all1000 updates /6.144M transitions in655.469s, saved0/250/500/750/999, and
exported a finite normalized61→14 ONNX. All17 Adam states have20000 steps;
critic count6144000, actor updater count0 and frozen affine unchanged. Initial
model states match the same stock control, with only the declared LR difference.
The reduced recipe hashes agree; the full steady hash matches its sealed source
reference. No intermediate checkpoint was evaluated or selected.

Foundation first-update mean KL.0102780 and maximum minibatch mean.0202781
pass the unchanged.05/.10 caps. The reward was active on all24000 control steps,
with6144000 valid and0 invalid projections. Mean cost.160415408 and actual
weighted reward−.160415404 agree within float arithmetic. All reward/optimizer
metrics are finite, with0 NaN terminations and1006 stochastic training falls.
Falls are higher than steady892 and cumulative-heading844; this is not an
unqualified training-stability improvement. The shared fixed budgets, one
training seed and nondeterministic GPU trajectories do not establish causality
or significance from these counts.

### Unchanged held-out decisions

Final-only evaluation took60.771s. All12 cases /96 worlds completed240
first-episode steps with no terminal, motor/range or idle-gate failure. BOTH
sealed walking/idle references pass: moving speed **67/72** (straight20/24,
negative yaw23/24, positive yaw24/24), idle speed/yaw **24/24**. The original
instantaneous-yaw gate still rejects all9 moving cases (moving yaw0/72).
The full gait-aware route gate also rejects all9 moving cases; **4/72** moving
worlds pass every component, versus steady3/72 and cumulative-heading2/72.
Idle route cases3/3 pass. No new integrated steering/obstacle/hopping/football
skill, video, policy publication or physical-motion approval follows.

| Command vx / yaw | Speed MAE: steady → projected-rate | Full XY endpoint/T: steady → projected-rate | Full heading endpoint/T: steady → projected-rate |
| --- | --- | --- | --- |
| .30 / 0 | .070188 → .070786 | .085446 → .095027 | .074679 → .112477 |
| .20 / −.20 | .048545 → .043801 | .053056 → .052003 | .081787 → .096306 |
| .20 / +.20 | .043694 → .046949 | .058954 → .057887 | .095747 → .097628 |

Speed/XY rates are m/s; heading rates rad/s. These are mean full-settled
ENDPOINT errors divided by interval duration, not trajectory maxima or full
qualification. Relative to steady, negative-turn speed MAE improves9.8%, while
positive-turn speed worsens7.4%. Turning XY endpoint error improves about2%,
but heading endpoint error worsens51% straight,18% negative and2% positive.
All three heading endpoint errors also worsen versus the cumulative-heading
round. Actual mean body vx.237835/.164670/.160506 and body yaw
−.013631/−.208444/+.156488 do not override the world-path result. The coordinate
change alone did not deliver the desired general steering gain.

Held-out maximum force utilization **.581466**, raw force p99 **.348880Nm**,
minimum named motor-hinge margin **.122988rad**. This leaves about.01112Nm
below the numerical.36Nm p99 cap, larger than the prior cumulative-heading
margin, but remains modeled short-test headroom, not measured hardware safety.
Stochastic training maxima are separate: utilization1.067539, soft-limit
fraction.073754, rated-speed exceedance.000709, mechanical power6.567563W,
thermal proxy.145772. The thermal proxy is not a measured motor temperature.

### Read-only failure decomposition and next direction

| Command | Full XY endpoint AND maximum passes | Full heading endpoint AND maximum passes | Complete cycles | Cycle heading endpoint failures | Additional peak-only heading failures |
| --- | --- | --- | --- | --- | --- |
| .30 / 0 |16/24 |12/24 |162 |89 |17 |
| .20 / −.20 |22/24 |14/24 |172 |77 |28 |
| .20 / +.20 |23/24 |12/24 |168 |74 |37 |

Every moving world has6–8 complete physical cycles, with no cycle-duration or
right-touchdown-count failure. Across502 cycles there are240 heading endpoint
failures and82 additional peak-only failures;38 cycle XY endpoint failures
(30 straight,3 negative,5 positive). Unlike the three missing-cycle worlds
in steady, the remaining problem is route accuracy, not absent gait coverage.
Fewer peak-only failures do not offset worse mean heading endpoints or qualify
the complete route. All original thresholds and rejected results remain intact.

Do not automatically extend this actor, stack these rejected reward changes,
or spend another budget on a blind weight/coordinate sweep. Next use the retained
steady/projected-rate/cumulative traces to localize signed heading drift and
within-cycle excursions by contact phase and command, then predeclare ONE
bounded gait-coordination lesson informed by that diagnosis. This is an offline
simulation diagnosis, not raw perception, an actor observation expansion or a
new acceptance definition. A new learner is not launched by this closeout;
harder skills remain behind the foundation and retention gates.

### Durable assets and scoped workload restoration

The native packet contains100 sealed files /255754153 bytes. PTs and complete
raw traces stay native; the Mac retains only35945-byte verified summary and
15079-byte inventory in
`artifacts/training/v8-projected-yaw-bias-db9a77ae-20261014-closeout/`. The
independent CUDA-hidden verifier recomputed checkpoint/optimizer clocks,
affine/init/recipe equality, raw hashes, legacy/route/BOTH retention decisions
and paired descriptive outputs against the exact clean executed source.
Its status is `verified-evidence-not-policy-acceptance`, with both policy and
physical authorization false. The evidence verifier is retained at
`/home/converge/.local/state/verify-projected-yaw-bias-db9a77ae-closeout.py`.

Max sampled GPU use1754MiB, minimum free14440MiB, maximum temperature67°C.
DINO PID2029 stayed intact. The scoped service proved Duck exit and DINO-only
GPU ownership before restoring ONLY original user vLLM, active with GPU engine
PID1696570 at9692MiB. User subject worker and both system scopes remain
inactive. At closeout the Duck service is inactive/success, no Duck GPU process
remains, and the native executed checkout is clean atdb9a77ae. No100.98 or
unrelated workload was changed.

Closeout checks:29 focused reward/driver/retention tests pass (6.76s), all67
local links in the two affected documents resolve, and diff-whitespace checks
pass. The native executed source stays pinned todb9a77ae; a later documentation
commit is not represented as the checkpoint's training source.

| Retained evidence | SHA256 |
| --- | --- |
| Foundation result | `dfc412c77fcf4a40ee37808ca275c1bdc9f2184770517bc73102ceb592676bf4` |
| Foundation initial PT | `5f9f8778078dd343df7533d5b6873c58989f1656807e67fbeff3d5a8a2c141c7` |
| Final model999 | `dc9f4861910c91ed8a58a11f62ad785908b9e5b152716ab2a775c27fa6b89fed` |
| Normalized foundation ONNX | `335d81e1c19cb602177a43a1a941ed2f52b960a2d25361ccf44332c20539d3a6` |
| Initial evaluation | `07674c55979a02ab3af7944078a4fbe5a73742216c8bed8d3ad0e7d8e68e8e89` |
| Final evaluation | `baf23e84b5e63e875268c5a2a73dab07d7c67e5c03bae2299f09b3913e220e1c` |
| Verified summary | `5c411b9d72a988a66245e34deda1dc53875180d414200479b85b21e366dd90fc` |
| Full packet inventory | `36e4a90ea17cea80fccf20adf951e927143ed5f37d6b4f20acc0facb808cadc7` |
| Restoration receipt | `d5fcd35d3f0e907d645b4d1006173059b65eaabb87bc74b0700dc83ac0e5c1f5` |
| Restoration idle proof | `8611b46dc6d2240b1a004bf1b4c9307bb4b82ae4aac208699e4bc15b51130e4c` |
| Restoration context | `a52660efe8c59afa7d9cec32c9700093d6e57e2c867d2c4eb68fd22179696094` |
| Independent verifier source | `1ecc36490245656ea80520d3b4ac3fdbb80d73196b2787207338b77d6bfb1cbd` |
