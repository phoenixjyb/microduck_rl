# V8 projected yaw-rate bias — prospective coordinate-frame contrast

Follow the owner's approval after the
[mixed cumulative-heading round](2026-10-13-v8-projected-heading.md). That
round retained walking/idle, but worsening straight tracking and2/72 complete
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
remain separate. No learner has yet launched from this declaration; record
executed source and later evidence-document revision independently.
