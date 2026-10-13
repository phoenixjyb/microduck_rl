# V8 projected-world heading lesson — prospective reward replacement

The owner approved the next lesson after the
[steady-steering round](2026-10-13-v8-steady-steering.md). That round retained
walking/idle and improved average heading, but all9 moving route cases still
failed:149 cycle endpoint heading failures,125 additional peak-only failures
and three turning worlds without enough physical cycles. Do not reinterpret
those results or relax their gates.

## Fixed hypothesis and recipe

Protocol `v8-projected-heading-v1`, declaration-date run ID
`v8-projected-heading-<source8>-20261013`. Make **one learning-recipe
replacement** against the steady recipe: remove `yaw_bias` (one-second EMA
body-yaw residual), add `heading_error`, fixed weight **−1.0**, reference time
**1.0s**, command `twist`. Use the actual world XY projection of the trunk's
body-X axis from the simulator's wxyz root quaternion, not body angular velocity.

At a new command segment, anchor actual and target heading to the **pre-action**
projected heading. Each reward step advances target by consumed twist yaw×.02s;
actual heading continuously unwraps successive projected angles. Positive cost
is `abs(target_unwrapped − actual_unwrapped) / 1.0s` (rad/s); actual weighted
reward must equal−cost. The error itself is never wrapped modulo2π, so missing
a complete turn cannot look correct. No EMA, action filter, angle clipping,
adaptive weight, checkpoint search or after-the-fact parameter tuning.

Per-world episode resets and any change of the consumed3D twist start a new
segment at the next accurate pre-action pose. Counter rollback clears history;
forward skips/missing priming refuse. Reward and metric repeated reads at the
same post-step must be idempotent, with independent cloned snapshots. Include
terminal samples before autoreset; never read their reset poses as outcomes.

For finite poses whose projected axis norm²<1e−4, heading is undefined. Charge
fixed positive cost1.0, mark projection-invalid and reanchor on valid recovery;
do not invent `atan2(0,0)` steering or reward the invalid state. Nonfinite inputs,
malformed/nonunit quaternions or invalid counters/dt/config refuse. The existing
qualification still rejects undefined heading; this training fallback is not a
tracking pass or hardware-safety claim.

Installed mjlab1.3.0 reward/metric root poses are stale by one physics substep.
The normal post-step `sim.forward` refreshes them before the next action. Account
for last-post→next-pre heading correction in continuous actual heading while
leaving target unchanged; otherwise the reward would lose a quarter of motion
at .005s/.02s. Do NOT force an extra simulation forward or change any other
reward's timing. A synthetic constant-turn regression must show only the fixed
one-substep phase offset, not accumulating lost motion.

This is privileged **reward state only**. The61D actor does not gain absolute
world heading, position, target history, raw perception or recurrent memory.
The experiment tests whether better incentives produce a more consistent
steering gait; it does not establish general pose-aware route correction.
Keep all other rewards, curricula, plant/BAM settings, raw14D actions, actor
layout/noise/delays, command sampler, frozen affine, std.3, fresh critic/Adam/
clocks and cautious LR1e−4/adaptive KL.01 unchanged. Prove complete config
equality after removing each recipe's owned reward and read-only metric; record
the different full recipe hashes honestly with stable CLI/import callable names.

## Budget, prerequisites and decisions

Fresh original-v8 deterministic mean only, not rejected final PPO state. Match
initial actor/critic bytes and empty17-parameter Adam layout against the same
retained stock controls (seeds947/953/967), with only the already declared
initial-LR difference. Smoke64×5 → benchmark256×10 → original actor baseline
evaluation → foundation256×1000 → **only final999** evaluation. Save
0/250/500/750/999. Same caps300/300/600/2100/600s, sequential outer4500s,
8GiB RAM/200% CPU/Torch20% GPU, no restart/enable or overlapping GPU learner.

Before the full budget require exact clean tested source, authenticated runtime,
policy/manifest/control/baseline bytes, finite complete smoke/benchmark and
normalized export, active actual reward telemetry and benchmark projection
<2070s. First update retains20 raw minibatches and must have mean KL≤.05 and
maximum minibatch mean≤.10. Write guard evidence and explicit aborted checkpoint
before refusal; no automatic rate/weight/budget retry after numerical failure.

Capture actual heading error, target/actual unwrapped heading, cost, weighted
reward, valid-projection and segment-reset masks after reward/before reset.
Require24 snapshots/update, finite cost≥0, reward≤0, reward=−cost, complete
transition-weighted eight signed-command buckets and actual command dwell.
Report invalid-projection exposure explicitly rather than labeling it zero error.

Held-out protocol is unchanged: seeds839/853/857,8 worlds,240×.02s,startup60,
commands idle/.30 straight/.20 with±.20yaw. Save raw world XY, quaternion,
contact-cycle and body traces first, hashes, legacy and full route decisions.
Every cycle AND the full settled interval must pass endpoint and trajectory
maximum checks; existing motor/hinge/rated-speed/fall/idle gates stay fixed.
Only a complete12-case route matrix qualifies a simulation foundation; source,
training completion, numerical qualification, publication and physical motion
remain separate. No video, obstacle/hop/football promotion or physical motion
follows from a partial numerical gain.

Keep **both** speed/idle references: original cautious final999 and latest
steady final999. Each requires moving speed passes≥66/72, EACH command's mean
body-speed MAE≤its sealed mean+.02m/s and idle all24 at speed≤.03/raw yaw≤.10.
The latest comparison is an additional bar, never a replacement that weakens
the existing cautious gate. Authenticate both summaries before launch.

Latest steady source `7f55ded7c615e0adda4f859e6e05dde40c447e66`, final PT
`d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f`, summary
`182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155`.
Cautious source `934bdff47a63d458f571189fe5db155fcd4fd8d0`, final PT
`820c86e28b3d05d8bd292b63777b78cfa0b4657d69cc028ad17f8afe1ab91738`, summary
`5c8542012dd31a42e6482300290de20ea1a90b68735a720f94f7c7b2f20f0b32`.

Preserve DINO/unrelated workloads on100.100; only pause originally active
**user** vLLM under the existing scoped lease/UUID/ownership fence, and restore
it after proven Duck GPU exit. Leave subject-worker/system scopes inactive as
found. Do not touch100.98/NX/FilmBrain, packages/drivers, physical devices or
stale September schedules. Diagnose failures read-only before fixes.

Preparation:1595 affected local tests passed (18 existing warnings,45.23s),
and58 focused tests pass after final receipt-count review. These include actual
MDP→RewardManager→metric→terminal snapshot consumption, ±3rad/s500-step
stale-pose/wrap regressions, metadata/layout/rollback refusal, normalized actor
import, recipe namespace identity and both retention controls. Compile,
diff-whitespace and local document links pass. The new reduced recipe matches
the exact sealed steady recipe hash in all three modes, not merely a mutable
factory definition. Exact native checks and CPU preflight precede any pause.
No learner launched from this declaration yet. Report the executed source
separately from any later evidence-document commit.
