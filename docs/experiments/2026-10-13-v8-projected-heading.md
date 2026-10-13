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
the reduced steady recipe in all three modes. Separately, preflight binds the
FULL steady reference recipe hash to the authenticated sealed summary, not
merely a mutable factory definition. Exact native checks and CPU preflight
precede any pause. The completed outcome below is distinct from preparation;
the later evidence-document commit is not the executed source.

## Completed round — mixed result, not promoted

Closeout recorded2026-10-14 Asia/Shanghai for the Oct13 declaration-date run.
Execution source **`fe23cc80fa9f4d624c8d79d792e0f45281239580`**, clean native
checkout, passed **1595 affected native tests** (18 existing warnings,45.89s)
and the pre-pause CPU control/recipe/baseline checks. Independent imported-module
validation also accepted smoke/benchmark receipts; the original cautious
retention decision recomputes exactly unchanged from the old sealed raw cases.
No weight, rate, sampler, plant, observation or qualification gate was retuned.

Retained native packet:
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/v8-projected-heading-fe23cc80-20261013`.
Sequential user service
`recomo-duck-v8-projected-heading-fe23cc80-20261013.service` finished inactive,
`Result=success`,exit0. Durations:4.416s smoke,7.974s benchmark,58.900s initial
evaluation,**657.207s foundation**,60.546s final evaluation. Foundation completed
**1000 updates /6,144,000 transitions**,844 exploratory training fall terminations,
0 NaN terminations and0 invalid-heading samples. The first-update KL mean
**.01028152**, maximum minibatch mean **.02059027**, passes the fixed .05/.10
guard; smoke/benchmark passed their own guards.

All1000 recorded update rows are finite. Actual heading reward activity covers
24000/24000 foundation steps; mean cost.35839354, actual weighted reward
−.35839354,6144000 valid samples and33530 segment-reset samples. This is not
the same quantity as the old EMA cost, so their magnitudes are not an improvement
score. The independent CPU verifier authenticates source/clean checkout,
controls and full reference recipe hashes, every retained file, initial/final
PTs, frozen actor affine/count0, critic count6144000,17 Adam states at20000
steps, environment clock24000, save cadence0/250/500/750/999 and normalized
export. Verification is **evidence packaging**, not a requirement that a
numerical gate pass or a policy-acceptance certificate.

### Held-out comparison

All12 final cases (96 first episodes) complete240 steps without a fall/terminal
or motor/hinge/rated-speed gate failure. Idle passes3/3 cases and24/24 speed/yaw
worlds. **Both** sealed walking/idle retention references pass: moving speed
66/72, each command within its reference mean+.02m/s. Straight speed passes
18/24 (steady21/24); left/right each24/24 (steady22/24 and23/24). Legacy moving
yaw still passes0/72, and all9 moving cases still fail complete route
qualification. Only **2/72** moving worlds pass every route component (steady3/72).

| Moving vx / yaw | Steady speed MAE | New MAE | Steady full heading error / T | New heading error / T | Steady full XY error / T | New XY error / T |
| --- | --- | --- | --- | --- | --- | --- |
| .30 / 0 |.070188 |.078519 |.074679 |.100865 |.085446 |.098923 |
| .20 / −.20 |.048545 |.038625 |.081787 |.083080 |.053056 |.046338 |
| .20 / +.20 |.043694 |.037378 |.095747 |.085890 |.058954 |.050854 |

Speed/XY rates are m/s and heading rates rad/s. These are mean full-settled
ENDPOINT errors divided by interval duration; they are not trajectory maxima
or full qualification. Turning speed MAE improves roughly20%/14%, turning XY
error13%/14%, and right heading error10%. Straight heading error worsens35%
and XY error16%; left heading error is roughly unchanged/slightly worse.
Actual body forward speeds.223949/.166000/.168542 and body yaw
−.014098/−.189108/+.164172 do not override the measured world-path result.
One training seed, limited held-out commands and nondeterministic GPU
trajectories support a descriptive contrast, not a statistical or physical claim.

Held-out maximum force utilization **.599805** (dimensionless), raw force p99
**.359883Nm**, minimum named motor-hinge margin.089597rad. The accepted raw
p99 cap is.36Nm, leaving only about.00012Nm in this test: do not interpret a
pass as motor robustness or measured hardware headroom. Stochastic training
maxima differ: utilization1.061131, soft-limit fraction.062698, rated-speed
exceed fraction.000605, absolute mechanical power5.979826W, thermal proxy
.135098. The proxy is not measured motor temperature.

### Read-only failure decomposition and next direction

| Command | Full XY endpoint AND maximum passes | Full heading endpoint AND maximum passes | Complete cycles | Cycle heading endpoint failures | Additional peak-only heading failures |
| --- | --- | --- | --- | --- | --- |
| .30 / 0 |13/24 |13/24 |163 |76 |36 |
| .20 / −.20 |24/24 |16/24 |166 |66 |54 |
| .20 / +.20 |23/24 |16/24 |165 |76 |51 |

Every moving world now has6–8 physical cycles, with no cycle-duration or
right-touchdown-count failure. This removes the three deficient physical-cycle
worlds seen in the steady run, but accuracy still fails:218 cycle heading
endpoint failures and141 additional peak-only failures across494 cycles.
Straight has44 cycle XY endpoint failures; turning cycles have none. No
threshold was relaxed and no alternative definition/checkpoint was selected.

Do not automatically extend this actor or call the reward replacement a broad
success. Preserve both rejected checkpoints for comparison. The cumulative
command-relative target is hidden reward state, not an actor observation;
this experiment does not establish that the61D feed-forward policy can correct
arbitrary accumulated heading error. Next isolate **projected-world yaw-rate
bias tracking** while retaining the successful one-second bias filter and
walking/idle bars. That would separate the coordinate-frame change from this
round's cumulative-target change, using local orientation/rate information
already represented by proprioception. Predeclare/test that bounded contrast
before another budget; do not add absolute heading/position observations or
relax route gates without a separately approved contract change. The next
learner is not implemented or launched by this closeout. No harder skill,
MP4, policy publication, raw perception or physical motion is authorized here.

### Retention and workload evidence

The native packet has100 sealed files /255584862 bytes. Full PTs/raw traces
remain native; Mac retains only36117-byte summary and15079-byte inventory in
`artifacts/training/v8-projected-heading-fe23cc80-20261013-closeout/`.
Max sampled GPU use1754MiB, minimum free14440MiB, temperature67°C. DINO
PID2029 stayed intact; only original user vLLM was paused/restored, active with
GPU engine PID1252086 at9692MiB after Duck exit. User subject worker and both
system scopes remain inactive. Restore context, DINO-only idle proof and
restored GPU ownership are separately hash-bound. The independent verifier is
`/home/converge/.local/state/verify-projected-heading-fe23cc80-closeout-v2.py`;
it runs CUDA-hidden against the exact clean executed checkout and recomputes
route/legacy/both retention decisions, including rejections.

| Retained evidence | SHA256 |
| --- | --- |
| Foundation result | `b0e88b27803ed9261aeb60b22f5617ac572bd796e5fdbdc34103b6870de2a80e` |
| Foundation initial PT | `5f9f8778078dd343df7533d5b6873c58989f1656807e67fbeff3d5a8a2c141c7` |
| Final model999 | `95dde4bcf89c13462efa9c0f599b20239a13058dd596718360e09668d2b97440` |
| Normalized foundation ONNX | `c2e93d538ecd5e2178e40377161bb868a64c7b812a8ad751ac1cf92f0608632f` |
| Initial evaluation | `6246b58b9b384bf358ab00dc515fd3b5c676a8296b66ac55e5b5945064f2ae83` |
| Final evaluation | `50c1fd17fd3a35d05b7511dd503d8f9b9f63ac5903a5bd9a4f2696018c05b605` |
| Verified summary | `8a64c7b07c410f4f4adc01c72422532e2be1c843a6ac8cb3be25eeb5d59ffcad` |
| Full packet inventory | `30aa449966a0efef0ac0ec8d50795564f72234cd2e4cf2f214deee61681e5e97` |
| Restoration receipt | `fa1d20d3ec0fa3168620869282aeab83db86870d39fd7c736bfde578d14aecb6` |
| Restoration idle proof | `c7abea7f0225772e60066877b9bc04fb088f7eeefef34338d603bdce7a9cea3c` |
| Restoration context | `8179fddfe202dc0046eec32175eb6a36bf4049e852f9e89db30efdde8898f4cf` |
| Independent verifier source | `72ee1c7d969710c5fd91a88ca0ab11b70d5b2aa61f6e31f97c25937be384c046` |
