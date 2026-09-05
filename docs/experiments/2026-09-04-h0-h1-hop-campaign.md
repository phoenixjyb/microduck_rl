# H0/H1 matched-mechanics hop campaign

Date: 2026-09-04

Decision: **runtime smoke and matched-arm diagnostic pilot passed; the full
K3900 H1 matrix and bounded H1-P revision were rejected; no hop policy accepted**

## Current evidence

The periodic-hop task is already registered for three mechanically matched
70 g boot arms:

| Arm | Spring | Travel | Purpose |
|---|---:|---:|---|
| Locked | locked | 0 mm | geometric and distal-mass control |
| K2500 | 2,500 N/m | 12 mm | softer non-bottoming candidate |
| K3900 | 3,900 N/m | 12 mm | stiffer measured-model candidate |

All arms share the same learner configuration, one-second phase command,
reward budget, action-rate cost, robot mass boundary, and observation/action
spaces. The task measures rise above the last stance sample rather than an
absolute body-height datum. Airborne reward requires both feet off the ground
and positive body rise, preventing foot flutter or a stationary tuck from
counting as a hop.

The implementation exposes spring energy, loaded compression, bottoming,
airborne rise, landing force, fall, and numerical metrics. The repository suite
passes 612 tests, including cross-arm configuration and stateful rise-tracker
tests. No hop checkpoint, TensorBoard run, or retained hop artifact exists on
100.100, so source readiness is not training evidence.

The assumed damping ratio remains the documented simulation uncertainty. No
result from this campaign may be used as a physical spring prediction or motion
authorization.

## Stage naming

This campaign follows the accepted capability specification:

- H0: matched mechanical/configuration controls;
- H1: periodic in-place hop;
- H2: commanded one-shot jump;
- H3: running jump with stable landing and speed recovery.

Low-obstacle jumping follows only after H3. It does not replace the independent
structured-geometry bypass branch.

## Predeclared runtime smoke

Run only K3900 first with seed 42, 64 environments, and five learning
iterations. This is a wiring smoke, not a training result. It must establish:

- successful task construction and checkpoint/log creation;
- finite observations, rewards, losses, actions, and hop metrics;
- present `hop_spring_energy_*`, `spring_compression_loaded_mean`,
  `spring_bottomed_fraction`, `hop_rise_*`, landing, and fall metrics;
- no NaN termination or non-finite optimizer state;
- no competing GPU service and no physical motion.

A five-iteration policy is not expected to hop and cannot pass H1.

## Bounded matched-arm pilot after smoke

Only if the runtime smoke passes, predeclare one diagnostic seed per arm before
launching it. Locked, K2500, and K3900 must use identical environment count,
iteration budget, checkpoint cadence, and learner settings. Read metrics in
this order:

1. spring energy and loaded compression, to prove that compliant arms store
   energy;
2. bottoming fraction, to reject a soft arm that simply slams the end stop;
3. both-feet-airborne fraction and rise distribution;
4. landing survival/fall rate;
5. action rate and motor envelope.

No height comparison is interpretable if the compliant arm stores no energy or
regularly bottoms. A sprung arm must be compared with Locked, never with an
unmatched rigid running policy. Long 8,000-iteration, three-seed training is
not authorized by this smoke; its budget and promotion matrix will be fixed
only after the matched pilot demonstrates a real hop signal.

No MP4 is recorded until a numerical H1 candidate survives held-out evaluation.

## Runtime-smoke result

The exact K3900 smoke completed all five iterations on the 100.100 RTX 4090
Laptop GPU with seed 42 and 64 environments. It wrote `model_0.pt`,
`model_4.pt`, ONNX, parameters, Git identity, and TensorBoard events under
`logs/rsl_rl/hop_k3900/2026-09-04_16-48-33_h1-smoke-f209d75-k3900-s42/`.

At the final logged iteration:

- hop spring energy mean/peak was 0.0076/0.0611 J;
- loaded spring compression mean was 0.0015 m and p95 compression was
  0.0025 m;
- bottomed fraction was zero;
- airborne rise mean/peak was 0.0025/0.0094 m;
- landing-force mean was 6.8620;
- NaN termination was zero and every reported scalar was finite.

The untrained five-iteration policy still fell frequently
(`Episode_Termination/fell_over = 2.1667` in the final interval). That is
expected smoke evidence, not H1 performance. The smoke proves that the spring
stores measurable energy and that the logging path is live; it does not prove a
repeatable hop.

Retained artifact SHA-256 values:

- `model_4.pt`:
  `d2e59fe45a5dd66b95cd19ef527d75efeef90efd7c853c7105f5b5d0e365d412`;
- TensorBoard event:
  `f717f82b6e17f14357d01ee7287691b83d4b54d8c19b1c0cd44dbb01fd3de173`;
- ONNX:
  `44f06b9057d80406137fe957d23364dcc7e341aeeb1dad6ee0a02f0ddb08cb33`.

## Predeclared matched-arm pilot

The next bounded diagnostic uses the same seed 43 for Locked, K2500, and K3900,
256 environments, 256 iterations, and a 16-iteration checkpoint interval.
Every arm starts from scratch with identical PPO settings; no checkpoint is
warm-started from the smoke or from another arm.

Read retained checkpoints at common iterations 64, 128, 192, and 255. Stop the
line before a long campaign if all three arms remain effectively stationary,
if any scalar becomes non-finite, if a compliant arm has no measurable loaded
compression/energy, or if bottoming is persistently at least 1%. The pilot is
diagnostic only: a promising signal still requires a separately fixed
three-training-seed budget and held-out H1 evaluation.

## Matched-arm pilot result

All three seed-43 runs completed the fixed 256-environment, 256-iteration
budget successfully on the 100.100 RTX 4090 Laptop GPU. They used the exact
`fe2ea3a612bec10f314436fd83e3952611c7a546` source snapshot, started from
scratch, and wrote checkpoints at the common iterations 64, 128, 192, and 255.
Every inspected scalar was finite and NaN termination stayed zero.

The common iteration-255 comparison is:

| Arm | Rise mean | Rise peak | Spring energy mean/peak | Loaded compression | Compression p95 | Bottomed | Landing force | Fall scalar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Locked | 0.00005 m | 0.00101 m | 0 / 0 J | 0 m | 0 m | 0% | 9.179 | 10.208 |
| K2500 | 0.01600 m | 0.05431 m | 0.0194 / 0.3633 J | 0.00354 m | 0.00578 m | 0.858% | 11.637 | 13.167 |
| K3900 | 0.01443 m | 0.06947 m | 0.0136 / 0.2604 J | 0.00239 m | 0.00388 m | 0.034% | 12.817 | 13.333 |

Locked remained effectively stationary while both compliant arms developed a
centimetre-scale rise that grew across the common checkpoints. This passes the
diagnostic signal gate: compliance stores measurable energy and enables a
behaviour that the mass-matched locked control does not discover. K2500 stayed
below the 1% bottoming stop line, but ended only 0.142 percentage points below
it and its bottoming rose monotonically from zero at iteration 64 to 0.858% at
iteration 255. K3900 produced the largest peak rise while retaining much more
travel margin, so it is the sole candidate for the next H1 training gate.

The reported `fell_over` values are interval logger scalars, not normalized
episode failure rates. They remain high enough that this pilot cannot establish
repeatable takeoff, landing survival, or H1 acceptance. No MP4 is recorded.

Retained iteration-255 artifact SHA-256 values:

| Arm | `model_255.pt` | TensorBoard event | ONNX |
| --- | --- | --- | --- |
| Locked | `f7e23c0ff6d44747f14fd9d47dd18745432ef7458f742bc88e46fc2a4caa3457` | `76890a9fd1163cc1589af2e416784568982f5c9385f6744321d6a853fa43adf2` | `c5aa0df2e39276118d9d15ce6dd393d5e43e78009d0e0045ff538fde0d0d961b` |
| K2500 | `0eb128a3c8f3cca1dd25957cda28595099dfda600dede3002e9b5335fb333c3a` | `d70fd33123a1ec60651d4c8b23a2cc26d22726004036967502d069d236aa367d` | `e8217bdaaa278ae688328be6d8a5599889bafb941dacf50db4ad5fa1d113c01e` |
| K3900 | `126ede4469c6258e8e4e90ba5060fa0a1a6f631219f5c591953aafd9f0af0872` | `57f851e02f1c0c1c843798e4d100187b0a27a7c88d13f84a2dcd96126fdd49dc` | `835c8860bbda0d8c07576bc5c58af84dabae4b35049cb22767389831161deaab` |

## H1 held-out evaluation protocol

`scripts/evaluate_hop_checkpoint.py` implements protocol
`H1-periodic-hop-heldout-v1`. It uses held-out seeds 211/223/227, 128
environments per seed, six one-second command cycles, phase zero at episode
start, no push event, and zero auxiliary head/body commands. Only the first
episode of each vectorized environment counts: all states after its first done
are masked so an automatic reset cannot hide a fall.

The evaluator reports completed and successful cycles, both-feet-airborne rise
mean/p50/p95/peak, qualified-hop landing fraction, per-episode maximum planar
drift, falls, non-finite state, spring bottoming, touchdown-force p95/peak,
action magnitude/rate, and the running campaign's full motor envelope. It has
the following fixed all-seed promotion gates:

- cycle success at least 90% per seed and qualified-hop landing at least 98%;
- at least 80% of first episodes complete all six successful landed cycles;
- median cycle rise at least 0.02 m and peak rise no greater than 0.10 m;
- zero falls, NaN terminations, non-finite episodes, and rated motor-speed
  exceedance;
- planar-drift p95 no greater than 0.10 m;
- spring bottoming no greater than 1%, torque-utilization p99 no greater than
  0.90, and near-stall exposure no greater than 0.25%.

Landing force and action metrics remain mandatory evidence but are not given an
invented physical threshold in simulation. The current gates constrain their
observable consequences through landing completion, no-fall, drift, spring,
and motor limits. A later hardware phase must supply physical force/thermal
limits before motion.

### Runtime-smoke result

The evaluator ran its complete 3 x 128 x 6 matrix against the seed-43 K3900
pilot checkpoint at iteration 255. It retained JSON and CSV successfully and
rejected the checkpoint as expected. In every seed, all 128 first episodes
performed one qualifying landed hop and then fell before completing the first
cycle: cycle success was 16.67%, complete-episode success was zero, and the
aggregate fall count was 384. Drift p95 was 0.900--0.964 m. Maximum observed
rise was 0.1196 m, rated-speed exceedance was 0.981--1.135%, torque-utilization
p99 was 1.0675, and near-stall exposure was 22.04--22.81%. Spring bottoming and
non-finite counters remained zero.

This confirms both the measurement path and the reason the short pilot was not
accepted: it learned a single unstable launch, not periodic in-place hopping.
The retained evaluator outputs are:

- JSON SHA-256:
  `58236d30d84357f9817fd09cb09a974b29e3694cb80d0bb25d355fac66993aca`;
- CSV SHA-256:
  `b054d4aed7759eeda015a595dd82b4f1b428abcccaf5c197287bb22d49714e95`.

## Predeclared K3900 H1 training campaign

Train K3900 from scratch with seeds 47, 53, and 59. Each run uses 256
environments, 8,000 iterations, and a 500-iteration checkpoint interval under
otherwise identical PPO/task settings. Runs are sequential on 100.100; the
other two seeds do not share its GPU concurrently.

Evaluate the common iterations 500, 1,000, ..., 7,500, and 7,999 using the
immutable H1 protocol above. The earliest common iteration advances only when
all three training-seed policies independently pass every held-out gate. At
that iteration, choose the representative policy by highest minimum held-out
episode-pass fraction, then highest minimum cycle-success fraction, then lower
maximum torque-utilization p99, then lower training seed. No final-checkpoint
default or visual selection is allowed.

`src/mjlab_microduck/hop_checkpoint_sweep.py` enforces that rule from retained
evaluation JSON. It rejects incomplete seed sets, unexpected held-out seeds or
protocol identity, checkpoint/iteration mismatches, stored decisions that
disagree with recomputed gates, and any artifact that drops the no-motion
boundary.

`scripts/evaluate_h1_campaign.py` is the fail-fast campaign driver. It verifies
that every one of the 48 predeclared checkpoints exists before allocating the
GPU, evaluates in iteration-major three-training-seed blocks, reuses only JSON
whose checkpoint hash and complete H1 protocol identity match, then writes the
manifest and invokes the selector above. An interrupted evaluation therefore
resumes without silently mixing checkpoints or leaving a misleading partial
all-seed iteration.

K2500 remains a documented mechanical sensitivity arm, not a co-candidate;
revisit it only if K3900 cannot pass H1. No MP4 is recorded until the numerical
matrix accepts a representative checkpoint.

## Full K3900 matrix result

All three 8,000-iteration training runs completed and retained every common
checkpoint at iterations 500, 1,000, ..., 7,500, and 7,999. The sequential
campaign evaluator then completed all 16 iterations x 3 training seeds = 48
checkpoint evaluations under held-out seeds 211/223/227. The deterministic
selector rejected the matrix: no iteration had three independently accepted
training-seed policies, `selected_checkpoint_iteration` is null, and physical
motion remains unauthorized.

Retained evidence on 100.100:

- output directory:
  `artifacts/evaluations/h1-k3900-6c4e470-full/`;
- campaign manifest SHA-256:
  `562e100c2b19b1c28024d435d2ab218c77a1a0c4cd06b07d04d9e99cbecc8dca`;
- deterministic sweep SHA-256:
  `5525c18052f584db540af0f46baea34e05e546566080670d907bc0aa8671677f`.

The closest causal diagnostic was the seed-47 policy at iteration 6,000. Its
minimum held-out cycle-success fraction was 98.828% and its maximum torque
utilization p99 was 0.8735, so the policy learned the repeated launch/landing
rhythm and passed the torque-p99 gate. It still had zero fully passing first
episodes, five falls, 0.4995 m maximum held-out drift p95, 1.148% maximum spring
bottoming, 0.0639% maximum rated-speed exceedance, and 0.5356% maximum near-stall
exposure. Its retained evaluation JSON SHA-256 is
`eaff48a912d08b7f488b6070dc46f8f3c23ecb5fe2dbd89b8ae6b776b8cc1a10`.

The evidence separates discovery from acceptance: mature policies reached
roughly 96--99% minimum cycle success, but none learned a stable, in-place,
motor-envelope-compliant six-cycle episode across all held-out seeds. The base
hop task instruments the motor envelope but does not put the existing
`motor_torque_load_cost` on the reward path. A longer repeat of the same task is
therefore not justified. Locked, H2, MP4 generation, and physical motion stay
closed.

## Predeclared H1-P revision

The smallest next revision is one K3900-only, from-scratch H1-P task that keeps
the same 61-D observation, 14-D action, one-second phase, simulator mechanics,
PPO configuration, and immutable held-out evaluator. It changes only the
training objective in two directly diagnosed ways:

1. register the existing normalized motor-torque/thermal-load cost at the same
   0.70 soft-limit fraction and 4.0 over-limit gain already tested by the
   motor-aware Run stage, with fresh-run weights -0.25, -0.75, -1.25, and -2.00
   at iterations 0, 1,000, 2,500, and 4,000;
2. progress one coherent height envelope at iterations 0, 2,000, and 4,000:
   `(target rise, Gaussian std, upward-velocity saturation)` =
   `(0.020 m, 0.010 m, 0.70 m/s)`, `(0.030 m, 0.015 m, 0.85 m/s)`, then
   `(0.040 m, 0.020 m, 1.00 m/s)`.

Each height stage keeps `std = target_rise / 2`, preserving the original
near-zero reward at zero rise, and keeps ballistic saturation above the target.
This avoids weakening the anti-tuck shape while asking the policy to establish
a lower-energy stable cycle before the final 40 mm objective. No task geometry,
acceptance threshold, or perception input changes.

Implementation must first pass focused curriculum/config tests and a five-
iteration CPU/runtime smoke. Only then may a bounded seed-67 K3900 diagnostic be
launched. Its predeclared checkpoints are iterations 500, 1,000, 2,000, 3,000,
4,000, and 5,999; the unchanged H1 evaluator is applied to each in order. A
three-seed promotion campaign is not authorized by that diagnostic and must use
fresh seeds fixed in a later experiment document. If no diagnostic checkpoint
improves episode survival while reducing drift and motor-envelope violations,
stop rather than spending GPU time on Locked or H2.

### H1-P source and runtime smoke

Commit `6ed782233e1ad70224d57e3a632443a1d2d20059` implements the distinct
`Mjlab-Hop-H1P-Flat-Sprung-K3900-MicroDuck` task. Focused H1-P configuration
tests pass 6/6 on both the development Mac and 100.100; Python compilation and
the repository diff check also pass. The registered task keeps the H1
observation/action contract and K3900 scene while exposing both curricula at
their predeclared initial values.

The seed-66, 64-environment, five-iteration GPU runtime smoke completed with
service result `success` in three seconds. It logged finite reward and optimizer
scalars, `Curriculum/hop_height_envelope = 0.0200`,
`Curriculum/motor_torque_load_weight = -0.2500`, the three motor-training
metrics, zero NaN termination, and zero spring bottoming. Frequent falls are
expected from an untrained five-iteration policy and are not H1 evidence.

Retained smoke directory on 100.100:
`logs/rsl_rl/hop_k3900_h1p/2026-09-05_00-17-31_h1p-smoke-6ed7822-s66/`.
SHA-256 values:

- `model_4.pt`:
  `d78875671be30e5d5148afd36e81330a47701d036c04af13422d20e4588d646d`;
- ONNX:
  `a9f7e7e695492c4e375e3954015385e3e9e6db85191fe384a26c604bb09ee302`;
- TensorBoard event:
  `52f26b1d518b85e7ab0e123677e47d0895249eebe8305318ec066ec752828e66`.

This passes the H1-P wiring gate only. It authorizes the already predeclared
seed-67 diagnostic, not a multi-seed promotion campaign, Locked control, H2,
video, or physical motion.

### H1-P seed-67 diagnostic result

The fixed 256-environment, 6,000-iteration diagnostic completed successfully
in 64 minutes. All reported training scalars stayed finite, NaN termination
remained zero, and checkpoints 500/1,000/2,000/3,000/4,000/5,999 were retained.
The unchanged H1 evaluator then completed its held-out seeds 211/223/227 for
all six checkpoints. Every checkpoint was rejected.

| Iteration | Min cycle success | Min episode pass | Falls | Max drift p95 | Max bottoming | Max rated-speed exceed | Max torque p99 | Max near-stall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 16.67% | 0% | 384 | 0.933 m | 0.539% | 1.179% | 1.068 | 8.680% |
| 1,000 | 40.49% | 0% | 370 | 0.936 m | 0.107% | 0.084% | 0.945 | 0.980% |
| 2,000 | 67.19% | 0% | 237 | 0.879 m | 0.015% | 0.099% | 0.848 | 0.376% |
| 3,000 | 85.68% | 0% | 95 | 0.899 m | 0.077% | 0.042% | 0.890 | 0.681% |
| 4,000 | 91.54% | 0% | 53 | 0.721 m | 0.033% | 0.018% | 0.870 | 0.634% |
| 5,999 | 96.61% | 0% | 23 | 0.594 m | 0.431% | 0.021% | 0.834 | 0.464% |

At iteration 5,999, median rise was 0.0458 m and the cycle, landing, rise,
rise-ceiling, spring-bottoming, torque-p99, NaN, and finite-state gates passed.
The episode-pass, fall, drift, rated-speed-exceedance, and near-stall gates did
not. The progressive motor-aware objective reduced bottoming, rated-speed
exposure, torque p99, and near-stall exposure relative to the earlier seed-47
iteration-6,000 diagnostic, but it did not improve whole-episode survival or
drift: episode pass stayed at zero, falls rose from 5 to 23, drift p95 rose from
0.499 m to 0.594 m, and minimum cycle success fell from 98.83% to 96.61%.
Because this was one training seed per objective, the comparison diagnoses the
revision; it is not a seed-consensus effect estimate.

Retained output directory on 100.100:
`artifacts/evaluations/h1p-k3900-e7b15bb-s67-diagnostic/`. Evaluation JSON
SHA-256 values by iteration are:

- 500: `14086086cb05896cbcedb7d3719c17ed188019188a84007911acf27a80b468b6`;
- 1,000: `c2223a6fda344191eb4863413317f809bd61ce2188accdca557774fe9a957dbf`;
- 2,000: `e53f0fc8f2b40daf99d95d12060ff5ba69676ddfa87b472e6f876300b989127a`;
- 3,000: `1596bf0b27d1551448ab677474bff698b92a4d8620288e3daaf1ff2d4e2c9643`;
- 4,000: `c8f9061fa929989dbb222f1b94778684e706b1d04625f5e251ef1f74306e4fac`;
- 5,999: `99907a27954ba31b2d39fcdf427e0c947b4f1ddbb9b9dc82096ff92f411494c3`.

The predeclared stop condition is met: H1-P reduced motor demand but did not
improve episode survival and drift together. Do not spend more GPU time on this
objective, a Locked control, or H2. The next work is a separately reviewed
stability diagnosis of lateral drift and falls; it is source/design work, not
authorization for another training run, video, or physical motion.

## Predeclared source-only H1-S stability revision

The first stability audit found one direct training/evaluation mismatch. The
periodic command is `[cos(2*pi*phase), sin(2*pi*phase), 0]`, whose planar norm
is always one. The inherited `stillness_at_zero_command` term pays only when
that norm is below 0.01, so its configured weight 3.0 contributes exactly zero
throughout every H1 and H1-P episode. At the same time, hop construction removes
both velocity-tracking rewards because they would interpret phase as desired
translation. No remaining reward directly prices planar body velocity, while
the held-out protocol rejects drift p95 above 0.10 m. This explains why the
policy can optimize repeated hops without optimizing the in-place criterion.

The actor also omits base linear velocity (the critic retains it as privileged
state). Adding actor state or a world-position tether would be a larger policy-
contract change and is explicitly out of scope for this first diagnosis.

H1-S is therefore one source-only change on top of H1-P: replace the dead
command-gated stillness function with the same Gaussian applied to planar body
speed on every phase, retaining its existing weight 3.0 and `vel_std = 0.07`.
It changes no observation, action, simulator mechanic, hop/motor curriculum, or
held-out H1 gate. The distinct task is
`Mjlab-Hop-H1S-Flat-Sprung-K3900-MicroDuck` so historical H1/H1-P configurations
remain reproducible.

This source slice does not authorize GPU use. If later approved, run a seed-68,
64-environment, five-iteration wiring smoke first, then a paired seed-67,
256-environment, 6,000-iteration diagnostic with the same checkpoint cadence
and evaluator. Before any multi-seed promotion, its iteration-5,999 result must
strictly improve H1-P's episode pass above zero, falls below 23, and drift p95
below 0.594 m while not regressing cycle success, spring bottoming, rated-speed
exposure, torque p99, or near-stall exposure. Full H1 acceptance thresholds do
not change. Failure of that causal gate stops H1-S; it does not authorize an
actor-observation change, Locked, H2, video, or physical motion.

The causal comparison is encoded by:

```console
python -m mjlab_microduck.hop_revision_gate BASELINE_JSON CANDIDATE_JSON \
  --output OUTPUT_JSON
```

It accepts only the exact H1-P and H1-S task identities, final `model_5999.pt`
evaluations, held-out seeds 211/223/227, 128 environments, six cycles, and the
retained no-motion boundary. Its output records both input JSON hashes and every
strict-improvement/non-regression comparison; only an `advance_to_multi_seed`
decision satisfies this diagnostic gate.

### H1-S source and runtime smoke

Commit `c9b045d0bc144de32c8d9acfc0422adc27618601` implements the distinct
H1-S task, and commit `6d5ddf99c5daf6296f8a1ad43e2f2ea6fb30a3f4` encodes the
deterministic causal comparison. The complete hop test pair passes 115/115 on
100.100; the focused causal-gate suite passes 3/3 on both the development Mac
and 100.100.

The predeclared seed-68, 64-environment, five-iteration runtime smoke completed
with service result `success`. The newly active
`stillness_at_zero_command` log was nonzero at 0.0109--0.0144, directly proving
that H1-S repaired the dead reward path. The height and motor curricula loaded
at 0.0200 m and -0.2500, respectively. Reward and optimizer scalars were finite,
NaN termination and spring bottoming were zero, and the final motor soft-limit
exposure was 0.3313. Frequent falls remain expected from an untrained
five-iteration policy and are not H1 evidence.

Retained smoke directory on 100.100:
`logs/rsl_rl/hop_k3900_h1s/2026-09-05_12-38-07_h1s-smoke-6d5ddf9-s68/`.
SHA-256 values:

- `model_4.pt`:
  `6f5f793523f90bb00e674f92d3fdec33448d8902ed1c2436f23a86937035d50c`;
- ONNX:
  `0d817ef686fe94c755e0392cd1920e2e7c266225c92c2739fb0a0efc1346b4c4`;
- TensorBoard event:
  `9474ab5b18e8310f3c2c9065b5bdf12dde06df28afa664fbfae7935948cd299a`.

This passes only the H1-S wiring gate and authorizes the predeclared paired
seed-67, 256-environment, 6,000-iteration diagnostic. It does not authorize a
multi-seed promotion, Locked control, H2, video, or physical motion.

### H1-S seed-67 diagnostic result

The paired seed-67, 256-environment H1-S run completed all 6,000 iterations
successfully and retained every 500-iteration checkpoint. The same held-out
seeds 211/223/227, 128 environments, six cycles, and unchanged H1 evaluator
were applied at the six predeclared diagnostic iterations:

| Iteration | Min cycle success | Min episode pass | Falls | Max drift p95 | Max bottoming | Max rated-speed exceed | Max torque p99 | Max near-stall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 22.79% | 0% | 384 | 0.990 m | 0.227% | 1.200% | 1.068 | 6.644% |
| 1,000 | 50.13% | 0% | 344 | 0.944 m | 0.065% | 0.422% | 0.780 | 0.362% |
| 2,000 | 12.50% | 0% | 0 | 0.0186 m | 0% | 0% | 0.279 | 0% |
| 3,000 | 11.85% | 0% | 0 | 0.0220 m | 0% | 0% | 0.294 | 0% |
| 4,000 | 12.89% | 0% | 0 | 0.0200 m | 0% | 0% | 0.277 | 0% |
| 5,999 | 11.98% | 0% | 0 | 0.0227 m | 0% | 0% | 0.298 | 0% |

The deterministic final-checkpoint comparison returned `stop`. H1-S improved
falls from 23 to zero and drift p95 from 0.594 m to 0.0227 m, and it did not
regress the spring or four motor-envelope comparisons. It failed the two
behavioral causal checks: episode pass did not improve above zero, and minimum
cycle success collapsed from 96.61% to 11.98%. The always-active stillness
objective therefore solved locomotor drift by suppressing the hop itself, not
by teaching a stable in-place hop.

Retained training checkpoint on 100.100:
`logs/rsl_rl/hop_k3900_h1s/2026-09-05_12-39-46_h1s-k3900-c9b045d-s67-6000x256/model_5999.pt`,
SHA-256
`57dd8ef9157131ee7445705dd6f747d56afe9a41b6292cbee3bfd914f6b1ba98`.

Retained evaluation directory:
`artifacts/evaluations/h1s-k3900-c9b045d-s67-diagnostic/`. Evaluation JSON
SHA-256 values by iteration are:

- 500: `1c0f499a02060810ffb37936dc856437ef6940b2f04b5b56c1bee513893d798f`;
- 1,000: `c21896d6028eff4115a64a0e2e626865a8997b5ec81d784ce11a6ca061fd93a7`;
- 2,000: `1f86381ee780d28e94a75c3314a21b654ea7b177f8ca88261e4eacaa6497ed02`;
- 3,000: `e360447f21e74469315cd1358924adde5b8d3131ed8c743b6879bfb309f05cde`;
- 4,000: `3b05192dc49c3e2b21d6891b4ded4279b45fe2fbdc1dfe8e42d590565f7f1017`;
- 5,999: `a48fa4a5bb99256b40ea23bebbac2b3ba721925f96b620a12377c2030bf8719b`.

The retained causal comparison JSON SHA-256 is
`2f1fe1f58986df53bfa11b59b3a52c7cc4ec6345b01f30c7047a896c5f0d1e56`.
The first evaluation wrapper attempt expanded its loop variable prematurely
and failed before opening a checkpoint or using the GPU; the corrected retry
used six explicit sequential commands and completed successfully.

The H1-S stop condition is met. Do not run a multi-seed promotion, Locked
control, H2, video, or physical motion. Any subsequent stability revision must
first predeclare a reward that distinguishes planar translation from the
vertical hop instead of applying full-strength stillness throughout the cycle.

## Predeclared bounded-cost H1-T revision

H1-T is the smallest causal follow-up to H1-S. It composes directly from H1-P,
not from rejected H1-S, and replaces the inherited dead zero-command stillness
term with one negative planar-translation cost:

`min((vx^2 + vy^2) / 0.20^2, 1)`.

The term is zero at rest, ignores vertical velocity, and is capped at one. It
therefore prices translation without paying a positive per-step dividend for
remaining still. Non-finite planar velocity maps to the maximum cost. Its
weight progresses gently at fresh-run iterations 0, 2,000, and 4,000:
`-0.05`, `-0.10`, and `-0.20`. Even at the final stage its per-step magnitude
cannot exceed 0.20, versus H1-S's weight-3.0 positive stillness opportunity.

The distinct task is `Mjlab-Hop-H1T-Flat-Sprung-K3900-MicroDuck`. H1-T changes
no actor or critic observation, action, mechanics, phase command, H1-P height
or motor curriculum, held-out evaluator, or physical-motion boundary. In
particular, base linear velocity remains privileged critic state rather than a
new real-robot actor dependency.

The first runtime action is only a seed-68, 64-environment, five-iteration
wiring smoke from scratch. It must complete with finite reward and optimizer
metrics, emit a nonzero `planar_velocity_cost` value and its initial `-0.05`
curriculum weight, load the H1-P height and motor curricula, retain `model_4.pt`,
and show no NaN termination. Five-iteration falls are diagnostic only. Smoke
success does not constitute H1 performance evidence and this source slice does
not launch the longer diagnostic.

The later paired causal diagnostic, if separately continued after reviewing the
smoke, is seed 67, 256 environments, 6,000 iterations from scratch with retained
checkpoints 500, 1,000, 2,000, 3,000, 4,000, and 5,999. Each checkpoint uses the
unchanged H1 held-out seeds 211/223/227, 128 environments, and six cycles. The
final iteration advances only if every predeclared check passes:

- minimum cycle success is at least 90%;
- minimum episode pass is above zero;
- total falls are below paired H1-P's 23;
- maximum drift p95 is below 0.30 m;
- spring bottoming, rated-speed exposure, torque-utilization p99, and near-stall
  exposure do not regress from paired H1-P.

The deterministic gate is:

```console
python -m mjlab_microduck.hop_revision_gate BASELINE_JSON CANDIDATE_JSON \
  --revision h1t --output OUTPUT_JSON
```

It retains exact input hashes and accepts only final `model_5999.pt` evaluations
from the exact H1-P and H1-T task identities under the frozen protocol. Any
failed check stops H1-T and parks the periodic-hop branch; it does not authorize
multi-seed promotion, Locked, H2, video, raw perception, actor-observation
expansion, or physical motion.

### H1-T source and runtime smoke

Commit `117c881ece760ce8b18a1195c7c8ff5839b98c8e` implements and predeclares
H1-T. The combined hop reward, configuration, registration, and causal-gate
suite passes 124/124 on both the development Mac and 100.100; remote validation
ran with CUDA hidden.

The seed-68, 64-environment, five-iteration smoke completed with service result
`success`. All five mean reward, value-loss, and surrogate-loss readings were
finite. The new `planar_velocity_cost` was nonzero at every iteration
(-0.0002 initially and -0.0011 finally), and its curriculum weight remained at
the expected initial `-0.05`. The H1-P height and motor curricula loaded at
0.0200 m and -0.2500. NaN termination and spring bottoming remained zero; final
motor soft-limit exposure was 0.3400. Falls are expected from this untrained
five-iteration policy and are not H1 performance evidence.

Retained smoke directory on 100.100:
`logs/rsl_rl/hop_k3900_h1t/2026-09-05_15-38-29_h1t-smoke-117c881-s68/`.
SHA-256 values:

- `model_4.pt`:
  `754d4e89fc5538104bfed3fa4df913e025357cd31fb26d338c8d770c1c6f23ac`;
- ONNX:
  `70ffc439e59d60342e1a7318114909967c7667cbe1da42cddb449ea1ce4bcbb8`;
- TensorBoard event:
  `42a22aae35208eb745320d1e301047a26a2298607c11a17c63fb56eea8b30768`.

The smoke service released the GPU (12 MiB, 0% utilization, 45 C), both
protected AI Mission services remained inactive, and the exact training
worktree remained clean. This passes only the H1-T wiring gate. The predeclared
seed-67 diagnostic remains closed pending explicit continuation; it does not
authorize multi-seed promotion, Locked, H2, video, raw perception, or physical
motion.

### H1-T seed-67 diagnostic result

The paired seed-67 H1-T diagnostic completed all 6,000 iterations successfully
with finite reward and optimizer metrics. The final curriculum values were the
predeclared 0.040 m hop rise, -2.00 motor cost, and -0.20 bounded planar cost.
NaN termination remained zero. The service released the GPU before evaluation,
and all six checkpoint evaluations then ran sequentially under the frozen H1
protocol.

| Iteration | Min cycle success | Min episode pass | Falls | Max drift p95 | Max bottoming | Max rated-speed exceed | Max torque p99 | Max near-stall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 16.80% | 0% | 384 | 0.918 m | 0.0109% | 1.637% | 1.068 | 20.294% |
| 1,000 | 30.99% | 0% | 384 | 0.956 m | 0.602% | 0.482% | 1.068 | 1.815% |
| 2,000 | 50.91% | 0% | 347 | 0.968 m | 0.557% | 0.212% | 0.870 | 0.677% |
| 3,000 | 84.38% | 0% | 137 | 0.827 m | 0.176% | 0.134% | 0.831 | 0.454% |
| 4,000 | 93.75% | 0% | 48 | 0.668 m | 0.105% | 0.125% | 0.780 | 0.369% |
| 5,999 | 99.35% | 0% | 4 | 0.476 m | 0.194% | 0.571% | 0.798 | 0.305% |

The deterministic causal gate returned `stop`. Relative to paired H1-P at
iteration 5,999, H1-T preserved the hop (99.35% cycle success), reduced falls
from 23 to 4, reduced drift p95 from 0.594 m to 0.476 m, reduced bottoming,
torque p99, and near-stall exposure, and remained within the absolute rise and
spring/torque ceilings. It nevertheless failed three predeclared causal checks:
episode pass remained zero, drift did not cross the 0.30 m causal threshold,
and rated-speed exposure regressed from 0.0208% to 0.571%. Under the full H1
acceptance protocol, the remaining nonzero falls and near-stall exposure also
fail their stricter gates.

Retained training directory on 100.100:
`logs/rsl_rl/hop_k3900_h1t/2026-09-05_15-47-06_h1t-k3900-117c881-s67-6000x256/`.
Final `model_5999.pt` SHA-256:
`454bd7db3da50896c2b00cebd72ea7eecf1fee3e9802a6ee03693e8fa858040a`.

Retained evaluation directory:
`artifacts/evaluations/h1t-k3900-117c881-s67-diagnostic/`. Evaluation JSON
SHA-256 values by iteration are:

- 500: `adf5a64612bbd76c3d5e12070d52fa2a4c85f39e0508a50b243bbf30c92b517b`;
- 1,000: `582ed7fc660249f7948f71c0dadcf02e7334ba3603fc44e95300ee1ddabedca9`;
- 2,000: `ac9a4086902c5f375e7f9ca266f4084f6ffb42bffadb05e22745168e44a97a3f`;
- 3,000: `0d2798143d80b1dbcfbbb31746e3356764fad4a6ca32e492fa8cabc3c025930d`;
- 4,000: `4564a4e98f532481a1e23a1238727c8b17244ddbce1ec3ae8b3997d0e37526f2`;
- 5,999: `867e1c9c1b4ba4d4a1385bd72a4ef8cdf9ff60120f70a3077e6e1b89c8ac91be`.

The causal comparison JSON SHA-256 is
`2e8731a7e1fb552f7badbd46b082cee75f10e3412341f9c17d7db55e5fcb5ef8`;
it retains baseline JSON hash
`99907a27954ba31b2d39fcdf427e0c947b4f1ddbb9b9dc82096ff92f411494c3`
and candidate JSON hash
`867e1c9c1b4ba4d4a1385bd72a4ef8cdf9ff60120f70a3077e6e1b89c8ac91be`.

The H1-T stop condition is met. Per the predeclared boundary, park periodic-hop
training here: do not run multi-seed promotion, Locked, H2, or video. The next
curriculum work should return to the already functional motor-aware running and
compact-observation obstacle negotiation path rather than spend more GPU time
on another hop reward revision.
