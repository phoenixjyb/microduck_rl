# H0/H1 matched-mechanics hop campaign

Date: 2026-09-04

Decision: **runtime smoke and matched-arm diagnostic pilot passed; K3900 is the
H1 training candidate; no hop policy accepted**

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

K2500 remains a documented mechanical sensitivity arm, not a co-candidate;
revisit it only if K3900 cannot pass H1. No MP4 is recorded until the numerical
matrix accepts a representative checkpoint.
