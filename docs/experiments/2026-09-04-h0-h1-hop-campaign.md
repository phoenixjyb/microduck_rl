# H0/H1 matched-mechanics hop campaign

Date: 2026-09-04

Decision: **implementation-ready; no hop policy accepted or trained in this
campaign yet**

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
