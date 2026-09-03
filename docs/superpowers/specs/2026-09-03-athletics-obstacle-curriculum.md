# MicroDuck capability curriculum: locomotion, hopping, jumping, and obstacles

Status: design accepted for incremental implementation on
`feat/athletics-obstacle-curriculum`.

This document turns the current experiments into a capability curriculum with
explicit dependencies and promotion gates.  It does not authorize physical
motion.  Every result below is simulation evidence unless a later section says
otherwise.

## Outcome

Train three reviewable low-level skills before attempting a generalist policy:

1. motor-aware locomotion,
2. in-place hop and commanded jump,
3. obstacle bypass.

A small supervisor may select among accepted skills only after each skill has
passed its own held-out evaluation.  The first supervisor consumes structured
geometry and commands; it does not consume camera images and it does not gain
motor or motion authority.

This decomposition keeps failures attributable.  A monolithic policy that
learns running, jumping, route choice, and perception at once would make a
collision impossible to assign to sensing, planning, or control.

## Current evidence and starting point

| Capability | Evidence | Decision |
| --- | --- | --- |
| Stage 1 running | `model_4999.pt`; five-speed, three-seed evaluation | Retain as the faster experimental control. |
| Stage 2 motor-aware running | `model_7998.pt`; 3,000 additional iterations; five speeds by three seeds | Accepted only for simulated commands from 0.5 to 0.8 m/s. |
| Replay | H.264 recordings at 0.5 and 0.8 m/s | Accepted as visual evidence, not as a metric substitute. |
| Hop task | Rigid, k2500, and k3900 variants plus hop metrics and tests | Implementation-ready, but no campaign checkpoint is accepted here yet. |
| Obstacle observation | Seven-channel geometry contract and scene adapter | Accepted interface; perception remains separate. |
| Single-box avoidance | Several bounded pilots and held-out evaluations | Learning is real, but no checkpoint satisfies the promotion gate below. |

The obstacle experiments exposed two important failure modes:

- training longer is not monotonically better; final checkpoints frequently
  become slower or take wider detours;
- one training seed can look substantially better than another.

Consequently, the final checkpoint is never selected merely because it is
final.  Intermediate checkpoints and training-seed robustness are part of the
curriculum, not post-hoc diagnostics.

## Dependency graph

```text
motor-aware stand/walk/run
          |
          +--> periodic hop --> commanded single jump --> running jump --+
          |                                                            |
          +--> single-obstacle bypass --> varied static obstacles ------+--> skill supervisor
                                                                            |
                                                                            +--> obstacle sequence
```

The hop and bypass branches may be trained independently after the motor-aware
baseline.  A policy is not warm-started from an unaccepted parent.

## Invariants shared by every stage

### Policy and perception boundary

- Locomotion RL receives robot state, task commands, and the documented compact
  obstacle estimate only.
- The obstacle actor fields remain `range`, `bearing_sin`, `bearing_cos`,
  `width`, `height`, `closing_rate`, and `valid`.
- Raw RGB, depth images, detector embeddings, simulator entity IDs, and future
  state are forbidden actor inputs.
- The asymmetric critic may consume exact simulation geometry.  The actor may
  not.
- Noise, latency, dropout, and field of view are introduced only through an
  explicit sensor model.  Their final values must come from measured perception
  performance or a labeled stress test.

### Motor and numerical safety

Every promoted checkpoint must satisfy all of these hard gates on every held-out
seed:

- zero NaN terminations and zero non-finite steps;
- zero rated-speed exceedance in the intended command envelope;
- p99 torque utilization no greater than 0.90 at a 0.5 m/s command and no
  greater than 0.95 at a 0.8 m/s command;
- near-rated-stall exposure no greater than 0.25% at 0.5 m/s and 1.0% at
  0.8 m/s;
- no unexplained action-magnitude or action-rate regression against the parent
  checkpoint;
- no physical motion based only on these simulation metrics.

Mechanical power and the thermal-load proxy are comparison metrics, not
physical temperature predictions.  A real robot still needs voltage, current,
joint-speed, and motor-temperature telemetry with an independent stop path.

### Experiment control

- Change one curriculum axis per experiment.
- Train at least three independent training seeds before promotion.
- Use distinct environment seeds for development and held-out acceptance.
- Retain a checkpoint at least every 16 learning iterations during bounded
  pilots.
- Evaluate intermediate checkpoints.  Stop extending a run when two successive
  retained checkpoints regress on the primary gate without improving a declared
  secondary objective.
- Preserve rejected checkpoints and their evidence; never relabel them as
  successful runs.
- Run one GPU job at a time and verify an idle GPU plus protected-service state
  before starting it.

## Stage L: motor-aware locomotion

### L0 — static support and command contract

Purpose: prove standing, reset, command, observation, and action contracts.

Promotion:

- deterministic CPU smoke passes;
- zero falls and non-finite values in the fixed standing case;
- actor and critic dimensions match the checkpoint manifest;
- exact checkpoint and source hashes are retained.

### L1 — walking and low-speed tracking

Commands advance from standing to 0.25 and then 0.5 m/s.  Heading and lateral
commands remain disabled until straight tracking passes.

Promotion:

- observed speed at least 80% of command at 0.5 m/s;
- no non-timeout episode endings across three held-out seeds;
- motor hard gates pass.

### L2 — motor-aware running

Commands expand to 0.8 m/s.  Stage 2 `model_7998.pt` already passes this stage
in simulation and is the parent for both hop and obstacle branches.

Commands above 0.8 m/s are extrapolation probes, not curriculum promotion
evidence.  A faster policy may remain on the Pareto frontier without replacing
the motor-aware baseline.

## Stage H: hop and jump

### H0 — matched mechanical controls

Run rigid, locked-at-matched-mass, k2500, and k3900 configurations with the same
policy architecture, randomization, reward budget, and action-rate penalty.
Spring damping remains a declared uncertainty until measured on hardware.

Promotion to learning requires non-zero spring compression in sprung arms and
near-zero bottoming in the stiffness arm being evaluated.

### H1 — periodic in-place hop

Start with one externally commanded period and the existing phase signal.  The
policy learns load, launch, flight, and landing while horizontal commands remain
zero.  Do not add obstacle geometry yet.

Promotion:

- both feet become airborne in at least 90% of commanded cycles;
- landing survival is at least 98%;
- zero NaNs and all motor hard gates pass;
- sprung peak rise exceeds the matched-mass locked control by at least 10% on
  the median training seed without materially increasing falls;
- spring bottoming remains below 1% of measured samples.

If no sprung arm beats the locked control, retain the negative result and use
the rigid/locked policy for the next stage.  Do not tune damping without a
measurement merely to obtain a positive result.

### H2 — commanded single jump

Convert the periodic skill into one-shot `prepare`, `launch`, `flight`, `land`,
and `settle` phases.  Begin from standing with a fixed target rise, then vary the
target within the H1 demonstrated envelope.

Promotion:

- at least 90% successful one-shot jumps across held-out seeds;
- apex error within 20% of the command;
- at least 98% stable land-and-settle outcomes;
- no forward displacement beyond a 0.10 m in-place corridor;
- motor hard gates pass.

### H3 — running jump

Warm-start from accepted L2 and H2 skills through a declared migration or
distillation step.  First fix approach speed at 0.5 m/s and landing height equal
to takeoff height.  Vary approach speed only after landing is stable.

Promotion:

- at least 85% clean clearance and stable landing;
- post-landing route speed recovers to at least 70% of command within 1 s;
- no obstacle contact, NaN, or motor hard-gate violation.

## Stage O: obstacle bypass

An obstacle attempt resolves as exactly one clean pass or collision.  Falls,
timeouts, and NaNs are reported separately and are never counted as clean
passes.  The primary success metric is:

`clean_pass_rate = clean_pass_events / (clean_pass_events + collision_events)`

### O0 — fixed-command control

Use the migrated Stage 2 actor with zero-initialized obstacle columns.  This is
the causal control: it proves what straight running does without learned use of
obstacle input.

### O1 — one centered box, exact geometry

Command 0.25–0.5 m/s, use one stationary box, exact actor geometry, no dropout,
and no sensor noise.  Permit lateral motion but retain a fixed route heading.
Train bounded pilots, checkpoint every 16 iterations, and select the earliest
checkpoint that passes.

Promotion across three training seeds and separate held-out environment seeds:

- clean-pass rate at least 70% on every training seed and at least 75% in the
  pooled held-out evaluation;
- zero falls, NaNs, and non-finite steps;
- pre-obstacle route speed at least 0.25 m/s under a 0.5 m/s command;
- mean clean-pass lateral excursion no greater than 0.45 m;
- mean passage time no greater than 4.5 s;
- motor hard gates pass.

No current obstacle checkpoint passes all of O1.  The collision-focused policy
misses speed and corridor gates, while faster intermediate checkpoints miss the
clean-pass-rate gate.

Protocol audit: the pilots retained before this design used lateral placement
randomization up to ±0.35 m and up to one step of actor-observation latency.
They are therefore reclassified as pre-O1 diagnostic evidence, not O1
acceptance attempts. `O1-centered-exact-v1` fixes the box 1.15 m ahead on the
centerline with zero observation lag and zero sensor degradation. O2 and O3
introduce those axes later through distinct task variants.

### O2 — lateral placement and geometry

After O1, widen box lateral placement, then vary width and height independently.
Only one of placement, width, or height changes at a time.  The actor receives
the same seven fields.

Promotion keeps the O1 hard gates and requires no more than a five-percentage-
point pass-rate drop in any geometry bucket.

### O3 — measured sensor degradation

Introduce range/bearing noise, one-step latency, dropout, and field-of-view
limits in separate runs before combining them.  Exact-geometry O2 remains the
control.

Promotion requires at least 90% retention of the exact-geometry pass rate and
no new motor or numerical failure.

### O4 — obstacle sequences

Begin with two widely separated static boxes so each maneuver resolves before
the next begins.  Reduce separation only after the two-box case passes.  Dynamic
obstacles and raw perception remain out of scope.

## Stage S: skill selection

The first supervisor chooses among `run`, `bypass-left`, `bypass-right`, and
`jump`.  It consumes route command plus the same structured obstacle geometry.
Its output is a discrete skill and bounded skill parameters, not joint actions.

Rules before learning a supervisor:

- each selected low-level skill already passes its own gate;
- a deterministic rule-based selector is retained as the baseline;
- the supervisor cannot select jump for an obstacle outside H3's demonstrated
  height/width envelope;
- loss of a valid obstacle estimate falls back to a stop/slow command supplied
  by the execution layer, not a guessed maneuver from the policy.

Only after the selector passes may policy distillation into a single network be
considered.  Distillation must reproduce each specialist's acceptance suite and
does not retire the specialists.

## Checkpoint selection protocol

For every training seed:

1. reject any checkpoint that fails numerical or motor hard gates;
2. reject any checkpoint below the stage's minimum success rate;
3. among survivors, retain the Pareto frontier for success rate, route speed,
   passage time, and lateral excursion;
4. prefer the earliest checkpoint when two candidates are equivalent within
   evaluation noise;
5. rerun the shortlisted checkpoint on held-out environment seeds;
6. promote only if all training seeds have at least one checkpoint in the same
   bounded iteration window that passes.

This protocol selects a robust learning window rather than cherry-picking one
fortunate seed.  The current evidence suggests an O1 candidate window roughly
34–50 iterations after the obstacle warm start, but that window is diagnostic
until the O1 gate passes.

## Evidence artifacts

Every promoted stage retains:

- source commit, branch, task ID, full invocation, device, and seed;
- input/output checkpoint hashes and migration metadata;
- training logs and intermediate-checkpoint hashes;
- per-seed JSON evaluation plus a pooled summary;
- motor-envelope metrics in the same acceptance record;
- MP4s for representative success and failure cases;
- an explicit `accepted`, `rejected`, or `diagnostic-only` decision.

The repository, checkpoint, rendered replay, and eventual physical result are
separate evidence tiers.  Success at one tier does not imply success at the
next.

## Immediate implementation order

1. [x] Add resolved-attempt and clean-pass-rate fields to the retained obstacle
   evaluator.
2. [x] Produce a deterministic checkpoint-sweep summary that compares training
   seeds by iteration without choosing a winner from the final checkpoint.
3. [x] Combine obstacle outcomes and motor-envelope metrics into one acceptance
   record.
4. [ ] Run a three-training-seed O1 campaign within a fixed 64-iteration budget.
5. [ ] If O1 passes, proceed to O2.  Otherwise change exactly one reward or
   command axis and repeat the bounded campaign.
6. [ ] Start H0/H1 as a separate campaign; do not mix hop rewards into O1.

## Physical gate

No curriculum stage authorizes physical movement.  Before the first robot test,
require a no-motion inference replay, joint-order and sign verification,
command timeout, independent emergency stop, conservative torque/current
limits, temperature monitoring, tethered low-energy stance test, and a human
operator with clear space.  Simulation thermal proxies are not a substitute for
measured temperature.
