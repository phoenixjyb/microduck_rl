# C1 stock Happy Hop: preparation, not a replay result

Date: 2026-10-01 Asia/Shanghai. Starting source:
`ee23009bd1cb7e30b42cb0b39d583861e04c21d5`, branch
`feat/athletics-obstacle-curriculum`. The approved next step from the
[community intake](2026-10-01-community-skills-research.md) is an independent
stock/backlash walk → settle → one-shot hop → land/settle → walk diagnostic.
Historical sprung-K3900 H1, obstacle, stance and football gates are unchanged.

**Current decision: preparation tested; matched closed-loop replay blocked by
missing entry-policy and task/runtime provenance.** No inference, simulation
stepping, GPU job, recording or physical motion was performed in this chunk.
Compiling a stock XML below is a separate structural check, not a BAM rollout.

## Source audit and concrete launch blockers

The pinned [Happy Hop release](https://huggingface.co/joanfox/microduck-happy-hop/tree/c0447da668255436eea070e862ad8eb89bad2ba1)
contains its ONNX, README, manifest and media, but not the entry walker or the
training task. Its policy is retained locally at
`artifacts/community/intake-20261001/happy-hop/policy.onnx`, SHA256
`abd6db1bca2f2a583508cfd9a6fd144a7cce65b302827b9b9a0c5227fc2acae9`.

The [pinned card](https://huggingface.co/joanfox/microduck-happy-hop/blob/c0447da668255436eea070e862ad8eb89bad2ba1/README.md)
specifies 61/14 float32 inputs/actions, 50 Hz, one 20 ms action delay, scale 1.0
around HOME, a baked observation normalizer, no low-pass, all 13 command values
zero, stable standing entry and a 3 s one-shot. The manifest names
`walking_backlash_model_5000.onnx` after 1 s of exact-zero velocity, but provides
no walker hash, location or producing-run provenance. Current official
`velstand` is not thereby the same entry policy.

The declared upstream base `d424a0c899f6b33cbd3daeb279913134349c0b63` is available
in local Git. Searching its complete task tree found no `HappyHop` task.
Its [task registry](https://github.com/pollen-robotics/microduck_rl/blob/d424a0c899f6b33cbd3daeb279913134349c0b63/src/mjlab_microduck/tasks/__init__.py)
does not contain the advertised
`Mjlab-HappyHop-WalkTransition-Clearance-Flat-Backlash-MicroDuck`.
Its [replay script](https://github.com/pollen-robotics/microduck_rl/blob/d424a0c899f6b33cbd3daeb279913134349c0b63/scripts/infer_policy.py)
has no `--happy-hop` option and uses simplified position actuators and servo-only
observations. It cannot reproduce the card's BAM/output-side backlash rehearsal.
Do not patch a flag onto that script and call the result matched.

The base backlash recipe supplies useful family context, but its generic BAM
delay is 3–6 control steps and the 1.75 A cap is commented out. The hop task's
overrides, sensor/velocity lags, normalization input scales, collision setup,
HOME and handoff initialization need the missing task overlay/run config.
The card's physical-test claim also still conflicts with its manifest's
hardware-untested status; neither establishes our acceptance.

A bounded additional check of the inspected release history, author's public
[GitHub account](https://github.com/joanfox), upstream source and exact-name
searches did not recover a safely bindable overlay or walker. This is not proof
that no private/unindexed copy exists. No third-party source archive or pickle
checkpoint was executed, and no message was sent to the author.

The smallest source handoff needed is:

1. Exact entry walking ONNX, SHA256, producing run/checkpoint and export source.
2. Happy Hop task patch/source plus environment/agent config for
   `2026-08-31_20-39-21_happy_hop_clearance_35mm_stage1`, `model_1255.pt`;
   especially BAM/current/delay/contact/observation overrides.
3. Transition implementation specifying raw previous-action and action-queue
   initialization and carry/reset across walker → hop → walker.

Until resolved, an official-walker substitution is a **different predeclared
experiment**, not an author-matched replay. It needs a separate decision before
running. There is no new long training launch authorized by these tests.

## What was independently verified

`git diff` against the declared base found no changes in the tracked stock
backlash XML/scenes/meshes used here. The only robot-directory differences were
an unrelated obstacle prop and removal of a ball scene. This establishes
source-byte equality to the base, not the missing hop task's compiled plant.

| Local source file | SHA256 |
| --- | --- |
| `scene_backlash.xml` | `9ad82184844332f3999a3bb4e634700e8d8010f9974744c16f9ac174243cf488` |
| `robot_allcollisions_backlash.xml` | `248c712e13258104ccab3f96c5cc47c7faa8f718f56f2c7de03efda653c8fb6b` |
| `robot_walk_backlash.xml` | `43686e85bf0fa0c77c89673d546c483d4a22d06a9c81fff3a673bfe9aae36d0d` |

Native CPU MuJoCo compiled `scene_backlash.xml`: `nq=35`, `nv=34`, `nu=14`.
All 14 actuators have their corresponding `passive_<name>_backlash` joint.
The compiled servo order is left hip yaw/roll/pitch, left knee/ankle,
neck pitch, head pitch/yaw/roll, right hip yaw/roll/pitch, right knee/ankle.
No `mj_step`, policy inference or BAM torque update was called. In particular,
the raw scene still contains position actuators; compilation is **not** evidence
of a correct BAM controller or task collision overrides.

Current local versions observed: MuJoCo 3.10.0, better-actuator-models 1.0.1,
ONNX 1.22.0, ONNX Runtime 1.24.4. No package, lockfile or remote runtime was
changed. No 100.98/100.100 service or GPU process was touched.

## Fixed proposed C1 measurement protocol

Implemented in [community_hop_rehearsal.py](../../src/mjlab_microduck/community_hop_rehearsal.py),
protocol `community-hop-c1-first-attempt-v1`, trace schema
`community-hop-c1-trace-v1`. This is a **trace contract and scorer**, not a
simulation runner. Its observation/history helpers are proposed family-contract
building blocks, not confirmed Happy Hop training/export parity.

| Time, seconds | Policy intent | Full 13-value command |
| --- | --- | --- |
| 0–3 | walk | forward 0.20 m/s; every other slot zero |
| 3–4 | walking policy settles | all zero |
| 4–7 | one-shot hop | all zero |
| 7–9 | walking policy lands/settles | all zero |
| 9–12 | resume original course | forward 0.20 m/s; every other slot zero |

This deliberately permits stopping through the hop and handoffs while requiring
speed reacquisition, consistent with the obstacle manoeuvre principle. It does
not claim a hop directly out of an active walking stride.

The proposed nominal probe budget is one attempt, 12 simulated seconds,
2,401 states at 5 ms including the terminal state; 50 Hz inference/command
updates, no fresh inference at terminal time. A future CPU runner must have a
60 s wall-clock cap, no automatic reset/retry, no speed/height escalation, and
save an incomplete/failure result if interrupted. **The probe remains blocked**;
the matched source and settings must be bound before this budget is used. An
untouched confirmation/retention matrix still precedes any promotion.

The shared one-step queue starts with zero raw action offsets at HOME, and
carries history across switches without resetting/filtering/clipping. The
observation packer concatenates angular velocity, unit projected gravity,
14 output-side HOME-relative positions, 14 output-side velocities, 14 previous
raw actions, and 13 commands; it does not run a second normalizer. Firmware
position feedback must read servo+backlash, while rotor velocity remains
motor-side for BAM back-EMF/friction. Sensor lag, ordered IDs, HOME, normalizer
parity and these proposed switch-state semantics remain **unbound** until the
author's overlay is recovered. The helper is not a torque controller.

Trace fields include exact commands and raw/applied actions; world-course
velocity, base XY and roll/pitch/yaw; angular velocity; two foot clearances,
contacts and normal-force magnitudes; 14 motor currents/torques/rotor speeds
and soft-limit flags; body-contact and reset flags. Positions/velocities follow
a fixed initial world +X course. Foot clearance must be the minimum physical
sole/collision-surface height above the floor, not a foot site/origin or base
rise. Contact must come from actual floor contacts, not a kinematic height flag.
`body_contact` means any non-foot robot/floor contact. Motor current must come
from the modeled voltage/back-EMF equation, not a clipped-torque guess. Proposed
soft exposure uses the centered 0.9 hard joint-range factor; that proxy still
needs source binding, not a mechanical-safety claim.

Numerical diagnostic indicators are fixed before any model execution:

- Exact contiguous 5 ms samples and 20 ms held actions; one-step delay persists
  at every handoff. Any command/history/cadence violation rejects the trace.
- No reset, nonfinite state, non-foot body contact or >=60 degree roll/pitch.
  The first such event ends scoring; a later retry cannot improve the result.
- Before switching to hop: the full preceding 1 s command is exactly zero;
  the final 0.5 s has both foot contacts, planar and vertical speed <=0.05 m/s,
  roll/pitch <=10 degrees and angular-speed norm <=0.5 rad/s. Unstable entry
  ends scoring at the switch; it does not receive a credited hop.
- Exactly one qualified airborne episode: both feet without contact and
  simultaneous >=30 mm physical clearance for at least three 5 ms states
  (10 ms span), reached within the hop window; then actual touchdown.
- The full first-touchdown+100 ms force window must exist. Report its maximum
  summed foot force **and** the entire hop/settle force peak, including any
  later bounce; no calibrated impact ceiling is invented.
- Final landing-settle second meets the same stability checks; maximum XY
  drift from the hop switch during hop/settle <=0.10 m.
- Final seconds of initial and resumed walking each have forward MAE <=0.08
  m/s, lateral speed peak <=0.05 m/s and heading error peak <=15 degrees.
- Modeled current peak <=1.75 A plus 1e-6 numerical tolerance, no declared
  soft-limit exposure, finite derived peaks. Torque, rotor speed, absolute
  mechanical power and force peaks are reported, not renamed rated limits.

These thresholds are independent **research indicators**, not old H1 promotion
criteria or calibrated hardware limits. The scorer authenticates neither its
declared trace nor its plant binding. Even a synthetic/declared trace clearing
every indicator returns only `declared-trace-indicators-pass`, with behavioral,
training, transition and physical-motion admission flags false. Thermal-model
verification and calibrated impact limits are explicitly false, too.

## Validation retained in source

[Synthetic tests](../../tests/test_community_hop_rehearsal.py) cover observation
ordering/output-side sums, no extra normalization, delay carry, substep hold,
unstable-entry abort, first-fatal/no-retry handling, malformed/truncated traces,
one-foot/separate-height/too-short false hops, repeated hops, delayed force
peaks, floating-point window boundaries, derived overflow and each fixed gate.
They are not robot or learned-skill results.

- 68 C1 tests; 254 passed in the combined C1, community inspector and existing
  static skill compatibility/binding/retention selection. CUDA was hidden.
- Separately, 12 existing hop evaluation/revision/sweep tests passed, retaining
  the unchanged historical H1 gate behavior. Full repository tests were not run.
- `git diff --check` passed; the new Markdown rendered with two tables and all
  three local links resolved. Updated curriculum/research links also resolved.
- Importing the new module loaded neither Torch, ONNX Runtime, MuJoCo nor BAM.
- A bounded Luna read-only audit identified the missing source contract and
  reviewed the scorer; integration made the impact-window completeness gate
  explicit and added entry-abort, repeated-hop and substep-force safeguards.
- No closed-loop replay, inference parity, GPU training, MP4 or physical test
  was run. These remain separate delivery gates.

Next gate: obtain/bind the immutable source handoff above, or explicitly
predeclare a substituted-entry diagnostic under a new identity. Only then
implement and test the native-CPU BAM runner against the actual task/transition
contract and execute the single bounded first attempt.

Subsequent user decision: a separately labelled official-walker substitution
is approved for predeclaration. See [C1-S](2026-10-01-community-hop-c1s-predeclaration.md)
for its explicit surrogate and baseline-first budget. This does not resolve
the original author-matched C1 blockers or claim a replay has executed.
