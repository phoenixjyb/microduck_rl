# HC4-R2 student-state teacher correction

Date: 2026-09-04

Implementation commit: `2fb8dfbec42d26dc57341b6573b2a6574656a258`

Decision: **protocol and collector accepted; training and closed-loop selection
remain pending retained evidence**

## Purpose and unchanged boundaries

HC4-R seed 42 passed offline imitation but regressed from seven to ten
collisions against HC4-LH in its three-seed 0.90 m matrix. Its low validation
error therefore did not predict closed-loop safety. HC4-R2 changes one training
input: it adds deterministic-teacher labels at states actually visited by the
HC4-R student. Architecture, optimizer, training seed, frozen locomotion actor,
exact obstacle geometry, command bounds, and evaluation matrix remain fixed.

The student still emits only forward-speed and yaw-rate commands. The teacher
labels a cloned supervisor state before the student command is applied, so the
counterfactual label cannot mutate execution. No camera pixels, joint targets,
physical robot motion, or authority expansion are introduced.

## Correction dataset contract

For each supervisor update, the collector retains:

- the 17D compact observation reached under HC4-R execution;
- the deterministic teacher's bounded forward-speed/yaw command;
- HC4-R's executed bounded command;
- a collision-safe episode key and terminal outcome code;
- frozen locomotion and student-checkpoint hashes.

Only episodes that terminate as clean pass, collision, or attempt timeout are
retained. Partial episodes and episodes with a fall or NaN termination are
excluded. Teacher-success collection and student-state correction collection
are mutually exclusive modes.

The retained runtime smoke used four environments at 0.30 x 0.90 x 0.00 m and
seed 131. It produced 337 finite samples from four clean episodes. Mean
teacher/student absolute disagreement was 0.00414 m/s in speed and 0.04292
rad/s in yaw; p95 disagreement was 0.01215 m/s and 0.12068 rad/s. The smoke
dataset SHA-256 is
`46deda7dd1827bfedc6a30edf138f6864a0e2c383b1241d2329ad3c50402e565`;
the rollout-report SHA-256 is
`9d2b68c48660caf84895e1fd3e42bb09786c41cd464513c921a3a05d3372fd5f`.

## Predeclared collection and training

Correction-data seeds are 131, 137, and 139. Each covers 0.30/0.40 m/s,
obstacle range 0.90 m, and lateral positions -0.08/0.00/+0.08 m with 64
environments for 700 steps. The executed student is the rejected HC4-R
checkpoint SHA-256
`641fb7ae5cd0a1a780b2ce8ca759e3d2ce668651b7f23834bdc79685bb88bf3f`.

If all shards are finite and carry exact provenance, one HC4-R2 candidate will
be trained with seed 42 for 200 epochs on the original HC1 near-range teacher
dataset plus all three correction shards. Reusing the architecture,
hyperparameters, and seed isolates student-state corrections as the changed
axis. The trainer refuses the HC4-R2 stage unless both teacher and correction
dataset types are present.

## Predeclared closed-loop gate

Held-out seeds are 149, 151, and 157. Seed 149 is the pre-screen; continuation
starts only if there is no fall, NaN termination, non-finite step, rated motor
speed exceedance, or obvious collision regression. HC4-R2 will be paired with
both HC4-LH and HC4-R on the same six cells.

Promotion requires all of the following across the three-seed matrix:

- zero falls, NaN terminations, non-finite steps, and rated-speed exceedance;
- fewer collisions than HC4-R and no more collisions than HC4-LH;
- no lateral/speed cell with more collisions than HC4-LH;
- no timeout regression and clean-pass rate at least as high as HC4-LH;
- maximum per-cell motor torque-utilization p99 no greater than 0.60.

Offline imitation success does not override these closed-loop gates. No MP4 is
recorded and no range-gated composition is built unless the numerical candidate
passes. HC4-LH remains selected throughout, and 0.90 m remains outside its
accepted envelope.
