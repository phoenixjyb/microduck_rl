# HC4-R2 student-state teacher correction

Date: 2026-09-04

Implementation commit: `2fb8dfbec42d26dc57341b6573b2a6574656a258`

Decision: **accept HC4-R2 as a specialist for the tested exact-geometry
0.30/0.40 x 0.90 m simulation envelope; do not replace HC4-LH wholesale**

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

## Retained correction shards

The three full collection runs completed without falls, NaNs, non-finite
steps, or timeouts. They retained 105,886 finite samples from 1,415 resolved
episodes, including all eight collision episodes:

| Seed | Samples | Episodes | Clean | Collision | Dataset SHA-256 |
|---:|---:|---:|---:|---:|---|
| 131 | 35,673 | 481 | 476 | 5 | `e145e34c9ac61cc3e2778151139847cbc98ede0be2dea4850066e584652bc08a` |
| 137 | 34,287 | 449 | 447 | 2 | `3f20078db7cfc0e460ab0564de608735d5c919c50df5efe4d7dc6eb53fcfb7ca` |
| 139 | 35,926 | 485 | 484 | 1 | `e871dda49fbe2155f0e9fbc3699abd609a0c711a757dde824c889107527fbce5` |

The corresponding rollout-report hashes are
`26af74e57ba14597d97a184a8b018b52e0558b4082f435130419cf91f6c79b4a`,
`bef7f33af820f94fae27299c076473e052aa5dfb759322db31d75225ee7c4f85`,
and `ecb7ca4d74d18218f81f035a632781a7262c7f431da4b3c7aabbaa21669bb1a5`.

## Offline fit

The fixed seed-42 candidate trained for 200 epochs on 213,551 total samples
from 2,881 episodes. Epoch 199 minimized validation MSE. Validation speed MAE
was 0.001658 m/s and yaw MAE was 0.017016 rad/s, inside the 0.025/0.050 gates.

The retained checkpoint is
`../artifacts/hc4r2-bc-796634d-s42/supervisor.pt`, SHA-256
`c4ba5925de7144373c94145b57b5e7a7ae3e1fc89bc7c2c3203f8724bdebf1b7`.
Its manifest SHA-256 is
`8b24cc0df80be770c19255d2b6b6ccc499cdd183cc86ad05a8f3736992c9f3c0`.

## Three-seed closed-loop decision

The actual seed-149 pre-screen and seed-151/157 continuation produced:

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC4-LH | 1,348 | 8 | 0 | 1,356 | 99.410% | 7.559 s | 0.5465 |
| HC4-R | 1,422 | 4 | 0 | 1,426 | 99.719% | 7.387 s | 0.5677 |
| **HC4-R2** | **1,402** | **2** | **0** | **1,404** | **99.858%** | **7.433 s** | **0.5650** |

HC4-R2's two collisions were centered: one at 0.30 m/s and one at 0.40 m/s.
It was collision-free in all four shifted cells. HC4-R recorded two centered
collisions at each speed. HC4-LH recorded seven centered collisions and one at
0.30 m/s by -0.08 m. All controllers retained zero falls, NaN terminations,
non-finite steps, timeouts, and rated motor-speed exceedances.

The report hashes are:

- HC4-R2 seed 149: `e4dbbdaa6418146618994f6721c37a2b8521c55f6c55725e65548fb69e01f495`;
- HC4-R2 seeds 151/157: `6f5327298bf46c8322c478ccff2ce3163cccf33e156e0d9316eb5ea852235a74`;
- HC4-R seed 149: `1ecc8b5a8cd42e02d7b4b9e05e51887b2d8fd2fbe0f0e2f5b48810d5e36cdb90`;
- HC4-R seeds 151/157: `bff8855612723bbff0102696edadf76d969f702b3462e2dc2e5db39a7b68e278`;
- HC4-LH seed 149: `65e6868cd37a29de46f4bdd2b3359cb94c241093ffdf809483921a7d7d7f38a0`;
- HC4-LH seeds 151/157: `d218fdb3841d74aae145a379554e291e445271548b6a6dd67756897e73f74389`.

HC4-R2 passes every predeclared numerical gate and is accepted only as a
near-range specialist for exact structured geometry at obstacle range 0.90 m,
speeds 0.30/0.40 m/s, lateral positions -0.08/0.00/+0.08 m, and the retained
box geometry. HC4-LH remains selected for its existing farther-range envelope.
A unified controller requires a separately tested range/speed gate with
invalid-geometry fallback; this acceptance alone does not authorize that
composition, camera perception, or physical motion.
