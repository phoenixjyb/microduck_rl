# HC4-U4 near-envelope student-state correction

Date: 2026-09-06. Status: correction data admitted; seed-42 fit predeclared, not a policy.
Parent HC4-U3 is rejected at held-out physics seed 311. See the retained U3
experiment for exact failed cells, hashes, and the limits of causal diagnosis.

## Hypothesis and frozen boundaries

HC4-U3 improved far-range timeout counts but added one near-left collision at
0.30 m/s, 0.90 m, -0.08 m. Existing correction labels were reached by older
networks. Test student-state coverage across the **whole six-cell near envelope**,
not only the observed failed cell. This hypothesis is unproven; representative
trace zero was not the colliding environment. No evaluation-seed trajectories
or labels enter training.

The sole proposed training axis is appending fresh near-envelope correction
samples reached under frozen U3. Retain its three independent 64x64 phase experts,
17D input, two bounded speed/yaw commands, seed 42, 200 epochs, batch 1024, AdamW
0.0003 / weight decay 0.00001, episode-disjoint 20% validation, and minimum
validation-MSE selection. Keep all thirteen parent corpus shards in their exact
order. No oversampling, relabeling, initialization transfer, reward adjustment,
new sensor, raw perception, motor-model change, or physical motion.

## Collection predeclaration

- Frozen student: `artifacts/checkpoints/hc4u3-ab1d831-s42/supervisor.pt`, SHA-256
  `6c6546448340530e4cdb7e4381247f644211029c7750146c0da6dbcf7c60aa2d`.
- Frozen locomotion actor SHA-256:
  `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Training-collection seeds **317, 331, 337**, sequentially. Each covers speeds
  0.30/0.40 m/s, forward 0.90 m, laterals -0.08/0/+0.08 m, 64 environments and
  700-step ceiling per cell. Exact compact geometry and 12-second timeout.
- First terminal attempt only: 384 distinct attempts per shard. Teacher labels
  use the unchanged deterministic teacher with cloned state while U3 executes.
  Resolve simultaneous outcomes with the declared failure-priority protocol.
  Keep collision and timeout examples; exclude partial, fall, NaN and unknown
  terminal episodes. The collection gate requires all 384 valid episodes, so
  any excluded episode closes that shard rather than changing its denominator.
- Before full collection, a four-environment smoke at 0.30/0.90/-0.08 with seed
  317 verifies the newly combined first-attempt/correction mode and outcome
  labeling. Retain it separately and **never include smoke samples in training**.
- The full seed-317 shard must pass before seed 331; seed 331 must pass before
  seed 337. Gate: at least 20,000 finite matched 17D samples, all 384 episode
  keys present with consistent valid outcome codes, exact actor/student/seed/
  window/teacher identities, no fall/NaN/nonfinite/hard/other/rated-speed failures,
  torque p99 <=0.60 in every cell, labels inside existing command bounds, and
  nonzero teacher/student disagreement. Outcomes must agree with every report
  cell and pooled counts. Collision count is descriptive for correction data,
  never an acceptance result. Code: `mjlab_microduck.hc4u4_collection`.

## Next gate after collection

Do not train until all three full shard reports and admission receipts are
retained and hashed. Freeze the ordered sixteen-shard corpus and exact U4 stage
identity in a tested, committed trainer contract. Then one bounded seed-42 fit
may run if the deadline permits. The checkpoint hash must be committed before
closed-loop evaluation. Offline MAE thresholds remain 0.025 m/s and 0.050 rad/s.

Fresh held-out physics seeds for U4 are **347, 349, 353**, in order, with no
reuse of 293/307/311 for acceptance. The same twelve-cell matrix, 64 first
attempts/cell, 700-step ceiling, paired near/far specialists, and failure-priority
v2 protocol remain. Preserve every U3 per-cell and pooled outcome, before/after
speed, and motor gate without relaxation. Stop the candidate on any failed
seed; no additional acceptance seeds or cherry-picked checkpoint retry.

## Runtime and authority

Use one GPU workload at a time on 100.100. Recheck source, checkpoint, idle-GPU,
protected-service, and deadline gates before each launch. Bound smoke to three
minutes and each full collection service to ten minutes; no new collection
after 06:30 and reserve at least fifteen minutes before the 07:00 cutoff.
Later fitting/evaluation must receive explicit source-bound runtime budgets.
Keep protected services inactive and 100.98 reserved for FilmBrain. No hopping,
H2, video, raw perception, observation expansion or physical motion. Failed U3
is a data-collection student only, never a promoted controller.

Source validation before launch: 100 focused collection, first-attempt rollout,
and unchanged numerical-gate tests passed locally with CUDA hidden. The runtime
smoke and full-shard validation remain required; unit tests alone admit no data.

## Retained collection and frozen fit

Collection source: `e1e701a95d36857fc1d2e403cabf140d74fdfa8d`. The smoke and all
three full collection services completed with exit zero. Each full admission
object was independently recomputed from its report and dataset and matched
exactly. All three full shards had 384 clean outcomes, no collision/timeout/
fall/nonfinite/rated-speed failures, and passed the original motor checks.
These are collection results from U3, not acceptance of a new U4 policy.

Artifacts live under `artifacts/evaluations/hc4u4-e1e701a-s{seed}-corrections`
on 100.100, backed up under `artifacts/overnight-20260906-u4` locally. Each
directory contains `hierarchical-teacher-evaluation.json`,
`teacher-correction-dataset.pt`, and `admission.json`.

| Seed | Samples | Report SHA-256 | Dataset SHA-256 | Admission SHA-256 |
| --- | ---: | --- | --- | --- |
| 317 | 29160 | `ef34146fecd3113e4060fac7cded8241218727de282baf0f2e0281ff0886c329` | `515745a8c173ae27c02986c9e24a469ac8161c7e426fcecd057eed50d41ec1f2` | `23f79449729a8cafb2ad1e7563bb94b030ad1bc49a612812d6d55f71f5d52309` |
| 331 | 29043 | `6abbe05d1f2db363035a9805eab83d17fdde7ec5bc384a58f7f31908ac73a94c` | `5ff128a9bb8ac1943d288328a746c8cd9bf6712597fd9f8be60906ce7996bd98` | `fcc7f04032350fe545e7b371178f9055c08ff02030857072bd04d6a701ec3dc4` |
| 337 | 29329 | `e2b1aaabf33334067dad8e0c994760728fea9e2833826fb2bb0872e09cffeec1` | `47f13a95c3c28cc0e908de6963c7a9ed188965c32466d410db41edbfffeef3c0` | `4f6b2ea6bd8de03c74d82728d32a1eb5504cce7df213502156f52a9ff9fa49a2` |

The separate four-environment smoke retained 313 samples and four clean outcomes:
report SHA-256 `7f3aaa58c5f82f3c47b54ddf5703dde0dc5fc27ffdac263bb1cf6c8522087c76`,
dataset SHA-256 `66a25efdcd6585b62c090de62f1af61218040080d984869f8867b5315a1a5877`.
Its `smoke-admission.json` is explicitly not training-data admission.

The exact ordered sixteen-shard hash tuple is now
`HC4U4_REQUIRED_DATASET_SHA256` in `obstacle_supervisor_bc.py`: all thirteen
U2/U3 parent hashes unchanged, then 317, 331, 337 above. The parent paths are
retained in `artifacts/checkpoints/hc4u3-ab1d831-s42/supervisor.json` (manifest
SHA-256 `58d8327a6271a4702a10af48bcbd597837d4232c8b729ec9c3473a12f1da860c`).
The new stage is `HC4U4-near-state-correction-phase-BC`, sharing the unchanged
phase-expert implementation. Trainer rejects any other corpus order, correction
identity, seed, epoch count, batch size, or optimizer/model/gate configuration.
The loader requires the existing three-expert architecture and frozen gait;
recording remains unsupported. `hc4u4_gate` preserves the v2 numerical checks
while requiring U4 identity and only fresh seeds 347, 349, 353.

CPU reconstruction of the frozen seed-42 episode split gives 520,787 samples
and 6,678 episodes: 415,947 training samples / 5,342 episodes and 104,840
validation samples / 1,336 episodes. The legacy `successful_episodes` manifest
key counts all retained episodes, including older correction failures; it is
not a success-rate claim. Phase counts (approach/interaction/recovery) are
16,196 / 252,435 / 147,316 in training and 4,227 / 63,198 / 37,415 in validation.
Appending data necessarily changes the deterministic episode split and SGD
sequence; this is a data-augmentation experiment, not a fixed-split causal proof.

After focused tests and this contract are committed and pushed, permit one
fresh-initialized seed-42 fit under a retained service with `RuntimeMaxSec=1200`.
Write `artifacts/checkpoints/hc4u4-<fit-source-short-sha>-s42/supervisor.pt` and
its existing progress/optimizer and manifest files; never overwrite/retry a
rejected fit. Verify exact corpus/split, finite losses, offline gates, and back
up artifacts. Commit the final checkpoint hash before any held-out rollout.
Each held-out seed matrix is sequential near source, far source, then candidate
in one retained service capped at 1,200 seconds. No fit or matrix may start after
06:25 Shanghai, so its maximum runtime leaves fifteen minutes before 07:00.
All per-seed decisions must be recomputed from retained reports before advancing.
Passing all three remains a single-training-seed diagnostic, not multi-seed
training promotion, range-noise acceptance, video admission, or physical safety.

Pre-fit source validation: 144 focused U4 contract/collection, phase-supervisor,
legacy BC, first-attempt rollout, U3 and U1 gate tests passed locally with CUDA
hidden (61.29 seconds). Remote repetition and real-corpus contract validation
are required before launching the fit.
