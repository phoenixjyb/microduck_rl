# HC4-U4 near-envelope student-state correction

Date: 2026-09-06. Status: predeclared bounded correction collection, not a policy.
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
