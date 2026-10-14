# Matched learning-seed replication of staged head goals

Status: **driver implemented and CPU-tested; native admission and launch pending**.
This design follows the sealed [staged-head pilot](2026-10-14-v8-staged-head-goals.md).
It is not permission to promote a teacher, change evaluation gates or run hardware.

## Question and frozen pilot outcome

Pilot seed967 improves moving speed passes67→72/72 and complete-route worlds
7→13/72, but fails its own paired retention requirement because control has
67<69 speed passes against the original provisional parent. Both moving route
matrices remain unqualified. The pilot decision stays rejected. One stochastic
paired learner cannot establish robust benefit or explain the control loss.

The next question is reproducibility of the SAME curriculum—not another
head-range/reward/horizon/budget sweep. Authenticate pilot summary
`a81efd4ea694dd0d5147bd45c575544c273f910e7f5470de6c186a4bf1fc2b15`,
inventory `dc420efcea50e0f355d715f29a88c77c005f1062ba515db794762ddaf4871f6b`
and source `bb223f668c556de2dc651ef40a381ac5fcd5a0ad` before implementing admission.

## Predeclared sequence

1. Implement a dependency-injected replication driver with fixed prospective
   learning seeds977 and983, and CPU tests for source/run/seed/recipe identity.
2. Admit only the first matched seed977 pair after ordinary64×5 smokes,
  64×320 frozen-actor boundary probes and256×10 benchmarks for BOTH arms.
3. Run control and staged256×1000 sequentially, independently from the original
   headroom parent `dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a`.
   Do not load either pilot endpoint. Preserve all PPO tensors/Adam/critic,
   fresh local clocks and frozen actor affine/count0 exactly as in the pilot.
4. Evaluate only each final999 under the unchanged12-case held-out protocol.
   Independently seal results, temporal exposure and scoped restoration.
5. Only if seed977 passes EVERY unchanged paired diagnostic/retention gate,
   admit the already declared seed983 pair with the same budgets and protocol.
   If seed977 fails, stop replication and diagnose; do not reroll seeds or
   change a gate. Seed983 is not a rescue attempt after failure.

Each pair keeps control at parent head ranges throughout; staged small0–11999,
intermediate12000–17999, parent18000–23999. All other rewards, command sampling,
DR, BAM, runtime, actor61→14, unfiltered/unclipped actions and non-head curricula
are fixed. Full learning seed is the only change from the pilot. Smoke947,
benchmark953 and boundary971 remain fixed admission seeds, not new result arms.
Use distinct source-and-learning-seed-bound run directories and receipts.

Save initial and0/250/500/750/999 checkpoints, verify final env24000/Adam80000/
critic24576000, and retain exactly500/250/250 staged updates with actual exposure.
Never select a favorable intermediate checkpoint.

## Unchanged decision and operational limits

For EACH pair require ≥6 additional complete moving-route worlds versus equally
continued control, no moving-speed-count loss, no per-command heading endpoint
growth>.01rad/s or XY growth>.02m/s, safety/idle retention and all THREE walking
references in BOTH arms. The original-parent speed bar stays69/72 and mean
per-command MAE growth≤.02m/s. Keep every complete-cycle AND full-trajectory
endpoint/maximum gate. Qualify neither a pair nor a policy by pooled averages.

Only two independently passing prospective pairs can support a reproducible
curriculum-helpfulness claim. Even that is not full route qualification, an
obstacle/hop teacher or physical acceptance. Do not reclassify the rejected pilot,
substitute it for a missing confirmation arm or relax gates if control loses speed.

Per pair reuse the bounded guarded sequence: stage caps300/180/300/2100/600s
per arm, CPU comparison300s, outer7800s, MemoryMax8GiB/CPU200%/Torch20%GPU.
Both conservative benchmark projections must fit2070s before either foundation.
Run only one GPU workload at a time on100.100. Preserve DINO and unrelated work;
pause/restore only originally active USER vLLM after proven Duck exit. No100.98,
packages/drivers/reboots, raw perception, actor-input expansion, MP4, publication
or physical motion. The second pair needs review of the first sealed decision;
there is no automatic long-job queue in this design.

## Implemented first-pair driver

[The replication wrapper](../../src/mjlab_microduck/v8_staged_head_replication.py)
uses immutable per-seed recipe objects and explicit helper/argument injection.
No mutable module-level seed switch or temporary global recipe replacement is
used. Both the environment and PPO agent get the declared foundation seed;
other modes keep947/953 and the frozen probe keeps971. Receipts distinguish
the prospective `learning_seed` from their actual `stage_seed`.

The driver reauthenticates the rejected pilot summary/inventory before resolving
the ORIGINAL parent, never either pilot endpoint. It preserves all six short
admissions, temporal command/reward telemetry, first-update KL refusal,
checkpoint cadence, three walking references and unchanged decision thresholds.
Source/run identities include both seed and exact tested revision.

Seed983 may be configured and CPU-preflighted, but all training, evaluation and
comparison entrypoints remain locked. A later reviewed change must bind the
exact independently sealed seed977 decision before unlocking it. There is no
arbitrary previous-result path, automatic rescue seed or queued second learner.

The152 affected Mac CPU tests passed in174.14s, including both-seed object
isolation, cfg/agent seed agreement, metadata JSON roundtrips, seed/run mismatch
refusal, locked983 paths, unchanged pilot recipe against the small retained
seal and historical continuation tests. Compilation and diff checks passed.
Native tests, parent/runtime preflight and fresh GPU admissions remain separate.

First-pair invocation shape:

```bash
python -m mjlab_microduck.v8_staged_head_replication \
  --learning-seed 977 --source <source40> \
  --run-id v8-staged-head-replication-s977-<source8>-20261014 --preflight
```

The `boundary-smoke` and `compare` subcommands also require `--learning-seed`;
ordinary stages use `--arm control|staged --mode smoke|benchmark|foundation|final-eval`.
The guarded service is prepared only for977; its cap/ownership/restoration
contract is unchanged. Implementation/source evidence is not a completed run,
curriculum confirmation or accepted motion skill.
