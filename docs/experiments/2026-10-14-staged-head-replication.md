# Matched learning-seed replication of staged head goals

Status: **seed977 completed and rejected; replication stopped; seed983 remains locked**.
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

## Seed977 native admission and retained service

Executed source `986c6f4facb44333a3daa76d1f411058c0820ac7` passed all152
native CPU tests in18.28s. The unchanged six pilot training receipts also
reauthenticated under the refactored default helpers. Native preflight bound
the exact rejected pilot seals, original parent actor/critic/Adam tensors,
runtime pins and seed977 matched recipes. Neither pilot endpoint is adopted.

The nonenabled user service
`recomo-duck-v8-staged-head-replication-s977-986c6f4f-20261014.service`
started2026-10-14 19:35:07 Shanghai with the unchanged7800s/8GiB/CPU200% cap
and post-exit scoped restoration. Retained assets belong under
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/v8-staged-head-replication-s977-986c6f4f-20261014`.

| Admission | Control | Staged |
| --- | --- | --- |
| Ordinary64×5 smoke seconds | 4.388 | 4.408 |
| Smoke KL mean / max minibatch mean | .018335 / .032874 | .020124 / .034112 |
| Frozen64×320 boundary falls / resets / NaNs | 0 / 0 / 0 | 0 / 0 / 0 |
| Actual range action steps | 320 parent | 80 small / 80 intermediate / 160 parent |
| Benchmark256×10 seconds | 8.016 | 7.852 |
| Benchmark KL mean / max minibatch mean | .016533 / .021425 | .013359 / .023881 |
| Conservative1000-update projection | 1262.38s | 1237.82s |

All six short admissions passed before either foundation. Both projections
fit2070s; actor tensors/affine/count0 stayed unchanged in the frozen probes.
Staged intermediate/parent actual outside-prior samples in all four components
were `[682,783,703,651]` / `[2068,2698,3107,2683]`; no resampling was forced.
Smoke and benchmark keep actual stage seeds947/953, while receipts correctly
bind the prospective learning seed977. The live foundation launch records977.

Short receipt hashes (control / staged):

- Smoke:
  `b81c2676b5d01b170c8a6993de122c92427fcc10ea1ceb35b9820a3ac5681fc3` /
  `8738c1a541936ff8308a55dfbe0fe26b95f60e69b7f5b7ebb7bc77b3f2804189`.
- Boundary:
  `d21088a677cb7e91dcc38f94b4738589049f28346ba012fbf4d60c8307007cf3` /
  `aacf62d48d67faf70e19658416b12a36cbf9e5b7b2748109781853b5a725d633`.
- Benchmark:
  `da63d36c050b48bbd5d1a031832113bb5f2266f6bd009a4b3f7d3dfd16a8a544` /
  `2017141b6a73618a2204c834cedfbc5331a046f8b36b581ca32c3f53a6a79f5b`.

At the first retained foundation check control reached update140/common3360/
Adam62800; latest NaN terminations0. First-update KL mean.0150344/max.0209004
passed, with original inherited LR.00011390625000000005. Durable initial PT
SHA `c6b09dc25d4693074e086faa5d5745ce04f5aa30cc12a03e9531f65e6a7e9562`
matches the original parent-preserving initialization. GPU occupancy1710MiB,
free14484MiB at64C comprised preserved DINO2029 and only the leased Duck
child; all four protected scopes were inactive during training. This is a
live snapshot, not completion or a numerical qualification result.

Executed handoff/context/restorer/unit hashes respectively:
`90fef04cb1e65620a66dc3ae742e9dc9de775a3dbdcbef9e3a715112f7df5def`,
`0a54f40603b7c1df0b2c18c85952f23c110d6b2f7fef6c895065ca47d4a9ecbe`,
`089457b02bed6eaf35c012cff3cd872b4980c81c22bcac6f732d639e98f0b0df`,
`91319a60f527ddbe1c2effdf2a0e5a4de6bde0de0ba6b2af9acf228f2056a05c`.
That snapshot preceded completion. The service subsequently finished both final999
checkpoints, both held-out evaluations, the unchanged three-reference comparison
and guarded restoration, as retained below. Seed983 has no enabled job and stays
locked. Never edit or fast-forward an active training checkout.

## Seed977 completed closeout and rejection

The sequential service exited successfully on2026-10-14: comparison completed
at20:01:23 Shanghai and scoped USER vLLM restoration at20:01:52. This operational
success is not a policy pass. Both foundations completed256×1000 updates with
6,144,000 transitions each, zero NaN terminations and every declared0/250/500/750/999
checkpoint. Final iteration999/local env24000/Adam80000/critic24576000 and frozen
actor affine/count0 were independently rechecked. Initial tensors and optimizer
match the original parent, not either rejected pilot endpoint.

| Retained result | Control | Staged |
| --- | --- | --- |
| Foundation seconds | 645.66 | 643.45 |
| Training falls across256 worlds | 809 | 615 |
| Complete moving routes, out of72 | 15 | 8 |
| Moving speed passes, out of72 | 67 | 67 |
| Idle route/speed worlds | 24/24 | 24/24 |
| Held-out motor/range/idle safety | pass | pass |
| Projected-heading / cautious walking references | pass / pass | pass / pass |
| Original-parent walking retention, speed bar69 | fail | fail |

All12 held-out cases per arm completed without terminal substitution. All nine
moving cases remain route-not-qualified in both arms. The deterministic
`staged-head-replication-rejected` decision has four failures: route improvement
is−7 rather than at least+6; negative-turn heading endpoint regression exceeds
.01rad/s; and BOTH arms fail original-parent speed-count retention67<69. Equal
paired speed counts do not satisfy original-parent retention.

### Read-only diagnosis, without another learner

The negative-turn mean full-trajectory heading error rises
.053102854→.096465957rad/s (+.043363103). Mean body yaw rate changes
−.189123→−.114103rad/s against the−.2 command. Complete negative-turn routes
fall5→1/24; cycle heading endpoint failures rise28/152→64/148. Its speed MAE
improves.055099→.042331m/s and cycle XY failures fall4→0, but neither compensates
for lost turning. Straight routes also fall6→3/24 with speed passes unchanged19.
Positive-turn routes remain4/24. This is a measured steering/retention failure,
not evidence of a motor safety violation or an evaluator crash.

Actual staged exposure is exactly12000/6000/6000 action steps; all four head
components have nonzero sampled commands throughout. Intermediate outside-prior
counts are `[1076188,1072970,1018507,1027508]`; parent-stage counts are
`[842526,836630,865078,875886]`. Thus a dead curriculum or missed range mutation
does not explain this rejection. Training falls are lower in staged, but held-out
route quality is worse: training stability/reward averages are not sufficient.
The observed seed-dependent reversal does not establish the causal mechanism;
do not assert that a specific range boundary or reward caused it without a
separately declared diagnostic. No intermediate checkpoint was selected/evaluated.

The prospective replication stops here. Do not unlock983 as a rescue, reroll a
seed, soften route/retention gates, promote either checkpoint, or start an
obstacle/hop teacher from these results. Any follow-on experiment needs a fresh
question and frozen protocol addressing straight-speed retention and negative
turn response; it is not an extension of this rejected confirmation campaign.

### Exact retained evidence and source repair

Native retained state:
`/home/converge/.local/state/v8-staged-head-replication-s977-986c6f4f-20261014-handoff`.
The independent CPU-only verifier reauthenticated all receipts, initial/final
tensors, temporal exposure, all trace/route hashes, the three walking references,
deterministic comparison and live restoration cgroups. It sealed151 run files,
373,219,566 bytes. Raw traces/checkpoints stay only on100.100; the Mac keeps just
the small summary/inventory in ignored artifacts.

- `replication-summary.json` SHA256:
  `a344c638a7da8f398cc1832d4da41680f8cdeb5c0dc251366b3d8b9f64c82634`.
- `replication-manifest.json` SHA256:
  `66081b90b72ed98a4ae01cda7198b09085ed83167166d4027e486ea61c8cdfaa`.
- Historical decision SHA256:
  `6786ae7b4b4c81c154fb0a5fb24d4d6fa68a5f3bca052fbc7f245fe4b1aa8246`.
- Final control / staged PT SHA256:
  `a03b77282b480daca6e6ff8b3e61f1b6f16c5c4416c6962edd1cb5b503e951f7` /
  `e7a6a8f5eeb505a66b128b95813345189ba097a2e0a815d9ab80502820c015eb`.
- Final evaluation control / staged result SHA256:
  `5559acb977ca5781b8b8422857b55ab71c2f03251d5c199f22b49e4a47bff8fd` /
  `dfa105554ce36e0f0e9d27119173f509713c6295f37e5cf03a14bb9bf30e0728`.
- Native independent verifier SHA256:
  `a338b1c3ffe9d62ead13794dbd301ed3d54d30cef4a60c0b4b6ade1df7271929`.

Observed Duck GPU peak1754MiB, minimum free14440MiB and maximum temperature67C.
The exit guard proved DINO2029-only idle before restoring only the original USER
vLLM; live GPU owners are DINO2029 plus vLLM2163857 in that service's cgroup.
The other three protected scopes remain inactive. No further GPU job is queued.

The historical comparison exposed a formatting defect: the failure string
`fewer-than-six-additional-complete-moving-worlds` was serialized as a character
list by the arm-name remapper. Its bytes and decision remain untouched. The
source repair preserves strings and relabels only structured arm records;
thresholds, groups and decisions do not change. The regression plus replication
and exposure suites passed67 CPU tests on the Mac in11.80s; diff checks passed.
This tested reporting repair and Luna's independent receipt audit do not qualify
a motion skill or replace the numerical rejection.
