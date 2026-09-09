# B1-N strict checkpoints and retained evaluation bundles

This completes the strict model-restoration and file-bundle components following
the [first-attempt recorder](2026-09-09-stance-attempt-trace.md). It does not launch
PPO, evaluate a trained stance candidate or change prior gait/hop checkpoints.

## Model and checkpoint contract

Read-only inspection of installed RSL-RL5.0.1 found separate `MLPModel` actor and
critic classes, not the older `ActorCritic` API. The owned builder uses stock
models with actor44 inputs/10 outputs, critic50 inputs/1 output, ELU hidden
layers128/128/64 and no empirical normalizer. Evaluation takes the deterministic
Gaussian mean; no sampled action or distribution update occurs. The existing
action limiter still clips/rate-limits this raw output before motor application.

RSL's Gaussian `std_type=scalar` means a state-independent diagonal scale, with
one learned direct-space parameter per action, all initialized to0.3. It does
not mean one shared parameter or log-space optimization. This makes the earlier
scalar-std declaration explicit without changing the installed convention.

Fresh initialization uses only the CPU generator inside a restoring RNG context.
It deliberately avoids `torch.manual_seed`, which also seeds/schedules CUDA
generators. Saving or auditing a checkpoint must not alter training randomness.
The fresh seed521 actor/critic state hash is
`27e98087a3043f8a4b4fdc9b81958260db849cd4dbfec5dc5cb6a8810e0b7e8b`.

The builder pins RSL version and these selected implementation files:

| RSL file | SHA-256 |
| --- | --- |
| models/mlp_model.py | `6219ebb3ed4df036dae7ff1c30d0fbb169eaaa659e44a659dd757a292124476b` |
| modules/mlp.py | `bad934b364b26cd47b6d1612e00ace107a425ae4d272c0cd0c82cbfc57bbcdbc` |
| modules/distribution.py | `4631eae1939dcd6b065d79c3c0128a88ba2c6126d08f46944c8795682a208a1b` |

These selected pins are not complete binary/runtime equivalence. Full launch
identity and dependency coverage remain the guarded runner's responsibility.

`stance_checkpoint` exports only actor/critic tensors and exact metadata, with
protocol `football-b1n-evaluation-checkpoint-v1`. Metadata binds architecture,
source, runtime/training-launch hashes, purpose, seed, environment count,
iteration and fresh initialization. Pilot uses seed521/512 worlds; disposable
smoke uses seed523/64. Only pilot iterations128/256/384/511 can be loaded for
the held-out evaluation; smoke and initialization exports cannot be promoted.

The loader hashes bounded bytes before weights-only CPU deserialization, builds
fresh models, validates exact tensor names/shapes/float32/finite values and
positive Gaussian scale, and loads strictly. It rejects old61D gait weights,
extra normalizers/optimizer state, changed activation/configuration, malformed
critic weights and metadata drift. Restored evaluation parameters are frozen.
This export is **not a resumable PPO checkpoint**. The eventual trainer must
separately save durable optimizer, iteration and RNG state; no resume support
or completed training provenance is claimed here.

## Exclusive artifact publication and verification

`stance_evaluation_bundle` binds independently supplied checkpoint identity and
trace binding to exact launch/runtime/checkpoint bytes. Launch metadata includes
the binding before its own SHA is assigned, avoiding a self-referential hash.
Training-source and evaluator-source identities are distinct and retained.

Before publishing, it strictly restores the checkpoint, replays the first-attempt
trace and recomputes every recorded deterministic actor action. It permits only
atol1e-6/rtol1e-5 cross-backend action arithmetic error and records the maximum
residual; these tolerances do not relax physical or performance scoring gates.
The new directory is exclusive. Every file and directory entry is fsynced, with
the manifest written last. A partial failed write is retained without a success
manifest and cannot be silently overwritten or resumed.

The exact seven-file inventory is `launch.json`, `runtime.json`, `checkpoint.pt`,
`trace.pt`, `restore.json`, `score.json` and `manifest.json`. A read-only verifier
requires an independently retained manifest SHA, checks every filename/hash/
length, restores and rescales no observations, recomputes actor actions and
numerical scores, and compares both receipts. Rehashing an invented positive
receipt still fails recomputation. Use only owned guarded-run artifacts, not
arbitrary tensor archives; size limits are not a hostile-input sandbox.

Runtime JSON is currently byte/source-bound, not fully interpreted as a compiled
plant/reset/contact contract. These libraries do not attest live service state,
checkpoint training history, complete sensor/motor provenance or an actual GPU
capture. They preserve `provenance_validated=false`, `checkpoint_admitted=false`,
`learned_stance_accepted=false` and `physical_motion_authorized=false`.

## Validation and next work

Tests use freshly initialized synthetic checkpoint metadata, never pretend those
weights underwent128 training iterations. A single real Warp CPU policy tick
supplies an interface fixture, not a full hold, trained candidate or new CUDA run.
Tests cover strict restore, caller RNG preservation, malformed weights, legacy
shapes, smoke/initialization refusal, exact byte hashes, changed actions, corrupt
files, rehashed false receipts, symlink paths and a simulated disk failure.

Local validation:221 focused CPU tests passed in15.50s, including38 new model/
bundle cases. The builder forces CPU device placement without changing the
caller's default device and refuses non-float32 initialization. Markdown HTML,
table and relative-link checks passed. Exact-source Linux confirmation is next.

Remaining before training: complete the compiled-plant/reset/contact binding and
the exact four-checkpoint/three-seed/128-world evaluation matrix; finish PPO
transition/timeout semantics, full training-state saves and bounded launch
assembly. Then run the declared disposable64-world/16-iteration smoke and use
measured timing before the fresh nominal-stance pilot. Do not rerun closed CUDA
or native-hold probes, relax historical gates, or claim learned football balance.
