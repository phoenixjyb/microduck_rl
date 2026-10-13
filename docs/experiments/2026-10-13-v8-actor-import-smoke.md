# V8 actor-only import and five-update integration smoke

The [three retained native reference matrices](2026-10-13-v8-backlash-reference.md)
fail the unchanged moving tracking gates. The v8 mean actor nevertheless supplies
more useful signed turning than the old archive. This is a candidate seed for a
new learning recipe, not an accepted teacher. Do not resume its unavailable PPO
state or invent an exploration variance, critic, Adam history or sample count.

## Source-only bridge contract

`mjlab_microduck.v8_actor_seed` accepts only policy SHA256
`2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`.
It binds the metadata and complete graph: raw61 → Sub(mean) → Div(exported
positive denominator) → 512/256/128 ELU → linear14. All initializers, wiring,
shapes, coefficients and finite values are validated before modifying the target.
No feature remap, external normalization, output filter or low-kp target clipping.

Only the deterministic MLP and affine transform are imported. The target must
use `mjlab_microduck.v8_actor_seed:FrozenV8Actor`, whose constructor establishes
`eps=0`, `until=0` on the stock empirical normalizer. Its `_std` buffer holds
the exported denominator, NOT a recovered historical standard deviation;
`_var=denominator²` is derived affine buffer consistency, not measured history.
The target updater counter stays zero and is labeled as such. This class is
mandatory in the saved runner config and checkpoint reconstruction: Python
`eps`/`until` are not stored in a generic MLP's state dict. The pinned import
digest is a persisted actor buffer. Unseeded inference/export and a second import
are refused. Train mode cannot silently unfreeze the observation transform.

The candidate must match CPU ONNX Runtime and the stock RSL export-wrapper
forward on supplied raw float32 probes, maximum absolute error ≤`2e-5`, before
the caller is changed. Copy parameter values in place; preserve fresh optimizer
parameter identities and critic/exploration state. This is import consistency,
not simulation, deployable-export or capability qualification. The regular
runner export/metadata path remains mandatory for a trained policy.

## Predeclared integration smoke — not a longer training admission

`python -m mjlab_microduck.v8_actor_smoke --source <exact-tested-commit> --run-id <unique-id>`.
Protocol `v8-actor-only-native-integration-smoke-v1`.

- Native100.100, exact clean source, frozen installed runtime and unchanged BAM.
- Existing plain `Mjlab-Velocity-Flat-MicroDuck` foundation smoke recipe, seed947,
  64 environments, five PPO updates, 24 steps/update: 7680 sampled transitions.
- Actor mean/affine from pinned v8; fresh state-independent Gaussian std0.3
  (scalar parameterization), stock fresh critic/Adam, zero target iteration and
  curriculum clocks. The std is a new target choice, not upstream provenance.
- Preserve base smoke rewards, plant, noise/delay, sampler and stock curricula.
  This intentionally tests integration into our target recipe; it does not
  reproduce v8 training conditions or assert compatibility for a long warm start.
- CPU parity on the actual initial raw64×61 observations; bind live named
  14-motor order/HOME and the plain BAM action contract. No physical hardware.
- Initial checkpoint, per-update finite reward/optimizer and motor metrics,
  final `model_4.pt`, frozen normalizer counter, final regular normalized ONNX
  and create-exclusive launch/import/result receipts retained. Warning-only
  exporter failure is fatal to this smoke receipt.
- Existing gross motor abort guards, NaN termination guard and command coverage
  checks; these are process guards, not substitutes for the unchanged held-out
  tracking/range/motor gates. Training falls must be reported, not hidden.
- Single retained user service: internal/TERM cap300s, outer600s, 8 GiB RAM,
  200% CPU, Torch20% GPU allocation, existing occupancy/temperature envelope and
  exclusive lease. Pause and restore only the originally active user vLLM;
  preserve DINO and every unrelated workload. No automatic follow-on job.

Before launch: focused CPU tests, actual artifact parity, native CUDA-hidden
affected checks, exact source/artifact/runtime/service/GPU identity and guarded
restoration preflight. After completion: verify counters/checkpoints/exports,
report finite losses/falls/motor metrics and restored original workloads.
No long foundation job, checkpoint-selection search, accepted motion skill,
obstacle/hop/football progression, MP4, raw perception or physical motion is
admitted by this smoke. A benchmark and a complete target curriculum/held-out
protocol must be separately predeclared before longer learning.

## Preparation evidence

Local focused importer/reference/integration checks passed91 tests with two
legacy-export deprecation warnings. Actual pinned v8 bytes
matched both Torch actor and stock export-wrapper forward on128 deterministic
raw probes with maximum error `1.9073486328125e-6` (tolerance `2e-5`); normalization
updates in train mode left the target counter at zero. These include raw zero,
upright/forward command and seeded affine-scale perturbations, not physical
robot measurements. The actual-artifact test also reconstructs the actor from
its state and checks the regular runner's serialized ONNX against the original
CPU ONNX on all128 probes.

## First native attempt — constructor rejection before learning

At source `371474f360bbab1aa81dae86cc22a0b321182d4a`, native affected checks
passed1307 tests with18 warnings. The new retained service then stopped with
`ValueError: single raw actor observation group required` before constructing
the algorithm, importing the actor or taking any PPO update. The packet contains
only `launch.json` and `failure.json`, with no checkpoint/update log. It remains
native at `artifacts/training/v8-actor-smoke-371474f3-20261013`; failure SHA256
`5375f85924bad72bbfa61fba0ac3790cbbd9badf320865b68bd8d539939ebd6b`.
The guarded original-vLLM restoration receipt has SHA256
`c4a1858c8b71c0f6437bcc324958808bc10629714d974987475dcc70667abd0b`;
DINO and the other protected scopes were preserved.

Read-only diagnosis checked the actual registered cfg and installed RSL resolver:
they use/preserve `('actor',)`, while the bridge compared only to `['actor']`.
The correction accepts list or tuple containers ONLY with that exact singleton
order; it changes no observation layout, graph, normalizer, reward, action,
plant, gate, budget or seed. Add a real serialized-cfg/PPO construction regression
test, repeat focused/native checks, and use a new source-bound retained run ID.
Never restart or overwrite the failed packet. The same five-update predeclaration
applies to the corrected attempt; no longer job or motion claim is admitted.

## Corrected native smoke — complete, not a skill qualification

Source `c178fbcc53c9d7e16d9bd09b5b21896fecf18645` passed97 local focused tests
(two export deprecation warnings) and1313 native affected tests (18 warnings).
`recomo-duck-v8-actor-smoke-c178fbcc-20261013.service` completed successfully:
five PPO updates,120 environment steps,7680 transitions, finite losses and
three training falls, with zero NaN terminations. Learner time was4.5023 s;
the outer worker including export checks took4.6404 s. This tiny2.4-s per-world
window is not a stability, tracking, obstacle or hop qualification.

The actual initial raw64×61 actor import matched CPU ONNX with maximum error
`3.2782554626464844e-7`. The CPU closeout verified changed actor/critic weights,
all17 Adam parameter states at step100, initial empty Adam, iteration0→4 and
curriculum clock0→120. The imported actor transform stayed byte-equal with
count0; the new critic accumulated7680 samples. Normalized metadata-bearing ONNX
was produced by the normal registered runner save/export path and CPU-smoked.

Maximum training force-utilization p99 was1.067539, soft-limit exposure6.0593%,
rated-speed exposure0.06975%, mean absolute mechanical power4.5446 W and thermal
proxy0.127936. These stochastic-training metrics passed the declared gross
process guards, **not** the stricter deterministic held-out motor/skill gate.
GPU peak1504 MiB, minimum free14690 MiB, maximum temperature55 C. Original user
vLLM was restored; DINO and the other protected scopes remained preserved.

Native packet: `artifacts/training/v8-actor-smoke-c178fbcc-20261013`.
The create-exclusive CPU inventory/summary certify:

- Result SHA256: `bfd1eb3081bc0dcdc6ce8d291714c93944d4a6e92a87a92c83d732c9269c8e68`.
- Initial PT SHA256: `1f4c8b3c1538ce88c1cb0409658c998ce0d6454b960214775e422015d898b2e3`.
- Final PT SHA256: `9f5762e866d125cb657c02256b6bef37c23485e60fffe65eeba7f28beff2e5d7`.
- ONNX SHA256: `1db7c36bfbffc308247696435e07098d24c4133b8d4ea59b8eb6e4e611f30ba8`.
- Inventory SHA256: `d105c72bb404827dfae9a409ed4c415ccda34a1d664b3f699144b4b63a5bd753`.
- Summary SHA256: `4c19c641eebc97cb7a183be0aa21337537e88bdfff8f789ccd9bb320b85c49a9`.
- Restoration SHA256: `108bf9e08d8b425a5b2d4bd8e4a090479fdde16a779fc0edb37e1a7f31f40e26`.

Only the lightweight inventory/summary are copied to the Mac (`-closeout`),
with matching hashes. No motion acceptance or physical claim follows. Next
predeclare a256-world timing benchmark, exact actor-checkpoint held-out path,
and a bounded foundation budget before launching further learning; retain all
old gates and reject rather than promote any failed endpoint.
