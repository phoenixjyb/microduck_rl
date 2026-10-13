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
CPU ONNX on all128 probes. The native integration smoke has not yet run.
