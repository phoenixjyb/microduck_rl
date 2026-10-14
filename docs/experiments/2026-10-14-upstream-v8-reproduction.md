# Upstream v8 reproduction: source and deployment rehearsal

Goal: reproduce a proven walking/turning baseline before adding more rewards.
Simulation demonstration, robust navigation and physical deployment are separate
tiers. No new training or policy acceptance is claimed by this source slice.

## Recipe restored

Pinned source: `3750776761ecfaa5671b36972087cafba45b8527`, upstream
`improve_standup_velstand`, task `Mjlab-VelStand-Rough-Backlash-MicroDuck`.
Official v8 refers to run `pollen-robotics/mjlab_microduck/0045i2si`, checkpoint2749,
Hub revision `665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`.

The earlier integration retained the older VelStand recipe. This slice restores
the exact pinned `distill.py`, `microduck_velstand_env_cfg.py` and
`test_velstand_cfg.py` files, plus their transitive SO(3)/sitting reset helpers
from `mdp.py`, without replacing unrelated custom MDP code. The restored recipe
includes teacher-only calibrated yaw labels, finer yaw/in-place training and
expanded recovery starts. Student commands and the 61D→14D actor contract stay
unchanged. It is a source reproduction, not proof of a completed learning run.

Git blob IDs, in the same three-file order:
`793ea33fa7c13087f3b707232387e555796e69f6`,
`ff5b6dab1f137f86f65364c0983d777626794f4e`,
`15e69d7c4d21a5248f1e9804302564f8a7af6221`.

## Finite CPU rehearsal

`scripts/rehearse_official_velstand.py` reuses `infer_policy.py` and existing
MuJoCo scenes rather than introducing a new simulator. It checks the pinned
official ONNX SHA256
`2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`,
requires empty `CUDA_VISIBLE_DEVICES`, rejects GPU ONNX providers, and checks
raw observations/actions. BAM M6,50Hz control,200Hz physics,7.4V,kp200 and
fixed four-control-step delay are explicit. No action filtering, remapping,
autoreset or physical motion occurs.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/rehearse_official_velstand.py --output /tmp/duck-v8-new.json
```

Default: four continuous four-second segments after one HOME reset: idle,
straight(.3m/s), left(.2m/s,+.2rad/s), right(.2m/s,−.2rad/s). Use `--case` for
an isolated command, `--combined` for12s total, and `--scene groundcontact` to
exercise the original deployment scene instead of default `allcollisions`.
Each output must be new; total requested simulation≤30s. Output separates
actual body COM velocity, projected heading and initial-heading displacement,
and retains finite state, fall samples and raw motor-force diagnostics.

Neither flat CPU scene reproduces rough terrain, passive backlash, full training
DR or recovery curricula. This is explicitly a deployment rehearsal, not full
upstream plant qualification. Continuous segments after a fall are not
independent command trials. A moving rehearsal failure is evidence to investigate
the transfer boundary, not justification to change historical training gates.

## Next learning gate

Faithful VelStand BC requires the actual teacher PPO checkpoints:

- Stand: `pollen-robotics/mjlab_microduck/69u48n8l`, `model_9750.pt`.
- Walk anchor: `pollen-robotics/mjlab_microduck/441tzs6d`, `model_3750.pt`.

At inspection, neither known native expert cache contained them. The official
ONNX is not a replacement PPO teacher checkpoint. Resolve exact public assets
or the owner's available checkpoints; do not silently disable BC or fake them.
Then smoke the unchanged task and profile larger parallel batches to choose a
measured resource/transition budget. Only one GPU workload may run under the
existing lease; original vLLM/DINO ownership must remain protected. No automatic
reward-tweak queue, obstacle promotion or physical deployment follows source tests.
