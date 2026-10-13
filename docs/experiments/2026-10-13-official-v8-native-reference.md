# VelStand v8 native reference — predeclared next diagnostic

The [archived native reference](2026-10-13-official-native-reference.md) failed
all nine moving cases. A fresh upstream release is now available; evaluate its
actual bytes before deciding whether it can support our motion curriculum.

## Provenance, not acceptance

Official [HF release commit](https://huggingface.co/pollen-robotics/microduck-policies/commit/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448),
also tag v8, `665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`, published October 12.
The manifest records clean training source
`3750776761ecfaa5671b36972087cafba45b8527`, branch `improve_standup_velstand`,
run `pollen-robotics/mjlab_microduck/0045i2si`, checkpoint2749,
task `Mjlab-VelStand-Rough-Backlash-MicroDuck`. Its smoother-recovery/yaw-tracking
description is an upstream claim, not evidence of passing our current plant/gates.

Retained locally under `artifacts/community/intake-20261013/official-velstand-v8/`:

- `policy.onnx`: 793705 bytes; SHA256 `2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492`.
- `manifest.json`: SHA256 `c566390738dd90a54c1e2de829aaab0e904c636949e2c2c6158ee7bbb889b818`.

Static inspection recognizes the baked affine prefix and the same raw
61→14 interface, metadata term/motor order, HOME and action scale1. The numerical
normalizer is NOT assumed identical to the older artifact. The ONNX is still not
a PPO checkpoint. No weights are reverse-converted or automatically warm-loaded.

## Source lesson and attribution limits

The [exact source patch](https://github.com/pollen-robotics/microduck_rl/commit/3750776761ecfaa5671b36972087cafba45b8527)
calibrates yaw on the upright walk teacher's input to produce better BC labels;
the student/environment's raw command is unchanged. Zero-yaw labels and fallen
stand-expert routing are preserved. It also enables fine yaw-only tracking and
changes in-place command sampling. The broader lineage adds recovery coverage
and teacher masking. Thus v8 is a bundled reference, not causal proof that one
remap solves turning. Source comparison shows no framework pin, robot/BAM,
velocity-observation schema or external scale change; the v7 policy document at
that source is stale and is not used as v8-result evidence.

No runtime remap is introduced in our diagnostic. No upstream code is merged
into our training recipe merely because its commit or release description exists.

## Fixed numerical and operational contract

Entry: `python -m mjlab_microduck.official_native_reference --reference v8 --source <exact-new-commit> --run-id <unique-v8-id>`.
Protocol `official-velstand-v8-native-reference-v1`; output
`artifacts/evaluations/official-v8-native-<source>-20261013`.
Only these two explicit artifact hashes are allowed by the typed CPU adapter;
there is no arbitrary-model loader or live-main download during evaluation.

Use the identical collector/task/plain native BAM plant, seeds839/853/857,
four idle/straight/mirrored-turn commands, eight first-episode worlds, 240 steps,
startup60, health/lease checks and all tracking/motor/range gates from the earlier
declaration. Preserve body/gait traces and explicit raw-force units. No reward,
physics, head/body command, command remap, normalizer, action filter or gate change.
The prior archive takes 56.421 seconds on this collector; cap the new evaluation
at 600 seconds, with outer service900 seconds, 8 GiB/200% CPU and guarded
restoration of only the originally active user vLLM. Preserve DINO and unrelated
workloads. Native CUDA-hidden affected tests and exact bytes/source/ownership
must pass before launch; no duplicate or concurrent GPU job.

Retain per-case/aggregate identities, hashes and numerical decision in a fresh
directory. Historical archive results are fixed-protocol descriptive context,
not a matched learner causal control. Even a numerical pass is not physical or
automatic teacher admission. If it fails, diagnose its actual response and plant
boundary; do not weaken gates or repeat the older C1-S attempt. No automatic PPO
job, hopping/obstacle/football promotion, video or physical motion follows this
check. Results pending.
