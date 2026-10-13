# V8 on the flat walking backlash plant — predeclared diagnostic

The [plain-plant v8 check](2026-10-13-official-v8-native-reference.md) gives useful
mean turning but insufficient forward speed and fails all moving tracking cases.
The published training task uses mechanical backlash and encoder-through-backlash
feedback, while that check uses the plain plant. Test this specific boundary
before choosing another motion-learning recipe.

## One registered plant contract; not full training reproduction

Use the SAME pinned v8 ONNX/manifest/source lineage, raw61→14 inference, baked
normalizer exactly once and unchanged commands. Only evaluation task changes
from `Mjlab-Velocity-Flat-MicroDuck` to its existing registered
`Mjlab-Velocity-Flat-Backlash-MicroDuck` twin. The twin retains the walking
collision family and flat terrain. It adds serial passive backlash hinges and
the corresponding firmware/observation encoder view. These required mechanical
and sensing changes are one coupled plant contract, not separable causal effects.

This is NOT the full upstream Rough/VelStand/all-collision training plant and is
not described as such. In particular, do not naively swap to VelStand's play
config, which removes first-fall termination and adds topple pushes; those would
confound the unperturbed first-episode battery. Keep Velocity's fall termination,
reset distribution, sensors and all other configuration; clear curricula and
remove only the existing push event exactly as in the original collector.

The BAM subclass `BacklashEncoderBamActuatorCfg` uses the same XL330 M6 settings
as plain `FrictionDRBamActuatorCfg` (firmware kp200, VIN6.5–8.2, existing action
delay3–6), but closes feedback through the passive hinge. No class/physics code
is modified. The adapter default stays strict plain BAM; the new profile must be
explicit and exactly match the existing subclass. Preserve motor metadata order,
HOME, scale/history and output checks. Never clip low-kp targets or add filtering.

## Fixed case, safety and operation contract

`python -m mjlab_microduck.official_native_reference --reference v8-backlash --source <exact-new-commit> --run-id <unique-id>`.
Protocol `official-velstand-v8-backlash-reference-v1`; output
`artifacts/evaluations/official-v8-backlash-native-<source>-20261013`.
The collector's task override is allowed ONLY for the typed pinned v8 artifact
and that exact task; never for PPO checkpoints, arbitrary tasks or the older
archive. All preexisting PPO collectors keep their original default task.

Same seeds839/853/857 × idle/(.3,0)/(.2,−.2)/(.2,+.2), eight worlds, 240×.02 s,
startup60, body/gait traces and ordered create-exclusive case/result files.
Same per-world tracking, first-terminal, torque-utilization ≤.60 (raw p99≤.36 Nm),
rated-speed and named 14-motor-hinge range/stop gates. Passive hinges are not
mistaken for actuated servo columns. Position diagnostics explicitly label motor
hinge qpos; do not misrepresent them as output-encoder or physical-robot limits.
This check cannot qualify the robot or deployment.

Require tested exact committed source, CUDA-hidden native affected checks,
verified fixed policy/manifest bytes, actual robot metadata/HOME binding and
fresh GPU/service ownership. Retain the existing exclusive lease and occupancy/
temperature envelope. One retained eval-only service, timeout600s/outer900s,
8 GiB/200% CPU, Torch20% GPU memory; no overlapping Duck or unrelated GPU job.
The previous two matrices take ~56 s, but first compilation of the new morphology
may add time: the cap remains strict; preserve failure evidence rather than extend
an active job. Pause/restore ONLY the originally active user vLLM with the existing
post-exit guard; preserve DINO and other workloads.

A pass is numerical diagnostic evidence only, not teacher/physical acceptance.
A failure must be retained and diagnosed, not followed by a blind scalar test,
automatic PPO job, gate weakening, harder-skill promotion or video. No raw
perception or physical motion.

## Retained result — rejected

Executed at exact source `ebb9550855fb42ed760e0f9d1ec164933ac8945f` by
`recomo-duck-official-v8-backlash-native-ebb95508-20261013.service`.
All 12 cases completed their 240 steps without a terminal world. The service
returned success in 86.3764 s, including the new morphology's compilation;
this is execution success, not motion acceptance. All nine moving cases failed
the unchanged per-world tracking gate. Moving speed passed in only 5/72 worlds,
instantaneous yaw in 0/72; idle passed both in 24/24.

| Command vx / wz | Mean actual body vx / wz | Speed MAE | Yaw MAE |
| --- | --- | --- | --- |
| 0 / 0 | -0.000067 / 0.000055 | 0.000945 | 0.006858 |
| 0.3 / 0 | 0.133790 / 0.009986 | 0.166210 | 0.272604 |
| 0.2 / -0.2 | 0.086546 / -0.227004 | 0.113454 | 0.263426 |
| 0.2 / +0.2 | 0.088197 / +0.143149 | 0.111803 | 0.259940 |

These are descriptive means across 24 worlds per command, not substitutes for
the per-world gate. Relative to the plain-v8 matrix, backlash slightly increased
forward motion but did not resolve tracking. This does not isolate mechanical
backlash from its coupled encoder view, or reproduce the full upstream plant.

Maximum case force-utilization p99 was 0.535273 (raw force p99 0.321164 Nm),
below the retained utilization 0.60 / raw 0.36 Nm gate. Minimum named motor-hinge
hard-stop margin was 0.203695 rad; no rated-speed or range gate failed. These
motor-hinge diagnostics do not qualify output encoders or physical hardware.
GPU peak 1802 MiB, minimum free 14392 MiB, maximum temperature 63 C. The original
user vLLM was restored; DINO was preserved and all three other protected scopes
remained inactive. Native affected checks passed 1269 tests with 16 existing
warnings before launch. The earlier unrelated whole-suite collection/ADA failures
remain separate; this does not claim a passing whole-repository suite.

Full traces stay native at
`artifacts/evaluations/official-v8-backlash-native-ebb95508-20261013`.
The create-exclusive CPU closeout validated all 15 files and recorded:

- Result SHA256: `ee05ad9fdb9ce51f4c2a2d59339e765a6c607d302e8b7f66a53f8d046768c192`.
- Inventory SHA256: `0117b535f392927777b6cb2b6f5eedfe111c2c9c8524e7b441a66ffba92c6492`.
- Summary SHA256: `6647fbe6d2908c90854572d7c2e6aae67b5aa66425bf8ea3546b8a42e10ad19c`.
- Restoration receipt SHA256: `f0546b67d7dc5eba68684fecd86ac427f0ba8809d4eb2d34c511f253c62c5c1d`.

The lightweight inventory and summary were received on the Mac under the same
run name with `-closeout`, and their hashes matched. No full-trace duplication,
policy promotion, video or physical motion followed. The next source-only slice
is a pinned actor-mean import bridge: verify its raw-input output against v8 ONNX,
preserve exported normalization, and explicitly start critic/optimizer/history
fresh in any separately predeclared training experiment. ONNX contains none of
that historical PPO state; import parity alone cannot qualify a motion skill.
