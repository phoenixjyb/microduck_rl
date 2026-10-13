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
perception or physical motion. Results pending.
