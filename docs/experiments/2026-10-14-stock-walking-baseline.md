# Stock walking baseline, from scratch

First milestone: visible upright forward walking and both-sign turning. This
is a foundational gait, not VelStand recovery, obstacle negotiation, hopping,
football balance or hardware acceptance. Exact VelStand teachers are unresolved;
this run needs none. Close the custom v8 reward-tweak line rather than extend it.

## Frozen learning recipe

Task `Mjlab-Velocity-Flat-Backlash-MicroDuck`; fresh random actor/critic/Adam and
normalizers, no resume, ONNX import, BC or W&B network. The velocity recipe and
backlash wrapper match source `3750776761ecfaa5671b36972087cafba45b8527`.
Retain its BAM walk/backlash plant,61D observations,14 raw actions,50Hz control,
24 steps/update, PPO5 epochs/4 minibatches/adaptive LR.001, normalization and
all existing DR/rewards/curricula. Operational logging is TensorBoard, local
checkpoints every250 updates and final checkpoint, with no video during training.

This is not reproduction of the published VelStand v8 learning run. Flat stock
walking is deliberately simpler and sufficient for the first gait milestone.
Use the standard training function; do not silently narrow commands, disable
head goals or insert a new smoothing/reward/observation wrapper.

## Admission and learning budget

Learning seed1019 is fixed before runs. First smoke64×5, then sequential fresh
profiles1024/2048/4096×20 with the same seed. Each short child≤300s; one retained
profile service≤1800s including exit/restoration. Select the largest safely
fitting batch, using measured steady iteration time and GPU/RAM occupancy;
the profile checkpoint is not a warm start or a skill verdict.

Target first training slice:6000 updates with the selected batch, unchanged
stock curricula. At4096 worlds this is589,824,000 transitions (96× a previous
256×1000 diagnostic). Do not assume4096 fits until measured. Predeclare the
selected batch/source, a conservative runtime projection and a bounded retained
training service before launching. If the projected run exceeds10800s, do not
quietly shorten learning into another tiny diagnostic: choose a separately
declared continuation plan or report the capacity constraint.

## Shared-host ownership

100.100 only, exact clean source, GPU
`GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`, existing owned lease. Preserve verified
DINO2029 and all unrelated services. Reuse scoped original USER vLLM pause and
post-exit restoration; all other protected scopes remain inactive. Never train
alongside vLLM or on100.98/NX/FilmBrain. Existing resource bounds remain: Torch
20% GPU allocator, total GPU≤6144MiB/free≥8192MiB/temp<75C, serviceMemoryMax8GiB
andCPU200%. A safety monitor checks live ownership/temperature during each run;
any violation terminates only its own Duck process. Outer service bounds and
ExecStopPost retain restoration independently of this chat. No packages,
drivers, reboot, credential setup, publication or physical motion.

## Behavior check after learning

Keep historical failures unchanged. Inspect the final checkpoint in its own
flat/backlash training plant, then through the CPU deployment rehearsal with
the matching exported normalizer. First report forward speed, both turn signs,
upright survival and idle drift alongside a visual rollout. A useful simulation
demo need not satisfy every prior diagnostic precision gate, but neither a
finite checkpoint nor a demo means robust navigation or hardware qualification.
Only then proceed to stop/recovery and an obstacle supervisor over the working
gait; hopping/football remain specialist tracks.

## First native admissions and thermal stop

Source `275ef098dac8fad5c9812baf6f17be8d559a81fd` passed24 focused checks on
Mac/native (6.56/5.25s). Seed1019 smoke64×5 completed6.13s; final checkpoint and
61D→14D ONNX were finite, NaN termination maximum0, GPU peak1542MiB/temp57C.
The1024×20 profile completed23.68s, finite checkpoints/exports/scalars, zeroNaNs,
23,584 final transitions/s, peak2688MiB and67C.

The2048×20 profile was stopped at about iteration16 by the unchanged shared GPU
budget/temperature guard (exit143). Last successful samples were4060MiB used /
12134MiB free, temperature72→73→74C. The memory evidence points to the75C thermal
threshold, although the failed sample itself was not retained.4096 was not run.
No failed profile is accepted; preserve its partial artifacts. Scoped restoration
completed with vLLM owning the GPU again and DINO2029 preserved.

Next bounded admission is **thermal-1024:1024×120**, seed1019, fresh stock recipe,
CPUQuota100% instead of200%, child≤600s and service≤1000s. This is a sustained
resource test, not a promoted gait or warm start. It retains the75C cutoff and
all other bounds. Choose a full training batch only after stable temperature
and measured runtime; do not relax cooling limits to get4096. The native profile
assets remain at `artifacts/training/stock-walk-profile-s1019-20261014`.
