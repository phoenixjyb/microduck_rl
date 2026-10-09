# Ada unconstrained Duck forward baseline

Branch `feat/athletics-obstacle-curriculum`, base
`fb6dffdbef490f66f32d73efa51930eae6eabca1`.

The [post-reboot Torch/Warp health check](2026-10-09-ada-runtime-smoke.md)
completed, but did not qualify simulation or authorize training. The next fixed
question is whether the actual Duck plant transfers and executes one stock
MuJoCo-Warp forward on Ada, and what complete CPU/GPU residuals it produces.
This new diagnostic does not modify or bypass the old numerical admission,
Blackwell collector, solver-origin gate, runtime, policy, reward or curriculum.

## Predeclared scope and limitations

One fresh child builds the existing `stance_warp_runtime.build_entity().compile()`
plant, matching a second fresh same-host CPU plant descriptor and all its asset
hashes. Topology is 21 qpos, 20 velocity DOFs, 14 controls and 16 bodies. It uses
the standing keyframe, zero velocities and controls, and the existing placement
routine leaving each foot approximately 0.1 mm above the floor. Placement already
calls native forward; afterward explicitly zero warmstart, time, applied forces
and controls, and cast qpos to float32 then back to float64. Native MuJoCo and
Warp receive those exact representable state values. Model coefficients remain
native double versus Warp float32; their rounding is part of this comparison.
Cross-platform CPU model bit identity is not asserted.

Two identical GPU worlds, dense solver configuration, `nconmax=128`, `njmax=512`,
one stock forward call, no graph. Require native `nf=nefc=ncon=0` and GPU
`nf=nefc=[0,0]`, shared `nacon=[0]`. BAM is **not initialized or computed**;
its compiled friction is zero. This deliberately isolates reset, kinematics,
composite inertia and unconstrained gravity/dynamics. It does **not** exercise
motor friction, contact forces, Newton constraint solving or rolling-ball balance.
The independent reviewer recommended prepared BAM friction for solver coverage;
the owner narrowed this first run to an explicitly unconstrained baseline.
A constrained/motor-aware follow-up requires a separate predeclared probe.

Retain every element of 20 fields: qpos, qvel, time, warmstart, ctrl, applied
generalized and Cartesian forces, body positions/quaternions/inertial transforms,
subtree CoM, body/composite inertias, spatial velocity, bias/actuator/constraint
generalized forces and smooth/final acceleration. Complete shape and finiteness
checks cover both worlds, not a prefix. Position/CoM residuals are metres;
quaternions/transforms dimensionless; generalized state/force/acceleration arrays
contain mixed translational/angular units, and spatial inertia arrays contain
mixed mass/inertia/moment units. Retained field identities preserve those distinctions.

Save native float64 and Warp float32 arrays in separate uncompressed NPZs with
file and per-field hashes. Report max absolute and RMS differences against native
double and bit mismatches against native cast to float32. These are descriptive:
**no new dynamics tolerance or numerical acceptance gate is introduced**.
qpos/qvel/time/warmstart/ctrl/applied-force bytes must remain unchanged by forward
on each backend. Zero integrations, motor preparations, optimizers or policy calls.
All exclusive-GPU, runtime-origin, solver/simulator qualification, training,
learned-skill and physical-motion flags remain false, regardless of residuals.

## Execution bounds

Use only 100.100's exact existing frozen venv, clean source branch and installed
versions, actual sm89 RTX 4090 Laptop 16376 MiB and driver 595.91.07. Reuse the
unchanged [health supervisor guards](2026-10-09-ada-runtime-smoke.md): read-only
existing advisory lock, CPU-hidden owner, fresh CUDA-0 child, exact UUID and device,
before/during/after foreign PID/name and protected-service preservation, at least
10240 MiB free, aggregate growth at most 2048 MiB, below 65 C and utilization at
most 85 percent. Grounding DINO must remain untouched. The stopped-vLLM authority
does not authorize other workload stops, package/driver changes or service restoration.

Same 180-second user service, 150-second owner / 120-second child watchdog,
reserving at least 20 seconds of the owner window for post-child closeout,
6 GiB RAM, 200 percent CPU, 64 tasks, Nice 10, 16 MiB per-file, no core/restart,
control-group kill, 10-second stop timeout; four thread limits 1 before imports.
Bind committed new module and unchanged helper bytes. Source-specific absent
evidence/cache directories; child-only PCH off, driver compiler cache disabled.
Two complete arrays total below 2 MiB per NPZ; no shared cache mutation or overwrite.
Failures retain evidence, retire only the owned child and require read-only diagnosis.
One bounded GPU attempt only after focused CPU tests and independent review.

The positive collection decision is
`ada-unconstrained-duck-forward-collected-pending-reception-not-training`,
not a numerical pass or accepted retention. A separate CUDA-hidden, bounded
post-exit receiver binds a fresh same-host plant descriptor and uses
`receive_payloads()` to reload both NPZs with `allow_pickle=False`, validate full
dtype/shape/finiteness and independently recompute every residual row.
Post-exit reception must rehash all files, recompute all residuals from full NPZs,
bind the same completed service invocation and confirm no Duck process remains.
Retain ignored assets on native and Mac, with source/tests/results in the fork.
No training, videos, perception or hardware motion follow from this collection.

Independent pre-execution review cleared this narrowed protocol after requiring
mandatory full-array reception, strict placement/counter fields, observed stop
timeout and closeout reserve. Authenticated NPZ bytes are decoded from the same
in-memory bytes that were hashed, not a reopened path. Focused tests also cover
foreign/service drift, admission refusal, child failure and owned-only cleanup.
