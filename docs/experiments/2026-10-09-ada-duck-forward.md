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

## Retained execution and reception

Execution source `d04747c7f6a3c9029e3276f6eeee1c9ec3c5a20f` passed **192
cases across five focused CPU test files on each host**, zero failures, errors
or skips. Mac pytest time was 25.96 seconds; Linux 41.23 seconds, in a capped
CUDA-hidden service. Actual testcase multisets and the complete XML bytes match
the independently retained closeout bindings. These are CPU checks, not CUDA
training acceptance.

The single capped GPU collection completed in **37.40 seconds**, including fresh
MuJoCo-Warp compilation. Two worlds and all 20 fields were finite. All seven
state/control/applied-force fields stayed byte-identical on each backend, and
their CPU/GPU comparison residuals were zero. Both CPU and GPU counters confirmed
the predeclared empty-contact, empty-friction, empty-constraint baseline.
The actual compiled native selected-fields SHA remained
`6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f`.
The stock forward compiled solver modules as well; empty active constraints do
not imply no solver code was compiled or touched, and do not qualify its behavior.

The 72 sampled telemetry rows showed peak aggregate usage 1262 MiB, minimum free
14932 MiB and peak temperature 51 C. This is sampled usage, not a true peak or
exclusive reservation. Grounding DINO PID 1592 remained present; mission services
remained inactive. GPU memory returned to 961 MiB used / 15232 MiB free.

| Field | Maximum absolute CPU-double/GPU-float difference | Float32 bit mismatches, both worlds |
| --- | ---: | ---: |
| `xpos` | `1.1213985913471891e-08` m | 40 / 96 |
| `subtree_com` | `1.7975138266734803e-08` m | 78 / 96 |
| `ximat` | `1.7022679843492483e-07` dimensionless | 224 / 288 |
| `qfrc_bias` | `4.756721674326059e-07`, mixed generalized units | 24 / 40 |
| `qacc`, `qacc_smooth` | `2.179735020035878e-06`, mixed acceleration units | 40 / 40 each |

These differences are descriptive, **not a bit-exact pass or a relaxed physics
tolerance**. The retained complete arrays also include every other declared
field and residual. No inference about constraint solving, long rollouts,
locomotion, hopping or hardware follows from this fixed unconstrained state.

GPU invocation `f3061a4c1ac74be3bdb87f47723ea2b0` retired with MainPID 0,
exit 0, Result success, no restarts, empty ControlGroup and absent original cgroup.
The separate 90-second-capped CPU receiver invocation
`951ee2938d5641ee81885dcba9b3ff8e` independently reloaded and recomputed both
full NPZs, matched a fresh same-host plant, matched both 192-case XMLs, and retained
its own observed resource caps. It also retired successfully with MainPID 0 and
empty ControlGroup. A second Mac reception reproduced every full-file hash and
every residual row; it deliberately did not assert Mac/native plant bit equality.

The six core GPU files total **89377 bytes** and are retained both natively under
`artifacts/evaluations/ada-duck-forward-d04747c7f6a3/` and on Mac under
`artifacts/tools/ada-duck-forward/ada-duck-forward-d04747c7f6a3/`. Both test XMLs,
receiver source and closeout are retained under `artifacts/tools/ada-duck-forward/`
on both hosts. Private compiled caches remain on native only, avoiding duplicate
cache storage. These ignored assets are not added to Git history.
Independent Luna reception also reproduced all payload, report and XML bindings;
the owner reviewed that result before accepting retention, not simulator admission.

| Evidence | SHA256 |
| --- | --- |
| GPU report | `5cefad1522a59495483cf0af9139a30200e0e21e3eb8e1a66ac0617e1e729c06` |
| Child receipt | `da1d30a1ac2ebc312ad458ffea9431f0795b5d57e9270a80729656b2977cfcfc` |
| Native-double full fields | `d29ad756325b9f5cd67443e235c50bf3bbf997bf7dbb8ef00bf8a06ab3b23c12` |
| Ada-float32 full fields | `d5fa5dfea7c33d517bd4acb55e1b8966b0789525be448e288a1f69935e0d1db4` |
| Independent native closeout | `e99efd21d877417a12dc987283b64dca447a6faa399fe2b2cd5ad979b3e316e4` |
| Mac 192-case XML | `ce1cc75d8a1f2daae078cbc8163a665bd4bb513ac1510f477b4f736b8e707407` |
| Linux 192-case XML | `18d5ecd6512bb88419ee9e8f605f00363299ea8c419e804bbd096a6295e07fc1` |
| Receiver source | `bc9b1db3c5a66cd64a99048538dd263f60ad73b76790f2a948d73217fd73b127` |

## Next numerical gate

This baseline establishes retained execution of a real model's unconstrained
forward on Ada, not simulator qualification. Next predeclare a similarly bounded
forward-only **motor-friction/contact** fixture, with genuine zero-error BAM
preparation, complete input binding and active-constraint/contact correspondence.
Keep integration and policy work separate. Do not reuse the old Blackwell origin
collector or change its guards to treat this Ada result as solver admission.
The old full-rollout numerical failure is still unresolved; no PPO, video, new
skill promotion or physical motion is authorized by this collection.
