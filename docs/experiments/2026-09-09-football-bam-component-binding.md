# Football motor binding: CPU component prerequisite

Predeclared protocol: `football-bam-cpu-component-v1`. This is not an optimizer
run, a standing controller, a ball rollout or a replacement for historical gates.

The installed native `bam.mujoco.MujocoController` subtracts joint-friction
constraint rows using joint IDs. A floating-base MuJoCo fixture instead exposes
DOF indices in those rows: hinge joint ID 1 has DOF address 6. The installed
training adapter `bam.mjlab.BamActuator` uses DOF indices correctly. Also, native
`Model.compute_frictions` and the Torch m6 friction budget differ in their
quadratic sign/equal-magnitude masks. Neither dependency is edited here; do not
silently substitute native formulas or infer that existing GPU training is wrong.

## Bounded implementation and tests

Build a fresh full-collision robot from the existing XML, without floor or ball,
at HOME_FRAME and zero velocity. This suspended, unstepped component fixture is
deliberately not a load-bearing stance. Apply the actual repository
`FrictionDRBamActuatorCfg` and BAM `edit_spec` before compilation: motor-mode
actuators, unit gear, declared armature/force ceiling and friction constraints.
Retain all asset and relevant runtime source/parameter hashes.

Bind one native MuJoCo snapshot to owned float32 CPU tensors consumed by the
actual training adapter's `compute` method. Select nominal voltage 7.5 V, drop
gain 0.1 V/Nm and unit gain/friction scales within existing startup ranges.
No Warp/Entity initialization or GPU backend equivalence is claimed. This probe
bypasses the command-delay path and applies no computed torque to MuJoCo.
Full runtime initialization, delay semantics and integrated contact dynamics
remain explicit follow-up gates.

Evaluate only three independent position-target offsets: 0, +0.02 and -0.02 rad
on all 14 controlled hinges. Each sample begins with zero previous motor torque;
native gravity/bias and constraint fields come from `mj_forward` at the same
saved state. Record computed torque, friction budget, damping and voltage.
No `mj_step`, inverse-dynamics solve, policy, synthetic support or pose search.

Focused regressions must cover DOF-versus-joint constraint indexing, exclusion
of non-friction rows, finite/shape rejection before compute, fresh owned buffers,
real motor spec conversion, positive finite friction, unchanged native state,
lagged voltage sag/reset, and reuse of the installed training compute path.
Synthetic component inputs are test fixtures, not robot performance evidence.

Do not declare B1 accepted or launch its optimizer from these component tests.
Next predeclare a bounded native contact/hold diagnostic with explicit timing,
delay, stop conditions and metrics; reconcile Linux state and tests before GPU
work. The existing curriculum cutoff remains September 10 at 07:30 Shanghai.

## Local validation

All 11 new component tests passed within a 409-test CPU regression run in
17.51 seconds, with CUDA hidden and an allowlisted environment. This includes
the retained B0-A input test; no optional skip was counted as a pass. Initial
test-only failures were diagnosed as the distribution name being
`better-actuator-models` (not its import name `bam`) and a mixed float32/float64
synthetic comparison. Metadata lookup and the test input dtype were corrected;
neither BAM implementation nor any acceptance threshold was changed.

The native friction-index fixture observes hinge joint ID 1, DOF address 6 and
friction row ID 6. The training adapter attributes that row to DOF 6 and ignores
non-friction rows. Separate component tests retain the m6 formula distinction
for same-sign loads and equal-magnitude opposed loads. These are compatibility
findings, not a new motor calibration or proof of historical training failure.

The latest bounded SSH attempt returned `No route to host` for 100.100. Current
remote GPU/service state remains unknown; no remote process, service, network
setting or training worktree was changed. The interrupted remote B0 wrench
report still needs reconciliation on reconnect before any rerun or overwrite.
