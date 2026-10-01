# C1-S native CPU runner implementation

October 1, 2026, starting from `04a98b7abfc02fe62c1b0422e963543525fc1f43`
on `feat/athletics-obstacle-curriculum`. This chunk implements the prerequisites
of the [fixed C1-S declaration](2026-10-01-community-hop-c1s-predeclaration.md).
It does **not** execute either retained community policy, consume either public
diagnostic attempt, train, render video, change a remote service or authorize
physical motion. Separate execution approval remains required.

The declaration bytes and frozen [C1 measurement scorer](../../src/mjlab_microduck/community_hop_rehearsal.py)
are unchanged. The declaration's historical `predeclared-runner-not-implemented`
status is not rewritten to fabricate a result or execution approval. This
implementation record is the new readiness evidence, not a new experiment.

## Implemented contracts

- [Native plant and preflight](../../src/mjlab_microduck/community_hop_surrogate.py):
  hash-bound XML, tracked asset bytes, installed BAM Git revision/parameters,
  frozen package versions and policy metadata. The exact verified ONNX buffers
  are retained for any later approved session; no mutable path is reopened for
  inference. XML includes and meshes likewise compile from verified buffers.
- The stock scene's fourteen position actuators become unclipped, unit-gain
  torque motors. BAM XL330 M6 uses 7.4 V, gain 200 and its 1.75 A PWM-current
  limiter. Firmware position feedback is servo plus passive backlash; motor
  velocity alone supplies back-EMF/friction. Unreachable high-speed current
  limits are measured as violations, not concealed by a hard torque clip.
- All stock collision masks/contact parameters, equality constraints and
  passive hinge properties are preserved. Only motor DOFs receive BAM armature,
  damping and friction updates. The external-load estimate uses the preceding
  constraint solve, subtracting each motor DOF's own dry friction; its **current**
  torque/velocity supply the BAM budget. Constraints are refreshed after those
  edits so the new friction budget enters the same physics substep.
- Actual sole clearance is the minimum world height of compiled mesh vertices,
  not a foot/body origin. Compiled mesh transformations are not applied twice.
  Both sole contact forces and every nonsole floor contact are collected each
  5 ms substep. World course velocity refers to the free-joint origin; policy
  angular velocity/gravity use the base frame.
- [Walker-only adapter](../../src/mjlab_microduck/community_hop_baseline.py): the
  nine shared frozen C1 gates plus full 8–9 s stability and zero qualified
  airborne episodes. It does not require a hop or apply C1's overall no-hop
  rejection to the control case.
- [Sequential runner](../../src/mjlab_microduck/community_hop_diagnostic.py):
  600 control ticks, 2,400 physics advances and an inclusive terminal state;
  carried raw-action, 20 ms delayed-action and one-tick encoder-velocity
  histories at every switch. No filtering, clipping, retry or switch-time reset.
  An unsafe state or unstable entry is detected before invoking the next policy.
- An owning parent bounds native imports/setup and each attempt in a separate
  child process group: setup ≤60 wall seconds, each case ≤60, campaign ≤180.
  It terminates only its owned child on timeout/interruption. Each complete
  journal record is flushed/fsynced; a partially written line cannot become a
  passing result. Nonfinite failures use lossless JSON tags, not numeric clamps.
  Load a tagged trace with `decode_numbers` before using the frozen scorer.
- Each case verifies its source/policy/compiled-plant binding against setup
  before creating sessions, and the parent checks it again. Malformed traces,
  empty/time-limited attempts and identity drift retain a rejected report and
  valid prefix. A failing control prevents the transfer case. Unique directories,
  exclusive writes and SHA256 manifests preserve earlier evidence.

MuJoCo frame/constraint details were checked against the pinned 3.10 primary
[mesh reference](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#asset-mesh),
[velocity/contact API](https://mujoco.readthedocs.io/en/3.10.0/APIreference/APIfunctions.html#mj-objectvelocity)
and [DOF-friction constraint source](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_core_constraint.c).
The installed BAM dependency was neither patched nor upgraded.

## Safe entry point and evidence

Default command (static preflight only):

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.community_hop_diagnostic
```

It stores a unique directory beneath
`artifacts/community/c1s-velstand-substitution-v1/`, including copied declaration,
effective compiled MJB, settings/policy/source receipt, child logs, decision and
hash manifest. The receipt explicitly records zero sessions/inferences/physics
steps. No remote GPU host is needed for this gate.

After a **separate run go-ahead**, the same entry point with
`--execute-approved` asserts operator approval for only the fixed, baseline-first
two-case CPU probe. The repository must be clean on the exact feature branch.
Private child flags are implementation details, not alternative launch commands.
Neither the declaration nor this document grants that go-ahead.

Focused checks cover baseline gates, actual compile/motor/contact/frame bindings,
high-speed limiter exposure, synthetic ONNX CPU execution, mocked closed-loop
switches, malformed prefixes, source drift, sequential gating and a deliberately
timed-out **fixture** child. The small native physics steps and synthetic constant
ONNX used by tests are not executions of the retained velstand or Happy Hop
weights and are not skill evidence.

Validation command:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_community_hop_predeclaration.py tests/test_community_hop_rehearsal.py \
  tests/test_community_hop_baseline.py tests/test_community_hop_surrogate.py \
  tests/test_community_hop_diagnostic.py tests/test_community_policy_inspection.py \
  tests/test_skill_compatibility.py tests/test_retained_skill_binding.py \
  tests/test_skill_retention_plan.py tests/test_hop_evaluation.py \
  tests/test_hop_revision_gate.py tests/test_hop_checkpoint_sweep.py
```

Results: **337 scoped tests passed**, including 59 new baseline/plant/runner
checks and the unchanged historical measurement/retention selections.
`git diff --check` passed; the new and linked Markdown rendered and all local
links resolved. A read-only Luna review found two evidence-handling gaps;
integration fixed malformed-prefix rejection and per-attempt binding checks,
then added regression tests and reran the selection. A fresh isolated import
of the watchdog modules loaded no MuJoCo, NumPy, ONNX/Runtime, Torch, Warp or BAM:
native imports stay inside the hard-bounded child. The default entry point was
also exercised against the actual retained artifacts in static mode only.

Full repository tests, public-policy rollouts, GPU qualification, training and
visual/physical acceptance are not part of this chunk. All author replication,
impact calibration, thermal verification, skill acceptance, multi-seed promotion,
training/deployment and physical-motion flags remain false. An eventual all-pass
C1-S result can only be a nominal substituted-entry diagnostic candidate; it
does not recover the [matched author's C1](2026-10-01-community-hop-c1-preparation.md)
or relax H1, obstacle, stance or football gates.
