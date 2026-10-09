# Saved prepared-pose solver-response follow-up

Base `6fe37f27cd4ea1009a78502a8f29b60cb4bdc7fe`, exact branch
`feat/athletics-obstacle-curriculum`. This separates the next response arm from
the completed [collision-only intervention](2026-10-10-ada-saved-pose-collision.md).
Current changes are a **read-only installed-source audit and tests**, not a
solver runner or an executed experiment. No library, plant, driver, motor,
acceptance gate, curriculum or training protocol changes.

## Source audit and dependency decision

`ada_solver_stage_audit.py` checks all69 installed MuJoCo-Warp Python files
and three Warp loader/config files against wheel RECORD. Six relevant files
also match explicitly frozen SHA256s. AST receipts preserve exact lexical
call order, arguments, line references and conservative model/data attribute
references. Conditional and unreachable calls remain labeled **lexical**, not
claimed executed. This inventory is not a transitive kernel read/write proof.
The audit imports no NumPy, Torch, Warp or MuJoCo runtime and executes no physics.

The frozen primary source establishes:

- `forward.py:596-619`: `fwd_position` starts with kinematics, then com_pos,
  camlight, flex, tendon, crb, tendon_armature; conditional factor_m and collision;
  constraint construction; disabled island path; transmission. Thus public
  fwd_position/forward cannot preserve a supplied geometry intervention.
- `forward.py:1230-1257`: ordinary forward passes factorize=False to position,
  then runs velocity, optional control callback, actuation, acceleration with
  factorize=True, solver and sensors. Position and acceleration factorization
  choices must not be silently changed.
- `smooth.py:602-632`: com_pos consumes supplied body/inertial/joint poses and
  writes subtree_com, cinert and cdof, not those supplied poses.
- `constraint.py:2503+`: construction resets counts and constructs friction,
  limits and contact rows. Its outputs include contact.efc_address; candidate
  addresses-1 are retained before construction, not required unchanged afterward.
- `solver.py:3341-3387`: `wp.copy(d.qacc, d.qacc_warmstart)` copies the **source
  warmstart into destination qacc**, because Warp `copy(dest, src)` is defined
  at `warp/_src/context.py:8708+`. When warmstart is disabled the source is
  qacc_smooth. Forward solve does not integrate or itself update qacc_warmstart;
  the `_advance` integration path does, and is excluded.

The source audit explicitly checks the two copy argument lists, not an assumed
arrow direction. The current retained plant has zero equalities, activations
and mocap worlds, seven sites, and14 joint transmissions. Site poses are not
among the earlier nine-field collision bundle. Do not generalize that bundle
to every dynamics dependency or initialize fixture sites from allocation poses.

## Proposed smallest bounded arm, not yet execution-ready

Use the existing authenticated predecessor source/report/files and exact
float32 seven-state and motor inputs. Fresh native plant, all347 model-array
bindings and unchanged dense default options must match the prior arm.
Fresh Warp CPU data allocation still counts its one internal native kinematics
call separately. No fixture native/Warp kinematics, integration, BAM recompute,
constraint/force injection, altered solver settings or model repair.

In addition to the nine saved collision poses, supply **site_xpos/site_xmat**
from the same authenticated prepared bank, with complete logical layout and
byte checks. This is an eleven-field prepared-pose dynamics intervention,
not a retroactive claim that the eleven measured Ada GPU-boundary arrays were
captured. Require rigid/no-SDF/no-contactfilter, all other callbacks absent,
no equalities/mocap/activations/tendons/body transmissions and the actual
joint-only transmission topology. Unsupported paths fail before physics.

Keep the frozen position-stage order, omitting only fixture kinematics and
the factor_m branch excluded by ordinary forward's factorize=False. Do not
move construction after velocity merely because a different ordering could
work for this zero-velocity fixture. Keep camlight/flex/tendon/tendon_armature
calls in order even when topology makes them no-ops. Recompute com_pos, crb,
transmission and normal velocity/passive/bias/actuation/acceleration outputs
from the supplied inputs; do not reuse preceding GPU-derived mass or forces.
Sensors/energy must either be executed in original order with supplied site
poses or be proven force-independent under explicit guards before exclusion.
The exact implementation and focused tests are a further review gate.

Retain complete actual model/data byte/layout inventories, including all
allocated Contact and Constraint fields, at these boundaries:

1. Before collision, after nonkinematic position prerequisites.
2. After collision, before make_constraint: all16 generated candidate fields,
   complete eleven actual poses, unchanged state and zero EFC/solver counters.
3. After make_constraint/transmission: actual EFC addresses/counts and rows.
4. Immediately before solve, after velocity/actuation/acceleration: full Data
   arrays, mass factors, qacc_smooth, qfrc_smooth, warmstart, every constraint
   input and all actual candidate fields, not a hand-selected subset alone.
5. After solve: full Data arrays, complete active row force/state, generalized
   qfrc_constraint and qacc, solver counters and complete generated contacts.

Synchronize and check all eleven pose bytes, seven state bytes, model bindings,
capacities/options and callback/static identities at each applicable boundary.
Preserve pre-constraint and post-constraint candidate tables separately.
Only EFC addresses may change in the contact struct after construction; report
and fail on unexpected changes in the other candidate fields. Reconstruct
J.T@force and ordered-pair world resultants over all active rows/contacts, using
existing arithmetic conventions with no rounding, spatial pairing, contact-count
normalization, imposed force, tolerance or acceptance promotion.

This produces a **new CPU-generated manifold followed by its CPU solver
response**. It is neither integrated state replay nor an Ada-versus-CPU
same-manifold solver control. The retained Ada post-solve table lacks
includemargin and complete measured GPU collision-boundary poses. Residuals
against historical Ada resultants can be descriptive only. A causal downstream
solver control needs branching from an identical complete pre-solver bank;
do not reconstruct a purported GPU input manifold from its solved outcomes.

## Execution gates retained

No response physics has run under this document. Before a runner may execute:
implement the closed CLI/receiver and complete boundary checks; test them on
Mac, independently review, commit/push the exact feature source, fast-forward
the clean native worktree and run its focused CPU checks. Predeclare its exact
source, absent output/cache paths, invocation and resource limits first.

Future native response cap: CPU-only user service, literal CUDA_VISIBLE_DEVICES
empty, one thread per pool, Type=exec/RemainAfterExit, RuntimeMaxSec180,
MemoryMax6G, CPUQuota200%, TasksMax64, Nice10, LimitFSIZE16M, LimitCORE0,
Restart=no, KillMode=control-group, TimeoutStopSec10. Fresh private CPU cache
and passive existing-executable provenance follow the preceding arm;
loaded_binary_bytes_bound remains false. Stop before any integrator/step.
No GPU Duck job, PPO, raw perception, policy promotion or physical motion.
Preserve Grounding DINO and inactive protected services. The07:00 Shanghai
deadline and ten-minute closeout reserve remain unchanged.

Source-audit CLI only: `python -m mjlab_microduck.ada_solver_stage_audit
--output <absent-absolute-json> --source <exact-clean-SHA>` with CUDA hidden.
It needs no runtime cache because no runtime is loaded. JSON source bindings
and both-host audit results will be retained separately from future physics.

Audit/test service cap60s, MemoryMax2G; CPUQuota200%, TasksMax64, Nice10,
LimitFSIZE16M, LimitCORE0, Restart=no, KillMode=control-group, TimeoutStopSec10,
Type=exec/RemainAfterExit and thread-pool1. These limits apply only to this
read-only source audit and its three-file contract suite, not response physics.
Output JSON is41838 bytes before exclusive serialization/source additions.

## Preparation checks

Mac focused audit11/11 and combined audit/saved-collision/metadata191/191
passed (combined1.93s). Import audit asserts none of NumPy/Torch/Warp/MuJoCo/
MuJoCo-Warp runtime modules loaded. Changed/missing source inventories, ambiguous
functions, lexical branch interpretation and the no-physics AST fence are tested.
Independent Luna review passed11/11 in1.28s and confirmed the source order,
deferred factorization, extra site/state guards and interpretation limits.
The review initially reversed Warp copy arguments; the owner checked the
installed primary API and the reviewer corrected this before integration.
The audit now records and checks the actual source/destination argument lists.
No runtime library or installed source was changed. Neither source review nor
these tests authorize solver execution or training.
