# Flat-floor BAM hold diagnostic (not learned B1)

Predeclared `football-flat-hold-v1`: one native CPU rollout, at most 500 steps
of 0.002 s (1 s). No optimizer, policy, GPU, ball, disturbance or physical motion.
Use the B0 full-collision rigid robot and actual BAM training computation from
the component binding, nominal 7.5 V / 0.1 V/Nm sag / unit scales / firmware kp200.
Do not modify the installed BAM or historical tasks. Add only a plane with
sliding friction 0.6, no torsional/rolling friction, condim3; retain XML contact
settings and effective per-contact parameters. Do not add tethers or support.

Keep HOME_FRAME hinge angles and identity root orientation; place vertically
by 40-step bisection in root-z [0.04, 0.30] m so nearest foot is 0.1 mm above
the plane. Require both foot gaps in [0,0.3] mm and no forbidden penetrating
contact. Hold the initial hinge targets using a fixed three-physics-step FIFO
delay (6 ms), primed with those targets. This is an explicit native diagnostic
delay, not validation of mjlab's randomized delay implementation.

At each boundary call mj_forward, snapshot the resulting native loads, evaluate
BAM, and check the proposed torque before applying it to motor-mode controls.
Copy its friction/damping budgets to controlled DOFs and call mj_step. Refresh
with mj_forward at the next boundary, recording its forces with that boundary's
state and previous controls. This extra forward solve is explicit; GPU numerical
or solver equivalence is not claimed.

Abort before another step for any nonfinite state/output, MuJoCo warning,
forbidden penetrating contact, root height below 0.08 m, torso tilt above
0.35 rad, root linear speed above 1 m/s, hinge speed above 10 rad/s, proposed
or applied motor torque above 0.36 Nm, or any joint outside its hard limits.
After 0.1 s, require both feet to carry positive normal force (>0.01 N).
These conservative diagnostic stops are not calibrated hardware ratings or a
relaxation of historical motor gates. Inspect the final boundary too.

Retain every boundary's state, support forces/contact parameters, joint speeds,
applied torques, soft-range exposure (central90%), absolute mechanical power
and torque-squared integral (load proxy, not temperature); retain proposed
torques and friction/damping before each applied step. No auto-reset or rerun
after an abort. Synthetic tests may inject failures but are not rollout evidence.

Completion means only that this fixture stayed within these stops for 1 s.
It does not accept B1, demonstrate perturbation recovery, establish long-duration
thermal safety or support football balancing. Use the first outcome to decide
the smallest follow-up; a B1 optimizer still needs a full lesson declaration,
Linux initialization tests and a verified exclusive GPU window.

## Retained first outcome: diagnostic abort, not B1 acceptance

The code/protocol was committed at `a37d00c` before the one native rollout.
All 424 targeted CPU regression tests passed in 18.51 s, including 15 new hold
tests. Tests exercise synthetic early/final stops without running the full
native hold. The original suspended component fixture remains the default;
the new explicit `flat_floor=True` option adds the plane only for this protocol.
Its earlier report remains unchanged and pinned to its historical source.

The exclusive local report is
`artifacts/diagnostics/football-flat-hold-local-a37d00c.json`, SHA256
`6bd04008ac8161f55f7a3c77c45b10ccee58c37cba6d4e8de5c5107ca2ea768f`.
It retains 475 boundaries and 474 applied commands. The first trial aborted at
474 physics steps / 0.948 s for **tilt**, with final tilt 0.351286 rad against
the unchanged 0.35 rad stop. No rerun, automatic reset or limit adjustment.

Both feet still carried support (3.61856 / 3.61752 N); root height was 0.111678 m.
There were no MuJoCo solver warnings. Maximum applied hinge torque was
0.0944433 Nm, maximum hinge speed 0.382216 rad/s, and soft-range exposure stayed
zero. Minimum modeled voltage was 7.45447 V. Maximum absolute mechanical power
was 0.0650463 W. The left-rectangle boundary torque-squared load proxy was
0.0106164 Nm²s; this is not a servo temperature or continuous-time torque integral.
The previously documented import-registration warning appeared again; normal
Linux registry and runtime initialization still require separate verification.

## Read-only trace diagnosis and next lesson

Replaying retained qpos through kinematics/COM calculation only (no integration)
shows predominantly forward pitch: approximately 0.04868 rad at 0.2 s,
0.17792 at 0.6 s and 0.35129 at the final boundary. Roll stayed small. Robot COM
x moved from 0.00056 m to 0.03426 m; final loaded contact x positions were about
0.02587–0.02594 m. The COM projection was thus approximately 8.3 mm ahead of
those loaded contact points. This is consistent with forward tipping, not a
complete dynamic stability certificate or proof of a unique root cause.

Despite fixed HOME targets, the knees drifted about +/-0.159 rad and ankles
about -/+0.098 rad; the firmware position loop alone did not maintain torso
orientation under this plant. The evidence does not identify a torque-ceiling
failure and provides no justification for raising the torque/tilt stops.

Next design B1 as actual closed-loop stance stabilization, with explicit torso
orientation/angular-velocity and joint feedback, bounded target corrections,
and nominal balance before pushes. Predeclare observations, policy/normalizer
initialization, rewards, seeds, budget, held-out cases and retention thresholds
before an optimizer. Do not train the football specialist yet or claim an
existing gait/hop checkpoint meets this new stance requirement.

100.100's read-only SSH check timed out. No GPU job, protected service, network
setting, remote worktree or 100.98 workload was changed. Reconcile remote
evidence and validate the exact Linux runtime when connectivity returns.
