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
