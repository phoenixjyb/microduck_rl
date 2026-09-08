# B0 frozen-pose point-wrench diagnostic

Protocol `football-b0-relaxed-point-wrench-v1` tests a necessary condition before
load-bearing simulation, not a learned policy or a motor-demand calculation.
The [geometry fixture](2026-09-09-football-b0-contact-fixture.md) remains unchanged.

## Model and test

Use actual MuJoCo contact candidate positions from the nominal full-collision
fixture, two foot contacts, and (for the free ball only) one floor contact.
Fixed-ball/world-floor collision is absent and no floor force is fabricated.
Require condim 3 and distances inside the force-generation margin. Retain the
effective contact friction and margins, rather than inferring them from the ball
geom: foot contacts have sliding coefficients 1.0, while ball-floor has 0.6.
Nonzero unused friction entries do not add contact couples at condim 3.
See [MuJoCo's contact model](https://mujoco.readthedocs.io/en/latest/computation/#contact).

Solve robot force and moment balance about its COM; for a free ball, also solve
ball force and moment balance about its center, with equal/opposite foot loads
and a floor reaction. Gravity is 9.81 m/s². Internal joint torques cannot supply
a missing net external wrench. Treat every contact force as unbounded and signed:
unilateral, friction and motor constraints are deliberately relaxed. No contact
couples, hidden support wrench, integration or MuJoCo force solve are introduced.

The linear least-squares objective scales moment rows by a fixed 0.11 m length.
Report residual forces in N and moments in Nm, matrix rank and singular values.
Numerical consistency requires maximum scaled residual <= 1e-6 N (rcond 1e-12).
This is a new algebraic diagnostic tolerance, not a replacement locomotion gate.
Failure rejects equilibrium of this exact frozen contact/COM geometry even under
relaxed forces. Passing would still require friction, joint torque, actuator and
contact-compliance checks. Least-squares forces with residuals are not supporting
loads and must not be converted into an accepted motor-demand claim.

## Initial CPU result

The nominal pose is inconsistent in both cases: maximum scaled residuals are
approximately 0.017441 N (fixed) and 0.022971 N (free), above 1e-6 N. Robot pitch
moment residuals are about 0.001522 Nm and 0.002527 Nm respectively. This is
consistent with the previously noted COM/support-line offset, not proof that
football balance is impossible. The next pose may shift its COM or contacts;
rolling balance is dynamic and need not maintain instantaneous static equilibrium.

Next: a separately declared small pose/contact search, followed by constrained
force feasibility and motor-demand instrumentation on a pinned motor plant.
Do not conceal this result by adding fictitious contact moments or changing
friction, reward or acceptance thresholds. Flat stance B1 remains a separate
prerequisite; free-ball training has not started.

Implementation: [football_wrench_probe.py](../../src/mjlab_microduck/football_wrench_probe.py).
Tests include analytic symmetric fixed/free solutions, equal/opposite forces,
moment signs, translation invariance, COM-offset rejection, rank deficiency,
invalid input and native no-integration checks. The combined football subset
passed 29 tests locally with CUDA hidden. No new GPU training or policy
capability is claimed.

The broader CPU regression set passed385 tests in18.45 s; after tightening the
exact two-distinct-feet topology check, all29 football tests passed again in5.23 s.
The three changed experiment documents were rendered to HTML in memory and local
links checked. This is a document structure check, not a visual robot recording.

## Renewed work window

On September9 the user renewed simulation/curriculum work through **September10
07:30 Asia/Shanghai (September9 23:30 UTC)**. The existing 15-minute continuation
was updated rather than duplicated. It preserves the GPU lease, independent job
timeouts, checkpoint retention and protected-service boundaries. The previous
September9 cutoff is superseded; no protected service restoration is authorized.
