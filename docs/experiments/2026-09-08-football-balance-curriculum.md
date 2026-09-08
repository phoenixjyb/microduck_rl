# Football balance: a separate simulation curriculum

User requirement, September8: stand on a rolling football and stabilize while
preserving walking, obstacle avoidance, stabilization and eventual hopping.
The renewed work window ends **September9 07:30 Asia/Shanghai**, or September8
23:30 UTC. This authorizes gated simulation/development, not physical motion or
unconditional GPU use. Keep100.98 and unrelated services/workloads untouched.

## Plant and feasibility first

Interpret football as a soccer ball, initially a nominal0.11 m radius /0.43 kg
shell. IFAB's [Law2](https://www.theifab.com/laws/latest/the-ball/) specifies
68–70 cm circumference and410–450 g mass; the proposed numbers lie within those
reference ranges. They do not calibrate stiffness, pressure, damping, rolling
resistance, friction or the real ball's inertia. A rigid sphere with soft solver
contacts cannot establish robustness to the deformation of an inflated ball.

The existing `robot/microduck/ball.xml` is a70 mm diameter,15 g kicking ball.
Roller-foot tasks likewise are not policies that stand on a separate rolling
ball. Preserve both unchanged. The new
[B0 prototype](../../src/mjlab_microduck/football_balance_plan.py) builds an
explicit fixed/free ball-only MuJoCo world with thin-shell inertia2mr²/3 and no
robot or actuator. Its [CPU physics tests](../../tests/test_football_balance_plan.py)
check compiled topology/inertia, passive rest/rolling and invalid parameters.
Zero torsional/rolling friction in this first prototype is an explicit idealized
assumption, not calibrated football dynamics or a training environment.

Geometry matters: the kick task documents nominal foot centers near y=+/-42 mm.
On a110 mm sphere that corresponds to a local normal tilt of about22.4 degrees.
This is a geometric estimate, not a solved robot stance. B0 must still inspect
the selected robot's actual full foot pads, ankle/hip reach, joint limits,
contact normals, support forces, reset penetration and motor loading. Keep the
same declared robot mechanics throughout each experiment. A larger support ball
may be a separately named scaffold; shrinking it later is a new difficulty axis.

## Progression

| Stage | Lesson | Evidence before proceeding |
| --- | --- | --- |
| B0 | Ball dynamics and full robot/foot fit | Explicit ball inertia/friction/compliance, feasible nonpenetrating stance, contact instrumentation; ball-only tests are insufficient |
| B1 | Flat stance and small disturbance recovery | Measured low motion, bounded settling, motor/action-rate limits; zero command alone is not standing |
| B2 | Start placed on a fixed ball | Stable support from feet only, no floor/body contact, foot slip and COM drift bounded; record fixture assistance |
| B3 | Start placed on a prescribed slowly rolling ball | Relative balance across direction/speed bins; record all prescribed motion/work, never call this free-ball success |
| B4 | Start placed on a freely moving/rotating ball | Unassisted coupled robot/ball dynamics, no hidden weld/tether/mocap drive, finite bounded balance and drift |
| B5 | Free-ball disturbances and bounded steering | Change one axis at a time: pushes, direction, speed, then plant uncertainty; retain earlier performance |
| B6 | Mount/dismount and integrated retention | Separate approach/mount/balance/dismount phases, contact-safe transitions, retained walking/avoidance/stance/hop tests |

Being reset on top is allowed during B2–B5 and must be labeled; it is not mounting.
Do not reward unrestricted ball speed or indefinite survival after falling to
the floor. The main task is balance relative to the moving ball. Ball route/speed
control is a later, explicitly commanded objective, not a hidden extra demand.

Use a separate balancing policy first. Its structured observations may include
ball center/velocity/angular velocity relative to the robot, foot contact and
slip, support-relative torso orientation/velocity and valid/fresh metadata,
alongside proprioception. Exact simulation state is privileged sensing, not
validated hardware perception. Do not silently expand the trained61D gait actor
or splice its normalizer into a new observation layout. Raw cameras stay outside
locomotion RL. Delay/noise/dropout belong to separate later sensor lessons.

Each training stage requires its own predeclared parent, restore semantics,
plant/observation/reward protocol, seeds, training budget, common checkpoints,
held-out matrix and numeric thresholds before GPU use. Log first falls, loss of
support, forbidden body/ground contacts, slip, relative drift/tilt, ball motion,
settling time, torque/speed limits, power/load proxies and every assistance force.
No stage is accepted by reward or video alone. Preserve motor gates rather than
loosening them to make the stunt look successful. Long-duration robustness and
calibrated servo thermal safety are different claims.

## Retention and overnight priority

First finish the [frozen foundation speed map](2026-09-08-foundation-rethink.md)
with tested remote CPU/runtime/source and service-timeout gates. Use its evidence
to choose the smallest next gait/stance training lesson. No selected moving
speed band or standing controller is currently admitted by that unfinished map.
Do not spend the night launching unrelated reward changes just to occupy the GPU.

Ball work may proceed independently through B0 analysis and tested simulator
construction; ball-policy promotion still needs B1 and then sequential B2–B6.
Keep walking/route, obstacle approach/avoidance/recovery, stance and hop/landing
ledgers separate. H1-T remains rejected; there is no accepted hop to claim retained.
Rigid-foot and sprung-foot policies require mechanical alignment before routing.
The historical four-skill retention matrix does not automatically cover this
fifth skill; B6 needs a reviewed expanded transition and interruption matrix.

Only launch jobs that finish with closeout before the cutoff. Use one GPU job
at a time, strict source/checkpoint/runtime pins, allowlisted environments,
durable checkpoints/reports and independent service timeouts. At07:30 start no
new work, close owned jobs safely, leave protected services unchanged, retain
the final report and remove the continuation automation. The schedule is a work
window, not a promise that football balancing will be learned overnight.

Initial validation:11 new B0 tests passed on native CPU MuJoCo (ball only).
The combined football/foundation-map/launch-guard regression set passed165
tests in16.41 s with CUDA hidden. The curriculum was rendered to HTML and its
table/local links checked programmatically; no rendered robot or policy rollout
is claimed. These tests do not solve the remaining full-robot B0 feasibility gate.
