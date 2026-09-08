# Football balance: a separate simulation curriculum

Latest foundation evidence: the [completed frozen map](2026-09-09-foundation-command-map-results.md)
does not admit a moving-speed band. Continue ball B0 geometry and distinct stance
work; do not promote a walking policy to ball balancing from that result.

The [B0 full-collision contact fixture](2026-09-09-football-b0-contact-fixture.md)
now has tested fixed/free geometry candidates. This is unstepped near-contact
evidence; stance, supporting loads and rolling dynamics are still unverified.

User requirement, September8: stand on a rolling football and stabilize while
preserving walking, obstacle avoidance, stabilization and eventual hopping.
The September9 renewal extends the work window to **September10 07:30 Asia/Shanghai**, or September9
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

A September9 CPU-only geometry inspection compiled `scene_walk.xml`, reset its
STAND keyframe and called `mj_forward` without stepping. Compiled rigid-walk sole
mesh world bounds span approximately54 mm forward by41.2 mm lateral per foot;
foot sites are at y=+/-41.82 mm. The pad inner edges are near+/-21.22 mm and outer
edges near+/-62.43 mm. Thus foot-center normals are not actual sphere-contact
normals. This nominal geometry (robot mass0.73724 kg) neither solves ball stance
nor transfers the rigid-walk controller to a full-collision or sprung plant.

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

The [frozen foundation speed map](2026-09-09-foundation-command-map-results.md)
is complete and must not be repeated. Diagnose its retained traces and use that
evidence to choose the smallest next gait/stance training lesson. No moving
speed band or standing controller is admitted by the completed map.
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

## Historical handoff at 58b1f29 (superseded by the completed map above)

At source `58b1f293c7df7520d899ba09e4c772f69ae7dd49`,100.100 ran the football,
foundation-map and historical regression set in the allowlisted child environment
with CUDA hidden: **337 passed,2 skipped in18.75 s**. The two optional real-model
tests expect Mac mirror paths. Both actors were separately strictly restored on
CPU from their actual Linux paths, preserving their declared hashes and saved
iterations/common-step counters; CUDA remained uninitialized.

The host's initial check showed no running Duck service or compute PID, GPU0 at
0% utilization/45 C/12 MiB, and both protected services inactive. Historical
failed/exited services were left unchanged. Recheck all live state before GPU use.
The fork branch and training worktree were fast-forwarded to the same exact
tested source. HTTPS publishing failed authentication; the existing SSH transport
worked without changing stored credentials or Git remote configuration.

Retained on100.100 under
`artifacts/evaluations/foundation-command-map-cpu-preflight-58b1f29`, mirrored on
the Mac under `artifacts/diagnostics/foundation-command-map-cpu-preflight-58b1f29`:

- `actor-loading.json`: SHA256 `4c14beee2ee543f77a8505c680b936539b889fe9ecb1c882713171b1dbf38d75`.
- `cpu-tests.json`: SHA256 `3ecd071dc76c7baf48f73c3b902b79463e418b25baf0722a786d72d3151c69e1`.
- Original checkpoint: `logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt`.
- Narrow checkpoint: `artifacts/experiments/f1r-width-paired-s421-v1/narrow-pilot/model_8498.pt`.

That historical handoff preceded the completed map; do not repeat it. The existing
same-thread continuation retains ID `microduck-curriculum-through-sep-9-07-30`,
but was updated on September9 to the September10 cutoff above at15-minute cadence.
The CPU motion decomposition and B0 geometry fixture are complete. The
[relaxed contact-wrench check](2026-09-09-football-b0-wrench-probe.md) rejects
the nominal frozen pose. The subsequent [bounded adjusted stance](2026-09-09-football-b0-adjusted-stance.md)
has local idealized contact-force feasibility. A separately declared
[0.5-cap allocation](2026-09-09-football-b0-margin-allocation.md) adds modeled
friction margin at the same saved pose, but not compliant contact stability or
motor acceptance. Next bind the motor/contact dynamics and the smallest flat-stance
lesson. No football optimizer or balanced rolling policy is running at this update.

The next [CPU BAM component binding](2026-09-09-football-bam-component-binding.md)
checks the actual training computation and motor-mode spec before integrated
contact tests. Native and training BAM adapters are not interchangeable without
checking constraint indexing and m6 friction semantics. This component probe
bypasses command delay, writes no native controls and is not a B1 standing result.
