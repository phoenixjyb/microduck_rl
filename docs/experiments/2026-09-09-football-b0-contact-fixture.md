# Football B0: rigid full-collision contact fixture

Variant `football-b0-rigid-full-collision-geometry-v1` is a separate CPU geometry
fixture, **not a balancing policy or accepted stance**. It uses the existing
`robot_allcollisions.xml`, nominal HOME_FRAME joint positions, and the declared
110 mm radius /430 g thin-shell ball. Existing walking, spring, kicking and
roller-foot tasks are unchanged. XML actuator definitions are present but never
stepped; their equivalence to the training BAM model is not asserted.

Implementation: [football_contact_fixture.py](../../src/mjlab_microduck/football_contact_fixture.py).
[Native CPU tests](../../tests/test_football_contact_fixture.py) cover fixed/free
topology, inertia, preservation of source/pose/control state, joint-range checks,
real mesh–sphere distances and normals, forbidden-contact reporting and failure
to fabricate a nominal fit on a20 mm-radius ball.

## Construction and measurement protocol

The free variant has two free joints (robot and ball), no equality constraints,
no mocap driver and no hidden tether. The fixed variant has only the robot free
joint and explicitly reports fixed-ball assistance. A1 mm ball contact margin
exposes near-contact candidates; candidate contacts are not supporting forces.

Keep the nominal joint pose and horizontal root coordinates unchanged. Search
only root height, with40 bisection steps on the upper hemisphere, until the
nearest foot is0.1 mm from the sphere. This is neither inverse kinematics nor
static equilibrium optimization. It cannot establish reachability from another
pose or mounting capability.

Use `mj_kinematics`, `mj_comPos`, `mj_collision` and `mj_geomDistance`, never
`mj_step` or the force solver. Report actual compiled collision mesh distances,
nearest points, sphere normals, selected robot/ball and robot/floor clearances,
collision candidates, robot COM and every hinge's range. Only foot–ball and
ball–floor pairs are allowed in the candidate contact list. Engine collision
filters still apply to self contacts; this is not exhaustive unfiltered
mesh-intersection proof. Visual meshes are not added as collision proxies.

The report retains the robot XML hash, but does not claim every mesh/asset byte
is pinned. A future dynamics/training experiment must additionally bind those
assets, motor implementation and its full numerical protocol.

## Initial native CPU observation

Both fixed and free variants produce a nominal near-contact candidate, with all
14 hinges in range and no forbidden candidate contacts. The closest foot gaps
are approximately0.1000/0.1008 mm. Sphere contact points are near y=+26.42/-26.47 mm,
with normals tilted about13.9 degrees. The minimum selected forbidden external
clearance is about18.98 mm. These differ from the earlier foot-center estimate,
which was not a contact solution.

The rigid robot mass is0.73724 kg; its nominal placed COM is approximately
(0.00056,-0.000014,0.35552) m, while the ball center is(0,0,0.11) m. Its small
forward COM offset and two-foot support geometry still require a load/equilibrium
check; they do not prove static balance on point-like contacts.

No time was integrated, actuator force computed, robot weight supported or ball
motion demonstrated. B0 load/contact dynamics and B1 flat stance remain open.
Next: explicitly test a load-bearing contact/force model and joint torque demand,
then define the smallest stance lesson. Do not start B2/B3/free-ball training
from a geometric near-contact flag.

Local validation:8 new native fixture tests and the complete focused regression
set passed375 tests in18.72 s with CUDA hidden. Changed Markdown documents were
rendered to HTML and their tables/local links checked programmatically. No
screenshot, force-bearing simulation, GPU evaluation or training was performed.

The fixed/free geometry report from100.100 at source `980661c` is retained as
`artifacts/evaluations/football-b0-contact-980661c.json` and mirrored locally under
`artifacts/diagnostics/`. Both copies have SHA256
`a0aa7d6c82154ba8424ad3e77540705eedca6c7ddd919b5f1ca6862b023c2150`.
The subsequent [point-wrench diagnostic](2026-09-09-football-b0-wrench-probe.md)
tests this frozen pose without claiming load-bearing or learned balance.
