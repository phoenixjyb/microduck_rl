# B0-A: bounded pose adjustment and ideal static loads

Predeclared before executing the pose search. This is CPU-only fixture analysis,
not an optimizer training run, a learned controller, or a change to historical
gates. Preserve the nominal B0 geometry and rejected frozen-pose evidence.

## Fixed search scope

Use the free-ball, full-collision rigid fixture at source `24e061c`, nominal
110 mm /430 g shell, unchanged HOME_FRAME hinge positions, contacts and gravity.
Vary only root x/y/z by at most5 mm from the nominal placed pose and root roll
and pitch by at most0.05 rad, yaw fixed at zero. Rotation is R_y(pitch) R_x(roll).
No joint optimization, mesh edits, contact couples, assistance or motor changes.

Five residuals in meters: two foot gaps minus0.1 mm, robot COM x/y relative to
the ball center, and horizontal distance from that center to the line through
the two actual collision contact positions. Solve with bounded least squares,
max80 function evaluations (finite-difference evaluations counted separately),
ftol/xtol/gtol1e-11. Require all residuals <=1e-8 m, joint ranges valid, no
forbidden contacts and exactly one contact per foot plus one ball-floor contact.
Failure is retained; no wider search or parameter changes within this protocol.

## Contact-constrained static load check

At the adjusted pose, use actual contact positions, frames, effective friction
and condim3. Solve equal/opposite robot/ball force and moment balance under
gravity, with the four rays n +/- mu1*t1 and n +/- mu2*t2 at every contact and
nonnegative ray coefficients. Minimize the sum of coefficients. This is a
declared conservative pyramidal Coulomb force model, not the simulator's soft
contact constitutive solution. Moment equations are scaled by0.11 m. LP primal
and dual tolerances1e-9; independently require scaled residual <=1e-6 N and
nonnegative coefficients within1e-9 N. No tensile force or contact couple.

## Ideal joint demand, not motor acceptance

Only after feasible contact loads, compute generalized gravity as the sum of
body-COM translational Jacobians transposed times each body's weight, and
generalized contact loads from contact-point Jacobians. Required ideal hinge
torque is the negative sum. Check root force/moment residuals and independently
validate gravity by finite differences of gravitational potential energy.
Keep radians, N and Nm explicit. No passive friction, gearing, BAM voltage-loop
execution, dynamic inertia, speed or calibrated thermal limit is modeled here.
Do not normalize to a made-up continuous motor limit or claim BAM feasibility.

Pin the XML and every referenced local mesh byte plus implementation/runtime
versions in the retained report. No MuJoCo integration, forward contact force
solve, GPU process, policy inference or physical motion. Algebraic feasibility
does not prove compliant contact equilibrium, stability, settling, robustness,
mounting or rolling. Next is the explicit motor/contact dynamics binding and
flat stance B1 lesson; no B2/B3/free-ball promotion from this diagnostic alone.

## Initial local CPU result

The fixed bounded search converged in4 solver function evaluations. Root offset
was approximately(-0.47532,+0.01227,-0.00139) mm, roll-0.00358 degrees and
pitch-0.22515 degrees. All hinges stayed at HOME_FRAME, forbidden contacts were
absent and the five residuals were below1e-8 m. This is a placed pose, not mounting.

The declared unilateral friction-pyramid LP found a feasible allocation, with
scaled residual about1.83e-9 N. Foot vertical loads were approximately3.6185 and
3.6138 N; ball-floor load11.4507 N. The allocation also has opposing lateral
foot forces of about2.1784 N. Minimizing normal force selected a near-friction-edge
allocation, not a robust margin or minimum-motor-load solution. Retain the actual
pyramid utilization/slack per contact; do not interpret LP feasibility as safe
friction margin or unique physical contact loads.

For that allocation, peak ideal hinge torque was approximately0.21490 Nm at the
left hip roll (right hip roll about-0.21475 Nm). These are rigid static generalized
torques, not executed servo torque, continuous ratings or thermal acceptance.
Free-root residuals were about2.1e-10 or smaller in their respective N/Nm units.
Gravity Jacobians independently matched central differences of gravitational
potential across every velocity coordinate. The37-test football subset passed
locally with CUDA hidden; no MuJoCo forward force solver or integration ran.

The first two read-only SSH attempts this heartbeat timed out before remote
state could be established. Do not advance the training worktree or launch a
job until connectivity, current Git identity, active workloads and protected
service state are rechecked. The previous interrupted remote wrench report
remains unreconciled. Continue local tested development without network or
service changes; never claim the host is currently idle from an older snapshot.

## Validation and limitations

The393-test broad set returned392 passed /1 failed in20.94 s. The failure was
the unchanged `test_surviving_cpu_descendant_closes_the_cell_instead_of_advancing`:
Darwin `os.killpg` raised `PermissionError: [Errno 1] Operation not permitted`
during SIGKILL cleanup of owned group92316. Read-only process inspection found
no remaining group member. The isolated test then passed in5.16 s, and the
combined football/campaign set passed103 tests in12.37 s without code changes.
This characterizes an intermittent failure, not a repaired supervisor or a clean
broad-suite result. No cleanup guard was disabled or modified. Revalidate the
supervisor on the Linux training host before any GPU launch.

The three changed Markdown documents rendered to HTML and their local links
passed a structural check. There was no visual policy render or MP4. The new
eight stance tests cover analytic load solutions, unilateral refusal, bounded
pose changes, asset hashes, all-coordinate gravity differentiation, free-root
balance and explicit non-execution of dynamics. CPU source tests and ideal
static feasibility do not accept a learned controller or a physical motor load.

## Retained local report and next handoff

Source `e79aadf` was committed and pushed to the exact fork feature branch. The
allowlisted Mac CPU run retained
`artifacts/diagnostics/football-b0a-stance-local-e79aadf.json`, SHA256
`d15226a8eacb69a347f4ece725528ecb102e9fde05e69e9035cb861c6e5bebf6`.
It includes exact Git/probe hashes, NumPy/SciPy/MuJoCo/Python versions, all fixture
mesh/XML hashes, pose, contact forces and ideal joint torques. The two foot
pyramid utilizations are0.99811 and1.00000: the second has essentially no slack.
This is an existence witness at the boundary, not a robust load allocation.

Three bounded SSH attempts in this heartbeat timed out; two ping probes received
no replies. The Mac route to the host is via `feth3199`, with direct SSH port22;
no proxy jump, routing edit or service action was used. Remote state and the
older interrupted report remain unknown. Do not infer that a newer commit was
installed there. On reconnection, reconcile first, then fast-forward only a clean
idle worktree and run the targeted Linux CPU tests before any GPU work.

Next unfinished local chunk: declare a margin-aware force allocation diagnostic
without rewriting this minimum-normal-force result, then bind the actual BAM
actuator/contact dynamics and predeclare B1 flat stance. B0-A need not be repeated
to occupy the GPU. The September10 07:30 cutoff and all original skill gates stay
unchanged; no new optimizer, policy checkpoint or video was produced here.
