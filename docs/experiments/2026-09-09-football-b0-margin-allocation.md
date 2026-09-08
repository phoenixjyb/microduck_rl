# B0-M: fixed interior force-allocation diagnostic

Predeclared before execution: use only the retained B0-A pose in
`artifacts/diagnostics/football-b0a-stance-local-e79aadf.json`, SHA256
`d15226a8eacb69a347f4ece725528ecb102e9fde05e69e9035cb861c6e5bebf6`.
Do not repeat the pose search or overwrite the original near-boundary allocation.

Rebuild the same full-collision free-ball fixture, require every referenced
mesh/XML hash to match, restore the exact qpos, keep zero velocity/control and
refresh kinematics/contact candidates only. Reconcile contact positions/frames,
effective friction, robot/ball masses and COM with the retained geometry before
solving; allow1e-8 absolute numerical tolerance for geometry and no asset changes.

Make exactly one new allocation using the existing four-ray LP, with a fixed
**0.5 utilization cap**: rays use half the actual sliding friction coefficients,
while the simulator model/contact coefficients stay unchanged. Minimize the same
normal-force objective under identical force/moment equations and numerical
tolerances. Independently compute utilization against the actual full-friction
pyramid, requiring <=0.5+1e-8 and positive supporting normal force at all three
contacts. This is a more conservative feasibility set, not a new friction model,
real-world friction calibration, runtime controller or relaxed historical gate.

If feasible, recompute ideal hinge torques for this allocation and compare the
same named joints with the original allocation. Both are ideal rigid static
loads; neither executes BAM, proves motor capacity, establishes compliant contact
equilibrium or preserves balance under disturbance. If infeasible, retain the
failure without adjusting the cap or scanning additional values in this protocol.

Retain input hash, exact code/runtime/asset hashes, unchanged pose, both force
allocations, full-friction utilization/slack and ideal torque comparison. Output
is a new exclusive file outside the input. No GPU, simulator integration,
policy/checkpoint update, video, raw perception or physical motion. Then proceed
to the explicit BAM/contact-dynamics binding and flat stance B1 declaration.

## Initial local CPU result

The single0.5-cap allocation was feasible without changing the saved qpos,
contact coefficients, robot XML or meshes. Actual full-pyramid utilization was
0.499149 /0.500000 at the feet and effectively zero at the floor. Full-cone
foot slack was approximately1.66187 /1.65622 N in the scaled tangential-L1
representation. This is an algebraic margin relative to the declared ideal
friction model, not a measured reserve on an inflated football.

Peak ideal hinge torque was0.0789003 Nm at the left hip roll, versus0.2149040 Nm
in B0-A; right hip roll was-0.0787376 Nm. The knees need about0.0556 Nm. These
different allocations use the exact same pose and gravity, showing the prior
high hip load was allocation-dependent, not a unique minimum motor demand.
The chosen0.5 cap was not tuned or swept, and this result does not rewrite B0-A.
Free-root residuals were about2.1e-10 or smaller in their respective N/Nm units.

All42 football tests passed locally with CUDA hidden, including the real retained
input test. That test is explicitly skipped on a checkout without the retained
input, so a remote test count must disclose whether the input was installed.
The previously observed intermittent Mac process-cleanup failure is not repaired
by this work; the cleanup supervisor and all historical gates remain unchanged.

100.100 did not respond to this heartbeat's initial bounded SSH check. Reconcile
its current state and the previously interrupted remote report before any remote
worktree change or GPU launch. Local analysis does not confirm current GPU or
protected-service state.

The current broad CPU regression set passed398 tests in21.33 s, including the
retained-input case and the unchanged process-supervisor tests. The earlier
intermittent Darwin cleanup failure remains recorded in B0-A; a passing run
does not establish its root cause or repair. Both changed Markdown documents
rendered to HTML and their local links passed structural checks; no robot video
or dynamic balance visualization was generated.

## Retained report and next gate

Code source `bd5502e` was committed and pushed to the exact fork feature branch.
The allowlisted native Mac CPU report is retained at
`artifacts/diagnostics/football-b0m-margin-local-bd5502e.json`, SHA256
`077d6c8f87b2373eef1d5e8afb9b77bf3c86b02f5f108958dc8709bfb6f7e048`.
It retains both allocations, their named ideal torque comparison, original input
hash, pose/asset hashes and exact current code/runtime versions. The source
input remains byte-identical and no pose search was repeated by B0-M.

Both bounded SSH attempts during this heartbeat timed out, so current remote
branch, GPU occupancy and protected-service state could not be confirmed.
No remote worktree, service, GPU job or network setting was changed. The older
remote wrench report remains unconfirmed. On reconnect, reconcile that report
without overwriting it, verify host/GPU/service state and a clean branch, then
fast-forward the feature worktree. Install the exact retained B0-A input only
after checking the destination, verify its hash, and require the real-input
Linux test to run rather than count an optional skip as a validation pass.

The next local implementation target is a pinned BAM/contact-dynamics fixture
and an explicitly declared B1 flat-stance lesson. Do not spend GPU time rerunning
the closed speed map, B0-A or B0-M, or claim these algebraic results as learned
football balance. The existing September10 07:30 cutoff remains unchanged.
