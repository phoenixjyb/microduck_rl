# Moving-corner skill preparation

Status: **new teacher source and synthetic tests, not a trained or accepted skill**.
The user's renewed request starts this bounded development slice; it does not
reopen an expired overnight launch window. No PPO, GPU rollout, policy switch,
raw perception, videos or physical robot motion is launched here.

## Why this lesson

The useful next capability is following a bent route and reacquiring its heading
and nominal speed. It builds toward corridors and obstacle composition without
overwriting the locomotion actor. The retained
[HC0 command envelope](../hierarchical_obstacle_controller.md#hc0-measured-command-envelope)
supports moving yaw at a0.30m/s command but explicitly rejects in-place
±0.3rad/s for that checkpoint. Thus the first candidate is a **moving30-degree
corner**, not an assumed stop/pivot/restart primitive. Command support does not
establish actual0.30m/s speed: the historical straight response was about0.21.

`corner_navigation.py` supplies a separate, unwired high-level teacher emitting
only forward-speed and yaw-rate commands. Actor weights, observation ordering,
normalizer, joint actions, motor limits and installed libraries stay unchanged.
The teacher is not the obstacle17D network, and no compatibility or policy switch
is inferred from having two velocity outputs.

## Geometry and control contract

The external route provider supplies a fixed local XY frame with origin at the
start, measured pose/heading, **both world-frame velocity components**, yaw rate,
tilt, grounded/contact validity, collision/fall and a monotonic timestamp. These
are teacher inputs, not new locomotion actor observations or camera features.
Timestamp/domain checks do not authenticate the publisher or its coordinate frame;
the runtime adapter must establish those bindings separately.

The proposed first route is1m straight, a signed30-degree circular arc of radius1m,
then1m straight along the new heading. Later60/90-degree cases exist in source
tests only. `APPROACH -> TURN -> RESUME -> DONE` describes teacher sequencing,
not learned acceptance. Arc radial error and outgoing lateral error must stay
within0.15m. Heading/tilt limits are5degrees. Commands are bounded at0.30m/s and
±0.30rad/s, with speed/yaw slew0.20m/s² and0.60rad/s². These are proposed command
limits, not measured acceleration, motor safety or hardware calibration.

Startup begins from zero command; turning uses measured body-forward speed for
curvature feedforward and bounded route/heading correction. It never commands an
in-place pivot. The candidate holds the same nominal command through the arc,
but a future learning objective must not penalize necessary slowdown inside the
maneuver. Approach and recovery still require actual nominal-speed tracking.
The existing obstacle interaction exemption and all old protocols remain unchanged.

Recovery is assessed from world velocity projected onto the **new route**. It
requires a contiguous sampled0.50s span within±0.03m/s of nominal, aligned within
5degrees and settled yaw within0.05rad/s, completed within2s of entering resume.
At50Hz this needs26 samples, not25. A later bad sample resets the current dwell;
an earlier recovery event cannot grant immediate exit after another speed loss.
Floating geometric boundaries use only a1e-12 roundoff allowance; no existing
acceptance gate or retained report is rescored.
Timestamp boundary comparisons use the same1e-12 arithmetic allowance; a dwell
that completes after the2s deadline plus that allowance cannot bypass failure.

Missing, stale/future, duplicate/backward or nonfinite observations, controller
gaps over50ms, lost ground support, fall/collision, excessive tilt, route escape,
missed recovery, and20s attempt timeout latch failure. New instance means a new
explicit attempt; no automatic reset or post-terminal continuation is accepted.
Grounded means at least one valid support, **not double support throughout walking**.
The proposed no-flight domain and contact validity still need runtime qualification.
Failure produces a slew-limited zero target; it is **not** a verified safe stop.
An independent safety layer must handle support loss, actual stopping and motors.

## Ordered development and training gates

1. Review and test this source-only teacher. Synthetic geometric traces are not
   dynamics, inference, motor evidence, simulator qualification or learned skill.
2. Complete the applicable native/GPU simulator admission already recorded in the
   [current control investigation](2026-10-10-ada-measured-boundary-control.md).
   Bind the exact frozen base, plant, actor/action mappings and route-state adapter.
   The October10 getter adapter remains mock-only; do not bypass that gate.
3. Predeclare a capped **frozen-base baseline**, before training: mirrored±30-degree
   routes plus matched straight controls, retained first-attempt traces and actual
   approach/recovery speed, heading/lateral/radial error, fall/collision/timeout,
   command slew, rated-speed and original motor-envelope/nonregression metrics.
   Select fixed seeds, world count, source/artifact hashes, motor gates, collection
   protocol and runtime/retention caps before observing any outcome. This source
   slice deliberately does not invent launch-ready identities or a seed verdict.
4. If the frozen teacher/base fails, identify whether command authority, gait,
   speed tracking or path geometry failed. Only then predeclare the smallest
   supervisor- or locomotion-learning revision; do not change several axes at once.
   If the frozen system already passes, retain that capability instead of training
   the same solved cells simply to spend GPU time.
5. After independent held-out confirmation and original straight/stance/recovery/
   obstacle retention, expand30→60→90degrees, then bind obstacle/corridor geometry.
   In-place pivots require their own admitted primitive. Hopping and rolling-ball
   lessons remain separate tracks with their unchanged feasibility/retention gates.

This is a finite skill-development ladder, not permission to repeat infrastructure
checks indefinitely. The next measurable outcome is the baseline corner trace;
source tests alone cannot provide it.

## Source validation

The six-file CPU regression (corner teacher, recovery measurement/control,
hierarchical teacher, retention plan and obstacle freshness) passed300 tests
in7.01s with CUDA hidden. It includes129 corner tests: mirrored/rotated geometry,
world-velocity projection, dwell interruption, exact/late recovery deadlines,
stale/future/duplicate inputs, latched failures, timeout/overshoot, command bounds
and import isolation. Two **ideal kinematic** fixtures follow commands from the
origin through the arc to exit with perfect actuation, no mass/contacts/actor or
robot physics. Neither these nor synthetic pose sequences are runtime rollouts.
`py_compile`, document-link checks and `git diff --check` passed. Independent
source review and exact native CPU delivery checks are retained separately below.
Independent read-only final review passed129/129 focused tests in0.23s using the
existing virtualenv with CUDA hidden, and found no blocking geometry/state/config
issue. It confirmed that the timeout lower bound is optimistic route arithmetic,
not a guarantee of feasibility under actual command slew or robot dynamics.

### Exact-source CPU delivery

Source `a93cfa4716ed40e0a11fbd581bc7617faacf04cb` was pushed to the fork
feature branch and fast-forwarded into the clean native100.100 worktree.
The source bundle SHA256 is
`211e26d8d4aa5bfc68914384d55f40bf53954f472aba23413546619a11592acc`.
Committed-source Mac/native regressions passed300 tests in6.72s/5.39s. All300
ordered JUnit identities match with zero failures/errors/skips. Retained files
under `artifacts/tools/corner-navigation/` on both machines:

- `a93cfa47-mac-tests.xml`,1090371B, SHA256
  `f7c4980387042e8da10541cee8a09f514ff6731d05c60c175c43e820e6bedee3`;
- `a93cfa47-linux-tests.xml`,1090377B, SHA256
  `f22604658409076b3220d6eec5c1c3ecaa357084977467cb0028eb0f812bdc88`;
- `a93cfa47-terminal-evidence.json`,8001B, SHA256
  `98c4a119b642defff111dff478f8eb5578c0012c7bc88e5d3fca014f86fbab5f`.

The CPU-only `microduck-corner-contracts-a93cfa47.service` finished exit0,
PID0/empty cgroup, elapsed6.878578s, invocation
`4da8a0af5e694ff0b396e22ae02a8245`. Its terminal transcript and exact unit
definition retain the six-file argv, CUDA-hidden/thread1 environment and
60s/2GiB/CPU200%/Tasks64 resource caps. Owner validation checked those fields
and exact argv using the previously hash-bound quoted-unit parser, not a lossy
space-split rendering. Grounding DINO1592/946MiB remained the sole GPU process;
all four protected AI Mission system/user scopes stayed inactive, and the
post-run lease file was observed empty. No GPU workload, policy update, robot physics
rollout or new skill acceptance was performed by this delivery.
