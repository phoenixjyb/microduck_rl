# Frozen-gait heading hold before obstacle training

Status: **predeclared, not yet executed**. User approved the heading-hold test
and requested obstacle avoidance without sacrificing stabilization or hopping.
This diagnostic does not claim either skill retained by an integrated system.

## One change, fixed identities

Use the retained, unaccepted F1-R narrow model8498, SHA256
`7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`.
Compare heading hold OFF and ON, without optimizer steps, normalization updates,
reward or physics changes. This is not an accepted replacement for its parent.
Keep the61D actor and original sensor noise/delay. Externally supplied exact
simulation heading relative to the initial route is the only added controller
input; no camera, learned perception or actor-observation expansion.

Controller at50Hz: wrap heading to[-pi,pi], command yaw rate
`clip(-1.0*heading_error, -.35, .35)`rad/s with1rad/s² slew limit (.02 per step).
Initialize previous yaw command to zero. Forward command remains.30m/s,
lateral command zero. No integral, lateral-position controller, deadband,
gain search, speed adaptation or joint-target authority.

Both OFF and ON explicitly use the already-tested fresh raw-twist delivery
adapter before actor normalization, changing only61D indices48:51. Never
recompute the observation group or resample delayed sensors within a step.
OFF's constant command equals the cached command; ON's issued command must
equal the actor input in the same step. Retain issued/cached/consumed commands,
heading and actual route positions; reproduce the controller rule from them.
This is a distinct protocol, not a pass of historical fixed-zero-yaw F1.

## Budget and unchanged gates

- Branch `feat/athletics-obstacle-curriculum`, exact clean pushed source;
  host100.100 only, frozen runtime hashes, idle GPU, protected services inactive.
- Protocol/output: `heading-hold-frozen-narrow-s443-v1` under
  `artifacts/evaluations/`; existing output refuses a retry or overwrite.
- Seeds443,449,457; OFF then ON for each.8env,400steps,.02s dt,100startup
  and300settled, first terminal stops all environments. At most6cases/48
  first episodes. No training. One sequential user service, hard900s,
 90s/child and internal remaining-budget guard. Start before Sept7 12:30 Shanghai.
- OFF safety/coverage failure stops before ON. An ON numerical failure stops
  the remaining pairs. No retry, gain tuning or further GPU stage after failure.
- Keep all F1 absolute speed/heading/lateral/stable-window/motor gates and
  paired nonregression margins, using OFF as reference. A corrected heading
  alone is not full speed acceptance. Absolute-motion failures are not relabeled
  harmless sway, and no old gate is weakened.
- Additional motor-retention checks: each named pre-reset joint torque p99
  <=OFF+.02; squared-load proxy and mean absolute mechanical power<=OFF×1.05.
  These are predeclared diagnostic margins, not physical continuous ratings.
- Decisions: reference-safety-stop, numerical-gate-stop, runtime-failure-stop,
  or at most heading-diagnostic-support-only. All obstacle/hop/stabilization
  retention/policy/physical-motion admission flags remain false.

## Capability preservation and next curriculum

Before and after, verify original locomotion parent7998 SHA
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`,
the narrow diagnostic model, and retained H1-T5999 SHA
`454bd7db3da50896c2b00cebd72ea7eecf1fee3e9802a6ee03693e8fa858040a`.
File identity protects research assets; it does NOT establish functional
retention of a combined controller. H1-T remains rejected, not a validated hop.
No H2, hop retraining, video, real robot motion or service restoration here.

If heading hold is supported, next predeclare straight/stop/disturbance
retention and supervisor integration before wider obstacle placement. Obstacle
supervision may reduce speed during interaction; nominal-speed tracking resumes
before/after avoidance. Keep structured perception and locomotion separate.
Hopping requires its own accepted stability/motor gates before routing can
select it. Locomotion currently uses rigid feet while H1-T uses sprung K3900:
an eventual combined robot must use one explicitly agreed, matched plant, not
silently switch foot mechanics when switching policies.

Retain reports, complete command/route traces, exact model/source hashes,
decision and manifest; mirror locally and verify before pushing results.

## Prelaunch verification

Local CPU-focused regression: **685 passed**, two existing actuator/site-pattern
warnings, no skips (14.18s). Coverage includes real observation-manager command
injection without sensor resampling, mocked same-step rollout consumption and
terminal stopping, controller trace reconstruction, paired motor rejection,
and sequential campaign success/reference/numerical/runtime stop paths with
manifest hashes. This is software evidence, not simulation skill acceptance.
The remote rejected H1-T checkpoint hash was also verified unchanged before
launch; all three retained checkpoint sentinels are checked again by the job.
