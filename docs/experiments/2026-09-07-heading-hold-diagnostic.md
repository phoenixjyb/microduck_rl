# Frozen-gait heading hold before obstacle training

Status: **completed, numerical-gate-stop**. The protocol below was committed
before execution at `72fa70ca035624a727c98c5d1f69b5ffce1e4c0d`.
User approved the heading-hold test
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

## Retained result: September 7, 11:36 Shanghai

The identical 685-test CPU regression passed on 100.100 in 10.34s (same two
warnings). The sequential user service
`microduck-rl-heading-hold-72fa70c-s443.service` ran 11:36:13–11:36:49,
exited 0, and retained **numerical-gate-stop** rather than policy acceptance.
Only seed 443 OFF/ON ran: two cases, 16 first episodes, 400 steps each.
Seeds 449/457 were deliberately not run after the first failed pair.
No optimizer steps, retry, gain adjustment, further GPU stage or video occurred.

| Settled metric | OFF | ON |
| --- | ---: | ---: |
| Body-forward mean (m/s) | 0.275398 | 0.275605 |
| Route-forward mean (m/s) | 0.270235 | 0.275115 |
| Worst absolute heading (rad) | 0.577876 | 0.157293 |
| Mean absolute cross-route speed (m/s) | 0.083598 | 0.075879 |
| Worst sampled cross-route displacement over whole run (m) | 0.506726 | 0.199146 |
| Pooled torque-utilization p99 | 0.585425 | 0.584964 |
| Mean squared utilization proxy | 0.0427573 | 0.0427124 |
| Mean absolute mechanical power (W, per actuator sample) | 0.148947 | 0.148589 |
| Stable route-speed window within deadline | 2/8 | 2/8 |

Both cases had no terminal/fall, nonfinite or safety failure, no rated-speed
exposure, and complete raw command/route traces. The deterministic failed-gate
list is exactly `straight-body-mean-outside-band`, `cross-route-motion`.
ON still has two slow body means (0.261598 and 0.254266m/s) below 0.27m/s;
all eight absolute cross-route speed means (0.073576–0.082777m/s) exceed 0.05.
Only 2/8 meet the instantaneous stable-speed window, despite the pooled mean.
The classification's first speed failure does not erase this additional window
shortfall or imply that all other speed criteria passed.

The fixed heading intervention reduced worst heading by **72.78%** and worst
sampled lateral excursion from 50.7cm to 19.9cm in this pair. All predeclared
paired torque, per-joint, load, power and speed nonregression margins passed.
This is a useful single-seed causal diagnostic, not multi-seed robustness,
standing/disturbance retention, physical motor validation or obstacle admission.
The highest ON named-joint p99 is right_hip_pitch **0.717471**; the pooled
0.584964 must not be represented as every motor being below 0.60.

Read-only decomposition of route velocity into yaw-aligned forward/lateral
components leaves mean absolute lateral speeds 0.071846–0.079515m/s under ON,
close to OFF's 0.070720–0.079148. Heading correction addresses route rotation,
but not this remaining lateral motion or the two slower environments. Do not
relabel the residual as harmless sway or loosen gates. A next training proposal
should target this low-level lateral/speed deficit with a matched motor-aware
control, before expanding obstacle geometry; no such training is launched here.

The command audit reproduces every issued yaw value from its retained heading
and verifies exact issued/actor-input equality. ON peak yaw command was
0.157293rad/s; maximum sampled change was 0.019892rad/s per step, below 0.02.
There were 3,193 cached-vs-issued yaw components that differed under ON;
the fresh adapter delivered the corrected values without a second sensor draw.
OFF had zero cached/issued differences. Hash-preserved checkpoints remain the
original parent, unaccepted narrow candidate and rejected H1-T; none is promoted.

### Evidence identities and closure

All **6 payload files, 3,290,707 bytes**, plus manifest, were mirrored and hash
verified on both hosts. Reconstructing the traces and rerunning `compare` from
the reports reproduced the retained failed-gate decision on both hosts.

- Remote root: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum/artifacts/evaluations/heading-hold-frozen-narrow-s443-v1`.
- Local mirror: `/Users/yanbo/Projects/microduckPlayground/microduck_rl/artifacts/diagnostics/heading-hold-frozen-narrow-s443-v1`.
- Manifest SHA256: `b0e1d2af4bbadaa8bea71f331063c97d1d3fa1e1c24062053ae234f45c2726b1`.
- Decision SHA256: `e138ac70dd9870467b59937fe56544453a7d36e4e3dda820f5c5703e5ed3f461`.
- OFF report SHA256: `0fa4c79753b7cb18e94c0a3debb20c5e7d42c6a19e72bf8715aca318ac690d16`.
- ON report SHA256: `b348677ab29b4405178c24dfab42b62a0aa19a38f267c5e872cfeedc12359d41`.

Observed GPU during OFF: one Duck compute PID, 55% utilization, 50°C, 503MiB;
this is a sample, not a full-run thermal peak. After completion: no compute PID,
0% utilization, 48°C, 12MiB. Both protected system services remained inactive;
unrelated services and 100.98 were untouched. No physical motion was authorized.
The bounded diagnostic is closed, with obstacle/hop/integrated-stability gates
still closed and no background Duck job left running.
