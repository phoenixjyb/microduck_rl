# MicroDuck athletics and obstacle curriculum

This document defines the training order and evidence gates for teaching the
MicroDuck to run, hop, jump, and negotiate obstacles in simulation. Perception
is a separate subsystem: locomotion consumes compact geometry and maneuver
intent, never raw images.

The curriculum is deliberately sequential. A policy advances only after a
retained, multi-seed evaluation passes. A later skill must not hide a regression
in an earlier one.

## Capability graph

1. **Foundation** — stand, recover, and track low-speed commands.
2. **Motor-aware running** — track 0.5 m/s without exceeding the rated motor
   envelope. This is the parent for obstacle locomotion.
3. **Bypass navigation** — learn a compact, route-preserving detour around a
   tall obstacle.
4. **Hop** — leave and regain the ground in place with controlled landing.
5. **Forward jump** — add forward displacement to the accepted hop primitive.
6. **Obstacle jump** — clear only low obstacles whose geometry is within the
   accepted jump envelope.
7. **Skill routing** — an external supervisor chooses stop, bypass, or jump;
   it remains outside the locomotion policy and retains safety authority.

Running, bypassing, and jumping are separate policy/evidence tracks until each
is stable. They may later share a network, but a shared network does not weaken
their individual acceptance gates.

## Bypass ladder

The failed centered-box campaign started at the graduation problem. The next
campaign starts with a visible lateral hint and changes one difficulty axis at
a time.

| Stage | Obstacle placement | Command | Sensor | New difficulty | Promotion target |
| --- | --- | --- | --- | --- | --- |
| OA0 | 1.15 m forward; signed absolute lateral offset 0.24–0.30 m | fixed 0.30 m/s | exact, zero lag | learn “move away, pass, return” | >=85% clean passes per train seed |
| OA0R | unchanged | unchanged | unchanged | time-step-normalized terminal outcome only | same OA0 gates |
| OA0P | unchanged | unchanged externally | unchanged | suspend speed shaping only in the avoidance zone | same OA0 gates |
| OA1 | 1.15 m forward; signed absolute lateral offset 0.10–0.16 m | fixed 0.30 m/s | unchanged | smaller lateral hint | >=80% per seed |
| O1a | 1.15 m forward; centered | fixed 0.30 m/s | unchanged | remove lateral hint | >=75% per seed |
| O1b | unchanged | fixed 0.40 m/s | unchanged | speed only | >=72% per seed |
| O1 | unchanged | fixed 0.50 m/s | unchanged | speed only | existing exact O1 gates |
| O2a | forward position 0.90–1.30 m; centered | accepted O1 speed | unchanged | range only | O1 gates across range bins |
| O2b | O2a plus lateral position -0.25–0.25 m | unchanged | unchanged | bearing only | O1 gates across bearing bins |
| O2c | O2b plus measured width/height envelope | unchanged | unchanged | geometry only | gates across geometry bins |
| O3a | unchanged | unchanged | measured range noise | range noise only | no more than 5 percentage-point pass loss |
| O3b | unchanged | unchanged | add measured bearing noise | bearing noise only | same regression bound |
| O3c | unchanged | unchanged | add measured latency | latency only | same regression bound |
| O3d | unchanged | unchanged | add measured dropout | dropout only | fail-safe stop plus retained recovery |

The signed absolute offset in OA0/OA1 samples both sides but excludes the hard
center band. This supplies a bearing sign while keeping the actor interface
unchanged. O1 remains protocol `O1-centered-exact-v1`; training scaffolds must
not rename or weaken that benchmark.

### Route-preserving objective

The bypass policy must do all three actions in order:

1. commit laterally early enough to maintain the collision margin;
2. continue forward past the obstacle;
3. converge back toward the original route.

Reward alone is not accepted as evidence. Each stage reports clean passage,
collision, fall, non-finite state, pre-obstacle route speed, passage time,
maximum lateral excursion, and post-pass route-return error. A bounded attempt
horizon marks lingering as failure. Its initial value must accommodate the
stage speed and distance; it is reduced independently only after clean passage
is reliable.

## Hop and jump ladder

Obstacle jumping starts only after motor-aware running and the independent hop
track both pass.

| Stage | Task | Difficulty axis | Core evidence |
| --- | --- | --- | --- |
| H0 | symmetric in-place hop | target apex | takeoff, airborne phase, landing, no fall |
| H1 | repeated in-place hops | repetition count | height consistency and thermal estimate |
| J0 | forward jump without obstacle | forward distance | landing stability and route error |
| J1 | fixed low obstacle | obstacle height | collision-free clearance margin |
| J2 | varied low obstacle | height bins | per-bin pass rate |
| J3 | low obstacle with sensor perturbation | one sensor axis per stage | bounded degradation and fail-safe stop |

A high or uncertain obstacle is never silently assigned to the jump policy.
The supervisor checks accepted height, width, approach-speed, landing-zone, and
motor envelopes; otherwise it selects bypass or stop.

## Evidence and selection rules

- Use at least three independent training seeds and at least three held-out
  evaluation seeds per candidate.
- Compare exact checkpoint iterations shared by every training seed. Select the
  earliest passing iteration, never the final checkpoint by default.
- Keep protocol identity, source commit, parent checkpoint hash, training seed,
  evaluation seeds, and checkpoint hashes in retained manifests.
- A stage fails on any fall, NaN termination, or non-finite step in its bounded
  acceptance matrix.
- Run motor-envelope and reset-safe action-rate comparisons only after the task
  gate produces a survivor.
- Record MP4s only for a numerically accepted candidate; video is qualitative
  evidence, not a substitute for the metrics.
- Every result remains simulation-only. No report or checkpoint authorizes
  physical motion.

## Current evidence and next implementation gate

The 2026-09-03 exact O1 campaign trained seeds 42–44 for 64 iterations from the
same motor-aware warm start. Its best pooled clean-pass rate was 14.606% at
iteration 8061, against the 75% campaign target. There were no falls, NaN
terminations, or non-finite steps. Training traces showed increasing reward but
falling route speed and growing lateral displacement, consistent with a
side-step-and-linger strategy. The retained acceptance decision is rejected.

OA0 then added signed-offset sampling, a bounded attempt horizon, route-return
success, and timeout-aware evaluation. Its earliest common checkpoint reached a
25.134% pooled clean-pass rate; later checkpoints regressed to 7.665% while
episode length approached the seven-second limit. RewardManager scales rewards
by the 0.02-second control step, so the original terminal `+10/-10` contributed
only `+/-0.2` per outcome—far less than the reward earned by lingering for one
extra second.

OA0R is therefore the next single-axis experiment. It leaves placement, speed,
sensors, horizon, success geometry, and acceptance gates unchanged, and adds a
time-step-normalized `+20` success / `-20` collision-or-timeout impulse. No
later curriculum stage starts unless OA0R passes its retained three-seed sweep.

OA0R also failed: its earliest common checkpoint reached 26.134% pooled clean
passes, then regressed to 8.025%. The larger outcome signal modestly improved
the inherited checkpoint but did not resolve the objective conflict. OA0P is
the next single-axis experiment. The external command remains 0.30 m/s and
normal linear-speed plus route-progress shaping remains active while the
obstacle is at least 0.60 m ahead. Both are suspended while the obstacle is in
the interaction zone and resume once its center is behind the robot. Collision,
timeout, route-return, angular tracking, motor, and acceptance contracts do not
change. Thus OA0P permits braking or slowing for safety without rewarding
lingering.

The bounded OA0P seed-42 pilot also failed. Its clean-pass rate fell from
18.154% at iteration 8000 to 3.320% at iteration 8061. Approach speed fell from
0.208 to 0.179 m/s and recovery speed stayed below 0.212 m/s. Because OA0P
separately measures approach, interaction, and recovery, this is evidence that
the shared 14-joint PPO policy is forgetting its locomotion skill, not merely
choosing a legitimate low speed inside the maneuver. Seeds 43 and 44 are not
started.

The next implementation track is therefore hierarchical. The accepted
motor-aware locomotion policy remains frozen and consumes a bounded velocity
command. A lower-rate obstacle supervisor consumes compact obstacle geometry
plus route state and produces only a forward-speed scale and yaw-rate command.
It cannot write joint targets. See `hierarchical_obstacle_controller.md` for
the contract and staged gates.

HC0 then established a measured command envelope, HC1 supplied successful
deterministic teacher trajectories, and HC2 trained a 17D behavioral-cloning
supervisor while keeping the 61D motor-aware actor frozen. On the bounded cells
0.3x1.15, 0.5x1.15/1.40, and 0.8x1.40 m, three-seed HC2 closed-loop evaluation
retained 759 clean passes, one collision, and eight timeouts (98.83% pooled),
with zero falls, NaNs, non-finite steps, or rated motor-speed exceedance.

This does not retroactively pass the rejected direct-joint O1 campaign. HC2 is
accepted only for its named simulation envelope, and still misses the 4.5 s O1
passage-time target. The next single-axis stage is HC3 supervisor-only
fine-tuning for passage time and residual outcomes. Nominal speed tracking
remains active in approach and recovery but is not imposed during interaction.
