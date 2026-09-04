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
passage-time target. HC3-A/B/C then fine-tuned only the supervisor for passage
time and residual outcomes. All three candidates were rejected by the retained
HC2 safety-regression gate. HC3-D balanced all four accepted HC2 cells during
training and retained each iteration for selection. Its safety-neutral snapshot
was 0.017 seconds slower than HC2; its faster snapshot improved weighted passage
by 0.094 seconds but recorded four collisions. HC2 therefore remains the
accepted controller. Nominal speed tracking remains active in approach and
recovery but is not imposed during interaction.

See `experiments/2026-09-03-hc3-supervisor-ppo.md` and
`experiments/2026-09-03-hc3d-balanced-supervisor-ppo.md` for the retained HC3
configurations, hashes, measurements, and next design gate. The next experiment
freezes HC2 yaw and may learn only a bounded interaction-speed reduction; it
does not expand placement. HC3-E implemented that boundary. Its seed-109
checkpoint matched HC2's collision count across six evaluation seeds, removed
one timeout, and improved weighted passage by 0.027 seconds. However, only two
of three independent training seeds passed the sensitive-cell pre-screen; seed
113 recorded three collisions. HC3-E is therefore not promoted and no new MP4
is recorded. HC3-F averaged the three seed-specific speed-head updates while
preserving the HC2 anchor exactly, but its 0.80 m/s by 1.40 m sensitive cell
recorded two collisions and stopped before the full matrix. The final bounded
speed-head candidate kept only update coordinates whose direction agreed
across all three training seeds. HC3-G still recorded three pre-screen
collisions, so the HC3 speed-head line is closed and HC2 remains accepted. The
next stage measures unchanged HC2 at lateral obstacle placements before any
new policy training. See
`experiments/2026-09-03-hc3f-seed-averaged-speed-head.md` and
`experiments/2026-09-03-hc3g-seed-consensus-speed-head.md`.

HC4-L then measured the placement gap and trained one lateral-placement BC
specialist from deterministic teacher data at +/-0.12 m. The specialist cut
shifted-placement collisions from 48 to zero on held-out seeds 61--63, but it
recorded four centered collisions versus HC2's two and therefore could not
replace HC2 wholesale. HC4-LH composes the byte-exact HC2 center model with the
HC4-L specialist behind a 0.06 m reconstructed route-lateral gate. Its actual
36-cell held-out matrix retained 2,408 clean passes, three collisions, and nine
timeouts among 2,420 resolved attempts (99.504% clean), with zero falls, NaNs,
non-finite steps, or rated motor-speed exceedances. Center collisions remained
at two and shifted collisions fell from 48 to one, so HC4-LH is accepted only
for the exact centered/+/-0.12 m simulation envelope.

The boundary diagnostic then showed that the HC4-L specialist generalizes to
the previously unseen +/-0.04 m points much better than HC2. A threshold-only
HC4-LH candidate reduced the exact-geometry center band from 0.06 to 0.02 m;
no network was retrained. Across 42 held-out cells at lateral positions
-0.18/-0.08/-0.04/0.00/+0.04/+0.08/+0.18 m, it retained 3,092 clean passes,
six collisions, and ten timeouts among 3,108 resolved attempts (99.485%
clean), with all hard safety counters clean. This supersedes the 0.06 m gate
for exact structured geometry only. The next single-axis stage varies forward
range while keeping speed, lateral positions, box geometry, and sensor quality
fixed. See `experiments/2026-09-03-hc4l-lateral-placement.md` for hashes,
comparisons, assets, and the simulation-only decision boundary.

HC4-R next tested the single-axis 0.90 m near-range boundary at only 0.30 and
0.40 m/s. A deterministic-teacher corpus of 107,665 samples passed its data
gate, and the seed-42 BC specialist passed the offline imitation thresholds.
It did not pass paired closed-loop selection: across held-out seeds
109/113/127 it recorded 1,433 clean passages, ten collisions, and zero
timeouts among 1,443 resolved attempts, versus HC4-LH's 1,364 clean passages,
seven collisions, and zero timeouts among 1,371. HC4-R was 0.177 seconds
faster by weighted passage time but introduced three net collisions. It is
rejected, HC4-LH remains selected, and 0.90 m remains outside the accepted
controller envelope. The next near-range design must address student-state
covariate shift before another training candidate is launched. See
`experiments/2026-09-03-hc4r-near-range.md`.

HC4-R2 is the next bounded attempt. It keeps the rejected HC4-R student in
simulation and asks the deterministic teacher to label states reached under
that student's execution. Resolved clean, collision, and timeout episodes are
retained with student commands and outcome codes; partial and hard-failure
episodes are excluded. Training keeps the HC4-R architecture, optimizer, and
seed fixed and adds three correction-data seeds, so student-state coverage is
the only intended change. See
`experiments/2026-09-04-hc4r2-student-state-correction.md` for the immutable
collection seeds and closed-loop gates.

HC4-R2 passed those gates. Across held-out seeds 149/151/157 it recorded 1,417
clean passages, two collisions, and zero timeouts among 1,419 resolved
attempts, versus four collisions for HC4-R and eight for HC4-LH on the same
matrix. Every shifted cell was collision-free; one centered collision remained
at each speed. Hard safety counters and rated motor-speed exceedance stayed
zero, and maximum per-cell torque-utilization p99 was 0.5650. HC4-R2 is
accepted only as an exact-geometry specialist at 0.30/0.40 x 0.90 m. HC4-LH
remains selected for its existing farther-range envelope. The next gate is a
bounded range/speed composition with an invalid-geometry fallback, followed by
an actual closed-loop boundary matrix; no physical motion is authorized.

HC4-R2H implements that composition without retraining. It selects HC4-R2 only
for valid structured geometry at reconstructed route-forward range no greater
than 0.95 m and nominal speed no greater than 0.40 m/s; every other observation
preserves HC4-LH. Invalid geometry still triggers the execution-layer immediate
stop outside recovery. The stateless composition passed focused unit and
actual-checkpoint CPU wiring checks, but failed the three-seed closed-loop
gate: it recorded three collisions versus two for the paired selected
baselines. Its lower timeout count and faster passage do not override the
collision regression. The threshold diagnostic and MP4 were not run. HC4-LH
therefore remains selected; the next design must latch one specialist for an
entire episode rather than switching on instantaneous range. See
`experiments/2026-09-04-hc4r2h-range-speed-composition.md`.
