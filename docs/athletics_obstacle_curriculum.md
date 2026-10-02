# MicroDuck athletics and obstacle curriculum

This document defines the training order and evidence gates for teaching the
MicroDuck to run, hop, jump, and negotiate obstacles in simulation. Perception
is a separate subsystem: locomotion consumes compact geometry and maneuver
intent, never raw images.

The curriculum is deliberately sequential. A policy advances only after a
retained, multi-seed evaluation passes. A later skill must not hide a regression
in an earlier one.

September8 addition: [football balancing](experiments/2026-09-08-football-balance-curriculum.md)
is a separate B0–B6 track: plant feasibility, flat stance, fixed-ball support,
prescribed rolling, free rolling, perturbations and mounting/dismounting with
retention. This is a new requirement, not an achieved capability or a renamed
ball-kicking/roller-foot task. The renewed simulation/development window ends
September9 07:30 Asia/Shanghai; old expired campaign windows below remain historical.

October1 addition: the [community skills research and intake](experiments/2026-10-01-community-skills-research.md)
identifies a stock/backlash one-shot hop reference and a recurrent basketball
balance reference. These are separate candidate investigations, not accepted
skills or replacements for the historical sprung-K3900 H1 and soccer-ball
B0–B6 gates. Pinned public artifacts are quarantined for CPU structural inspection;
no new training or controller switching is admitted by the research.

The [C1 preparation](experiments/2026-10-01-community-hop-c1-preparation.md)
now supplies tested first-attempt trace scoring and proposed timing/history
helpers. These are not a replay result. A matched Happy Hop diagnostic remains
blocked on the exact entry walker, missing task/run overlay and transition
semantics; the current official walker is not silently substituted.

The separately approved [C1-S substitution diagnostic](experiments/2026-10-01-community-hop-c1s-predeclaration.md)
is now predeclared: a walker-only control precedes one conditional hop handoff
on an explicit CPU surrogate. It is not author replication, a completed replay
or skill admission; tested runner/plant binding still precedes execution.

The [C1-S native CPU runner](experiments/2026-10-01-community-hop-c1s-runner.md)
and separate walker-only adapter are now implemented and fixture-tested. Static
preflight binds the pinned policies and compiled BAM surrogate without inference
or physics stepping. The subsequent [October 1 result and development window](experiments/2026-10-01-duck-development-window.md)
ran the approved walker-only control safely, but rejected approach and resume
speed tracking. Happy Hop was not invoked. The consumed C1-S attempt does not
establish a hopping failure or admit a retry; no old curriculum gate changed.
Bounded development continues through October 1 18:00 Asia/Shanghai under the
new window, not an expired September campaign.

September16 addition: the B1-N stance track's
[residual-lean lesson](experiments/2026-09-16-stance-lean-lesson.md) is a
**predeclaration, not a result**. The completed frozen initializer/final
comparison decided `final-numerical-rejected`: the iteration-127 export held a
stable, almost drift-free stance and cleared every gate except tilt. The lesson
exists only to separate an optimizer-budget explanation from a
mechanical-feasibility one, and it changes exactly one axis — budget and
starting weights. Its 0.0873 rad tilt gate is unchanged and is not to be
relaxed, and no reward coefficient is touched.

September17 addition: that lesson has now **run and been evaluated**, and the
declared decision is **`lean-lesson-passed-nominal`**. The 256-update
weight-initialized continuation (source `7bbc75fb`) was evaluated on all four
common checkpoints across seeds 541/547/557 at 128 environments each, and
**1,536 of 1,536 attempts cleared every gate with zero hard failures**. Worst
per-attempt final-second tilt p95 was 2.25299 deg at iteration 64, 2.40004 at 128,
1.55985 at 192 and **1.02774 at 255**, against the unchanged **0.0873 rad
(5 deg)** gate — cleared by more than a factor of two, with no relaxation and no
re-weighting. Explanation 1 (optimizer budget) is therefore supported and
explanation 2 (mechanical feasibility) is disfavoured, which is the entire
question the lesson was built to answer.

This admits **a nominal stance candidate only**. Fresh-seed training replication
and small-disturbance recovery still precede any B1 acceptance, and no fixed-ball
or rolling-football training follows directly. Every admission flag remains false,
including on this pass. See the
[residual-lean lesson](experiments/2026-09-16-stance-lean-lesson.md) and its
[evaluation timing probe](experiments/2026-09-17-stance-lean-lesson-evaluation-probe.md).

### October 1 WSL execution and next curriculum gate

The [packed runtime qualification](experiments/2026-10-01-wsl-packed-runtime-qualification.md)
completed on exact source `be2d59661af293b0d67ae20d2e16db50514cce14` on 100.98,
with same-input device checks and full collection timing fitting the fixed WSL
budget. The [packed replication](experiments/2026-10-01-wsl-packed-replication.md)
has completed all three matched seeds **577/587/593** on that frozen source.
Each complete archive and copied bytes passed verification on WSL and Mac:
771 exports including initializers, 768 finite optimizer receipts and 18,432
ordered finite-reward ticks in total. This is an execution milestone, **not** a
new stance, obstacle, hopping or football achievement. The original CLI window
ended October 1 at 18:00 Shanghai; no confirmed background automation exists.

Training and capability evaluation remain separate. Every complete archive was
authenticated before the single [WSL held-out timing probe](experiments/2026-10-01-wsl-packed-evaluation-probe.md).
Any future attempt requires separate fresh authority and repeat entry checks.
That fixed 255/541/128-world single case is not the twelve-case evaluation or the
three-training-seed verdict. Full evaluation must retain 64/128/192/255 ×
541/547/557 for each of the three training seeds: **4,608 first attempts** in
the complete campaign, never a smaller denominator after a missing run.

The single timing probe at evaluator source `d4e96cda975f` failed before publishing
its bundle: CPU replay rewrote the CUDA-origin binding, which the strict packed
validator refused. The failed attempt is retained on both hosts, with no usable
timing measurement or skill verdict. A bounded CPU repair now separates checker
device from capture metadata without weakening either capture or scoring gates.
This does not authorize an automatic retry. A separately reviewed fresh fixed
probe, then a successful measured-cap full-evaluator declaration, still precede
nominal replication acceptance; no recovery or ball lesson starts from a training
completion alone.

After that closeout the user renewed work. The separately predeclared
[repaired timing probe R1](experiments/2026-10-01-wsl-packed-evaluation-probe-r1.md)
allows one fresh fixed case through October 1 **19:00 Shanghai**, with unchanged
600/960-second caps and all entry gates. It does not renew training or enable the
full evaluator; if qualification leaves insufficient time, launch nothing.

R1 has now completed its fixed **255 / 541 / 128-world** case: **128/128**
complete numerical passes without gate relaxation. Its unchanged conservative
timing rule yields **3,350-second service / 3,290-second child** and **4,010
seconds including closeout/margin**, within the existing WSL ceiling. WSL
rehash/replay passed. All 14 copied Mac file hashes match, but Mac full replay
refused `compiled plant mismatch`: five native arrays differ at last-bit scale
between ARM Mac and x86 Linux. The exact gate was not relaxed. This is one
diagnostic case, not replicated stance. Qualify a compatible independent CPU
replay runtime before advancing; then a reviewed, tested full
evaluator must still retain all 4,608 attempts and apply the original cross-seed
decision. No further GPU job fits the remaining single-probe authority.

The renewed CPU-only [independent Linux investigation](experiments/2026-10-01-independent-linux-r1-replay.md)
authenticated all copied bytes and matched the entire compiled plant on
100.100, but strict replay refused an actor-error receipt mismatch (saved 0;
recomputed 8.34465e-7). A single AVX2 dispatch check also refused. The gate was
not relaxed. A separately declared portable CPU profile produces identical
actor-output hashes on both hosts for the retained 32,000 inputs; this is only
candidate arithmetic evidence, **not** successful original R1 replay or a skill
verdict. The optional profile checker is opt-in and does not change any runner.
A separately reviewed fresh capture under that profile and independent exact
whole-bundle replay still precede full-evaluator development and the 4,608-attempt
campaign. 100.100's current NVML mismatch remains unrepaired; 100.98 remains the
candidate GPU host, subject to fresh entry gates and GPU-job authority.

Following renewed authority, [R2 portable replay probe](experiments/2026-10-01-wsl-portable-probe-r2.md)
is separately declared for October 1 **21:00–22:00 Shanghai**. Its distinct
trace/probe protocols bind the invariant CPU profile before actor capture and
replay; existing gates, old artifacts and all capability flags remain unchanged.
It permits one fixed 255/541/128-world capture, not a new learner or full matrix.
Independent whole-bundle replay remains required before evaluator advancement.

October 2 renewed work runs through **October 3 08:00 Shanghai**, under the
[current continuation](experiments/2026-10-02-overnight-continuation.md), not the
expired windows above. The separate [R4 portable capture](experiments/2026-10-02-wsl-portable-probe-r4.md)
completed on source `07466b58021c084dbc2f688effa936d9466ba456`: all 128 nominal
first attempts passed and same-host actor replay was exact. The original
training initializer/archive authentication passed separately under AVX2.
Strict independent native-Linux replay **passed on October 3 at 00:05** after
complete archive delivery: all 14 files and 621,439,568 bytes matched, all 128
attempts were recomputed, and actor error was exactly zero with CUDA hidden.
Timed-out partial copies remain unaccepted and preserved. No full matrix,
new learner, disturbance recovery, hopping or football capability follows from
the single capture. A separately bounded CPU-only delivery closeout preserves
all original GPU limits and every acceptance gate. The separately declared
4,608-attempt evaluator has a [separate predeclaration](experiments/2026-10-03-portable-packed-full-evaluation.md)
and focused implementation checks. After successful original and portable Linux
CPU qualification, its source-frozen sequential CUDA evaluation started on
October 3 at 00:56 Shanghai and finished successfully at 03:11. All **4,608/4,608**
attempts passed, and the CPU-only supervisor replayed the entire serialized
matrix to re-derive `lean-replication-passed`; every checkpoint passed for every
training seed. This clears nominal simulation replication only. The additional
whole-matrix CPU closeout passed again with CUDA hidden, and all 329 retained
files / 22,352,528,834 bytes were hashed and rechecked. Every
capability/physical admission flag remains false. Disturbance recovery and
football promotion are not established by the nominal result.

The unchanged [replication rule](experiments/2026-09-17-stance-lean-replication.md)
requires each training seed to have at least one checkpoint passing every gate
at all three evaluation seeds, with at least 122/128 passes each. It does **not**
require the same passing iteration across all three training seeds. Report all
passing checkpoints and preserve that original rule. Before a later recovery
experiment, separately predeclare which candidate is selected; do not select a
different checkpoint after observing disturbance outcomes.

Only after complete, independently replayed nominal replication should the next
bounded lesson be **flat-ground small-disturbance recovery**. Predeclare
directions, magnitude, onset, duration, settling/recovery and motor/action-rate
criteria first. The current B1-N runtime explicitly rejects applied external
forces: do not silently disable that guard or label assisted trajectories as
nominal stance. A future recovery path needs an explicit perturbation contract
and focused fixtures, then its own source/timing/held-out gate. A nominal pass
alone still does not accept B1.

The next separately declared [D0 one-substep force fixture](experiments/2026-10-03-single-substep-force-fixture.md)
qualifies the external-force timing and independent physics replay first. It has
five fresh one-world 2-ms cases, no actor/checkpoint/optimizer, and no recovery
acceptance. The nominal no-push guard remains unchanged. Only a successful D0
result may precede a separately declared frozen-policy D1 recovery baseline.

After B0 full-robot/foot feasibility and B1 stance/recovery, football progresses
to fixed support (B2), prescribed slow rolling (B3) and only then free coupled
rolling (B4). Ball-only tests or an unstepped geometry fixture do not clear B0.
Obstacle bypass remains a separate phase-aware track: slowing is allowed in the
interaction zone, but approach/post-pass speed and route recovery remain gates.
Sprung hopping's historical rejections and community-policy source/handoff gaps
remain separate; neither is erased by a flat-stance run. Retention across these
tracks precedes any combined-skill claim.

No real Duck is available. Current stance receipts explicitly have no thermal
model and no applicable spring-bottoming model. Modeled torque/soft-limit
counters, numerical tests and GPU temperature must not be reported as robot
motor thermal calibration, spring safety or physical readiness.

## Capability graph

September7 integration boundary: the historical milestones below do not
constitute one fully validated combined robot. F1/F1-R/F1-L still fail current
straight-speed/route/motor-nonregression gates; H1-T is rejected and periodic
hopping is parked. Exact-geometry obstacle specialists remain retained only
within their named historical envelopes. See the current
[heading-hold diagnostic](experiments/2026-09-07-heading-hold-diagnostic.md).
The subsequent [F1-L continuation](experiments/2026-09-07-f1l-lateral-continuation.md)
reduced lateral motion in its first held-out pair but increased motor load and
was rejected. Its final comparison was recovered from stored data on CPU;
neither gait promotion nor a further seed sweep of F1-L is admitted.
Renewed September7 overnight authorization starts the separately predeclared
[F1-M motor-weight contrast](experiments/2026-09-07-f1m-motor-continuation.md).
The [overnight curriculum and budget](experiments/2026-09-07-overnight-curriculum.md)
define the gait, stop/recovery, obstacle-diversity and independent-hop sequence,
retention requirements and September8 07:00 Shanghai cutoff.
F1-M also stopped at its first pair: modest load/power improvements did not
pass speed, lateral or named-joint nonregression gates. Its next step is
motor-timing diagnosis, not an admitted obstacle or hopping stage.

For the requested capable obstacle avoider with stabilization and eventual
hopping, maintain a separate retention ledger:

| Track | Required before integrated promotion | Current boundary |
| --- | --- | --- |
| Stand/stop/disturbance recovery | First-attempt stability and motor evaluation of the proposed integrated controller | Not established by straight walking |
| Route/speed tracking | Per-env approach/recovery speed, heading, lateral displacement and motor gates | Heading improved; F1-L lateral gain failed motor limits and retained speed deficits |
| Obstacle avoidance | Clean pass, collision, timeout and route/speed recovery by placement/speed bin | Historical specialists are not universal acceptance |
| Hop/landing | Independent H1 survival, drift, spring and motor gates | H1-T rejected; no validated hop to claim retained |
| Skill transitions | Matched mechanics, bounded command/action transitions and all earlier retention tests | Not yet admitted |

Train the obstacle supervisor separately from locomotion; preserve frozen
checkpoint identities. Speed may drop during collision avoidance, not silently
in approach/recovery. No raw-image learning is required. Separate checkpoint
files do not by themselves prove behavioral retention: test the composed
controller too. Rigid-foot locomotion and sprung-foot hop experiments cannot
be promoted as the same plant without explicit mechanical alignment.

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
| H0 | matched Locked/K2500/K3900 controls | mechanics only | identical learner/task and valid instrumentation |
| H1 | periodic in-place hop | target apex and repetition | takeoff, airborne rise, landing, no fall |
| H2 | commanded one-shot jump | target rise | apex error, settle stability, in-place corridor |
| H3 | running jump | approach speed | clearance, landing stability, speed recovery |
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

HC4-R2L implements that single change with explicit per-environment reset
state. Initial 0.90 m valid low-speed episodes choose HC4-R2; initial 1.15 m,
high-speed, or invalid episodes choose HC4-LH. The choice cannot change until
the execution layer reports that episode done. Its first gate reuses the
HC4-R2H seeds only as a causal regression check; fresh seeds are separately
predeclared for promotion. HC4-LH remains selected until both stages pass. See
`experiments/2026-09-04-hc4r2l-episode-latched-composition.md`.

HC4-R2L's causal matrix matched the paired sources at two collisions and eleven
timeouts and removed HC4-R2H's added collision, but its clean-pass rate was
99.5002% versus 99.5008% for the sources, narrowly below the fixed 99.501%
continuation floor. The fresh promotion and threshold matrices were therefore
not run. HC4-LH remains the accepted composite, while HC4-R2 remains a retained
near-range specialist rather than a unified runtime replacement.

HC4-E1 now fixes the evaluation denominator before another composition attempt.
Future candidates use exactly one first terminal attempt per environment and
exclude all post-reset samples; expected, completed, unresolved, hard-failure,
and unclassified terminal counts are explicit. Legacy fixed-step results and
the HC4-R2L rejection remain unchanged. A four-environment accepted-HC4-LH
wiring smoke is the only runtime action authorized by this source slice. See
`experiments/2026-09-05-hc4e1-fixed-attempt-protocol.md`.

HC4-U1 is the next predeclared candidate. It removes runtime specialist
selection by fitting one unchanged 64x64 supervisor to the exact ordered union
of accepted far/lateral teacher data and near-range teacher plus student-state
corrections. Its first closed-loop evidence uses fresh seeds and HC4-E1's fixed
64-attempt denominator, paired against HC4-R2 at 0.90 m and HC4-LH at 1.15 m.
Approach and recovery speed may not regress, while interaction speed remains
deliberately ungated. See
`experiments/2026-09-05-hc4u1-unified-supervisor.md`.

HC4-U1 passed its offline fit and improved the seed-193 aggregate matrix to 766
clean passages, zero collisions, and two timeouts, versus 760, two, and six for
the paired specialists. It nevertheless failed the frozen per-cell timeout
gate in the 0.30 m/s far-center cell: one timeout versus none for HC4-LH. Fresh
seeds and video were not run. The smallest defensible next revision is new
student-state correction data for that single lingering mode, not a rerun or a
post-hoc gate relaxation; HC4-LH and HC4-R2 remain the accepted specialists.

HC4-U2 predeclares that one-axis correction: HC4-U1 executes only the failing
0.30 m/s, 1.15 m far-center cell while the deterministic teacher labels its
student-reached states. Seed 233 is a bounded collection pre-screen before
seeds 239/241. Any later candidate keeps HC4-U1's architecture and base data
and must use fresh evaluation seeds 251/257/263 under the unchanged HC4-E1
gate. See `experiments/2026-09-05-hc4u2-far-center-correction.md`.

HC4-U2 passed collection and offline fitting but failed fresh seed 251. It had
763 clean passages, zero collisions, and five timeouts versus the paired
specialists' 762, zero, and six, yet the far-center 0.30 m/s cell regressed by
one clean passage and one timeout. Seeds 257/263 and video were not run. Since
HC4-U1 and HC4-U2 both improved aggregates but missed independent local gates,
the unified-BC correction line is closed pending a new architectural
hypothesis. The next recommended obstacle track measures compact-observation
range-noise sensitivity inside each accepted specialist envelope before any
new training.

O3a now begins with a bounded, paired HC4-LH pre-screen rather than another
training candidate. It perturbs only compact range by replayable uniform error
in `[-0.02, +0.02]` m while exact simulator geometry continues to determine
visibility, collision, passage, and termination. The magnitude is a
provisional simulation stress level, not measured sensor evidence, so even a
pass cannot complete O3a. Seed 271 compares exact and noisy conditions at
0.50 m/s, 1.15 m forward, and lateral positions -0.08/0.00/+0.08 m under the
HC4-E1 fixed-attempt denominator. See
`experiments/2026-09-05-o3a-range-noise-prescreen.md`.

That HC4-LH pre-screen passed at seed 271. The exact matrix recorded 191 clean
passages and one collision; the paired range-noise matrix recorded 192 clean
passages and zero collisions, with no timeouts or hard failures and all frozen
phase-speed and motor gates satisfied. This is one-seed simulation sensitivity
evidence, not proof that noise improves control and not measured sensor
acceptance. Only a separately predeclared HC4-R2 near-range screen may follow.
That screen is now frozen at seed 277 across HC4-R2's accepted 0.30/0.40 m/s,
0.90 m forward, and -0.08/0.00/+0.08 m lateral cells, using the same paired
exact-versus-two-centimeter protocol and numerical gates.
HC4-R2's seed-277 exact and noisy matrices each completed all 384 attempts as
clean passages. All phase-speed and motor gates passed, so two additional
predeclared seeds may now establish or reject multi-seed sensitivity for both
accepted specialists. The range bound is still provisional simulation stress,
not a measured sensor specification.
Seeds 281 and 283 are the only frozen continuation seeds. Each specialist runs
exact baseline, noisy condition, and deterministic comparison sequentially;
any local fail stops that specialist. A three-seed pass establishes only the
provisional two-centimeter simulation envelope, after which measured sensor
calibration—not another invented perturbation—is the next O3a gate.

The September 6 continuation completed HC4-LH's three seed gates, but HC4-R2
seed 281 added one centered collision at 0.30 x 0.90 m under range noise.
Its seed 283 was not launched and the two-specialist O3a campaign is not
accepted. The exact-geometry specialists remain retained. HC4-U3 separately
tests a new architecture over HC4-U2's unchanged exact-geometry corpus: three
independent approach/interaction/recovery experts using the existing phase
observation. This is a single training-seed diagnostic with fresh held-out
seeds 293/307/311 and unchanged local outcome, speed, and motor gates. See
`experiments/2026-09-06-hc4u3-phase-separated-supervisor.md`.

The independent hop branch now follows the canonical H0--H3 naming above. H0
source/configuration checks and the first retained matched-arm runtime campaign
are complete. In the fixed seed-43, 256-iteration diagnostic, Locked ended at
0.0010 m peak rise, K2500 at 0.0543 m, and K3900 at 0.0695 m. K3900 combined
the largest peak with only 0.034% bottoming, versus 0.858% for K2500. This is a
real compliance-dependent learning signal, but the interval fall counters are
still high and no policy has passed H1.

K3900 is therefore the sole next H1 training candidate. The deterministic,
reset-aware `H1-periodic-hop-heldout-v1` evaluator is now implemented with
explicit hop-cycle, rise, landing, stability, spring, action, and motor metrics.
Its fixed matrix uses seeds 211/223/227, 128 first episodes per seed, and six
cycles per episode; automatic-reset states are excluded. The runtime smoke
correctly rejected the iteration-255 pilot: all 384 first episodes landed one
qualifying launch and then fell, with zero completed cycles and material motor
envelope violations. The next campaign is now fixed at K3900 training seeds
47/53/59, 256 environments, 8,000 iterations, and common checkpoints every 500
iterations; only the earliest iteration passing every held-out gate across all
three policies may advance. See
`experiments/2026-09-04-h0-h1-hop-campaign.md`.

That complete 16-iteration by three-training-seed matrix has now finished. None
of its 48 checkpoints passed across all training seeds, so K3900 H1 is rejected
and Locked, H2, and hop video remain gated. Mature checkpoints learned the
cycle, but still failed whole-episode survival, drift, rated-speed, near-stall,
and commonly fall/bottoming gates. The next predeclared H1-P revision adds the
already-tested motor-torque cost and a time-staged 20/30/40 mm height envelope;
it preserves the task mechanics and held-out evaluator. See
`experiments/2026-09-04-h0-h1-hop-campaign.md`.

The bounded H1-P seed-67 diagnostic also completed and was rejected at every
predeclared checkpoint. Its final checkpoint improved motor and spring metrics,
but held-out episode pass remained zero with 23 falls and 0.594 m drift p95.
The fixed stop condition therefore closes further H1-P GPU work: do not run a
multi-seed promotion, Locked control, H2, or video. Any next hop revision must
first diagnose lateral stability and fall causality in a separately reviewed
source/design slice.

That audit found the inherited planar-stillness reward was dead: the periodic
phase command has unit magnitude, but the reward pays only below command norm
0.01. With velocity tracking correctly removed, H1/H1-P had no direct planar-
velocity objective despite the evaluator's drift gate. Source-only H1-S swaps
that dead gate for the same always-active Gaussian stillness shape without
changing observations, mechanics, curricula, or acceptance gates. No H1-S GPU
run is authorized by the source slice. See the H0/H1 experiment document.
