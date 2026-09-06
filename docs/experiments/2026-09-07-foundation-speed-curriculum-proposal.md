# Foundation speed curriculum proposal

**Status: design only; not an approved or launch-ready training experiment.**
This proposal is within the overnight curriculum-design scope. Updating the
locomotion actor is outside the currently frozen-gait recovery-PPO scope and
requires explicit agreement. No new GPU job, model, video or physical motion is
authorized by this document. Deadline remains September 7, 07:00 Shanghai.

## Why the curriculum needs a prerequisite

The closed [recovery A/B](2026-09-06-overnight-recovery-curriculum.md) produced
eight clean obstacle attempts but zero qualifying nominal-speed recoveries.
The distinct [straight control](2026-09-07-straight-speed-response.md) measured
about .210 m/s body-forward and .185 m/s route-forward at a .300 m/s command,
without obstacles or a supervisor. Historical HC0 measured .211 m/s for the
same nominal command. Strict CPU actor/normalizer restoration passed.

The [command-delivery audit](2026-09-07-command-delivery-audit.md) found a
20 ms changed-command lag and implemented an unwired input adapter. The lag
cannot explain the sustained deficit at a constant command. CPU tests already
prove the input-timing discrepancy; a GPU test solely repeating that fact is
not justified. No timing-only result could certify nominal-speed recovery.

Consequently, avoiding contact, restoring the route, tracking actual speed,
and staying within motor limits remain separate capabilities. Success at one
must not be substituted for the others. This proposal does not reopen any
historical rejection or reinterpret its command labels as measured speeds.

## Observed reward contract, hypothesis and uncertainty

Saved Stage-2 configuration used linear tracking weight 4.5 and
`std=sqrt(.15)`. The installed reward is
`exp(-((command_x-vx)^2 + (command_y-vy)^2 + vz^2) / std^2)` in body coordinates.
With lateral and vertical speed exactly zero, the .30 m/s target gives:

| Body-forward speed | Normalized linear-tracking reward | Meets ±.03 m/s? |
| --- | --- | --- |
| .21 m/s | .9474321065 | No |
| .27 m/s | .9940179641 | At the lower boundary |
| .30 m/s | 1.0000000000 | Yes |

These are verified reward-function calculations, **not measured episode
rewards**. Real lateral/vertical motion reduces the value; averaging velocity
before a nonlinear reward does not recover the mean reward. The total
objective also includes posture, action smoothness, gait and motor terms.
The motor-load curriculum increases its penalty from -.25 to -2.0; this alone
does not prove that motor cost caused undertracking. Configured .30 command
coverage existed, but realized occupancy and learning balance are not established.

Hypothesis worth testing after approval: focused low-speed command practice
may improve tracking under the unchanged motor objective. If that fails, a
separate predeclared tracking-reward revision can test sensitivity. Do not
change command distribution, reward width and motor limits together and then
claim which change worked. Do not assume that .30 m/s is physically achievable
under every retained motor/domain setting before observing it.

The 61D actor has angular/gravity and joint/action observations plus commands,
not an explicit measured base-linear-velocity or absolute route-heading input.
Improved training may reduce bias without guaranteeing closed-loop speed or
long-horizon route tracking. Keep this observability limitation visible; do
not silently expand actor inputs or claim a yaw-rate command holds an absolute
heading. Any such architecture change requires a separate agreed revision.

## Smallest proposed progression

| Stage | One learning question | Entry/exit condition |
| --- | --- | --- |
| F0: input contract | Are issued commands the raw inputs actually used by the actor? | Source tests pass; future integration must retain prepared, inference-completed and step-completed evidence separately. No policy acceptance from these checks. |
| F1: .30 straight foundation | Can a separately authorized derived locomotion actor track .30 with zero yaw, no obstacle and the same motor/domain constraints? | New experiment with frozen reward coefficients first; per-environment body and route tracking, finite/stable episodes and motor gates all required. |
| F2: bounded speed transitions | Does a common accepted actor retain .30 and handle neighboring speeds, stops and recovery transitions? | Add one speed/transition band at a time. No automatic expansion to .5–.8; those old commands exceeded today's .60 torque gate. |
| F3: obstacle integration | Can a frozen accepted foundation actor and bounded supervisor negotiate then restore route and nominal speed? | New joint-system campaign, new identities/seeds and matched comparisons; old supervisor acceptance does not transfer automatically to a changed gait. |
| F4: perception-contract robustness | Does accepted locomotion/avoidance tolerate structured-observation noise, latency and dropout? | Separate perturbations and per-bin gates; perception remains external, not raw-camera RL. |

F1 first changes the training command distribution to a fixed .30 forward,
zero lateral/yaw, zero head/body target task. Retain the actor's 61D observation
definition, actuator model, motor limits, physics, domain randomization and
reward coefficients. This is an intentionally narrow task-distribution
revision, not an assumed general walking improvement. Do not concurrently
introduce the fresh supervisor-timing adapter into this constant-command task.
Any narrower reward width or new motor penalty is a later, separately declared
revision only after the first result is diagnosed. No selected seed or training
budget has been assigned: the proposal cannot be launched as-is.

F2 must include stand/stop retention before claiming a useful general controller.
Training performance at one fixed speed is not acceptance at other speeds or
under route/yaw changes. Promote a common checkpoint window across the
predeclared independent seeds, not each seed's independently cherry-picked best.
Do not train jumping/hopping as a remedy for these locomotion prerequisites.

## Gates to preserve and make explicit in the new experiment

- Preserve the .30 target and ±.03 tolerance; do not relabel .21 as success.
  Separate body-forward speed, initial-route projection, lateral motion and
  yaw drift. Report per-environment values as well as aggregates.
- Preserve the recovery criterion: a sampled .50 s in-band span completed
  within 2.0 s of entering recovery; at 50 Hz this requires 26 samples.
  This is distinct from settled mean speed. Missed/censored/unobserved windows
  are not successes. F1 must define its startup/settled windows before launch.
- Keep existing no-fall/no-collision/no-timeout/nonfinite/rated-speed gates and
  legacy pooled torque-utilization p99 ≤.60. Also retain the pre-reset motor
  stream, soft-limit exposure, power and squared-load proxy; do not call that
  proxy a physical temperature or replace the legacy statistic after the fact.
- Predeclare additional yaw/lateral bounds and retained-control nonregression
  tolerances before observing candidates. Until specified, F1 is not launch-ready.
- Keep every first-attempt terminal, including startup failures; do not count
  post-reset states as continuation of the evaluated episode.
- For F3, approach and recovery track nominal speed. During interaction, allow
  slowdown without nominal-speed tracking pressure; require safe clearance,
  bounded progress and timeout compliance. Returning to speed does not excuse
  a collision, and avoiding collision does not excuse failed recovery.
- A failed stage stops its ordered campaign. Retain evidence and diagnose;
  no retry with altered seeds/thresholds until something passes. A new actor
  would require a new campaign, not resumption of seed-379 A/B or its blocked PPO.

## Required before any launch

Explicitly agree to the separate F1 locomotion-update scope. Then predeclare:
exact task/config diffs and frozen source/model/runtime hashes; training and
held-out seeds; command/physics/domain distributions; optimizer and reward
settings; environment count, iterations and checkpoint cadence; shared
checkpoint-selection rules; full numerical gates; and the ordered failure-stop
policy. Specify what transfers from the parent checkpoint, including optimizer
and normalizer state, and test resume/curriculum step indexing.

Benchmark a separately bounded smoke before reserving a longer GPU window.
Every launch needs a clean exact branch, idle exclusive GPU, retained unique
output directory, hard service runtime and checkpoint/closeout margin before
the authorized deadline. Keep 100.98 and unrelated workloads untouched and
both protected AI Mission system services inactive. No physical duck exists;
all proposed acceptance remains simulation-only.

## Overnight handoff

This closes the immediate diagnosis/design chain without launching more GPU
work. There is no new numerical gate that clears recovery PPO. The useful next
training decision is whether to authorize the separate F1 locomotion scope.
Do not add more diagnostic wrappers or spend GPU time merely to fill the
remaining window. Read-only durability/service checks and concise final
closeout remain in scope. At/after 07:00 stop new Duck work, verify retained
state, leave protected services unchanged and delete the heartbeat as directed.

Validation: two new CPU tests verify the proposal's reward arithmetic against
the installed function, including nonzero lateral/vertical motion. The focused
regression suite passed **610 tests locally** (8.84 s), without skips; two
existing actuator/site-pattern warnings remain. Local document-link targets and
diff whitespace were checked. No production task or reward configuration changed.
