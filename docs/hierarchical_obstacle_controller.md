# Hierarchical obstacle controller contract

## Why the architecture changes

OA0, OA0R, and OA0P all updated one policy from compact obstacle observations
directly to 14 joint targets. Pass rate and approach speed regressed with
training even when speed rewards were active on approach. The obstacle learner
was overwriting a gait that already worked.

Obstacle negotiation is therefore split into two time scales and two authority
levels. The accepted motor-aware locomotion actor is frozen. A new supervisor
may modify only velocity commands inside measured bounds; it cannot emit joint
targets or alter motor limits.

```text
external geometry + route state
            |
            v
  obstacle supervisor (10 Hz)
    [speed scale, yaw rate]
            |
      clamp / fail-safe stop
            |
            v
 frozen locomotion actor (50 Hz)
       14 joint targets
```

Perception remains separate. Training receives the same externally supplied
range, bearing, width, height, closing-rate, and validity fields used by the
existing observation contract; no raw camera pixels enter RL.

## Supervisor interface

Inputs are normalized, timestamped values:

- nominal forward command;
- obstacle range, bearing, width, height, closing rate, and validity;
- route lateral error and route heading error;
- estimated forward speed;
- previous supervisor action.
- execution-layer maneuver phase (`approach`, `interaction`, or `recovery`);
- the latched bypass side chosen from obstacle bearing (with a deterministic
  tie-break for a centered obstacle).

The phase and side are explicit because a memoryless actor cannot safely infer
whether a temporarily invalid obstacle is already behind it, nor choose a
repeatable side for a perfectly centered obstacle. They are route-controller
state, not perception features or simulator entity identifiers. The resulting
HC2 supervisor observation is 17-dimensional.

The first bypass supervisor has two outputs:

- forward-speed scale in `[0, 1]`;
- yaw-rate command clamped to the frozen locomotion policy's validated range.

Lateral velocity stays zero until the base locomotion policy has independent
evidence for lateral-command tracking. Invalid or stale obstacle input selects
the fail-safe stop path, outside learned policy authority.

## Phase rule

- **Approach:** pass through the nominal speed command and penalize tracking
  error.
- **Interaction:** permit braking and any lower forward speed; optimize
  clearance, bounded progress, and command smoothness. There is no nominal
  speed-tracking objective in this phase. The controller may keep speed when
  the measured command envelope permits it, but is never penalized for slowing
  to avoid contact. Timeout still fails the attempt.
- **Recovery:** converge to the original route and restore nominal speed.
  Success requires route error within 0.15 m and speed at least 0.22 m/s for a
  retained 0.5-second window, rather than terminating at the first crossing.

## Curriculum and gates

| Stage | Change | Required evidence |
| --- | --- | --- |
| HC0 | frozen-policy command matrix, no obstacle | stability, speed/yaw response, motor envelope |
| HC1 | deterministic teacher on OA0 geometry | collision-free traces and command bounds |
| HC2 | imitate teacher; frozen base actor | >=85% OA0 passes across three seeds |
| HC3 | RL fine-tune supervisor only | improve HC2 without locomotion regression |
| HC4 | reduce lateral offset toward center | one geometry band per stage |
| HC5 | vary range, then width/height | per-bin pass gates |
| HC6 | add measured sensor noise, lag, dropout | bounded degradation and fail-safe stop |

Each stage retains checkpoint/source identity, independent training and
held-out seeds, collisions, timeouts, falls, non-finite states, approach speed,
interaction speed, 0.5-second recovery speed, route error, command slew, and
the existing torque/power/thermal envelope. MP4s are recorded only after the
numeric gate has a survivor.

## HC0 measured command envelope

HC0 was measured against the frozen motor-aware `model_7998.pt` checkpoint
(SHA-256 `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`).
The evaluator records the actual command tensor as well as the requested
command, preventing heading or forward-only sampling overrides from being
mistaken for policy response.

- At 0.30 m/s, requested yaw `-0.6, 0, +0.6` rad/s produced mean observed yaw
  `-0.582, -0.058, +0.511` rad/s with no falls, NaNs, or motor-speed
  exceedance. Torque-utilization p99 stayed at or below 0.535.
- At 0.50 m/s, yaw response weakens but remains useful; torque-utilization p99
  stayed at or below 0.720 and near-stall exposure below 0.073%.
- At 0.80 m/s, hard yaw is not an interaction command: observed yaw undertracks
  and torque-utilization p99 reaches about 0.928. Straight 0.80 m/s remains an
  approach/recovery command inside the previously accepted locomotion envelope.
- In-place `+/-0.3` rad/s is not a usable turn command for this checkpoint.

The first supervisor therefore clamps interaction commands to 0.30 m/s and
`+/-0.6` rad/s. Nominal approach/recovery commands may range to 0.80 m/s. A
3-second time-to-contact trigger, computed from the external range and closing
rate fields, begins braking and steering earlier at high approach speeds.

## HC1 and HC2 retained results

The three-seed HC1 matrix at forward placements 1.15 and 1.40 m retained
1,147 clean passes, 15 collisions, and 16 timeouts across 1,178 resolved
attempts (97.37% pooled). Successful teacher data came only from these cells:

- 0.30 m/s at 1.15 m: zero collisions and one timeout across three seeds;
- 0.50 m/s at 1.15 and 1.40 m: 100% clean passes across three seeds;
- 0.80 m/s at 1.40 m: 100% clean passes across three seeds.

Successful episodes from those cells produced 69,441 HC2 samples from 765
episodes. The 17D behavioral-cloning supervisor reached held-out mean absolute
errors of 0.0039 m/s for forward speed and 0.0226 rad/s for yaw. This was an
offline gate only.

Closed-loop HC2 evaluation across the same four cells and three seeds retained
759 clean passes, one collision, and eight timeouts across 768 resolved
attempts (98.83%). There were zero falls, NaNs, non-finite steps, or rated motor
speed exceedances. The highest per-cell torque-utilization p99 was about 0.745;
the highest near-stall exposure was about 0.20%.

HC2 is accepted only as a simulation supervisor for this bounded envelope. It
is not accepted for 0.8 m/s at 1.15 m, the 0.90 m near-obstacle stress case,
general O1/O2, sensor degradation, physical motion, or the 4.5-second O1
passage-time target. HC3 should improve passage time and residual failures
without changing the frozen gait or reinstating interaction speed pressure.

## Replay matrix contract

Numerical evaluation precedes video. The first centered-box matrix crosses
nominal speeds `0.30, 0.50, 0.80` m/s with obstacle forward positions
`0.90, 1.15, 1.40` m. Representative survivors are then recorded from an
external tracking camera at 960x540, with both the duck and obstacle visible
during negotiation. Filenames contain nominal speed and exact obstacle
position, and each MP4 has a JSON sidecar containing checkpoint hash, teacher
configuration, phase speeds, outcome events, and physical-motion authority
set to false.

## Next bounded implementation slice

1. [x] Measure HC0 against the frozen motor-aware checkpoint and retain actual
   applied-command, tracking, stability, and motor evidence.
2. [x] Implement and unit-test the stateful, two-output deterministic teacher
   while keeping the 61D locomotion actor frozen.
3. [x] Complete the HC1 speed-by-position matrix across held-out seeds and
   retain only successful, command-bounded teacher episodes for imitation.
4. [x] Train the 17D HC2 supervisor by imitation on accepted HC1 trajectories,
   then evaluate it without teacher intervention.
5. [x] Record representative numeric survivors with duck and obstacle visible.
6. [ ] Accept an HC3 fine-tune on passage time and residual failure outcomes
   without allowing interaction speed tracking to fight collision avoidance.
   HC3-A/B/C were implemented and measured but rejected for collision/timeout
   regression.
7. [ ] Expand placement only after HC3 retains the accepted HC2 envelope.

MP4 inspection never substitutes for the numeric matrix gate.

No GPU supervisor training starts until HC0 and the adapter tests pass.
