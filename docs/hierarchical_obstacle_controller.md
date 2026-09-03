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
  clearance, bounded progress, and command smoothness. Timeout still fails the
  attempt.
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

## Next bounded implementation slice

1. Run HC0 in simulation against the frozen motor-aware checkpoint using a
   fixed command matrix: forward commands `0.0, 0.15, 0.30` m/s crossed with
   yaw commands `-0.6, -0.3, 0.0, 0.3, 0.6` rad/s.
2. Reject command cells with falls, non-finite state, unstable tracking, or
   motor-envelope violations; the surviving envelope becomes a hard clamp.
3. Implement and unit-test the two-output supervisor action adapter with the
   frozen actor hash pinned.
4. Validate a deterministic OA0 teacher before training a supervisor.

No GPU supervisor training starts until HC0 and the adapter tests pass.
