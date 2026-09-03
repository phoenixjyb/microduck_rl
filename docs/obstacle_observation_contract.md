# Obstacle observation contract v1

The obstacle-avoidance locomotion policy does **not** consume camera images,
depth maps, or detector features. Perception is a separate component that
supplies one compact estimate for the nearest relevant obstacle. This keeps
raw-camera perception outside locomotion RL and gives simulation and the robot
the same stable policy interface.

## Actor input

The actor receives seven normalized values in this exact order:

| Index | Field | Definition | Range |
| ---: | --- | --- | ---: |
| 0 | `range` | forward surface range / 2.0 m | [0, 1] |
| 1 | `bearing_sin` | sine of base-frame bearing | [-1, 1] |
| 2 | `bearing_cos` | cosine of base-frame bearing | [-1, 1] |
| 3 | `width` | obstacle width / 0.50 m | [0, 1] |
| 4 | `height` | obstacle height / 0.25 m | [0, 1] |
| 5 | `closing_rate` | closing rate / 2.0 m/s; positive means approaching | [-1, 1] |
| 6 | `valid` | 1 for a current estimate, otherwise 0 | {0, 1} |

Sine/cosine bearing avoids a discontinuity at plus/minus pi. Values outside
the physical envelope are clipped. If the estimate is absent or contains a
non-finite field, every channel is zero; `valid` distinguishes that state from
an obstacle at the origin.

The canonical implementation is
`mjlab_microduck.tasks.obstacle_observation.encode_obstacle_observation`.

For simulation, `encode_relative_obstacle_observation` deterministically maps
relative position and velocity in the robot base frame into the same layout.
Its conservative surface-range approximation treats `width / 2` as a planar
footprint radius. A later scene adapter remains responsible for transforming
world state into the base frame and selecting the nearest relevant obstacle.

`mjlab_microduck.tasks.mdp.obstacle_geometry_observation` is the first scene
adapter. It reads a named simulator entity, transforms its position and
velocity into the robot base frame, and masks it by range and horizontal field
of view. It is intentionally not registered in an actor configuration until a
tested obstacle entity and sensor-perturbation model are present.

`ObstacleSensorModel` and `encode_perturbed_obstacle_observation` provide the
sensor-perturbation boundary. They apply bounded symmetric noise independently
to the five physical fields and whole-estimate dropout before normalization.
Their defaults are exact and dropout-free: task-specific values must be
explicitly justified by perception measurements or a documented stress test.
Optional noise and dropout samples make retained evaluations reproducible.
The simulator scene adapter exposes these bounds as explicit parameters and
applies them before normalization; its default path remains exact and does not
consume random numbers.

Observation latency is not implemented inside this encoder. The eventual
actor term must use MJLab's reset-aware observation delay buffer; this avoids
state leaking across environment resets.

## Training boundary

Simulation should derive the physical fields from ground-truth geometry, then
apply an explicit sensor model before encoding the actor observation. That
sensor model will independently randomize measurement noise, latency, dropout,
and limited field of view. It must not expose future state or simulator-only
identifiers to the actor.

An asymmetric critic may additionally receive unperturbed ground-truth
geometry. The deployed actor must receive only the same v1 estimate that the
external perception stack can supply. Perception model training and validation
remain a separate project and acceptance gate.

## Next integration gate

The first entity is a 100 x 200 x 100 mm, 2 kg free-moving box described by
`robot/microduck/obstacle.xml`. `reset_obstacle_ahead` places it in the robot's
reset-yaw frame and clears its velocity. Its 200 mm lateral dimension matches
the width passed to the observation adapter.

Before obstacle-policy training, register a short flat-terrain curriculum that
starts with this single stationary box and adds sensor perturbations only in
later stages. Run a CPU-only configuration smoke test first. GPU training
starts only after those checks pass and after an idle-GPU gate.

`Mjlab-Run-Obstacle-Flat-MicroDuck` now provides that configuration. It stages
placement from 1.0-1.3 m down to 0.6-1.0 m, command speed from 0.5 to 0.8 m/s,
and sensor perturbations from exact measurements to the documented stress
envelope. The actor receives delayed/perturbed observations; the asymmetric
critic receives exact ground truth. Play mode remains deterministic.

The first task commands straight forward motion only: lateral and yaw commands
are zero, so avoidance cannot be confused with an unrelated turning command.
A conservative planar envelope uses a 0.12 m robot radius and 0.10 m obstacle
radius. Clearance cost begins 0.15 m outside that envelope, collision incurs a
penalty and terminates the episode, and clean passage is measured along the
fixed reset heading. Logged metrics include mean clearance, resolved attempts,
clean-pass and collision rates, and forward-speed diagnostics.

The first GPU pilots were gated on a reviewed warm-start migration because
adding the seven obstacle channels changes the first actor layer from the
retained Stage 2 checkpoint. Every new campaign must still copy the old columns
exactly, initialize only the seven new columns, and prove numerically equivalent
outputs within floating-point tolerance when those channels are zero.

`python -m mjlab_microduck.checkpoint_migration SOURCE DESTINATION` performs
that migration without overwriting either file. It expands actor 61→68 and
critic 76→83, preserves every old first-layer column, initializes only new
columns to zero, gives new normalizer channels mean 0 and variance/std 1, and
expands Adam moments with zero new columns. The output records the source
SHA-256 and migration contract under `infos.obstacle_warm_start`.

`python -m mjlab_microduck.obstacle_smoke CHECKPOINT OUTPUT_DIR` is the bounded
runtime gate. It disables external model uploads, allows at most 256 simulation
environments and two learning iterations, strict-loads actor/critic/optimizer,
and retains a smoke checkpoint. Passing this gate proves runtime compatibility;
it is not evidence that obstacle avoidance has been learned.

`python -m mjlab_microduck.obstacle_baseline CHECKPOINT OUTPUT_DIR` runs a
bounded, inference-only baseline (at most 256 environments, 1,000 steps, and
five seeds). It fixes the command to straight 0.5 m/s, disables actor sensor
noise through the deterministic play config, and retains collision, fall,
timeout, clean-pass, non-finite, clearance, and forward-speed evidence. Its
output explicitly remains untrained-policy evidence.

`python -m mjlab_microduck.obstacle_checkpoint_sweep MANIFEST OUTPUT_DIR`
compares retained O1 evaluations without running simulation. The manifest names
each candidate's training seed, checkpoint iteration, and evaluation JSON
explicitly; paths alone never decide provenance. The report checks the O1
obstacle gates, compares exact iterations shared across training seeds, and
selects the earliest survivor rather than the final checkpoint. It is always
marked `diagnostic-only` until motor-envelope and action-regression evidence are
combined by the next acceptance step.
