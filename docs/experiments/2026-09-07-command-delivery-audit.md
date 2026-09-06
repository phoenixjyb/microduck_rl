# Command delivery and play-noise audit

## Scope and finding

Read-only continuation of the [actor input audit](2026-09-07-actor-input-and-speed-contract-audit.md),
at source `a31ad149a535faf06fa92f457f0467200c256968` on the exact feature
branch. No simulator or optimizer was launched. Historical numerical rejections
remain closed; this audit cannot admit recovery-only PPO.

**A concrete source-level timing discrepancy is reproduced:** the hierarchical
rollout writes a new supervisor twist before policy inference, but passes the
actor observation returned by the previous environment step. That observation
contains a cloned/concatenated copy of the previous twist, not a live view of
the command manager. The new command normally reaches actor input on the next
50 Hz step: a **20 ms lag on a changed command**, within the 10 Hz supervisor
cycle. The representative trace records the newly issued command; it must not
be interpreted as proof that this command was seen by that same policy call.
The recording script has the same source ordering. This is a source/CPU timing
finding, not a measurement of its behavioral impact in simulation.

The installed environment computes commands before observations at the end of
each step. The constant-command straight diagnostic uses these returned
observations without changing the command between steps. A CPU test confirms
that its command input stays correct. Therefore the timing discrepancy does
**not explain the sustained .30 → .21 m/s deficit** in the closed straight
control, nor clear that prerequisite by itself.

## Installed implementation and counterexample

Inspection and tests use the real installed `ObservationManager` with synthetic
CPU sensor tensors, and the real `generated_commands` function. No replacement
observation-manager implementation is mocked in. The manager:

1. clones each term and concatenates actor inputs;
2. returns its existing cache from `compute()` / wrapper `get_observations()`;
3. refreshes the command copy during the next `compute(update_history=True)`.

The CPU counterexample starts with command `[.3, 0, 0]`, builds observations,
then writes `[.1, 0, -.4]` into the command manager. The existing actor tensor
and `compute()` still contain `[.3, 0, 0]`; the next step refresh contains the
new command. No claim of matching observed robot response is made.

Calling `compute_group("actor", update_history=False)` is **not a safe fix**:
it still samples fresh noise and unconditionally appends to delay buffers.
A second CPU counterexample shows one-step-delayed joint velocity advancing
again during that extra same-step call, and CPU RNG changing. Recomputing all
observations would change the sensor protocol as well as command timing.

Installed source SHA-256 values match on the Mac and 100.100:

| Installed file under `mjlab` | SHA-256 |
| --- | --- |
| `managers/observation_manager.py` | `db9feb383786b4b144efc8cc71c9a9804b6611b3d148b6ceed283cd66eef53a8` |
| `rl/vecenv_wrapper.py` | `d458aa421d72c979d0269a24e51a4b49a02e4ec8cb5a9d10e47290aa5d86a3e2` |
| `envs/manager_based_rl_env.py` | `a381027e336d6313cd338d541230b36864657edac89708a33e7e60c0c6fb74d2` |

## Candidate primitive, not a rollout switch

`command_delivery.fresh_actor_twist` is added as an **unwired** opt-in primitive
for a future distinct `frozen-actor-fresh-twist-v1` protocol. It accepts the
raw, pre-normalizer actor tensor and an already bounded command. It verifies
the exact 61D layout and an unscaled, unclipped, noise-free, undelayed twist
term, then returns an independent clone with only columns 48:51 updated.
All other sensor values, manager cache, command tensor and CPU RNG are preserved.
The existing actor still owns normalization; passing normalized values is not
supported. This primitive is not a command limiter or an authority grant.

No call site is changed in the hierarchical evaluator, recording script,
straight control, historical diagnostics or PPO training. No dataset/report
is relabeled. Wiring the primitive changes an evaluation protocol and must
receive a new identity with explicit issued-versus-actor-consumed command
evidence. It cannot be used to rerun a rejected historical cell for acceptance.

## Play noise remains deliberate and separate

Current play actor corruption is enabled. Uniform additive ranges are angular
velocity ±.03, gravity ±.01, joint position ±.001, joint velocity ±.25. Twist,
head and body commands have no observation noise. The IMU term parameters set
a **6 degree** maximum mounting rotation, not the helper's 1 degree default.
Gravity and angular velocity use the same per-environment quaternion, sampled
once and cached for the environment lifetime. Tests verify shared bias and no
new RNG draw on subsequent calls. This is separate from per-step observation
noise and sensor-delay sampling. Neither noise removal nor IMU-range changes
were made; no result here assigns causality for yaw drift or speed deficit.

## Validation and next bounded handoff

The first 18 CPU tests passed locally (6.49 s): real-manager stale-command and
constant-command controls; unsafe whole-group refresh; command-only clone
isolation; layout/transform/dtype/shape/nonfinite guards; configured play noise
and shared fixed IMU bias. These are source-contract tests, not GPU acceptance.

Next inspect the supervisor-training action/observation ordering and define a
single opt-in command-delivery contract shared by future training/evaluation,
with a focused lifecycle test before any GPU request. Preserve existing default
behavior and protocol identities. A future bounded timing-only simulation must
be separately predeclared as diagnosis, with both issued and actor-consumed
command traces and unchanged physical/motor/speed gates. It cannot clear the
constant-command low-speed deficit, admit PPO, or revive historical failures.
Do not substitute low-level locomotion retraining without explicit agreement.
Keep source/CPU work inside the 07:00 Shanghai deadline; do not expand GPU work
merely to keep the device busy. Keep both protected system services inactive.

Pre-commit regression: **591 focused CPU tests passed locally** (8.79 s), with
no skips and two existing actuator/site-pattern warnings. One SSH probe timed
out; a bounded retry succeeded, verified clean `a31ad14`, no compute process,
and the matching runtime hashes above. No service or network setting changed.

## Training-loop audit and shared future interface

At `2f5d2d245b23601ab39fdbbeb1157527d4d3fe43`, inspection of
`obstacle_supervisor_ppo.train_hc3_supervisor` confirms the same ordering:
write the supervisor command, then call the frozen locomotion policy on the
previous observation before each five-step low-level block. It also writes
per-cell nominal commands after reset observations have already been obtained;
`get_observations()` returns the cache and does not refresh those command rows.
This is not an evaluation-only discrepancy. Its learning/reward impact has
not been measured and no old checkpoint is relabeled.

`prepare_actor_command_input` now provides the **unwired** common boundary for
future training and evaluation. The caller must select an explicit protocol:

- `frozen-actor-cached-twist-v1`: retain the cached raw actor command;
- `frozen-actor-fresh-twist-v1`: replace only the raw twist slice with the
  already bounded issued command, before normalizer/inference.

Both modes clone the input TensorDict and retain independent issued, cached
and actor-input command snapshots. Reports contain row-wise input differences
and a control-step index. They explicitly label the boundary as **prepared raw
input**, with `actor_inference_executed=false` and `simulation_executed=false`:
preparing a tensor does not prove a policy call or dynamics step occurred.
Later command resets or caller edits cannot rewrite the separately cloned
snapshot. Sensors, command limits, normalization, rewards and motor authority
are unchanged. The adapter does not call the actor, environment or optimizer.

`require_matching_delivery` rejects missing/unknown identifiers and a changed
training/evaluation timing pair. Matching identifiers establish compatibility
only, not speed tracking or policy admission. Future adopters must persist the
identifier in both training manifests/checkpoints and evaluation reports.
Historical artifacts lacking this metadata cannot be assigned a new identifier
by guesswork; a source-linked audit is needed, without mutating originals.

CPU lifecycle coverage uses two independent real observation managers with
synthetic sensors and the same command schedule: a flat evaluator loop and a
two-block, five-step training loop. Cached timing differs from the issued
command at steps 0 and 5 for both environments, and step 3 for the reset row;
fresh timing has no such input mismatch. Each loop emits identical command
snapshots for its chosen protocol. This is a synthetic lifecycle test, **not**
execution of the complete trainer/evaluator or a trained robot in simulation.

### Superseding handoff

The shared input contract is source-tested but still has **no call sites** in
historical evaluation, recording or PPO entrypoints. Next, if useful, implement
one isolated, bounded timing-diagnostic entrypoint with a separate protocol and
full lifecycle tests. Keep historical defaults untouched. Any GPU run requires
a predeclared distinct question beyond what these CPU tests already prove,
new run identity, input snapshots tied to actual inference/step completion,
the frozen checkpoint, unchanged gates, an idle GPU and a hard deadline margin.
Do not run a timing-only GPU experiment merely to repeat this source finding.
It cannot admit recovery PPO or explain away the constant-speed deficit. The
separate low-speed locomotion training scope still requires explicit agreement.

Shared-boundary closeout: **608 focused CPU tests passed locally** (8.39 s),
including 35 command-delivery cases. No skips; two pre-existing actuator/site
warnings remain. No existing entrypoint imports the new adapter, and no GPU
job, optimizer step, checkpoint change or protocol-default change occurred.
