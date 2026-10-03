# Scheduled recovery PPO transition bridge

This is a composition adapter and integration harness, **not an admitted trainer
or new learned skill**. It uses the exact frozen seed577 iteration255 parent from
the retained nominal replication. Historical fresh-model seeds/checkpoint
allowlists, the original24-step adapter, package/driver pins and actor observations
remain unchanged. No raw perception, video or physical motion is involved.

## Implemented source contract

`stance_recovery_ppo_bridge.RecoveryPPOBridge` loads the actual parent through
`prepare_policy`. Proposed learner seeds653/659 and two or64 worlds are separate
from the historical parent identity. Held-out schedules are refused by the
learner; a training row keeps its fixed schedule across episode-clock resets.

The new rollout horizon is28 transitions (0.56 simulated seconds), not24
(0.48 s). A20-physics-step pulse at step250 finishes at step270 and is fully inside
the new horizon. Two-world and64-world buffers both divide exactly into four
minibatches, avoiding dropped remainder samples. This horizon does not span later
onsets or a full five-second episode; those need separate qualification.

The adapter constructs fresh real storage/Adam over the same restored actor and
critic. A private CPU RNG scope covers both stochastic action sampling and the
entire minibatch update. It restores caller RNG even on sampler/update errors.
This is not CUDA RNG qualification or resumable state persistence.

The transition path calls only `step_with_schedule`. PPO stores the raw sampled
action, corresponding log-probability, value and pre-action observations. Motor
clipping/slew/delay are separate runtime effects. Timeout rewards bootstrap from
the returned terminal observation exactly once; stock `time_outs` extras are
omitted. True failures receive no timeout bootstrap.

Terminal records must be present exactly for done rows and are copied before
storage/reset. Selective reset must return those records, restart only the done
episode clocks, retain the fixed row schedule, restore motor/control state and
preserve untouched row clocks, controls and native qpos/qvel/time. Synthetic
fixtures require explicit opt-in and a fixture label; they do not establish those
native physics claims.

One update maximum,28 transitions, five epochs and four minibatches yield exactly
20 Adam steps. Configuration, parent model keys/shapes/dtypes, positive Gaussian
scale, finite parameters, complete finite gradients and every Adam state are
checked. Errors fault the composition; it offers no retry, resume, CLI, export or
job-admission method. Synthetic optimizer changes are not a new student checkpoint.

## Checks and remaining native gates

Local bridge + existing preparation/PPO tests: **63 passed in8.69 s** with
`CUDA_VISIBLE_DEVICES=''`. They include paired private-RNG reproducibility, full
minibatch RNG isolation, raw-vs-clipped actions, terminal-value-vs-pre-step-value
timeout bootstrap, partial resets, 28-vs24 horizon, finite/invalid optimizer paths,
held-out refusal and immutable model schema. No native bridge run has occurred.
Ruff was unavailable; no package installation was performed.

Before native collection, predeclare an exact source and capped CUDA-hidden CPU
service, authentic parent bytes, frozen CPU math profile, two fresh native rows
(zero and early +x pulse), learner seed653,28 transitions, no optimizer update,
and retained whole control/force/terminal/storage evidence. Check private and
caller RNG, exact raw action storage, complete20-step pulse delivery/cleanup and
fresh fixed approach clocks. Stochastic rows have different sampled actions;
matched frozen-parent prefixes belong to the separate unchanged-parent screens.
This is a transition gate, not a deficit screen.

A full native timeout/selective-reset check is still needed separately. Do not
shortcut it by moving episode counters, replacing a real result with a synthetic
terminal, discarding full storage through private mutations, or relabeling the
28-step harness as a250-tick trainer. A later adapter/run must explicitly declare
its longer rollout/update limits and evidence protocol before execution.

Real curriculum training still requires a repeatable unchanged-parent deficit on
predeclared training cells, untouched held-out acceptance criteria, complete
source-specific transition/finite-optimizer/native reset gates, nominal regression,
and measured CUDA throughput/memory/cutoff reserves. The small CPU dose screen
found no deficit, so it does not authorize training those already-passing cells.
