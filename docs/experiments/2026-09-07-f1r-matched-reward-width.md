# F1-R: matched reward-width diagnostic

Status: **predeclared; execution not yet verified**. Separately authorized by
the user's “sure go on” after the closed F1 pilot. This does not reopen that
pilot, restore its deleted automation or renew its one-hour work window.

## Question and causal limits

Does narrowing linear-velocity reward std from sqrt(.15) to sqrt(.05) improve
fixed .30 m/s tracking relative to an otherwise matched continuation?
Reward weight stays4.5. The function penalizes all three body velocity errors,
not just forward speed; extra balance-motion penalty is a risk, not a benefit
assumed in advance. One matched training seed supports a diagnostic only.

Both arms start from original model7998, SHA256
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`, not the
failed F1 candidate or either smoke. Use the same parent and strict actor,
critic, normalizer, exploration and Adam restoration implemented by F1.
Parent env step192000 and next update7999 are unchanged. Keep the61D actor,
physics, rigid feet, domain events/noise/pushes, motor weights/limits,
optimizer settings, fixed commands and completed curricula identical.
This is not physical hardware evidence; the user has no real Duck yet.

## Fixed budget and execution order

- Protocol: `f1r-width-paired-s421-v1`; branch `feat/athletics-obstacle-curriculum`.
- Worktree100.100: `/home/converge/work/microduck_rl-athletics-obstacle-curriculum`.
  Require clean pushed exact source, frozen dependencies/parent hash and idle
  GPU. Leave100.98 and unrelated workloads alone. Protected AI Mission
  services must remain inactive; never restore them here.
- Retained root: `artifacts/experiments/f1r-width-paired-s421-v1`.
  Existing output is a hard stop, not a resume/overwrite invitation.
- One sequential user service, hard runtime3300s; internal total3150s.
  Launch only before September7 11:00 Shanghai. Expected runtime roughly
  12–18minutes based on the preceding287.5s/500-update pilot, not a guarantee.
- First two smokes: control then narrow, both seed419,256env,10updates,
  hard180s each. Retain initial.pt and model8000/8008. Both must complete;
  each measured learning time×50×1.5 must be below870s before its pilot.
- Then control followed by narrow, both **seed421**,256env,500updates,
  hard900s each;24steps/env/update =3,072,000new samples per arm.
  Save every50 global labels8000,8050,...,8450 and final8498; final common
  step204000. Earlier checkpoints are durability evidence, never candidates.
- Initial restored checkpoints must match byte-for-byte between the two
  smokes and between the two pilots. No training from smoke or historical
  F1 candidate. Fresh same-seed resets reduce confounding but do not make
  different learned trajectories share subsequent noise or establish
  deterministic CUDA execution.
- Each GPU child is a separate process and exits before the next idle check.
  Training keeps all F1 raw finite checks and stochastic abort limits:
  rollout fall fraction>.50, pooled pre-reset torque p99>.85, rated-speed
  exceed fraction>.01, GPU temperature>=80C or protected service active.
  These are NOT deterministic evaluation acceptance limits.

## Fixed evaluation and decisions

Evaluate original parent at seeds431,433,439 first. Any safety/coverage
failure stops. Then for each seed in that order evaluate control followed
by narrow. Each case retains8 first episodes,400steps at50Hz,100startup and
300settled, no training pushes, constant(.30,0,0). Hard90s/case. At most
9cases/72episodes; report the actual prefix, including all failures.

Control undertracking/heading/lateral performance failures are retained as
the baseline and do not by themselves skip the treatment. A control safety
failure stops before narrow. A narrow failure stops the remaining seeds.

Keep ALL F1 numerical gates: startup included, no terminal/nonfinite/command
mutation, complete coverage, all/settled torque p99<=.60, zero rated-speed
exceedance; every body/route mean.30±.03, each stable26sample(.50s) route
window within2s; max heading<=.25rad, each mean absolute cross-route speed
<=.05m/s. Keep historical paired parent nonregression margins (.02 torque,
.01m/s cross-route mean, .10rad heading, .01m/s per-env speed error).
Additionally apply those same nonregression margins against the new matched
control. No reward change to the evaluation environment is needed: inference
does not optimize rewards. Report body/route/torque/power/load effects as
descriptive paired differences, not causal proof across training seeds.

Decisions: `reference-safety-stop`, `numerical-gate-stop`,
`runtime-failure-stop`, or at most `single-seed-diagnostic-support-only`.
All policy-acceptance, F2, obstacle and physical-motion flags stay false.
No extension, restart, checkpoint selection, another reward revision, Locked,
hop/H2, video, actor-observation expansion or raw perception is authorized
by a failed result. Stand/stop retention and independent training seeds
remain unresolved even if this diagnostic passes.

## Added measurements, not replacement gates

Retain full pre-control samples of actual base-link position projected into
the fixed initial-heading route frame, plus signed body/route/cross-route
velocity and heading. The last sample precedes the final action; do not
mislabel it the episode endpoint. Stop on the first terminal and never
include reset position. Positions are read from simulation, not reconstructed
by integrating velocity. Retain max absolute cross-route position and signed
cross-route velocity mean for drift-versus-sway diagnosis; no new position
threshold or removal of the old absolute-velocity gate.

## Retention and closure

Entrypoint: `python -m mjlab_microduck.foundation_reward_experiment --source SHA`.
Retain launch/configs, all checkpoints, TensorBoard/rollout metrics, child
logs, per-case trace reports, deterministic decision and SHA256/byte manifest.
Mirror artifacts locally; verify hashes and reproduce the decision from
retained JSON before committing results. No indefinite monitor is needed
for this bounded service. Stop at its decision and report next evidence gate.

Prelaunch local regression:661 passed in10.59s, no skips, two existing
actuator/site-pattern warnings. Includes config identity, matched nonregression,
first-failure ordering, subprocess budget/module, route-frame invariance and
first-terminal trace exclusion. Remote tests and GPU execution remain separate
checks; this paragraph does not claim either has happened.
