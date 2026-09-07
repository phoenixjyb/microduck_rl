# F1-R: matched reward-width diagnostic

Status: **closed; numerical-gate-stop** (execution below). Separately authorized by
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

## Executed result — September7 10:11–10:23 Shanghai

Execution source `77a1d612495c484b42a0c58702d55eae163bff1b` was committed and
pushed before launch. Both hosts passed661 CPU tests (local10.59s,
remote7.91s, no skips, same two existing warnings). HTTPS push authentication
failed; the existing authenticated GitHub SSH path succeeded without credential
or remote-configuration changes. No failed HTTPS delivery was called a push.

One sequential user service, `microduck-rl-f1r-77a1d61-s421.service`, started
10:11:53 and emitted its final numerical decision10:23:31. No other GPU job
was launched. After completion the GPU was idle,45C/12MiB, with no compute
process; both protected services remained inactive and the remote worktree
remained clean at execution source. The transient unit was subsequently
collected; completion evidence is its retained results/decision and journal,
not an assumed persistent service exit-status field.

### Training and checkpoint integrity

| Arm | Updates | Learning wall s | Fall-containing rollout windows | Max rollout torque p99 | Max rated-speed exceed fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control smoke419 |10|6.3456|0|.729209|0|
| Narrow smoke419 |10|6.3356|0|.733033|.000011626|
| Control pilot421 |500|286.9103|8|.737774|.000069754|
| Narrow pilot421 |500|289.2436|12|.777159|.000186012|

Each nonzero pilot fall window involved1/256 environments; no training abort
limit was crossed. Do not call either pilot fall-free. Each run's74
TensorBoard scalar tags were finite, with10/500 scalar entries respectively;
the named NaN-state termination signal stayed zero. Sampled training GPU
temperatures peaked at59C. Max soft-limit exposure fractions were.013451
(control pilot) and.016578(narrow pilot). Training quantiles/exceedances are
not the stricter deterministic evaluation metrics and not pooled quantiles
across all rollouts. Total reward is not directly comparable across different
reward functions.

All26 scheduled post-update checkpoints were loadable, finite and had exact
iteration/common-step identities. Four initial snapshots also retained;
their actor, critic and optimizer state trees matched the original parent
exactly. Initial files matched byte-for-byte across each paired smoke/pilot.
Final pilot step204000, actor normalizer count789504000 in both arms
(parent786432000 plus3,072,000 samples). No counts were reset. Both final
actors changed from the parent; neither smoke was used as a training parent.

### Held-out pair431

All three parent evaluations passed safety/coverage but undertracked. Then
control431 and narrow431 ran. The narrow arm failed; paired433/439 were
**not executed**. Actual coverage:5cases/40 first episodes, not9/72.
All five cases had400 samples, zero terminals/nonfinite/command mutation,
zero rated-speed exceedance and all/settled torque p99 within.60.

| Settled metric | Parent7998 | Control8498 | Narrow8498 |
| --- | ---: | ---: | ---: |
| Mean body-forward speed m/s |.206439|.222818|.264707|
| Mean route-forward speed m/s |.187413|.214368|.252882|
| Body mean in.27–.33 band |0/8|0/8|4/8|
| Route mean in.27–.33 band |0/8|0/8|3/8|
| Timely.50s stable route window |0/8|0/8|0/8|
| Max heading drift rad |1.392097|.618995|.764543|
| Mean absolute cross-route velocity m/s |.098537|.094326|.093739|
| Legacy torque utilization p99 |.487920|.494691|.546806|
| Squared-load proxy |.029923|.033612|.038670|
| Mean per-joint absolute mechanical power W |.107126|.115519|.134028|

Narrow versus matched control: body speed+18.80%, route speed+17.97%, torque
p99+10.53%, squared-load proxy+15.05%, absolute mechanical power+16.02%.
This is a single matched training seed and one held-out paired case, not
generalization or a multi-seed causal estimate. The proxy is not physical
temperature. Absolute torque passed, but its+.052115 increase over control
exceeded the predeclared+.02 nonregression margin. Parent-relative torque
also regressed beyond its margin. Max heading increased by.145548rad.

The exact narrow failure list is:

1. `straight-body-mean-outside-band`
2. `heading-drift`
3. `cross-route-motion`
4. `torque-nonregression`
5. `matched-control:torque-nonregression`
6. `matched-control:heading-nonregression`
7. `matched-control:route_forward_per_env_mean-nonregression`

No checkpoint search, omitted-case retry, extension or next curriculum stage.
All authority/admission flags remain false.

### What the new traces establish

The CPU-only audit reproduced all five reports' velocity summaries and
per-environment stable-response windows from retained400×8 samples. Position
and signed-velocity summary fields also matched their traces. This adds
consistency evidence; it is not a new acceptance rule or hardware proof.

Narrow431 last sampled cross-route positions span−.777141 to+.697977m
at7.98s. Its largest sampled absolute cross-route displacement is.777141m,
versus.553864m for control431. These are actual projected positions, not
velocity-integral estimates. Drift is therefore real in this case; the
absolute-velocity failures cannot all be dismissed as harmless sway.
Some low-heading-error environments still have substantial oscillatory
lateral velocity, so do not equate every absolute-speed sample with net drift.
The final post-action position was not sampled and is not claimed here.

### Interpretation and next boundary

The narrower reward is a useful speed-learning signal in this paired case,
but it does not deliver motor-efficient straight-line tracking. Faster body
motion alone is not F1 completion, nor evidence for obstacles or hopping.
Next design work should diagnose route/heading feedback and the speed-versus-
motor-load tradeoff before another training revision. Keep the current61D
actor's lack of explicit base-linear-velocity and absolute-route-heading
inputs visible as a hypothesis/limitation, not a proven cause. No actor-input
expansion, looser gate or further reward revision is launched by this result.

### Retention

Remote originals stay under the predeclared experiment root; local mirror is
`artifacts/diagnostics/f1r-width-paired-s421-v1`. Manifest lists74 payload files
totaling159,024,034bytes, plus the manifest itself. Full hash/size verification
passed on both hosts with zero mismatches. One transfer was interrupted by
an SSH `Host is down` transport error after training/evaluation had finished.
Read-only connectivity checks recovered without a restart; the retry copied
the verified narrow final checkpoint first, then completed the remaining files.
No experiment or failed case was rerun. Numerical decision was independently
reproduced from the retained parent/control/narrow431 JSON; all five traces
passed the CPU consistency audit without rerunning simulation.

Post-run local focused regression:670 passed in12.75s, no skips, two existing
warnings. The added tests include full retained manifest hashes, actual
five-case accounting, raw-trace reconstruction, unchanged failed decision
and synthetic trace-corruption rejection. This offline audit was added after
GPU completion; execution identity remains77a1d61, not the evidence commit.

| Artifact | SHA256 |
| --- | --- |
| Control final8498 | `38904154f1f682b7622e84a83511edf848f2f2c091ab1748b3986b3ae2aebf90` |
| Narrow final8498 | `7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f` |
| Decision | `fde5ce1a0659a1bbbc759215c7548c182a7f61adae25a8b94f60cfbb92db8440` |
| Manifest | `ead4f9b6cd4ef9dcde184f5f70485da9a9c223efc80226afb86f71c183c3f753` |
| Parent431 report | `3a74b2eadabeb190fe45194b2e737dd051452f887bac719d9d5aa1eaf966068f` |
| Control431 report | `60df36a37970bdded4e640326dad3ca1c6416d415a25803b7362b20b5f7c2d12` |
| Narrow431 report | `af6463f91bcea9d1a90c1e2c40c45ae3645b28d48fa73df9d631de1e3d197a20` |
