# Frozen actor input and speed-contract audit

## Scope

Continuation of the closed [straight-speed control](2026-09-07-straight-speed-response.md).
No new simulation, optimizer update, gate relaxation or checkpoint selection.
The seed-379 recovery A/B and seed-383 straight control remain closed failures
of their respective speed gates. The recovery-only PPO prerequisite is unmet.

Frozen actor: `model_7998.pt`, SHA-256
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`,
under remote `logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee`.
Saved `params/env.yaml` SHA-256:
`7697d55a64d76b8c1fdb207640a86fc59cfbc010c7e0ed9146535355230cca50`;
`params/agent.yaml` SHA-256:
`3ec05f69852d09151d0d4a67130acb8875275c41e4b3ce55bb510edff2888b76`.
Saved YAML was inspected without constructing its Python-tagged objects.

## Read-only configuration findings

Selected saved fields agree with `speed_response_control.prepare_config()`:

- Actor term order and function paths: base angular velocity, projected gravity,
  relative joint position, relative joint velocity, last actions, twist command,
  head command, body command. The checkpoint expects 61 inputs and 14 outputs.
- Term scales/clips are unset; history length is zero. Angular velocity/gravity
  delays are 0–1 steps; joint velocity delay is exactly one; other terms have
  zero delay. Concatenation is enabled. Actor corruption is **enabled in both**
  saved training and current play configuration; play must not be described as
  noiseless merely because it is an evaluation task.
- Joint-target action scale 1, offset 0 with default offsets enabled; simulation
  timestep .005 s and decimation 4 (50 Hz control).
- Actor MLP 61 → 512 → 256 → 128 → 14, ELU, observation normalization enabled,
  Gaussian distribution with scalar standard-deviation parameterization.

This is not an exhaustive equivalence audit of all physics, parameters, package
versions, stochastic effects or checkpoint-training runtime. Head/body commands
were randomized in training and intentionally fixed to zero in the control.
No identified difference so far establishes a configuration regression.

The saved forward-command curriculum starts at 0–.5 m/s and raises the maximum
to .65 then .8. Therefore **.3 m/s was within the configured training range**;
do not call it an unseen command. Configured coverage is not evidence of
accurate learned tracking or of the realized training-command distribution.

## CPU restoration diagnostic

`python -m mjlab_microduck.checkpoint_inference_audit --source <exact SHA>`
requires CUDA hidden, a clean exact remote source and the frozen actor hash.
It creates one exclusive `artifacts/evaluations/frozen-actor-input-audit-v1`
output. It reconstructs the installed RSL-RL MLP, strictly restores every actor
key (including normalization buffers), and probes synthetic 61D inputs twice
in evaluation mode. Finite results, unchanged state and CPU RNG preservation
are loading checks, **not** speed tracking or complete runtime equivalence.
No simulator is constructed and no optimizer step is taken.

The preliminary CPU inspection restored all keys. Saved normalizer count is
786432000; standard deviation spans .0028858329–3.5669066906, with finite
61-dimensional mean/variance/std buffers. Installed runner/algorithm source
loads `actor_state_dict` strictly and selects evaluation mode for inference;
normalization is part of that actor state. The repository runner does not
override checkpoint loading. A retained execution of the tested helper follows
its source commit; no behavioral acceptance follows from a successful load.

## Historical speed evidence

These are original reports, not new trials or retrospective acceptance. They
use different seeds, environment counts and sample windows from seed 383.

| Retained protocol | Command | Measured body-forward mean | Torque-utilization p99 |
| --- | --- | --- | --- |
| HC0 envelope, seed 41, zero yaw | .300 m/s | .2114190012 m/s | .5113741755 |
| Stage 2, seeds 41/42/43 | .500 m/s | .4128075242–.4167431593 m/s | .6898138523–.7040922642 |
| Stage 2, seeds 41/42/43 | .800 m/s | .6179999709–.6224805117 m/s | .9023922682–.9235647321 |
| Closed straight control, seed 383 | .300 m/s | .2102451335 m/s | .5082634687 settled |

HC0 recorded the actually applied forward command as .3000000715 m/s and
measured mean yaw rate −.0581833646 rad/s despite zero commanded yaw; fall rate
was zero. The older Stage-2 report does not contain equivalent applied-command
or yaw fields: absence must not be treated as zero or as command verification.
Both historical reports used 64 environments, 100 warmup steps and 600 measured
steps; the recent control used 8 environments and 300 settled steps after 100
startup steps. Agreement of approximate means is historical context, not a
matched reproducibility experiment.

Original remote reports and SHA-256:

- `artifacts/evaluations/hc0-command-envelope-a1b3611-s41/checkpoint-evaluation.json`:
  `d3c0ce5775d0f8109e2f449b00ff29fdb7ef4e40b9e259dad895e09702cbcd14`.
- `artifacts/evaluations/run-stage2-motor-envelope-5speed-3seed-36667ee/checkpoint-evaluation.json`:
  `9053a929e5d0faf92b9150f223486ae2bf5766b92ba79ff868326ac169f61a47`.

Inference: appreciable undertracking predates tonight's recovery measurement
code. It is not established to be a new obstacle-controller regression. The
historical Stage-2 label “accepted .5–.8 m/s commands” neither means measured
speed equals the command nor grants today's stricter .60 torque admission.
Changing a gate, relabeling the goal as .21, or increasing the command to mask
the deficit would not establish the requested recovery-speed capability.

## Next bounded handoff

Retain this audit and its inputs, run focused CPU tests, commit and push. Keep
the GPU idle unless a distinct, predeclared diagnostic is justified. Next audit
command delivery/observation timing and the IMU/noise play contract with focused
source-level counterexamples. Preserve command coverage versus learned
capability as separate claims. Do not expand actor observations or silently
substitute low-level locomotion retraining for the blocked recovery-only PPO.
If no concrete contract defect is found, present the separate low-speed
locomotion calibration/training scope for explicit agreement rather than
spending GPU time on the closed diagnostics. All overnight deadline and
protected-service rules remain unchanged.

## Retained execution and closeout

Source **`1ee0f7b6c5ee12ec10dabbf83ab64731bc516361`** was tested locally
(571 CPU tests, 9.91 s), pushed to the exact fork feature branch, then
fast-forwarded onto the clean idle remote worktree. The same **571 tests passed
on 100.100** (8.18 s). Two pre-existing actuator/site-pattern warnings remain.

The CPU-only helper completed successfully on 100.100: all state keys matched,
both model and normalizer were in evaluation mode, two 2-by-14 outputs were
finite and equal, and state was unchanged. It retained
`artifacts/evaluations/frozen-actor-input-audit-v1/inference-audit.json` with
SHA-256 **`3cbc27cf51c6f858d6f9a3f9cc868b084d042331d00d110b9b5c7032e952c91f`**.
All behavioral, complete-runtime-equivalence and physical-admission flags are
false. No GPU simulation or optimizer was run for this audit.

Local `artifacts/diagnostics/frozen-actor-input-audit-v1` contains byte-identical
copies of that report, both saved YAML files and both historical reports (named
`hc0-envelope.json` and `stage2-envelope.json`). Their five hashes match those
recorded above; original remote files remain unchanged. Added hash-bound tests
keep loading success separate from behavior and prevent the retained command
labels from being mistaken for achieved speed or current .60 torque admission.

Evidence-closeout validation: **573 focused CPU tests passed locally** (9.63 s),
including both retained-evidence checks without skips; the same two existing
actuator/site warnings remain. The closeout changes no runtime configuration.

The next audit is now [command delivery and play noise](2026-09-07-command-delivery-audit.md).
It reproduces a changed-command timing lag in the hierarchical call ordering,
not a cause of the constant-command speed deficit. Read its latest handoff;
historical rollouts and all failed gates remain unchanged.
