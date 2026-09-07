# F1-M same-source OFF/OFF replay control

Status: predeclared, not run. Zero optimizer updates; tonight's paired-pilot
count stays1/3. This is a measurement control, not a retry of the closed
motor-timing experiment and not another reward-weight revision.

## Read-only initialization/backend inspection

`speed_response_control.run_control` sets the declared Torch seed, constructs
the seeded environment, wraps/resets it, then loads the frozen checkpoint and
gets the inference policy. The initial observation fetch does not add a reset.
The evaluation path does not call the training-only `configure_torch_backends`
helper or enable deterministic algorithms. No seeding, reset or precision
setting is changed for this control.

The frozen installed mjlab environment delegates to `utils/random.py`:
`seed_rng` seeds Python, NumPy, Torch and calls Warp `rand_init`. Its own
comment says MuJoCo Warp is not fully deterministic. It assigns
`PYTHONHASHSEED` during execution, not before Python startup; this does not
establish equal interpreter-start hash ordering across fresh processes.
The installed helper also defaults `torch_deterministic=False`. These are
possible contributors, not proof of this campaign's numerical cause.

Read-only inspection on100.100 confirmed these additional helper hashes:

- `utils/random.py`: `5ba445435efa5f064aac4b572d035c6a6e98ca3ce459faf1f00dc6f73cbabbc8`.
- `utils/torch.py`: `87e0673c850ae6cd5306c9533c928ddb570396323b904f7bbbab589990bc44b1`.
- `rl/vecenv_wrapper.py`: `d458aa421d72c979d0269a24e51a4b49a02e4ec8cb5a9d10e47290aa5d86a3e2`.

Upstream's [determinism issue562](https://github.com/google-deepmind/mujoco_warp/issues/562)
was still open when checked September7; it requests an exact-output option
and says Warp implementation needs investigation. This supports investigating
repeatability, not attributing our divergence to a specific solver bug.

## Fixed two-case protocol

Module `mjlab_microduck.foundation_replay_control`; output
`artifacts/experiments/f1m-replay-off-off-s503-v1`. Same clean pushed feature
branch and frozen runtime on100.100. Verify the closed F1-M83-payload manifest,
decision, checkpoints, and failed timing5-payload manifest/decision before and
after. Use the original narrow model8498 (SHA
`7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`).

Run exactly two fresh child processes, `first` then `second`, on seed503 with
the same source, task, heading hold ON, commands, initial episode,400steps,
8env, physics and actor61D. Raw route and command traces remain enabled as in
F1-M, but **raw motor-history serialization is explicitly disabled in both**.
No new first-step hooks, actor/action capture, forced synchronization, backend
precision options, deterministic settings or environment fixes are introduced.
Each child must use the same numerical environment inherited from the single
sequential service. It must not inherit state from the first simulation process.

Record read-only numerical settings before/after each evaluation: allowlisted
environment variables, deterministic-algorithm flag, cuDNN options, Torch2.9
precision settings, thread counts, default dtype and CUDA initialization flag.
Also record the interpreter hash of one fixed public string. The hash probe
can differ when process-start hash seeding is uncontrolled; retain and report
that fact instead of pretending the assignment during env construction fixed
it. It is not an independently changed treatment and cannot identify a cause.
Require the other numerical settings to agree across matching before/after
snapshots or stop as a fingerprint mismatch, not a repeatability conclusion.

## Gates, budgets and decisions

Before each case, require the existing two-idle-sample GPU gate and both
protected SYSTEM services inactive. Each child120s maximum, one retained user
service420s hard cap with40s closeout reserve. No new GPU job after06:00 and
no child extending past September8 07:00 Shanghai. No other GPU workload,
no100.98, no service restoration or physical motion.

Validate unchanged checkpoint/source/controller/trace identity and first-attempt
absolute safety. Stop before `second` if `first` has any terminal/nonfinite,
coverage, rated-speed or absolute torque failure. An unsafe second case also
stops without a repeatability verdict. Existing speed/lateral/nonregression
failures remain retained and do not become policy acceptance.

For two valid safe controls, compare **every report field exactly**, retaining
all typed JSON differences and earliest changed route-trace indices plus
per-column maxima. Do not average, ignore fields, set post-hoc tolerances or
choose favorable runs. Outcomes are:

- `recording-disabled-exact-match-in-this-pair`: only these two cases match;
  not universal determinism, no third ON case automatically admitted.
- `recording-disabled-same-source-divergence`: the current unchanged evaluation
  process can differ without motor-history recording. Recording is not a
  necessary condition for this observed divergence; the particular solver,
  initialization, hash-order or execution cause remains unidentified.
- Safety, settings or runtime failure: preserve and diagnose; no repeatability
  conclusion, no automatic retry.

After either measured result, hash/mirror/reconcile the evidence and test/commit
the closeout. Stop GPU work for this protocol. Do not run a third ON case or
paired pilot2 until a new predeclaration explains the next measurement need.
Never turn a failure here into an accepted gait, obstacle or hopping skill.

## Prelaunch validation

Local focused regression:760 passed, no skips, two existing actuator/site
pattern warnings,18.36s. New tests verify non-mutating fingerprints, exact
typed comparison, separate hash-probe reporting, source/recording/settings
refusal, first-unsafe stop, two-case ordering, no optimizer call, deadline
refusal and immutable manifest coverage. The three newly inspected runtime
helper hashes match between local and100.100. `git diff --check` passes.
