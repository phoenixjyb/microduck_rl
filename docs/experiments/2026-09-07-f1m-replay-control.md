# F1-M same-source OFF/OFF replay control

Status: **completed; safe same-source recording-disabled reports diverge**. Zero optimizer updates; tonight's paired-pilot
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

## Retained result: September7 23:05 Shanghai

Source `79074f7b446a1a1ac854335855a9f2b10ceebce5`; service
`microduck-rl-replay-off-off-79074f7-s503.service` ran23:05:13–23:05:53 and
completed normally. Remote regression also passed760 tests in17.81s with the
same two existing warnings, no skips. Normal exit means a retained comparison,
not an accepted policy or exact repeatability.

Both cases completed400steps/8env with no absolute safety failure and no
motor_trace field. Their full reports differ at28,563 typed JSON leaves.
First changed velocity is step3/environment0/body-forward; first position
change is step4/environment0/route-forward. Maximum per-sample differences:
body-forward0.049969m/s, route-forward0.060063m/s, cross-route0.091312m/s,
heading0.060312rad; maximum route/cross position differences0.012718/0.027555m.
The decision is `recording-disabled-same-source-divergence`, not a solver-cause
diagnosis. This shows raw motor-history recording is not necessary for the
observed divergence in the current same-source execution path.

All captured numerical settings match across the two before/after snapshots:
Torch float32,1intra-op/16inter-op threads, deterministic algorithms false,
cuDNN enabled but benchmark/deterministic false, both precision options `none`,
CUDA uninitialized before and initialized afterward. `PYTHONHASHSEED` is unset
at child entry and becomes503 during environment construction in each. The
fixed-string hash probes differ between processes and do not change after
seeding, confirming that interpreter-start hash identity was not controlled.
No setting was changed during this diagnostic. Hash variation is a measured
remaining factor, not proof that it caused the simulation differences.

All10 payloads (7,261,512bytes plus manifest) are mirrored under
`artifacts/diagnostics/f1m-replay-off-off-s503-v1`; hashes/bytes and exact file
coverage verified. CPU recomputation reproduces the complete deterministic
comparison, including all differences, trace extrema and fingerprint verdict.
Both historical F1-M and failed timing manifests/payloads were verified unchanged
before and after the control; checkpoint sentinels remain intact.

- Manifest: `2d0b1c5b572eccdce205e9fe51f35c20dfebf4fef32545ac9c95f7ae82590d90`.
- Decision: `8557576964007fa6978cc3da3f0cd9c313e78409fd25f1c053b08d180d4d622b`.
- First report: `6df1b35d789b3d2822e98fe2b493f9267949c6323bbb6fd80aa8ea8f567a422a`.
- Second report: `cf0f5b4c632aadec6aadb996cb7d2f2c6156a0fdf6af951a5422b8f6186029c4`.

GPU returned idle,0%,46C,12MiB; no new training or third case was launched.
This protocol is closed. The smallest justified next control is to pin only
`PYTHONHASHSEED=503` **before child interpreter startup**, keep motor-history
recording OFF and every physics/policy/Torch setting unchanged, and compare
two fresh cases in a new predeclared directory. Do not claim that this will
fix MuJoCo Warp, or alter the old decision if the next control matches.
