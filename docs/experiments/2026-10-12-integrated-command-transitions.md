# Integrated stop/start and signed-turn transition preparation

Status: **CPU-only offline plan/scorer; no dynamics rollout, trained transition
skill, collector qualification or physical acceptance**.

The user approved this parallel slice while the
[fresh motor/range-aware foundation](2026-10-12-fresh-motor-range-foundation.md)
trains at exact native source `d0cba9c0735fa49fc86b42383e756a5fbd6c70a5`.
Do not edit, synchronize, restart or add a GPU job to that checkout while its
retained learner/evaluation service is active. All development below is on Mac
`integration/upstream-2026-10-10`; no training recipe or policy is changed.

## Fixed next-skill matrix

Protocol `integrated-command-transitions-v1`, seeds1351/1361/1367,8 independent
worlds per seed/sequence,50Hz. Nine cases/72 first attempts, never a pool of
retries or successful fragments. Commands below are `[body-vx,0,body-yaw-rate]`
in m/s and rad/s. A world experiences one complete sequence without resets or
policy/controller switching. These are externally supplied step commands, not
an action filter or a new actor input.

|Sequence|Ordered stages|Control steps|
|---|---|---:|
|Stop/restart|idle2s → .30 straight4s → stop4s → .30 restart4s|700|
|Left/right return|.20 straight3s → (.20,-.20)3s → straight3s → (.20,+.20)3s → straight3s|750|
|Right/left return|.20 straight3s → (.20,+.20)3s → straight3s → (.20,-.20)3s → straight3s|750|

Signs are explicit; route-relative left/right interpretation must be bound by
the actual simulator adapter. No in-place pivot, obstacle, hopping or football
is added to this matrix. Total52,800 world-control samples. Same actor61D and
raw/unclipped/unfiltered14 BAM actions; fixed plant and installed runtime pins.

## Trace and collector boundary

`integrated_transition_gate.py` uses only the standard library. It does not
import Torch, Warp, MuJoCo or mjlab, execute inference, initialize a GPU, simulate
physics or collect a rollout. It accepts the following retained trace shape:

- Root: protocol, source40-hex, checkpoint SHA-256, compiled plant SHA-256,
  frozen runtime versions, step_dt.02, actor_width61/action_width14,
  frame_phase `post-physics-pre-reset-last-substep-v1`,
  first_episode_only=true, automatic_retry=false,14 unique actual servo names,
  matching compiled hard joint ranges and ordered cases.
- Cases: exact seed/sequence order from the plan, worlds0–7 in order.
- World: world index, initial world-frame XY pose and chronological frames.
- Frame: step0 onward, response time `(step+1)*.02`, applied command and exact
  matching raw actor twist, body-frame linear velocity3, body yaw rate,
  world-frame XY pose, nonnegative tilt angle,14 named qpos/force/motor-speed
  columns,14 applied raw actions, grounded/terminal/fall/reset booleans and
  first-episode id0. Closed field names and finite scalar/vector coverage.

Command-to-declared-target comparison allows1e-7 solely for float32 target
representation; actor twist must equal the actual supplied command exactly.
Response timestamps allow1e-9 arithmetic error; dwell/deadlines use integer
sample indices, not approximate timestamps. Grounded means at least one valid
foot support, not double support throughout moving gait. The actual contact
sensor/threshold and validity must be bound by a separately tested collector.

The future collector must clone all post-physics state at the registered
pre-reset metric boundary, verify actual named joint/actuator mapping, bind
plant/runtime/model hashes and command/actor/action order, and terminate each
world's retention at its first terminal. A backend may internally reset that
world, but no later state/action can enter the retained first-attempt trace or
be used to repair its result. Missing terminal flags cannot hide resets.

Root identity strings are validated **claims**, not authentication of a model,
plant or collector. An input byte hash binds what was scored, not where those
bytes originated. The scorer cannot detect a falsified sensor/episode flag.
Actual capture binding, qualified sensor semantics and same-checkpoint
foundation evidence remain required before any runtime/skill conclusion.

## Numerical diagnostics and proposed transition gates

Each complete phase scores body-vx and yaw MAE after its first60 control steps.
Reference limits remain.03m/s idle/.10m/s moving and.10rad/s yaw. This is a
separate transition protocol, not a change to the existing foundation matrix's
single startup exclusion, case-level motor reduction or acceptance decision.

Additional transition diagnostics require a continuous sampled.50s band within
2s of each command change and another such band at phase exit. At50Hz that is
**26 samples/25 intervals**. A band whose last sample is exactly2.00s passes;
one ending2.02s fails. Broken bands restart the dwell counter. Moving bands use
absolute vx/yaw errors<=.10; idle/stop bands require grounded support, full3D
linear-speed norm<=.03 and yaw error<=.10. Early recovery cannot rescue an
unsettled phase exit. These new transition conditions do not replace foundation
admission or assert calibrated stopping capability.

Report stop-path distance from measured world XY, including the segment from
the pre-stop entry state to the first response sample. Also report phase path
length, tilt, recovery timing, motor power/thermal proxies and action magnitude/
rate p99. Stop distance has **no invented pass threshold** in this source-only
slice; use real admitted traces to predeclare a justified later bound.

All first-attempt frames—including startup and command-change transients—enter
the safety diagnostics: zero terminals/falls/rated-speed exceedance, torque
utilization p99<=.60 against.60Nm nominal, every named joint's minimum hard-range
distance>=-.005rad and right-hip.05rad stop exposure<=1%. The existing no-load
speed reference is123rpm converted to rad/s, matching `tasks/run.py`; no new
hardware limit is claimed. Left-hip exposure is reported separately. These
references are applied **per world**, so failure cannot disappear in a mean;
that stricter reduction is declared only for this new transition protocol.
Metrics use last-substep samples, not a claim to cover all internal substeps.

A terminal-prefix trace is valid failed evidence, never a complete success.
Unexplained truncated nonterminal traces produce `transition-evidence-incomplete`.
Malformed schema, flags, nonfinite data, step gaps, post-reset/terminal frames,
wrong commands, matrix holes and mismatched identity/layout fail closed.
Extreme finite values use robust mean reductions; nonfinite derived values are
rejected rather than emitted as invalid JSON. Raw actions are not filtered,
clipped or reinterpreted by the scorer.

## Commands and next admission boundary

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.integrated_transition_gate --plan --output artifacts/evaluations/integrated-transition-plan-v1.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.integrated_transition_gate --input TRACE.json --output REPORT.json
```

Output paths must be new; the CLI refuses overwrite. Input is bounded to64MiB
and reports its exact SHA-256. `transition-matrix-pass-diagnostic-only` means
only supplied numerical traces passed. Policy acceptance, simulator qualification,
collector qualification and physical motion authority remain false in all
reports; synthetic fixtures cannot unlock training, video or skill promotion.

After the active run finishes, first verify its final5999 bytes and original
12-case decision. Only a same-checkpoint foundation pass permits separately
declaring/testing a bounded **inference-only transition collector** and idle-GPU
capture under this plan. Keep original foundation results, policy normalizers,
observations, BAM state/history and motor/range gates intact. If transitions
fail, classify stop/restart, signed-turn response or safety failures before
considering a small new learning revision. Obstacle slowdown/recovery evaluation
and the existing replay tools follow later; they are not launched here.

## Source validation

CUDA-hidden transition/stage-exit/retention/corner regressions passed255 tests
in332.38s on the memory-constrained Mac. The final transition file contributes
45 synthetic tests. They cover complete9-case/72-world scoring, single-world
attribution, exact/late dwell deadlines, interrupted final dwell, wrong/stale
commands, gaps and duplicate frames, typed flags, missing worlds/cases, nonfinite
vectors, terminal prefixes, truncation/reset laundering, airborne/vertical/lateral
false stops, named-column permutation, joint-range boundaries, all-phase motor
gates, finite extreme-value reductions, CLI input hashes, stdlib import isolation
and refusal to overwrite outputs. Fixtures are ideal scoring data, not measured
duck dynamics. An initial duplicate-frame fixture was corrected to recognize
the earlier bounded-length rejection; no gate was relaxed.

Independent read-only review exercised additional boundary/overflow probes and
found no remaining blocking issue after corrections. The speed reference was
checked against the existing123rpm constant. Review highlighted overflow from
unscaled accumulation; normalized finite reductions and a serializable extreme-
value regression now cover it. Stop dwell additionally checks actual declared
ground support and full3D stillness, not vx alone. None of these changes touched
the active learner or its original foundation scorer.

Documentation link targets, stdlib-only import AST checks and `git diff --check`
passed. Machine plan was generated by the CLI on Mac at
`artifacts/evaluations/integrated-transition-preparation-20261012/plan.json`;
SHA-256 `1004b3b8bd7102b552cd44974238802aa7d3392cc51726c4b32bdfe02fc6e18a`.
No checkpoint weights were copied, no collector/inference/physics rollout ran,
and no policy, package, service, perception or physical-motion change was made.

The parallel read-only training check reached update3835/6000 with durable
model_3750.pt, all retained rows finite, zero NaN terminations and no failure
file. Latest-sample hip-stop exposure was0 for both names and torque p99.773765;
this is training telemetry, still above the held-out.60 gate, not acceptance.
GPU1717MiB/free14476MiB/62C and all four protected AI-service scopes inactive.
Native source remained frozen at `d0cba9c0`; later Mac commits must not be
synchronized into it while training/evaluation remains active.
