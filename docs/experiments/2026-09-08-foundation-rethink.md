# Foundation rethink: diagnose, then teach distinct capabilities

Status: CPU audit, experiment design and locally tested diagnostic implementation.
The user approved development after the overnight closeout. No GPU evaluation, optimizer, scheduling, remote
deployment or physical motion is authorized by this document. Source inspected:
`35e7c6537eab179aa92158b21c8552a603b45d9e`. Historical ledgers and gates stay intact.

## Findings and what they do not establish

| Question | Evidence | Consequence |
| --- | --- | --- |
| Was .30 m/s unseen? | Original Stage-2 configured range included it; historical .30 command produced about .21 m/s | Not an unseen-command explanation; realized coverage and learned tracking differ |
| Is there a demonstrated reliable lower-speed range? | Existing reports do not establish one under the current per-environment motor/route criteria | Do not assume .10 or .20 is easier, or relabel .21 achieved at .30 command as successful .21 tracking |
| Is the basic interface mismatched? | Selected current training/evaluation actor observations, action config, simulation config and decimation compare equal; corruption is enabled in both | No selected-config mismatch found; this is not full historical/runtime equivalence |
| Are the learning and test domains identical? | F1-Y training retains x/y velocity pushes of +/- .30 m/s every 3–6 s; evaluator removes pushes. Training's collapsed CoM curricula request +/-15 mm trunk and +/-10 mm head; evaluator clears curricula and retains +/-3 mm event defaults | Command learning and robustness are mixed. Document effective reset parameters, not just initial saved YAML; no causal attribution yet |
| Is heading control matched? | Training uses zero yaw or reset-sampled yaw commands; fixed evaluation uses the external heading-hold loop and fresh command delivery | Yaw coverage is not the same as heading-loop trajectory exposure; existing ON/OFF evidence already bounds this explanation |
| Did short pilots prove convergence? | Each arm is 500 updates x24 steps x256 environments =3,072,000 transitions, 240 simulated seconds per lane, with resets | Substantial parallel data but no convergence proof; longer training is a hypothesis, not the established cure |

The actual `com_range_curriculum` mutates the live event manager's copy. A CPU
test invokes it with a synthetic manager and confirms the listed range changes;
it does not execute a reset or establish actual sampled body inertias in an old
rollout. Training sets the restored common step before its first wrapper reset.
Both inspected event configs initially serialize +/-3 mm, so a raw YAML-only
comparison misses this distinction. Push and CoM differences are deliberate
configuration differences, not a newly proven implementation bug.

Evaluation also retains different reward weights/width from the fine-tuning
objective. Those reward values are not the numerical speed/motor acceptance
criteria, and inference does not perform optimizer updates. Do not change play
rewards and claim that this fixes tracking. The actor has no explicit measured
base-linear-velocity or absolute-route-heading input; that is an observability
constraint, not proof that a 61D policy cannot learn this task.

The previous timing audit already reproduced a changed-command lag; that does
not explain constant-command undertracking. The retained heading ON/OFF trial
improved route heading but still missed speed/lateral criteria. Do not repeat
either closed diagnostic merely to collect another favorable result.

## Revised lesson structure

1. **Map the frozen foundation before teaching it again.** Measure commanded
   versus achieved speed, including per-environment spread and startup. Keep
   rigid-foot walking separate from sprung hopping. No new accepted speed yet.
2. **Learn commands, then transitions.** If the map supports a usable starting
   band, compare fixed-speed practice against balanced command practice in a
   separately declared, matched domain. Add speed transitions only after steady
   tracking. Stop/stance/restart require their own measured-motion protocol;
   zero command is not a safe-stop controller.
3. **Add robustness as an explicit axis.** First establish baseline behavior;
   then stage pushes and parameter variation separately with baseline retention.
   Any simplified training domain gets a new identity. It cannot pass the old
   robustness protocol by substitution. Return to the full declared domain
   before robustness or integrated-controller admission.
4. **Obstacle supervision on a frozen admitted gait.** One generous-clearance
   obstacle first; then placement, turn direction and nominal-speed bins.
   Slowdown is allowed during negotiation. Approach and post-passage recovery
   must track nominal speed; collision-free passage alone is insufficient.
5. **Separate hop repair, then integration.** Crouch/rebound, low takeoff,
   controlled landing and settling are proposed lessons, not demonstrated
   skills. Keep original H1 gates. Resolve rigid/sprung mechanics before routing
   between policies; check walking/stance retention and transitions explicitly.

## Proposed next GPU diagnostic: frozen command-response map

Design identifier: `foundation-command-map-v1-draft`. This is not an executable
launch declaration. Its only question is where existing frozen policies track
commands under one declared evaluation domain. No reward or optimizer changes.

- Frozen identities: original Stage-2 model7998 SHA256
  `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`
  and research narrow model8498 SHA256
  `7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`.
  Neither is newly admitted; do not choose a later checkpoint after seeing cells.
- Proposed command bins .10, .20, .30 m/s; each starts a new first-attempt cell,
  not a transition episode. This maps three points, not a continuous interval.
  No .50/.80 extension and no standing claim from moving-speed cells.
- Development seeds503,509,521 are already used development data, **not fresh
  confirmation seeds**. Fixed seed-major, speed-ascending, original-then-narrow
  order: at most18 cells /144 first episodes, 8 environments, 400 steps, .02 s,
  first100 startup and300 settled. Seed pairing is not guaranteed identical
  random disturbances after divergent trajectories; retain reset/RNG metadata.
- Preserve the existing evaluation domain, sensor noise/delay and heading hold
  settings; use fresh raw command delivery. Log effective event parameters and
  model state, controller commands actually consumed, route positions, joint
  speeds/torques, power/load proxies and first terminals. No full observation
  recomputation within a step. No normalizer update or optimizer restoration
  for inference. Test these invariants before implementation can launch.
- Existing .30 acceptance stays unchanged. For the *new descriptive map*,
  report each moving bin's own command +/- .03 m/s, .50 s stable span completed
  within2 s after the100-step settling point, heading<=.25 rad and absolute
  lateral mean<=.05 m/s per environment. Preserve legacy pooled torque p99<=.60
  and zero rated-speed exposure/nonfinite/terminals, all/settled accounting and
  named-joint/load/power outputs. These exploratory bins are not policy admission
  or physical motor ratings; no pooled statistic conceals individual joints.
- An absolute safety, coverage or runtime failure stops the whole map. A
  tracking/performance miss marks that cell failed but does not select a retry:
  the remaining predeclared map may continue for diagnosis only. This is an
  explicit new coverage protocol, **not reopening** the old first-pair-stop
  acceptance campaigns. Publish failures and unexecuted cells; no promotion,
  causal learning comparison or generalization claim follows from this map.
- Budget proposal: one sequential service, at most120 s/cell, 60 s setup and
  180 s closeout reserve =2400 s hard cap. No automatic retry/extra cell. Validate
  startup timing before reserving this window; exact runtime source, output
  identity, tested evaluator and new user-approved time window remain unbound.

The current `speed_response_control` and `foundation_evaluation` hardcode .30
and its report contract. Do **not** monkeypatch their constants or reinterpret
historical reports. A future parameterized diagnostic needs a separate schema,
tests and explicit no-admission output while leaving those defaults unchanged.

## Decision after the map, before another optimizer run

- If no bin satisfies its descriptive motion/motor checks, do not advance to
  obstacle RL. Diagnose named-joint/phase loads and effective domain/controller
  response first; do not assume a lower target or more updates solves it.
- If at least one bin works for all measured development seeds, call it only a
  candidate practice band. Predeclare an independent confirmation protocol
  before claiming reliability; do not interpolate to an entire speed range.
- Only then draft a matched learning experiment from one fixed parent:
  fixed-command control versus balanced commands within the supported band,
  with the same objective, controller, physics/domain, optimizer restoration,
  environments and budget in both arms. Do not simultaneously remove pushes,
  narrow CoM randomization, add actor inputs and change rewards.
- A provisional learning-curve budget could be2000 updates per arm, with common
  checks at500/1000/2000, but it is **not selected or authorized here**. Choose it
  from observed throughput and prior learning curves; fix the final selection
  and stopping policy before launch. Do not extend an unsuccessful500-update
  run after seeing its result or choose each seed's best checkpoint.

## Evidence pointers and verification

- [Original actor/speed audit](2026-09-07-actor-input-and-speed-contract-audit.md)
- [Command timing audit](2026-09-07-command-delivery-audit.md)
- [Original curriculum proposal](2026-09-07-foundation-speed-curriculum-proposal.md)
- [Heading ON/OFF result](2026-09-07-heading-hold-diagnostic.md)
- [Latest paired yaw result](2026-09-08-f1y-command-support.md)
- [Readiness and missing acceptance](2026-09-08-curriculum-readiness.md)
- [New CPU contract tests](../../tests/test_foundation_rethink_contract.py)

Validation: five new CPU checks plus the existing speed-curriculum, yaw-pilot,
heading-hold and foundation-pilot tests passed: **76 passed in10.30 s** with
CUDA hidden. Markdown was rendered to HTML; its seven-row/three-column table
and all seven local links were checked programmatically, not by screenshot.
Diff whitespace was checked. No GPU behavior or remote validation was run.
No production configuration, model, historical report, acceptance gate, remote
worktree or service changed. This plan and its tests are local, uncommitted work.

## Development slice 1: CPU trace scorer

The subsequent user-approved development chunk adds
[`foundation_command_map.py`](../../src/mjlab_microduck/foundation_command_map.py)
and [synthetic tests](../../tests/test_foundation_command_map.py). This implements
the ordered 18-cell schedule, per-speed raw trace scoring, heading-rule and
issued/consumed-command checks, motor accounting, first-terminal boundaries
and ordered-prefix reconstruction. It imports no simulator or launcher and
performs no policy inference, checkpoint load, optimizer step or output write.

The trace protocol `foundation-command-map-trace-v1` is distinct from the draft
campaign identity and historical .30-only reports. Inputs must be finite CPU
float32/float64 tensors with explicit ordered joint names and boolean terminal
flags. Supplied traces are not trusted producer evidence: checkpoint-byte,
runtime and producer verification stay false, along with policy/training/
physical admission and continuous-speed-range validation. Metadata labels alone
cannot prove a model ran. Invalid/overflowed inputs raise rather than produce
a passing report. Later execution code must retain that as a runtime stop.

Performance misses remain failed cells but allow the next fixed descriptive
cell; safety/coverage failures close the prefix. Reordered cells, duplicates,
post-stop cells and samples after any environment's first terminal are refused.
No automatic resets, retries, seed selection or interpolation are performed.
Motor force and speed retain their distinct capture timing; the reported
power/load values remain proxies, not instantaneous calibrated thermals.

Remaining after slice1, before any GPU request: a separate parameterized capture loop and
same-step command adapter, immutable source/runtime/checkpoint/config binding,
effective reset-parameter capture, CPU lifecycle tests with stubbed execution,
exclusive output handling, runtime/idle-GPU guards, explicit user-approved
window and retained sequential service. This slice is **not launch-ready**.
It does not modify the old evaluator, its constants, saved results or gates.

Slice validation: **165 CPU tests passed in43.45 s**, including30 new map tests
and the prior foundation/rethink, speed-curriculum, yaw, heading, original
speed-response and recovery-measurement coverage. The new .30 cross-check
matches shared historical summary arithmetic without altering the old module.
Inputs, CPU RNG and historical source bytes remain unchanged by scoring in
the tested cases. Markdown-to-HTML checks verified the seven-row/three-column
table and all nine local links; no screenshot-based check was performed.
All four new files (plan, rethink tests, scorer, scorer tests) remain local and
uncommitted. No remote validation, service change or GPU work was performed.

## Development slice 2: borrowed-session capture core

[`foundation_command_capture.py`](../../src/mjlab_microduck/foundation_command_capture.py)
adds parameterized configuration, a bounded capture loop and an explicit file
pin checker. It creates no environment or service and has no CLI. The future
launcher supplies an already initialized session and frozen actor, owns cleanup
in `finally`, and provides the external hard process timeout and GPU exclusivity.

The core uses the existing fresh-command adapter before the actor's own
normalization, without recomputing sensors. It records cached/issued/consumed
commands, actual supplied route positions/velocities, pre-reset motor samples,
legacy post-step motor samples and terminal flags. It accepts the installed
wrapper's integer0/1 terminal output only after validation, then normalizes it
to boolean. Any terminal stops all environments before a new inference call;
the pre-reset motor sample remains separate from reset-sensitive legacy data.
The exact .003 m trunk/head CoM event ranges are checked on the borrowed live
event manager. This is not full effective-plant or randomization verification.

Actor parameters and buffers (including normalizer state) are hashed before and
after capture; all submodules must be in evaluation mode. Commands are checked
again across inference before physics. Nonfinite results, missing pre-reset
capture, invalid terminal values, state mutation, clock regression or elapsed
budget throw `CaptureFailure` with partial in-memory raw frames and separate
completed inference/step-call counts. A prepared frame is not a completed step.
The caller still must persist these failures durably; this core writes no files.

The cooperative cell budget is at most120 s and checked around each call and
after scoring. It **cannot interrupt a hung native call** and is not a hard
service timeout. Capturing supplied callbacks never proves actual physics ran;
the output keeps physics/runtime/checkpoint-loading verification and all
admission flags false. Tests use actual installed observation and motor-stream
code, with synthetic CPU state, a stub policy and a stub wrapper, not MuJoCo.

`bind_files` verifies the selected checkpoint pin and caller-supplied runtime
file hashes/byte counts, rejects symlinks and empty runtime pin sets, and reports
only the checked file bytes. It does not deserialize checkpoints, establish
runtime-pin completeness, prove a loaded actor matches the checkpoint, or lock
files against concurrent changes. Factory integration must bind the reviewed
pin manifest/source, verify before and after loading/capture, strictly load only
inference state and preserve the resulting evidence. Do not turn its byte-check
flag into a full-runtime or policy-acceptance flag.

Remaining before a GPU request: tested native-session construction/strict
checkpoint loading; source and complete runtime/config binding; exclusive durable
outputs and failure receipts; cleanup/timeout/idle-GPU guards; then a separately
approved window and sequential service. No new training or GPU evaluation is
authorized by this development slice. All changes remain local and uncommitted.

Slice2 validation: **217 CPU tests passed in34.90 s**, including17 new capture
tests and the scorer, rethink, command-delivery, heading, original speed and
recovery regressions. Initial fixture failures were diagnosed against the
installed wrapper: the fixture now supplies its actual TensorDict/0-or-1 integer
terminal contract; production input validation was not weakened. A changing-
heading test observes one cached read plus exactly one sensor update per step,
with fresh yaw commands and no extra same-step observation computation.
Markdown-to-HTML table/link inspection and new-file whitespace checks passed;
no screenshot, native simulator, remote validation or service work was run.

## Development slice 3: native factory and durable cell evidence

[`foundation_command_session.py`](../../src/mjlab_microduck/foundation_command_session.py)
implements strict checkpoint loading, an owned native session and exclusive
per-cell evidence. It has no CLI, campaign loop, training runner or service
launcher. [CPU tests](../../tests/test_foundation_command_session.py) cover
actual checkpoint restoration and synthetic lifecycle/failure injection.

The loader hashes the exact bytes passed to `torch.load(weights_only=True)` on
CPU, validates the pinned iteration and saved common-step counter, and restores
only the 61D MLP actor and its normalization state. Strict key, shape, dtype and
tensor equality checks reject silent conversion or incomplete restoration.
Synthetic inference probes must be finite/repeatable and preserve state and CPU
RNG. No optimizer or critic is constructed/restored. Both separately retained
real checkpoints passed this test locally: original7998/common-step192000 and
narrow8498/common-step204000, with the hashes declared above.

The native factory explicitly selects CPU or CUDA0, constructs an environment
and wrapper, restores common time after the wrapper reset (the retained
evaluation ordering), then installs the pre-reset motor stream. It closes a
successfully constructed environment on wrapper/stream setup failure, capture
failure, normal completion or post-capture identity failure. It snapshots the
parameterized environment/agent YAML and records the existing selected runtime
pins before/after capture. Constructor-internal failure and hung native calls
still require external process teardown. Full runtime/plant equivalence is not
established by those selected pins or initial YAML snapshots.

Each output directory is exclusive and cannot be reused. Complete control frames
are cloned into a flushed/fsynced JSONL journal before the next control step.
The writer retains tensor traces, cached commands, loading/runtime evidence and
scores, then writes a file-hash manifest last. The cooperative <=120 s budget
covers construction, capture, cleanup and retention, with a final check before
manifest publication. It is not a hard timeout and cannot interrupt blocking
filesystem, constructor, CUDA or native simulation calls.

Capture exceptions retain partial tensor frames and a failure receipt where the
filesystem remains writable. A hard-killed process can leave only a journal or
partial files; missing, partial or failure manifests must never be read as a
successful cell. A decision file alone is provisional: the final manifest is
the closure marker. A filesystem failure cannot guarantee durable receipts and
is reported as an additional failure, not hidden. `captured-diagnostic-only`
means capture/evidence closure, not behavioral success: terminal/safety cells
still classify as `safety-or-coverage-stop`. All policy/training/physical
admission flags remain false, including for correctly loaded real checkpoints.

Validation: **259 CPU tests passed in22.41 s** with CUDA hidden, including28
new session tests, strict real-checkpoint loads, callback isolation, cleanup,
tampering, dtype rejection, full400-step journal retention, exclusive writes,
manifest hashes, storage and retention-timeout failures, plus the earlier trace/capture/configuration/command/speed
regressions. Native environment/wrapper construction is stubbed in lifecycle
tests; no actual MuJoCo rollout, CUDA evaluation or remote validation was run.
Markdown was rendered to HTML and its table/local links inspected
programmatically; this was not a screenshot-based visual review.

The three development slices and plan are committed together in the local
feature branch; the earlier "uncommitted" statements describe their original
slice handoffs. No historical evaluator, numerical gate or result is changed.
No service, automation, remote worktree or GPU workload was changed.

**Remaining launch gate:** bind a reviewed exact source/runtime manifest and
effective reset/RNG evidence, complete the sequential-map driver and its
failure-prefix checks, then establish an authorized runtime window with an
exclusive idle-GPU guard and external hard process timeout. This commit is
diagnostic development, **not launch-ready** and not a newly trained capability.

## Development slice 4: sequential supervision and evidence reconciliation

[`foundation_command_campaign.py`](../../src/mjlab_microduck/foundation_command_campaign.py)
adds the sequential driver, launch-plan validation, read-only host/source/GPU
checks, an advisory GPU lease, a supervised per-cell process and a raw-evidence
reader. [CPU tests](../../tests/test_foundation_command_campaign.py) exercise
the complete fixed order and failure boundaries without native GPU execution.
The public campaign entry is a Python API, not an installed service or scheduler.
The module CLI is only its supervised child and requires an inherited lease and
the exact byte-pinned parent plan. No current launch plan or time window has
been created, and no child was launched on100.100 in this development slice.

The launch plan must identify the exact clean feature-branch commit, Linux
machine ID, both pinned checkpoint paths, an explicit UTC start/deadline window
and reviewed environment-local runtime file hashes. Required source-file pins
include the environment, observation/metrics managers, entity data and RSL MLP.
These are a minimum selected-file scope, **not complete runtime equivalence**;
review the real host's dependency/asset coverage before supplying a launch plan.
Source, selected runtime and checkpoint bytes are checked before and after each
cell and before final closeout. The parent hides CUDA; only its guarded child
selects CUDA0. An environment allowlist excludes API/connector credentials and
Python/preload/library-path injection. Its compatibility with the actual Linux
runtime remains to be validated; inherited user environments are not assumed
equivalent to this cleaner execution environment.

The fixed destination is
`artifacts/evaluations/foundation-command-map-v1` in the authorized worktree.
It must not exist; neither closed nor partial output is reused. Cells execute
in the predeclared seed/speed/policy order. The reader rehashes the exact bytes
it deserializes, reconciles all journal rows against the tensor trace, checks
loading/configuration/capture identity and recomputes the numerical score and
ordered prefix. A renamed, partial, altered or failed manifest cannot become a
successful cell merely by presenting a decision JSON. Even reconciled evidence
does not independently prove native physics or admit a policy.

Performance misses remain failed descriptive cells and may continue through
the fixed matrix. A terminal, motor safety/coverage failure, child error,
evidence mismatch, source change or exhausted reserve stops without retries,
replacement seeds or later cells. Per-cell receipts are durable before the
next cell. Runtime failures retain a separate campaign failure record and the
unverified/unexecuted suffix, preserving partial child artifacts. A complete
18-cell map still grants no training or physical admission.

The shared Duck lease is advisory: it prevents cooperating jobs from racing,
but cannot reserve the GPU against arbitrary external allocators. Two idle
samples are required before and after each child. While the child is active,
read-only probes require both protected services inactive, no foreign compute
PID and GPU temperature below80 C. A conflict stops only the owned Duck process
group; no foreign PID is killed and no service is started/stopped. Sampling
cannot promise zero overlap with an independently launched workload.

A separate watchdog kills the owned child group at the120 s cell cap even if
its native call or the supervisor's telemetry callback blocks. A zero-exit child
with surviving group members is a runtime failure, not successful cleanup.
Already dead leaders are reaped before cleanup signals: CPU testing on macOS
found and fixed an EPERM cleanup edge case for a watchdog-killed zombie.
Tests also confirm that an unrelated disposable CPU process remains alive.
Logs, child PID/timing and available telemetry are retained on failure; a hard
kill may leave only the already-synced journal, never a passing cell manifest.

**The whole-service hard cap is still an external launch requirement.** The
driver checks a2400 s total budget and180 s closeout reserve, but an actual user
service must independently enforce `RuntimeMaxSec=2400` and group-wide cleanup
(`KillMode=control-group`) if the supervisor itself hangs or is terminated.
No service definition was installed or invoked by these tests.

Remaining before GPU execution: review/bind the host-specific launch manifest,
effective reset/RNG evidence and clean child-environment compatibility; validate
on the remote CPU; establish the user-approved window and retained hard-capped
service. No scheduler, optimizer, video, raw perception or robot motion is added.

Slice4 validation: **328 focused CPU/regression tests passed in19.70 s** with
CUDA hidden, including62 new campaign tests and the existing checkpoint,
capture, scorer, curriculum, command-delivery, recovery and idle-GPU coverage.
The process tests use disposable CPU sleepers, not simulation or training.
Watchdog-start failure also reaps an already launched child rather than leaving
it running without supervision.
The supervised-child `--help` entry point also passed with CUDA hidden.
Markdown-to-HTML table/local-link validation and staged whitespace review passed;
no screenshot-based visual review, native rollout or remote validation was run.

## Slice5: sampled reset evidence and cleanup race regression

The native session now retains initial per-environment qpos/qvel and encoder
bias, selected model fields plus the event manager's expanded randomization
fields, live CoM ranges, reset/common-time counters and available Python,
NumPy and Torch RNG states inside `runtime.json`. Sampling happens after wrapper
reset and saved common-time restoration, before the first actor input; it calls
neither observation recomputation nor a physics step. Empty derived fields
(for example, zero tendons) retain explicit shapes. The campaign reader checks
this metadata in addition to its existing hashes and trace reconciliation.

This is sampled starting-condition evidence, **not a complete replay state**:
Warp RNG and all internal solver/history buffers are not exported. It does not
prove deterministic physics, full runtime equivalence or policy acceptance.
Native host construction and the reviewed service launch plan remain pending.

CPU tests reproduced a Darwin race where a watchdog-killed child becomes a
zombie between `poll` and cleanup's signal. Reaping only before the signal was
insufficient. On EPERM cleanup now waits for that owned leader and retries the
group signal, preserving descendant cleanup; persistent permission errors still
fail. No external workload or GPU service was involved in these tests.

Validation: **352 focused CPU/regression tests passed in17.75 s**, CUDA hidden,
including reset snapshot nonmutation, corrupt evidence, empty-field shapes,
strict actor restoration, campaign readers and disposable-process lifecycle
checks. No GPU map or optimizer was started by this slice.

Remote CPU follow-up at `b0dac8276580aef70294c846e7576009a5cd8001`:
350 tests passed and2 optional Mac-path checkpoint tests skipped in18.77 s.
Actual native CPU construction then caught an adapter mismatch before any actor
step: mjlab's ModelBridge exposes `TorchArray` proxies, not Tensor subclasses.
The reset reader now explicitly unwraps only that installed adapter via its
read-only `detach` method; a real CPU Warp-bridge regression accompanies the fix.
The initial failure JSON remains in
`artifacts/evaluations/foundation-command-map-native-cpu-b0dac82`; its diagnostic
wrapper also attempted an unsupported closure label, leaving an unsealed failed
directory. It is not a map result and will not be reused or called successful.

At `0f03f052368dfa496b0936097b357cf87f1c887a`, native CPU construction succeeded
in0.94 s after imports, with no policy steps and Torch CUDA uninitialized. The
sealed evidence is at `artifacts/evaluations/foundation-command-map-native-cpu-0f03f05`.
The recorder captured15 model fields (including two empty tendon fields), all
eight initial states and the available RNG states. The full remote allowlisted
CPU regression then passed352 tests with2 optional Mac-path skips in18.68 s.

Host launch review expands the selected byte pins to Python sources in mjlab,
rsl_rl and mujoco_warp plus the Warp bridge/context/code-generation and version
files. This still excludes some native libraries/assets and is explicitly not
complete runtime equivalence. Review found eight genuinely empty package
initializers; runtime hashing now accepts and pins empty source bytes rather
than omitting those files. Empty checkpoint/report evidence remains rejected.
The failed draft plan preflight wrote no launch plan and started no GPU process.

The single frozen-map invocation will use the exact reviewed feature SHA,
machine ID and checkpoint paths, a clean CPU-only parent, the shared advisory
GPU lease and an independent `RuntimeMaxSec=2400`, `KillMode=control-group`
user service. It must finish before September8 23:30 UTC. No optimizer is part
of this map; a safety/coverage failure closes its prefix without a rerun.
