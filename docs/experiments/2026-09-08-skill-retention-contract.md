# CPU-only skill retention prerequisites

This follows the closed F1-Y pilot, the third and final paired pilot this
window. No further GPU experiment or training is permitted before the
September8 07:00 Shanghai shutdown. Current task: bounded CPU implementation,
tests and evidence organization, not controller deployment or admission.

## Why mechanics and interfaces come first

The historical locomotion actor and rejected K3900 H1-T hop actor cannot be
switched merely because both emit14 motor actions. The sprung model adds two
passive foot joints, pad geometry/mass, travel, preload and damping; the locked
control also differs from the rigid robot despite zero travel. `tasks/sprung.py`
adjusts height rewards and spring monitoring accordingly. The obstacle warm
start utility also explicitly expands61/76 actor/critic inputs to68/83, which
is not an interchangeable input schema. No new model or migration is created
by this work. A saved checkpoint is neither proof of retained behavior nor an
accepted fallback under tonight's stricter foundation protocol.

## Slice1: byte-bound descriptors and conservative comparison

`mjlab_microduck.skill_compatibility` is a standard-library-only API. It
hash-verifies every referenced file without loading checkpoints, interpreting
evaluation decisions, compiling simulation, or writing files. Its strict
`skill-descriptor-v1` mapping requires an exact40-character training source,
skill ID and nine references, each with relative POSIX path, positive integer
byte count and lowercase SHA256:

| Reference | Content required of a future artifact-bound producer |
| --- | --- |
| checkpoint | Exact immutable weights/normalizer checkpoint |
| mechanics | Effective robot dynamics/contact/actuator model, assets and simulator options; not just a robot label or nominal spring constant |
| actor_interface | Ordered observation terms, dimensions, units/frames, preprocessing/noise, history and normalization semantics |
| action_interface | Ordered motor names, units, scale/offset, clipping, action processing and reset/history semantics |
| control_timing | Physics step, decimation, control/supervisor periods and latency semantics |
| command_sensor_envelope | Commands and external structured sensor fields, units/frames, bounds, validity/staleness/dropout assumptions |
| runtime | Exact relevant simulator/inference dependency and implementation identities |
| evaluation_protocol | Immutable numerical protocol, gates, seeds and execution scope |
| evaluation_report | Original decision and evidence reference, including rejected results |

The API verifies bytes, not the completeness or truth of their descriptions.
Unknown schema/fields, malformed hashes/counts, missing/altered files, traversal
and symlink references fail closed. Distinct mirror paths are allowed. No
untrusted pickle/YAML execution is needed. Audit only quiescent retained roots;
this is point-in-time verification, not a filesystem lock or authenticated
provenance mechanism.

Comparison requires exact content identity for all six static specification
fields. Different checkpoint weights or evaluation reports do not themselves
imply a mechanics mismatch. It never unions command envelopes, infers model
identity from filenames, migrates observations or trusts an `accepted` value
inside a report. Multiple mismatches are returned in fixed order.

Even `declared-spec-match-only` sets all of these to false:

- Descriptor-to-runtime binding verified.
- Behavioral retention verified.
- Policy acceptance.
- Transition authorization.
- Physical motion authorization.

Tests use explicitly synthetic fixtures. **No real skill descriptor has yet
been generated or admitted.** This helper is not wired into a live policy
selector; there is no selector launch this window.

Local validation:60 focused synthetic tests passed; the foundation, command,
motor, obstacle-supervisor and retention regression set passed950 tests with
three existing actuator/site warnings. `git diff --check` passed. Ruff is not
installed in the frozen local environment; no dependency was installed to run
it. Implementation/evidence commit
`7401415190f2e928caa65dd7ee769b35364812c5` was pushed to the fork. After
confirming a clean exact branch, no running Duck service/compute PID and both
protected SYSTEM services inactive,100.100 fast-forwarded to that exact commit.
Its CPU-only compatibility/yaw/migration/inference regression passed113 tests
in8.47s; the remote worktree remained clean. This verifies source delivery and
CPU checks, not any live policy switch or behavioral retention.

## Remaining bounded CPU slices

1. Derive one artifact-specific descriptor from retained source/config/model
   evidence. Do not use fabricated hashes or infer effective dynamics from an
   action count. Missing historical information remains an explicit unknown,
   not a default match. Add focused extraction and tamper tests.
2. Inventory the preserved rigid locomotion, obstacle and rejected K3900 hop
   artifacts separately. Verify checkpoints and retained rejection decisions;
   keep different mechanics in separate groups. Bind interface ordering and
   normalizer restoration using the existing CPU inference audit where it
   applies; do not extend its61D assumptions silently to other actors.
3. Implement a CPU retention-plan checker that requires original per-skill
   protocols and explicit directed transitions, interruption/stop cases and
   missing/stale sensor handling. It may report missing evidence but cannot
   manufacture a behavioral pass. No GPU retention collection this window.

## Slice2a: real F1-Y artifact binding, explicitly incomplete

`mjlab_microduck.retained_skill_binding` now audits the closed yaw fixed8998
artifact without launching simulation. It pins the original F1-Y manifest,
checks all105 payloads and exact inventory, then links the checkpoint, saved
environment/agent YAML, launch/result/backend records and rejected503 report.
Weights are loaded CPU-only with `weights_only=True`; the known61-input,
14-output actor tensor layout, finite values, dtypes and8998/216000 counters
are checked without running inference. The motor-column list in the original
report is retained as a recorded list, **not** promoted to proof of action order.

Important extraction finding: the YAML stores `spec_fn` and observation
functions in Python tags, with empty scalar values. A plain BaseLoader mapping
therefore loses their callable names. The new parser uses node composition,
retains tags as inert strings and preserves ordered entries/scalar spelling.
It never constructs Python objects, imports tagged factories or invokes them.
Duplicate/non-string keys, cyclic aliases, malformed/multiple documents and
excessive depth/expansion/size are rejected. Existing command-coverage audits
only use scalar bounds, so their original results are not changed by this.

Saved actor term order is `base_ang_vel`, `projected_gravity`, `joint_pos`,
`joint_vel`, `actions`, `command`, `head_command`, `body_command`. Saved timing
is0.005s physics with decimation4, matching the report's0.02s control period.
These are retained declarations, not resolved historical model equivalence.
The saved factory tag names `microduck_constants.get_walk_spec`; no standalone
effective compiled-model/asset inventory was produced by this campaign.

The output decision is `partial-artifact-binding-only`, with `descriptor=null`.
Five explicit unresolved requirements remain:

1. Historical effective compiled model and asset inventory.
2. Resolved actor columns and preprocessing binding.
3. Resolved action/joint ordering, scaling, offset and reset binding.
4. Complete runtime and actuator implementation binding.
5. Accepted per-skill and transition-retention evidence.

No synthetic placeholder fills these gaps. All admission/transition/retention
flags remain false. A later CPU reconstruction must be labeled reconstructed
and compared against historical evidence, not presented as a captured training
model. No controller switch or new training follows from this report.

Mac output: `artifacts/audits/f1y-retained-skill-binding-v1.json`,130,409 bytes,
SHA256 `62a3bf5a34abda909246205011a54c1c7261f571ba7ddc9e5afce2a8ea502bf2`.
The output records exact extractor/helper source hashes and never writes inside
the closed evidence root or overwrites an existing report. Local regression
995 passed,3 existing actuator/site warnings, including45 binding tests and
the real retained-artifact check. Implementation commit
`6d60bf13764e7db0971edf26c1a9ed5c286cb45c` was pushed and fast-forwarded on
100.100 only after clean-branch, idle-GPU and inactive-protected-service checks.
Remote focused regression158 passed in9.07s. Its independently generated
`artifacts/audits/f1y-retained-skill-binding-v1.json` has exactly the same
SHA256 and bytes as the Mac report. GPU remained0%,45C,12MiB; no training,
simulation rollout, service restart or skill transition was performed.

Next CPU step: inspect the historical robot/actuator assets and resolved mapping
evidence behind these gaps. Preserve the partial report unchanged; any later
reconstruction or expanded inventory gets a separate versioned artifact, not
an overwrite or an automatic upgrade to a complete descriptor.

The later behavior sequence remains foundation gait, stop/recovery, frozen-gait
structured obstacle supervision, one-axis obstacle diversity, separate hop
repair, then matched-mechanics transitions. Avoidance can slow within the
maneuver zone; nominal speed tracking applies before and after. Every new
composed controller needs all earlier admitted skill matrices rerun. H1-T is
still rejected: no claim that hopping has been retained or completed.

Future training needs a separately approved window and predeclared hypothesis,
not another experiment squeezed into this exhausted three-pilot budget.

## Slice2b: historical source and current CPU robot reconstruction

`robot_source_reconstruction.py` verifies working bytes against Git blob IDs
from the exact training source2614d09 before importing its config factory.
Coverage is97 historical project Python files, the robot XML and all38 directly
referenced meshes:136 files,23,193,150 bytes. Includes, unknown file-bearing
assets, traversal, duplicate meshes and source symlinks are refused rather
than silently omitted. These files are recoverable source evidence, not a
captured historical compiled model.

The rebuilt config's tagged robot, simulator, ordered observations, actor,
actions, commands, clipping and timing declarations exactly match slice2a.
Only `Entity` robot construction and MuJoCo CPU compilation are performed;
the saved simulator options are applied to that robot-local model. No scene,
environment reset, randomization, runtime actuator initialization, actor
inference, physics step, optimizer update or CUDA initialization occurs.

Observed robot-local topology is nq21/nv20/nu14,16 bodies,75 geoms and38 meshes.
The existing direct-unit-gear hinge validator resolves14 motors in the recorded
order, with local joint indices0–13. This is a reconstructed motor mapping,
not yet the complete resolved policy-action/preprocessing pipeline.

The report retains selected finite compiled arrays and227 current dependency
file hashes: BAM/mjlab Python sources and the actual bundled XL330 M6 JSON.
The distribution name is `better-actuator-models`1.0.1, distinct from import
name `bam`; an initial metadata lookup test exposed that naming difference
and was corrected after inspecting installed metadata and the lockfile.
Remaining package versions: mjlab1.3.0, MuJoCo3.10.0, MuJoCo Warp3.8.1,
torch2.9.1 and Warp1.12.0. Native binary/runtime state is not fully inventoried.

| Retained identity | SHA256 |
| --- | --- |
| Historical robot XML |`07af5e482200dfd2a7bf80ddde371153c88768a3a5aebc2f58ea42136fc7dadb`|
| Current bundled BAM XL330 M6 JSON |`61c699362fb3fabdde93eeba5e1ad3bf4ef9ca2f71d03e316b1924ff005b20d3`|
| Mac reconstruction report |`cc75a3e26d7fd2aeb8a724a3d0089aebe5764cffe5e622fd550aed85a10531d8`|

New output `artifacts/audits/f1y-robot-only-cpu-reconstruction-v1.json`,91,868
bytes on the Mac. Its decision is
`reconstructed-robot-only-not-historical-equivalence`; every historical-model,
policy-action-pipeline, behavioral-retention and admission flag remains false.
The slice2a report and all closed campaign payloads remain unchanged.

Local validation:24 focused reconstruction tests and1019 regression tests
passed, with four occurrences of the existing actuator/site warning. Next
verify the same source on100.100 and retain its independent CPU reconstruction.
Compare source/assets/config/mapping identities exactly; retain any compiled
numeric differences descriptively, without a tolerance-based policy pass or
another GPU experiment.

### Independent100.100 reconstruction

Source `d9eb4f70ddb86f4cbdf3cb49b77d6bc7e8cf83c3` was pushed and
fast-forwarded only after a clean exact branch, no Duck process/GPU workload
and inactive protected SYSTEM services were verified. Remote focused182 tests
passed in9.28s with one existing actuator/site warning. The independent Linux
report is91,870 bytes, SHA256
`59b801970e97036c1bf66be345ac46855ab8f6b71c2dd532c0e383f13e4fd1eb`.
It is retained at the same audit-relative path on100.100 and mirrored without
overwrite as `artifacts/audits/f1y-robot-only-cpu-reconstruction-v1-linux.json`
on the Mac. GPU remained idle0%,45C,12MiB.

Read-only exact recursive comparison found **11 differing scalar leaves**,
all in `reconstructed_robot.arrays.body_inertia`. Every other field matches
exactly, including136 source identities,227 dependency hashes, package
versions, configuration-match evidence, dimensions, names, indices, timing
and the other selected compiled arrays. Inertia differences are at indices
`[1,1]`, `[1,2]`, `[3,0]`, `[3,2]`, `[4,0]`, `[6,1]`, `[10,0]`, `[10,2]`,
`[12,0]`, `[12,2]`, `[13,0]`. Maximum absolute difference is
`1.0842021724855044e-19` at `[10,0]`: Mac
`0.0003210675976941665`, Linux `0.0003210675976941666`.

This is descriptive cross-host compiler output, not a tolerance-based pass,
established cause of the earlier rollout divergence, bit-exact compilation
claim or historical-training-model equivalence. Neither report is normalized
or overwritten to hide the differences. Both remain outside skill admission.

Next bounded CPU slice: resolve the reconstructed policy action pipeline
(ordered targets, scale/default offsets and reset behavior) with synthetic
CPU state and the actual installed manager, while keeping that evidence
separate from the historical randomized actuator state and behavioral gates.

## Slice2c: installed action manager, synthetic CPU state

`action_pipeline_audit.py` uses the actual `Entity` target resolver/writer,
`ActionManager` and `JointPositionAction`, but supplies only isolated synthetic
CPU buffers for default positions, encoder bias and target positions. An
already-initialized Entity is refused. No actuator execution, scene/environment
reset, physics, policy inference or CUDA initialization takes place. Original
source/assets and all105 campaign payloads are verified before the audit; the
rebuilt declarations match the immutable slice2a report.

Fourteen positive/negative one-hot probes establish the reconstructed input
column-to-joint order. Additional probes confirm the saved configuration's
scale1, construction-time default offset and no manager-level clipping. The
saved runner clipping configuration is also None; the wrapper itself is not
executed in these probes. The actual target transform is:

```text
position_target = raw_action * 1 + default_joint_position_at_construction
                  - current_encoder_bias
```

Encoder bias is read at each apply call; the default offset is a clone captured
when the term is constructed, not a live view of a subsequently changed
default-position buffer. The probes preserve caller actions and torch RNG.

| Sequence tested in CPU buffers | Observed result |
| --- | --- |
| Process zero, then apply | Configured posture minus encoder bias, not zero joint angles or a proven safe stand |
| Selective/full manager reset | Selected raw/current/previous histories clear; processed-target cache remains |
| Apply immediately after reset, without new processing | Previous processed target is reused; this is a deliberate out-of-order diagnostic |
| Process fresh zero after reset, then apply | Stale processed target is replaced with the configured posture/bias target |

The normal installed environment's **source ordering** processes fresh actions
before the decimation/application loop. That source check is separate from
the executed manager probes; the full environment is not executed here.
Consequently, the reset-only observation is not evidence that ordinary
training applied stale targets or an explanation of the F1-Y rejection.

Future skill transitions must include a fresh processed command before the
next target application, plus separate actuator-delay/state and observation
history handling. A manager reset or zero action is not a validated stop or
stable landing. No production manager/reset code has been changed, and the
earlier reconstruction/binding reports are not upgraded or overwritten.

Mac output `artifacts/audits/f1y-action-pipeline-synthetic-cpu-v1.json`,30,706
bytes, SHA256
`c453ace1ccef3e01379d854b6f766600f21ce34fff252127aeadbfc40aba5107`.
Decision `synthetic-action-pipeline-only-not-retention`; full environment reset,
historical randomized actuator state, safe-stop, closed-loop retention and
transition/admission flags remain false. Local regression1038 passed, including
19 action audit tests;17 occurrences of the existing actuator/site warning.
Implementation `a36f9c25623a3ea031da3cf84874c92f21f6749a` was pushed and
fast-forwarded on100.100 after clean-branch/idle-GPU/inactive-protected-service
checks. Remote focused201 tests passed in9.64s with14 occurrences of the same
actuator/site warning. Its independently generated audit report matches the
Mac SHA256 exactly. GPU remained0%,45C,12MiB and the worktree stayed clean.

Next bounded CPU slice: a fail-closed retention-plan checker and evidence-gap
inventory. Require original per-skill numerical protocols, compatible mechanics
and interfaces, directed transitions, fresh command processing, actuator-delay
and observation-history handling, interruption/stop cases and stale/missing
external sensor behavior. Do not mark a skill admitted from these synthetic
probes or start collecting new GPU rollouts. The plan must identify missing
evidence explicitly; accepted walking/hopping or safe-stop behavior cannot be
manufactured from a saved checkpoint, source-level order or manager reset.

## Slice3: retention-plan coverage and explicit unbound evidence

`mjlab_microduck.skill_retention_plan` is a standard-library-only checker. It
reads and verifies retained file references, but never loads a policy, runs
physics, evaluates a numerical report, or admits a skill. The tracked initial
plan is `2026-09-08-retention-plan-v1.json`. Its entries are deliberately empty:
the earlier F1-Y partial binding and synthetic action probes are not complete
skill descriptors or behavioral-retention evidence.

The fixed coverage catalog requires:

- Four separate skill records: foundation, stop/recovery, obstacle, hop. Each
  needs a byte-verified descriptor, its original numerical protocol, a composed
  controller retention procedure and the resulting retention report. A supplied
  original protocol must have the same bytes/hash as the descriptor's protocol.
  The new retention report need not equal the original checkpoint's report.
- All12 directed requests between those four skills, each with procedure and
  report. Reverse direction is independent. This is a request matrix, **not**
  permission to switch directly: incompatible or unsupported requests need an
  explicit refusal or a separately tested route through stance/landing.
- Sixteen cross-cutting cases: fresh commands/actions before application;
  actuator delay/internal state; action history/processed cache; observation
  history/normalizer handling; stop/stance and restart/speed reacquisition;
  interruption before takeoff, in air and during landing; stale, missing and
  invalid structured input plus recovery; permitted slowdown inside avoidance;
  nominal speed before/after; per-placement/speed-bin/worst-seed reporting; and
  independent training with untouched confirmation seeds.

An interrupted airborne hop cannot be treated as an instantaneous zero-action
stop. Its procedure must address abort/landing and subsequent stable stance.
These procedures must cover the applicable directed requests and phases, not
just mention a generic reset. The checker audits references, not the meaning
or completeness of text inside a referenced procedure; that remains review and
eventual numerical work. No new thresholds replace any original F/S/O/H gates.

Absent/null known references produce explicit gaps; unknown keys, malformed
references, tampered bytes and symlinks are errors. A complete descriptor pair
is compared on all six existing static fields. Missing descriptors stay
unknown; rigid-versus-sprung or interface differences cannot become a match
from a shared14-action dimension. A complete *reference* catalog can still
contain mismatches and is never a behavioral pass. Every result keeps original
gate verification, runtime binding, retention, safe-stop, policy acceptance,
transition, GPU collection, training and physical-motion flags false.

The initial plan has72 **unbound reference slots**, not72 failed behavioral
tests and not a claim that no historical protocols or reports exist. Original
protocol/evidence retrieval and actual per-skill descriptor derivation remain
unfinished. Four semantic gates stay unresolved even for fully populated
synthetic test fixtures. The report records its canonical plan hash and the
checker/descriptor-verifier source hashes. JSON duplicate keys, non-JSON numeric
constants and inputs over1MiB are refused; CLI outputs are exclusive-create.

Reproduce without a GPU or rollout, from the repository root, writing only to
an existing audit directory outside the closed campaign roots:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m mjlab_microduck.skill_retention_plan \
  --root . --plan docs/experiments/2026-09-08-retention-plan-v1.json \
  --output artifacts/audits/skill-retention-plan-v1.json
```

This is still the CPU-only remainder of the exhausted3/3 pilot window. Next
bounded work may bind original per-skill protocols and inventory historical
artifacts, preserving rejected status and mechanics groups. Do not populate
missing behavioral evidence with synthetic action tests, rerun GPU evaluations,
or launch another training revision before a separately authorized window.

Local validation:59 new focused tests and1097 regression tests passed, with17
occurrences of the existing actuator/site warning; `git diff --check` passed.
Initial audit `artifacts/audits/skill-retention-plan-v1.json`,9847bytes, SHA256
`092352bc4287b74bb9b7c7cc4a934059eef8a906be1a0e1b8efd462e7e86da05`,
decision `incomplete-reference-coverage`. This artifact preserves all72 unbound
slots and the four semantic gaps. Source hashes and exact reference verification
make this reproducible planning evidence, not a new behavior evaluation.

Implementation `74949bb92ca708b7428423fbc095b98c93cdbe76` was pushed to the
feature branch and fast-forwarded on100.100 only after clean exact-source,
no-running-Duck/no-compute-PID and inactive-protected-service checks. Remote
focused207 tests passed in6.09s with14 existing actuator/site warnings. Its
independently generated9847-byte report matches the Mac SHA256 exactly. GPU
remained0%,45C,12MiB, both protected SYSTEM services inactive and the worktree
clean. No controller, actuator, GPU rollout or training service was launched.

## Slice4: retained H1-T evidence and original protocol inventory

`retained_hop_inventory.py` audits a selected11-file set: H1-T5999 checkpoint,
saved environment/agent YAML, all six historical evaluation JSONs, paired
H1-P5999 JSON and original H1-T/H1-P comparison. The checkpoint and eight
decision/report hashes match the September5 ledger. The two YAML hashes are
new quiescent inventory observations, not independent historical provenance.
No checkpoint is deserialized and no YAML object is constructed in this audit.

The tracked inventory is `2026-09-08-h1t-retained-inventory.json`, SHA256
`dc2d351c73e1b16c2b75d3ff2a76a04dea8e66d8748986ec0c14510b5b65a4cb`.
All11 selected payloads,4,967,886bytes, are mirrored on the Mac under
`artifacts/retained/h1t-seed67-v1/`, preserving their original repo-relative
paths. Remote originals are unchanged. This is not a complete training or
evaluation-directory backup: intermediate checkpoints, baseline checkpoint
bytes, CSVs and raw rollout samples are outside this selected inventory.

Before importing the old CPU summary reducers, the audit verifies current bytes
of `hop_evaluation.py`, `hop_revision_gate.py` and
`scripts/evaluate_hop_checkpoint.py` against Git blobs at
`aa48f178cc9cb5791ee95e520cf6ef9fc14b2456`. It does not execute the rollout
script. The original seeds211/223/227,128 environments and six cycles remain
unchanged. All six full H1 summary decisions reproduce as rejected. The paired
final causal decision reproduces as `stop`, including every metric, criterion,
status, report hash and checkpoint identity. Only the two reader-local report
locations are normalized to relative paths for cross-host comparison; original
stored host paths are checked first and no checkpoint metadata is normalized.

At5999, the original full H1 failure set is:

| Gate | Retained observation | Original full H1 criterion |
| --- | ---: | --- |
| Minimum episode pass fraction | 0 | At least0.80 |
| Total falls | 4 | Zero |
| Maximum per-seed drift p95 | 0.476065069437027m | At most0.10m |
| Maximum rated-speed exposure | 0.005710565485060215 | Zero |
| Maximum near-stall fraction | 0.0030487352050840855 | At most0.0025 |

The separate causal revision gate still fails episode pass above zero, drift
below0.30m and rated-speed non-regression. Its looser drift diagnostic must not
replace full H1 acceptance. Likewise the old sprung H1 torque ceiling0.90 is
not the rigid foundation ceiling0.60: do not pool those mechanics or import
the looser limit into an integrated controller. The measured99.35% minimum
cycle success is evidence of hopping events, not an accepted hopping skill.

Output `artifacts/audits/h1t-retained-summary-inventory-v1.json`,26,182bytes,
SHA256 `5a1cbe0dff6fc8e644d230ef2fe0cd6c2d785a31de9f512a797d1bd67666ea7a`.
It keeps `descriptor=null`, identifies the declared sprung-K3900 group and
leaves effective mechanics, raw rollout reconstruction, historical config
identity, behavioral retention and every admission/transition/training/motion
flag false. The earlier72-slot retention plan is not filled with this rejection
inventory: a historical summary audit is not a composed-controller retention
report or a complete descriptor.

Reproduce using the Mac mirror as `--root`, or the remote repository as `--root`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m mjlab_microduck.retained_hop_inventory \
  --root artifacts/retained/h1t-seed67-v1 --repo . \
  --output artifacts/audits/h1t-retained-summary-inventory-v1.json
```

No H1/Locked/H2, video, new GPU run or policy switching is authorized. Next
bounded CPU slice: inventory the historical obstacle/stop-recovery protocols
and retained terminal decisions, keeping prerequisite failures separate from
skill acceptance and preserving the frozen low-level actor identity.

Local validation:23 new inventory tests,32 combined hop/inventory tests and1129
regression tests passed in42.69s with17 occurrences of the existing actuator/
site warning. Tests cover all11 payload tampering cases, manifest replacement,
exact non-path causal comparison, retained rejection and refusal to deserialize
checkpoints. `git diff --check` passed. No new raw rollout samples were collected.

Implementation `662dce5636d4c5f97e45a1d919f920d4bd2f29a8` was pushed and
fast-forwarded on100.100 after clean-source, idle-GPU, no-running-Duck and
inactive-protected-SYSTEM-service checks. Remote151 focused CPU tests passed
in1.23s. Its independently produced26,182-byte audit matches the Mac SHA256
exactly. GPU remained0%,45C,12MiB with no compute PID; both protected services
remained inactive and the worktree clean. This verifies inventory delivery and
summary reconciliation, not a new hop rollout or behavior acceptance.

## Slice5: frozen obstacle/recovery baseline and terminal boundary

`retained_recovery_inventory.py` verifies the closed seed379 eight-file
experiment and three checkpoint references: original motor-aware7998 gait,
near HC4-R2 supervisor and far HC4-LH supervisor. All11 selected payloads,
5,020,145bytes, are mirrored on the Mac under
`artifacts/retained/recovery-seed379-v1/` with original repo-relative paths.
The earlier eight-file diagnostic mirror and remote originals remain unchanged.
Checkpoint hashes match the original launch receipt; the eight experiment
hashes match the September6 ledger. No weights are deserialized.

Tracked inventory `2026-09-08-recovery-retained-inventory.json`, SHA256
`6d49fa4a5af7ef18018c8fc8a25c5f175f7ca03203ad5660ca75b1588e3be43f`,
also pins nine selected validator/controller/observer/rollout source files to
Git blobs at `bd2a20a6232f647e656a2dfe6788b037e29a06b1`. This is not a complete
transitive dependency or historical runtime binding. The audit constructs the
expected argv without executing it, verifies the exact original frozen models,
single child index0/exit0 receipt and uncapped near-A1 command, then runs only
the existing pure `recovery_ab.evaluate_paths` reducer. Its canonical result
equals both original decision receipts exactly. The closed evaluation directory
must contain precisely its eight files: an extra A2/B report or symlink is an
error, not permission to extend this failed prefix.

The historical result remains `numerical-gate-stop`, solely `recovery-window`.
Eight clean obstacle passages, zero collisions/falls/timeouts and legacy pooled
torque p99 of0.5060326457023621 do not clear the recovery requirement. All eight
entered recovery; none held0.30+/-0.03m/s for the sampled0.50s span within2s.
All eight are `window-missed`, not unobserved or censored. Mean recovery speed
was0.20347188413143158m/s over1382 samples. Approach mean0.03261695057153702m/s
has only15 startup-inclusive samples and must not be reported as steady-state
approach performance. Interaction mean0.19954714179039001m/s is descriptive;
the curriculum intentionally permits slowdown during the maneuver.

The gait and near supervisor were used by this one retained child. The far
supervisor was hash-verified as a declared reference but **not evaluated** here.
No A2 repeat, capped-B comparison, later placement/speed cell or recovery-PPO
pilot ran. The actual decision is a failed prerequisite, not proof that a cap
does or does not improve recovery. Standstill, safe stop/restart, interruptions
and stale/missing sensor behavior were not evaluated. The report used exact
structured geometry with zero noise/dropout; no raw perception is involved.

Output `artifacts/audits/recovery-retained-inventory-v1.json`,11,321bytes,
SHA256 `0fd893d5b1459d9d62a7ea64a28244c0b2e156e41d034af3411686378417c522`.
Its descriptor stays null and its declared rigid-locomotion mechanics group
stays separate from sprung H1-T. Runtime/mechanics binding, raw reconstruction,
behavioral retention and all admission/transition/training/motion flags remain
false. Retention-plan slots are not filled with this failed baseline report.

Reproduce using the Mac selected mirror, or use `--root .` on100.100:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m mjlab_microduck.retained_recovery_inventory \
  --root artifacts/retained/recovery-seed379-v1 --repo . \
  --output artifacts/audits/recovery-retained-inventory-v1.json
```

Focused local validation:23 new inventory/receipt tests plus the existing
recovery suites passed132 tests. Checks include every selected payload's bytes,
extra/symlinked evidence, changed source/model/command, extra children, original
failure preservation, and refusal to load checkpoints or call the launcher.
The broader local regression passed1152 tests in41.04s, with17 occurrences of
the existing actuator/site warning; `git diff --check` passed.

Next bounded CPU slice: consolidate the evidence-backed curriculum readiness
matrix and explicit missing acceptance cases. Keep F speed/motor repair before
further obstacle progression; do not reinterpret clean passage as retained
recovery, standing or hopping. No further GPU experiment in the consumed3/3
window, and the07:00 Shanghai cutoff/protected-service rules remain unchanged.
