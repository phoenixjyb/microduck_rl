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
the real retained-artifact check. Remote reproduction is a separate gate.

The later behavior sequence remains foundation gait, stop/recovery, frozen-gait
structured obstacle supervision, one-axis obstacle diversity, separate hop
repair, then matched-mechanics transitions. Avoidance can slow within the
maneuver zone; nominal speed tracking applies before and after. Every new
composed controller needs all earlier admitted skill matrices rerun. H1-T is
still rejected: no claim that hopping has been retained or completed.

Future training needs a separately approved window and predeclared hypothesis,
not another experiment squeezed into this exhausted three-pilot budget.
