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
it. Remote CPU validation remains a separate source-delivery check.

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

The later behavior sequence remains foundation gait, stop/recovery, frozen-gait
structured obstacle supervision, one-axis obstacle diversity, separate hop
repair, then matched-mechanics transitions. Avoidance can slow within the
maneuver zone; nominal speed tracking applies before and after. Every new
composed controller needs all earlier admitted skill matrices rerun. H1-T is
still rejected: no claim that hopping has been retained or completed.

Future training needs a separately approved window and predeclared hypothesis,
not another experiment squeezed into this exhausted three-pilot budget.
