# External obstacle freshness: non-wired CPU contract

This addresses the source-design gap in the
[readiness matrix](2026-09-08-curriculum-readiness.md), not a demonstrated sensor
failure on a real duck. The prototype is
[`obstacle_freshness_contract.py`](../../src/mjlab_microduck/obstacle_freshness_contract.py).
It classifies declared metadata and names required retention-case families.
It does not synchronize clocks, validate perception, change controller phases,
update receiver history, deserialize policies, launch simulation or emit motion.
No production module imports it. Trained actor/supervisor dimensions and the
existing seven-channel obstacle packet remain unchanged.

## Metadata stays outside the actor

The caller supplies a receiver-owned expected context. Incoming packets cannot
silently replace it. The exact envelope fields are:

| Field | Required meaning |
| --- | --- |
| `source_id` | Expected external estimate publisher |
| `stream_epoch` | Explicit publisher stream/restart identity |
| `episode_id` | Expected environment/route episode identity; reset cannot inherit an old packet |
| `timebase_id` | Receiver-owned monotonic domain after an independently validated clock mapping |
| `sequence` | Nonnegative integer sample sequence within that context |
| `captured_ns` | Acquisition time in the declared receiver domain, not sender wall time |
| `first_received_ns` | Original first receipt in that same domain; not refreshed on polls/retransmission |
| `estimate_status` | Exactly `detected`, `clear` or `invalid`; ambiguous no-detection must not default to clear |
| `payload_sha256` | Lowercase64-hex identity of the associated payload, whose bytes are not checked here |

Context labels, times and a formatted digest are declarations, not proof of
clock synchronization, publisher authenticity, payload binding or a physically
clear route. Those gates remain false. A future adapter must define the exact
payload byte encoding, frames/units, empty/clear semantics and field-of-view
coverage, then verify them before this metadata could reach a controller.

## Timing and history rules

`assess` requires both `max_capture_age_ns` and `max_receipt_age_ns` as positive
integer inputs. **There are no operational default limits.** Calibration,
transport latency and a separately predeclared stress envelope must justify
them before integration. Both ages are checked independently. Equality to a
bound is within the metadata window; one nanosecond beyond is stale. A newly
received old capture is still stale when its capture-age bound is exceeded.

Require `captured_ns <= first_received_ns <= now_ns`. No sender/receiver clock
offset is guessed and no future timestamp is silently clamped. A receiver
clock moving behind `last_check_ns` returns a clock-regression classification.
Caller context/configuration/history errors raise; malformed current packets
return an untrusted classification. Unknown fields and boolean/negative/float
timestamps or sequences are not silently coerced.

The caller owns persistent history; this function never advances it:

- `previous` is the latest structurally and temporally valid, ordered envelope
  in the exact context, including stale packets and invalid estimates. A newer
  invalid estimate must prevent fallback to an older good packet.
- Lower sequence numbers or newer sequences with backward acquisition/receipt
  times are rejected. Same-sequence packets must match every immutable field.
  A changed digest, status or timestamp is a duplicate conflict, not new data.
- Rechecking an identical cached packet is allowed, but its age grows from the
  original timestamps. No-new-arrival at a control tick is represented by that
  cached envelope, not a new receive time. `packet=None` means no sample is
  available; it does not manufacture a fresh clear observation from history.
- Fresh/stale/invalid-estimate results may supply the next ordered watermark to
  a future receiver implementation. Missing, malformed, identity/time/order
  faults and duplicate conflicts must not replace it. Advance the check-time
  watermark on non-regressed checks, including failures; retain it on rollback.
- Explicit startup, restart, episode changes and stream rebinding need their
  own trusted handshake/history-reset contract. Never adopt a packet's new
  epoch or erase history merely because a check failed.

The prototype does not implement that receiver state machine or authorize
automatic resumption. In particular, a fresh packet after a fault only passes
metadata classification; reacquisition and route/phase continuity remain
separate tested requirements.

## Classifications and per-phase evidence

| Classification | Interpretation, not control authority |
| --- | --- |
| `fresh-detection-metadata` | Declared detection is within both age bounds; payload/truth still unverified |
| `fresh-clear-metadata` | Publisher explicitly reports clear with current metadata; not inferred from silence |
| `invalid-estimate` | Publisher reports an invalid estimate even though metadata age is current |
| `stale` | Capture and/or original receipt age exceeded its supplied bound |
| `missing` | No available packet |
| Schema/identity/time/order faults | Untrusted input; never a fresh-clear substitute |

Fresh metadata names a payload/producer-semantics validation case family.
Untrusted input names a different family depending on phase:

- Approach, interaction, recovery or stance: grounded sensor-loss stop and
  reacquisition evidence. This does **not** change the frozen controller's
  intentional route-driven recovery on invalid geometry after an observed pass.
- Pre-takeoff: interruption-before-takeoff evidence.
- Airborne: interruption/landing-before-stop evidence, not instant zero actions.
- Landing: landing/recovery interruption evidence.

These are future test obligations, not a dispatcher, a tested stop policy or
entries automatically admitted into the72-slot retention plan. Every result
has `motion_command=None`; safe-stop, policy/transition/training/physical
authority, clock/payload/producer binding, runtime wiring and operational-limit
selection remain false. Even a fully fresh packet grants no motion authority.

## Synthetic verification and remaining work

The48 focused tests use toy integer nanoseconds solely to probe boundaries.
Their100/80ns limits are deliberately synthetic, not a recommended sensor rate,
latency budget or safety setting. Coverage includes independent age limits,
delayed capture, immutable cached polls, duplicate changes, out-of-order samples,
new invalid packets followed by old good ones, identity/epoch/episode/timebase
mismatch, future/rolled-back time, malformed fields and all seven phase families.
No tensor, camera, simulator, checkpoint or actuator is involved in this module.

Next CPU-only work may review the receiver lifecycle and write additional
adversarial contract tests. Actual payload/clock binding, numerical age limits,
receiver integration and closed-loop interruption/reacquisition testing remain
unimplemented or unvalidated. No training/GPU job is reopened; stop all new Duck
work at07:00 Shanghai and retain the unchanged protected-service boundary.

Validation:48 prototype tests passed; the focused freshness/obstacle/retention
set passed138 tests. The broader local regression passed1200 in43.66s with17
occurrences of the existing actuator/site warning. `git diff --check` passed.
Programmatic Markdown-to-HTML verification confirmed two tables (10 and7 rows,
two cells each) and both links resolve. No screenshot-based UI check was made.
A source search found no production importer of the new module.

Source `bf56a065896d80afb4969f7ef8d256265f1159df` was pushed and fast-forwarded
on100.100 after exact clean-source, idle-GPU/no-running-Duck and inactive-
protected-service checks. The same138 focused CPU tests passed there in5.69s.
GPU remained0%,45C,12MiB with no compute PID; both protected SYSTEM services
remained inactive and the worktree clean. No receiver or policy was deployed.

## Lifecycle review: test bookkeeping is not a deployed receiver

Seventeen additional sequence tests use a test-only history fixture to exercise
the documented packet and check-time watermarks. They cover repeated cache
polling through expiry, missing/new data in every phase, newer invalid/stale
samples blocking old good ones, conflicting retransmissions, unsolicited
context changes, explicit test episode rebinding, future timestamps and receiver
rollback after malformed input. The classifier remains pure; production code
is unchanged. No operational latch, handshake, clock mapper, receiver loop or
automatic controller resumption is implemented by this fixture.

These65 combined freshness tests preserve `motion_command=None` and false
authority flags throughout each sequence. A later fresh classification after
an error remains metadata-only: a production fault latch/reacquisition decision
needs separate implementation and closed-loop evidence. This review does not
fill retention-plan references or reopen the three-pilot budget.

Local validation after the lifecycle review:65 combined freshness tests and1217
broader CPU regression tests passed (42.94s,17 existing actuator/site warning
occurrences). No production source changed in this lifecycle slice.
