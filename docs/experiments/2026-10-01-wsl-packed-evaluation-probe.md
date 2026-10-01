# WSL packed held-out evaluation timing probe

Predeclared October 1, 2026, **before this probe's execution**. The active CLI
authority ends at **18:00 Asia/Shanghai / 10:00 UTC** today. This is not a full
evaluator, a new training recipe, a checkpoint promotion or a capability claim.
Keep active learners frozen at their qualified source; install this later source
only after every owned learner has completed and its full archive is durable.

## Entry gate and fixed case

The [packed replication](2026-10-01-wsl-packed-replication.md) archives remain
separate from the probe's current evaluator/runtime source. Authenticate one
**complete** packed training archive: successful child/report, exact inventory
and every file hash, initializer plus all 256 exports, 6,144 ordered tick records,
256 finite optimizer receipts and all four common checkpoints. No partial final
checkpoint or legacy training archive is a substitute. The frozen host, stack,
plant and checker must still match the completed training launch.

Use the new [probe-only runner](../../src/mjlab_microduck/stance_packed_evaluation_probe.py),
not the historical full evaluator. Select training seed **577**, first in the
predeclared order, checkpoint **255**, evaluation seed **541**, **128 worlds**,
one nominal first attempt each, up to **250 ticks / 5 seconds**. An attempt is
never reset to obtain a second chance. Use the original scorer and strict
**0.0873-radian** gate; do not score this one case as the full matrix.

The full held-out protocol stays **64/128/192/255 × 541/547/557 = 12 cases**,
1,536 attempts per training seed, requiring at least **122/128** in every
evaluation seed at a common checkpoint. All three training seeds remain in the
cross-seed denominator. Short cases may retain diagnostic evidence but cannot
size a full-length evaluation or be relabeled as complete timing measurements.

## Source, runtime and archive bindings

Protocol `football-b1n-wsl-packed-evaluation-probe-v1` has distinct source/seed
service and output prefixes. Its bundle uses the separate
`football-b1n-packed-evaluation-probe-trace-v1` protocol, which allows only the
fixed 255/541/128/CUDA case, with exact packed-mode and checker-SHA fields.
The strict replication checkpoint loader may restore it, but it is not a
registered full evaluation continuation. Historical binding bytes and loader
routes remain unchanged. The bundle binds evaluator source/compiled plant
separately from checkpoint training source, launch, report and checkpoint hashes.

Construct an actual eager packed runtime. Check its immutable mode, CUDA device,
disabled graph and checker source before collection and after every step.
Use the existing owned-case path, including strict restore, all boundary/contact/
motor/control checks, bundle publication and independent actor/trajectory replay.
No optimizer, policy export, perception, motor/contact/plant change or video.

## Fixed probe bounds, not borrowed full-evaluation timing

One retained user service only, after a fresh idle GPU/lease check:

- **600-second child / 960-second service**, independently checked; neither can
  be extended or retried automatically after an error or timeout.
- `KillMode=control-group`, `MemoryMax=6G`, `CPUQuota=200%`, `Nice=10`.
- 600-second closeout reserve and 60-second margin in a fresh at-most-hour
  absolute probe window wholly before today's cutoff.
- The existing shared FilmBrain lease, two-sample idle gate, live sole-owner,
  GPU memory/temperature and system/user protected-service checks. Preserve
  FilmBrain and unrelated workloads; never start competing GPU jobs.

These are an explicit bounded single-case experiment, **not** the historical
4090 full-evaluation cap or the WSL learner's 4,320/4,380-second training cap.
Nothing here establishes that the future twelve-case evaluator fits.
The probe-specific 600-second child wrapper leaves up to 360 service seconds
for parent replay/closeout, rather than letting a near-limit 900-second child
consume all but 60 seconds. Service setup also uses this allowance, so it is
not a promise that every probe finishes. The independent hard service bound
still closes a slow or hung replay. Historical 900-second wrappers are unchanged.

## Measured full-service sizing rule

Record the supervisor's monotonic time immediately before child launch and pass
it to the child in the same clock domain. Refuse missing, nonfinite, future or
out-of-budget starts. The first environment's `prelude_seconds` now includes
child interpreter/import startup, authentication, idle checks and file reads,
which the old case-local timer omitted. Record actual environment construction
and the full owned-case wall time separately.

After child exit, time **only the supervisor's additional independent replay**.
The owned-case time already includes its own immediate replay, so do not count
that replay twice. Retain observed total supervisor duration through idle
closeout and top-level hashing; the final report write is outside this observation
and covered by the fixed reserve. Recompute unattributed overhead as observed
duration minus the four nonoverlapping components, refusing backwards overlap.
That residual includes per-case receipt writes, logging and garbage collection
as well as one-off setup. A one-case probe cannot separate them: conservatively
project **all** residual work twelve times, not once. This intentionally
overestimates any truly one-off work instead of silently omitting repeated tails.

```text
unit = environment + owned_case + supervisor_independent_replay
unattributed_overhead = observed_probe - (prelude + unit)
prediction = prelude + 12 * (unit + unattributed_overhead)
service = ceil(1.25 * prediction)
child = service - 60
```

This is one full-length timing sample, not a maximum over multiple samples.
The fixed factor and explicit closeout reserve are unchanged. Require
`service + 600 + 60 <= 7200` before any later full-evaluation declaration, and
then separately require that whole budget fits actual remaining authority.
An oversized result records `timing-rejected-no-full-evaluation`; do not widen
the WSL ceiling, drop
cases, omit replay or reuse a different host's measurement. A fitting result
records `probe-measured-full-evaluation-disabled` and still leaves
`full_evaluation_enabled=false` until a reviewed source-bound
measured-cap handoff registers a distinct full evaluator.

## Retention and current state

Retain launch, plant, complete case bundle/manifest, case receipt, measurements,
child log/status/telemetry and timing report in an exclusive source/seed path.
Rehash and replay on WSL, then verify copied bytes independently on Mac; offline
consistency is not independent GPU authentication. The one-case scorer's values
are diagnostic only, not a matrix decision or football/hop/obstacle acceptance.
Offline verification also checks the exact successful child-receipt fields,
positive PID, integer zero exit, finite positive elapsed time within 600 seconds
and a telemetry list. Child elapsed cannot exceed observed supervisor duration,
which cannot exceed the 960-second service bound. Independently rehashing an
internally contradictory timeout/status receipt cannot make it acceptable.
Preflight refusal may produce no report because no child ran; a service-level
kill may leave a partial single-use archive. Both fail retained verification,
and neither authorizes reuse, retry or cap extension.

Implementation is complete. **78 focused CPU fixture tests** passed, followed
by **652 selected CPU checks** across the new probe and unchanged runtime,
qualifier, learner, archive, checkpoint, trace, control, plant, bundle and scorer
contracts. Markdown rendering/local links and `git diff --check` passed. Full
repository tests and live WSL probe/replay are separate, not established by
these fixtures. A bounded Luna test slice and read-only review were integrated;
review found repeated loop-tail undercount and parent replay headroom, corrected
before execution with conservative residual projection and the 600-second child.
An additional read-only timeout/closeout audit found an offline child-receipt
consistency gap, tightened before installation. The revised probe's **100
focused tests** and **339 selected CPU tests** passed, including rehashed
malformed/over-limit child receipts and contradictory child/service timing.
This changes only retained-evidence checks, not live caps, runtime or skill gates.

After all three learners completed, their full archives were verified on WSL and
the copied bytes independently replayed on the Mac. **339 selected CPU tests**
also passed on Linux in an isolated source-only validation worktree, explicitly
loading the new source with CUDA hidden and uninitialized. Fixture tests used
the default profile, not a globally forced WSL profile. Production does not use
the validation process's `PYTHONPATH` override or a rewritten environment.

The exact fork tip `d4e96cda975f943ac89aacbbed62e99f1c9f00d1` was transferred by
a hash-checked Git bundle, preserving the clean training source as detached
`artifacts/tools/runtime-packed-training-be2d59661af2`. The validation snapshot
is `artifacts/tools/probe-validation-d4e96cda975f`. These are source snapshots,
not separately restored binary environments. Only after Linux validation and
fresh frozen-host/two-sample-idle checks was the runtime fast-forwarded to this
exact evaluator tip; no dependency, asset, driver or service configuration changed.

The single probe launched at approximately **17:14 Shanghai** in
`microduck-wsl-packed-eval-probe-d4e96cda975f-seed-577.service`, invocation
`2fc1ba2aefd84ce5a1e3e14fc9488ace`, with unchanged **600/960-second** hard bounds,
6 GiB/200% CPU/Nice 10/control-group resources and deadline **1790848800**
(today 18:00 Shanghai). Independently checked launch SHA256:
`2d4284cc12b2f0347574bd2c566023b12ce7f646cc3843008da07488098a5739`.
It **failed**, not timed out, after **191.47832137392834 child seconds**, with
child exit 1. The user service is `failed / Result=exit-code / ExecMainStatus=1`,
MainPID 0. No Duck compute process remained; the GPU was idle at 35 C. FilmBrain's
two user services remained active with zero restarts and the protected AI services
remained inactive. No service, dependency, driver or training artifact was changed
to diagnose this failure. Keep this runtime's exact source frozen; subsequent CPU
repair commits are not a live GPU retry or runtime installation.

### Failed attempt retained, not usable timing evidence

The exact error was **`ValueError: fixed packed probe case`** along
`evaluate_owned_case -> write_bundle -> trace.encode -> trace.replay ->
FirstAttemptTrace -> validate_binding`. Replay replaced the declared CUDA capture
device with `cpu` before revalidating the strict packed binding. The CUDA-only
binding correctly refused that altered declaration. This is a replay-device
bookkeeping bug, not a numerical checkpoint rejection or evidence of a failed
learner. It occurs before bundle directory creation; the in-memory trajectory was
not retained and cannot be salvaged as a completed measurement.

Both WSL and Mac retain exactly these **four files**, independently rehashed,
with the report's three payload hashes checked against their bytes:

| File | SHA256 |
| --- | --- |
| `child.log` | `25d9ab1f295eed13898c55b1e196a143a07a7c3157642af2e231354b1a71187b` |
| `launch.json` | `2d4284cc12b2f0347574bd2c566023b12ce7f646cc3843008da07488098a5739` |
| `runtime.json` | `d790ead1c4ed5f8ba672ad58d9e3db8c4f06ced01ecfc0634d9fd986decf6e4a` |
| `report.json` | `b6279b74961a207d9c661e74983d75f471545bd300daf618a628411d058c5ea1` |

The archive is
`artifacts/evaluations/stance-wsl-packed-eval-probe-d4e96cda975f-seed-577`.
The report has decision `failed`, optimizer steps zero and every admission,
full-evaluation, skill and physical-motion flag false. Its error note retains
child PID 1413484, elapsed time and 152 telemetry samples. There is **no case
receipt, complete bundle, manifest or timing measurement**. This inventory/hash
check is a failed-attempt audit, not successful `verify_retained` evidence. Do not
derive full-service caps or a skill verdict from the failed child's wall time.
The actual copied archive was also supplied to `verify_retained` with its exact
independent hashes: it correctly refused it with `successful timing-only probe
report`, before case replay, with CUDA hidden and uninitialized.

### Bounded CPU repair and next gate

The repair preserves the original binding throughout replay and separates the
CPU checker's tensor device from recorded capture-device metadata. The public
capture constructor and live append still require the actual declared device;
packed bindings still require 255/541/128/CUDA and the packed checker fields.
Only the private CPU rehydration path uses CPU tensors without rewriting the
binding. Replay continues to report `provenance_validated=false` and all
admission/physical-motion flags false.

New regression fixtures exercise actual replay/encode/hash verification at 128
worlds and actual bundle publication plus strict checkpoint, actor, plant and
motor-control replay. Fixture CUDA-origin metadata is synthetic, not GPU capture
authentication, and the tests leave CUDA uninitialized. Malformed bindings,
CPU-authored packed capture and discontinuous traces remain refused. No original
scoring, motor/contact/plant contract, learner recipe or timeout is relaxed.

Validation of this repair: **53 trace tests** and the real packed-bundle regression
passed, then **474 affected CPU checks** passed together on Mac with CUDA hidden
and explicitly uninitialized. The first parallel broader run passed 420 checks
but one unchanged standalone-import subprocess exceeded its fixed 30-second
timeout during elevated Mac load. That exact test passed in isolation, and the
complete sequential 474-test run subsequently passed without a timeout change.
Focused Luna test development and read-only review were integrated; the owner
added/reviewed the actual bundle roundtrip. Full repository tests and repaired
live GPU timing have not been run.

Repair source **`05ff5574bd67daf42f48cc299f3ac83355fee59a`** is committed and
pushed to the exact fork feature branch. A second hash-checked source bundle,
SHA256 `7b8c47d7c665712164bb84359aaf8d543ff39d80fe9762763391a512cc7dd153`,
created the isolated detached Linux worktree
`artifacts/tools/replay-validation-05ff5574bd67`. The same **474 CPU checks passed
on Linux in 40.72 seconds**, explicitly loading the repaired trace module from
that snapshot with CUDA hidden and uninitialized. The main WSL runtime remains
clean at `d4e96cda975f943ac89aacbbed62e99f1c9f00d1`; no live installation,
environment restoration, binary equivalence or GPU replay is claimed. The source
bundle and detached validation/training source snapshots remain retained.

An additional bounded **CPU-only full-horizon trace fixture** ran from that
exact Linux repair snapshot: 128 inert synthetic worlds, 250 policy ticks and
2,500 recorded physics counters each. Real `encode` and hash-verified `verify`
completed with the packed CUDA-origin binding intact and every provenance,
admission and motion flag false. Serialized trace size was **267,936,199 bytes**,
within the unchanged 512 MiB trace limit. No actor inference, optimizer, actual
physics or CUDA context was run; the fixture's synthetic numerical passes are
not learned performance. This checks trace shape/serialization/full-horizon
replay only, not full bundle/control size, peak-memory qualification or GPU
timing. Its 120-second CPU watchdog was not a probe cap extension.

There is **no automatic retry** of this consumed single-use probe, no full packed
evaluation enabled and no checkpoint promotion. The next GPU step needs a
separately reviewed fresh probe identity and authority, the tested repair source,
fresh frozen-host/idle/lease guards, the same fixed case and unchanged hard bounds,
and sufficient remaining authorized time. Only a successful complete retained
measurement can justify the later measured-cap full-evaluator declaration.
The runner's current October 1 cutoff must not be reused on another day: a new
window needs an explicit tested date-bound declaration before any new launch.
The historical full evaluator still refuses WSL preparation and its registry
excludes this probe-only continuation. Hopping, obstacles and rolling football
remain behind their separate numerical gates.
