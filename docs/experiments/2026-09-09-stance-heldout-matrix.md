# B1-N held-out matrix and caller-owned collection

This CPU-tested component follows the [PPO precursor](2026-09-09-stance-ppo-adapter.md).
It implements the declared held-out matrix and first-attempt collection without
creating a simulator, allocating a GPU, installing a service or training a model.
No full matrix or new Duck performance result was produced by this coding chunk.

## Immutable matrix and numerical decision

The plan requires ordered checkpoints128,256,384,511 from one pilot identity:
same training source, runtime, launch hash, initializer, seed521 and512 worlds.
CPU preflight must restore all four actual checkpoint byte streams and validate
the exact runtime/compiled-plant bytes before any case allocation. Metadata-only
planning is not preflight, a launch authorization or evidence of training history.

Each checkpoint has seeds541,547,557 and128 worlds:12 cases and1536 complete
first attempts, with384 attempts at final511. Case names, byte bindings and
bundle launch hashes are derived deterministically. Actor inference is fixed
to CPU for exact alignment with bundle replay; physics is declared CUDA for
the actual matrix. Device or matrix changes require a new reviewed plan.

At least122/128 passes are required for **each** final seed (ceil(0.95*128)).
Earlier checkpoints remain diagnostic, even if all their attempts pass. Missing
cases, duplicate/missing world IDs, short prefixes, changed seeds, contradictory
counts or absent replay checks refuse matrix publication. All original per-attempt
physical gates remain in the existing scorer. A complete failure is retained as
a failure; it is not silently retried or replaced with a new episode.

Even a positive matrix decision says `nominal-candidate-numerical-only`, with
provenance, checkpoint admission, learned stance and football acceptance false.
The nominal fixture has no randomized perturbation distribution: three seeded
repetitions test repeatability, not three independently varied environments or
generalization. Live seed initialization and GPU supervision must still be
performed and independently retained by the future guarded launcher.

## Collection and durable verification

The worker accepts an already owned environment and a strictly restored frozen
CPU actor. It collects up to250 policy ticks, including every physics boundary
and opt-in control evidence. It never resets a world. Finished worlds freeze
while siblings finish, and collection stops when all first attempts are complete.
The environment's compiled model and exact nominal reset are checked before
the convenience worker takes a step. Existing output directories are refused
before simulation work, not overwritten or treated as a resumable case.

A monotonic budget is checked before each policy tick. If it expires after a
tick, the partial evidence can be retained but cannot satisfy matrix completion.
This check cannot interrupt a hung physics kernel; an independently timed
outer service remains mandatory. Zero-tick collection refuses publication.
Exceptions are not converted into successful case receipts or auto-retries.

The convenience worker publishes the existing v3 bundle, immediately verifies
its byte hashes and replays checkpoint, physical, plant and control checks.
The matrix writer takes independently retained bundle hashes, verifies every
case sequentially, and publishes `matrix.json` last with fsync and exclusive
creation. The reader requires an independently expected matrix hash and plan,
rechecks exact directory inventory, verifies every bundle again and recomputes
the decision. Rehashing an invented positive summary cannot make it pass replay.
Retain the exact plan bytes independently before any future live execution.

## Validation and next boundary

Tests use untrained weights with explicitly synthetic checkpoint metadata,
synthetic matrix score fixtures and short real CPU collection. Aggregator tests
stub bundle verification solely to test dispatch/thresholds; those empty test
directories are not physical evidence. Separate worker/bundle tests exercise
actual short CPU traces, first failure, wall-budget partial capture and refusal
of bad resets or existing output. No test reports a trained policy passing H1/B1.

Local validation passed all 358 CPU regression tests in 27.10s, including 28 new
matrix/worker cases. Markdown HTML and relative-link checks passed.

On 100.100, implementation commit
`4afc79c6ac9327a87013f5ed7c46adc17558f0af` passed all 424 CPU regression tests
in 36.70s with `CUDA_VISIBLE_DEVICES` empty, including the 66 shared campaign
supervisor tests. The exact feature branch was clean after testing. GPU readback
was 0% utilization, 12 MiB and 44 degrees C, with no compute PID. Both protected
system services (`recomo-ai-mission-vllm.service` and
`recomo-ai-mission-subject-model-worker.service`) remained inactive.

No GPU allocation, optimizer or full matrix run occurred; the only new physical
traces were short CPU test fixtures. These checks validate infrastructure, not
learned standing, hopping or football balance.

Next assemble the source/runtime/asset/seed-bound GPU adapter and independently
timed single-workload launcher, then the declared disposable64-world/16-iteration
smoke. Measure timing and inspect finite metrics before scheduling a fresh pilot.
Real simulator resume still requires physical/motor/FIFO/solver-state retention;
the learner-only codec does not provide it. Preserve protected services,100.98
and historical checkpoints; do not replay old closed diagnostics or admit ball
balance from infrastructure tests.
