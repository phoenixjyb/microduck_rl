# October 6 source-only entry-repeat analysis

## Predeclaration

Start at `2137d39a98ca0a38fe57c2c6b4b745f73853094f` on the existing feature
branch. Permit only this document,
`src/mjlab_microduck/stance_forward_entry_repeat_analysis.py` and its test.
Preserve all prior helpers, source pins, native artifacts and numerical gates.
Native 100.98 stays at `2ecee471f7b999822ed19defd7e2d1c7ad09df07`.
No GPU job, hook, training, installed-package change or protected-service
change is part of this slice. Start no new work at/after 08:00 Shanghai.

## Whole-output comparison

The [entry reference](2026-10-06-forward-entry-reference.md) accepts bounded
caller-supplied bytes, not authenticated kernel-entry provenance. Reuse it to
derive the nine-group float32 candidate. Bind both the input snapshot and the
complete output bank to supplied whole-byte hashes before decoding either.
Accept exactly 32 repetitions of 64 worlds / 16 bodies / 3 or 6 components.
The maximum output bank is 786,432 bytes; accept no variable repetition count,
alternate topology, schedule callback or per-body mask.

Compare all scalar raw bits in every repetition against the CPU candidate.
Also compare all repetitions at each fixed output cell, distinguish signed-zero
bit variation from numeric delta, and count complete distinct snapshots.
Report body 0, body 1 and all other bodies separately, but omit none from the
complete mismatch decision. Keep a bounded first-eight mismatch sample with
repeat/world/body/component and expected/observed uint32 values. These are
scalar mismatch counts, not a Hamming distance between bit strings.

Only a fully bit-identical, single-variant bank is `exact-reference-and-stable`.
A constant wrong bank, varying bank or mismatch outside bodies 0/1 remains
`numerically-negative`; never throw away a finite numerical negative or select
only favorable repetitions. This is a calculation label, not acceptance of a
device reduction order, determinism, original causal hypothesis or training.
All entry/source/launch/order/cause/full-window/training/physical flags stay
false even when the candidate matches. A supplied hash binds bytes, not their
stage, capture procedure or GPU provenance.

## Next native experiment remains separate

Prefer a CoM-only actual-entry control first, because its reduction precedes
the other recorded inertia fields. Authenticate actual mass-weighted input
after `_subtree_com_init` and before any accumulation, not normalized output.
Compare both fresh input byte vectors before interpreting paired outputs.
Reset the complete aliased working array from the identical input for every
repeat, retain original kernel and reversed levels, then compare concurrent
levels with the nine single-writer groups. Preserve complete outputs including
root 0; disclose synchronization/readback timing perturbation.

The separate native supervisor still needs an exact clean source revision,
fresh installed-source/package/model-array/dispatch/stream checks, cooperative
GPU lease, unique retained directories, actual child PID/parent/invocation
observations, service/process/log/artifact caps, deadline margin and independent
CPU receiver. It must not construct actors, critics, learners, rollout storage
or graphs, run integration or assume the old post-forward scratch state was
reset. A stage sample from a newly reconstructed model is not retrospectively
an original-rollout entry capture. Existing partial results and the rejected
full-window gate remain unchanged.

## Validation

All 50 new CPU tests passed, including independent integer ancestor sums,
constant wrong results in each body group, separate root-zero/root-one
variation, signed-zero differences with zero numeric delta, complete all-negative
accounting with eight bounded samples, detached inputs/results, and type/shape/
hash/topology/nonfinite rejection before inappropriate decode. The three new
source-only test files passed together: `134 passed in 0.33s`.
The reviewed 53-file CPU regression plus these three files passed:
`1723 passed in 157.99s`, zero skips. Ruff, formatting, diff checks and owner
review passed. A fresh isolated import of the analysis and its dependencies did
not import Torch, Warp, MuJoCo, MuJoCo Warp, mjlab or rsl-rl.
Synthetic all-negative reports for both component widths serialized with
`allow_nan=False` under 4 KiB; all 98,304 / 196,608 scalar mismatches were
retained. No native run is authorized or claimed by this document, and the
retained native test/failed-closeout evidence is unchanged.
