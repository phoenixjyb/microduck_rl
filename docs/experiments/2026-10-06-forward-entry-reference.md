# October 6 source-only forward-entry reference

## Predeclaration

Start at `a340f37c6cfb7bf7c45e1ea79b896ceef66ed936` on
`feat/athletics-obstacle-curriculum`. Change only this document,
`src/mjlab_microduck/stance_forward_entry_reference.py` and its focused test.
Preserve all previous source, exact numerical gates and immutable runtime
evidence. This slice permits pure CPU arithmetic, not native capture or a GPU
job. Native 100.98 remains at `2ecee471f7b999822ed19defd7e2d1c7ad09df07`.
Start no new work at/after 2026-10-06 08:00 Asia/Shanghai.

## Why completed-forward fields are insufficient

The pinned `smooth.py` whole SHA is
`63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f`.
The local installed file was read and its whole hash checked. This does not
freshly authenticate a native installation or demonstrate kernel execution.

`com_pos` initializes `subtree_com = xipos * body_mass`, accumulates it in-place
through reversed levels, then divides by `body_subtreemass` before producing
`cinert` and `cdof`. The existing post-forward `subtree_com` is therefore not
the weighted pre-accumulation input. Multiplying a rounded, completed CoM by
subtree mass is not an exact reconstruction of the original per-body input.

`rne` initializes world/branch accelerations, computes per-body `cfrc_int`,
accumulates it in-place, then produces `qfrc_bias`. The acceleration sensor's
postconstraint RNE later overwrites `cacc`, `cfrc_int` and `cfrc_ext`; the
existing `EarlyInertiaTrace` explicitly excludes those scratch fields. Its
`cvel` and `cdof_dot` are recorded, but they do not supply a stage-boundary
`cfrc_int` snapshot. Do not label a sensor-overwritten force as bias-RNE input.

The authenticated partial serial constructor pair has identical recorded
`cinert`, `cdof`, `cvel` and `cdof_dot`, while one `qfrc_bias` scalar differs.
That narrows the missing observation, not its cause. `fwd_velocity` invokes
`tendon_bias` after `rne`; do not assume the completed bias is a direct
observation of `_qfrc_bias`. A future stage control must retain the actual model
tendon/armature configuration and distinguish any later contribution. The
frozen seven-field topology does not authenticate those model fields.

Other locally read installed source hashes: `forward.py`
`c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3`,
`sensor.py`
`2ba411f182ba5fb7d88e85050c55ef4055fafefe2d26555f429ecd4e0d80c19e`,
and `io.py`
`731a31f254274c6e04d9843b11bdcb0502cf9d49abf26aa1c82e52e1a13fa75a`.
These are local source reads, not new native execution evidence.

## Bounded CPU reference

Accept only plain, exactly sized bytes containing finite little-endian float32
values for the pinned 64-world/16-body topology. Bind the complete input bytes
to the supplied lowercase SHA-256 before decoding. Accept exactly two named
references: `subtree_com` (3 components) or `rne_backward` (6 components).
Authenticate the existing exact topology and derive the separate body-zero
activity rule through the source-only forward-reduction planner.

Apply the nine single-writer groups in their declared order. Every nonzero body
adds its evolving vector to its parent using float32 addition; input/output
aliasing is preserved inside the private working copy. Body 1 actively targets
root 0; only body 0 is inactive. Return immutable expected bytes and detached
metadata. Bound each input/output snapshot to at most 24,576 bytes. Reject
nonfinite inputs and overflow/nonfinite intermediate predictions.

Do not reuse the CRB checker's all-bodies-except-1 invariant for these operators.
A sibling-sum change at body 1 can propagate to body 0 because `[1] -> [0]` is
active. A future isolated repeat must compare the complete output and report
both bodies separately, retaining any mismatch elsewhere rather than silently
masking it. This is structural propagation, not an observed device order.

This is a candidate recurrence for supplied bytes, **not an authenticated
kernel-entry fixture**. A caller can supply normalized CoM or sensor forces
with a correct byte hash; the helper cannot infer their stage or provenance.
It must retain explicit false flags for entry capture, source authentication,
kernel/order observation, original cause, full-window, training and physical
acceptance. Do not synthesize an admission decision or add a native runtime
hook. Synthetic integer sums and signed-zero cases establish CPU/reference
behavior only.

## Future native boundary, not authorized by this slice

A separate control needs a predeclared actual entry snapshot: immediately after
`_subtree_com_init` and before the first `_subtree_com_acc`, or after bias-RNE
`_rne_cfrc` and before its first `_cfrc_backward`. Bind the exact function/call
site (distinguishing bias and sensor RNE), source, compiled topology, model
inputs, same-device aliased input/output array, stream, dimensions and complete
scratch reset. Authenticate every retained byte independently. Declare whether
readback, synchronization and dispatch interception perturb timing. A separate
isolated fixed-input repeat may distinguish arithmetic variation, but cannot
by itself prove the uninstrumented original rollout cause. Preserve the original
full-window rejection and all GPU lease/resource/service gates.

## Validation

All 44 new synthetic reference tests passed. They cover independently computed
ancestor sums including root zero, per-addition float32 rounding, signed zero,
immutable output/detached metadata, shape/type/hash rejection before decode,
nonfinite inputs and intermediate overflow, and non-admitting stage flags.
Reference plus planner/full-tree checks passed: `137 passed in 0.49s`.
The reviewed 53-file CPU suite plus the two new files passed:
`1673 passed in 212.45s`, zero skips. Ruff, formatting, diff checks and owner
review passed. These do not replace the retained native 1,581-test receipt or
the failed CPU closeout. No new native execution is part of this declaration.
