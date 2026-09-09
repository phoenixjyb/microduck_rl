# B1-N forward repeatability: diagnosis and next isolation test

The [six-case throughput probe](2026-09-09-stance-throughput-probe.md) remains
`differential-rejected`. Its faster graph candidate is not enabled in training.
This chunk adds an offline, tested diagnosis of immutable evidence and specifies
the next bounded experiment. It does not launch another GPU job or train weights.

## Reproducible CPU diagnosis

`python -m mjlab_microduck.stance_prefix_diagnosis` requires an independently
supplied source-report SHA256. It verifies the exact 34-file inventory, every
file hash, the original deterministic decision, every prefix descriptor and
captured-value hash, then deserializes with weights-only loading onto CPU.
Frames are checked against the existing physical-state schema; nonfinite values
anywhere, unsupported types and unverified files fail closed. No simulator is
run. Comparison preserves ordered contact records, signed-zero bit differences,
numerical differences, discrete values and structural mismatches separately.
No tolerance, sorting of contact rows or rounding can turn a failure into a pass.

The derived output must be fresh and outside the immutable input directory.
It never changes the original decision and always reports diagnostic-only,
no graph equivalence, no training admission and no physical-motion authority.
All per-field maxima retain their concrete tensor index and ordered path;
chronological boundary divergence is separate from dictionary field order.
This is a comparison report, not a simulator-resume state or causal proof.

Run from the repository root with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python \
  -m mjlab_microduck.stance_prefix_diagnosis \
  --input artifacts/evaluations/stance-throughput-f2ff27e434f4 \
  --report-sha256 b9c1ab3b2148a851389e87dc91c9a0dfda00734e11390d0e25ef955b99832373 \
  --output artifacts/evaluations/stance-throughput-f2ff27e434f4-prefix-diagnosis-v1.json
```

The actual Mac result is 1,076,349 bytes, SHA256
`898403098bb6435419a3dbcc871553dd472493b905fdea01c29ca0dc9d8ecb0b`.
Its original report and all original raw files remain unchanged. Source-bound
Linux regeneration produced the identical bytes and hash, as recorded below.

Observed in all four pairs (64 eager/eager, 64 eager/graph, 512 eager/eager,
512 eager/graph):

- The complete initial retained boundary is bit-identical, not just qpos/qvel.
- The first differing retained boundary is tick 0, boundary 1, immediately after
  the first integrated step and its post-step solve. Differing fields are qpos,
  qvel, root/joint velocity and actor/critic observations. The pre-Euler solved
  acceleration is **not retained**, so this does not attribute the discrepancy
  to Euler itself rather than its input solve.
- All compared integer/boolean/string/None values and structures match across
  the four complete retained ticks, including terminal contact metadata. Floating
  forces and states still differ. No metadata permutation was observed here.
- Four ticks cover at most 40 physics steps (80 ms); the support-loss stop becomes
  active at step 50. The injected world-0 tilt tests a synthetic terminal, not
  a natural fall or a successful standing policy. Later trajectory behavior,
  support-loss behavior and full five-second acceptance remain untested here.

## Source inspection: hypotheses, not a proven root cause

The installed, pinned MuJoCo Warp source contains these relevant operations:

- `constraint.py`, `_friction_dof`: active friction rows receive indices through
  `wp.atomic_add(nefc_out, worldid, 1)` (line 1394 in this exact source).
- `solver.py`, `update_gradient_grad`: the gradient squared norm is accumulated
  with `wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)` (line 2256).
- The dense constraint-force accumulation at lines 1999-2008 is a sequential
  row loop. Do not cite the sparse atomic force path as this plant's dense path.

These are candidate assembly/reduction sites to examine. Their existence alone
does not prove that a race, contact order, reduction order or graph implementation
caused this run's divergence. The retained prefix lacks the necessary pre-Euler
solver/constraint state. Do not patch installed kernels or alter solver options
on this evidence alone.

| Installed source | SHA256 |
| --- | --- |
| mujoco_warp/_src/constraint.py | `b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53` |
| mujoco_warp/_src/solver.py | `bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a` |
| mujoco_warp/_src/forward.py | `c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3` |

## Predeclared next experiment: same-input forward only

Protocol: `football-b1n-same-input-forward-v1`. Implementation and its CPU tests
are still required; this document is **not a launch-ready source-bound plan**.
Before any launch, bind the reviewed implementation commit, runtime hashes,
compiled plant, source-snapshot hashes, output path and service to the plan.

Use one owned runtime at a time, first 64 worlds then 512, seed 523, unchanged
nominal plant and zero controls. There are no policy ticks, BAM updates, Euler
steps, optimizer steps, trajectory resets between trials, or checkpoint inputs.
After normal runtime initialization, retain a complete model/data array-content
snapshot and bind every array pointer, layout and scalar/static option. This
includes warmstart, model damping/friction and scratch contents, not just qpos.
Before **each** measured forward call, restore those same contents in place,
synchronize both streams and require an exact input-content hash. Graph capture
must preserve the bound allocations; restore again after capture before any
measurement. Do not assume capture is a no-op for dynamic contents. Unknown
fields or an unbounded snapshot fail before trials; do not omit them to fit.

At each world count run eight calls in the fixed order E, G, G, E, G, E, E, G
(four eager and four graph), with identical restored inputs. Eager calls the
unchanged stock forward; graph calls the same complete captured forward. Keep
the existing fresh finite, capacity and active-contact checks after each call.
Never substitute a separately split phase graph for the candidate under test.

Retain each input hash and post-call raw output before proceeding:

- qpos/qvel/time and qacc_warmstart, to check no integration/input drift;
- qacc, qacc_smooth, qfrc_bias, qfrc_actuator, qfrc_constraint and solver_niter;
- active contact records and decoded forces, preserving original ordering;
- nefc and each world's active constraint type/id/J/D/aref/force/state rows,
  so a row-order difference is distinguishable from changed solved values.

Bind the actual fields against the pinned runtime during implementation. Retain
raw arrays and their hashes, not only maxima. Keep inactive scratch bytes in the
input snapshot; do not mislabel inactive scratch as an active physical quantity.
The snapshot covers owned model/data, not the entire CUDA process or ephemeral
solver allocations. Both arms use the same objects to avoid confounding fresh
allocation differences with graph dispatch.

Compare all six within-eager pairs, all six within-graph pairs and all sixteen
cross-arm pairs at each batch size, with no numerical tolerances. If any input
hash differs, classify `invalid-input-control` and stop. If same-input eager
outputs vary, classify `forward-baseline-nonrepeatable`: integration and learning
are then unnecessary for variation to occur, but the precise kernel cause still
needs evidence. If eager repeats match and graph differs, classify
`candidate-difference-requires-diagnosis`. If all outputs match, classify only
`same-input-forward-repeatable-in-this-sample`. No classification admits graph
training, restores the rejected trajectory result or validates motor safety.

Use one retained sequential user service, the existing shared GPU lease,
120-second child watchdog, 180-second service cap, 80 C stop threshold and
600-second closeout reserve before September 10 07:30 Shanghai. Require idle
GPU0 and both protected system services inactive before and throughout the job.
Cap each batch's input snapshot at 256 MiB and the retained experiment at 1 GiB;
abort cleanly rather than trimming evidence. Preserve unrelated workloads.
No automatic retry, raw perception, MP4, installed-library patch or physical
motion. Diagnose any failure read-only first.

## Path back to training

The immediate aim is to separate same-input forward variation from accumulated
trajectory divergence, not to make a strict test pass by changing its tolerance.
An eventual numerical-equivalence protocol must have independently justified
engineering-unit error budgets, replicated held-out inputs, unchanged discrete
stops and complete trajectory coverage. The 0.4025 N observed support difference
is not itself an acceptable tolerance. No such tolerance-based gate is declared
or passed by this document. After accepted optimization evidence, run a separate
optimized training smoke with full optimizer and evidence costs before deciding
whether the original 512-world, 512-update pilot fits its 30-minute cap.

## Validation and delivery

The initial 18 diagnostic CPU tests passed in 17.22s. After adding explicit
missing-boundary coverage, the combined 87 diagnostic, throughput, attempt-trace
and contact-evidence tests passed in 31.04s. These include bit-versus-numerical
comparison, nonfinite rejection, contact ordering, chronological divergence,
hash-before-deserialization, input immutability, deterministic regeneration and
no-admission behavior. Synthetic bundles are test fixtures, not physical evidence.
Markdown HTML rendering, table structure, local links and `git diff --check`
passed.

At pushed source `8feabd98e46047d0092e610349fb1b806758ed27`, all 479 Linux stance,
GPU-idle and foundation-campaign tests passed in 48.48s without skips or
deselection. The 19 new tests are CPU diagnostic checks, not CUDA evidence.
Independent regeneration on 100.100, with CUDA hidden, produced the identical
1,076,349-byte diagnosis and SHA256 shown above. The analyzer's own source SHA256
is `8533a247c111419fdfa17572024bba62eb3a0691f5ff41c7dc88166c273b3c8b`.
All three inspected installed source hashes also matched on 100.100.
The report is retained outside the immutable probe directory on both machines.

Delivery state: CPU diagnosis complete and cross-machine reproducible;
same-input GPU isolation specified but not implemented or launched. The numerical
root cause, graph equivalence, optimized training smoke, full pilot timing and
learned stance capability remain open. No weights or protected services were
changed by this chunk.
