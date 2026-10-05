# October 6 fixed-input CRB reduction diagnostic

## Question and authority

Continue the newly authorized source/simulation window through **2026-10-06
08:00 Asia/Shanghai (00:00 UTC)**, using only 100.98 via `gw98-direct`.
The clean local base is `2df4948e5af2c0e10636d61c8d70f0a685fd4d82` on
`feat/athletics-obstacle-curriculum`; the retained native execution checkout is
`f190d24dbb4b185f581462c36c0b2104a743cdfd`. Freeze a new tested source before
native execution. All previous source, protocols, failed namespaces and raw
artifacts remain unchanged. This is not an extension of the October 5 deadline.

The original CUDA64 full pair remains rejected by exact semantic comparison.
The earliest observed difference is composite inertia at event 0 / step 0,
world 53, before any applied pulse. Identical completed-forward root `cinert`
and child `crb` values admit different scalar float32 addition orders. They are
**not captured kernel-entry operands**. The new question is narrower: can the
unchanged installed accumulation kernel produce different bits on repetitions
of an identical, explicitly derived reduction fixture without a launch observer?

No optimizer, policy inference, learned capability, full-window acceptance,
raw perception, physical motion, package/driver change, or protected-service
restoration is authorized by this experiment. Training stays gated.

## Immutable inputs and derived fixture

Authenticate the original 23-file failed-pair inventory, the two compiled-map
files and both order-oracle JSON files before interpreting any retained value.
No pickle or original GPU tensor deserialization is needed.

- Original source: `65abe930caa16476e4d8ece44f937492da47ce2e`.
- Oracle comparison SHA256:
  `7731633338071aeb1435c0dd613f1216cc1b528b15edc8e658cbcad8691fa45c`.
- Oracle report SHA256:
  `38d860e97cdbe898c560b8522c3191348984c778f8d772d899f6f02ef7a5517a`.
- Installed smooth source SHA256:
  `63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f`.
- Derived full baseline SHA256:
  `554aa8cccf4dbede45ac8f03b3d6e9e8433b0f567f30da454ab2cacd754a80d6`.
- Full seven-key launch-topology JSON plus newline SHA256:
  `2e9eb90bd53525e4c4af45a880481fe6cab874be549d9260b5a92d8a397ebe19`.
  This is distinct from the oracle's compiled component-map topology hash,
  which covers plant, body and degree-of-freedom mappings rather than launch IDs.

Compile the exact plant topology on CPU without a physics step. Bind all body
parents and seven reversed levels, including level index 4 `[2,7,11]` and their
common parent body 1 (`trunk_base`). Populate a contiguous float32 `[64,16,10]`
fixture with root `cinert` at body 1 and the authenticated child composite
inertias at bodies 2, 7 and 11. Only the other, unused bodies are synthetic zeros.
Verify every used JSON value against its independently retained uint32 bits;
preserve signed zeros. Label this **derived-complete-forward-reduction-fixture**,
with `actual_launch_inputs_captured=false`.

## Frozen execution design

Use pinned Warp 1.12.0 / MuJoCo Warp 3.8.1 and the installed, unmodified
`smooth._crb_accumulate` kernel. No new deterministic option or kernel rewrite.
Run exactly **32 repetitions**, 64 worlds and three nodes, dimension `(64,3)`,
with level `[2,7,11]` and independently compiled parent IDs.

Preserve production aliasing: pass the **same working CRB array** as input and
output. Before each repetition, reset the complete working array from the
immutable fixture on the same Warp stream. Copy the complete result to separate
owned GPU storage after each launch. Do not insert the per-level CRB observer,
host readback or host synchronization between repetitions. Synchronize/read back
only after the bounded batch. Copies are part of this explicit diagnostic
schedule; it is not a timing-equivalent substitute for the original rollout.

Retain all 32 complete `[64,16,10]` outputs as little-endian float32 bytes:
**1,310,720 bytes**, with an independently retained **40,960-byte** baseline.
No compression, omission, tolerance or signed-zero canonicalization. No actor,
model, Adam or rollout storage is constructed; those checks are not applicable,
not silently claimed satisfied. Preserve and retain caller CPU RNG bytes and
require that Torch CUDA RNG remains uninitialized; the GPU child uses Warp only.

The initialized-GPU producer publishes exclusively created, bounded durable
raw artifacts and hashes. It never calls the CUDA-hidden checked CRB archive
encoder/decoder. A separate CUDA-hidden supervisor authenticates whole bytes,
reconstructs the independently derived fixture and checks all output bodies.
Bodies other than parent 1 must remain bit-identical to the baseline.

## Resource and ownership gates

Fresh source-tagged user services use `RemainAfterExit=yes`, CPU quota 200%,
Nice 10, KillMode `control-group`, and memory maximum **6 GiB**. CPU tests cap
at **300 s**; the supervised run caps at **600 s**, its isolated GPU child at
**240 s**, with a separate **300 s closeout reserve** plus **60 s margin**.
Tests reserve the full tests/run/closeout/margin window. Refuse insufficient
time rather than extending the deadline. The final exact test count is frozen
by the owner after reviewing both new test files.

Use the existing FilmBrain cooperative lease; preserve the regular lock file
and inherit the held descriptor to the owned child. Require exact native root,
clean feature source, frozen environment, machine/GPU/driver identity, one
running Duck service and the actual supervisor PID/child parent relationship.
Require idle GPU before/after and only the owned compute PID during execution,
temperature below 75 C, at most 5 GiB used VRAM and at least 6 GiB free.

Keep both ReCoMo AI mission services inactive in both managers. FilmBrain
observatory/video state and all unrelated workloads must remain unchanged.
No service other than a newly owned diagnostic unit may be changed.

Logs cap at 1 MiB; JSON at 2 MiB; individual binary artifacts at 2 MiB and the
complete declared new inventory at 8 MiB. Retain partial files on failure.
Authenticate the exact successful CPU test receipt before native execution.
Retain exact successful terminal invocation, PID 0, zero restarts and caps
before cleaning up only an already-exited owned unit. Rehash immutable inputs,
source/runtime identity, and all new artifacts at independent closeout.

## Decision boundary

Different bits across identical fixture repeats demonstrate variability of the
isolated reduction under this declared schedule. They do **not** identify actual
atomic order or prove it caused the original trajectory rejection. Constant
results across 32 repetitions mean only *variation not observed*, not proof of
determinism. Unexpected scalar results, changed inputs or non-root mutations
require retained read-only diagnosis; do not discard contrary evidence.

Any next observer-on/off experiment must retain repeated controls, not infer
observer causality from a single differing stochastic sample. The original
64-world / 28-call / 280-step, step-250 pulse protocol remains a separate full
window numerical gate. A derived fixture never replaces it.

NVIDIA's [Warp 1.15 release notes](https://github.com/NVIDIA/warp/releases/tag/v1.15.0)
describe opt-in deterministic atomic execution absent from our pinned 1.12
environment. That is a future separately isolated compatibility option, not
permission to upgrade this environment or treat new arithmetic as original
replay acceptance.

## Results

The owner-reviewed 46-file CUDA-hidden contract suite passed **1,415 tests,
zero skips**, in 205.90 seconds locally. Focused coverage includes exact fixture
bits, signed zero, whole output bytes, pairwise repeat extrema, source/runtime
guards, distinct hash domains, service caps and terminal invocation, sanitized
environments, and the mocked 32-launch aliased command schedule. The routing
skill was used to separate bounded implementation and independent read-only
review; the owner integrated and checked the changes. These remain source/CPU
test results, not native GPU or learned-policy evidence.

After freezing the count, the same full suite passed again in 215.08 seconds.
The final narrow environment check also pins the existing sanitized startup
profile: `ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`, and one thread
each for OMP, MKL and OpenBLAS; it does not inherit credentials or injection paths.

No new native job or numerical gate acceptance is claimed by this declaration.
