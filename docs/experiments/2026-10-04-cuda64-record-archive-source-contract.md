# CUDA64 record archive: source-only consistency and force-window audit

Status: **source implementation; no native collector, returns or training job**.
The user renewed bounded continuation after the October 3 20:00 window closed.
This slice does not reopen that expired declaration or infer a new overnight
deadline. It follows the [collector](2026-10-03-cuda64-transition-source-contract.md)
and [finite-return helper](2026-10-03-cuda64-finite-returns-source-contract.md),
without changing either frozen contract or the D1 parent weights.

## Retention and consistency boundary

The new [archive module](../../src/mjlab_microduck/stance_recovery_cuda_record_archive.py)
owns nested tensors, including tensors inside tuples. `retain_record_cpu`
validates the inherited shared lease **before** traversing a collector record
or transferring a CUDA0 tensor. A future native supervisor must explicitly
request `capture_control=True`; the collector's default is false, and the
archive rejects missing controls rather than silently accepting that default.

`encode`, `check` and `verify` are CPU-only. They require CUDA hidden and
uninitialized for consistency checking. The format admits 1–28 ordered records,
64 rows, the existing source-bound schedule and explicitly supplied compiled
body descriptor, and only declared seeds 653/659. There are no filesystem
writes, service launches, optimizer calls, checkpoints or admission APIs.

The tree is bounded to depth 32, 250,000 nodes and 128 MiB of traversed tensor
bytes, counted conservatively even for aliases. Dense supported CPU tensors
and finite primitive leaves are required. Capture owns copies; direct checking
validates every nested value without cloning it. Serialization is capped during
writing at 256 MiB, with a deterministic size refusal even if PyTorch's zip
finalizer masks the original write error. Verification checks the caller's
independently retained **whole-byte SHA256 before CPU weights-only decoding**.
These are archive limits, not a native process-memory or runtime guarantee.

The checker binds source/seed/cursor and false flags; raw action Gaussian
log-probability arithmetic; advancing, continuous private RNG bytes and their
receipt digests; reward and terminal copies; actual row clocks; typed physical
boundaries, observations, motor commands and controls; and copied before/after
reset rows. It stops at the first observed terminal and rejects a later record.
Synthetic terminal/reset fixtures establish source consistency only. A fresh
280-substep window cannot naturally reach the 2500-substep timeout, so a
reported timeout or injected clocks are refused.

## Reachability is per row, not a successful-run Boolean

At 10 substeps per 20-ms policy tick, 28 calls span only **0.56 simulated
seconds**. Each force entry is bound to an accepted motor proposal and the
actual before/after boundary clocks. Complete external and generalized-force
arrays must equal the declared schedule before integration and be zero afterward.
Repeated physical windows, missing entries and incorrect row masks reject.

| Declared onset | Onset substep | Exposure possible in a fresh 28-call run |
| --- | ---: | --- |
| 0.5 s | 250 | A 20-substep pulse can finish at step 270 |
| 1.0 s | 500 | Not reached |
| 1.5 s | 750 | Not reached |

A prefix ending at step 250 has not applied the first pulse; ending at step
260 has delivered only 10 of its 20 substeps. The report labels every row's
window `not-reached`, `partial` or `complete`, retaining expected, observed and
nonzero delivered counts. A mixed batch with any incomplete nonzero row cannot
claim complete nonzero delivery. A zero-wrench control never counts as a
completed nonzero push.

Three phase headers must be present when a window is active, but this module
does **not** replay their solved payloads. It does not recompute motor/BAM
outputs, rewards from physics, actor/critic outputs, CUDA RNG draws, solver
results or trajectories. Caller RNG snapshots, model/Adam snapshots and
prepared-object provenance still require separate independent retention and
replay. All associated qualification fields and all eight capability flags
remain false, even when every arithmetic/copy check passes.

## Next native gate remains open

The fresh October 4 00:11 Shanghai read-only check found the exact clean WSL
feature branch still at `1b96ccaac1326d6f0a1e00cdff8950b91a2916be`.
The Blackwell GPU was 0% / 695 MiB / 30 C with no compute PID or active Duck
user service. FilmBrain observatory/video-playground remained running at PIDs
521/298048 with zero restarts; both protected AI Mission services remained
inactive at system and user scopes. This is a host snapshot, not new runtime
or native collector qualification. No service, package, driver or WSL source
was changed.

Before a real collector probe, commit/push tested source, define a fresh
supervisor with exact preparation/model/caller-state inputs, actual stock
storage snapshots, mandatory control and complete solved-phase evidence,
source/host identity, one shared leased workload and bounded capture plus
independent replay budgets. Authenticate its
prerequisites again; old shadow-sampling timings and receipts do not qualify
CUDA64 physics or provide its resource budget. Choose a reachable early pulse
for a short probe; test later timing cells only under a separately declared
longer protocol.

The [closed catalog synthesis](2026-10-03-recovery-catalog-coverage.md) found no
measured deficit across the 45 existing push cells at its one evaluation seed.
Do not train a passing parent or relax gates to manufacture progress. Hopping,
obstacle negotiation/return-to-speed, and rolling-football balance retain their
separate acceptance ledgers. No physical Duck or physical-motion authorization
is available.

## Reviewed source evidence

The bounded Luna contributor's final focused suite passed **50 tests in
28.81 seconds**. The owner's final integrated 27-file regression passed
**787 tests in 104.34 seconds**, with CUDA hidden and no skips. It is the
unchanged 25-file `TEST_FILES` tuple in the
[closed sampler supervisor](../../src/mjlab_microduck/stance_recovery_cuda_shadow_probe.py),
plus the existing transition suite and the new
[archive suite](../../tests/test_stance_recovery_cuda_record_archive.py).
The sampler's old declared 679-test protocol is not widened or relabeled.
Ruff 0.15.7 check and format, Python compilation, both changed documents'
relative links, and whitespace checks passed.

The owner reviewed the source and sole test contribution together, tightened
whole-tree validation and producer-shaped controls, and reran the integrated
suite after the last failure/mask checks. The tests use explicitly synthetic
CPU arrays, eight-byte synthetic private RNG states, and a synthetic body
descriptor. Their phase payloads are placeholders: only header presence and
recorded force arrays are checked. Test lease/context seams do not enter the
production API. These are source tests, not CUDA sampling, solver execution,
physical trajectory replay, natural reset qualification or learned performance.
