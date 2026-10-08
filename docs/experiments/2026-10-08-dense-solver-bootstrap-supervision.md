# Dense solver scratch bootstrap and owned-child supervision

## Scope

Preparation for a fresh-process dense `_update_constraint` diagnostic on 100.98.
This does not execute a CUDA solver, qualify native numerics, identify the
runtime cause, authorize policy training, or demonstrate a physical Duck skill.
The historical initialization comparison remains rejected. Existing locomotion
and obstacle videos remain historical simulation evidence, not new results.

## Authenticated replay input

The original forward-four initialized bank is a replay input to the unchanged
whole `_update_constraint(..., track_changes=False)` caller. It is **not** the
input immediately before the dense target: its preceding EFC writer must still
run, and a new exact-launch before/after packet must bracket that target.

Retained raw root: `artifacts/evaluations/solver-init-tick-run-e45a59c412bf`.
Inventory: `artifacts/tools/solver-init-tick-closeout-e45a59c412bf/inventory.json`
(69,779 bytes, SHA256
`8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625`).
The authenticated `child.json` binds the recipe and snapshot directory
(2,139,030 bytes, SHA256
`d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91`).
Raw `solver-init/original/forward-04.initialized.bin`: 5,375,152 bytes, SHA256
`b47122d67e3568bf4e413c3905b0ca329e727ee900b168f93dd4c5f27bce4441`.
The owner independently authenticated all three before successfully decoding the
complete 66-field retained snapshot with the new helper.

## Scratch closure and sequencing

`stance_solver_scratch` is standard-library-only on import. It authenticates
the entire immutable bank before interpreting fields; checks literal recipe,
all 66 offsets/layouts/dtypes, active row counts, false done flags, and meaningful
finite active Jacobian entries; and rejects any active elliptic EFC row.
Pyramidal cone configuration alone does not exclude malformed elliptic rows.
The missing `contact.friction` carrier is unreachable only after this row check.
With `track_changes=False`, the frozen static branch does not access changed-row
tracking buffers. The helper restores the 24 arrays actually read or written by
the whole four-launch caller, including all five dense target buffers.

The one-shot restorer verifies destination identities, layouts and disjoint
allocations, then CPU staging identities, layouts and exact authenticated bytes.
It copies on the explicit supplied stream and synchronizes before a receipt.
Failure consumes the attempt; partial device writes are not rolled back. The
caller must authenticate the actual Warp functions, objects, stream and module,
exclude concurrent mutations, and supply the runtime guard. No general-purpose
hostile-code sandbox is claimed.

Allocate the frozen model/data/context using direct constructors, not the
higher-level stance runtime or a forward/init/solver call. The direct context
constructor allocates buffers. Compile the target's unchanged original module,
authenticate offline SASS against the retained CUBIN, and explicitly load it
while its executable cache is fresh. Only then restore and invoke the guarded
whole caller. Any constructor that launches or preloads the target must be
rejected by the final fresh-child admission checks.

## Process boundary

`stance_solver_supervisor` launches one reviewed literal argv without a shell,
in a fresh Linux session, with caller-supplied environment, bounded log file and
explicit inherited lease FDs. It binds `/proc` PID/start-time/session identities,
holds a root pidfd and observes exit with `waitid(P_PIDFD, WNOWAIT)`. The exited
root stays unreaped through the final session scan and cannot fork again or be
replaced by another SID owner. Missing root identity is a failure, not permission
to adopt a new session. It signals only freshly verified same-session pidfds.
A successful root that leaves live descendants is a failure, not a receipt.
Process state changes alone do not invalidate a stable birth identity; coarse
start-time stamps alone are not treated as unique process-generation handles.

Probe calls after spawn run outside the cleanup loop; a stalled probe cannot
block the child deadline. Probe error, timeout, child nonzero exit, leftover
descendants or supervision error trigger bounded TERM/KILL cleanup. Initial
admission is caller-bounded before spawn. `/proc` inventory and session sizes
are capped. Missing Linux pidfd support is refused before spawn.

The enclosing capped user service with `KillMode=control-group` remains mandatory
for attach failures, unreadable process inventories or detached descendants.
The helper alone does not provide that external backstop, acquire a GPU lease,
authenticate source/toolchain/cache, verify services, or grant GPU admission.
The reviewed child must join all subprocesses/tool grandchildren and must not
daemonize. A `/proc` scan is non-atomic: an arbitrary forking descendant tree may
evade it. The receipt therefore reports observed members, keeps
`process_tree_retirement_proven=false`, and requires independent retirement of
the enclosing capped service cgroup before native acceptance or lease release.

## Review and verification

An initial real-bank decode caught an incorrect synthetic matrix-dtype oracle:
`context.frame` is `warp._src.types.mat33f`, not the MuJoCo-Warp spelling.
It was corrected with an independent literal regression; the real authenticated
bank now decodes. The installed solver file was not changed.

The focused suite has 40 cases. It includes inert import, authenticated packet
and staging mutation refusal, one-shot failures, explicit stream copies,
synthetic proc birth/reuse protection, probe refusal/deadline, and an isolated
real Warp **CPU-only** allocation/copy test over 24 arrays with kernel launches
forbidden and no installed solver import. macOS checks pidfd refusal; WSL must
additionally execute real CPU child/grandchild cleanup and nonzero-exit cases.
Review also caught a pre-exit inventory race and stale timeout timestamps.
Unreaped-root final scans, post-inventory clock checks and actual probe-completion
timestamps now guard success. Attach failure uses the held root handle and
explicitly requires the cgroup backstop for uninspected descendants.
Integration and native closeout evidence will be recorded after verification.

## Next gate

Assemble and predeclare the full bounded native owner/child runner: exact source
and installed tree/library/tool pins, existing lease, protected services,
Windows plus WSL occupancy, finite time and artifact caps, fresh private cache,
deadline backstop, and independent retained-packet receiver. No CUDA execution
or training is implied by the helper/test results. Do not reuse an expired
historical diagnostic cutoff, alter packages/drivers, or disturb FilmBrain.
