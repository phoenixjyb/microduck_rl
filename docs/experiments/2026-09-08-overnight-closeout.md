# September8 overnight campaign: closeout preparation

**Status: pre-deadline snapshot, not final07:00 confirmation.** The authorized
window ends2026-09-08 07:00 Shanghai /2026-09-07T23:00:00Z. Final live checks
and deletion of the scheduled continuation remain pending at that boundary.
No further GPU work is permitted: all3/3 paired pilots are closed and rejected.

## Preserved training evidence

The [machine-readable inventory](2026-09-08-precloseout-inventory.json) records
the UTC time of a fresh read-only check on both hosts. Exact pinned manifest
and decision hashes, every payload's byte count/SHA256 and exact file inventories
match. Symlinks, missing files, altered bytes and unexpected files are refused.

| Closed pilot | Payload files | Payload bytes | Retained decision |
| --- | ---: | ---: | --- |
| F1-M | 83 | 163,963,281 | numerical-gate-stop |
| F1-N | 101 | 179,111,100 | numerical-gate-stop |
| F1-Y | 105 | 179,459,039 | numerical-gate-stop |

Total:289 payloads,522,533,420bytes, plus the three manifest files. They include
90 `.pt` files: initial snapshots and smoke/pilot checkpoints, **not90 distinct
accepted policies**. Mac roots are under `artifacts/diagnostics/`; remote roots
are under `artifacts/experiments/`, using the campaign names in the inventory.
This is point-in-time artifact verification, not a concurrent filesystem lock
or a new rollout evaluation. Original fixed-final selection and rejected gates
remain unchanged; no later candidate seeds or extra arms were collected.

Selected original gait, near/far obstacle-supervisor and rejected H1-T weights
are also mirrored and audited separately, as documented in the
[retention contract](2026-09-08-skill-retention-contract.md). The H1 and recovery
summary decisions reproduce on CPU. Source/spec identity, saved weights and
summary reconstruction do not prove a composed controller retained its skills.

## What the CPU remainder produced

- Byte-bound compatibility checks; partial real artifact/config binding;
  robot-only reconstruction and isolated action-pipeline probes.
- A retention-plan coverage checker that leaves72 missing references explicit
  and grants no behavior/transition authority.
- Selected H1-T and recovery379 inventories, with original failures preserved.
- An [evidence-backed readiness plan](2026-09-08-curriculum-readiness.md):
  foundation repair, stop/restart, structured obstacle progression, separate
  hop repair and finally matched-mechanics integration.
- A [non-wired freshness metadata contract](2026-09-08-external-freshness-contract.md)
  with synthetic boundary and lifecycle checks. No operational age limits,
  payload/clock binding, receiver integration or motion commands were selected.

No physical motion, raw perception training, harder obstacle job, H2 or MP4 was
performed in this remainder. No policy is newly admitted. The current bottleneck
remains foundation speed/motor performance and missing retention acceptance,
not a shortage of additional overnight seeds.

## Final boundary checklist

At or after07:00, start no new Duck task. Complete only the shutdown evidence:

1. Check the current clock, exact feature-branch HEAD and clean state on the Mac
   and100.100; record the actual final commit rather than this snapshot's head.
2. Confirm no running Duck user service or Duck compute PID; check GPU state.
   A retained `active/exited` unit with MainPID0 is not a running training job.
3. Confirm saved checkpoint/inventory identities remain intact. If an unexpected
   Duck job is still active, diagnose read-only and preserve its latest durable
   checkpoint before stopping only that exact authorized Duck service.
4. Keep SYSTEM services `recomo-ai-mission-vllm.service` and
   `recomo-ai-mission-subject-model-worker.service` inactive. Do not restore them
   or alter unrelated workloads/100.98/FilmBrain without an explicit request.
5. Delete `microduck-curriculum-through-sep-8-07-00-shanghai` using the product's
   automation tool, then report the final verified state and remaining gates.

Do not describe this preparation document as proof that the deadline checks
or automation deletion already happened. Future training needs a separately
authorized window and a fully predeclared experiment; it cannot be inferred
from continued CPU tests or from any retained `exit0` campaign wrapper.

Preparation validation: the inventory's three totals reconcile exactly and its
`deadline_closeout_verified` flag remains false. Programmatic Markdown-to-HTML
inspection verified the four-row/four-column pilot table and all four links.
No screenshot-based UI check was performed. Local lifecycle/regression checks
passed1217 tests; this is source/evidence validation, not new policy evaluation.

Cross-host source verification: commit `24864da3e1a248efd3083545fa46d5e2c728a1a3`
was pushed to the exact feature branch and fast-forwarded onto a clean100.100
worktree. With CUDA hidden, the focused freshness/lifecycle, obstacle-observation,
hierarchical-obstacle and retention-plan suite passed155 tests in5.74s there.
The preceding live check found no running Duck unit or compute process, GPU0%
at45C/12MiB, and both protected SYSTEM services inactive. These remain
pre-deadline observations; they do not complete the final boundary checklist.
