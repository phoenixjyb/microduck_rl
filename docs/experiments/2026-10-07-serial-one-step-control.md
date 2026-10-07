# Fresh serial reductions during one nominal physics tick — October7

## Why this intermediate gate exists

The [actual-coupled RNE experiment](2026-10-07-rne-coupled-serial-control.md)
at50864a995f33 passed its separate paired and within-case forward gates, with
actual serial reduction outputs exact against independent arithmetic. It did
not integrate physics. Original repeats happened to be exact in that run but
still differed from reference arithmetic; historical failures remain negative.
That evidence cannot qualify a moving-state window or a learner.

The next smallest controlled question is whether serial CRB, CoM and bias RNE
remain correctly dispatched and arithmetically consistent during **one nominal
20-ms tick**, and whether two fresh executions repeat. This is an intermediate
engineering diagnostic, not the28-call/0.56-s policy capture/replay gate. It has
no actor, critic, learner, optimizer, rollout storage, external push or perception.
It must not be relabeled full-window qualification even if every check passes.

Work only on100.98 via`gw98-direct`, frozen environment, clean feature branch,
lean worktree`/home/yanbo/work/microduck_rl-com-entry-20261006`, until2026-10-07
13:00Asia/Shanghai (Unix1791349200). Preserve FilmBrain, old checkouts, all
failed services and historical raw evidence; no100.100, package/driver/cache
changes, graph, recording, raw perception, protected-service restoration or
physical motion. Feature-branch source commits/pushes remain in scope.

## Distinct control, fixed execution and evidence

New `SerialStepControl`, protocol`microduck-serial-one-step-control-oct7-v1`,
is a separately reviewed source implementation. It must not change, subclass
past guards, or monkeypatch the caps/method checks of any previous controller.
One process-local scope owns only the three private smooth entry references
`crb`, `com_pos` and `_rne_cfrc_backward`, restoring only its own references.
Original initializer/downstream/public-forward/helper/kernel references remain
pinned. At the existing sibling group2,7,11 only, dispatch the original kernels
sequentially in body2,7,11 order on their actual initialized live aliased arrays.
CRB retains7 logical accumulation requests plus one denseqM request,9 actual
accumulations plusqM; CoM11 logical/13 actual requests; bias RNE7/9. Sensory
`rne_postconstraint` remains original direct passthrough, no launch interposer
or force snapshot inside that call. Readbacks, ID validation, synchronization
and changed dispatch perturb timing, explicitly and intentionally.

The exact count is **21 forwards**: one constructor reset plus10 substeps,
each with one pre-Euler and one post-Euler forward. Constructor ownership must
bind the exact runtime/model/data before stepping. Only the audited plain
`WarpStanceRuntime`, eager packed checks,64 CUDA0 worlds (or2 CPU test worlds),
exact rigid16-body/21qpos/20DOF/14-control/no-tendon plant, nominal fixed motor
fields, and one zero10-action tensor with`capture_control=True` are allowed.
No explicit reset after construction, no automatic reset, no retry or additional
tick. First terminal, rejected motor row, missing/extra reduction, foreign
reference, changed storage/device/stream, nonfinite or changed source fails
closed and remains retained. Do not normalize arrays or repair caller RNG.
The producer's `explicit_reset_calls=0` means no explicit reset **after**
construction; constructor initialization itself calls the pinned reset once.
It is a source-bound claim checked against the observed21-forward sequence,
not a separate instrumentation counter or proof about an earlier job.

Two fresh serial cases execute sequentially in one isolated owned CUDA0 child,
under the existing inherited shared GPU lease. Identical caller seeds
CPU673/CUDA677 and private CPU977/CUDA983 forks account for constructor draws;
caller streams must be restored. Retain complete constructor/private step-end
RNG states, actual descriptor,21 full initialized/accumulated CoM and RNE
snapshots per case, exact dispatch/count/storage/source receipts, and complete
17-field constructor/final frames. These two frames are **different-time**
states: no within-case unchanged-frame exactness requirement is appropriate.

Retain all10 actual motor proposal/control snapshots, commands, torque, accepted/
rejected/live masks, before-step counters, voltage/gain/friction/damping, target,
correction, three-slot delay queue and committed control/history, not just maxima.
Both cases must execute10 accepted2-ms integrations in all64 rows, remain live,
have no terminal/timeout, leave applied external/generalized forces zero, retain
finite rewards, and have no graph/model/optimizer/storage construction.

## Separate checks, not a widened acceptance label

1. Independently authenticate every whole retained file and terminal before
   JSON/array decode. Bind exact source/tree/whole leaves, current test receipt,
   installed-source hashes, closed prior proof, actual service invocation/PIDs,
   lease, monitoring and host preservation.
2. Check all21 live initialized-to-output CoM and RNE recurrences, including
   body1-to-root0, and all dispatch groups/aliases/counts and sensory21 passthroughs.
3. Check literal initial home/zero inputs, accepted10-step clocks/counters,
   nominal zero-action targets/FIFO/correction, position-only commands, finite
   motor arrays, torque ceiling0.36Nm, retained voltage/gain and nonnegative
   friction/damping. Motor records are consistency evidence, **not independent
   BAM recomputation, physical motor characterization or thermal simulation**.
4. Compare complete corresponding raw packets,17 physical fields and the
   complete finite float32 reward/dense-term ledger across the two fresh
   cases. Preserve all negative counts/deltas/hashes; no tolerance turns a
   mismatch into bitwise exactness. Do not require unchanged state across time.

Emit separate moving-reduction arithmetic, execution-consistency and paired
repeat decisions. All historical-cause/original-pair/full-window/training/
physical admission flags remain false on every outcome. A positive20-ms result
permits only consideration of a separately predeclared longer no-update
capture/replay gate, not optimizer updates or a claim of new Duck skills.

## Required launch preparation and resource caps

Seven new owned paths: this declaration, three `stance_serial_step_` source
modules (control/probe/receiver) and their tests. Start at evidence-only
commit`f42a3343705b7b0321ed7f5af219022a844eb522`.
Preserve all previous source leaves. The bound inventory has663 whole leaves;
the73-file ordered CPU suite SHA256 is
`1f341edaf05efe261b6888b4e649aaca71a4ebb4b12cf00b8d33e0e6689f68f5`
(canonical JSON including terminal newline), with2250 tests and no omissions.
These counts must be independently remeasured before native launch. Initial source
work alone and an old CPU receipt cannot authorize a native run.

Intended retained units `microduck-serial-step-{tests,run}-<source12>.service`
and unique corresponding artifact roots. Existing cooperative zero-byte regular
GPU lock is opened/verified/inherited, never created, unlinked or reset. One
child only, no overlapping GPU task. Service300s, child240s, independent closeout
300s plus60s deadline reserve;6GiB CPU, CPUQuota200%, Nice10, no restart,
KillModecontrol-group, stop10s, RemainAfterExit/MemoryAccounting enabled. CPU
file cap64MiB; native individual files1MiB, total below16MiB, receiver output
below4MiB. Temperature below75C, used≤12,288MiB/free≥10,240MiB; require idle
before/after and observed sole owned child, retain honest WDDM unavailable memory.

The exact inventory is22 files: eight raw packets per case and six metadata/
log/test-proof files. Motor bytes563200 and masks7040 per case are complete,
not prefixes. Child JSON has a1MiB cap to retain all21×31 array-layout receipts
and21×27 actual accumulation body-ID trace rows in both cases. RNE snapshots
must be tied to the actual bound force-storage identity and complete24576 bytes.

These are preparation requirements, not a declaration that launch is ready.
Controller, producer, independent receiver and adversarial focused tests must
be reviewed, current complete CPU regression must pass, exact source pushed and
native fresh qualification closed before any native stepping. Failures are
diagnosed read-only before changes, and failed source-addressed namespaces are
never overwritten or retried. At13:00 start no new Duck work; retain durable
state and verify no owned active GPU process without restoring protected services.

## Source review and CPU checks before transport

The owner reviewed the controller, producer and independent receiver, including
the live runtime sequence. A separate read-only integration review found no
blocking producer/receiver mismatch. Runtime step/forward/reset identities and
code are pinned; all historical admission flags plus `native_qualified` remain
false. Focused tests:90 passed in9.47s. The complete73-file source/CPU suite:
**2250 passed in179.20s**, with CUDA hidden, on the Mac. Ruff lint/format and
staged whitespace checks passed; the referenced prior experiment exists.
These are source/CPU checks, not a native CUDA result. Native qualification and
whole-byte independent receivers must still close on the exact pushed revision.
