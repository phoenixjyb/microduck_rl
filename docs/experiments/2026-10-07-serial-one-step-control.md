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

## Closed native result: arithmetic exact, moving repeat negative

Execution source`26e2ee8244b7bd05fb3290d655113e538bd1a989`, tree
`e3f919145e145281b49168d352b964523a3297bb`,663-leaf whole SHA256
`a0677ebf4facdc92cebb255faed309c1b6101b651dd95ace9306e8244d14bd50`.
The clean native worktree was fast-forwarded through the pushed fork tip using
a verified bundle; the old2ECE worktree and all prior evidence remained untouched.
Frozen six-package/version and three installed-source-tree bindings were
unchanged before and after execution; no driver/environment update was made.

Fresh CPU qualification: `microduck-serial-step-tests-26e2ee8244b7.service`,
invocation`e13b99f253a74e68928cb2fdfe114600`, actually observed owner3633063,
10:20:41–10:23:33CST. **2250 passed in136.28s**, no skipped/error/failed tests;
same successful terminal invocation,PID0, no restart, full declared caps.
Peak cgroup memory5700673536 bytes, below the6GiB cap. Raw receipt/JUnit/log
are durable at `artifacts/tools/serial-step-tests-26e2ee8244b7/` on both hosts.
Independent whole-byte closeout is
`artifacts/tools/serial-step-tests-closeout-26e2ee8244b7/`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| inventory.json | 317 | `86956ed41f194cdff76b4fa79095f5651e63495d5a53678ed344fe21a1c083fc` |
| terminal.json | 391 | `bcf3d3e1603aa37ce7b07476dbae3dffa9f4d143d7aad620e1570f7c9454a56e` |
| verification.json | 1881 | `193d4fdb402e3083f4e7fa2975e3e9a083a82b1f8ae2c884b863f859ec9ec268` |
| mac-verification.json | 1418 | `eceb855b147a63e2902ea77a78f69b901a8b49983f3e7e21c3733585a260e65b` |

GPU execution: `microduck-serial-step-run-26e2ee8244b7.service`, invocation
`c0aed85c5c19437185f1fed6e6f23cca`, observed owner3641571, actual child3642418
with PPID3641571,10:25:59–10:26:39CST. Service/child completed successfully
without restart; terminalPID0, full caps, peak cgroup memory3170848768 bytes.
The26 real monitor rows observed only the owned GPU child, maximum33C,
maximum1134MiB used/minimum23028MiB free. WDDM per-process memory unavailability
was retained honestly. Closure GPU idle at30–31C/661MiB, no compute PID.
FilmBrain remained active at521 and298048 with zero restarts; both protected
AI-mission services remained inactive in user/system namespaces. Lease inode35886
remained the existing zero-byte regular file.

All22 raw files are durable on both hosts at
`artifacts/evaluations/serial-step-run-26e2ee8244b7/`.
Both native and Mac independent receivers authenticated every whole file before
decode and emitted the same61159 bytes. Closeout
`artifacts/tools/serial-step-run-closeout-26e2ee8244b7/` is durable on both:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| inventory.json | 2500 | `60138c8381f91fb2c31541c51fc8e5dcfe2eaa1a6a3b46c8b59715403c3dce2c` |
| terminal.json | 390 | `232b94280d51a431c4027db38b97d911d303f7d0d9bea117a1182f0af094c721` |
| receiver.json and mac-receiver.json | 61159 each | `d8c38bf63780f40aaa0fc40492bd5a43b7e40c3290195ec59555b781727eee71` |
| verification.json | 1801 | `046293cc74747690f041da7f9aac5912a83c1a8127bf90cc67ca969384e14c7e` |

The three separate decisions are:

- Arithmetic: `moving-reductions-reference-exact`. All21 actual CoM recurrences
  and21 actual bias-RNE recurrences in each case match independent float32 math.
- Execution: `one-nominal-tick-consistent`. Both cases integrate10 accepted
  2-ms substeps in all64 worlds, remain live, and retain identical complete
  masks and RNG packets. Peak torque is0.01356003899127245Nm in both cases.
  No force, terminal, reset after construction, graph, actor, storage or optimizer.
- Paired repeat: **`one-tick-paired-negative`**. Constructor17 fields are exact;
 12 of17 final fields differ. Overall`fresh-one-step-serial-repeat-negative`.
  CoM weighted banks match, but CoM initialized/RNE input/RNE output/motor/frame
  packets differ. The complete dense-term ledger differs even though both
  reward minima/maxima equal0.08811777830123901.

| Final field | Different float32 cells | Maximum absolute delta |
| --- | ---: | ---: |
| cinert | 165 | 4.092726157978177e-12 |
| cdof | 1199 | 3.4924596548080444e-10 |
| qM | 1778 | 2.9103830456733704e-11 |
| qLD | 1183 | 2.3283064365386963e-10 |
| cvel | 4905 | 7.450580596923828e-08 |
| cdof_dot | 5118 | 3.725290298461914e-08 |
| qfrc_bias | 430 | 4.656612873077393e-10 |
| qfrc_smooth | 669 | 5.587935447692871e-09 |
| qpos | 629 | 1.280568540096283e-09 |
| qvel | 1079 | 8.940696716308594e-08 |
| qacc_warmstart | 1221 | 1.0967254638671875e-05 |
| ctrl | 716 | 3.725290298461914e-09 |

Full world counts, raw hashes, ordered-bit distances and finite deltas remain
in the receiver; near-zero ordered-bit distances are not physical magnitudes.
Tiny finite differences do not establish a physical hazard or a learned-policy
failure, but they **do not pass the predeclared bitwise gate**. No tolerance,
normalization, source-addressed rerun or RNG repair is used to turn this negative
positive. Historical failures remain unchanged and every admission/cause flag
stays false. There is no new learned skill, thermal acceptance or motor hardware
characterization in this result.

Next: a bounded read-only first-observed-divergence analysis over all retained
21 reduction records and10 complete motor proposals, before changing simulation
or launching another native job. Current17-field frames do not contain solved
`qacc`, constraint forces, complete contact/constraint state or solver intermediates;
the first observed difference cannot establish the first runtime cause. A new
solver-boundary capture would need a distinct tested/predeclared protocol and
fresh source qualification. Longer capture/replay and training admission remain
closed, even though the captured reduction arithmetic is exact.
