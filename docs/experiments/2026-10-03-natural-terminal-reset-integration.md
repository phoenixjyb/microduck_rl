# Natural terminal/reset integration: bounded CPU2 diagnostic

Status: **saved-record-only repair closed the full-timeout/reset integration
gate. Selective reset, recovery acceptance and training admission remain open.**
This is a sibling of the closed 28-transition/one-update diagnostic, not an
extension, recapture or relabeling of that evidence. The campaign deadline stays
2026-10-03 **20:00 Asia/Shanghai**; expired protocols remain unchanged.

## Hypothesis and separate gates

The unchanged exact D1 parent can be sampled in a fresh actual CPU2 scheduled
runtime through the first naturally observed terminal, retaining the real
terminal observation/critic target and native reset evidence. The old 28-step
RSL storage and GAE remain unchanged. No optimizer or RSL storage is created
for the new record-only ledger; no `process_env_step`, cursor assignment, early
reset, clock injection, synthetic failure or actor-observation expansion is
permitted. The diagnostic updated actor is not used.

The exact parent checkpoint SHA is
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`;
its actor/critic state SHA is
`e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329`.
Use seed **653**, the existing training dose rows `zero-wrench` and
`+x-2n-20steps-t250`, CPU2 eager packed runtime, unchanged plant, 44 actor/
50 critic/10 action dimensions and the frozen CPU math profile. The zero row
has its declared zero-force observation window at physics step 500 for 10
substeps; the +x row receives 2 N at step 250 for 20 substeps. Actual full
coverage therefore means window counts `[10,20]`, delivered nonzero counts
`[0,20]`, with full force arrays cleared before each unforced post solve.

Collect once, up to **250 policy calls / 2,500 accepted Euler substeps / five
simulated seconds**, or stop at the first natural terminal, deadline or fault.
Retain owned per-call actor/critic inputs, raw Gaussian action, log probability,
mean/scale, private RNG chain, actual clock increments, physical boundaries,
control/motor proposals and every scheduled force phase. The first terminal's
critic value must use its retained pre-reset observation; apply the existing
`reward + 0.99 * V(terminal)` bootstrap only to timeout rows, exactly once.

On that actual terminal, retain all terminal records and before-reset clocks,
live masks, qpos/qvel/time, controls and force arrays; reset exactly the real
done rows once; retain the returned records and complete post-reset evidence.
Then stop the entire collection immediately. Do not continue a new episode.

- Both rows naturally reaching the common 2,500-step timeout can close the
  **full-timeout and timeout-reset** integration gate. It does not prove
  selective reset, because there is no untouched live sibling.
- A naturally staggered first terminal can close a **selective-reset** gate
  only if the still-live sibling's actual clocks, kinematics, time and controls
  remain exact. It does not borrow full-timeout qualification.
- An early physical failure, deadline or Python fault remains retained partial
  evidence. Never retry or inject a terminal to manufacture either gate.

Independent closeout creates no environment. It rehashes the immutable raw
files and validates recorded-state/control/plant/force continuity, exact
unchanged-parent stochastic sampling, critic/bootstrap math, private/caller
RNG and actual terminal/reset records. This is not fresh whole-trajectory
physics re-simulation, thermal modeling, recovery acceptance or physical motion.

Recorded force-prefix consistency and full scheduled-window coverage are
separate facts. Every natural-terminal gate requires all actually retained
accepted force phases to be checked; only full-timeout requires the complete
250-call / 2,500-step schedule and both full windows. Pre-action critic input
is bound to the previous retained observation, not the after-action boundary
whose realized correction has already changed.

## Prerequisites and execution bounds

Before admission, commit/push the reviewed implementation and focused tests,
verify the exact fork revision and clean WSL worktree, and retain a capped
frozen-profile native source-test receipt. Require the existing closed saved
transition audit plus the exact **31-file** one-update prerequisite at source
`0f245126837220297d5bd93d58c021997d3c9fb3`, independent closeout SHA
`0e1154de7c2093a1ee9e9bb0fa4433b28577f448edccbbe6b8f973fcdae233ba`.
Rehash its inventory; its diagnostic weights are never the new parent.

Run and closeout are separate sequential retained user services. Predeclare:

| Bound | Fixed value |
| --- | --- |
| Collection wall cap | 180 seconds |
| Run service cap | 360 seconds |
| Independent closeout service cap | 300 seconds |
| Retention margin / complete launch reserve | 60 / 720 seconds |
| Memory / CPU / priority / kill scope | 2 GiB / 200% / Nice 10 / control group |
| Raw capture limit | 128 MiB, plus the exact 256,368-byte parent |
| Run inventory / closed inventory | 5 / 6 files |

These are conservative diagnostic ceilings, **not** a measured CPU2
full-episode throughput guarantee. The existing CPU1 full-screen and short
CPU2 measurements justify an in-scope bounded timing probe, not extrapolated
training throughput. A hard kill may leave only parent/launch/journal evidence;
such a run cannot be called captured, complete or accepted.

Use the unchanged existing shared GPU lock, never unlink it, and require two
idle GPU samples before and after each service. Only the named Duck service
may be active. Freeze the exact WSL source while it runs; preserve FilmBrain,
all three previously failed service invocation IDs and historical raw evidence.
Keep both protected AI Mission services inactive in system and user scopes.
Use startup `CUDA_VISIBLE_DEVICES=` plus the exact existing profile settings;
do not alter drivers, packages or host services. Diagnose any failure read-only
before separately reviewed repairs. No retry of a failed unit or old namespace.

The numerical gate may open the next runtime-preparation step only. It changes
none of the eight existing capability/admission flags. A training curriculum
still needs a reproducible **unchanged-parent** deficit, retained passing cells,
harder held-out cases and separately qualified native learner throughput.

## Reviewed source checks, before native admission

The 14-file focused regression suite passed **302 tests in 22.98 seconds**
on the Mac with startup `CUDA_VISIBLE_DEVICES=`. Ruff and `git diff --check`
passed. These are source/schema and existing regression checks, not the new
native terminal/reset evidence. A first local command omitted the required
CUDA-hidden setting; eight existing guarded CPU tests refused that command.
The corrected startup setting passed without weakening their guards.

The review separately checked initial versus post-action observations, exact
sequential float32 time increments, original controls retained before
collection, frame/terminal linkage, restored done-row state and untouched
sibling controls. Pure fixtures test malformed receipts and contradictory
claims without presenting synthetic terminals as native qualification.

## Native v1 refusal and v2 predeclaration

Source `f04e07ce157d247a61558701019dab885be84f74` was committed/pushed and
fast-forwarded into the clean WSL worktree. Its capped native source-test unit
passed **302 tests in 28.27 seconds**, invocation
`e0c6bca383b54952a7fa832873d4a0e1`; source-test receipt SHA
`91399a9ff4e6c0c0cccff8376c888d348326f6b66819c25f2babdd25085c2f09`.
The source transition receipt SHA is
`d8dcc5918662fc19f19a3676bdda53200c0a59ce3e2fefa5e3cb84257e1522b2`;
external run-admission receipt SHA is
`04ab7ecb41a6e757218b02ffbf4f1f847a4c9b31f9aa249574a52c870961922f`.

The native v1 run unit `microduck-cpu-terminal-run-f04e07ce157d.service`,
invocation `4f00a3301c27432781dfc184030a07f0`, failed with exactly
`TypeError: dict() got multiple values for keyword argument 'cpu_math_profile'`.
Read-only diagnosis verified the host context already owned that field and
the launch constructor supplied it a second time. The failure occurred after
CPU environment construction but **before launch serialization or any policy
collection**: zero calls, zero optimizer steps, no capture or reset evidence.
Runner elapsed time was **3.789056434063241 seconds**, memory peak 574,451,712
bytes. Do not call this a failed learned policy or completed timeout attempt.

The immutable original directory contains exactly the 256,368-byte parent and
1,791-byte report. Report SHA:
`136f069740526963f6db0160533a617d56c821a2e88b4841b6e8e2957525db2f`.
The retained diagnostic receipt SHA is
`5c9524f1e02a878d63c1249e02fca8833004dd7ef62532d8c961bac95d6ed9b7`;
the exact 7,695-byte invocation journal SHA is
`d759f90ed3da24572975aa182be02b9c8aa9fb60bd1b15ced41e555bfeccd7b2`.
Both files live in
`artifacts/tools/terminal-launch-construction-failure-f04e07ce157d` on WSL;
all four files were mirrored and byte-verified on the Mac under
`artifacts/retained/terminal-launch-failed-f04e07ce157d.zujnXr`.
The failed service and all three earlier failed services remain unchanged.

The separately reviewed repair uses supervisor protocol
`football-b1d-cpu-stochastic-first-terminal-probe-v2`. It emits the verified
context's CPU profile once, explicitly compares it with the actual capture
profile, and tests the real launch constructor, not only hand-built receipts.
Every fresh admission and postcheck authenticates the original failure bytes
and all four failed service invocation IDs. The failed source/namespace cannot
be reused. The trace protocol, unchanged parent, seed, force schedule, numeric
gates and all resource/cutoff bounds above remain identical. Commit/push and
capped frozen-profile native source tests must pass before a **new source-bound
unit** can collect once; never restart or overwrite the v1 attempt.

Repair source checks: the unchanged 14-file regression selection passed
**309 tests in 23.77 seconds** with startup CUDA hidden; Ruff and
`git diff --check` passed. The seven added cases cover the forbidden failed
namespace, authenticated original failure, whole-file tampering and actual
launch construction/profile disagreement. Native v2 qualification remains
open until the separately capped run and independent closeout finish.

## Native v2 collection and immutable closeout failure

Exact source `9ad48b96abf6528c2f837399dc8fedd83a286464` passed the capped
frozen-profile native 309-test selection before collection. Its successful run
unit `microduck-cpu-terminal-run-9ad48b96abf6.service`, invocation
`f67c95456c3842f5a54035aa4d82a149`, captured **250 policy calls** in
28.991513116052374 seconds; the complete runner took 39.78638452687301 seconds.
Metadata records `first-natural-terminal` with no collection failure. No
optimizer or policy update occurred.

The original directory
`artifacts/evaluations/stance-wsl-cpu-stochastic-first-terminal-9ad48b96abf6`
contains exactly these five immutable files:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| checkpoint.pt | 256368 | `2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5` |
| launch.json | 25858 | `4529c25ce1a728805196a7c8b7be290ab87f019239aae24cf40a6da31fb50bf5` |
| capture.pt | 49950070 | `82ef6175674a520448a3e6cfc19ca075377f1f2f90428cb945883f1e6695db76` |
| capture.json | 2664 | `76633ec5c77aed675a632190d9c1b2615178930e9f2816a3f0427d19f79ad6e8` |
| report.json | 5584 | `712a7201bad070b8c431f3310cb27566799f14af7832bda77af0feff9237a98e` |

Independent closeout unit `microduck-cpu-terminal-closeout-9ad48b96abf6.service`,
invocation `ab8b2d101f6f42a1ba94d7f8f9aa5df4`, failed with exactly
`ValueError: trace tensor layout: original reset qpos`. It remains failed,
PID 0, restart count 0, status 1. Memory peak was 574603264 bytes.
No independent-closeout receipt was written. This is a verifier failure, not
evidence of policy failure or accepted terminal/reset integration.

Read-only, hash-before-load diagnosis found the actual collector retains
`terminal_reset.initial_qpos` as float32 **(2,21)**, exactly equal to the
physical trace's initial qpos. The scorer incorrectly demanded **(21,)** and
broadcast it. Both actual rows record timeout at step 2500, neither terminated;
before/after live masks are `[false,false]` / `[true,true]`, reset counters are
`[2500,2500]` / `[0,0]`, and one actual reset was called. Those are recorded
facts, not an independently accepted gate. There is no untouched live sibling,
so this recording cannot qualify selective reset.

Durable diagnosis directory
`artifacts/tools/terminal-reset-qpos-failure-9ad48b96abf6` contains exactly a
receipt and the original invocation journal. Receipt SHA:
`fd83806a4fb92807e35289208f3d33c95ae6af1ef7d12c8ba0fbfedd953c2b1a`.
The 6203-byte journal SHA is
`cce0d9b412a5bd86e3d485b0a188a4e40dc6fa93ea000707ec2bf760e0ea4378`.
All five failed service invocations, prior raw artifacts, closed one-update
evidence, FilmBrain and protected services remain unchanged.
All seven original/diagnostic files were copied and byte-verified on the Mac
under `artifacts/retained/terminal-reset-failed-9ad48b96abf6.7kplFw`, totaling
50265589 bytes. A file-only regression over the authentic saved qpos passed
the strict per-world check and rejected single-row/wrong-dtype variants. It
does not qualify this Mac for the frozen WSL inference/replay profile.

## Saved-record-only repair predeclaration

The smallest source correction validates both recorded and physical initial
qpos as finite CPU float32 **(2,21)** and compares every row exactly. It does
not reshape, broadcast, weaken tolerance, change the collector, or alter the
numerical gate. Original trace source SHA
`c18ea88da58ba8eac3f387e02175f1646ccbe0529ce17f716c30d91439524453`;
fixed strict-layout trace SHA
`a746923941c514e58e9b785b4fa645dc75275e9f7eee457999b3427be110f1c0`.
The unchanged original probe SHA is
`f863a9edd9c034de9c14aa6230bde6734c0e9924c70a3f795ca55eea2695ca13`.

A new source-bound sibling audit must authenticate those original five files,
both pinned failure files and the untouched failed invocation. Its transitive
source inventory allows **only** the strict terminal-trace leaf to differ from
the immutable artifact source. Actual host context must match the launch apart
from the evaluator revision. All earlier profile/source/parent, failure and
closed-optimizer prerequisites remain required.

The audit creates **no environment**, policy rollout, simulator reset, storage
or optimizer. It independently verifies the saved raw stochastic policy, RNG,
physical/control/full-force ledgers, terminal critic/bootstrap and real reset
record under the frozen CPU profile. It writes only a new unique three-file
audit directory, never the original namespace. Use one retained user service:
**180-second cap / 240-second launch reserve / 2 GiB / CPU 200% / Nice 10 /
control-group**, startup CUDA hidden, shared GPU lease and two idle checks
before and after. Commit/push and capped native source tests precede launch.
Do not retry the failed closeout or recapture an episode to repair the receipt.

The strict-layout source/schema checks passed **105 tests in 7.70 seconds** on
the Mac; the unchanged 14-file regression selection then passed **314 tests in
23.39 seconds**. Ruff and `git diff --check` passed. Distinct per-world positions,
single-row broadcasting, wrong dtype, nonfinite data and malformed physical
layout are covered. These checks do not close the native gate. Any accepted
saved-record decision must still preserve all eight capability flags as false;
full timeout/reset, selective reset, skill improvement, thermal modeling and
physical acceptance remain separate claims.

The saved-record sibling is
`football-b1d-cpu-stochastic-first-terminal-receipt-repair-v1`, implemented in
`stance_recovery_terminal_receipt_repair.py`. Its unique service/output names
include the evaluator source; the original artifact source stays `9ad48b96`.
The combined **15-file** source/regression selection passed **338 tests in
69.16 seconds** on the Mac with startup CUDA hidden, including the authentic
saved failure reader and strict source-closure checks. Ruff and diff whitespace
checks passed after owner review. These tests are not native replay evidence.
Final focused checks after the canonical-newline writer/concise CLI review and
three additional whole-file-tampering cases passed **27 tests in 25.18 seconds**;
Ruff and diff whitespace checks passed again. The future native selection has
341 tests; its actual pass count and no-skip journal must be checked before audit.

## Native saved-record repair: closed integration, unchanged policy

Exact evaluator source `3338b5f381e9a9ae39f73240e5ad49b3b1bd547c` was
committed, pushed and fast-forwarded into the clean WSL worktree. Before audit,
the capped native source-test unit
`microduck-terminal-repair-tests-3338b5f381e9.service`, invocation
`23f7f077f931497989079526f34aadd0`, passed **341 tests in 41.17 seconds**,
without skips. Its receipt SHA is
`c06c486fa60b9d5af6253a6d0f867bb4b9ad5c3791713198c6b01e28810d2218`.
The source-transition and external admission receipt SHAs are respectively
`28438e834745d6dab82cd8cd185b977545fe7bc25548ceb510f792a31477eac1` and
`3395c17d68b143425180237156bd472933df374c10476ce86b7c73aa3f95a7a8`.

The saved-record audit unit
`microduck-cpu-terminal-receipt-repair-3338b5f381e9.service`, invocation
`25578ce25d4248178039a62f4a3ee425`, finished successfully, status 0,
inactive, PID 0 and restart count 0. Audit elapsed time was
**20.182555282022804 seconds**. It created no environment, rollout, simulator
reset, storage, optimizer or revised checkpoint. All original five artifact
hashes, both failure files, all five failed unit invocations and the 95-leaf
source closure were authenticated again; only the predeclared strict-layout
trace correction differed from the original closure.

The new three-file directory is
`artifacts/evaluations/stance-wsl-cpu-terminal-receipt-repair-3338b5f381e9`:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| launch.json | 40175 | `47fd7b5ea76550367a080604e38c3266547cd46640bddbf165510bf520f3a699` |
| source-inventory.json | 22760 | `09e5e579051a651c2e6997a94017df34345b890b0ccfdaba1173e0130e3ca10e` |
| receipt.json | 50061 | `53af66761f38c0612f929e3b0af1e8587956f6e4b3b33378767a7b18d31076a5` |

The independent decision is **`cpu-natural-full-timeout-and-reset-replayed`**.
The retained unchanged parent's 250 stochastic policy calls and 2,500 accepted
physics steps passed exact actor/critic/RNG, control/plant and every force-phase
check. Full window counts were `[10,20]`, nonzero delivery counts `[0,20]`.
The terminal critic bootstrap and one original common timeout/reset passed;
`original_full_timeout_reset_qualified=true` and
`original_selective_reset_qualified=false`. There is no untouched live sibling.
This verifies the existing recording, not fresh whole-trajectory resimulation,
motor thermal modeling, learned skill improvement or physical qualification.
All eight capability/admission flags remain false.

Post-audit verification retained the exact process and unit-manager journals
plus a separate receipt in
`artifacts/tools/terminal-repair-verified-3338b5f381e9`; receipt SHA
`3552da22826a0073db57ed48ade51aea7bac8bea48567f3c7ef5e9b3999177f6`.
The source-test and audit process-journal SHAs are
`a2037651726a84132d12ffe6ef696eea9e3cee31a0096737910eca20749a0539` and
`20d8c43bd531012b55c19936e3f33421b8f9d54f88fb7befbf4494e1957446e4`;
their manager-journal SHAs are
`2619b640151d2a5767fd855a9ac4be745a3dceede59d8a6f04de768ee1c583ce` and
`d77d3107e1c13c221b1950fef5ed303f35a1b744ba4999a3b73019bf41ca2491`.
Manager journals report source tests at **2.0G memory / 2.0G swap peak** and
audit at **979.3M memory / 0B swap peak**. Successful tests are not a native
training-throughput measurement; the test swap observation is not zero.

Eleven new metadata/journal files totaling **298264 bytes** were copied and
whole-byte verified on the Mac under
`artifacts/retained/terminal-repair-closed-3338b5f381e9.KLlNRi`. The original
seven-file raw/failure mirror was not recopied. GPU postchecks found idle
occupancy, approximately 30 C and 696 MiB in use; FilmBrain and both protected
service scopes remained unchanged. The failed original closeout stays failed
and its namespace remains immutable.

The next scientific step is a separately predeclared unchanged-parent gentle
timing/diagonal gap screen. The previously passed strong cardinal matrix does
not justify training it again merely to consume GPU time. Any future learning
requires a reproducible trainable deficit and its own native learner gate.

Closing documentation checks passed: **132 focused terminal/repair tests in
27.11 seconds**, file-only receipt semantics and all eleven documented hashes,
local-link checks and diff whitespace checks. The Mac checks authenticate
retained files and source behavior; they do not replace the actual frozen-profile
native audit above.
