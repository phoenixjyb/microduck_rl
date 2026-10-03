# Natural terminal/reset integration: bounded CPU2 diagnostic

Status: **v1 failed before collection; bounded v2 construction repair prepared,
not yet natively admitted**.
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
