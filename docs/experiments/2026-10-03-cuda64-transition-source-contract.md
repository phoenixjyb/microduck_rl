# CUDA64 transition collector: source contract, not native qualification

Status: **owner-reviewed, source-tested implementation; no native collector job launched**.
This follows the independently closed
[fixed-input CUDA sampler](2026-10-03-cuda64-shadow-sampler-probe.md). That gate
used synthetic zeros and never stepped physics or wrote rollout storage. It
cannot be borrowed as a real CUDA64 transition, terminal/reset or training gate.
The current authorization ends **2026-10-03 20:00 Asia/Shanghai**.

## Smallest implementation boundary

The new sibling
[collector](../../src/mjlab_microduck/stance_recovery_cuda_transition.py)
does not change the CUDA-hidden CPU bridge, parent allowlists, reward gates,
actor44/critic50/action10 layout, motor model or physics runtime. It accepts only
the fresh prepared D1 CUDA64 objects and exact `ScheduledRecoveryRuntime` at
zero episode clocks, with an unchanged source-bound fixed row declaration.
The inherited shared lease is checked before CUDA access. An initialized single
visible CUDA0 and exact frozen host/source/profile remain required by the private
RNG context. No synthetic/device bypass is exposed in the production API.

There are at most **28** actual stock `PPO.act` / `process_env_step` transitions.
Both original bound methods, actual prepared model/storage/Adam ownership,
unchanged D1 model state, fixed schedule and current row clocks are checked.
The private CUDA state is scoped around each stochastic action; each scope must
advance it once, close cleanly at the expected count and preserve both caller
streams. Storage count must
match the collector's own successful-transition count.

The per-call order is:

1. Clone typed finite CUDA0 pre-action actor/critic observations; require the
   actor prefix. Sample the stock policy and retain raw actions, pre-action
   values/log probabilities and positive Gaussian mean/scale.
2. Pass a **separate raw-action clone** to the scheduled runtime, preserving
   unclipped PPO actions independently of motor correction/clamping. Retain a
   whole copied runtime result before any reset.
3. Validate finite environment reward, exclusive failure/timeout masks, actual
   runtime live complement, per-row clocks and 0–10 executed substeps. Live rows
   execute all ten substeps. A reported timeout must actually reach step 2500;
   no clock injection or forced terminal is an accepted native outcome.
4. Evaluate the critic **only for timed-out rows**, using the retained pre-reset
   critic observations. Store `environment_reward + .99 * V(terminal)` exactly
   once. Failure and live rows receive no bootstrap. Environment reward is not
   overwritten; empty stock timeout metadata prevents a second bonus based on
   the wrong pre-action value.
5. Store first and verify exactly one new storage slot containing the **raw**
   actions, pre-action observations/values/log-probs, target reward and done
   mask. Only afterward may the exact done mask be selectively reset.
6. Verify the returned terminal ledger, untouched row clocks, physical
   qpos/qvel/time, controls and observations; reset rows return to clock 0/live
   and their initial controls. Retain before/after reset data, private-state
   boundaries and the scope receipt. Return detached owned records rather than
   live aliases.

An exception at any stage latches `faulted`; no retry, duplicate write or resume
is permitted. The future supervisor must retain failed/partial child evidence
before diagnosing it. A completed collector has **no** returns/GAE, update,
optimizer-step, checkpoint or export API. Every record keeps all eight
capability/admission flags false, plus false native transition, natural-terminal
and selective-reset qualification.

## What the tests may establish

The focused suite uses explicitly **synthetic CPU stock RSL objects**, with
private context/device seams patched only in tests. It checks action aliasing,
terminal targets, raw storage, reset order/isolation and fault-latched refusal;
unpatched real CPU tensors are rejected by the production CUDA tensor guard.
Synthetic timeout clocks and synthetic terminals are source fixtures, not
naturally observed physical episodes or native CUDA evidence. Neither these
tests nor a finite result can admit a GPU training job.

## Next native gate remains separately predeclared

Before a collector probe: commit/push tested source; retain exact clean native
source and CPU test receipts; independently authenticate current prerequisites
and actual CPU parent construction; requalify the source-bound fresh preparation
and private sampler rather than replaying an old success Boolean. Declare the
exact 64-row physical schedule, seeds, plant/solver/force checks and numerical
gates, capped whole raw inventory, one leased workload, two idle samples and
unchanged FilmBrain/protected services. Measure/declare complete run plus
independent closeout budgets before launch. Old CUDA2 or shadow timing does not
provide CUDA64 physics throughput or a resource guarantee.

A 28-call fresh-runtime trace can reach only 280 accepted substeps, not the
2500-step natural timeout. Even a successful short probe leaves natural
timeout/storage/reset and untouched-sibling selective-reset gates open. They
require a separately designed naturally observed episode protocol; never inject
clocks or failures to manufacture qualification. Stock NaN-filling GAE and
CUDA optimizer replay remain unqualified for this sibling. A separate
[finite-return source helper](2026-10-03-cuda64-finite-returns-source-contract.md)
now computes checked GAE without invoking the stock NaN repair; it still has
no native qualification or optimizer/update path.

The [combined catalog coverage](2026-10-03-recovery-catalog-coverage.md) found
no unchanged-parent deficit across all 45 declared push cells. Keep that parent
unchanged; do not use this preparation work to launch
an unnecessary update. Hopping, obstacle avoidance and football balance retain
their separate ledgers, with no combined-policy or physical-motion claim.

## Reviewed source evidence

The final focused synthetic suite passed **44 tests in 7.36 seconds**. The owner
26-file recovery/preparation/sampler/transition regression passed **723 tests in
70.17 seconds**, with CUDA hidden and no skips. Separately, the unchanged CPU
bridge and stance-PPO tests passed **38 in 7.90 seconds**. Ruff 0.15.7 check and
format, compilation, document relative links and whitespace checks passed.

The bounded Luna test implementation/review used real CPU stock RSL storage
under explicitly synthetic private CUDA/context/runtime seams. It exposed a
TensorDict iteration error before integration; the owner corrected field-name
iteration and reran the final complete regression. Further owner review tightened
actual runtime clock/live binding, scope advancement/caller preservation,
untouched observation isolation and exact done-row reset qpos/qvel/time checks.
These are reviewed source checks only, not native CUDA transition/reset evidence.
The native WSL checkout remains frozen at the closed sampler source
`1b96ccaac1326d6f0a1e00cdff8950b91a2916be`; this collector has not been installed
there or used to change any Duck weights.
