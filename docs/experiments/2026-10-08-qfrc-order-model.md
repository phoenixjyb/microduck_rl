# Qfrc accumulation-order model, 2026-10-08

Status: bounded CPU arithmetic implementation and review; retained outcome pending.
No new simulation, GPU run, learner, policy promotion or physical motion.

## Exact source and retained dependency

Base `5456225b0b7ce14a0c42fbfeb3a87c493a56d9cd`, exact feature branch
`feat/athletics-obstacle-curriculum`. Only three new paths: this document,
`stance_qfrc_order_model.py`, and its tests. Historical owners, cutoffs, captures,
receivers, source fences and packages remain unchanged. Bind the analysis to a
clean committed source and execute it on Mac and WSL with CUDA hidden.

Use the [initialized row view](2026-10-08-initialized-row-side-view.md) and its
whole original capture: source `e45a59c412bfe6bcf2de751f98159aa7fe0db463`,
480 leaves / 673,600,544 bytes. Inventory SHA256
`8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625`,
historical receiver SHA256
`c91d812df1278d1cfe208951d8dad6f23f5240639989d56407a6ca38ed08f999`.
Reconstruct the entire historical Git source binding, reauthenticate every
leaf and reproduce the full receiver before interpreting the arithmetic.
Do not mutate the existing row-view module to expand its source fence.

## Predeclared mathematical hypotheses

Model the dense `qfrc_constraint` serial row loop from frozen solver.py lines
1988–2010, SHA256
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
Only active `row < nefc[world]` contributes. Require complete paired active
coverage, no done worlds, equal paired captured J and initialized force terms,
and equal paired construction subset. Reject nonfinite active inputs and
observed force words; do not inspect inactive padding as initialization evidence.

Use exact integer dyadics, not host floating-point arithmetic, for two explicit
binary32 hypotheses: separately rounded multiply then add, and single-rounded
fused multiply-add. Both start from +0, use round-to-nearest/ties-to-even, retain
signed-zero rules and gradual subnormal underflow, and refuse any rounded
overflow (including an unfused intermediate). No actual GPU FMA contraction,
flush-to-zero behavior or schedule is inferred from selecting a hypothesis.

For each of the 64 worlds / 20 DOFs, retain bitwise comparison summaries and
whole predicted-bank hashes for both candidates' original stored orders.
Separately evaluate candidate1 in candidate0's original row sequence via the
existing unique payload/address/id/type mappings and same-offset noncontact
markers. This is an **analysis-only virtual sequence**; no row bank is rewritten,
sorted, repaired or fed back to a simulator. Shared-sequence equality is expected
by construction once corresponding input words match, not a new numerical gate.
Report observed/model mismatch patterns and each modeled order sensitivity.

Positive reproduction would establish mathematical consistency, not a runtime
cause, compiled arithmetic, solver equivalence or training authorization.
Negative reproduction is retained, not retried under undeclared math choices.
Do not generalize a serial force-sum model to atomic cost or tiled hessian paths.
All five qualification/training/physical flags remain false.

## Checks and retained outcome

Mac focused checks: 83 passed (35 arithmetic, 34 initialized-row view, 14
historical receiver), with no failures or omissions; Ruff and whitespace checks
passed. Independent read-only review found no blocking arithmetic or mapping
issue and independently repeated all 35 arithmetic tests. Full retained replay
is intentionally pending until this three-path source is committed cleanly.

The arithmetic alternatives follow the distinctions explained in NVIDIA's
[floating-point guide](https://docs.nvidia.com/cuda/archive/11.5.2/floating-point/index.html).
That reference does not establish which instructions this retained GPU run used.
