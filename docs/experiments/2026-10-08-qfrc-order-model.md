# Qfrc accumulation-order model, 2026-10-08

Status: same-source Mac/WSL CPU replay complete; mathematical consistency only.
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

## Retained same-source outcome

Executed analysis source `5307fe32536bce5868b63aac9e5ee3f1ba36bfbd` on both
hosts with CUDA hidden and all four numerical thread settings at one. Each
ran the exact 83 tests with zero failures, errors or skips, authenticated all
480 original capture leaves, reproduced the whole historical receiver, then
evaluated the predeclared alternatives. No new simulation or GPU job ran.

The whole reports are byte-identical: 190,794 bytes, SHA256
`154f7626bf0615420616ea709e3c24f89d10a1277b603ed4c48993b2fb53daee`.
The inputs cover all 2,944 active rows / 58,880 matched J-force term pairs;
1,616 contact row offsets differ. The observed two-candidate force banks have
477 differing words out of 1,280 (64 worlds by 20 DOFs).

| Predeclared model | Candidate0 mismatches vs observation | Candidate1 mismatches | Stored-order repeat differences | Observed difference mask reproduced |
| --- | ---: | ---: | ---: | --- |
| Fused multiply-add | **0 / 1,280** | **0 / 1,280** | **477 / 1,280** | yes |
| Separately rounded multiply then add | 529 / 1,280 | 574 / 1,280 | 300 / 1,280 | no |

Both analysis-only virtual common sequences are exact, as expected by
construction from equal paired input words. No captured bank was rewritten,
sorted or fed back to the simulator. The fused model's two stored-order output
hashes exactly match their corresponding observed hashes:

- Candidate0: `d83ee91b33f18b0c4d0ea20758ec9027a81717520029eb68848ba89fa133635f`.
- Candidate1: `13efc8a89215218f0a34d6bcecbf98520fb4eb525dbf36c7110ae85d3504bde3`.

This is **exact mathematical reproduction consistent with row-order
sensitivity under the fused hypothesis**. It is not evidence of the actual
compiled contraction setting, device instructions, FTZ behavior, atomic
schedule, a whole-solver cause or native reproducibility. Atomic cost and tiled
hessian remain separate paths. All five qualification/training/physical flags
remain false; decision `qfrc-arithmetic-order-model-only`.

### CPU owners and transfers

Mac owner PID is retained in its whole proof; elapsed 28.953721 s. WSL user
unit `microduck-qfrc-order-model-cpu-5307fe32536b.service`, invocation
`2b8adfbddd5347cfa680aec3c03974a2`, owner PID 4147736, finished successfully
with status zero and MainPID zero, 10:24:53–10:25:01 Asia/Shanghai. Its owner
elapsed 7.976959 s. Native enforced ceilings were 420 s, 200% CPU, 6 GiB memory,
64 tasks and 16 MiB/file; these ceilings are not measured peaks.

| Whole proof | Bytes | SHA256 |
| --- | ---: | --- |
| Mac | 1,042 | `a6fa30139b40221bc247a67914bd60fe2d55a2221de33c371291017a90e4438c` |
| WSL | 1,076 | `c36bde995891771cbd086c2f66fd6b6214ef78f1db23138a2df16da39a198757` |

Each proof binds exactly four leaves: JUnit, pytest log, report and replay
stderr. Both hosts reauthenticated both anchored proofs and all four leaves
after mutual transfer, including canonical serialization and whole-report
equality. Original directories are
`artifacts/tools/qfrc-order-model-cpu-5307fe32536b` on their respective hosts;
cross-host copies use `qfrc-order-model-native-5307fe32536b` on Mac and
`qfrc-order-model-mac-5307fe32536b` on WSL. The 673 MB raw capture was reused,
not duplicated by this analysis.

At native completion the exact GPU UUID remained unchanged, 30 C, 664 MiB,
0% utilization, with no compute process. FilmBrain services remained at their
original active PIDs, protected AI-mission services remained inactive, and the
frozen `.venv` symlink was unchanged.

## Next gate

Do not promote a policy or start training from this diagnostic. First review
an isolated native compiled-device/instruction binding for this exact dense
loop, with declared cache-state and byte bindings, before any new runtime
capture. A controlled runtime experiment needs its own bounded predeclaration;
do not alter the historical capture, fence, row order or solver as a shortcut.
Only a separate reproducibility/qualification gate can unlock the planned
multi-seed stance/recovery matrix. Retained obstacle skill, stance candidates,
hopping and rolling-ball curricula keep their separate acceptance status.
