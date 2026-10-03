# CUDA64 finite returns: source-only, no optimizer or admission

Status: **owner-reviewed, source-tested implementation; no native returns job launched**.
This separate [finite-return helper](../../src/mjlab_microduck/stance_recovery_cuda_returns.py)
follows the [28-transition collector contract](2026-10-03-cuda64-transition-source-contract.md).
It does not call stock PPO's inherited NaN-filling return implementation and
does not alter the frozen CPU bridge or any previously closed source/artifact.
The authorization cutoff remains **2026-10-03 20:00 Asia/Shanghai**.

## Fixed calculation and refusal boundaries

Only an exact healthy `CudaTransitionCollector`, with all 28 transitions in its
64-row CUDA0 storage and phase `full`, may invoke this helper. Its source,
model, fixed schedule, runtime clocks and inherited lease are checked before
CUDA operations. Stored rewards/values must be finite float32 `[28,64,1]`;
done masks must be uint8 of that shape with values only 0/1. Existing returns
and advantages must still be unwritten zeros. There is no resume, arbitrary
rollout/device input, synthetic-fixture argument, CLI or export path.

The actual critic evaluates the current next observations after any completed
reset. With fixed gamma `.99` and lambda `.95`, the backward recurrence is:

```text
continuation = 1 - done[t]
delta[t] = reward[t] + gamma * continuation * next_value - value[t]
advantage[t] = delta[t] + gamma * lambda * continuation * advantage[t+1]
return[t] = advantage[t] + value[t]
```

For the final row, `next_value` is the actual next critic value; otherwise it is
the following stored pre-action value. A done mask cuts both terms, so a reset
sibling/episode cannot leak into the preceding terminal. Timeout bootstrap
already added **once** by the collector is not repeated here. Failures never
receive a timeout bonus.

Raw advantages are normalized globally using the sample standard deviation
(`correction=1`) and epsilon `1e-8`, matching the declared existing finite CPU
contract. Every intermediate, return, raw advantage, mean/std and normalized
target must be finite. Finite-input overflow also rejects; there is no
`nan_to_num`, clipping or gate relaxation. Constant advantages yield finite
zeros through the fixed epsilon.

All target calculations complete before either storage write. Input rewards,
values and done masks remain unchanged, as do D1 model parameters, empty Adam
and private RNG. A CPU/CUDA0 `fork_rng` protects caller streams even on failure;
the math body must not consume either stream. The helper writes only returns
and advantages, never clears storage or samples minibatches, and sets phase
`returns-computed`. A repeat, incomplete rollout or error latches the collector
faulted and forbids retry; a partial copy failure is not silently repaired.

The result retains whole input/target tensors and next critic observations,
with `native_finite_gae_qualified=false` and all eight capability/admission
flags false. There is **no optimizer call or new checkpoint**.

## Evidence tiers and next gate

Positive source tests use the same explicitly synthetic CPU stock RSL collector
fixtures, with only private context/device/math-RNG seams patched in tests.
Independent CPU reference arithmetic, terminal-mask fixtures, zero variance,
nonfinite/overflow refusal and caller/private RNG preservation check the
contract, not native CUDA arithmetic or naturally observed terminals.

The native worktree remains at closed sampler source
`1b96ccaac1326d6f0a1e00cdff8950b91a2916be`. Neither this helper nor the collector
has been installed or exercised there. Fresh physical CUDA64 collection and
independent retained replay, natural terminal/storage/reset and native finite
return replay remain separate predeclared gates. Native optimizer qualification
is still unimplemented, and a learning job also requires an independently
replicated measured deficit. The passing push catalog does not justify an
update, new skill claim or physical motion.

## Reviewed source evidence

The bounded test contributor's focused suite passed **58 tests in 8.53 seconds**;
the owner's separate focused rerun passed **58 in 8.66 seconds**.
The owner reran the complete 26-file recovery/preparation/sampler/transition
regression after the final test additions: **737 tests passed in 78.63 seconds**,
with CUDA hidden and no skips. That suite is the unchanged 25-file `TEST_FILES`
tuple in [the closed sampler supervisor](../../src/mjlab_microduck/stance_recovery_cuda_shadow_probe.py)
plus `tests/test_stance_recovery_cuda_transition.py`; the old sampler's declared
679-test gate is not widened or relabeled. Ruff 0.15.7 check/format, Python
compilation, relative links across three documents, and whitespace
checks passed.

The tests include an independent Python geometric-series oracle for constant
rewards and zero stored values: an ordinary final bootstrap, a terminal at
step 5 that cuts the earlier episode, and a final terminal that suppresses
bootstrap. Separate tests exercise the actual math RNG context with the real
CPU `fork_rng` and explicitly fake CUDA state APIs: no draws succeed; CPU or
synthetic CUDA draws reject and restore both callers; a body exception after
both draws preserves the original exception and restores both callers.
Those fake CUDA APIs are **not** CUDA execution or native RNG qualification.

The bounded Luna test contribution was reviewed by the owner before the final
regression. The earlier **732-test / 68.86-second** run preceded the additional
five oracle/RNG checks and is not the final source suite. No policy, package,
driver, native worktree, protected service or retained artifact was changed by
this source slice.
