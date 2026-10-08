# Retained Gauss arithmetic side view

## Predeclared scope

This is a new CPU-only side view based at
`0a66611a00b57bc28d95e2cec18e86b6211ecefe`, with exactly three changed paths:
the `stance_solver_gauss_audit` module, its focused tests, and this document.
Historical captures, receivers, replay source fences and installed libraries
remain unchanged. No GPU job, simulation tick, policy, optimizer, video or robot
motion is authorized by this checker.

Authenticate the frozen solver source and all three historical anchors before
JSON parsing or numerical decoding:

- Solver: 102688 bytes, SHA256 `bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
- Inventory: SHA256 `8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625`.
- Metadata: SHA256 `d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91`.
- Original forward-4 initialized packet: 5375152 bytes, SHA256 `b47122d67e3568bf4e413c3905b0ca329e727ee900b168f93dd4c5f27bce4441`.

Reuse the sealed 66-field decoder, exact recipe and all-active boundary. It
already rejects any active elliptic row: `contact.friction` is omitted but its
read is unreachable for the retained rows. `cone=0` alone is not that proof.
Neither this side view nor the prior replay supports elliptic-contact inputs.

## Arithmetic and decisions

For the pinned 20-DOF recipe, Gauss runs one thread per world in ascending DOF
order, without an atomic reduction. The mathematical expression is
`0.5 * sum((Ma - qfrc_smooth) * (qacc - qacc_smooth))`.
The setup kernel resets active Gauss to zero. Compare only captured
`context.gauss`, not `context.cost` (which additionally contains EFC row cost),
and not `context.prev_cost` (which holds an earlier cost).

Use exact rational arithmetic over decoded binary32 values. Report both explicit
IEEE binary32 round-to-nearest/ties-to-even hypotheses: separate rounded product
and accumulation, and contracted product-plus-accumulation. Both round each
subtraction and the final halving. Retain bit mismatches for both without
selecting one as the device implementation. Gradual underflow is modeled; signed
exact-zero identity and GPU flush-to-zero behavior are not inferred.

The predeclared numerical consistency bound is
`gamma_80 * sum((abs(Ma)+abs(smooth)) * (abs(acc)+abs(acc_smooth))) / 2
+ 80 * 2^-149`, where `gamma_80 = 80*u/(1-80*u)` and `u=2^-24`.
This conservative RNE-model bound accounts for subtraction, product, accumulation,
halving and subnormal rounding. It is not a promise about arbitrary compiled
CUDA. Compare the actual captured error as an exact rational, not the rounded
JSON display. Reject nonfinite values, overflowed hypotheses, malformed byte
schemas, nonzero done masks and out-of-bound results. Even success leaves every
qualification flag false.

## Checks and retained output

Run focused rounding, cancellation, underflow, malformed-byte, nonfinite and
authentication-order tests alongside the existing scratch/replay contracts.
Run the new checker on Mac and WSL at the same clean exact source revision with
`CUDA_VISIBLE_DEVICES=''`; outputs are exclusive files in
`artifacts/tools/retained-gauss-arithmetic/`. Authenticate and compare the two
outputs after separating their platform-specific source/path identity, if any.

## Remaining gate

This cannot verify `efc.Ma` production, setup input/output transitions, row-cost
atomic ordering, the following gradient/Hessian stages, CUDA dispatch, a full
simulation window or learning. The next native diagnostic must have its own
predeclared source/capture fence and test evidence; it must retain pre/post setup
and Gauss boundaries, preserve the all-active non-elliptic guard, and independently
prove capped-unit retirement before a further GPU job. Training remains paused.
