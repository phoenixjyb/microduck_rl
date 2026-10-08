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

## Retained closeout

Executed source: `d7dd2e673d36a02256c749c7c20cff62f1de9806` on the clean exact
feature branch on both Mac and 100.98. The same twelve focused test files (the
eleven retained scratch/replay contracts plus the new Gauss tests) passed:

- Mac: 472 passed, no failures/errors/skips; 18.68 seconds. XML 337297 bytes,
  SHA256 `cb7efc21feb47d94275f70f229fe3c4183f499fd22142fa591393c6c957830aa`.
- WSL: 472 passed, no failures/errors/skips; 17.73 seconds. XML 337301 bytes,
  SHA256 `46a1de0e1e5767d1d64ff9c9a1425844953bf44ee172da9d16ac6b3e67a56daf`.
- WSL CPU unit: `microduck-gauss-cpu-d7dd2e67.service`, invocation
  `b7775f9bf6ae4d5ab24d7defa21bf762`, `Result=success`, `ExecMainStatus=0`,
  `MainPID=0`, empty cgroup, zero restarts; retained exited state is not a
  running workload. CPU-only resource caps were 150 seconds, 4 GiB RAM,
  200% CPU quota, Nice 10, 64 tasks, 16 MiB/file and control-group termination.

The Mac and WSL checker reports were independently generated from their retained
anchors, copied back without overwrite, and compared as complete bytes. Both
are 29150 bytes with SHA256
`1bfd090d46ffe7fda826b165969f5311674b73ddfcc3fcbfbddf8d3f4e3083be`:

- `artifacts/tools/retained-gauss-arithmetic/d7dd2e67-mac.json`
- `artifacts/tools/retained-gauss-arithmetic/d7dd2e67-wsl.json`
- Paired XMLs use the same directory and source prefix, ending in
  `-mac-tests.xml` and `-wsl-tests.xml`.

Decision: `retained-gauss-arithmetic-consistent-only`. All 64 active worlds
have zero out-of-bound results and zero bit mismatches under either hypothesis.
However, every captured Gauss value is only `3.963656371842135e-15`; the largest
mathematical error is `5.546696567206813e-22`, while the conservative bound is
`0.0006766296188161527`. This near-zero case cannot distinguish the arithmetic
hypotheses or stress a meaningful acceleration cost. It is not a new full-tick,
runtime-cause, simulation, learning or physical acceptance result.

A read-only Luna review found no concrete defects in the rational rounding,
bound scope, hash-before-parse order or CPU-only behavior. The review did not run
tests or touch the GPU. It helped retain the explicit separation of Gauss from
total constraint cost.

After the CPU checks, the frozen interpreter/alias, package versions, libraries,
Warp and MuJoCo-Warp source trees and disassembler pins reverified unchanged on
WSL. No running Duck unit remained. FilmBrain services remained active at
PIDs 521 and 298048 with zero restarts; both protected AI mission services were
inactive in user and system scopes. The point-in-time WSL GPU reading was
8018 MiB used, 16144 MiB free, 0% utilization and 32 C. This is shared capacity,
not an exclusive-idle claim. No GPU lease was acquired or GPU job launched by
this side view. The full repository suite was not run.

### Next bounded experiment design

Before another full simulation tick, implement and test a distinct caller-stage
capture protocol for the unchanged setup and Gauss kernels. Retain actual
pre/post `init_cost` arrays to verify reset and prior-cost transfer; retain
pre/post EFC and Gauss cost separately so row-cost atomics cannot be confused
with Gauss arithmetic. Include a predeclared nondegenerate Gauss-only input
control (for example literal per-DOF `Ma=1`, smooth force `0`, acceleration `1`,
smooth acceleration `0`, whose twenty-DOF Gauss result is `10`). Label this a
synthetic solver control, not a physical model state. Preserve the complete
non-elliptic restored bank, finite inputs, explicit held stream, exact kernel
bindings, resource caps and external closeout. Do not use these synthetic
inputs for training or treat a passed control as full-simulation admission.
