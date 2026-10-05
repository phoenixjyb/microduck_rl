# October 6 bounded CPU plant-field inspection

## Predeclaration

At local `07ef7544c1766ddeabc6f30d0c3de823064f6986`, a strict coordinate
diagnosis of the authenticated 2ECE partial trace stopped before emitting a
result: `local CPU selected-plant and assets match authenticated native descriptor`.
The failing tool whole SHA is
`e9a91ff5eb95a48a63ab1c151d535854bbca9fec81b2b6e1a01bfe635d6f5ad9`.
Read-only comparison found the same asset hashes and all other descriptor
fields, but a different compiled selected-field hash. Do not bypass that guard
to attach a local compiled model to the native trace.

The Mac CPU selected-field hash is
`7bdf08526f5178612ee3dc43eaf3cb751ace3dad8571bd535fbe946885e9f653`;
the authenticated native descriptor hash is
`6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f`.
This is not evidence of new packages, asset changes or a driver problem.
Actual array differences must be inspected before attributing it to architecture.

Permit a single **CPU-only** native model-field export at clean exact branch
`feat/athletics-obstacle-curriculum`, source
`2ecee471f7b999822ed19defd7e2d1c7ad09df07`, worktree
`/home/yanbo/work/microduck_rl-stance-replication-20260930`, via `gw98-direct`.
Do not sync source, modify installed packages, use CUDA, create policy models,
run integration, change an existing service or touch 100.100. Preserve all old
artifacts and failed services. Only this declaration and a later result/handoff
note may change in the fork. Finish with margin before 08:00 Shanghai.

## Exact tool, context and bounds

Reviewed export tool SHA:
`3ced21cdb5862ec1f4eb9a1ce8aa9ffe55af279ead5cfa66dced8219e6063ff7`.
It requires exact worktree, clean branch/source, Python/architecture and package
versions before/after; verifies builder/descriptor imports come from that
worktree; compiles only the CPU MuJoCo model; emits finite bounded selected
arrays and verifies they reproduce the descriptor's exact hash. It observes
CPU `ntendon`, DOF bindings and body names, never CUDA model fields. Torch CUDA
must remain uninitialized with `CUDA_VISIBLE_DEVICES` empty.

The reviewed Mac invocation passed on arm64 / Python 3.12.12. Its 38,830-byte
report whole SHA is
`a06f4dba7c06f16228880b8913c97ccbb9b165033e5385d5079323b369ac11f5`.
Its `ntendon=0` is a local CPU fact, not a native observation. An initial
unused-import lint warning was fixed; prior tool/data versions remain separate.
The reviewed tool passed Ruff, formatting and owner review.

Native context is x86_64 / Python 3.12.13. Both export contexts require exactly
torch 2.9.1, Warp 1.12.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, mjlab 1.3.0 and
better-actuator-models 1.0.1. Do not relax a mismatch.

Use only the unique retained user service
`microduck-cpu-plant-fields-2ecee471f7b9-20261006-0720.service` with
RuntimeMaxSec 120, MemoryMax 3 GiB, MemoryAccounting yes, CPUQuota 100%, Nice 10,
Restart no, KillMode control-group, RemainAfterExit yes, TimeoutStopSec 10 and
LimitFSIZE 128 KiB. Use one-thread numerical libraries and
`PYTHONDONTWRITEBYTECODE=1`. No GPU lease acquisition is needed for CPU-only
execution; leave the existing lease file and every unrelated workload intact.
Observe the actual active PID/invocation, then retain terminal exit/caps/peak
and the whole output/log hashes. Keep protected namespaces inactive and both
FilmBrain PID/restart identities unchanged.

Unique native output directory:
`artifacts/tools/stance-cpu-plant-fields-2ecee471f7b9-20261006-0720/`.
The export and log must each remain under 128 KiB. The helper's own selected
array cap is 128 KiB; total artifact budget is 1 MiB. Create no existing target
and overwrite no artifact. Receive/authenticate the complete output bytes on
the Mac before decoding, compare every selected field to the Mac export and
the retained native descriptor, and preserve exact finite discrepancies.

If the fresh native CPU compile matches the retained selected-plant descriptor,
its DOF map may label the authenticated scalar coordinate. Still retain CPU vs
CUDA observation separation; neither this export nor a coordinate diagnosis
captures RNE/CoM entry state or proves a runtime cause. Keep all full-window,
training and physical acceptance flags false. If the native CPU hash also
mismatches, retain that result and stop model-dependent interpretation.

## Result

The declared service completed from 07:17:45 to 07:17:55 CST, invocation
`1837a5778f8c40eba395321cfe626ec6`. PID 3271972 was observed active; the
terminal audit found `MainPID=0`, exit 0, `Result=success`, zero restarts and all
declared caps intact. Its raw terminal `MemoryPeak=966656` is retained as an
accounting value, **not an observed RSS or live-peak proof**. The 38,611-byte
`fields.json` whole SHA is
`e8160827a52db740a6edc2e89973e5313f0020ed7be79ef609c903a1cf8f6ace`;
the 4,437-byte `compile.log` whole SHA is
`6cd45b6b879425fdf0d9a10172385004c9b4b7db168d1273cd3fb8c30587d9c5`.
Both received files were authenticated before decoding. CUDA stayed hidden
and uninitialized; the native source remained clean at 2ECE.

The fresh native **CPU** descriptor exactly matches the authenticated native
descriptor; its CPU `ntendon=0`. All 34 selected array fields were compared
without a tolerance. The finite cross-host differences are:

| Field | Float64 reencoded differing scalars | Float32 reencoded differing scalars | Maximum absolute delta |
| --- | ---: | ---: | ---: |
| body_inertia | 11 | 0 | 1.0842021724855044e-19 |
| body_iquat | 17 | 0 | 2.8935187579293142e-15 |
| geom_pos | 109 | 10 | 1.3877787807814457e-17 |
| geom_quat | 72 | 15 | 1.5543122344752192e-15 |
| geom_size | 99 | 0 | 1.231653667943533e-16 |

All other selected field values match. These are comparisons of exported CPU
values reencoded as float64/float32, **not** full MJB fingerprints or observed
CUDA model bytes. They are consistent with cross-host compilation/arithmetic
differences, but do not prove an architecture cause, package/driver change or
the cause of the original same-host CUDA replay failure. Keep the failed strict
Mac model-binding guard; do not turn these deltas into an acceptance tolerance.

After reauthenticating the original 20 files and reproducing the exact partial
result hash, the separate coordinate diagnosis found both serial constructor
force negatives at `[environment=10, dof=4]`. The native CPU binding labels that
as `trunk_base_freejoint`, body 1, local DOF 4; it is a CPU annotation, not a new
CUDA binding observation. Bias values are `-0.004055817145854235` versus
`-0.004055815748870373` (bits `0xbb84e6a9` versus `0xbb84e6a6`); smooth force has
the corresponding positive values/bits. The other 15 recorded fields match
for that constructor pair. Constructor-to-frame-2 CoM negatives in each serial
child are eight scalars: four on root body 0 and four on trunk body 1, confined
to environments 10 and 53. These are post-forward observations, not actual
CoM/RNE kernel-entry state or proof of a particular reduction cause.

The first new analysis-helper attempt hit `AttributeError` before writing a
result because its import order selected an older private reader. The corrected
helper requires the literal authenticated reader/helper module paths before
analysis. It does not change either retained reader or the original failure.
The final 13,520-byte diagnosis whole SHA is
`79ce773aa3adb30bce03b28eff86f9c5de54b8ceb2e5015b1fb36a49d19bc45e`;
its tool SHA is
`d7f8d524247b100f30541751c867b1cb6d2f84e9014cf6dd50d3264ce83cb385`.
The 2,512-byte terminal owner packet whole SHA is
`c91a218e41095243ae55089f4a6202ee4054a2b8629a21bb306f2b887c6b3a73`;
its writer SHA is
`5d305c93334a14487666f20a68e88fc89ec8adc5452442a3c37b2ef1bbff00e5`.

Tools, reviewed Mac input, diagnosis, terminal packet and synthetic negative
test are separately retained at
`artifacts/tools/stance-cpu-field-diagnosis-20261006-0735/`. Complete file hashes
were checked on both hosts after transfer and the directory was synced. The
old 20-file inventory, seven-tool directory and failed services are unchanged.
The private negative tests passed: every terminal identity/cap mutation fails
before publishing, wrong whole-input hashes and symlink inputs are rejected.
Ruff and formatting passed; this documentation-only update does not need a new
native execution or another full source regression.

At the result audit the GPU had no compute PID, FilmBrain retained PIDs 521 and
298048 with zero restarts, and both protected services were inactive in user
and system namespaces. This is not a new GPU experiment or a repaired full-window
gate. Runtime-cause, full-window, training and physical acceptance remain false.
