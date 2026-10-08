# Dense solver probe preparation, October 8

Status: existing inspection tools resolved; CPU-tested dispatch/disassembly
guards prepared. **The capped native collector is not yet assembled or run.**
No historical source fence, solver, row order, numerical tolerance or training
gate is changed. The prior numerical rejection remains in force.

## Resolved tools without an installation

A broader read-only search found existing Triton 3.5.1 binaries under the frozen
WSL environment at
`/home/yanbo/work/microduck_rl-stance-replication-20260930/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin`.
Both identify themselves as **12.8.55**. They were not on PATH; the earlier
bounded Windows/system-CUDA-path inventory was incomplete, not an installation
failure. WSL has no `rg`, so the bounded package-tree search used `find`.

| Existing executable | Whole bytes | SHA256 |
| --- | ---: | --- |
| `cuobjdump` | 569,008 | `ad1d0f0699f46603416eb58e7b9fdf9a293e95f9312c9df7fd6fda56a6c30d41` |
| `nvdisasm` | 5,894,744 | `1080bc909a3020761ff4c78a9b7bd9b4937bdb9c0978709cbe52381de612a1f2` |

Their `--version` and cuobjdump `--help` completed with CUDA hidden and 10-second
timeouts. No driver, package, frozen environment or shared cache was changed.
Only an NVIDIA redistribution JSON manifest was downloaded while investigating
a fallback; no executable archive was downloaded or installed. Existing tools
are sufficient for the next protocol's byte pin, subject to checks at launch.

## Two separate guards

[Dispatch guard](../../src/mjlab_microduck/stance_solver_dispatch_guard.py):
standard-library import, no device initialization. It requires an externally
established `LoadedModuleBinding` for the literal target CUBIN, reads the pinned
installed solver path and compiles its Python source **without executing it**
to compare the actual loaded caller/kernel code objects. One controlled call
through unchanged `_update_constraint` must reach the literal dense branch.
The wrapper checks caller frame/code/globals/line and model/data/context
identities, exact dimensions and argument order, eager defaults, sm120 device,
held nonzero stream, buffer identities/pointers/layouts and nonoverlapping
ranges. The bounded recipe is 64 worlds, 20 DOFs, and `njmax=512`.

Only the owned launch hook is restored on exit; a foreign hook is preserved and
refuses a successful receipt. Direct target launches, duplicate dispatch,
changed source/code/defaults/stream/loaded binding, backward/tape/graph/recorded
launches, and buffer mutations are rejected. The guard delegates the original
call, synchronizes the same stream, and retains one observed Python launch.
It is **not** a GPU supervisor, binary loader, data-packet recorder, runtime
provenance proof, or numerical acceptance rule. It is not a sandbox against
hostile Python instrumentation. A future collector must supply those separate
bindings and preserve the model/context setup.

[Disassembly scope](../../src/mjlab_microduck/stance_solver_disassembly_scope.py):
reuses the unchanged bounded ELF parser to select exactly one literal dense
target forward symbol and whole function span. Offline cuobjdump output must
contain one exact `sm_120` envelope and function, the measured header flags,
contiguous 16-byte instruction offsets from zero, paired encoding words and
the measured terminator. Unknown formats, extra functions/architectures,
warnings, truncation or arbitrary trailing text are rejected. Reconstruct the
128-bit little-endian encoding bytes and require their complete length/hash
to match the selected CUBIN function. This binds **compile-input** bytes and
offline text, not actual driver-loaded instructions or dispatch.

Both keep native/full-window/runtime-cause/training/physical flags false.
No FMA/cause inference or numerical acceptance follows from a mnemonic string.
The [NVIDIA 12.8 manual](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html#cuobjdump)
documents function selection and paired instruction encodings; exact sm120
format details here come from the pinned installed tool's observed output.

## CPU-only tool-format calibration

The old retained original friction CUBIN was used solely to check tool output
format and encoding byte order, **not** as evidence about the solver kernel.
The intentionally nonexistent function selector returned exit zero with a
warning and no function. Exit zero alone is therefore insufficient.

For `_friction_dof_a00ce1f7_cuda_kernel_forward`, a separately retained offline
tool call produced 440,496 bytes, SHA256
`6e7d619c72b6ddf60976cb22f92be5eb0a699d45c6e3b16ac5cf3b389d07e2dd`,
and empty stderr. Its 1,376 encoding rows exactly reconstruct the 22,016-byte
friction function span, SHA256
`35dd8ca36669d21b5d140585b26c5974e8709250b246031ebb948facdbc2fd58`.
The retained CUBIN is 463,240 bytes, SHA256
`6da2211afb1d74efedb5dbc723e47bab089256fe97b0302def234f9f5c78ab61`.
The new literal solver selector correctly refuses it: no target promotion.

The complete calibration report is
`artifacts/tools/triton-binary-tools-oct8-0df0d07f/format-replay.json`, SHA256
`5dca3e83332907497e7f60c47d68e0a61f76535fdaa5a375ba987ab38ce0951e`.
This CPU replay does not initialize a GPU or observe solver instructions.
All original raw capture bytes remain unchanged. The parser's initial synthetic
format was incompatible with actual tool output; owner review and native
offline calibration corrected it before source freeze.

## Focused verification and remaining gate

Mac passed **210 checks in 11.39 s**, CUDA hidden: dispatch guard, disassembly
scope, static target, unchanged artifact binding, and historical ELF scope.
The dispatch fixtures execute real frozen Python function code with synthetic
device/buffer/launch objects; **they are not GPU evidence**. Disassembly tests
use synthetic ELF/text bytes. Independent read-only review prompted additional
wrapper-boundary and exception/restoration tests. Same-source WSL checks are
the next delivery step; a full repository suite has not been run.

The first same-source native CPU service at `1cbd8823f446` reached the end of
the test list but **failed** during JUnit serialization: pytest raised
`OSError: [Errno 27] File too large`. Its 8 MiB `LimitFSIZE` remained enforced;
the truncated XML is not accepted test evidence and the subsequent format
replay did not run. Read-only diagnosis found that the cap-test's automatic
parameter ID embedded its 8,388,609-byte payload in the test name. The Mac
XML was also oversized (8,685,489 bytes), with an 8,388,661-character name.
The repair assigns three short explicit IDs to that parametrization only.
The actual oversized input, refusal assertions, parser caps and native service
limits are unchanged. A fresh source-specific CPU service must pass with the
same cap; the failed service and its original artifacts remain retained.

At the turn's initial live check, WSL reported **73 C, 78% utilization,
7,293 MiB used / 16,869 MiB free**. No Duck CUDA child was started. The next
native protocol must separately bind the fresh child, clean source, frozen
packages/interpreter/runtime/compiler, installed tool bytes, private cache,
actual context/stream, explicit load and hook, and retained data packets around
the controlled dispatch. Require its reviewed resource/deadline supervisor,
GPU availability gate and workload-preservation checks before execution.
Neither these guards nor the earlier tensor-smoke protocol grants that
admission or resumes policy training.
