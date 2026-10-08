# Dense solver replay diagnostic — 2026-10-08

## Purpose and acceptance boundary

This experiment follows the authenticated scratch/bootstrap work at
`e6cf84d9615ee626390e2b691fbf08889efb0137`. It does not change the historical
solver-init rejection, rewrite rows, canonicalize inputs, change installed
libraries, or authorize training. It asks a narrower question: does one guarded
execution of the frozen dense target, inside its unchanged caller, agree with
an independent ascending-row float32 reference on the actual captured inputs?

The CPU owner and receiver import no CUDA packages. The fresh child authenticates
the exact source, frozen runtime, lease, unit and historical bank before CUDA
initialization. Source scope is exactly the new probe, receiver, their two test
files and this document. Both machines must pass the same eleven-file CPU suite
at the committed source before live admission; whole XML hashes are declared.

No simulation tick, solver initialization/search/iteration, policy, optimizer,
camera perception, learned-skill promotion, physical motion or MP4 is in scope.
All five qualification/authority flags remain false even on numerical agreement.
Compiled CUBIN bytes and full selected offline SASS are bound to observed Warp
load objects and the launch receipt; driver-consumed bytes/JIT code are not
observed. Readback changes timing. This is not a full-window CUDA qualification.

## Retained input and exact call

Input is the original arm's initialized forward-4 bank, not a synthetic zero
fixture and not the dense target's already computed before-input packet:

- `solver-init-tick-closeout-e45a59c412bf/inventory.json`: SHA256
  `8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625`.
- Historical `child.json`: SHA256
  `d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91`.
- `solver-init/original/forward-04.initialized.bin`: 5,375,152 bytes, SHA256
  `b47122d67e3568bf4e413c3905b0ca329e727ee900b168f93dd4c5f27bce4441`.

All three whole anchors precede JSON/schema decoding. Exactly 24 necessary
fields are staged and restored byte-for-byte into fresh allocations under one
dedicated stream. The frozen ABI stores schema field `contact.nacon` on
`data.nacon`; the real CPU allocation test covers this mapping.

The plant topology and selected compiled descriptor stay fixed; MuJoCo Warp
allocation uses 64 worlds, 20 DOFs, 512 constraint rows, 128 contacts/world and
the literal dense pyramidal Newton recipe. Allocation must not preload the
solver module in the CUDA context. The complete frozen solver module is
compiled for sm120 CUBIN and explicitly loaded after exact offline-disassembly
authentication. Installed `solver.py` remains SHA256
`bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.

Original `_update_constraint(False)` executes once. Capture synchronizes after
the preceding EFC writer, reads complete `nefc/J/force/done/qfrc_constraint`
banks before the target, and reads them again after it before the following
Gauss-cost launch. The pair is 5,515,904 bytes. Receiver requires unchanged
entire input banks, finite active products/output, all 20 output DOFs/world,
and bit-preserved output rows for done worlds. Inactive padding is retained,
not interpreted as physical data. Diagnostic tolerances are absolute/relative
`2e-5`; mismatch is reported, not repaired or promoted to a runtime cause.

## Native shared-capacity declaration

Host: `gw98-direct`, exact worktree
`/home/yanbo/work/microduck_rl-com-entry-20261006`, exact feature branch.
Frozen alias, library bytes, complete Warp source/header tree and complete
MuJoCo Warp Python source tree are authenticated before and after replay.
Torch 2.9.1, Warp 1.12.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, mjlab 1.3.0,
better-actuator-models 1.0.1 and Triton 3.5.1 remain unchanged. CUDA device is
`cuda:0`, sm120, UUID `GPU-7d72b360-33bc-2cee-3ff4-a954474011b5`, driver 595.95.
Process-local PCH disablement bounds generated files; no installed edits occur.

This is explicitly a shared-capacity lane, not an idle/exclusive GPU claim.
Both Windows and WSL samples must show at least 10,240 MiB free, at most
12,288 MiB used, temperature below 65 C, utilization at most 85%, and aggregate
growth no greater than 2,048 MiB versus the admission sample. Windows engine
activity is retained; it is never treated as proof of idleness. Existing
FilmBrain PID/restart identities and protected-service inactivity must remain
unchanged. No unrelated workload may be stopped or signalled.

Acquire the existing empty Wan lease nonblocking with no create/replace/write;
pin device 2096/inode 35886 and pass its held FD to the reviewed child. A unique
source-bound output directory, fresh private Warp/CUDA/XDG/Torch cache paths,
whole paired CPU receipts and owner declaration are required before spawn.

Effective user-unit caps: Type=exec, RuntimeMaxSec=480, MemoryMax=6 GiB,
CPUQuota=200%, Nice=10, TasksMax=64, LimitFSIZE=16 MiB,
TimeoutStopSec=10, KillMode=control-group, Restart=no. The newly supplied
wall-clock deadline must reserve the entire service, 120-second external
closeout and 60-second margin, and be at most 1,200 seconds away. No expired
historical cutoff is reused.
`system_ram_bytes` names the unit's hard MemoryMax, not a claim about host RAM
headroom. The exact expected user-1000 app.slice cgroup path is bound to the
source-bound unit ID and invocation before kernel membership is inspected.

The supervised child budget is 240 seconds; admission probes have 25 seconds
and owned cleanup has 5-second TERM/KILL budgets. Only the newly created,
birth-bound child session is signalled. The child has one joined disassembler
subprocess with a 20-second timeout and bounded retained output; it does not
detach or daemonize. Proc scans cannot prove an arbitrary forking tree empty.
On successful return the owner additionally checks the kernel unit cgroup has
only itself before closing its lease FD. On failure, retained failure evidence
and the enclosing unit cgroup backstop are mandatory; any inherited child FD
continues holding the lease. An external observer must verify MainPID=0, no
remaining unit cgroup tasks and final workload/telemetry state before receiving
artifacts or admitting another job. No owner receipt claims unit retirement.

## Collection and deterministic closeout

Do not launch from a dirty tree, use a general training command, or overlap
another Duck job. After a live run, retain the exact unit invocation, result,
exit status, process/cgroup closure and before/after workload samples separately
from the artifact receiver decision. Inventory every regular output leaf
(including empty logs) externally: at most 256 leaves, 16 MiB per leaf and
128 MiB total. Authenticate all leaves before interpreting receipt fields.
Bind that inventory hash and the independent receiver JSON/hash in the closeout.

The standalone runner is:

```text
CUDA_VISIBLE_DEVICES='' PYTHONPATH=<native-root>/src <native-root>/.venv/bin/python
  -m mjlab_microduck.stance_solver_replay_probe
  --source <exact-tested-commit> --deadline <fresh-unix-deadline>
  --cpu-evidence <whole-hashed-paired-cpu-evidence.json>
```

This command must run inside the exact capped source-bound user unit; direct
shell execution refuses admission. Failed output directories are preserved;
no retry overwrites them. Further numerical diagnosis or a new launch requires
review of the retained failure and a new source-bound declaration.

## Current delivery state

Runner/receiver assembly is under CPU integration and independent review.
An isolated real CPU allocation/restore check has already caught and corrected
the aggregate-contact-counter ABI mapping before any live launch. Native GPU
execution and training remain unclaimed. The final CPU evidence and any live
closeout will be appended only after their actual completion.

The native WSL CPU compile was rechecked read-only: descriptor SHA256
`91838baeac031299a008709e44cef879bdbc9bc769e84fcb2b318453ac115501` and selected
fields SHA256 `6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f`
match the retained native fixture. Mac ARM64 has a distinct selected-fields
fingerprint; all other descriptor fields match. Its whole descriptor SHA256 is
`2308c6a2e24f74646cefb264df0e3ce2e94957e2396f2af3fc7ee954e0118b85`. The CPU
allocation test checks the applicable platform fingerprint; it does not assert
cross-platform bitwise compiled-plant equivalence or diagnose that difference.

### First committed native CPU check: refused, no CUDA launch

Source `85742501c1c41a33ba34788273efd2fbd85b635b`: Mac eleven-file suite
382 passed in 16.23 seconds. XML: 325,243 bytes, SHA256
`37eb44e8abea9335bee3cf1f992a49650e3bdb1c7650b38a3e4d59ef8e9c174f`.
Native source/runtime/input CPU preflight authenticated tree
`044a56f63ce89389e47493068da981a9432ee8fb` (956 whole source leaves),
frozen library/source/tool pins and the retained input bank.

Native CPU unit `microduck-dense-solver-replay-cpu-85742501.service`, invocation
`205876c79293448783e840225691b0b0`, returned 1 failed/381 passed in 19.83 seconds.
It retired with MainPID=0, NRestarts=0, Result=exit-code and empty ControlGroup.
The failed unit/XML are preserved; no reset or overwrite is performed.
Failed native XML: 342,057 bytes, SHA256
`74d53b847b55a67ab217b13b0fc393145f9d2c9ebbb5e769db14e4b8fdb69bf3`.

Read-only diagnosis: the CPU-only allocation fixture incorrectly requested
`pinned=True` while CUDA_VISIBLE_DEVICES was empty. In the frozen CUDA-enabled
Warp build, `CpuPinnedAllocator.alloc()` calls `wp_alloc_pinned`, which returned
null with CUDA error 100 (no CUDA-capable device detected). Its error-reporting
branch then accessed missing `self.device`, masking allocation failure with
`AttributeError: 'CpuPinnedAllocator' object has no attribute 'device'`.

The fixture now uses ordinary pageable CPU staging. It still checks exact raw
bytes, actual ABI/allocation layouts and no kernel dispatch. The live child
continues requesting pinned staging only after authenticated CUDA admission and
initialization; CPU tests do not establish that GPU pinned-allocation path.
No installed library/source, driver or live workload is modified to hide the
failure. A new committed-source paired CPU run is required before CUDA.

### Repaired paired CPU check and first bounded GPU attempt

Source `7785922aea2e324b49a673e64892253d04a06eca`: both eleven-file CPU suites
passed all 382 tests, without failures, errors or skips. Mac: 17.42 seconds,
325,243-byte XML, SHA256
`dcff0a1867fd138c96925a5204846fda952c117482f1a50479ed697a06613b25`.
Native: 21.69 seconds, 325,247-byte XML, SHA256
`25f99708b8813553682ed06044c5f32119295f488c453baf0f505d92282f37be`.
Native CPU invocation `adc028fdd2064a4a86ee68e78a4ff939` completed successfully
with MainPID=0 and empty ControlGroup. Paired CPU evidence SHA256:
`b1b037a254903461f5b6eec6a19e7fd6a3d865b6d08056519c71705dee2546fb`.

Fresh shared admission at 18:45:32 Shanghai: both host counters reported
8,023 MiB used / 16,139 MiB free, driver utilization 0%, 31–32 C. Windows 3D
engine PID 51276 showed 12% activity; this was not exclusive/idle admission.
Protected services remained inactive; FilmBrain PIDs 521/298048 remained active
with zero restarts. Owner acquired the existing lease and declared deadline
1791457233, exact effective unit caps, complete source/runtime bindings and
the historical replay input before spawning its fresh child.

GPU unit `microduck-dense-solver-replay-7785922aea2e.service`, invocation
`81d9fd8851954b13a717a64b9284e912`, failed at executable construction:
`RuntimeError: Missing hash for kernel update_constraint_init_qfrc_constraint_dense
in module mujoco_warp._src.solver`.
The child reached authenticated CUDA initialization and fresh allocation, but
not CUBIN compilation/load, scratch restoration, target dispatch or numerical
comparison. MainPID=0, ExecMainStatus=1, no restarts and the owned cgroup absent
were independently checked. Owned-session cleanup reported no remaining
processes, no signals and no errors. GPU returned to 8,023 MiB used /
16,139 MiB free, utilization 0%, 32 C. No runtime/training/skill gate passed.

Retained raw directory: `artifacts/evaluations/dense-solver-replay-7785922aea2e`.
Four-leaf inventory SHA256:
`f5beff9ee24901fa523c36e4445e7c4fbc13ffed70156767bc4ef65983ad8397`.
Declaration: 261,477 bytes, SHA256
`44026add8560eab22ab369b14590c4a113cfe07df82663d11581c7f147976352`.
Child log: 5,670 bytes, SHA256
`2ecb5040f087bbe6e1ac1e7dafe010669d8fa493e10ac3b40c53de55d7c5f43d`.
Failure JSON: 367 bytes, SHA256
`22d81c5e2700d2137f7cc5446f285cbf60d7b7f60a6c00e6ad96e8a858a9d06a`.
All failed evidence/unit state is preserved; no reuse or overwrite is allowed.

Read-only CPU reproduction showed the fresh target's `kernel.hash` was None.
Calling frozen `module.get_module_hash(256)` published the target hash and
source-derived symbol with an empty executable cache. Installed Warp's
`ModuleHasher` explicitly publishes `kernel.hash`; `get_mangled_name()` refuses
it beforehand. The runner now resolves frontend hashes before the executable
snapshot, verifies unchanged options/empty caches and binds the snapshot to
the same hash/symbol. Its real CPU regression forbids module compile/load,
CUDA build/load and kernel launch during this operation. No installed helper
or library is patched, no row/input is repaired, and no GPU result is inferred
from the CPU hash reproduction. Another source-bound paired CPU run and fresh
declaration are required before any live retry.

### Hash-publication repair verification and handoff

Executable/source revision `ddf451ff72ca61b4c4ec88ac089a9d33c6485302`: eleven-file
suite passed all 382 tests on both hosts, including actual frontend hash
publication with compile/load/build/launch forbidden. Mac: 25.32 seconds,
325,243-byte XML, SHA256
`8a624c698098979502249d60c1fa3886783673a01161f20942aa67053f75ea30`.
WSL: 20.01 seconds, 325,247-byte XML, SHA256
`1e40e29e2be2f03ff7216ad1df30b03474d90b3f0938ca2e6e53466fb583ea64`.
WSL CPU unit invocation `e2b5b516b87c45c796dd7d0a63188d68` returned success,
MainPID=0, NRestarts=0 and empty ControlGroup. Both XMLs are retained locally;
the native XML is also durable on WSL. No full-repository test suite was run.

At 18:55:50 Shanghai, whole frozen runtime/library/source/tool pins were
reverified unchanged. Windows and WSL still reported 8,023 MiB used /
16,139 MiB free, utilization 0%, 31–32 C; Windows 3D PID 51276 showed 10%
activity. Both protected services were inactive, both FilmBrain PIDs and zero
restart counts unchanged, and no Duck service was running. The four-leaf failed
GPU evidence was copied to Mac and authenticated in full against its external
inventory (299,981 total bytes). The failed unit was not reset or reused.

This handoff is a documentation-only descendant of the tested repair revision.
It does not claim that the repaired child has completed CUDA compilation,
explicit load, scratch restoration or target dispatch. The next gate is a new
exact-source CPU prerequisite receipt and fresh shared-capacity/lease/deadline
declaration, followed by one bounded replay and independent unit-retirement /
artifact reception. Training, full-window qualification and physical motion
remain prohibited by this diagnostic evidence alone.
