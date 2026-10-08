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

Runner/receiver assembly has passed paired CPU checks, the original bounded
native replay, and a separate versioned destination-overwrite control. The v2
receiver authenticated all four packet banks, verified identical inputs across
arms, and found all 1,280 active outputs changed from the canary and matched the
independent reference. Native qualification and training remain unclaimed;
the append-only closeouts below distinguish each actual stage.

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

### Repaired hash preparation: actual compile/load, constructor refusal

Source `bb66741273a2c6a99901af3b74a94f7bb59ebcf1`: both eleven-file suites
passed 382 tests (Mac 19.01 s; WSL 19.62 s), without failures/errors/skips.
Mac XML: 325,243 bytes, SHA256
`9adf6d0f81a2a481a3b3ed0d372f1f27ce3fdae35dac1f209bfc102c180e6c32`.
WSL XML: 325,247 bytes, SHA256
`740c2a44ebfc860240ecfa34e9e0ca119a2495b76730eb3105e3f2cdf652ee1c`.
Paired receipt SHA256
`442d0ad0b5eb45a87a42f692c035fba104202575d10f8ea421f4bf75d0827f43`;
CPU unit invocation `a00d326c156640f7b452a58898a7a98e` retired successfully.

GPU unit `microduck-dense-solver-replay-bb66741273a2.service`, invocation
`c8280e415073412b976b9e5b46099816`, reached fresh sm120 compilation, selected
offline disassembly, explicit load and scratch restoration. It then failed at
the dispatch-guard constructor, before any target launch:
`TypeError: DenseSolverDispatchGuard.__init__() takes 1 positional argument
but 3 positional arguments (and 6 keyword-only arguments) were given`.
Read-only signature inspection found both dispatch-guard and packet-capture
constructors are keyword-only; the runner incorrectly called both positionally.
Both calls now use named arguments. Two CPU regression cases bind the actual
runner AST calls against the real inert constructor signatures, not permissive
mock signatures. Focused runner tests: 38 passed; paired current-source CPU
tests and a fresh declaration remain mandatory before another live attempt.

Retained raw directory `artifacts/evaluations/dense-solver-replay-bb66741273a2`
has 12 authenticated leaves / 2,727,832 bytes on WSL and Mac. External inventory
SHA256 `db150c571870c3c8703e877c7f559ce8ec682f497063ccaa60877e344b76ac51`;
external closeout SHA256
`0d3c8c2c33f91a3407712e3f4f4c6d4747efed915c1398ce7b146369f8cd6b46`.
Retained CUBIN: 524,128 bytes, SHA256
`7fc1fa79f7796fd1c2c199c485d03545d6eb5f9818e39aa7598e53cb99e175dc`;
target SASS: 364,943 bytes, SHA256
`66b87b99403f44925e3f07f66dd07ccd49830ce791147077f893ae93ecf6fec6`.
These files are forensic evidence, not a completed replay receipt.

External closure at 20:08:39 Shanghai confirmed MainPID=0, ExecMainStatus=1,
NRestarts=0, empty ControlGroup and absent exact kernel cgroup. No Duck unit
remained running. Whole source/runtime pins and FilmBrain PIDs 521/298048 were
unchanged, protected services inactive. Both GPU views returned 8,022 MiB used /
16,140 MiB free, utilization 0%, 33 C; Windows 3D activity remained present.
No numerical replay, training, skill, runtime-cause or physical gate passed.

### Constructor repair CPU check and stale admission refusal

Source `e2f05d69aae2504fa22971f4965108b871b3de7e`: all 384 focused tests
passed on both hosts (Mac 21.75 s; WSL 18.09 s), with no failures/errors/skips.
Mac XML: 325,587 bytes, SHA256
`e49dc482129d4dde8ebbae6196a71b1bb1990ed6d2dfdae77587ecb45387b493`.
WSL XML: 325,591 bytes, SHA256
`54a54233e9706123c9e7565b94a36e0c154f6871dce0c4214a17e4c24f953d8c`.
Paired receipt SHA256
`6721e8458678770561adc3981f3bb90a98309b69900b618365990685a5d2d8c3`;
CPU invocation `5b17158300c34e9fa82ed1b7e49707d3` completed successfully.

Unit `microduck-dense-solver-replay-e2f05d69aae2.service`, invocation
`1aeca57726824cfaa0e83a0e85f1719d`, refused at the first owner deadline guard
at 20:18:01 Shanghai. The earlier preflight's deadline `1791462429` no longer
left the required 660 seconds at execution. Exact error:
`ValueError: fresh bounded wall-clock deadline and full reserve`.
No declaration, output directory, lease acquisition or GPU child was created;
MainPID=0, ExecMainStatus=1, empty ControlGroup were observed. This is an
admission refusal, not a CUDA or constructor failure. The failed unit remains
preserved. The next distinct-source launch must calculate its deadline on the
native host immediately before systemd-run, without weakening the guard.

### Completed guarded replay and independently repeated receiver

Exact source `4579312b269439a6784cf0b9577398131b6af777`, tree
`8ff273b359ca4021d6c7a11cfc1bbb9935d99fbe`: all 384 focused tests passed on
both hosts (Mac 19.43 s; WSL 16.93 s), without failures/errors/skips.
Mac XML: 325,587 bytes, SHA256
`c6000edc7fd3db61694909b73901b59d8bd2bc3c7258fae9db2003b920e185d8`.
WSL XML: 325,591 bytes, SHA256
`d7ff9e8eeda8ae414774238e403a575d1656b58dbe83175732985fa5804fba2f`.
Paired CPU receipt SHA256
`7c9ba43d4e9c330c73c9cc4350c50d80d183acb0d9e140bdc4d83fbd6f230790`;
CPU unit invocation `4a3736c7546b4fa0a90eca13ea4fcb70` retired successfully.

GPU unit `microduck-dense-solver-replay-4579312b2694.service`, invocation
`3a02c18c10fb4e538671d5a3a08d7c51`, launched at 20:25:57 Shanghai using the
native-calculated fresh deadline `1791463257`. It completed with Result=success,
ExecMainStatus=0, MainPID=0, no restarts, empty ControlGroup and the exact kernel
cgroup absent. The owner observed child PID 121252/birth 206980092 exit and
checked only itself remained before releasing its lease. The external closeout
confirmed no running Duck unit; it does not rely on the owner's retirement claim.

Retained raw directory `artifacts/evaluations/dense-solver-replay-4579312b2694`:
25 whole authenticated leaves / 8,788,864 bytes, copied intact to Mac. Inventory
SHA256 `23cd2bc43a908dc81afad7a11a9ece4e1e8f3dde069c7c43adfdaa1975f37c19`.
External closeout SHA256
`3847a35563c6c658de5a5a4c43150c91c5332dd03214e1b4d22444e6a59bbb3d`.
The independent CPU receiver was run on both WSL and Mac against the complete
inventory before record decoding. Both produced identical canonical decision
bytes, SHA256
`418bea0953bb4923c5c98bdf059dba7d8a8dbc40a856ab9c832c892a2cd4b461`:
`authenticated-one-launch-numerical-replay-only`.

All 64 worlds were active (`done=false`) with 46 active EFC rows, and all 20 DOFs
per world were compared. Zero mismatches; maximum absolute error
`4.789326339960098e-7` under the predeclared abs/rel `2e-5` diagnostic comparison.
Maximum relative error was 1.0 for near-zero reference components; absolute
tolerance dominates these cases. No tolerance or serialized inputs were changed.
Entire input banks were unchanged. Both complete 2,757,952-byte packet banks
have the same SHA256
`4b38bda92d03ffc852989cf943374b9185799ce626342d6aef3e0ce0a641d0fe`.
Thus the restored output already agreed: this run does not demonstrate an
observable output overwrite, distinguish a hypothetical no-op numerically,
diagnose the historical initialization rejection or qualify a simulation tick.

Retained CUBIN: 524,120 bytes, SHA256
`0cd22da97ba4060012c4f36bb8cce7177ddb7dcaedffa5b6d929af99b8c464ec`.
Selected target SASS: 364,943 bytes, SHA256
`66b87b99403f44925e3f07f66dd07ccd49830ce791147077f893ae93ecf6fec6`.
Receipt SHA256
`4b0ffa52ca4d65089b2e5280b2e6363b2a507706aee6b7f5b9be6a35e185f0b3`;
supervision SHA256
`5382ba776eebe1fc54de77ba01672fe1f68e591a89d6af377bb67dbec415f753`.
Compile/load/object/stream/dispatch bindings passed; driver-consumed machine
code was not observed. All numerical/native/full-window/runtime-cause/training/
physical qualification flags remain false. No policy, optimizer, physics tick,
MP4, raw perception or physical motion was started. No full-repository suite ran.

At 20:27:14 Shanghai, whole runtime/source pins and FilmBrain PIDs 521/298048
remained unchanged, protected services inactive. Both GPU views returned
8,022 MiB used / 16,140 MiB free, utilization 0%, 32–33 C. Windows 3D activity
remained present; this was shared-capacity admission, not exclusive idleness.

### Next diagnostic gate, not a training promotion

The next bounded experiment should separately predeclare an output-overwrite
control: retain this exact historical replay as its unchanged reference arm,
retain every input bank byte, and use a documented finite non-reference canary
only for the destination `qfrc_constraint` in a distinct control arm. Authenticate
the canary before dispatch, require all active output DOFs to overwrite it and
match the independent reference, preserve done-row bits, and require no-op /
partial-write / input-tampering negative fixtures to fail the pure receiver.
This changes the scratch-output contract and must not be smuggled into the
completed protocol. It needs a reviewed declaration and paired CPU tests before
any fresh capped GPU launch. Even a passing control would not establish full
solver initialization, simulation stability, policy training or a learned skill.

### Predeclared two-arm destination-overwrite control

Source baseline: `dfb8339c4c47f809b6e7735e6e582939e8c36dd7`. Opt-in flag:
`--output-overwrite-control`. Probe protocol
`microduck-dense-solver-replay-overwrite-probe-oct8-v2`; receiver protocol
`microduck-dense-solver-replay-overwrite-receiver-oct8-v2`. Existing v1 decoding
and historical artifacts remain unchanged; v1 cannot masquerade as a control.
Source scope is still the same five reviewed paths since `e6cf84d...`.

One fresh child prepares the same frozen executable/device/context/stream and
allocations, then runs two sequential arms under the existing unit, child,
lease, workload, capacity, cache and deadline bounds. Arm one restores the exact
24-field historical bank and executes the unchanged caller once as before.
Arm two restores that complete bank again, then stages **only** destination
`data.qfrc_constraint` from an owned pinned CPU buffer, using the held copy
entry and explicit held stream. The copy is synchronized and its whole CPU
staging bytes rechecked before constructing the second one-shot observer.
No solver initialization, physics tick, input-row rewrite or installed edit occurs.

Canary recipe: little-endian float32, shape `[64,20]`, flat index `i=0..1279`:
`(+1 if i is even else -1) * (4096 + i % 32)`. Exactly 5,120 finite bytes,
SHA256 `4c52c8fc33c766f4cc6ffb68bc64d9ea076b0420cd4f79fe83ca12723fb623a9`.
The declaration pins this recipe, destination and exactly two target calls.
Reference `packet-before.bin`/`packet-after.bin` and distinct
`control-packet-before.bin`/`control-packet-after.bin` are retained in full;
the extra copy, scratch and guard receipts are bound to the v2 envelope.
Total four packet banks: 11,031,808 bytes, within the existing bounded inventory.

The whole inventory is authenticated before decoding. Both one-shot guard
records must bind the same CUBIN/context/device, source caller, five array
identities/pointers/layouts and stream. All four `nefc/J/force/done` input banks
must be byte-identical across arms and phases, including inactive padding.
The captured control-before output must equal the exact canary. Both arms must
match the unchanged ascending-row float32 reference. Each active DOF must
change bits from its canary preimage, with reference/canary separation greater
than sixteen times the existing combined abs/rel diagnostic tolerance. Done
worlds retain their canary output bits; an all-done matrix is rejected. No-op,
partial write, nonfinite/wrong output, canary/manifest tampering, cross-arm or
post-target input changes, altered done-row bits, substituted stream/destination,
aliased packet roles and stale inventory must fail focused CPU fixtures.

Acceptance is only `authenticated-two-arm-output-overwrite-diagnostic-only`.
All numerical/native/full-window/runtime-cause/training/physical qualification
flags remain false. It demonstrates destination overwrite on the retained
case, not a general GPU fix, stable simulation, policy or physical skill.
Paired exact-committed-source CPU evidence and fresh shared admission are required
before live launch. Failed artifacts/units are preserved; no automatic retry.

### Completed two-arm overwrite control and retained closeout

Exact implementation/source `96179731c9f408a466d20931c82709114647c59b`, tree
`79c676e61a3106026b3c345659d19a3aac06b95a`: all eleven focused files passed
412 tests on Mac (19.01 s) and WSL (17.56 s), no failures/errors/skips.
Mac XML: 329,915 bytes, SHA256
`af1cb6a742b4042b61d606390e480a8d35158f8f59210636d48d1085618f426e`.
WSL XML: 329,919 bytes, SHA256
`4b2b705ee8c658b7d6b530fba58106d6635855d7bfa9fe6c37f1fd2ed7cf73d2`.
Paired receipt SHA256
`d8a61ca22ae187d0c425d8997f0432508925ddd394985b58bcece27416e2bac2`;
CPU unit invocation `899828b2eb5d48648fc2a384e42f12c3` completed successfully.
Independent read-only review prompted the empty-coverage/canary-collision
negative fixtures, distinct v2 result protocol and mirrored copy-pointer check.
The historical v1 full receiver result remained byte-identical at its retained
SHA256 `418bea0953bb4923c5c98bdf059dba7d8a8dbc40a856ab9c832c892a2cd4b461`.

Unit `microduck-dense-solver-replay-96179731c9f4.service`, invocation
`a3ab6acab89f4903b373b4445eed510e`, launched at 21:23:57 Shanghai with native
deadline `1791466737` and `--output-overwrite-control`. It returned success with
ExecMainStatus=0, MainPID=0, NRestarts=0 and empty ControlGroup. Independent
closeout at 21:25:34 confirmed the exact kernel cgroup absent and no running
Duck unit. The owner observed child PID 135420/birth 207328088 retire, then
checked its cgroup contained only itself before closing the held lease.

Raw directory `artifacts/evaluations/dense-solver-replay-96179731c9f4` contains
27 whole authenticated leaves / 14,312,202 bytes, retained on WSL and copied
intact to Mac. Inventory SHA256
`87f6f7757beb73354fbb31d0d1ca92b580700c74303e0c0c40d43d84d4cfdf0f`;
external closeout SHA256
`8119103aa0657da223cc8fe207ea4f562912397088b4dafb6cf53787b0383f49`.
The receiver was independently repeated on Mac after whole inventory
authentication and produced identical canonical bytes to WSL, SHA256
`8aac2fce068851e2b9e64d6ed822519c4cb35f3a37d2361e5c5a0156a4c421fb`:
`authenticated-two-arm-output-overwrite-diagnostic-only`.

All 64 worlds were active. The captured control-before output exactly matched
the predeclared canary; all 1,280 active components changed from that preimage.
Reference and control had zero numerical mismatches, each with maximum absolute
error `4.789326339960098e-7`. All four complete input prefixes were byte-identical;
both arms' guards bound the same arrays, held stream and explicit loaded CUBIN.
Native done-row handling was not exercised because all done flags were false;
its preservation and all-done rejection were checked only by CPU fixtures.

Reference before/after and control-after complete packet SHA256:
`4b38bda92d03ffc852989cf943374b9185799ce626342d6aef3e0ce0a641d0fe`.
Control-before SHA256:
`ae96d9a4ca57ee8abccfc81c305b32644a27075994ad26c6d6ce505251c4bbf8`.
Each complete packet is 2,757,952 bytes. CUBIN: 524,128 bytes, SHA256
`28e5788f148d35aec6cc9b2ffcad20beefeb0a66c101b150162e0894b9b43f3a`.
Selected target SASS SHA256
`66b87b99403f44925e3f07f66dd07ccd49830ce791147077f893ae93ecf6fec6`.
Receipt SHA256
`70cf1039bb700768cace37dffb8b915296a053a3f20cbda90602e3345c0e0dc2`;
supervision SHA256
`ce021fe0e1023b719aa259c9a458edfe6ac2957b51fd4976f6b496be31da8f29`.

Whole frozen runtime/source pins remained unchanged. FilmBrain PIDs 521/298048
stayed active with zero restarts; protected services remained inactive. Final
views: 8,018 MiB used / 16,144 MiB free, utilization 0%, 32–33 C, Windows 3D
activity still present. No policy/optimizer, physics tick, video, installed
runtime/driver change, raw perception or physical motion occurred. The full
repository test suite was not run. All qualification and training flags remain
false: this excludes a numerical no-op for this destination on this retained
case, but does not prove driver-consumed bytes or a general solver/runtime fix.

Next gate: predeclare a bounded check of the remaining caller/initialization
outputs against retained CPU references, including setup and Gauss-cost
boundaries, before attempting a full simulation tick or policy training.
Do not infer broader stability from the single dense target's overwrite result.
