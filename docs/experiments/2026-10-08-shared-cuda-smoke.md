# Shared Windows/WSL CUDA capacity smoke, October 8

Predeclared after the user requested the next actions toward a small shared-GPU
run. This distinct protocol does not relax the existing idle or simulator
qualification gates. The current experimental simulator's numerical rejection
remains unresolved. No policy, PPO update, robot physics, raw perception,
checkpoint, video, or physical motion is in this test.

## Fixed experiment

One CPU-only owner and one fresh Torch-only CUDA child in the clean feature
worktree `/home/yanbo/work/microduck_rl-com-entry-20261006`, using the existing
frozen environment. Pin source, Windows/WSL machine, physical GPU UUID, driver
595.95 and package versions. Acquire the existing FilmBrain advisory lock
nonblocking without creating, replacing, writing or unlinking its file. This
lease excludes cooperating WSL jobs; it cannot reserve uncoordinated Windows
applications. Preserve both FilmBrain service PIDs/restarts and inactive
protected services, and stop only the owned child on a guard failure.

The child runs eight exact float32 products of fixed 4,194,304-element tensors
(1.25 times 2.5 equals 3.125), synchronizes and checks every value, then pauses
one second between rounds. Bind the CUDA child's literal 16-byte UUID to the
physical NVIDIA UUID, sm120 capability, name and frozen Torch CUDA 12.8 runtime.
Autograd is off; there is no optimizer. Use fresh
private cache paths without changing shared caches or installed packages.

Bound the owner service to 120 seconds, owner to 90 seconds, child to 60 seconds,
4 GiB system RAM, one CPU quota, Nice 10, 64 tasks and 4 MiB per output file.
An independent timer kills only the owned child at the earlier child/owner
deadline, even if telemetry blocks. Read-only subprocess timeouts are clipped
to remaining time. The service wall cap remains a separate backstop.
Limit the Torch allocator to 512 MiB and require its observed peak reservation
at most 128 MiB. **This is not a hard total-device allocation limit:** driver
contexts and non-Torch memory are outside that allocator cap.

Before, throughout and after execution, query Windows NVIDIA/CIM engines and
WSL NVIDIA telemetry. Require each GPU view to retain at least 10,240 MiB free,
at most 12,288 MiB used, below 65 C, and at most 85% utilization. Require
aggregate growth over its own baseline at most 2,048 MiB. Growth is not
attributed to Duck; Windows changes may conservatively abort this test. Any
telemetry/identity loss, guard breach or deadline stops only our child. Queries
are sequential and cannot eliminate races or prove desktop responsiveness.

Current observation: 7,422 MiB used, 16,740 MiB free, 32 C, 0% instantaneous
NVIDIA utilization. Windows previously showed intermittent `ugraf` 3D work.
That is sufficient apparent headroom for this bounded diagnostic, **not an idle
lease or long-running training reservation**.

## Retention and interpretation

The new [diagnostic module](../../src/mjlab_microduck/shared_cuda_smoke.py) writes
fresh launch, child log/receipt and owner report under a source-specific ignored
evaluation directory, retaining both telemetry views and whole output hashes.
Focused CPU tests and independent read-only review are required before a source-
bound native run. Preserve a failure and diagnose it before any fix or retry.

The only successful decision is
`shared-cuda-tensor-smoke-complete-not-training`. It means a bounded CUDA context
and deterministic tensor operations completed during observed shared load.
It does not demonstrate Duck simulation throughput, usable PPO capacity, the
solver's compiled arithmetic, repeatability, any learned skill or robot safety.
All five exclusive/simulator/training/skill/physical flags stay false. A later
simulator diagnostic needs its own reviewed declaration and numerical gates;
no existing learner may consume this report as qualification.

## Review corrections before source freeze

Independent read-only review found missing post-exit telemetry, delayed deadline
checks, an interpreter binding gap and an incomplete CUDA-runtime receipt.
Owner correction adds a separate guarded post-exit sample, independent owned-
child watchdog and deadline-aware queries, canonical `sys.prefix`/interpreter
bindings, and a closed runtime/device receipt. CPU-mocked supervisor tests cover
admission refusal, telemetry loss, resource/deadline failure, stubborn-child
termination, post-exit rejection and honest successful closeout. No old launch
guard or numerical acceptance rule was edited.

The frozen environment exposes Torch CUDA 12.8 and the UUID bytes property;
the [PyTorch 2.9.1 binding source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/cuda/Module.cpp#L962-L1008)
confirms the 16-byte UUID access. Bind those bytes directly rather than guessing
the string-format prefix. Ruff is not installed in the owner's environment;
do not modify the frozen environment to obtain it.

Final precommit Mac checks: **76 passed**, CUDA hidden, across the new diagnostic,
unchanged idle gate and execution-profile tests. Python syntax, relative links
and whitespace checks passed. A broader collected profile test exposed a fixture
that incorrectly assumed Torch was absent from the shared pytest process; only
the mocked owner namespace was corrected. The actual fresh-owner CLI guard
remains strict. Independent final read-only review found no remaining
implementation blocker; it did not execute a GPU test. A same-source native CPU
check and source-bound capped run remain separate delivery steps.

## First native attempt: admission refusal, not CUDA failure

Executed source `74c3ca1044036082d8d89acd4e02109d33307355`, module SHA256
`e63816f65d75c66a8fb3e06c70221215a44fc40cc1d09df7c147944332a95d4d`.
The same 76 CPU checks passed on WSL in 8.84 s (zero failures/errors/skips).
GPU diagnostic service `microduck-shared-cuda-smoke-74c3ca104403.service`,
invocation `e36faf3b64474b4991e99bd12ce1da93`, owner PID 13874, ran
14:13:37–14:13:42 Shanghai and exited 1 with **`shared thermal guard`**.
The capacity check refused **before creating a CUDA child**, private caches,
launch receipt, tensor, simulator or learner. MainPID is zero, no restart.

The original report records elapsed 5.288552 s, null child exit, empty child
inventory and false admission flags. Its baseline was unfortunately checked
before appending it, so its telemetry list is empty. Do not retroactively insert
a later observation or claim exact launch-time temperature/utilization.

Independent post-failure observations at 14:15:13–15 showed 95–96% NVIDIA
utilization, 82–83 C and 5,588 MiB used / 18,574 MiB free. Windows CIM separately
reported `ugraf` PID 5744 at 71% on the retained `0x0001224A` 3D engine.
The CPU-only observer at 14:15:52 showed both GPU views cooled to 62 C and 0%
sampled utilization, with the same memory. This is intermittent Windows demand,
not a continuously reserved GPU. Plenty of VRAM did not imply spare compute.

Whole retained evidence is authenticated on Mac and WSL:

| File | SHA256 |
| --- | --- |
| Original failed report | `3e12f3e502e824e7a959ec53a3dddb8f3661152f79e8571427dd86af6067e683` |
| Original WSL CPU JUnit | `ba0af81a0078726f745a301ade6f7dd9a469e8fcc720f2e7f68d01add20686b4` |
| Original Mac CPU JUnit | `66bbd1c19c020e63c7e931fe348aed2aa7091771b37893ba949d3ca408db163d` |
| Diagnostic journal | `0912765a8430e98c2bf3709c36427411c3e68fcca6f9d85ce7a9d7476c774125` |
| Separate post-failure telemetry | `121dcc707671dd473eb3f2e5fa5336cb01d7bef4645dc9640a519610c4ab1bf0` |
| Service closeout | `39b4a4963eb1b88bd4a603ba9efe7a330c69fd10ca61d6eb32f12909f49e8cfc` |

The original failed directory contains only `report.json` under
`artifacts/evaluations/shared-cuda-smoke-74c3ca104403`; separate diagnosis lives
under `artifacts/tools/shared-cuda-failure-diagnosis-74c3ca104403`. Preserve all
bytes and the failed unit. FilmBrain remained at PIDs 521/298048 with zero
restarts; protected services stayed inactive in user/system scopes. The frozen
interpreter, environment alias and packages were rechecked unchanged.

## Evidence-only repair and conditional one-attempt retry

Read-only diagnosis preceded the repair: retain the baseline **before** applying
capacity guards; retain the failing phase, pre-run services and existing lease.
Set the post-exit phase before querying so telemetry loss is correctly labeled.
Independent review checked this narrow retention correction. No resource,
thermal, demand, runtime, physics, numerical or old idle guard was relaxed.
The extended focused Mac suite passed **77** checks, CUDA hidden.

Only after committing/pushing the corrected source and passing those 77 native
CPU checks may one fresh source-specific diagnostic attempt run. Require two
new Windows/WSL observations below 50 C and at most 20% sampled NVIDIA demand
in addition to the unchanged headroom guards before launch. This stricter retry
precheck is not an idle claim. Do not loop through Windows work, extend the wall
cap or modify consumers to force success. Stop after that attempt and retain its
decision; even success does not admit a simulator or training.

### Corrected-source closeout: retry precheck also refused

Corrected source `13da386ee9b36ccedfeea6f9bf181fac2a2cc92f` was pushed and
installed by authenticated bundle fast-forward. Module SHA256
`6879f2af5ca6727c2b3a78cb0d9f0a0980979af0e9676b059f49ef7756063e15`.
Native CPU unit `microduck-shared-cuda-tests-13da386ee9b3.service`, invocation
`923059ba91af4031aadab720edd96cc8`, passed **77 checks in 5.73 s**, zero
failures/errors/skips. The new raw-baseline and phase tests are included.

The separately retained **CPU-only** retry precheck refused its first observation
at **14:24:26 Shanghai**: Windows and WSL each reported **84 C**, **5,738 MiB
used / 18,424 MiB free**; utilization snapshots were **87% Windows / 95% WSL**.
The Windows engine record showed `ugraf` PID 5744 at 70% 3D and 2% Copy on LUID
`0x0001224A`; an Intel-side instance was recorded separately, not summed.
Error remained `shared thermal guard`. The stricter two-cool-sample precheck did
not pass, so **no second GPU diagnostic service or CUDA child was launched**.
The first failed service remains preserved with MainPID zero. No timer extension,
memory/temperature relaxation, consumer stop or simulation/learner retry occurred.

`artifacts/tools/shared-cuda-smoke-cpu-13da386ee9b3` contains exactly these two
authenticated leaves on both hosts:

| File | SHA256 |
| --- | --- |
| Native 77-case JUnit | `85c44d1dee9e8a547fa4f23499ceb382dc0c5ce87ec32d9847b90f1fe22c3495` |
| Retry-precheck JSON | `1430ed8e64266c3894799ed43edcf5b46870d646322284e7a032bba27586a6df` |

Independent Mac replay checked whole hashes before parsing the XML/JSON, all 77
cases, the observed refusal, exact source and false flags. This is evidence
verification, not independent CUDA execution. FilmBrain PIDs/restarts and both
scopes of protected services remained unchanged. Current delivery is a reviewed,
CPU-tested capacity diagnostic with safely refused admission, **not a CUDA
success or a learned Duck skill**. Before another separately declared attempt,
verify cool/low-demand conditions and preserve all historical refusals; Duck
simulation/training still needs its separate unresolved numerical gate.
