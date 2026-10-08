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
