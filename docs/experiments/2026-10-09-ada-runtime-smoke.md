# Ada post-reboot runtime smoke

Feature branch `feat/athletics-obstacle-curriculum`, base
`9111cbe2e7feed684e8088b8b99b46c8f5e2f036`. This is a new, independent diagnostic
module and test file. No old source fence, driver profile, numerical gate,
Blackwell collector, solver, observation, reward, learner or checkpoint changes.

## Authority and fixed question

The owner authorized reboot of 100.100, rebooted it manually, then approved
temporarily stopping only `recomo-ai-mission-vllm.service` to recover capacity.
After that stop, its user service was inactive/MainPID 0. Grounding DINO stayed
running at PID 1592, using 946 MiB, and all other workloads were left alone.
This is not authority to disable services, stop Grounding DINO, restore unrelated
services, perform physical motion or launch an unbounded training job.

Post-reboot kernel and installed driver now both report 595.91.07. The actual
GPU is **RTX 4090 Laptop**, 16376 MiB, compute capability 8.9, UUID
`GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`. It is not a desktop 24 GiB 4090.
At the opening stopped-vLLM sample it had 15232 MiB free, 961 MiB used, 0 percent
utilization and 51 C. Capacity is point-in-time, not an exclusive reservation.

Question: does this frozen environment execute tiny, deterministic Torch CUDA
and Warp kernels on the actual Ada GPU after reboot? This is **not** a test of
Duck simulation, solver numerical correctness, training, hopping or avoidance.
The [entered solver collector](2026-10-09-entered-gradient-integration.md) stays
closed and remains Blackwell-specific. We do not use, modify or bypass it.

## Predeclared protocol

One stdlib-only CPU owner, with CUDA hidden, runs in the clean exact-source
checkout `/home/converge/work/microduck_rl-athletics-obstacle-curriculum` and its
existing `.venv`. Bind source SHA/tree/module bytes/lock bytes, machine ID,
interpreter/prefix and frozen versions: Torch 2.9.1, Warp 1.12.0, MuJoCo 3.10.0,
MuJoCo-Warp 3.8.1 and mjlab 1.3.0. No package sync, upgrade or driver work.
The old remote checkout must fast-forward to the reviewed source before running.

Acquire the existing, owned, regular, empty Duck advisory lock without replacing
or writing it; pass the same FD to the fresh child. Only this cooperating Duck
job is serialized. Foreign processes do not participate in that lock.
Require both AI mission GPU services inactive in system and user scopes. Retain
the original foreign compute PID/name list and require it unchanged during and
after the child. This is sampled preservation, not proof of uninterrupted foreign
service health or attribution of all GPU allocations.

Retain before/during/after NVIDIA telemetry. Require UUID/driver/SM unchanged,
at least 10240 MiB free, at most 12288 MiB used, aggregate growth at most 2048 MiB,
temperature below 65 C and utilization at most 85 percent. Non-simultaneous
queries cannot eliminate races or reserve capacity. Failures kill only the owned
child, retain evidence and require diagnosis before any retry.

The exec user service has a 180-second runtime maximum, 6 GiB system RAM,
200 percent CPU, 64 tasks, Nice 10, 16 MiB per-file limit, no core/restart,
control-group kill and a 10-second stop timeout. Owner deadline is 150 seconds,
child deadline 120 seconds with an independent watchdog and terminate/kill/reap
cleanup. All four CPU thread limits are 1 before imports. The exact service
properties and invocation are retained; external closeout checks retirement.

Create a fresh source-specific evidence directory and separate private Torch,
Warp, CUDA and XDG caches. Never overwrite a prior attempt or shared cache.
The child exposes only CUDA device 0; bind the actual Torch UUID bytes, model,
capability 8.9 and CUDA 12.8 build, plus Warp device ordinal 0 and architecture 89.
Compute `2*i+1` for every float32 index `i=0..31`, separately with Torch and one
Warp kernel. Synchronize, retain every result and independently compare to exact
CPU arithmetic. No autograd, optimizer, simulator, robot asset, policy or action.
The 64 MiB Torch allocator limit is not a hard total-VRAM cap: Warp/driver/context
allocations are instead constrained by sampled aggregate telemetry and watchdog.

The only positive decision is `ada-torch-warp-smoke-complete-not-training`.
Every exclusive-GPU, runtime-origin, solver, simulator, training, learned-skill
and physical flag stays false. A successful toy kernel does not identify the
solver's loaded machine code or resolve its historical numerical rejection.

## Verification and next gate

Focused CPU tests check inert imports, exact identity/resources, all 32 elements,
strict false flags, private evidence, lease and owned-child cleanup. A separate
CUDA-hidden CPU Warp kernel check verifies the actual toy kernel source compiles;
that is not GPU evidence. Independent review precedes source freeze and execution.

After a successful capped GPU run, retain byte hashes, source/installed-version
checks and external same-invocation retirement, with no remaining Duck process
and Grounding DINO still present. Then choose a separately reviewed Ada simulator
diagnostic for the unresolved runtime/numerical question. Do not promote the toy
smoke into a solver gate or restart PPO from it.

Pre-freeze independent Luna review found no remaining blocker for this limited
health-check scope. Owner review corrected an impossible used-plus-free CPU
fixture and changed source comparison to literal bytes, not stripped text.
Four-file CPU regression (new diagnostic plus unchanged shared smoke, idle gate
and execution profile) passed; the final post-freeze counts are retained with
the execution evidence. Separate CUDA-hidden Mac Warp CPU compilation produced
all 32 exact values. None of these checks is GPU or simulator qualification.

## First attempt and CPU-only compiler diagnosis

Execution source `cd6a73dac6a0b0f6660dd53b03986311e0ef28ca` passed **135 CPU
tests** on both Mac and 100.100. Its GPU unit invocation
`0931a758209c4448b70b9baf9388d7fb` failed with exit 1 after 1.73 seconds.
Torch reached and completed the synchronized CUDA calculation, but the overall
attempt did not produce an accepted child receipt. Warp initialized on sm89,
then NVRTC compilation failed **before module loading or kernel execution**:

`NVRTC_ERROR_COMPILATION (6)`; `warp/native/mat.h(1852): catastrophic error:
unable to obtain mapped memory`.

This is not a new driver mismatch, a robot-physics result or evidence of a CUDA
numerical failure. The failed unit retired with MainPID 0, exit 1, no restarts
and empty ControlGroup. Only Grounding DINO PID 1592 remained in compute telemetry.
The failed report/log/generated CUDA source remain untouched in the original
`artifacts/evaluations/ada-runtime-smoke-cd6a73dac6a0/` directory.

Read-only source review found Warp 1.12.0 defaults to automatic precompiled
headers (PCH). NVIDIA documents this compiler-state cache and its memory/file
behavior in the [NVRTC 12.9 guide](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html).
Two separate, 60-second-capped CPU compilation controls used the **same retained
2294-byte CUDA source**, architecture 89, frozen Warp, CUDA hidden, no device
load/launch, the same 6 GiB RAM/200 percent CPU/64 tasks/16 MiB file bound and
fresh caches. Hidden CUDA initialization listed CPU only; its expected CUDA
error 100 is not an exposed-GPU driver fault. With PCH on, the exact mapped-memory
error reproduced; the syscall trace showed repeated 64 KiB mappings from a
scratch FD through offset `0xfe6000`, followed by **SIGXFSZ** (file-size ceiling).
With PCH off, compilation completed in 0.40 seconds and wrote a 20816-byte CUBIN.
This isolates the PCH/resource-bound interaction without enlarging any limit.
No installed source, package, driver, service, solver or GPU gate was changed.

| CPU control evidence | SHA256 |
| --- | --- |
| identical input CUDA source | `299ba7653f66bb5abad4daf739367b5493524d1485278da03b916c5c54a93c91` |
| PCH-off generated CUBIN | `041a85e85bf0170a3a625767c6afeaddac1198d75b5814a054f200886b44480d` |

## Predeclared second revision

Protocol `microduck-ada-runtime-smoke-oct9-v2-no-pch` changes **only this
diagnostic child's compiler configuration** to `use_precompiled_headers=False`.
The strict child receipt records that false value, and the child disables the
driver compiler cache before imports. This does not disable or replace the
independent Warp private kernel cache. All deadlines, resource bounds, numerical
inputs, foreign-process guards and false qualification flags remain unchanged.
A new clean source commit creates a new absent evidence directory and unit;
never overwrite or relabel the failed first attempt. One capped GPU attempt is
permitted only after focused tests, reviewed diagnosis and source freeze.

## Retained accepted health-check closeout

Execution source `78e7ceb4b77a5a2ad1a74b5bfafa0da76c249330` passed **137 cases
across four CPU test files on each host**, zero failures/errors/skips. Mac took
6.39 seconds pytest time; the capped Linux CPU unit took 5.00 seconds pytest
time / 6.48 seconds service runtime. Independent post-exit reception matched
the actual testcase multiset, not merely the aggregate counts. A later doc-only
commit does not relabel these checks or the GPU execution source.

The sole revised GPU attempt completed in **1.76 seconds**, with all 32 Torch
and all 32 freshly compiled Warp float32 outputs exactly matching independent
CPU arithmetic. Torch peak allocated bytes were 1536, not a total GPU footprint.
Six sampled telemetry records showed peak aggregate used memory 1272 MiB,
minimum free memory 14922 MiB and peak temperature 47 C. Sampling cannot prove
the true instantaneous peak or attribute all growth to Duck.

Independent external closeout checked the exact source, source/module bytes,
all report file hashes and strict child fields; then matched GPU unit invocation
`0b8a76dee08545da8847b2c154299404` with MainPID 0, exit 0, Result success,
zero restarts, empty ControlGroup and absent original cgroup. Active/exited is
a retained completed unit, not a running job. Only Grounding DINO PID 1592 remained
in compute telemetry; both AI mission services stayed inactive in both scopes.
At closeout, memory returned to 961 MiB used / 15232 MiB free, 45 C and 0 percent
utilization. No driver, installed package, simulator, optimizer or policy changed.

Seven complete GPU evidence files, both original/revised test XML pairs, the
untouched failed attempt, exact retained generated CUDA source, both CPU compiler
controls/traces and helper bytes are retained on **both Mac and 100.100** under
`artifacts/tools/ada-runtime-smoke/`. Native original GPU directories remain in
`artifacts/evaluations/`. These ignored evidence assets are not uploaded into
Git history. The independent receiver reproduced all seven GPU byte bindings
after transfer to the Mac. Historical evidence and caches are not merged into
the new run or promoted into solver machine-code proof.

| Retained evidence | SHA256 |
| --- | --- |
| Mac 137-case XML | `c243a83711f4d447166b0012ffded47005070a63989af738d7084b6102aa127b` |
| Linux 137-case XML | `6b901b7cb839de3a1f00601976ff5ac8c492f7ec90822680637dfb544f107f36` |
| independent post-exit closeout | `9b2f0bdeeb5575f155762e2d2177fd117bbf551514c32122a978ec0c6a980f14` |
| accepted GPU owner report | `af457091e415827b8635fa0a313c181eec870e67c6decc5ac99d6e6605e9ca2b` |
| accepted GPU child | `8e4e6daed3a14ab2fdd18479a53715400e240eba240c694ae792302cf514bf3b` |
| untouched failed GPU report | `82f8d41e6d04145bffc5e8bd66db11661fcf1db42cb9573b1e480960b8cf05cc` |
| PCH-on CPU result | `8136fb3248dc0f8ea78a1574b51fcb30045d54a1220869d403993f0c2cbdf64c` |
| PCH-off CPU result | `a41f84e2c7775951aaea9072fb15244a6d76a711bb1348e54a3818ab44008db8` |
| PCH-on syscall trace | `246d2eb67e4fa6235792874bfb7d9d611f47c4eb90549263e27a529756c0a128` |
| PCH-off syscall trace | `52f78b7abb20f4a4fb4f9d2f5ba37f1cfb872e4618659104c4a12a9365fc3dbc` |
| CPU compiler-control helper | `6ea18250fd5c355484591c364c6a6d89498ae574f18a981dc1cb17feb4623586` |
| post-exit helper | `47362b222ef84b9ddf83bebe9c21847cdf79a8e118de1b743558e5cda0340aa1` |

Luna's independent read-only review accepted the matched CPU diagnosis and
limited second attempt; the owner integrated and verified the evidence. The
only accepted claim is **Ada Torch/Warp toy-kernel runtime health**. Runtime-origin,
solver, simulator, training, skill and physical qualification remain false.
The previous solver gates stay closed, and no new Duck behavior or video is
claimed. Next work is a separately declared Ada-compatible simulator/numerical
diagnostic; this health receipt is not a PPO launch permit. vLLM is temporarily
stopped, not disabled; it was not restarted during this work.
