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
