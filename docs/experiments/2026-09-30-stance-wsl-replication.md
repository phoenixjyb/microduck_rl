# B1-N replication migration to 100.98

Simulation-only predeclaration, 2026-09-30. The user authorized resuming training
and testing on 100.98. This changes the **execution host**, not the learner or
the [declared replication](2026-09-17-stance-lean-replication.md). Cross-host
binary/numerical equivalence is not claimed. Results will be labeled Blackwell
replication evidence, not a pure same-host learner-seed causal comparison.

## Fixed target and coexistence

- Isolated checkout `/home/yanbo/work/microduck_rl-stance-replication-20260930`,
  branch `feat/athletics-obstacle-curriculum`. The September 2 checkout is preserved.
- Explicit profile `MICRODUCK_STANCE_PROFILE=wsl-10098-20260930`; no arbitrary
  host/driver/path overrides. Machine `7d6778c98cb345788b8c1a410f19ad35`, GPU
  `GPU-7d72b360-33bc-2cee-3ff4-a954474011b5`, driver `595.95`.
- Exact frozen lock, six package versions and reviewed installed dependency
  Python-tree hashes remain mandatory. This is selected runtime qualification,
  not complete binary equivalence. An offline cache miss may be reconciled by
  copying the existing environment with the identical lock, followed by frozen
  sync and exact installed-tree checks; the old environment is untouched.
  The build backend may need a small network fetch even when runtime packages
  are already installed. No dependency pin is changed to satisfy a cache miss.
- FilmBrain remains active. Duck takes its existing `wan-gpu.lock` via the same
  nonblocking `flock` protocol. No lock unlink, service stop, restart, or config
  change. An occupied lock means no launch. A simultaneous preview request may
  be refused by FilmBrain's existing guard; no concurrent GPU worker is allowed.
- Both system and user protected AI Mission services must remain inactive.
  Idle gate: two zero-utilization samples, no compute PID, temperature below
  75 C, memory below 1024 MiB (observed WSL desktop baseline ~642 MiB).
  Live guard: no foreign compute PID, temperature below 75 C, total GPU memory
  <=5120 MiB and at least 6144 MiB free. These observations and the cooperating
  lock are not hard hardware isolation. Only the owned Duck process group may
  be terminated on a failed guard.

## Gates before the first learner

1. Review/test the pending replication evaluator and campaign. All three seeds
   must authenticate the claimed training source. Missing/mismatched evidence
   stays `lean-replication-incomplete`, never a smaller denominator.
2. Exact-source Linux CPU regression with `CUDA_VISIBLE_DEVICES=''`.
3. `stance_wsl_qualification`: a fresh absolute window, 900-second CUDA child,
   960-second `KillMode=control-group` user service and 600-second closeout reserve.
   Retain the exact two-world integration/isolation/reset cases, the frozen-parent
   collection probe (64 worlds, 24 ticks; two warmups and eight measured updates),
   and CPU-only optimizer timing stand-in. No checkpoints are exported.
4. Re-derive timing from measured maxima, including qualification setup, using
   the existing 1.25 factor. Admit a bounded learner only if the resulting child
   and service caps fit **within** 1693/1753 seconds. A slower host is rejected;
   the historical caps are not extended. Bind the successful qualification
   report/hash into each WSL training launch. This qualifies training timing
   only; evaluation timing must be separately measured before full evaluation.

## Next jobs and decision

Start learner **577**, then **587**, then **593**, sequentially. Each uses the
same pinned parent, fresh optimizer, 64 worlds, 256 updates and 24 ticks/update;
all exports and tick/update receipts are durable. No forward graphs. Each job
gets a fresh 41-to-60-minute absolute window and existing 1693/1753-second caps.
Numerical training completion is not a stance pass.

Before full evaluation, qualify its full-length repeating case on this host;
do not describe the old 4090 timing probe as a Blackwell measurement. Evaluate
common checkpoints 64/128/192/255 at held-out seeds 541/547/557, 128 worlds,
five-second first attempts, unchanged scorer and 0.0873-rad tilt gate.
Disturbance recovery comes only after all three learner seeds have a verified
campaign decision. Football balance, hopping promotion, obstacle-policy changes,
video and physical motion are outside this migration chunk.

## Retained status

Prelaunch inspection: GPU idle at 30 C, approximately 23 GiB free; FilmBrain's
guard and Duck use the same owned regular lock inode. Frozen lock SHA256
`84dc822f2d75a4368fe70f09ca0a2dcc7dbc1d523ccb35455d374d02c4c7f88d`.
No completed learner or cross-host capability result is asserted by this document.

## October 1: completed timing diagnosis and separate WSL budget

The original prelaunch above **did not admit training**. At exact source
`ab60e84ce71cbfd27ac4b9e0a67146e459c59ae0`, the 900-second qualification child
completed successfully in 168.314 seconds and integration/isolation passed.
The service then correctly refused the old training budget; its nonzero exit
is a recorded timing rejection, not a CUDA/numerical failure. GPU temperature
peaked at 47 C; service peak RAM was 3.69 GiB. FilmBrain stayed active and the
protected services inactive. Exact-source focused CPU regression: 274 passed.

Retained report SHA256:
`c6c561ba227f3d969295c20f167eeef3a29dff3c510fe2dd2b8eab51e3711843`.
Launch SHA256:
`83a42b403ab3e71136375cd54d58a2bfb58511fcf5dfc4f5a48eeaea3891fd7e`.
Measured collection maximum 13.426330681 s/update, optimizer maximum
0.096356995 s/update and combined setup 27.796749570 s. The unchanged 1.25
rule derives 3490 s prediction, **4303 s child / 4363 s service**. Therefore
the 1693/1753-second historical budget cannot be used on this host. This is
observed wall-time throughput, not evidence that more VRAM is required.

This section predeclares a **new, separate WSL-only wrapper**, not an extension
of the historical wrapper or a retry of a killed learner. Round the measured
service cap up to a whole minute: `ceil(4363/60)*60 = 4380`, less the unchanged
60-second margin gives **4320 s child / 4380 s service**. Fresh launch window
must exceed 5040 s (service + 600 s closeout + 60 s reserve) and be at most
7200 s. The original Linux constants, wrapper, plans and evidence stay frozen.

Before launching any seed, run a fresh source-bound 900/960-second qualification
under the revised implementation. It must authenticate every byte of the prior
timing archive, re-derive its maxima and rounded budget, match current
host/dependency/asset inputs, then repeat integration and timing. Both setup
measurements must independently be finite nonnegative floats before addition.
The fresh derived cap must fit the newly declared WSL budget; a slower result
again means **no training**, not another extension. This supersedes only the
host timing/window paragraphs above. The learner budget, seeds, checkpoints,
PPO/physics, evaluation protocol and all capability gates are unchanged.

Each WSL service is additionally limited to 6 GiB RAM and 200% CPU quota, Nice
10; the timing probe and learner use the same limits. Full WSL evaluation remains
blocked in code until its own full-length timing probe is declared and measured.

## October 1: second rejection and disposable training bring-up

Fresh qualification at `f5aa2d13ee5c5d9e6a399bad1627a6704ca94c98` again
passed integration, but measured maxima derive 4388/4448 seconds, outside the
declared 4320/4380 pair. Report SHA256
`ecb56444c8b3ed362ecff65fc599ea45706e071b272e622b55ae29fde0b1635a`.
The full replication stays blocked: no further budget extension and no seed
577/587/593 launch. GPU peaked at 46 C; the CUDA child exited zero in 148.027 s.

Predeclare a separate **disposable training bring-up**, not a shortened
replication or a curriculum promotion: reuse the existing smoke learner and
purpose, fresh seed 523, 64 worlds, 16 updates, 24 ticks/update, CPU PPO and CUDA
eager physics. Seed 523 is reused only as the same infrastructure smoke on a
different host; it is not claimed as new independent learning evidence. No
parent initialization, optimizer resume, or pilot/replication parent authority.
The existing 900/960-second smoke bounds, fresh absolute window and 600-second
closeout remain fixed. The measured frozen-parent update cost is a conservative
sizing reference for this short bring-up, not trajectory equivalence; an actual
overrun still kills only the owned child. Same GPU lock, memory, CPU, temperature,
source/runtime guards. Retain all 17 weight exports, 384 tick receipts and 16
optimizer receipts, verify finite tensors and exact hashes before reporting.
Success proves this isolated host can execute and retain a real short training
loop; it establishes **no trained stance, hopping, avoidance or football balance**.
