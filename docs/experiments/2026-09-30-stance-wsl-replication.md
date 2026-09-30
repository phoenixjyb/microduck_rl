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
