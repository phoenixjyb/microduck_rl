# One bounded pre-reset motor-audit smoke

Date: 2026-09-06. Predeclared before execution. This document and runner extend
the CPU-only measurement work at `ce37747596a429768bba08ada76d35187daa96a5`.
The user's subsequent approval permits this one short simulation-only validation,
not a training campaign or policy promotion. The previous measurement document's
no-GPU boundary describes its own source-only change, not this new authorization.

## Frozen experiment

- Protocol/output name: `motor-measurement-audit-s373-v1`.
- Host: 100.100, exact worktree
  `/home/converge/work/microduck_rl-athletics-obstacle-curriculum`, branch
  `feat/athletics-obstacle-curriculum`. Clean tested source commit recorded by
  `--source` in immutable launch/runtime metadata, committed/pushed before launch.
- One retained user service: `microduck-rl-motor-audit-s373-v1.service`;
  Type=exec, RemainAfterExit=yes, RuntimeMaxSec=600, TimeoutStopSec=15,
  KillMode=control-group, Restart=no. One child, timeout=240 seconds, no retry.
- Launch strictly before **2026-09-06 17:35 Asia/Shanghai** (09:35 UTC).
  No later than 17:45:15 under service bounds. This is not an overnight extension.
- Fresh diagnostic seed **373**, four environments, 700 control-step ceiling,
  first-terminal-attempt only, 12-second simulation timeout, speed 0.4 m/s,
  obstacle relative forward 0.9 m and lateral 0.0 m, unchanged zero-noise external
  obstacle observation. No raw perception or actor-observation expansion.
- Audit enabled with `--motor-measurement-audit`; trajectory recorder and all
  dataset collection disabled. No video. No optimizer/training process.
- Frozen actor SHA-256:
  `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Frozen U4 supervisor SHA-256:
  `29855a51df8fe885d6ffed7fedf028093a8449a68b10b4b0e8a4bde7069bcf5b`.
  Both are the retained paths bound in `rollout_repeatability.py`; no checkpoint
  change, resume or training. U4 remains rejected.
- Runtime version and three installed mjlab dependency hashes are hard-checked
  against the CPU-audited versions in `motor_audit_smoke.py` before execution.
- Preflight requires no compute processes, GPU utilization zero, memory below
  100 MiB, temperature below 80 C, and both protected system services inactive:
  `recomo-ai-mission-vllm.service` and
  `recomo-ai-mission-subject-model-worker.service`. Never change them. Preserve
  unrelated workloads; 100.98 is not used. Check occupancy again at launch.

## Separate decisions, unchanged stop rules

The measurement check requires the exact case/artifact/timing identity, 14 named
unit-gear joint columns, one capture per executed step, four complete terminal
attempts, every phase exercised, finite nonnegative statistics, correct per-joint
and pooled sample counts, phase partition, and force/reference normalization.
Terminal pre-reset, post-return and difference distributions each cover 56
motor samples. These summaries permit structural reconciliation, not independent
reconstruction of quantiles or proof of the actual physics timing from JSON alone.
Timing evidence combines the installed step-order CPU regression and the live
callback path. Terminal difference need not be nonzero to validate the observer.

Legacy runtime gate remains torque-utilization p99 <= **0.60 dimensionless**,
rated-speed exceedance fraction zero, no falls, NaN/nonfinite steps, hard/other
terminal failures or unresolved attempts. All outcome counts must reconcile.
Collision/timeout outcomes remain descriptive, never policy acceptance. A lower
pre-reset metric cannot rescue a failed legacy gate or change old results.

Decisions: `measurement-smoke-validated-not-admission` only when structural and
legacy runtime checks both pass; `legacy-runtime-gate-stop` when structure is
valid but the legacy check fails; `invalid-audit-stop` or `runtime-failure-stop`
otherwise. **Every decision stops this job**; none admits another GPU run.
The corrected utilization is descriptive, with no newly invented acceptance
threshold. Neither metric measures physical motor temperature, continuous-rated
torque, synchronized mechanical power, or all-physics-substep peaks.

One audit-enabled rollout cannot establish trajectory non-interference or
repeatability. Seed-359 recorder failure, seed-367 control stop, and U3/U4/O3a
rejections remain unchanged. No policy, recorder, dataset or physical admission.
The user has no physical duck; model/motor assumptions remain uncalibrated.

## Retention and verification

Runner: `python -m mjlab_microduck.motor_audit_smoke --source <tested-commit>`.
Exclusive output directory `artifacts/evaluations/motor-measurement-audit-s373-v1`:
launch.json, runtime.json, decision.json, rollout.log and
rollout/hierarchical-teacher-evaluation.json. Existing output refuses execution.
Retain any failure evidence unchanged; diagnose read-only before any proposed fix.
After process exit, verify GPU idle and protected services unchanged, independently
rerun the pure JSON validator, hash and copy the evidence locally, record the
outcome here, run focused CPU tests, commit and push this exact feature branch.
No follow-on training is authorized by the smoke result.

Prelaunch local validation: **359 focused CPU tests passed** with CUDA hidden
(10.70 s), including 54 new smoke/validator tests. The existing real-spec
actuator/site-pattern warning remains; named unit-gear JOINT mapping is tested.
Remote repetition at the exact pushed launch commit is required before launch.
