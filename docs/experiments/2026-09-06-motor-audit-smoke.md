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

## Retained result: measurement smoke validated, no policy admission

Executed once at source **`fb7b1dc0d32c569e40c9ee5f5bbd7f32a4558af3`**, after all
359 focused tests passed on 100.100 with CUDA hidden (6.22 s). Runtime versions,
all three installed dependency hashes and both model hashes matched preflight.
The retained service started **2026-09-06 17:17:40 Asia/Shanghai**, exited at
**17:18:01**, Result=success, ExecMainStatus=0, MainPID=0. RemainAfterExit leaves
the unit active/exited; that does not mean a Duck process is still running.
The sole child returned zero in **15.083693279419094 seconds**. No retry,
second rollout, training, video, dataset, perception or physical action followed.

Deterministic decision: **`measurement-smoke-validated-not-admission`**.
Measurement structure and the unchanged legacy runtime gate both passed.
The pure JSON validator reproduced the exact retained decision independently
on the host and on the local backup; all five files matched SHA-256 byte-for-byte.
All policy, runtime-equivalence, dataset, physical and further-GPU admission
flags remain false.

Post-result validation: all **359 focused CPU tests passed locally again**
(215.24 s), with the same existing real-spec pattern warning and no failures.
Both frozen model hashes were rechecked unchanged after the rollout.

- Four expected/completed first attempts, **4 clean passes**, zero collisions,
  timeouts, falls, NaN terminations, nonfinite steps, hard/other terminal events
  or unresolved attempts.
- **389** executed/captured control steps, **1499** active environment-steps,
  **20,986** motor samples, all finite; four terminal environment-steps and
  zero incomplete attempts. All 14 named joints and all three phases reconciled.
- Legacy torque-utilization p99 **0.5748031139373779**, below the unchanged
  pooled 0.60 limit. Legacy rated-speed exceedance fraction zero, speed-reference
  utilization p99 **0.4123215973377228**; thermal-load proxy mean
  **0.03993804752826691**, not a measured temperature.

The new pre-reset measurements (force divided by the 0.60 Nm model reference):

| Scope | Environment-steps | Pooled utilization p99 | Highest joint utilization p99 |
| --- | ---: | ---: | --- |
| All | 1499 | 0.5755217795570708 | right_hip_pitch: 0.7986437002817789 |
| Approach | 20 | 0.5778405760725338 | right_knee: 0.6865567748745283 |
| Interaction | 885 | 0.5203912352522216 | left_hip_pitch: 0.7094891627629595 |
| Recovery | 594 | 0.6324249580502511 | right_hip_pitch: 0.8497568269570681 |

The pooled statistic obscures greater phase/joint loads in this case. In
particular, recovery deserves attention before any motor-aware curriculum
revision. This is a descriptive finding, **not a new post-hoc per-phase or
per-joint acceptance threshold**. The pre-reset maximum sampled actuator force
was **0.6405236124992371 Nm** (normalized **1.0675393541653952**), above the
normalization reference. This is not proof of acceptable real motor operation;
the reference is not a continuous-duty rating or an asserted simulation clamp.
No cause for that peak or general cross-seed load pattern is established here.

Terminal motor samples visibly differ across the two sampling boundaries:
pre-reset absolute force p99 **0.47529449462890627 Nm**, post-return p99
**0.26755833625793457 Nm**. The p99 of the **paired absolute differences** is
**0.7220623001456261 Nm**, with maximum **0.7372826337814331 Nm**; it is not the
difference of those two quantiles. All three distributions contain 56 samples.
This supplies live evidence that post-return samples do not preserve the
terminal motor state. It does **not** reconstruct seed-367 or prove that reset
sampling caused its rejection. Historical decisions remain unchanged.

GPU observations: 0%/44 C/12 MiB before launch; 2%/46 C/397 MiB during, with only
the expected child compute PID 1214254; 0%/45 C/12 MiB at 17:18:21 after exit,
no compute PID. These are snapshots, not a continuous temperature-peak record.
Both protected system services remained inactive with MainPID=0 at closeout;
neither service was changed.

Remote evidence remains in the predeclared output directory. Local second copy:
`/Users/yanbo/Projects/microduckPlayground/microduck_rl/artifacts/diagnostics/motor-measurement-audit-s373-v1`.

| Retained file | SHA-256 |
| --- | --- |
| decision.json | `79bc321740fb6c43e60853942247ba4e21bce23a98bc375161277e58d05fab0c` |
| launch.json | `3dffa5eca397d4701d2b421f23a9bdc722ee37593d0e43d224d3d4bf45e6aafd` |
| rollout/hierarchical-teacher-evaluation.json | `60a2d0e23d8b3e1a65a63b47a97134aeadefde3602a157a78782637224b23b77` |
| rollout.log | `bfe9a2e7df89c792ad89e1b9e2bb6b3a568762715fcb34dd6b3d05919d2d4b70` |
| runtime.json | `2833a9bd4f88467e0bcea42d58d8dc411fa477758e830a7ba7c6d3d4d9a77913` |

Next boundary: use these retained phase/joint findings to propose a gentler
return-to-speed, motor-aware simulation curriculum with predeclared criteria.
This result alone authorizes neither its implementation nor another GPU run,
and still does not establish repeatability, sensor-recorder non-interference,
general policy acceptance or physical readiness.
