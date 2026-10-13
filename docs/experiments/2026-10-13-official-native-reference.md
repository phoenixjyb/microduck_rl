# Official VelStand on the current native plant — predeclared diagnostic

This is a fresh simulation reference check, not a repeat of the one-shot C1-S
surrogate attempt, a new PPO training run, or admission of a teacher. The recent
quiet-head sharpening and gentle-command lessons both failed moving yaw gates.
Measure an independent reference before spending another matched learner pair.

## Bound artifacts and interface

- Archived official policy: `artifacts/community/intake-20261001/official-velstand/policy.onnx`,
  SHA256 `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`.
- Archived manifest SHA256: `622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94`.
- Policy-set revision: `pollen-robotics/microduck-policies`
  `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`. This archived artifact is NOT
  asserted to be the later v7 policy or a recoverable PPO checkpoint. Its manifest
  has no exact training commit/run and the ONNX's run path is `None`.
- API-1 float32 `obs[1,61] → actions[1,14]`, sole CPU ONNX Runtime provider,
  verified bytes loaded directly into the session. Raw 8-world observations are
  inferred independently at batch 1. The recognized exported affine normalizer
  runs inside ONNX only; no second normalization, command remap or action filter.
- Actual eight-term actor order: angular velocity `0:3`, projected gravity `3:6`,
  joint position `6:20`, joint velocity `20:34`, previous action `34:48`, twist
  `48:51`, head `51:55`, body `55:61`. Native cfg order/scales/history, raw action
  scale 1, named motor order and HOME are checked. A 61-wide shape alone is not
  compatibility proof. The CPU deployment script and native cfg share this order.

## Unchanged held-out protocol

Entry point: `python -m mjlab_microduck.official_native_reference --source <exact-commit> --run-id <unique-id>`.
Protocol `official-velstand-native-reference-v1` uses the existing
`integrated_turn_foundation.evaluate_case` collector, with ONLY a typed ONNX
inference backend replacing the PPO loader. No PPO runner, optimizer or training
is used. Legacy `checkpoint_sha256` trace keys identify the ONNX bytes in these
reports; `inference_backend.ppo_checkpoint=false` explicitly distinguishes them.

Current task `Mjlab-Velocity-Flat-MicroDuck`, plain native BAM M6 plant, pinned
Torch 2.9.1 / Warp 1.12.0 / MuJoCo 3.10.0 / MuJoCo Warp 3.8.1 / mjlab 1.3.0 /
BAM 1.0.1. This plant is not asserted identical to the archived teacher's training
plant. Preserve play-mode observation/simulation configuration, pre-reset motor
stream, first-terminal semantics and head/body command handling.

Exactly 12 cases, ordered seeds 839/853/857 × commands `(vx,wz)`:
`(0,0)`, `(.30,0)`, `(.20,-.20)`, `(.20,+.20)`; vy zero. Eight worlds, 240 control
steps at .02 s; omit first 60 steps only for tracking. No command remap, searching,
case retry or cherry-picking. Capture body/gait traces and named joint diagnostics.

Reuse the existing numerical gates without relaxation: every world speed MAE
≤.03 idle / ≤.10 moving, yaw MAE ≤.10; no terminal or rated-speed exceedance;
case-pooled torque-utilization p99 ≤.60; every joint margin ≥−.005 rad and
right-hip-yaw near-stop exposure ≤1%. `torque_p99` is dimensionless
`p99(abs(force_nm)/.60 Nm)`, so its .60 gate is equivalent to .36 Nm raw p99,
not .60 Nm. New explicit raw-Nm/reference companion fields do not alter this
legacy metric or any sealed report. Soft exposure and thermal-load values are
simulation proxies, not measured hardware temperature or lifetime guarantees.

## Operations and interpretation

Run once in a unique `artifacts/evaluations/official-native-<source>-20261013`
directory. Retain source, policy/manifest hashes, runtime, ordered per-case JSON,
health and final decision or exact failure. Create-exclusive outputs prevent
overwriting an earlier attempt. CPU adapter/hook/matrix tests and native CUDA-hidden
tests must pass before the GPU check. One retained user service, external 600-second
evaluation timeout, 8 GiB RAM/200% CPU ceiling, Torch 20% GPU fraction; GPU occupancy
≤6144 MiB, free ≥8192 MiB, temperature <75°C and existing exclusive lease required.

Refresh exact GPU UUID/DINO identity and both-scope service ownership before launch.
Pause only the originally active USER vLLM; preserve DINO and unrelated workloads.
The service's stop-post guard restores only that vLLM after the Duck process/cgroup
exits, the lease is free and DINO is the sole remaining GPU workload. Do not launch
on top of vLLM/NX/FilmBrain. No 100.98, driver or package changes.

A passing reference matrix would only justify a separately reviewed teacher/plant
comparison, not automatic PPO warm start or skill promotion. A failed reference
must retain failures and investigate plant/source mismatch, not relax thresholds
or repeat C1-S. Historical learner packets are descriptive context, not a matched
causal control for this independent reference. No video, raw perception, physical
motion, hop/obstacle/football promotion or automatic next long learner is authorized
by this diagnostic decision.

## Completed and rejected

Source `3b958f5b0d9823b1c3fb5d8e9d366cd513e0b502`; retained native directory
`artifacts/evaluations/official-native-3b958f5b-20261013`. All 12 cases complete,
240 steps/eight first-episode worlds each, in 56.421 seconds. All nine moving
cases fail tracking: moving speed 1/72 worlds, yaw 0/72; idle speed/yaw 24/24.

| Command (vx,wz) | Mean speed MAE | Mean yaw MAE | Mean body vx | Mean body wz |
| --- | ---: | ---: | ---: | ---: |
| (0,0) | .000964 | .005659 | −.000174 | −.000346 |
| (.3,0) | .162761 | .278769 | .137239 | −.046081 |
| (.2,−.2) | .168983 | .229231 | .031017 | −.052203 |
| (.2,+.2) | .176715 | .226041 | .023285 | .011856 |

No terminal, rated-speed exceedance or motor/range gate failure. Highest pooled
utilization p99 .502438 = raw force p99 .301463 Nm; minimum all-joint margin
.203911 rad. Sampled GPU peak 1784 MiB, minimum free 14410 MiB, maximum 57°C.
The process succeeds but the numerical decision is `foundation-not-ready`, not
teacher/skill/physical acceptance. This archived reference is NOT a walking teacher.

Evidence seals (full traces remain on 100.100; lightweight seals suffice on Mac):

- Result: `dd19de98099474a0e255d7be4019d1c774c1f13c9f82b59a345d502e0c9b3507`.
- Fifteen-file native inventory: `426900a0314bd764ca2847e5f3de2a774ac40ab13f903283a81a57452f066129`.
- CPU-only summary: `a23ad79b235b6f07b644c0c84aafbe1de9d1741fbc662c25592081a9d6db3747`.
- Restoration: `8122ff8e9fffcd4e8bd898282690c075cd55118b2c0334968b321f9f9a6b2bb0`.

The create-exclusive closeout checks exact result hash, ordered case identities,
per-case/aggregate equality, artifact identity, finite values, unchanged gates,
force-unit conversion and retained original-service restoration. The user service
is inactive/success; user vLLM was restored (observed main PID883343, GPU engine
883655/9692 MiB) and DINO2029/946 MiB preserved. Other protected services stay inactive.
These process identities are point-in-time, not reusable host defaults.

Checks: Mac focused suite 74 passed; native affected suite 1256 passed/16 existing
warnings; restoration guard 3 passed/7 refusal subtests; native CPU 240×8 ONNX
calls .04035 seconds. The broad all-tests sweep timed out at 180 seconds and is
NOT a broad pass. Its first failure is the unchanged ADA audit's global
`sys.modules` isolation assertion after whole-suite collection loads Torch;
that file passes separately (88 tests). An unrelated ADA CPU fixture also fails
its retained raw-payload bound (54 passed/1 error in the diagnostic subset).
Those sources are unchanged from c5274c58; no ADA repair, driver or package change
was made. Exact logs are retained in native `.local/state/official-native-3b958f5b-*`.

The launch's initial SSH attempt timed out before connection; on restored
reachability its never-started service timestamp and absent output were verified
before one successful launch. Do not confuse the connectivity retry with another
evaluation attempt. Next assess the separately pinned upstream v8 release,
not another blind scalar/support learner or a relaxed gate.
