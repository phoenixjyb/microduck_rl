# Frozen stance evaluation: initializer versus final eager checkpoint

One bounded evaluation follows the completed fresh seed-563, 64-world,
128-update eager learning diagnostic. No additional optimizer run is authorized
by this evaluation. Keep all training inputs and existing skill policies intact.

## Predeclared comparison

- Exactly `initial.pt` (iteration -1) and `model_127.pt`, in that order. No best
  checkpoint search, intermediate substitute, resume or policy mutation.
- Each checkpoint uses seeds **541, 547, 557**, sequentially, **128 worlds** per
  seed. Six cases and **768 first attempts**, of which 384 use final weights.
- Fresh nominal placement, deterministic frozen CPU actor mean, eager CUDA0
  physics, **250 policy ticks / 2,500 physics steps / five seconds** maximum.
  Freeze on the first physical failure; never reset during evaluation.
- Reuse the unchanged every-boundary stance scorer, physical/motor stops,
  action/delay/control replay, compiled-plant checks, terminal contact checks,
  strict checkpoint restore and deterministic actor replay. No new tolerances.
- Report all per-attempt numerical gates, first-attempt durations and per-seed
  pass counts. The final arm must still pass at least **122/128 per seed** to
  meet the original per-seed numerical threshold. All six cases must complete;
  a timeout prefix cannot count as a completed comparison.
- Nominal seeded repeats are not randomized generalization trials. Durations
  are censored at five seconds. Descriptive improvement is not a statistical
  significance claim, multi-seed training replication or football balance.

The distinct eager trace protocol permits only iterations -1 and 127; its
checkpoint loader accepts only `eager-learning` exports. The original pilot
protocol still accepts only its own checkpoints 128/256/384/511, and its matrix
is unchanged. Neither loader aliases iteration labels or promotes a diagnostic
export into the original 512-world/512-update pilot.

## Immutable training evidence

Training implementation: `c8f6b994a2991e400bf6478b967c30e9b618db6a`.
Archive: `artifacts/evaluations/stance-eager-learning-c8f6b994a299`.
Independent completed-report SHA256:
`fc7356c3f3763bc3bf6e800e65adb1826cabe5a86b0705e6c3fbf3607dcd4948`.
Before deserialization, verify that hash and every file in its exact inventory.
Then check all completed training exports/receipts and select only the pinned
initializer and final export. The original archive must remain unchanged after
evaluation. The training service exited successfully after 128 updates; its
735.963-second child duration is not an evaluation timing estimate.

## Bounded launch and retention

Use one sequential user service on 100.100, a shared GPU0 lease, a **900-second
independent child watchdog**, **960-second whole-service timeout**, control-group
cleanup and an additional **600-second closeout reserve**. Require a new explicit
absolute window of more than 26 and no more than 60 minutes at entry. This does
not renew an old unattended schedule. Full worst-case evaluation timing is not
yet measured; retain partial completed cases on timeout and do not extend/retry
automatically or reduce coverage to manufacture completion.

Require an idle GPU, clean exact feature-branch commit, pinned dependencies,
assets and compiled plant, no competing compute owner, temperature below 80 C,
and both protected AI Mission system services inactive. Preserve unrelated
workloads and 100.98. Never restore protected services or cause physical motion.

Each case retains a complete hash-bound trace/control/checkpoint/runtime bundle,
its independently retained manifest digest, and a collection receipt. Reverify
all six bundles and recompute the final comparison before reporting completion.
Numerical rejection is a valid completed experiment, not an infrastructure
failure. Always keep `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` false. No MP4,
training promotion or next GPU job follows automatically.

Focused CPU tests and exact-source Linux verification must pass before launch.

The focused local selection passed **134 tests in 19.69 seconds**. It includes
real short CPU physics/control-bundle collection and replay for the new trace
identity, frozen diagnostic restore, protocol separation, missing/partial-case
rejection, the unchanged 122/128 boundary and independent service/hash gates.
This is not a real CUDA evaluation result. Exact-source Linux tests remain the
next launch gate.

The complete training archive was mirrored to the same relative path on the Mac:
**3,334 files, 86,811,418 bytes**, including the independently pinned report and
all 3,333 files it hashes. CPU verification restored and checked every export and
receipt before selecting final checkpoint SHA256
`46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93`.

## Completed frozen evaluation

Exact implementation `bd099638f52b8b44c05d0d7c9f02f62f0538417c` passed **665
Linux CPU tests in 96.11 seconds**, without skips or deselections. Before launch
the GPU was idle at 12 MiB / 46 C, with no compute owner and both protected
system services inactive. The user service
`microduck-stance-eval-bd099638f52b.service` started at **2026-09-09 16:50:10
Asia/Shanghai**, supervisor PID 711795 and sole GPU child PID 712047. It completed
with MainPID 0, Result success, ExecMainStatus 0 and active/exited retained state.
The child took **354.216 seconds**; peak sampled GPU temperature was **56 C**.
The GPU was subsequently idle at 12 MiB / 45 C, with protected services still
inactive. No optimizer, follow-on job, video or service restoration ran.

The unchanged scorer and full bundle replay completed all **768 first attempts**:

| Frozen weights | Evaluation seed | Mean first-attempt duration | Hard failures | Full stance passes |
| --- | --- | --- | --- | --- |
| Fresh initializer | 541 | 1.164516 s | 128 / 128 | 0 / 128 |
| Fresh initializer | 547 | 1.164609 s | 128 / 128 | 0 / 128 |
| Fresh initializer | 557 | 1.164438 s | 128 / 128 | 0 / 128 |
| Final iteration 127 | 541 | 5.000000 s | 0 / 128 | 0 / 128 |
| Final iteration 127 | 547 | 5.000000 s | 0 / 128 | 0 / 128 |
| Final iteration 127 | 557 | 5.000000 s | 0 / 128 | 0 / 128 |

Every final-policy attempt reached the five-second endpoint and passed all
individual stance gates **except final upright tilt**. Across the 384 final
attempts, final-second tilt p95 ranged **7.95283 to 8.13340 degrees**, above the
unchanged **0.0873 rad (approximately 5 degrees)** limit. Other worst-case final
metrics were:

- Maximum planar displacement: **0.01224868 m**, below 0.02 m.
- Final-second planar-speed p95: at most **0.00161868 m/s**, below 0.03 m/s.
- Final-second minimum height: at least **0.11518921 m**, above 0.105 m.
- Both-foot support after the required settling interval: **100%**, above 99%.
- Soft-limit joint-time exposure: **0%**, below 1%.
- No first-attempt hard failure in any final case.

This establishes a useful **nominal simulated stability improvement** over the
same policy initializer, but not an accepted upright stance. Five-second results
are horizon-censored; nominal repeats do not establish disturbance recovery,
independent-training-seed replication, physical motor safety or football balance.
The deterministic decision is **`final-numerical-rejected`**, not a GPU/runtime
failure. Do not relax the upright limit or promote this checkpoint.

### Retained hashes

Directory: `artifacts/evaluations/stance-eager-evaluation-bd099638f52b`.
The completed remote archive contains **59 files / 2,299,673,891 bytes**.

| Artifact | SHA256 |
| --- | --- |
| launch.json | `2e1b7ba2cc4805f4bf012aba1d572bc00e7ebf8563769c71173ba82ddc5bf969` |
| report.json | `a6f4e4ca5e7e81f8bed8ce2f5226e08422ad581836c727140356c0423bb2e297` |
| comparison.json | `9a3c524195a06ca646d711cfcb9a73485b131e22543da0349263e77e62c34b67` |
| initial-seed-541/manifest.json | `705b774d86191d7268c8cba6df6c9579ac78e777604dbb6aa21396be81a75c1c` |
| initial-seed-547/manifest.json | `60b0a2ff480f3a32dae52c08192f38282ab74bde9d1f3de38a25526126db7462` |
| initial-seed-557/manifest.json | `0f9515ce8dbb6cbb7499395d6d5217b8af56e6ef33f284ba4e3278f49316bd46` |
| final-seed-541/manifest.json | `35e66a0dba891cf4c90e14b04f4e70ee7b41a491dda87e6ea8f1ca9ff3edb337` |
| final-seed-547/manifest.json | `6a332b99f14fe75b180719b3cc1510dac6625e06331b521b4476086041176d5e` |
| final-seed-557/manifest.json | `5e807992758bd06cf69ab32ef208c432e7090bd36af7649192b45087a3dbc7cb` |

The GPU worker replayed each retained bundle and the CPU supervisor separately
reverified all six bundles and recomputed the comparison before publication.
The training archive was also rechecked unchanged. A separate post-run Linux
process, with CUDA hidden, independently verified the pinned report and launch
hashes, every top-level recorded hash, all six bundle manifests and their full
trace/control/actor/plant replay, the deterministic comparison, and the immutable
training archive again. It completed in **42.17 seconds**, reproduced all 768
attempts and `final-numerical-rejected`, and confirmed CUDA remained uninitialized.

The 2.3 GB evaluation mirror to the Mac is still transferring at this record;
do not call that copy complete or independently verified yet. The original
remote evidence is complete and verified. The separate 86.8 MB training archive
mirror described above is complete and verified on both hosts.

### Next bounded lesson

Target the remaining steady lean while retaining the successful five-second
hold, foot support and motor limits. The existing reward already rewards
uprightness; this single short training seed does not by itself establish that
the reward needs redesign. Predeclare the next training budget and checkpoint
evaluation before spending more GPU time. These weight exports are **not**
optimizer/simulator-resume checkpoints: a longer fresh run or an explicitly
designed weight-initialized run must be identified honestly and separately.
Do not move on to perturbations or ball support before the applicable stance
and replication gates are satisfied.
