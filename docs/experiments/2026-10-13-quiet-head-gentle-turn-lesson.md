# Gentle-turn command lesson with a simpler head task

Protocol `integrated-quiet-head-gentle-turn-v1`, branch
`integration/upstream-2026-10-10`. Source:
`src/mjlab_microduck/integrated_quiet_head_gentle_turn_lesson.py`.
This document is the predeclaration, not a result or skill admission.

## Why this lesson

The [simpler-head lesson](2026-10-13-motor-range-head-lesson.md) improved all
moving speed/yaw group means and motor/range aggregates, but still failed the
motion gate. The [sharper-yaw contrast](2026-10-13-quiet-head-yaw-focus.md)
subsequently worsened steering, drift and falls. Do not continue that reward
width sweep or warm from either rejected endpoint.

The next question is whether **easier signed-turn commands** improve acquisition
under the simpler-head baseline. Change only training yaw support: control
`[-0.50, 0.50]` versus gentleturn `[-0.25, 0.25]` rad/s. No staged widening or
reward schedule. This preserves both turn directions and contains the unchanged
held-out ±0.20 rad/s commands. Narrowing uniform support approximately doubles
the conditional signed-yaw target-band exposure, (.15,.25], from 20% to 40%;
measure actual action-step exposure rather than assuming this is achieved.

An earlier narrow-yaw probe used another parent, action cost, additive angular
objective, standing fraction and update budget. It increased target-band dwell
but failed its motion gates. Reuse its tested instrumentation, not its endpoint
or a pooled performance claim. This new contrast isolates command support under
the current parent and quiet-head stock objective.

## Matched parent and fixed behavior

Both start from the original unaccepted reference, not a promoted teacher:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Restore exact models, normalizers (43,008,000), all 17 Adam histories (140,000)
and learning rate; verify source iteration 999 / environment steps 24,000.
Reset learner/environment clocks to zero and materialize the seven source
curricula at their final conditions before applying the matched lesson.

Common head ranges remain the existing small nonzero four-slot lesson:
±.05 / ±.05 / ±.07 / ±.015 rad. Head tracking +2, bias +3, 2–5 s resampling and
zero-head probability 0 stay fixed. Twist vx [.2,.4] m/s, vy zero, resampling
3–5 s, standing fraction .25 and forward-only fraction .25 stay fixed; no
heading/world-frame commands, turn-in-place or initial-velocity sampling.

Both retain stock XYZ angular tracking +2 / variance .5, stock XYZ linear
tracking +2 / variance .1, action-rate −1, motor cost −1, bilateral named
hip-yaw qpos cost −4 / .15 rad margin, trunk-angular −.05 and whole-body
momentum −.02; forward-error cost remains zero. PPO, DR/noise/delays, BAM plant,
solver and reward aggregation remain unchanged. No head freeze or new penalty
is inferred from the previous head/yaw correlations.

The normalized actor contract stays raw 61 → raw 14, mouth excluded, no target
clipping or action filtering. Official ONNX export bakes normalization. No new
actor/critic observations, raw perception, controller switching or physical motion.

## Bounded admission and learning

| Phase per arm | Worlds × updates | Matched seed | Child cap |
| --- | ---: | ---: | ---: |
| Normalized ONNX smoke | 64 × 5 | 1673 | 600 s |
| Timing benchmark | 256 × 10 | 1693 | 600 s |
| Control / gentleturn | 256 × 1,000 | 1697 | 2,100 s |
| Final-999 held-out matrix | 8 × 240 per case | 839 / 853 / 857 | 900 s combined |

Focused source tests and complete native scoped CPU regression precede both
smokes and benchmarks. Require `benchmark_elapsed * 100 * 1.5 < 2040 s`;
never widen the cap after seeing timing. Run control → gentleturn → both
matrices sequentially. Each learner uses 6,144,000 transitions / 20,000
optimizer minibatches, final normalizer 49,152,000 / Adam 160,000. Retain initial
and 0 / 250 / 500 / 750 / 999 saves; evaluate only final 999, not selected saves.

Before action, check the entire live sampler configuration against the declared
configuration, cached actor twist slots 48:51 against the manager command, and
actual yaw support. Do not rewrite cached commands or consume extra RNG. Retain
the existing eight exclusive dwell bins (idle, straight, signed low/target/high)
with per-update and cumulative integer counts. High bins must be exactly zero
in gentleturn; do not fabricate high-yaw coverage. Both still require actual
idle, straight and both signed-turn exposure. Validate every emitted support,
action, linear, head and angular row, including .25 standing and stock angular
width/weight, without recomputing the stateful head EMA.

## Unchanged held-out gate and outcome rules

The full matrix remains seeds 839/853/857 × idle / (.3,0) / (.2,−.2) / (.2,+.2):
12 cases per arm, eight first-episode worlds, 240 steps, omit startup 60 only
for tracking. Per-world speed MAE ≤.03 idle / ≤.10 moving, yaw MAE ≤.10,
no terminal or rated-speed exceedance, case-pooled torque p99 ≤.60 Nm, all
14 hard margins ≥−.005 rad and right-hip stop exposure ≤1%. No relaxed gate.
Retain body-orientation and gait traces for every case using the existing
read-only pre-action API and truthful contact timing; no sensor refresh.

Report every group/world pass and regression, speed/yaw means, signed responses,
paired motor/power/thermal/range/stability metrics, actual target-band dwell,
falls total/first100/last100 and reward mass. The separate descriptive-benefit
rule requires all three moving speed means lower, no moving yaw mean regression
and no paired motor/range/stability regression; it never waives a hard gate.

No endpoint adoption, wider-turn claim, head-range expansion, longer training,
obstacle/hop/football promotion, MP4 or publication follows automatically.
Even a numerical foundation pass requires independent-seed confirmation and
retention/transfer checks; policy/simulator/physical acceptance stays false.

## Shared-host boundary

Use a fresh clean exact tested/pushed source on native 100.100; freeze it while
active. Recheck GPU UUID, retained DINO PID/exe/uid and all protected-service
scopes. Pause/restore only originally active user `recomo-ai-mission-vllm.service`;
preserve DINO, unrelated workloads and 100.98/NX. Owned lease, GPU used ≤6144
MiB / free ≥8192 MiB / temperature <75°C, Torch 20%, one CPU thread, 8 GiB /
CPU 200%; nonenabled preflight 45 min / matched 95 min / handoff 150 min services.
The owned restore guard requires no Duck child, released lease and DINO-only
GPU before restoring the original user service and proving its GPU ownership.
Unknown workloads cause refusal, not unrelated stops or concurrent GPU jobs.
No driver/package change is part of this lesson.

Source tests, exact source identity, native admission and results are pending.

## Source preparation evidence

CUDA-hidden single-thread focused regression: **217 passed in 8.80 s** across
the new lesson, head lesson, earlier low-yaw probe, yaw-focus, action/linear
telemetry and gait-trace tests. Paired configuration checks isolate only yaw
support; parent/optimizer contracts, evaluator forwarding, live sampler/cache
faults, stock angular terms and support-count validation are covered.

Owner review caught duplicate exposure bookkeeping before any GPU admission.
The lesson now reuses the inherited capture/step/update chain rather than
maintaining a second counter stream. Full 24-step observer-chain fixtures for
both arms verify 1,536 samples counted once and reset update counters, while
bad cached commands/support leave counters unchanged. This is CPU fixture
evidence, not a simulator rollout. Independent portable arithmetic tests also
cover support, standing fraction, cumulative continuity and malformed budgets.

Independent source/declaration/generated-ops-path review cleared. Five portable
arithmetic tests pass, with stock-angular and support/protocol/standing/count/
cumulative/budget faults; the inherited validator is replaced rather than
shadowed by a contradictory duplicate. Documentation links and whitespace pass.

Complete native regression, both normalized smokes, fresh host context and
measured timing remain mandatory. No learner has been admitted yet.

## Native admission passed; matched training active

Exact tested/pushed clean native source
`c5274c58c2dbba01558ba636c8b442c590da358a`; source is frozen while active.
Native scoped regression: **1,216 passed, 16 existing warnings in 37.64 s**.
Normalized-export smokes completed 4.659 / 4.632 s with zero falls/NaNs;
benchmarks 8.083 / 8.151 s, three / six falls, zero NaNs. Projected budgets
1,212.411 / 1,222.622 s stay below 2,040 s. The benchmarks admit runtime and
timing, not a motion improvement.

All four initial checkpoint hashes remain
`d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`,
with exact original parent models/normalizers/Adam/LR restored. Thirty closed
rows independently pass action, linear, head, angular and command-support
validators. Actual treatment high bins are zero, both signed target bins are
nonzero, and stock angular .5 / .5 / weight 2 remains unchanged in both arms.

The automatic training export hook repeats the existing warning
`'Logger' object has no attribute 'logger_type'` in all four preflight phases;
the previous yaw-focus preflight has the same four warnings. Read-only diagnosis
confirms the **separate explicit official normalized-export smoke** succeeds,
with finite [1,61] → [1,14] receipts and byte hashes bound by native admission:
control `8f2c2d12ab47c2eadaf4b335477adb601d8fce70e4d24daeb6de6307f55d1893`,
gentleturn `d4421cd15c5bc4654fae7f2a406ae26404f1bf9aa948d4c0ab8e476a340d1ab3`.
No logger/package/source fix was made during the active job. Later deployment
must still use the explicit official export route, not assume the hook worked.

Admission SHA `1c37ad7a19db819ea12ff3ee6cc55346622fecdff5c9d42386ce25427b5f13a3`;
fresh context `875894a512b090cbd2eccae6adfa1a9142a7a62c7c01f7a64ecd665ea99c089d`.
Closed JSON-only Mac packet
`artifacts/evaluations/quiet-gentleturn-c5274c58-20261013-preflight`; owner
compared both hashes to fresh native reads and reran all 30 five-contract checks.
Three inherited mock restore tests with seven refusal paths pass; generated
Python/shell and native systemd unit verification pass.

Run `artifacts/training/quiet-gentleturn-c5274c58-20261013`, retained services
`microduck-quiet-gentleturn-c5274c58-matched-eval.service` (MainPID 801302) and
`recomo-duck-quiet-gentleturn-c5274c58-handoff.service` (MainPID 799350):
control → gentleturn → both final matrices. First audit control 220/1,000,
214 training falls, zero NaNs, finite rows; GPU 1,710 MiB / 65°C, DINO 2029
preserved, all four protected scopes inactive during the scoped user-vLLM pause.
Final checkpoints, held-out outcomes, evidence seal and service restoration
remain pending. No motion capability, video or harder-skill promotion is claimed.

## Closed result: exposure shift achieved, motion gate still rejected

Both learners completed 1,000 updates / 6,144,000 transitions / 20,000 optimizer
minibatches: control 672.690 s, gentleturn 680.046 s. All declared saves, exact
initial-parent restoration and final normalizer 49,152,000 / Adam 160,000 are
verified. Final checkpoint hashes:

- Control `188475601ffa2004c12cf77b9870ccbe11d8f22c47911550d9dc1d510e0c4c84`.
- Gentleturn `80b5dbf22fe6049ba4f5a781b653977b8288eb0ebda79edf4a9bf9dd51a2e2e6`.

All 24 held-out cases complete the unchanged 240-step first episode. Idle speed
and yaw pass 24/24 worlds in each arm. Every moving case still fails tracking:
moving speed passes **53→51/72**, moving yaw **0/72 in both**. Both decisions
remain `foundation-not-ready`, not policy or simulator qualification.

| Command (m/s, rad/s) | Speed MAE control→gentleturn | Yaw MAE control→gentleturn | Speed passes /24 | Worst case torque p99 Nm |
| --- | ---: | ---: | ---: | ---: |
| Idle | .000947→.001087 | .006776→.007787 | 24→24 | .247408→.244288 |
| (.3,0) | .110048→.112365 | .227538→.275427 | 5→3 | .490175→.464991 |
| (.2,−.2) | .064319→.060304 | .232111→.276454 | 24→24 | .462060→.432989 |
| (.2,+.2) | .063210→.070974 | .245758→.266547 | 24→24 | .465213→.430127 |

No terminal, rated-speed exceedance, hard-range violation or right-hip stop
exposure occurs. All case torque p99 values remain under .60 Nm; minimum joint
margin stays positive but decreases .252958→.206727 rad. These safety-subset
passes are not motion qualification. Independent matched numerical review also
finds paired regressions: torque p99 3/12 cases, soft-limit fraction 3/12, power
2/12, thermal proxy 3/12, minimum joint margin 11/12. The descriptive-benefit rule
fails: straight and positive-turn speed means regress, as do all three moving
yaw means. Do not pool this seed-1697 control with earlier controls.

Training falls 934→1,035; first100 96→97, last100 101→119; zero NaNs.
Actual mean angular reward .738544→.729868, yaw-error square .457705→.460727,
XY-rate square .633483→.644428, raw linear reward .722670→.714689 and action
cost 1.036990→1.072214. These are telemetry, not causal explanations.

### Authenticated support and heading/gait evidence

The intended rehearsal actually happened: target-band (.15,.25] exposure among
nonzero-yaw action steps is 698,143 / 3,488,292 = **20.0139%** for control and
1,374,861 / 3,471,339 = **39.6061%** for gentleturn. Both treatment high bins are
exactly zero; all four required command groups have actual exposure. Every
1,000-row training log passes integer/bin/support/standing/clock/cumulative
continuity checks. Exposure is a measured lesson property, not an acceptance gate.

All 24 body/gait trace identities and complete clocks validate. Six stdlib
audit tests pass 11.411 s, using synthetic partial/complete idle/straight/arc
cases and rejecting stale pins, malformed counters and output overwrite.
The audit reuses the unchanged pinned heading/gait math and portable support
helper; no new physics, RNG, observation or reward call was introduced.

Straight heading is mixed, not uniformly better: mean absolute net change
.247862→.264221 rad over 3.58 s, while worst absolute change improves
54.770° (control seed857/world1) → 41.473° (gentleturn seed853/world6).
Signed means −.202717→−.075076 rad hide opposite-sign worlds and are not a
substitute for per-world evidence. Straight group mean absolute 25-sample
windowed body-wz error .078048→.081322 rad/s also regresses.

Negative-turn mean body wz −.114301→−.073065 rad/s (target−.2), world-heading
slope −.116220→−.073964; positive-turn mean body wz .073663→.121350
(target+.2), heading slope .076488→.117683. Arcs remain asymmetric and their
instantaneous yaw MAE worsens in both directions. Projected world heading and
body-wz command tracking are distinct; no integrated-heading gate was invented.
The retained gait/contact/head correlations remain descriptive, not causation
or permission to freeze the head or loosen the motor/turn gates.

### Evidence seal and original-service restoration

Native packet `artifacts/training/quiet-gentleturn-c5274c58-20261013`; JSON-only
Mac packet `artifacts/evaluations/quiet-gentleturn-c5274c58-20261013`. Native and
portable one-shot verification each pass: **59 files / 260,822,301 bytes**.
All 2,000 learner rows pass the five contracts; original decisions remain intact.
Models remain on native storage; derived proofs and analysis are retained there.

| Evidence | SHA-256 |
| --- | --- |
| Aggregate | `5d5af2abe20f7e827825411be78abeb55ef977659b311fc8fcc838d786df896a` |
| Native proof | `40a5a56b6e8468c9d338bf3b1e6815d6ca3f2e7c47eb35469e5598bb80e9bb15` |
| Retained manifest | `515015c226df66cc3a0403177778e748b928daa05739145a2102cc79bb7b066c` |
| Portable summary | `acd64deeb5a7046e2c54574db3b9f1fc7eb336690c5536125199f4a4f9f48d34` |
| Portable proof | `8144f35a22f3cabecaf575ba79fc12c21178a9224a9db73d449100107c23543b` |
| Heading/gait/support audit | `25e8a4fe2649c788ad8420edb34299f634e462ee63af2c03ffc0e41dbc07a903` |
| Restoration | `8c09ea65f21014c2e4cd3df539bb4351403d014f3cb3a544508a4bdb3b4136aa` |

All three Duck services finish inactive/success/MainPID0. Maximum Duck GPU
1,926 MiB / 69°C, DINO2029 preserved, all protected scopes inactive during
the owned pause. Original user vLLM is restored with GPU engine823303/9,692 MiB
alongside DINO2029/946 MiB; the other three protected scopes stay inactive.
No Duck GPU process remains; no unrelated workload, driver/package, physical
robot, video, publication or harder-skill promotion was changed.

Do not launch another support/width scalar repeat on this rejection. Next work
checks structural learning evidence and a compatible positive locomotion
reference before choosing a new predeclared experiment. Old optimizer state,
finite adaptation and command-to-gait response are hypotheses, not a proven
cause or authority for a blind reset, longer run or acceptance-gate change.
