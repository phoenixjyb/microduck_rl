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
