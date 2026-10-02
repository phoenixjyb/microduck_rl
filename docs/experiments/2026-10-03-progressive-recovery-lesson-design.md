# Next recovery lesson: source-only progressive design

This is **not a GPU-job predeclaration or an available learner**. It prepares
the next bounded work after the [D1-M frozen baseline](2026-10-03-frozen-five-direction-recovery-baseline.md)
closes. The D1-M source, checkpoint and runtime are unchanged: its retained fixed
five-case baseline has now passed. That does not qualify the proposed held-out
or stronger-push cells. No long training job may
borrow this document as execution admission before the October 3 08:00 cutoff.

## Learn only where the frozen policy has a measured deficit

First implement and qualify a separate finite schedule adapter/evaluator, then
screen the unchanged seed-577 / iteration-255 actor on the proposed timing and
dose cells. If it already passes, do not spend a training campaign relearning
those cells: retain the baseline and choose the next smallest evidenced deficit.
If it fails, preserve the failed first attempts and gate evidence before
predeclaring a weight-initialized recovery lesson. Training is a separate policy
specialist first, not a claim of one actor preserving walking/obstacles/hopping.

The pure [lesson plan module](../../src/mjlab_microduck/stance_recovery_lesson_plan.py)
proposes these bounded stages, never force-installation permission:

| Stage | Training pulse cells | Held-out additions |
| --- | --- | --- |
| gentle | 2 N / 20 ms at 1.0 s, four cardinal directions, plus zero control | Four float32-specified diagonal directions at 0.75 and 1.25 s |
| timing | Same gentle dose at 0.5 / 1.0 / 1.5 s, plus zero control | Same eight unseen diagonal/time cells |
| dose | Retain gentle cells; add 2 N / 40 ms and 4 N / 20 ms at the three training onsets | Same unseen gentle diagonal/time cells plus earlier stage retention |

The two added doses both have 0.080 N s impulse, but different peaks/durations;
measure them separately rather than equating their feedback/contact responses.
All pulses stay world-horizontal, at the actual trunk inertial COM, zero torque;
clear both complete applied-force arrays before the fresh unforced post-solve.
Keep original motor, height, tilt, support, joint-limit and first-terminal gates.
No camera perception, actor-observation expansion or thermal-model claim.

Each stage proposes 20% zero-command/no-push attempts and 80% uniformly sampled
explicit nonzero cells. This preserves quiet standing exposure during training.
Evaluation retains one full 5-second first attempt per world; no reset followed
by a successful recovery replaces a failure. Training may selectively reset only
after storing the transition and immutable first-terminal evidence.

Training seeds **653 / 659**, held-out seeds **671 / 677 / 683**, 64 training
worlds and 256 updates per stage are **proposed, not budget-qualified**. These
seed literals were checked against source/test/Markdown seed declarations before
this proposal. Seed separation is provenance, not proof of randomized state
coverage: an unchanged deterministic reset can produce identical states across
evaluation seeds. The unseen direction/onset cells supply the explicit held-out
variation; later initial-state/plant uncertainty needs its own measured bounds.

## Common windows and retention

Proposed common checkpoint iterations are **64 / 128 / 192 / 255**. Require both
training seeds and all three evaluation seeds/cells at **two adjacent common
checkpoints**—64+128, 128+192 or 192+255—not two repeats of one lucky checkpoint.
Each row has 128 complete first attempts, all passing unchanged numerical gates
and complete force checks. A missing/duplicate/partial row, changed checkpoint
or failed earlier cell blocks progression. Gentle/timing/dose held-out catalogs
contain 13 / 21 / 45 cells respectively, including zero and prior pulse cells.
This is a full qualification matrix, not a cheap throughput estimate.

The pure `promotion` function validates this proposed matrix schema only. Its
`raw_provenance_enforced_by_this_pure_screen=false` is deliberate: caller JSON
assertions do not authenticate captures. A future executor must hash raw bytes,
strictly restore the exact student weights, independently replay every action,
plant/control/force/terminal record and match each pre-push prefix to the
appropriate zero control before supplying receipts. Every admission flag remains
false in this source-only design, even when synthetic proposal rows pass.

Do not relax the displacement/settling/motor limits to make a stage graduate.
Retain zero-push performance and every previous push cell at each promotion.
Obstacle speed/route recovery and hopping are separate ledgers; H1-T remains
rejected. Football B2–B6 remain behind full B0 contact/motor feasibility and
held-out B1 stance/recovery, with rigid/sprung mechanical compatibility reviewed.

## Required implementation boundaries before any new job

The existing `LeanStanceLearner` starts from eager `model_127` with SHA256
`46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93`.
It does **not** accept the selected D1 `model_255`, SHA256
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`.
Matching 44D actor / 50D critic / 10 actions is not parent-identity compatibility.
Prepare a distinct hash-first loader for both parent model groups, install them
into the actual policy/optimizer modules with fresh empty Adam state, and retain
the exact initial state hash. No optimizer/storage/simulator/RNG/normalizer resume.
Do not widen the old parent or fresh-seed allowlists to disguise the new purpose.

The current D1 adapter only permits one-to-five unique cases and its trace only
one fresh world; its checkpoint/schedule binding is deliberately frozen. A
64-world learner therefore needs a **new explicit per-row schedule namespace**
and a CPU bridge calling that named pulse API, never patching nominal `step` or
its applied-force fault guard. The student checkpoint and replay protocol must
also be separate; never relabel new weights as the original D1 iteration 255.

Before a future GPU job: qualify exact source and actual portable CPU execution,
frozen parent installation and real CPU pulse/terminal/reset integration; retain
an actual bounded CUDA throughput/optimizer/capture probe; derive a service,
memory and closeout budget from those complete measurements; predeclare its
training/evaluation checkpoints and gates; verify clean exact source, the shared
lease, two idle samples and unchanged FilmBrain/protected services. Do not infer
64-world training throughput from D1-M's five sequential one-world evaluations.
Nothing here permits a job extending past the authorized cutoff or physical motion.

## Source checks

The focused pure-plan suite passed **68 tests in 0.11 s** on the Mac with CUDA
hidden. Combined with the unchanged D1/D0/runtime/portable-profile suites, the
integrated check passed **471 tests in 38.29 s**. Independent read-only review
found no concrete blocker in the finite force catalog, cumulative stage
retention, held-out semantics or fail-closed promotion schema. These checks are
source/math/schema evidence only, not real learner, force-adapter, randomized
recovery or GPU qualification.
