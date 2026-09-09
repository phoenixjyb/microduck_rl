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
