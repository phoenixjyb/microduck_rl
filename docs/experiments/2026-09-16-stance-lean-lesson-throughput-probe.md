# B1-N: lean-lesson throughput probe — sizing the watchdog from measurement

Protocol `football-b1n-lean-lesson-throughput-v1`. This is a **predeclaration**.
Nothing in this document has run. It exists because the
[lean-lesson predeclaration](2026-09-16-stance-lean-lesson.md) forbids sizing the
watchdog and service cap from an estimate, and because the estimate it gives is
known to be wrong in the optimistic direction.

It is a **timing probe, not a training run**. It admits no checkpoint, produces no
policy, and establishes no capability.

## Why an estimate is not good enough

The parent eager run completed 128 updates in 735.963 s of child time at 64
worlds — **5.7497 s per update**. Straight-line scaling to 256 updates gives
1,471.9 s (24.53 min).

That number is biased **low**, for a reason visible in the parent's own evidence:
its average includes early updates whose episodes terminated at roughly 1.16 s.
A weight-initialized policy starts from `model_127.pt`, which already reaches the
full five-second endpoint on every attempt. So the real run spends close to the
full 250-tick episode budget from update 0 onward, and every update costs more
than the parent's average.

Sizing a stop boundary from a number known to be too small is how a run gets
killed mid-flight or silently truncated. So the cap must come from measurement.

## What is measured, and where

The cost of one update has two components on two different devices. They are
measured separately, on the **same host**, and reported separately so neither can
hide inside the other.

**Component A — collection (CUDA0, the dominant term).**
64 worlds, 24 policy ticks per update, real eager full-collision physics, forward
graphs disabled. The policy is the **pinned frozen parent export**
(`model_127.pt`, SHA256 `46cd52b5…`), evaluated deterministically at every tick.
No optimizer is constructed and no weight is written: this is the same "never an
optimizer" shape as the earlier source-bound probes. Using the parent policy
rather than random actions is deliberate — it is what makes episode lengths match
the real run, which is exactly the quantity the parent's average got wrong.

**Component B — optimizer (CPU, the small term).**
The same `LeanStanceLearner` PPO update on 64x24 samples, timed on the host CPU
with CUDA never initialized. Measured on the host rather than on a development
machine because the two CPUs differ.

Two development-host measurements of this term, taken on different runs:

| Run | mean | max | share of the parent's 5.7497 s |
| --- | --- | --- | --- |
| first | 0.1164 s | 0.1758 s | 2.024% at the mean |
| second | 0.4565 s | 0.9801 s | 7.939% at the mean |

A third run exposed *why the warm-up matters*, and changed the plan below. Its
per-update series was:

```
2.0236  0.4876  0.6353  0.4684  0.2965  0.2846  0.2874  0.3979
```

The **first measured** update cost 2.02 s against a steady state of roughly
0.29 s. One discarded warm-up update had not absorbed the first-touch cost, so
that single outlier would have inflated the derived service cap from about
1,800 s to 2,657 s — a watchdog sized to a one-off transient rather than to the
run. The declared plan therefore discards **two** warm-up updates, and the
steady-state samples begin after both.

These figures are recorded as **expectations and as a warning about variance**,
not as the value used to set the cap: the host's own measurement is used, and it
is taken with the same `max`-based rule over the post-warm-up samples.

## Declared measurement plan

- **8 measured updates**, after **2 discarded warm-up updates**. The warm-up covers
  first-touch JIT, allocator growth and plant compilation, which are real but are
  not per-update steady-state cost. Two, not one, because a measurement showed the
  first post-warm-up update still carrying a 2.02 s transient against a ~0.29 s
  steady state.
- Every tick is timed individually and retained, so the per-tick distribution is
  available and not just a per-update total.
- Both components report **mean, median, p95 and max**, and the raw per-update
  series is retained. A single aggregate is not sufficient evidence.
- The measured update count is small on purpose: the probe must be cheap enough
  to sit inside one bounded window, and it must not become a training run by
  stealth.

## Declared cap rule, fixed before measuring

Let `C` be the maximum per-update collection seconds and `O` the maximum
per-update optimizer seconds over the measured (non-warm-up) updates, and let `S`
be the measured process start-up seconds up to and including the first reset.

```
predicted_child_seconds = ceil(256 * (C + O)) + S
service_seconds         = ceil(1.25 * predicted_child_seconds)
child_seconds           = service_seconds - 60
```

The 1.25 factor is a declared safety margin, not a fitted quantity. `child_seconds`
sits strictly inside `service_seconds` so the independent watchdog always fires
before the service kill. The **600 s closeout reserve is unchanged** and sits
outside both.

## Declared window and refusal rules

The probe's own caps are **fixed at 900 s child and 960 s service**, reusing the
proven eager-learning pair. They are not the numbers this probe exists to
produce — the caps it produces are for the 256-update lean-lesson run. The 900 s
bound comes from the fixed `supervised_stance_smoke` wrapper; `supervised_process`
is unusable here because it enforces `timeout <= CELL_SECONDS` (120 s), far too
short for a nine-update probe.

A **new explicit absolute launch window** is required. Authority must not be
derived from any expired cutoff: the `CUTOFF` value baked into the older CUDA
probe (`2026-09-09 23:30 UTC`) is expired and must not be reused. The window is
passed in at launch as an absolute Unix deadline; it is not written into this
document, because a date fixed in advance is not a window anyone can honour.

The probe **refuses to launch** unless all of the following hold:

1. The remaining window exceeds `service_seconds + 600 + 60` at launch, and is at
   most 60 minutes. For this probe `service_seconds` is its own fixed **960 s**, so
   the floor is **1,620 s (27 minutes)**.
2. The GPU is idle, no competing compute owner holds the lease, and the
   temperature is below 80 C.
3. `stance_cuda_probe`'s pins all match: machine ID
   `0c79e415429b4933a400159bfa79a34d`, driver `595.84`, GPU UUID
   `GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd`, branch
   `feat/athletics-obstacle-curriculum`, a clean tree, and HEAD equal to the
   declared source.
4. CUDA is hidden in the supervising process and never initialized there.

If the measured throughput does not fit the remaining window with the declared
margin, then **the lean-lesson budget is reduced or the run is not launched**. The
budget is never exceeded by weakening the stop, and `service_seconds` is never
raised to fit a budget that does not fit.

## What a pass and a failure mean

- **Pass:** the measured caps are recorded and become the lean-lesson run's
  `child_seconds` and `service_seconds`. The lean-lesson predeclaration is
  amended with those two numbers and nothing else.
- **Failure:** the probe is reported as failed and the lean-lesson run is not
  launched. No partial measurement is used to size anything, and the probe is not
  retried with a relaxed margin.

Neither outcome admits a checkpoint. Neither outcome says anything about tilt,
stance quality, or the 0.0873 rad gate, which remains untouched.

## Implementation status

Records implementation and validation only. No threshold, gate or cap rule above
is changed.

Implemented at `1e10cb7d` (`stance_lean_throughput`): the declared plan, the
two-component measurement, the nearest-rank summarizer, the cap rule, and the
window refusal rules.

Validation:

- Local focused CPU regression: **200 passed**, 2 deselected. The two deselected
  tests are the pre-existing eager wall-clock-budget errors that reproduce on
  unmodified source; they are unrelated to this work. Eight of the passing tests
  are new.
- **Exact-source Linux CPU regression: 618 passed in 92.73 s** at `1e10cb7d` on
  100.100 with `CUDA_VISIBLE_DEVICES` empty and CUDA never initialized.

**The probe itself has not been run.** It requires the GPU host and a fresh
declared window. Until it runs, the lean-lesson `child_seconds` and
`service_seconds` remain undeclared, and the lean-lesson run is not launched.

## What this does not authorize

No physical motion. No claim of learned stance, football balance, disturbance
recovery, or motor safety. `checkpoint_admitted`, `learned_stance_accepted`,
`football_balance_accepted` and `physical_motion_authorized` remain false
throughout. No checkpoint is exported, admitted, or retained by this probe. No MP4
is recorded. No gait, hop or obstacle policy or its evidence is modified or
re-adjudicated.

This document is simulation-only, and it is a declaration, not a result.
