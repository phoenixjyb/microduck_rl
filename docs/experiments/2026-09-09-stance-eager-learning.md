# B1-N: bounded eager learning, separate from the full pilot

The user renewed one next chunk after the September 9 14:00 unattended window
ended and asked when training could proceed. This declares one short learning
diagnostic using the already exercised eager runtime. It does not reinstate an
overnight campaign. Graph optimization remains a separate investigation, not a
prerequisite for every diagnostic optimizer run.

## Question and fixed experiment

Can a longer fresh run produce finite PPO updates and a useful training trend
within a short GPU budget? This is not a claim that stance has been learned.

- Fresh seed **563**, **64 worlds**, **128 updates**, **24 policy ticks/update**.
- No parent, resume or warm start; seed-523 smoke exports are not reused.
- CPU actor/critic/Adam with unchanged architecture, PPO math, reward, 44D actor
  observations and 50D critic observations. CUDA0 eager full-collision physics;
  forward graphs remain disabled.
- Unchanged BAM motor behavior, action slew and delay, support-loss, fall,
  forbidden-contact, torque, velocity, hard-limit and finite-state stops.
- Retain fresh weights and every completed update (initial plus 0 through 127),
  all 3,072 tick records, optimizer receipts, source/runtime/launch hashes and a
  final supervisor file-hash inventory. Exclusive, fsynced writes; no overwrite.
- Distinct checkpoint purpose `eager-learning`. The existing pilot evaluator
  rejects these exports. No optimizer/simulator resume files are produced.

The initial and final 16-update blocks may be summarized descriptively: reward,
optimizer losses, executed steps, termination causes, duration of **terminated**
episodes, timeouts and motor exposure. Do not select a best checkpoint or treat
autoreset training statistics as frozen-policy performance. Different seeds and
budgets prevent a causal comparison with the old 16-update smoke.

The original seed-521, 512-world/512-update pilot, 30-minute cap, checkpoints
128/256/384/511 and its held-out numerical criteria remain unchanged. This run
does not replace that pilot or relax its acceptance gates. No existing gait,
hop or obstacle policy is overwritten; football balance is not yet trained.

## Timing, ownership and stop boundary

The retained 64-world smoke took 90.215 seconds for 16 complete collection/
optimizer/evidence updates. Straight-line scaling gives **721.72 seconds
(12.03 minutes)** for 128 updates. This is an estimate from a short, different-seed
run, not a promise or an actual 128-update timing measurement. Initialization,
closeout, changed episode lengths and shared CPU load can alter the duration.

The child uses the existing independent **900-second watchdog** and its own
per-tick/update deadline. The exact user service must have **960-second
RuntimeMaxSec** and `KillMode=control-group`. An additional **600-second closeout
reserve** is required. A separately supplied absolute launch deadline must leave
more than 26 minutes and no more than one hour at prepare/supervise/child entry.
Do not derive renewed authority from an old campaign cutoff. There is no retry,
extension, automatic follow-on, video or physical-motion mode.

Before GPU allocation: focused tests, exact-source Linux CPU regression, clean
feature branch, frozen dependencies/assets, compiled plant and independently
retained launch SHA must pass. The parent owns the shared GPU0 lease; the child
checks the inherited lock. Check GPU idleness and exclusive compute ownership,
temperature below 80 C, numerical/backend warnings and unchanged live inputs.
Both protected AI Mission system services must remain inactive. Do not stop or
restart unrelated workloads, touch 100.98, or restore protected services.

On timeout or failure, preserve the durable completed prefix and diagnose
read-only; do not resume the partial experiment. On success, verify every
export, update receipt and tick inventory, mirror the hashes and evidence, and
report only `eager-learning-complete-not-capability`. No held-out acceptance,
repeatability tolerance, graph equivalence or real motor safety is inferred.

## Next performance gate

After this bounded run, review the numerical training evidence before proposing
another budget. A separate, predeclared frozen-policy evaluation is required
before making any learning-quality claim; its acceptance thresholds must not
be weakened to fit these diagnostic weights. Qualified nominal stance would
still precede disturbance recovery, fixed-ball support and rolling-football
balance. Existing locomotion/hopping retention remains a separate check.

The focused local learner/checkpoint/legacy-smoke selection passed **77 tests in
72.31 seconds**, including 128 actual CPU optimizer updates against an explicitly
synthetic environment, strict purpose rejection, export verification and failed
supervisor closeout. This is source/CPU evidence, not CUDA or learning-quality
evidence. Exact-source Linux regression and the retained GPU run remain pending.
