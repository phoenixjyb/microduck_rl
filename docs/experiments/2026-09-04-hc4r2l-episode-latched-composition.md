# HC4-R2L episode-latched composition

Date: 2026-09-04

Implementation parent: `048c327cc5c6e456bd994b6e75b1e139865b5fc9`

Decision: **do not promote; causal gate narrowly missed; HC4-LH remains
selected**

## Single changed axis

HC4-R2H's instantaneous range gate added one collision against its paired
source-controller baselines. HC4-R2L changes only selector lifetime. On the
first policy observation of each environment episode, it chooses:

- HC4-R2 when structured geometry is valid, reconstructed route-forward range
  is at most 0.95 m, and nominal speed is at most 0.40 m/s;
- HC4-LH otherwise, including initially invalid geometry.

That decision is retained for the complete episode. Range, heading, validity,
and speed changes cannot switch networks mid-maneuver. The rollout execution
layer supplies an explicit per-environment done mask; only that reset clears
the latch. The next observation then chooses for the new episode.

The embedded HC4-LH and HC4-R2 weights, 0.02 m lateral gate, 0.95 m range gate,
0.40 m/s speed gate, frozen locomotion actor, command limits, and exact
structured-geometry boundary do not change. Raw camera perception and physical
motion remain outside scope.

## Fail-closed implementation contract

- Checkpoint creation still rejects mismatched locomotion hashes, model
  configurations, source authority, or missing HC4-LH composition state.
- The loader rejects an HC4-R2L stage without the exact
  `latched-until-explicit-episode-reset` selector declaration.
- A batch-size or device change initializes a new conservative latch state.
- Initially invalid geometry selects HC4-LH for the episode; the existing
  execution layer independently commands immediate stop outside recovery.
- A reset mask with the wrong batch shape is rejected.

Focused tests must prove that changing instantaneous eligibility cannot alter a
latched route, that resets affect only named environments, and that checkpoint
composition plus runtime loading reconstruct the stateful wrapper.

## Predeclared evidence sequence

### A. Causal regression check

First rerun the exact HC4-R2H matrix at speeds 0.30/0.40 m/s, ranges 0.90/1.15
m, lateral positions -0.08/0.00/+0.08 m, and seeds 163/167/173. This is a
diagnostic reuse of already-seen seeds, not fresh promotion evidence.

Because the route is latched from initial geometry, the 0.90 m cells should
behave as HC4-R2 for the whole episode and the 1.15 m cells as HC4-LH. Continue
only if there are zero hard failures and rated-speed exceedances, no per-cell
collision increase, at most two aggregate collisions, at most eleven timeouts,
clean-pass rate at least 99.501%, and maximum per-cell torque-utilization p99 no
greater than 0.60.

### B. Fresh acceptance

If A passes, use fresh seeds 179, 181, and 191 on the same 36 cells, paired with
HC4-R2 at 0.90 m and HC4-LH at 1.15 m. Promotion requires:

- zero falls, NaN terminations, non-finite steps, and rated-speed exceedances;
- no per-cell or aggregate collision increase against the paired source;
- no timeout increase and no lower pooled clean-pass rate;
- maximum per-cell torque-utilization p99 no greater than 0.60;
- retained approach and recovery tracking; interaction speed remains
  descriptive rather than a nominal-speed gate.

Only after B passes may a centered 0.94/0.96 m threshold diagnostic be run on
the same fresh seeds. No MP4 is recorded before all numerical gates pass. A
passing result remains exact-geometry simulation evidence and does not
authorize camera perception, dynamic obstacles, or physical motion.

## Retained checkpoint and causal result

The canonical simulation checkpoint is
`artifacts/checkpoints/hc4r2l-d71c076/supervisor.pt`, SHA-256
`9973079826dd246e0326b16faead5959fa072ed84b6e136851f75ebe04a083f6`.
Its manifest SHA-256 is
`85c64d493a534b40e98c2bdf3d5dd4cfa21570e2ccac42da5ae8011c5cc4631c`.
The repository suite passed 612 tests. An actual-checkpoint CPU check confirmed
byte-exact HC4-R2/HC4-LH output selection across range changes and correct
reseating after explicit reset.

The 36-cell causal matrix completed successfully:

| Controller | Clean | Collision | Timeout | Resolved | Clean rate | Weighted passage | Max torque p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC4-R2L | 2,588 | 2 | 11 | 2,601 | 99.5002% | 8.0807 s | 0.5642 |
| Paired source baselines | 2,591 | 2 | 11 | 2,604 | 99.5008% | 8.0657 s | 0.5656 |

HC4-R2L matched the aggregate collision and timeout counts and had zero falls,
NaN terminations, non-finite steps, or rated motor-speed exceedances. It
removed HC4-R2H's added collision. However, its pooled clean-pass rate was
0.00058 percentage points below the paired baseline and below the fixed
99.501% continuation floor; weighted passage was 0.0150 seconds slower. The
distribution of adverse events also varied by seed/cell even though aggregate
counts matched.

The retained causal report SHA-256 is
`42eb472ed5349fc25661eadbe9559e57adb33ce4a107051f9b3080fdb9edfc9c`.
Because the predeclared continuation gate was not met, fresh seeds 179/181/191,
the 0.94/0.96 m threshold diagnostic, and MP4 recording were not run.

The tiny difference may reflect fixed-step episode-count sensitivity or GPU
simulation nondeterminism, but that is an inference rather than acceptance
evidence. Any future composition protocol should predeclare equal resolved
attempt counts per cell before another candidate is evaluated. This candidate
is not rerun or promoted post hoc.
