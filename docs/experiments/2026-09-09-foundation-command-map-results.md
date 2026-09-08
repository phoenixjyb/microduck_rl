# Frozen foundation command map: completed, no admitted band

Follow-up: [CPU motion decomposition](2026-09-09-foundation-motion-decomposition.md)
is complete; the original performance decision is unchanged. Football development
has reached a separate [B0 geometry fixture](2026-09-09-football-b0-contact-fixture.md),
not a stance or free-ball balance result.

The single predeclared map completed on100.100 from September9 00:13:05 to
00:19:38 Asia/Shanghai at source `ddca61c484812aa64f83e00fca9e81a92b6fabc8`.
Service `microduck-foundation-map-v1-ddca61c.service` exited0, with its independent
40-minute cap and control-group cleanup verified. No optimizer ran and no new
checkpoint was produced. This closes the [map](2026-09-08-foundation-rethink.md);
do not repeat it or reinterpret these development seeds as confirmation data.

## Numerical outcome

All18 cells /144 first episodes completed: seeds503,509,521, both fixed actors,
three commands, eight environments per cell,400 control steps at50 Hz. That is
7200 batched control steps /57,600 lane samples /1152 simulated lane-seconds,
not7200 physics substeps. The first100 steps were startup; statistics below use
the remaining300. Each table entry pools24 per-environment means across seeds.

| Command m/s | Original mean [environment range] | Narrow mean [environment range] |
| --- | --- | --- |
| 0.10 | 0.0834 [0.0542,0.1050] | 0.1222 [0.1094,0.1431] |
| 0.20 | 0.1452 [0.1306,0.1663] | 0.1832 [0.1693,0.2033] |
| 0.30 | 0.2086 [0.1941,0.2286] | 0.2677 [0.2496,0.2814] |

These are achieved route-forward speeds, **not relabeled successful commands**.
All18 cells missed performance criteria. No bin passes the complete descriptive
checks across all three development seeds, so there is no candidate admitted
practice band or obstacle-controller promotion. Only3 of144 environments met the
declared half-second stable-speed response within its deadline;141 missed it.
All cells also failed the lateral-motion criterion. The narrower-reward actor
shifts speed upward, improving average undertracking at.20/.30 but overshooting
.10; average improvement does not resolve spread, settling or lateral motion.

There were no sampled terminals, nonfinite traces or rated-speed exceedances,
and no safety-triggered prefix stop. Maximum cell pooled legacy torque p99 was
0.5692 against the unchanged0.60 gate. The largest named-joint pre-reset p99 was
0.6916 at the right knee in the narrow.30 group. These are normalized to the
declared0.6 Nm reference, not Nm values or calibrated thermal safety claims.
Named-joint exposure and power/load data remain in every raw cell report.

Mean absolute lateral speed ranges from0.0718 to0.0777 m/s across the six groups.
Signed per-environment lateral means span -0.02635 to+0.05670 m/s overall.
The difference suggests oscillatory sway contributes to the unsigned measure;
it does not prove that sway is harmless, remove drift, or justify changing gates.
The next diagnosis should distinguish signed drift, oscillatory motion and
forward bias using these retained traces before choosing another learning change.

## Evidence and scope

All18 cell manifests were rehashed, their journals reconciled against retained
tensors and their scores recomputed on the remote CPU after completion. The
deterministic ordered summary exactly matches the retained campaign result.
All nine original/narrow pairs have identical recorded initial qpos/qvel,
encoder bias, selected randomized model fields and exported RNG states. This
supports the paired starting-condition comparison; Warp/internal-state coverage
is incomplete and trajectory-level determinism is not claimed.

The exact source/checkpoint pins and292 selected dependency Python files passed
pre/post verification. This is not complete binary/asset/runtime equivalence.
Remote CPU validation before launch:352 passed,2 optional Mac-path checkpoint
tests skipped; both real actors had separately passed strict CPU restoration.
The final runtime-initializer change also passed66 campaign tests on100.100.

Remote raw directory: `artifacts/evaluations/foundation-command-map-v1`.
The Mac mirror at `artifacts/diagnostics/foundation-command-map-v1` has all18
cell file manifests independently hash-checked. Keep both retained, not in Git.

- Campaign result SHA256:
  `8dbc0f8fa5f41a3e35764b1e335003033f5cb72811fccc61bd14781efd7cce5d`.
- Retained campaign plan SHA256:
  `237ed6f32a6c8c174dabddaaaa8180a84d3d91e657ebfbafec67cd8bc15455b8`.
- Reviewed external launch plan SHA256:
  `efd26cd987f3aa9fca432de81ec6981a0466a3439b1858a8f3887b09b800e1cb`.
- Authoritative derived closeout:
  `artifacts/evaluations/foundation-command-map-v1-closeout-v2.json`, SHA256
  `7870ae8968289498618a730eab04f4b87543cf39533fbe342e565a2d30cf6750`.

The earlier closeout remains retained and linked by hash: its `physics_steps`
label incorrectly counted control batches. V2 only corrects those count labels;
all numerical outcomes, raw evidence and decisions are unchanged. A first
read-only closeout attempt lacked the user-bus environment for its service query;
the query was repeated with the explicit user runtime directory, not by rerunning
the GPU experiment. Earlier native CPU adapter failures likewise remain separate.

Closeout showed MainPID0, service active/exited with Result=success, no compute
PID, two0%-utilization GPU samples at44 C/12 MiB, and both protected services
inactive. Active/exited is a retained completed service, not running training.
No protected service or unrelated workload was changed.

## Next bounded chunks within the existing work window

1. CPU-only decomposition of the immutable map: report signed drift, velocity
   oscillation, forward bias, named-joint loads and their timing. Any smoothed
   signal is a separate explanatory diagnostic, never rescored acceptance.
2. Continue [football B0](2026-09-08-football-balance-curriculum.md): build an
   explicitly named full-collision contact fixture and verify actual foot/ball
   distances, joint reach, support normals and forbidden contacts. A static
   geometric fit still does not pass stance, load or free-ball dynamics.
3. Use that evidence to predeclare one smallest training lesson, with exact
   mechanics/observations/parent/restore/reward/seeds/budget/checkpoints, numerical
   abort and retention gates. No optimizer until the declaration and CPU tests
   are complete and the job plus closeout fits before07:30. Do not advance
   obstacle RL on the failed map or claim rejected H1-T hopping is retained.

The overnight authorization still ends September9 07:30 Asia/Shanghai.
No raw-camera RL, physical motion,100.98 work or protected-service restoration.

Closeout validation:355 focused local CPU/regression tests passed in18.56 s
with CUDA hidden. The result and two linked curriculum documents were rendered
to HTML and their tables/local links checked programmatically; no screenshot
review, MP4 or additional GPU rollout was performed during documentation closeout.
