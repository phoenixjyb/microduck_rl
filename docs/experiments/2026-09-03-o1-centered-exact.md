# O1 centered exact campaign, 2026-09-03

## Outcome

Decision: **rejected, simulation-only**. No checkpoint was promoted and no
physical motion was authorized.

The campaign used source `b43a482`, parent checkpoint SHA-256
`a4be40e798377f9aeca0ed9bb40946c1f57bcef6ff7ea70b257e509e38be422a`,
training seeds 42–44, and held-out evaluation seeds 41–43. Each training seed
ran 64 iterations with common retained checkpoints.

| Iteration | Pooled clean-pass rate | O1 obstacle gate |
| ---: | ---: | --- |
| 8000 | 7.580% | fail |
| 8016 | 11.707% | fail |
| 8032 | 13.747% | fail |
| 8048 | 11.941% | fail |
| 8061 | 14.606% | fail |

Every candidate had zero falls and zero NaN/non-finite events. At iteration
8061, the per-training-seed clean-pass rates were 15.913%, 11.272%, and 17.440%.
The best pooled result remained far below the 75% campaign gate.

## Diagnosis

Observed training traces ended with route speeds around 0.076–0.090 m/s and
lateral excursions around 0.466–0.505 m while mean training reward continued to
rise. Successful held-out passages were also often too wide or too slow.

Inference: the task permits a locally rewarding side-step-and-linger behavior.
This is not a numerical-instability failure and is not evidence that the motor
baseline failed. The campaign skipped the easier offset-assisted bypass stages
needed to teach early commitment and route return.

## Retained evidence

- Sweep campaign: `o1-centered-exact-b43a482-three-seed-20260903`
- Acceptance campaign: `o1-centered-exact-b43a482-acceptance-20260903`
- Selected checkpoint: none
- Motor/action acceptance: intentionally unverified because no obstacle
  checkpoint survived
- MP4: intentionally not recorded because no policy was accepted

The initial multi-seed shell launcher expanded its seed variable before the
remote shell ran. Seed 42 nevertheless completed and was retained with an
explicit seed manifest and verified hashes; seeds 43 and 44 then completed in
separate, explicit services. No checkpoint data was overwritten or lost.
