# MicroDuck evening handoff: October 3

Prepared at approximately **19:50 Asia/Shanghai**, before the **20:00** cutoff.
This records reviewed source and retained state, not final cutoff telemetry or
authority for another session. At the cutoff, start no new Duck development or
job; verify the final live state separately and leave services unchanged.

## Exact starting points

- Mac repository: `/Users/yanbo/Projects/microduckPlayground/microduck_rl`, branch
  `feat/athletics-obstacle-curriculum`. Reviewed implementation commit
  `b61eb5ad08e607a2623a56b0f228c095cfed031d` was pushed and independently matched
  to the fork before this documentation-only handoff.
- WSL: `gw98-direct`, repository
  `/home/yanbo/work/microduck_rl-stance-replication-20260930`, same branch,
  **clean frozen `1b96ccaac1326d6f0a1e00cdff8950b91a2916be`**. New transition
  and return helpers are not installed there. Do not interpret this intentional
  source difference as a completed native installation.
- GPU: RTX PRO 4000 Blackwell,
  `GPU-7d72b360-33bc-2cee-3ff4-a954474011b5`. The latest live sample before this
  handoff was 0% utilization, 696 MiB, 30 C, no visible compute PID and no active
  Duck user unit. This is a sample, not continuous GPU or motor telemetry.
- FilmBrain **user** services remain running with original PIDs 521 and 298048,
  zero restarts. Both AI Mission protected services remain inactive in both
  system and user managers. No restore/restart, driver/package change or
  physical motion was performed.

## What this window actually closed

1. [Fixed-input CUDA64 sampler](2026-10-03-cuda64-shadow-sampler-probe.md): native
   two-seed fresh capture/replay pairs match under the declared caller-state
   exclusions. Synthetic zero inputs, no physics, reward, rollout storage or
   update. The whole 32-file / 18,319,761-byte retained inventory was freshly
   rehashed on **both WSL and Mac**, including original journals and failure
   evidence. This is recorded provenance, not CPU replay of CUDA math.
2. [Recovery coverage synthesis](2026-10-03-recovery-catalog-coverage.md): 46
   retained complete attempts cover all 45 catalog cells, only zero duplicated;
   all nine gates pass in each. One deterministic evaluation seed, not a new
   experiment or randomized multi-seed recovery promotion. Keep D1 unchanged.
3. [Transition collector](2026-10-03-cuda64-transition-source-contract.md) and
   [finite-return helper](2026-10-03-cuda64-finite-returns-source-contract.md):
   reviewed source-only siblings with fault-latched refusal, raw-action storage,
   exactly-once timeout bootstrap, store-before-reset ordering, and finite GAE
   without NaN repair. The final 26-file CUDA-hidden regression passed **737
   tests in 78.63 seconds**; owner focused rerun **58 in 8.66 seconds**. Ruff,
   compilation, links and whitespace checks passed. No native collector/GAE,
   optimizer, new checkpoint or learned capability follows from these tests.

The original failed source-sync service remains failed with PID 0, no restarts
and exit status 1. Its missing-working-directory refusal is retained unchanged;
the separately declared corrected source-sync succeeded. Never reset, retry or
relabel the original failure to simplify the ledger.

The three original iteration-255 exports for training seeds 577/587/593 remain
present at 256,368 bytes each. Their original training reports were freshly
rehashed against the previously declared report pins. The selected D1 seed-577
export still matches
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`.
These checks did not rerun whole training authentication or the historical
[4,608-attempt nominal evaluation](2026-10-03-portable-packed-full-evaluation.md).

## Next bounded implementation, before any training

1. Implement a distinct native CUDA64 simulation **no-update** supervisor and retained
   record scorer around the new collector. Reauthenticate exact clean source,
   historical provenance, D1 construction, fresh native preparation and sampling.
   Declare the fixed 64-row schedule and capture actual controls/force phases,
   storage and private/caller RNG boundaries. Require whole-byte retention,
   independent native-profile replay, one shared leased workload, two idle
   samples and unchanged protected/FilmBrain states.
2. Predeclare measured resource/raw-size and full closeout budgets before a
   launch. Existing CUDA2 physics and fixed-input CUDA64 sampler timing do not
   establish CUDA64 physics-collector throughput. This handoff declares **no
   launch cap or execution admission**.
3. Keep natural terminal/storage/reset qualification separate. A fresh 28-call
   trace reaches at most 280 of 2500 episode substeps and cannot observe the
   ordinary timeout. Do not inject clocks or terminals. Common reset does not
   prove untouched-sibling selective reset. Qualify native finite-return replay
   separately; do not borrow synthetic tests as device evidence.
4. Before implementing or admitting a native optimizer diagnostic, retain and
   independently replicate a genuine new unchanged-parent deficit under an
   explicit initial-state/plant/terrain contract. Do not train passing catalog
   cells or resample held-out cells into training. Per-attempt curriculum
   assignment is not supported by the current fixed row map.

Nominal stance replication, fixed push baselines and sampler wiring are distinct
achievements. Hopping remains rejected under its retained full gates; historical
obstacle passage does not imply accepted post-pass speed recovery; rolling
football B2–B6 still require B0 feasibility and held-out B1 qualification.
All integration and physical-admission flags remain unchanged. There is no real
Duck available, calibrated motor thermal claim, raw-perception learning or
combined skill-retention result.
