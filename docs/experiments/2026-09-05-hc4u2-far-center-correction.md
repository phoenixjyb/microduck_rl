# HC4-U2 far-center student-state correction

Date: 2026-09-05

Parent: `67c7cfcbd7437abab0cee410d717887bacaf1c2c`

Decision: **offline candidate passed; fresh seed-251 pre-screen is next**

## Causal target and unchanged boundaries

HC4-U1 passed every seed-193 pre-screen check except per-cell timeout
non-regression at 0.30 m/s, 1.15 m forward, and 0.00 m lateral. It produced one
timeout where HC4-LH produced one collision and no timeout. HC4-U2 changes only
the student-state coverage for that far-center lingering mode. It does not
relax, merge, or reinterpret the failed gate.

The frozen motor-aware actor, HC4-U1 17D compact observation, 64x64 network,
command authority, teacher, optimizer, seed 42, 200 epochs, ten base shards,
box geometry, exact structured perception, and fixed-attempt closed-loop gates
remain unchanged. Interaction speed is still free to slow or brake; approach
and recovery speed must not regress. Raw camera input, joint-target authority,
and physical motion remain excluded.

## Predeclared correction collection

The student checkpoint is HC4-U1 SHA-256
`2196d2ed2dbc3e182fa0b36edf663d11187330d430cd319ceb368c8a28e9753b`.
The frozen locomotion actor SHA-256 is
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.

Three separate CUDA shards use seeds 233, 239, and 241. Each runs only the
0.30 m/s, 1.15 m forward, 0.00 m lateral cell with 64 environments and a
700-step ceiling. HC4-U1 controls the simulated robot while the deterministic
teacher labels the same compact student-reached states. Collection retains
samples only from episodes resolving as clean passage, collision, or attempt
timeout; partial, fall, and NaN episodes are excluded. Each service must set
`CUDA_VISIBLE_DEVICES=0`, run alone, and retain its report and dataset.

Seed 233 is the collection pre-screen. Seeds 239 and 241 run only if it has:

- the exact actor and student checkpoint identities;
- at least 4,000 finite samples from at least 60 resolved episodes;
- zero fall, NaN, and non-finite events;
- matching observation, teacher-command, student-command, episode-key, and
  outcome-code lengths;
- at least one nonzero teacher/student disagreement sample.

The same gates apply independently to seeds 239 and 241. Outcome mix is
descriptive; collision or timeout episodes are intentionally retained for
correction.

## Retained correction shards

All three CUDA services passed the collection gate with exact actor and HC4-U1
student identities, finite matched tensors, and nonzero teacher/student
disagreement. Together they retain 18,825 samples from 192 resolved episodes.

| Seed | Samples | Clean | Collision | Timeout | Dataset SHA-256 | Report SHA-256 |
|---:|---:|---:|---:|---:|---|---|
| 233 | 6,294 | 63 | 0 | 1 | `8bbf3560faeb2758c88b5326b5d35975d57db685d6ad47bde83ad691ac55fb71` | `a3cb048aac374fa709bf770f1b00b6eaf38b1aee0cf1335309e3d749a54df522` |
| 239 | 6,303 | 64 | 0 | 0 | `efa3ceec37ea9e4b9677175b38c696a8452463a76c122ca103c1693c363142fb` | `ff36f1d54e0ee760449358202bc64c3ece104bd9684a9fc1a7fa184d7d85e358` |
| 241 | 6,228 | 64 | 0 | 0 | `1fc67b945436700a1aa9a5fe711188ec80e7f9d1c26e0b15eb478279d5538bfb` | `3f4118375c021cb278002c4cb41b1caaba4888a32e03803935702253aa6af228` |

The trainer input is now exactly 433,255 samples from 5,526 episodes: the ten
ordered HC4-U1 shards followed by correction seeds 233, 239, and 241. It
rejects reordered or hash-mismatched input and verifies that the final three
shards name the exact HC4-U1 student checkpoint.

## Candidate and fresh evaluation gate

The exact trainer contract is now frozen. One seed-42 HC4-U2 candidate may
train with the unchanged offline MAE gates after source tests, a clean exact
branch, idle GPU, and protected-service preflight pass.

## Retained training result

The single seed-42 CUDA fit completed successfully with 433,255 samples from
5,526 episodes. It used 346,103 training and 87,152 validation samples. Epoch
198 minimized validation MSE at 0.00035634. Validation speed MAE was 0.003665
m/s and yaw MAE was 0.009566 rad/s, within the unchanged offline gates; all
reported values were finite.

- Checkpoint SHA-256:
  `ded75258b7a6467ff4460b9441b3a108c42acfcf10cab5d5bc65676ef2648629`.
- Manifest SHA-256:
  `ba8737605d241f1be6dac563860a3224443bb151115b053c2b198e7e2ea4996d`.
- Exact source/trainer-contract commit:
  `d56a23a41c3a9a3b134bf827c2669eb44b6a9e15`.

The offline decision is `offline-imitation-pass`; it does not establish
closed-loop safety. The deterministic fixed-attempt gate now freezes this
checkpoint identity and fresh seed 251 before runtime evaluation.

Closed-loop evidence uses the unchanged HC4-U1 twelve-cell matrix and HC4-E1
fixed-attempt protocol, but entirely fresh seeds: 251 is the pre-screen; 257
and 263 continue only on a complete pass. HC4-R2 remains the paired source at
0.90 m and HC4-LH at 1.15 m. Collision, timeout, clean-passage, approach,
recovery, fixed-denominator, and motor gates are byte-for-byte unchanged from
HC4-U1. Seed 193 is diagnostic history and cannot serve as HC4-U2 acceptance
evidence.

No training, fresh evaluation, MP4, broader geometry, sensor perturbation,
raw-perception work, or physical motion is authorized until the preceding
gate completes and the next exact artifact contract is committed.
