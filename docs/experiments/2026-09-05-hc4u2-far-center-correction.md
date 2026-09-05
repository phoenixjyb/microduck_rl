# HC4-U2 far-center student-state correction

Date: 2026-09-05

Parent: `67c7cfcbd7437abab0cee410d717887bacaf1c2c`

Decision: **predeclared collection only; candidate not yet trained**

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

## Candidate and fresh evaluation gate

Only after all three shard hashes are retained may the trainer contract be
updated to require the exact ordered HC4-U1 ten-shard base followed by
correction seeds 233/239/241. One seed-42 HC4-U2 candidate may then train with
the unchanged offline MAE gates.

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
