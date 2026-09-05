# HC4-U1 unified range/lateral supervisor

Date: 2026-09-05

Parent: `8fd5e07274ae65a43f26be6120eee24378667ea2`

Decision: **reject at the seed-193 pre-screen; do not run seeds 199/229**

## Purpose and single changed axis

HC4-LH is accepted over its farther-range exact-geometry envelope, and HC4-R2
is accepted as a near-range low-speed specialist. Stateless and episode-latched
runtime selection did not pass. HC4-U1 therefore trains one supervisor on the
union of the retained far/lateral teacher data and near-range teacher plus
student-state correction data. There is no selector and no runtime switch.

The 17D compact observation, 64x64 network, optimizer, seed 42, 200 epochs,
batch size 1,024, frozen motor-aware actor, command bounds, box geometry, and
exact structured perception remain unchanged. The network still emits only a
forward-speed scale and yaw-rate command. It cannot emit joint targets. Raw
camera perception and physical motion remain outside scope.

## Immutable ordered training set

The order is frozen because it affects episode-key namespacing, validation
splitting, and optimization. The trainer rejects missing, extra, reordered, or
hash-mismatched input.

| Order | Coverage | Samples | Episodes | SHA-256 |
|---:|---|---:|---:|---|
| 1 | HC2 0.30 x 1.15 | 18,613 | 190 | `660cdfa8b618f8af425baf0e2f9c3d7b01d59eab93fca24848c9a82a84408467` |
| 2 | HC2 0.50 x 1.15/1.40 | 34,234 | 385 | `18d4faf1b37c8fd9982677bd2bff7635f5d254483b8641bd9745315219cf38be` |
| 3 | HC2 0.80 x 1.40 | 16,594 | 190 | `0fddb6412ea39595bdceeba7bc762d3397f89af49d32811dd783e807be6314ea` |
| 4 | HC4-L 0.30 lateral | 31,622 | 387 | `76da0fd8fb9efe99e332ba9d7e787f7d3c607417ee48b906bb3091a6e941f15f` |
| 5 | HC4-L 0.50 lateral | 67,464 | 873 | `9bc85efa2917c16fb007fa51647e87c1115e87dd469dd7eb267c384af3a1fad9` |
| 6 | HC4-L 0.80 lateral | 32,352 | 428 | `3d9d24355457e033e42448dfb1b71438443ee65186b4f6d644d20127b6264026` |
| 7 | HC4-R near-range teacher | 107,665 | 1,466 | `69c8238505a4f60f8de9f17816993947597fbbf5920253898117e6667a961f06` |
| 8 | HC4-R2 correction seed 131 | 35,673 | 481 | `e145e34c9ac61cc3e2778151139847cbc98ede0be2dea4850066e584652bc08a` |
| 9 | HC4-R2 correction seed 137 | 34,287 | 449 | `3f20078db7cfc0e460ab0564de608735d5c919c50df5efe4d7dc6eb53fcfb7ca` |
| 10 | HC4-R2 correction seed 139 | 35,926 | 485 | `e871dda49fbe2155f0e9fbc3699abd609a0c711a757dde824c889107527fbce5` |

Total: 414,430 samples from 5,334 episodes. Every shard names the same
motor-aware actor SHA-256
`080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.

## Offline gate

Exactly one seed-42 candidate is trained on CUDA after a clean-branch,
idle-GPU, exact-hash preflight. It advances to closed loop only if validation
speed MAE is at most 0.025 m/s, validation yaw MAE is at most 0.050 rad/s, all
checkpoint values are finite, and the manifest retains the ordered inputs and
device. Offline fit never establishes obstacle safety.

## Fresh fixed-attempt closed-loop gate

Evaluation uses HC4-E1 protocol
`first-terminal-attempt-per-environment-v1`: 64 first attempts per cell, a
700-step ceiling, and no post-reset samples. Seed 193 is the pre-screen; fresh
seeds 199 and 229 run only if it passes.

The twelve cells are speeds 0.30/0.40 m/s, ranges 0.90/1.15 m, and lateral
positions -0.08/0.00/+0.08 m. HC4-U1 is paired against HC4-R2 in the six 0.90 m
cells and HC4-LH in the six 1.15 m cells. Every service must explicitly set
`CUDA_VISIBLE_DEVICES=0`, and GPU work must remain exclusive.

Each cell must have 64 expected, completed, and resolved attempts, with zero
unresolved, hard-failure, other-terminal, NaN, non-finite, fall, and rated
motor-speed events. Relative to its paired accepted source, HC4-U1 must have:

- no per-cell or aggregate increase in collisions or timeouts;
- no per-cell or aggregate decrease in clean passages;
- approach and recovery mean route speed no more than 0.03 m/s lower per cell
  and no more than 0.01 m/s lower when sample-weighted across the matrix;
- no interaction-speed target; slowing or braking remains legitimate there;
- maximum per-cell motor torque-utilization p99 at most 0.60.

Only a complete three-seed pass can accept HC4-U1 for this exact-geometry
envelope. A failed pre-screen stops continuation. No MP4, broader geometry,
sensor noise, raw perception, or physical motion is authorized before the
numerical gate passes.

## Retained training result

The exact seed-42 CUDA job completed successfully on 100.100 using all ten
ordered shards: 414,430 samples, 5,334 episodes, 331,598 training samples, and
82,832 validation samples. Epoch 198 minimized validation MSE at 0.00037494.
Validation speed MAE was 0.003528 m/s and yaw MAE was 0.010007 rad/s, so the
offline gate passed. All reported values were finite.

- Checkpoint SHA-256:
  `2196d2ed2dbc3e182fa0b36edf663d11187330d430cd319ceb368c8a28e9753b`.
- Manifest SHA-256:
  `eb6ecba9417409c40a5439f4c0cd705b44212690ad1864ac25440c51ff13b503`.
- Source/predeclaration commit:
  `6641e6c105c6c6c6866cc2d73f3dd966f42dcf4c`.

## Seed-193 pre-screen decision

The first candidate rollout completed all twelve simulation cells, but the
initial report write failed on a missing HC4-U1 stage-name mapping. It produced
no JSON and therefore no admissible numerical evidence. The mapping-only fix
was tested and pushed as `d197dadd23b19b113ba58f9650f53be24ce79d21`; the
same frozen matrix then ran once into a new output directory.

All three valid reports used 64 completed and resolved first attempts per cell,
with zero unresolved, hard-failure, other-terminal, fall, NaN, non-finite, or
rated-speed events. The aggregate result was:

| Controller | Clean | Collision | Timeout | Attempts | Pooled approach delta | Pooled recovery delta | Max torque p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC4-U1 | 766 | 0 | 2 | 768 | -0.00212 m/s | +0.00612 m/s | 0.5623 |
| Paired HC4-R2/HC4-LH | 760 | 2 | 6 | 768 | reference | reference | descriptive |

HC4-U1 passed aggregate outcome, every collision, clean-passage, phase-speed,
and motor gate. It failed exactly one predeclared check: at 0.30 m/s, 1.15 m
forward, and 0.00 m lateral, HC4-U1 had 63 clean passages and one timeout,
while HC4-LH had 63 clean passages, one collision, and zero timeouts. The
collision improved by one but timeout worsened by one, so the independent
per-cell timeout non-regression rule requires `stop`.

Retained SHA-256 evidence:

- HC4-U1 report:
  `c7a261e6cdc205f2be7778cce2f50d412ae2d9492c277a5d944c5876c64b6d99`;
- HC4-R2 near-source report:
  `95666f52b1d584dcbbb868ce66d2387329283020b88365f44daceed6a527fc4a`;
- HC4-LH far-source report:
  `51a53c05ccba5533823b594fb23d8f5a68ae29266bfa4e28119ba982326650c1`;
- deterministic decision JSON:
  `f99678d2c6363bf1273e97d5ab3a8895f3bb0fce640edfbd328720a7a5c99acb`.

The reports and decision are retained under
`artifacts/evaluations/hc4u1-6641e6c-s193-prescreen/` on 100.100. Seeds 199
and 229, MP4 recording, and promotion were not run. HC4-LH and HC4-R2 keep
their existing accepted specialist envelopes.

## Smallest next revision

The evidence does not justify weakening the gate or rerunning HC4-U1. A future
HC4-U2 may change only one training input: add deterministic-teacher labels at
states visited by HC4-U1 in the single failing far-center cell, while retaining
the HC4-U1 architecture, optimizer, base data, seed, and fixed-attempt gate.
Collection seeds, outcome-retention rules, exact hashes, and a stop condition
must be predeclared before that GPU work. This is a proposed next design, not
training authorization from this result.
