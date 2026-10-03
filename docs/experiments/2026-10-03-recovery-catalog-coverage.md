# Recovery catalog coverage: no measured training deficit

Status: **whole-byte checked metadata synthesis, not a new native experiment**.
The unchanged D1 seed 577 / iteration 255 policy was used by both independently
closed screens. No learner update, checkpoint selection or curriculum promotion
is introduced by combining their coverage.

| Closed screen | Complete first attempts | Distinct cells |
| --- | ---: | ---: |
| [Stronger cardinal dose/timing](2026-10-03-cardinal-dose-timing-screen.md) | 25 | Zero plus 24 cardinal cells |
| [Gentle timing/diagonal gaps](2026-10-03-gentle-timing-diagonal-gap-screen.md) | 21 | Zero plus 12 gentle cardinal and 8 diagonal cells |
| Union | **46** | **45**, with only the zero-control cell repeated |

The exact union of recorded cell IDs equals every cell in the existing
`cells("dose", held_out=True)` catalog: no omissions or additional cells. All
46 recorded first attempts have 250 policy calls / 2500 physics steps, zero actor
replay error, and all nine unchanged numerical gates passing. Each screen used
the single deterministic evaluation seed **671** and fresh first attempts.
Different evaluator source IDs and original failure/repair histories remain
separate; this synthesis does not relabel them as one newly collected matrix.

The October 3 evening check rehashed the full cardinal repair receipt and gentle
external receipt/report against their retained pins, then each of the 21 gentle
whole replay JSON files against the external inventory before comparing IDs
and gates. Exact anchors:

| Record | SHA256 |
| --- | --- |
| Cardinal independent repair receipt | `adba134d905c8e07c9b871cce85b26a5704335944b47b6001ddfa9ecd0c2fcac` |
| Gentle independent external verification receipt | `f6e0df22eabd737e047dc290d8202bbfd5d22452cfb868be2381fa13a66c3e76` |
| Gentle original report | `58fdf6863752f7e3e8238d8cebaf7dd2dd8b7d9fc731c1349c59d8b300570e54` |

Local evidence roots are
`artifacts/retained/cardinal-screen-closed-b1878b8715ef.gIcnKb` and
`artifacts/retained/gentle-gap-closed-bb7c059d5dda.n3GlnP`. The original whole raw
archives remain on WSL. This evening metadata check did **not** rereplay CUDA
math, resimulate physics or newly attest those historical services.

## Consequence for the curriculum

Keep the unchanged parent. The current gentle/timing/dose cells have not supplied
a measured deficit that justifies training them again. This is **not** the
proposed 128-attempt, multi-seed, two-adjacent-checkpoint promotion matrix;
one deterministic seed is not randomized generalization. All capability and
physical-motion acceptance remains unchanged.

Finish the separately tested
[CUDA transition contract](2026-10-03-cuda64-transition-source-contract.md),
then qualify physical feedback, naturally observed terminal/reset behavior,
finite GAE and native optimizer replay under fresh bounded protocols. Choose a
new recovery gap only with an explicit initial-state/plant/terrain declaration,
fresh first-attempt evidence and independent replication of any failure.
Do not increase forces, change held-out partitions, weaken motor/settling gates
or train a passing cell just to keep the GPU occupied.

Obstacle negotiation/return-to-speed, hopping and football balance remain
separate capability ledgers. Football B2–B6 still need B0 contact/motor
feasibility and held-out B1 qualification. This catalog result neither merges
those specialists nor proves standing on a rolling football.
