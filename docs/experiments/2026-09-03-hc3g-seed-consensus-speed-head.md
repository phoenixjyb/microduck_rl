# HC3-G seed-consensus interaction-speed head

Date: 2026-09-03

Source commit: `e63f708b7501df5f461e809e44f15cf6d728672a`

Decision: **rejected at sensitive-cell pre-screen; close HC3 speed-head line**

## Candidate contract

HC3-G was the predeclared final bounded speed-head candidate after HC3-F. It
formed each of the 65 HC3-E speed-head deltas relative to the byte-exact HC2
anchor. A coordinate was retained only when all three training seeds had the
same non-zero update sign; its value was their arithmetic-mean delta. A
disputed coordinate was restored exactly to HC2.

Twenty-six coordinates had unanimous direction and 39 remained at HC2. The
hidden layers, yaw row, non-interaction speed behavior, frozen locomotion actor,
structured-obstacle observation, and physical-motion boundary were unchanged.

## Exact artifact

The retained checkpoint is:

`../artifacts/hc3g-e63f708-consensus-s109-113-127/supervisor.pt`

SHA-256:
`02072b01d64e818a0bfb9f8f1cee32552e8fbe9ff488e29c43134d4966b2f941`

The JSON manifest SHA-256 is
`80bbf44c98d18cde7431a2cbdeec6bf961d927cb8c70e28e609cb8a68f2fe1c5`.
It records all three source checkpoint hashes, completed/configured iteration
counts, the unanimous-sign arithmetic-mean rule, and the 26/39 parameter
partition.

## Sensitive-cell result

The unchanged pre-screen used evaluation seed 41, 64 environments, and 700
low-level steps per cell.

| Speed x forward position | Clean | Collision | Timeout | Resolved | Passage | Torque p99 | Near-stall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 m/s x 1.15 m | 64 | 1 | 0 | 65 | 8.669 s | 0.578 | 0.0150% |
| 0.80 m/s x 1.40 m | 62 | 2 | 0 | 64 | 8.616 s | 0.743 | 0.2186% |
| Total | 126 | 3 | 0 | 129 | — | 0.743 | 0.2186% |

Both cells had zero falls, NaN terminations, non-finite steps, and rated
motor-speed exceedances. The candidate exceeded the complete one-collision
budget in the high-speed cell and recorded another collision at 0.50 m/s.
It therefore did not run the full matrix or receive an MP4.

The reports in `../artifacts/hc3g-e63f708-prescreen/` and the checkpoint are
byte-identical to the retained copies on 100.100.

## Terminal HC3 decision

HC3-A through HC3-G did not produce a candidate that improved passage time and
retained the HC2 collision budget. HC3-G was explicitly the last bounded
speed-head candidate, so this result closes that optimization line. We do not
select a favorable seed, search interpolation factors against seed 41, widen
learned authority, or spend more GPU time on rejected candidates.

HC2 remains the accepted simulation controller for the original four centered
cells. The next curriculum step is a diagnostic HC4-L lateral-placement sweep
using unchanged HC2—not another HC3 optimization. Placement data will be
measured first; no expanded-placement policy will be trained or promoted until
the new envelope and per-bin gates are retained.
