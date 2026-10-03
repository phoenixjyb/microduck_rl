# Renewed scheduled-CUDA integration window

The owner renewed simulation, preparation, gated training and development through
**2026-10-03 20:00 Asia/Shanghai**. The expired 08:00 protocols and their retained
successes/refusal remain unchanged. `stance_recovery_campaign_window` is a new
fixed authorization boundary, not a relabeling of those runs.

## Predeclared next numerical gate

The frozen seed-577 iteration-255 parent remains byte-identical:
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`.
The previous three full CPU dose cases passed; there is no measured reason to
train those cells. The next experiment qualifies only the scheduled force path
on CUDA before a broader unchanged-parent deficit screen.

Two sequential fresh one-world cases, evaluation seed 671, dose-stage held-out
schedule: `zero-wrench` and `+x-2n-20steps-t250`. Both run **60 policy ticks / 600
Euler steps / 1.2 simulated seconds**. The push is 2 N along world +x for 40 ms
at 0.5 s; the zero control retains its declared zero-force phase window at 1 s.
Each backend must have an exact matched pre-push prefix through physics step250.
This does not require CPU and CUDA trajectories to be bit-identical to each
other. A CPU actor must replay every recorded input exactly on the CPU scorer;
actual force arrays and forced-pre/integrated/unforced-post phases are checked.

Preparation: genuine CPU traces and whole consistency replay under a 120-second,
2 GiB, 200%-CPU, Nice10, control-group user service. The parent/prerequisite
receipts, entire closed inventories and current source/runtime/profile/services
are checked before simulation. All captures are retained before numerical
judgment; there is no retry, reset, optimizer step or checkpoint selection.

CUDA supervision: a separate 240-second, **3 GiB**, 200%-CPU, Nice10,
control-group user service holds the existing shared FilmBrain GPU lease. Only
one 180-second owned child may allocate CUDA0. Each case has a 60-second
collection cap; 30 seconds are reserved for serialization/retention. The
previous five-case CUDA run measured about 0.687 seconds per policy tick in its
full child including construction and serialization. The two 60-tick cases
predict about 82.4 seconds, or103 seconds with a1.25 factor; the fixed180-second
child cap also covers
imports, hash checks and construction. This is a new wrapper over the existing
owned-child watchdog, not an increase of any old public cap. The combined
one-world native child and CPU supervisor need more than the previous CPU-only
2 GiB cap: the previous full CPU dose service peaked at 1.3 GiB and its independent
CPU scorer at 0.515 GiB; 1.25 times their sum is about 2.27 GiB, rounded to 3 GiB.
No change to driver, packages, clock, source allowlists or protected services.

The supervisor checks source, FilmBrain, protected-service inactivity, sole-child
GPU occupancy, temperature, memory and a 1 MiB numerical-warning log bound.
Failure kills/reaps only the owned child group and preserves partial artifacts.
No overlapping unrelated compute is permitted, even if memory would fit.

Independent closeout: another CUDA-hidden 120-second, 2 GiB CPU user service
rehashes the statically fixed inventory and freshly re-scores both whole captures
and the matched prefix, for either a complete qualification or rejection.
Timeout/error paths retain their actual partial files and the failed report;
they require read-only failure diagnosis and an exact partial-file inventory,
not a successful closeout or an automatic retry. A failed preparation is likewise
not an independent qualification. Launch requires the complete
120 + 240 + 120 + 60 = **540-second**
prepare/supervise/closeout/margin reserve before the fixed20:00 cutoff.

Passing means only `scheduled-cuda-integration-qualified`. The five-second
full-duration gate is explicitly false, as are recovery, training/checkpoint,
football, physical-motion and independent native-attestation acceptance. There
is no thermal model or whole-trajectory fresh physics re-simulation. No video,
new long job or capability promotion follows from this short gate alone.

## PPO adapter work in parallel

The prepared real parent has private CPU RNG state but it is not connected to
sampling. The existing 24-transition buffer also ends at0.48 s, before the
earliest 0.5 s push. A separate composition adapter is being developed with a
28-transition buffer, private action/minibatch RNG, unchanged raw sampled PPO
actions, terminal-before-reset retention, exactly-once timeout bootstrap and
finite gradients/Adam. Its tests and native transition/timeout qualifications
are separate from this CUDA force-path experiment and from actual skill learning.

## Results

Completed at exact source `84ce52af2fc501830426a4290166f860a7d1a2fe`, with a clean
frozen WSL worktree throughout. The source/fork tip was independently verified
before the run. Later Mac bridge development did not change the active WSL source.

Three capped services completed successfully (no change/retry while active):

- CPU preparation: invocation `85edc408f197494cacb8f78b429e3279`, measured
  runner23.337526 s,684.1 MiB peak, no swap.
- CUDA supervision: invocation `6580bddddd064df793926b5ff73a1ddc`, measured
  supervisor107.254312 s, child92.399547 s,2.6 GiB service peak, no swap.
- Independent CPU closeout: invocation `79981cf045324aedac2cc638ade3cdfc`,
  measured scorer13.887622 s,1.0 GiB peak, no swap.

Decision: **`scheduled-cuda-integration-qualified`**. Both genuine CUDA cases ran
60 policy ticks /600 physics steps without terminal or soft-limit exposure. The
zero control had all10 declared zero-force phase checks; the push delivered all
20 nonzero force steps, with full forced-pre/integrated/unforced-post checks and
cleanup. Every actor input replayed exactly on CPU, with maximum action error0.
Whole control/motor-history and plant/kinematic consistency checks passed.

Zero/push collection times were39.224091 /38.516119 s. Peak planar speeds were
0.01514418 /0.05709777 m/s and peak tilts0.02043538 /0.02994196 rad. Both had a
maximum0.11519976 Nm motor torque and0.03776584 W absolute joint power; these
whole-window maxima are diagnostics, not motor thermal validation.

All72 telemetry samples permitted only the owned child PID2176412. GPU peak was
39 C and974 MiB, with at least23188 MiB free. Two idle samples passed after the
child and independent closeout. FilmBrain observatory PID521 and video playground
PID298048 stayed active with restart count0; both protected AI Mission services
stayed inactive in user and system namespaces.

CPU matched-prefix hash:
`eb0be5ba1c88bc9864a8c8912369c413b414a55be9dae3245e57a7ebd1ef86df`.
CUDA matched-prefix hash:
`655a0cd64741d8230396297c062716ea5687d26bfb7068476dc84a2013e27b46`.
Each is equal across its own backend's two cases; they are **not** equal across
CPU/CUDA, and no cross-backend bit-equivalence is claimed.

Exact retained hashes:

- launch: `06611ab94a25414f3f2d28c8e622cdd5ed830d324437cff555d6d5020fb3da21`
- report: `a1c2b4bd91ca1b3efef234cdb3bd1bddcc7975e6881f5e9e0cb91380945a4137`
- independent closeout: `cb2c5cb2a53c67c7f84bf01df1a884d5ab5365e816c95cde051fc050b121a241`
- zero CUDA raw: `270df02b88382e97946f9cf4bcfadd9d9efa014cab16ed03cd64e3e792368bcb`
- push CUDA raw: `61a36499d8fdd2d82817e972d71733f701eca13eac4f0ac6a1f55535fb48163d`

WSL root: `artifacts/evaluations/stance-wsl-scheduled-cuda-integration-84ce52af2fc5`.
The independent process rehashed the fixed15-file pre-closeout inventory and
freshly reproduced both whole scores and the decision. Including closeout, the
closed inventory is16 files /38655145 bytes. Every hash and byte count in the full
Mac mirror was independently compared with that closed inventory under
`artifacts/retained/scheduled-cuda-84ce52af2fc5.FI1dxc`.

Native WSL scoped source tests passed:132 in44.26 s, receipt
`e3a159379cbf0983fc1541c2dc2fcdc8e9033f4045daadc2a2124b9fd2af35d8`, invocation
`c4e14a9e90314d25b6552d0887f35926`. This broad test process reached its2 GiB cap
and1.9 GiB swap peak; that is distinct from the no-swap native preparation,
CUDA supervision and closeout. It is not evidence for increasing any job cap.
After bridge integration,170 relevant local tests passed in24.87 s.

The short numerical recovery evaluator deliberately still returns a false
full-duration/candidate gate: these are1.2-second traces, not5-second completed
attempts. The integration pass does not change any of the eight acceptance flags.
Optimizer steps remain0; no new learned policy, recovery/football capability,
thermal validity, whole-trajectory fresh physics or physical motion is admitted.

Next: predeclare and test a wider full-duration frozen-parent timing/direction
screen; independently qualify the two-row stochastic PPO transition trace. Train
only a repeatable measured deficit after the remaining native reset/optimizer and
held-out gates, not the already-passing cells. The20:00 goal remains active.
