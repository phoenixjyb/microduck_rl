# B1-N first-attempt trajectory capture and replay

Protocol `football-b1n-first-attempt-trace-v1`. This is the next evaluator
component after the successful [short CUDA integration](2026-09-09-stance-cuda-integration.md).
It does not launch an optimizer, run another hold, restore a checkpoint or admit
a nominal stance policy. The declared observations, rewards and gates are unchanged.

## Owned capture and numerical replay

`stance_attempt_trace.FirstAttemptTrace` consumes the caller-owned runtime's
initial snapshot and successive tick results. The caller must retain the actor
input before calling `runtime.step(actions)`: the runtime's initial tick snapshot
already contains the newly limited correction, not that pre-action actor input.
Both are retained separately along with the supplied raw actions and tick reward.
This component checks reward finiteness and that closed worlds receive zero; it
does not independently recompute dense rewards or prove network inference.

The trace starts at physics step0, owns CPU copies and accepts at most250 ticks
of up to10 physical substeps for at most128 evaluation worlds. It refuses reset,
skipped or globally duplicated substeps; mismatched policy input; changing frozen
physical state/observations; missing, changed or inconsistent terminal records;
nonfinite data; and warning flags. An invalid append adds no partial tick and
permanently faults the recorder. No reset or partial-success publication is allowed.

Adjacent ticks share a physical boundary. Its physical fields must match exactly;
only the live world's realized correction may change before new physics. Within
a tick, each world either advances exactly one physical step or stays bitwise
fixed. The first terminal's counter, qpos/qvel, state and observation must match
the retained trajectory. Excessive proposed torque is an explicit pre-step stop,
not an extra integrated sample. Full terminal contact JSON is retained, checked
for finite values and world identity, but geometry/force reconstruction against
the compiled plant is still a separate unfinished provenance gate.

CPU replay rebuilds these checks, removes only reconciled per-world frozen
duplicates, and feeds every actual boundary to the unchanged `score_attempt`.
Full attempts contain steps0..2500; early first failures remain failed complete
attempts and clean short prefixes remain incomplete. An earlier failed frame
cannot be hidden by later recovery or averaging. No score smoothing, denominator
changes, favorable seed/checkpoint selection or historical rescoring occurs.

## Binding and serialization boundaries

The exact binding records protocol, source SHA, runtime/launch/checkpoint SHA-256,
checkpoint iteration, evaluation seed, world count and explicit capture device.
It permits only checkpoints128/256/384/511 and held-out seeds541/547/557. CPU
fixtures cannot claim CUDA at capture. CPU replay of retained CUDA metadata is
not a new CUDA execution or proof of its origin.

`encode` first replays the payload, then returns CPU tensor artifact bytes, their
SHA-256, byte count and numerical score. It writes no files. The caller must
durably and exclusively store those bytes and a bound manifest. `verify` checks
the independently supplied SHA-256 before weights-only CPU deserialization and
full replay, with a512 MiB serialized-size cap. It refuses unexpected bindings.
The byte cap is not a general-purpose sandbox for hostile tensor archives;
use only owned artifacts from the guarded launcher, never arbitrary downloads.

Supplied hashes alone do not prove clean live source, correct checkpoint restore,
reset identity, actual policy inference, log supervision, or a complete384-attempt
checkpoint matrix. All outputs keep `provenance_validated`, `checkpoint_admitted`,
`learned_stance_accepted` and `physical_motion_authorized` false. The complete
launch/restore/plant/contact contract and matrix assembly remain required before
the separately declared training smoke or any candidate promotion.

## Validation and next integration

Tests include explicitly synthetic five-second successful traces (not learned
performance), partial prefixes, first physical and pre-step torque failures,
frozen siblings, missing/reset/skipped counters, hidden earlier failures, altered
terminal/observation/contact-world evidence, changed bindings and corrupted
serialized bytes. A short actual Warp CPU test performs two policy ticks with
bounded nonzero targets and an explicitly synthetic first-substep tilt. It checks
the real runtime interface without repeating the closed native hold or claiming
ball balance. No GPU work or optimizer is launched by this component.

Local validation:183 focused CPU tests passed in11.59s, including37 new capture,
replay, binding and serialization cases. Markdown HTML rendering and local-link
checks passed. Linux exact-source confirmation is pending at this commit.

Next integrate this recorder with a source/runtime/plant-bound artifact writer,
strict fresh-actor checkpoint loading and the exact held-out matrix, then finish
the PPO adapter and launch guards. Retain the CUDA probe unchanged; do not rerun
it as a substitute for completing the remaining implementation.
