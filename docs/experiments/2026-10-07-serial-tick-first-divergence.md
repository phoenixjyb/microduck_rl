# Retained one-tick first-observed divergence — October 7

This is read-only analysis of the [closed moving-state serial test](2026-10-07-serial-one-step-control.md), not a new physics run, a causal result, or a tolerance-based admission. Its execution source remains `26e2ee8244b7bd05fb3290d655113e538bd1a989`; no historical source-addressed artifact or protocol is changed.

`stance_serial_step_divergence.analyze_closeout` authenticates all five external closeout anchors before JSON decode, all 22 complete run artifacts before numerical decode, then reruns the independent receiver and requires its canonical output to equal both retained native and Mac receivers byte-for-byte. That receiver remains `fresh-one-step-serial-repeat-negative`, SHA256 `d8c38bf63780f40aaa0fc40492bd5a43b7e40c3290195ec59555b781727eee71`.

The analyzer compares all 21 snapshots in each of four reduction channels, all 150 motor field records, 40 complete masks/counters, 34 constructor/final frame fields, and nine reward-ledger fields. Counts and per-world differences are complete; at most four coordinates per field are examples. Float32 bit differences, signed zeros and ordered-bit distances are preserved. Observation ordering does not establish hidden kernel ordering, physical importance, or cause. Readbacks and serial dispatch already perturb execution timing.

## First recorded difference

Using zero-based indices, motor substeps 0 and 1 match in every recorded field. At substep **2** (the third 2-ms substep), all five commands, proposed torque, committed previous torque, voltage, gain, damping, correction, target, queue and control still match. **Only `committed.friction` differs: 345 float32 scalars, maximum absolute delta `3.725290298461914e-09`, maximum ordered-bit distance 3.** This is the first observed difference in the retained channels, not proof that all unrecorded inputs or solver state matched.

The reduction channels first differ later, at post-forward call 6 (after that third substep): CoM initialized entries 26 scalars, RNE initialized forces 2153 and accumulated outputs 2836. The weighted CoM output bank remains exact across all 21 calls. This supports inspecting the BAM friction candidate boundary before adding another solver observer; it does not establish propagation or blame BAM, CUDA, the driver, or the solver.

| Channel | Total differing recorded scalars |
| --- | ---: |
| CoM entries | 646 |
| CoM weighted | 0 |
| RNE entries | 33912 |
| RNE outputs | 42681 |
| Motor fields | 26283 |
| Masks/counters | 0 |
| Constructor/final frames | 19092 |
| Reward/dense ledger | 99 |

The final frame has 12 differing fields; all ten substeps were accepted, finite, live and nonterminal. None of these small finite deltas establishes physical risk or excuses the exact-repeat gate. Constructor/final frames are different-time states, so this analysis compares corresponding cases, not unchanged state over time.

## Retention and checks

Mac artifact `artifacts/tools/serial-step-divergence-26e2ee8244b7/mac-analysis.json`: **252029 bytes**, whole SHA256 `5c5d3959acfacea057bc3729a4f0a0270544e47189a7e8b164355f116fadf5b8`. The source helper itself does not import Torch, Warp or CUDA. Output is capped at 2 MiB; no tolerance, physics reexecution, runtime-cause, full-window, training or physical-admission flag is enabled.

Owner review corrected the scalar-mask first-event fallback and made the serialized byte count include itself exactly. Focused tests cover whole-file authentication before decode, wrong source/tree/leaf/terminal anchors, altered raw bytes, receiver disagreement, nonfinite/wrong-dtype fields, signed zero, observation ordering, mask mismatches, coordinate caps and output cap. The 13 analyzer tests passed; Ruff lint and formatting passed. The combined controller/producer/receiver/analyzer suite passed **103 tests in 11.55s** on the Mac.

After the exact reviewed source `cfb46ed3fff9e96740a0062c538b4665e478e5a9` was pushed, the clean native lean worktree was fast-forwarded from 26e using a SHA-verified bundle, with no running Duck job or GPU compute process. Under the unchanged frozen environment with CUDA hidden, the native analyzer independently reproduced **252029 bytes and the identical whole SHA256 above**; its 13 tests passed in 0.54s. Both `native-analysis.json` and `mac-analysis.json` are retained on both hosts and byte-identical; native files and directory were fsynced. This reproduction is CPU/read-only evidence, not a new native simulation or training result.

Next: inspect the exact installed motor/friction calculation and any uncaptured state read by it. Only after that read-only diagnosis should a smallest new capture be predeclared, tested, independently reviewed and source-bound. Preserve the frozen environment, cooperative GPU lease, FilmBrain, old failures and all negative evidence. No learner, optimizer, longer rollout, raw perception, video or physical motion is admitted. This work remains limited to 100.98 and the October 7 13:00 Shanghai cutoff.
