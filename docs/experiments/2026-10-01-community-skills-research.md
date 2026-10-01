# MicroDuck community skills: October 1 research and intake

Research snapshot: 2026-10-01 Asia/Shanghai. Starting local source:
`ade85fde1b4f8b28b349e5e9ab7bee2865a9684d`, feature branch
`feat/athletics-obstacle-curriculum`. This is a research/intake result, not a
training launch or a newly accepted robot capability. No physical duck is
available. Published measurements below are the authors' results, not ours.

## Findings worth using

The community now offers actual weights, reproducible evaluation packages and
new training tasks, not only demonstrations. The
[community index](https://github.com/joeynyc/awesome-microduck) was useful for
discovery; the primary repositories/model cards below were checked separately.
Search results and catalog counts are not acceptance evidence.

| Direction and primary source | What is available / reported | Implication for our duck |
| --- | --- | --- |
| [Official policy set](https://huggingface.co/pollen-robotics/microduck-policies/tree/1b56c396825c052a4e26e95cf2b8d8298af9e9b4) | Ten ONNX files. Its pinned manifest selects `velstand` as the default walk, alongside standing, sit/stand, picking, rolls, kicks and roller behaviors. | Audit a contemporary stock walk/stand reference; do not overwrite our frozen locomotion or automatically update its environment. |
| [Happy Hop](https://huggingface.co/joanfox/microduck-happy-hop/tree/c0447da668255436eea070e862ad8eb89bad2ba1) | Stock/backlash two-foot one-shot; 61 observations, 14 actions, 50 Hz, required one-step delay. The card reports Pollen's September 1 physical test. | Promising **separate stock-mechanics hop reference**, not evidence that our rejected sprung-K3900 H1 passed. Stable entry and landing handoff need local tests. |
| [Blind basketball balance](https://huggingface.co/HannesVonEssen/microduck-basketball/tree/6d8f74b97c75b1597efead1754ff54ca2af4900c) | ONNX, resumable checkpoint, source, configs and evaluation. The author reports 2,980/3,072 first-fall trials surviving 60 s on a free basketball, from proprioception with LSTM memory; hardware untested. | Directly relevant to rolling-ball ambition. A basketball, placed-at-apex reset and recurrent actor are **not** our soccer ball, mounting skill or 10-action stance plant. |
| [Robustified running](https://huggingface.co/HannesVonEssen/microduck-running/tree/dd11bd188edce3ebae8f60ce54c77b0d6098d00a) | ONNX, checkpoint, configs and evaluation; card reports 1.651 m/s nominal forward speed, with substantial heading/lateral drift. Simulation only. | A comparison candidate, not a physical top-speed specification or a route-tracking parent to adopt without motor/heading tests. Its mat-impact video contains scripted effects, not learned avoidance. |
| [Rough walk G](https://huggingface.co/RemiFabre/microduck-rough-walk-g/tree/242876a0aa8b40b702142fb0a5677fd43bc88a4c) | Rough-terrain ONNX; reported 12/110 falls versus 32/110 for the reference, with 17% more motor power. Weak turning and rubble limits remain. Simulation only. | Borrow the terrain ladder and paired motor comparison, not a universal rugged gait or a safe speed limit. |
| [Step-up](https://huggingface.co/Nupr-Haokun/microduck-step-up/tree/b98b48650ac443a17cd25eaee472d1844eb2b3e4) | Crossing plus recovery reported successful in 13/14 deterministic alignments of a 25 mm step. The actor does not detect the edge. | A later **externally aligned threshold-crossing** skill, preserving the perception/control split. Not general obstacle avoidance. |
| [Flamingo cycle](https://huggingface.co/RemiFabre/microduck-flamingo-cycle/tree/a69e4f6103a70e355983eb5d4e919da76778a40f) | Either-side single-foot hold and return; command slots encode flag/side, not velocity. Card reports failures under stronger backward pushes. Simulation only. | Useful support-transfer lesson after stance/recovery. Never pass ordinary velocity commands into its repurposed slots. |
| [TorchRL nine-skill prior](https://huggingface.co/torchrl/microduck-skills/tree/d6766234b24bf9cff6a6a8c55011a4584a8995d5) | GRU prior includes stand, four translation directions, turns, forward and in-place hops. Reports 32 held-out episodes per skill, one training seed, native CPU MuJoCo. Actor-only checkpoint, not a standard ONNX slot. | Borrow balanced skill sampling and per-skill evaluation. A separate backend/interface/license review precedes reuse; not a reason to replace our frozen stack. |
| [Roll/roller-slalom package](https://huggingface.co/langli11/microduck-tricks/tree/ae7284e0dc54e6ae10e5f099f4a4d3b93aed7158) | Three ONNX policies and evaluation scripts; card reports 200/200 walking-to-roll handoffs and an eight-gate roller slalom. | Borrow transition batteries and command-tracking comparisons. The [pinned evaluator](https://github.com/easyrider11/microduck_rl/blob/146777b989f47160d26590e29a9d13957b2c6a3c/scripts/eval_roller_slalom.py) uses a course-aware commander and virtual cone clearance, so this is not learned perception or physical obstacle-contact acceptance. |

Additional ideas, not immediate training priorities:

- [Microduck headstand](https://github.com/zachgarner/microduck-headstand)
  composes several acrobatic primitives: useful inspiration for evaluating the
  whole routine rather than only individual clips.
- [Microduck Circus](https://github.com/ros-claw/microduck) now supplies separately
  trained jumper/turner policies with a phase coordinator. Its current result
  passes only five of eight runs; turners use 73 observations, and coordination
  reads simulator rope state. Treat contact-fidelity limitations explicitly.
- [Vision soccer](https://github.com/toyhank/microduck-vision-soccer) keeps visual
  servoing outside the locomotion policy and separates its camera controller
  from the scored ground-truth baseline. This architecture fits our boundary;
  soccer play is not standing on a football.
- [AR control](https://github.com/kgediya/specs-microduck) and
  [MCP simulation control](https://github.com/aj-dev-smith/microduck-mcp) are
  interaction tools, not proof of new learned motor skills.
- [Microduck Academy](https://huggingface.co/spaces/tfrere/microduck-academy)
  is a discovery lead only: anonymous source/API fetch returned HTTP 401 in this
  pass. No internal implementation, cost cap or training outcome was verified.

## Details that prevent careless reuse

1. **Shared dimensions are necessary, not sufficient.** The official training
   repository documents the 61/14 contract and normalized ONNX exporter, but
   joint order, HOME offsets, action scaling, contacts, backlash, delay,
   filtering, command semantics and reset state must also match.
   [Upstream training source](https://github.com/pollen-robotics/microduck_rl)
   was inspected at develop tip `cfe1c2adcceb55f6b6e369c888b31c6873175c55`.
2. **Recurrent support is no longer merely a proposal.** Runtime
   [PR 231](https://github.com/pollen-robotics/microduck/pull/231) merged on
   September 10. It carries/clears explicit LSTM hidden and cell state, using
   model API 2. That does not update our old fork automatically, establish board
   latency, or support the TorchRL GRU checkpoint.
3. **Model card and manifest can disagree.** Happy Hop's pinned README reports
   the physical test and mandatory 20 ms action delay; its manifest still says
   hardware untested and omits an explicit delay field. Retain both, resolve
   the contract before replay, and do not derive our admission from either claim.
4. **A landing can look good but fail impact limits.** The separate
   [3–6 cm jump project](https://github.com/chenp9527/microduck-jump) reports no
   falls over 128 episodes but a 128.3 N landing peak against its 40 N target.
   Neither that target nor that result is our calibrated mechanical limit.
5. **Licensing is per artifact.** The eight individually inspected ONNX model
   repositories in the table advertise Apache-2.0. TorchRL's model metadata and
   the polite-bow lead did not expose a model license in this pass. Code,
   weights and robot meshes need separate attribution/license checks; no
   relicensing or public redistribution follows from this research.

## Proposed curriculum revision — separate tracks, unchanged old gates

This is a prioritized development proposal. Full seed/budget/reward/evaluation
predeclarations and source-bound host qualification still precede new training.
Historical H1 and football gates are not renamed or waived.

| Priority | Next bounded lesson | Evidence required before progression |
| --- | --- | --- |
| C0: now | Pin and inspect official stock stance/walk, Happy Hop and basketball candidates; keep them quarantined from every active controller | Exact revision/hash/license records and actual ONNX interface; no model execution or skill admission |
| C1 | Independent stock walk → settle → one-shot hop → settle → walk rehearsal | Matched BAM/backlash/collision assets, delay and filters; first-attempt takeoff, bilateral clearance, impact, drift, landing, motor limits and speed reacquisition; untouched confirmation and retention |
| F/S/O | Continue our foundation/stop/recovery and obstacle supervisor track | Existing speed/heading/lateral/motor and per-placement gates. Slowdown allowed only inside avoidance; approach/recovery speed and stale/missing sensor behavior stay required |
| B0/B1, then B2–B5 | Audit community ball plant and recurrent contract; complete our robot/football fit and stance replication/disturbances before ball progression | Explicit basketball versus football geometry/dynamics; no hidden support, reset-on-top clearly labeled, first fall and contact/motor accounting, recurrent parity/reset tests |
| Later | Small terrain ladder, one-foot support transfer, threshold crossing, then optional kicking/picking | One changed difficulty axis at a time, per-bin worst-seed results, previous capabilities retained |

The stock reference investigation can proceed on CPU without bypassing the
[WSL timing qualification](2026-09-30-stance-wsl-replication.md) for our bespoke
64-world stance learner. Community batched `mjlab` training and native-CPU
TorchRL are alternative pipelines to study, not measured speedups for our current
runtime. Do not launch a new long GPU job on an unqualified estimate.

Our current [capability graph](../athletics_obstacle_curriculum.md) retains narrow
exact-geometry obstacle specialists and a nominal stance lesson. It does **not**
establish general walking/stop/recovery, accepted hopping, football balance or a
composed all-skills robot. Do not call these community releases our achievements.

## Retained intake and executable check

Downloaded only the three small ONNX candidates, their README and manifest,
from immutable revisions into ignored local storage:
`artifacts/community/intake-20261001/{official-velstand,happy-hop,basketball}/`.
No checkpoint pickle, source archive or third-party executable was loaded.
The original upstream filenames for the first candidate are `velstand.onnx`
and the official set's manifest; its local ONNX alias is `policy.onnx`.

| Candidate | Hub revision | ONNX bytes | SHA256 |
| --- | --- | --- | --- |
| official velstand | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` | 793705 | `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45` |
| Happy Hop | `c0447da668255436eea070e862ad8eb89bad2ba1` | 793778 | `abd6db1bca2f2a583508cfd9a6fd144a7cce65b302827b9b9a0c5227fc2acae9` |
| basketball | `6d8f74b97c75b1597efead1754ff54ca2af4900c` | 2500571 | `e105148b160b3a86170621648215be38dd5018e955930ce182394f4513c7569b` |

Pinned supporting metadata hashes, in the same candidate order:

- README: `98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74`,
  `108c65ce78acfdeee87ca115e6bf3e638fbb18b4bcbf42ba058c89da2a128198`,
  `a2e8d266ebd6127dd501df375fd92fac45d4147e0e56146307ddd247b3a14f7c`.
- Manifest: `622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94`,
  `c3e351d46a0401d437916aa7520127505b50065cc0cd3225fc8408357f345089`,
  `bc3356d168c0496861f853bb22e36b295c054abcd62841f257d3c3925ef7ddba`.

The new [CPU-only inspector](../../src/mjlab_microduck/community_policy_inspection.py)
verifies the exact file hash, checks self-contained ONNX structure and reports
ordered graph inputs/outputs. It refuses external tensors and custom operator
domains, and never runs the graph. Classification of an API-shaped interface
does not prove its command semantics, normalizer or runtime compatibility.
All behavioral, transition, training and physical-motion flags remain false.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.community_policy_inspection \
  artifacts/community/intake-20261001/happy-hop/policy.onnx \
  --sha256 abd6db1bca2f2a583508cfd9a6fd144a7cce65b302827b9b9a0c5227fc2acae9
```

Completed local validation with CUDA hidden:

- 22 new inspector tests; 186 tests passed in the combined inspector, existing
  static compatibility, retained-skill binding and retention-plan selection.
- All three downloaded ONNX hashes match the Hub's published SHA256 entries.
  Each passes self-contained structural checking at IR 8 / opset 18.
- `velstand` and Happy Hop declare float32 `obs[1,61] -> actions[1,14]`.
  Basketball additionally declares float32 `h_in/c_in[1,1,256]` and
  `h_out/c_out[1,1,256]`; classification is `lstm-microduck-api2`.
- The actual-artifact inspection imported neither Torch nor ONNX Runtime,
  executed no inference, and left every admission flag false. This checks
  declared graph structure, not runtime action correctness or normalizer parity.
- Source review corrected sparse-initializer input handling against the
  [ONNX protobuf definition](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto):
  only its values name denotes the initializer, not the indices storage name.
- `git diff --check` passed. Markdown tables and local links were checked after
  rendering in memory. Full repository tests, Linux/WSL validation, closed-loop
  simulation, inference parity, GPU training, video and physical motion were
  **not run** in this chunk. No host service or environment was changed.

Next executable development chunk: bind a matched stock/backlash plant and
explicit delay/filter/entry contract, then implement a bounded CPU rehearsal
and first-attempt transition scorer for C1. Do not run it on the bespoke stance
plant or label it accepted from structural checks. Ball work first audits the
released plant and recurrent reset semantics against B0–B6; no direct football
promotion is granted.
