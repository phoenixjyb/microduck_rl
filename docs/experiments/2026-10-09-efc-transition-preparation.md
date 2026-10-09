# Positive EFC transition control and row-cost preparation

## Separate scope and evidence tier

Based at `b83bc7208b378d5ed949b75505ff4c916058139b`, exact feature branch
`feat/athletics-obstacle-curriculum`. This new CPU-only protocol
`microduck-efc-transition-preparation-oct9-v1` adds exactly three paths:
`stance_solver_efc_transition.py`, its focused tests and this document. Old
source fences, capture protocols, receivers, runtime and historical evidence
are unchanged. It does not admit a GPU child, acquire the FilmBrain lease or
change services. No training, video, actor/perception expansion or physical
motion. Synthetic scratch responses do not establish plant stability.

The completed caller-cost diagnostic exercised cost but reproduced existing
force/state bytes. Before further simulation acceptance, require a positive
force/state transition exercise and a row-cost oracle. Keep conditional
arithmetic consistency separate from native provenance and learned skills.

## Frozen response model

Source anchors, verified as whole installed bytes before the CLI audit:

- `solver.py`: SHA256
  `bba0c67182ade84f5375d6a066048e111edd3371b33d46a6f1246349f22bb30a`.
- `math.py`: SHA256
  `4c58d5e3864f7b841633d403d78e7702bb547359ee846d43810e3b31c63966c2`.
- `types.py`: SHA256
  `8f0b19d7b2bc039a419fa547ba6b732837327c0735f688ad34ace5afc4843e50`.

Use the original `_update_constraint(False)` row routing, all-active 64-world,
512-row storage, finite exact binary32 inputs and no active elliptic rows.
This bounded model requires stiffness `D` in `[2^-20,2^20]`, residual magnitude
at most `2^20`, frictionloss in `[0,2^20]`. Reject zero/subnormal/negative
stiffness, negative frictionloss, nonfinite/unrepresentable inputs and invalid
active row types. The actual `safe_div` substitutes MJ_MINVAL only for zero D;
that fallback and elliptic cone formulas are deliberately outside this model.
No zero-D case is silently treated as ordinary division.

| Row route | Condition | State | Force | Source cost term |
| --- | --- | --- | --- | --- |
| equality | row below ne | QUADRATIC=1 | -D*Jaref | 0.5*D*Jaref*Jaref |
| friction | Jaref <= -rf | LINEARNEG=2 | +frictionloss | -f*(0.5*rf+Jaref) |
| friction | Jaref >= rf | LINEARPOS=3 | -frictionloss | -f*(0.5*rf-Jaref) |
| friction | between thresholds | QUADRATIC=1 | -D*Jaref | 0.5*D*Jaref*Jaref |
| non-elliptic remainder | Jaref >= 0 | SATISFIED=0 | literal +0 | no atomic add |
| non-elliptic remainder | Jaref < 0 | QUADRATIC=1 | -D*Jaref | 0.5*D*Jaref*Jaref |

Here `rf` is the explicitly RNE-rounded `f/D`. Inclusive negative comparison
wins first, including f=0, Jaref=0. Force sign-of-zero is modeled explicitly;
unilateral negative zero is satisfied. Equality and quadratic costs use source
left-to-right rounded products. Friction cost retains both rounded half-rf
followed by addition, and fused half-rf-plus-residual models, then rounded
outer multiplication. This is not proof of compiler contraction, division,
reassociation, FTZ behavior or driver code. Gradual underflow is a stated model
assumption. Preserve signed zero in row term models, including negative-zero
frictionloss and threshold division; nonnegative adds initialized at +0 must
not produce a -0 aggregate under this model. Force/state outputs are checked as
exact bits. These conditional checks do not observe device rounding or timing.

## Arbitrary-order accumulation envelope

For each world, keep the two finite nonnegative modeled row terms. Let L/H be
the exact rational sums of per-row minima/maxima; let n be the number of atomic
terms, at most 512, and u=2^-24. Any serial order of the modeled nonnegative
binary32 RNE adds is checked within:

`[max(0, L-B), H+B]`, where `B = n*u/(1-n*u)*H + n*2^-149`.

The conservative n includes the first addition to positive zero. Reject unless
H+B is below the round-to-infinity midpoint `2^128 - 2^103`, so partial rounded
prefixes remain finite under the model. Exact Fractions decide acceptance;
serialized display endpoints do not. Tests include different accumulation
orders yielding different bits while both remain inside the envelope. This
does not qualify CUDA atomic scheduling or prove arbitrary generated arithmetic.

Complete EFC before/after banks must preserve all non-output bytes and inactive
force/state tails. Initial cost must be literal positive zero. Check every
active row force/state and each world cost; retain mismatch counts and branch
coverage. No claim that equilibrium replay changes forces.

## Prospective positive control

Control protocol `microduck-efc-positive-transition-control-oct9-v1` derives
exactly thirteen overrides from the sealed complete historical packet. Retain
all counts, types, IDs, J matrix, inactive tails and other arrays. Never replace
the historical reference or mutate its anchors.

- Reuse the seven declared Gauss-only controls unchanged: four drivers and
  Gauss/cost/previous-cost canaries. Expected twenty-DOF Gauss remains 10.
- Add five active-row overrides: D=2; friction-row f=2 (threshold 1);
  Jaref cycles through {-2,-1,-0.5,0,0.5,1,2} for friction, {-1,-0.5,0,0.5,1}
  for non-elliptic remainder and {-1,0,1} for equality, offset by world+row;
  force canary `-4096-world*512-row`; state canary CONE=4. The canary state
  does not turn these rows into elliptic contacts and is not a physical state.
- Dense output canary is `-8192-index` for each of 1280 DOFs.

Require every active force/state canary to be replaced, dense output to change
in every world, exact dyadic aggregate EFC cost and subsequent Gauss addition.
The historical recipe has ne=0, nf=14, nefc=46 in every world: 2944 active rows.
This control covers negative/positive saturated friction, quadratic friction,
satisfied/quadratic pyramidal contacts. Equality is tested as a row formula but
not inserted into this historical control. All dyadic row costs are multiples
of 0.25 with small sums, making every serial accumulation order exact for this
specific control. Do not generalize that exactness to other inputs.

Authenticate all eight complete boundary packets before decoding any; initial
bank must equal the historical 24-field bank plus exact thirteen overrides.
Reuse the existing unchanged four-stage bank analyzer for write sets,
continuity, setup/previous-cost transfer, tails, dense consistency and Gauss.
The new positive receiver adds EFC row/aggregate checks and canary replacement
gates. Pure fixtures can satisfy it: capture/native/physical qualification stays
false. A future native owner must retain actual complete boundaries separately.

## CPU audit and tests before native work

The CLI requires explicit hidden CUDA, clean exact source and the new three-path
fence. It hashes every committed blob, reauthenticates the prior complete native
inventory and external retirement through the old receiver, then hashes all
four additionally re-read EFC banks before additional arithmetic decoding.
It independently reproduces the old receiver output before adding this oracle.
Pinned prior source is `3f6ab6919db420b72de2b232de7eb0f2c150af6f`; complete
inventory SHA256 is
`69735a75f2185a43a03b5f955d284863f661edc4e6e5de72bf81d513860ba54d`, and
old receiver SHA256 is
`7c94ba163d484578ea8f448567ec55480a283f16169032e6c6238afbcd69b290`.
The oracle is a new interpretation of existing native data, not a new native
positive-transition run. `positive_transition_gpu_run` stays false.

Run the eighteen files in `stance_solver_efc_transition.TESTS` on Mac and WSL
with CUDA hidden, at the exact committed source. Retain paired XMLs and CLI
reports under `artifacts/tools/efc-transition-preparation/`. Require matching
collection counts and zero failures/errors/skips; this is not the full suite.
New tests cover branches/ties/zero signs, explicit unsupported domain,
underflow, finite bounded extremes, order-dependent accumulation, overflow
refusal, whole-byte/initial/tail/write-set guards, canary changes, all hashes
before decode, retained response audit and inert import. The dyadic prospective
fixture is generated from independent literal equations, not the oracle; a
second test executes the authenticated original kernel factory Python body
with CPU-only fake Warp/NumPy carriers and checks whole output bytes. It neither
compiles nor loads a CUDA kernel. Real CPU enum values are checked separately.

## Next native gate

The old two-arm native receiver requires paired non-Gauss outputs to agree,
which this new force-changing control intentionally violates. Its source fence
and semantics cannot be reused for this control. Before any native positive
run, prepare a new separately fenced owner/receiver using the unchanged
explicit module loader and caller observer. It must stage all thirteen fields
including int32 state, preserve exact fresh-module/source/runtime binds,
lease/capacity/deadline/resource caps, complete native inventory and external
unit retirement; independently receive both reference and transition arm, and
retain the new response/positive gates. No untested runtime monkeypatch of the
old collector or receiver. Broader solver-window/plant acceptance and motor
checks remain before curriculum training.

## Retained CPU preparation closeout

Tested source `383909def6380b910df5331a06b38766014f7287`, tree
`74e1c37b7cd0a7fe39ff18848426b57ddfb9270f`, was pushed to the fork and
fast-forwarded by a verified incremental bundle into the clean WSL worktree.
The exact three-path preparation fence and all committed source bytes were
checked independently by each CLI. No old protocol or runtime was changed.
The bounded read-only Luna arithmetic review identified signed-zero edge cases;
owner integration added explicit sign handling and three regression cases.

All **789 tests in eighteen focused files passed on both Mac and WSL**, with
CUDA explicitly hidden and zero failures/errors/skips. Mac pytest reported
92.97 seconds; WSL 61.81 seconds. This is not the full repository suite.
Retained under `artifacts/tools/efc-transition-preparation/`:

- `383909de-mac-tests.xml`: 928944 bytes, SHA256
  `dd5fa9d0a9080a3a2aa1d667962068ed2f27b7f44211319fdabfb4466152d578`.
- `383909de-wsl-tests.xml`: 928948 bytes, SHA256
  `4d47d4e5b2d9080de3b1358e05ba4a0773b2024578884a5d37ba3553dc9650b3`.
- `383909de-mac.json` and `383909de-wsl.json`: each 278485 bytes, byte-identical
  SHA256 `c436e79faafb5b8b33051a5d1377ff29c5f2d34fdc900b4868f3363c6d0e917e`.
- `383909de-source.bundle`: 16789 bytes, SHA256
  `339091160046fae91c2ada59361536b6b015d08d98cffcb1642c775094be8d67`.

Each independently received historical arm has 2944 active rows, zero force
and state bit mismatches, and zero world costs outside the conditional interval.
Each has **zero force/state transitions**, as expected for the previous native
capture: 896 quadratic friction and 2048 quadratic unilateral rows. The new
CPU fixture replaces all 2944 active force/state canaries and changes dense
output in all 64 worlds, covering five response branches. That fixture and the
authenticated kernel-body CPU stand-in are not a new native transition capture.
The CLI decision is `retained-efc-response-model-audited-not-native-transition`;
`positive_transition_gpu_run` and all six qualification flags remain false.

The CPU-only unit `microduck-efc-prep-cpu-383909de.service`, invocation
`75ecddff890d434da393f2620c85f16a`, completed successfully. Externally checked
MainPID=0, ExecMainStatus=0, Result=success, NRestarts=0, empty ControlGroup and
absent original kernel cgroup. Its retained active/exited state is not a running
workload. Caps were 180 seconds, 4 GiB memory, 200 percent CPU, Nice=10,
TasksMax=64, LimitFSIZE=16777216, KillMode=control-group and no restart.
No Duck user unit remains running; no GPU lease or device kernel was used here.

Fresh runtime verification passed without package/driver changes. Before and
after CPU validation, FilmBrain observatory PID 521 and video playground PID
298048 remained active with NRestarts=0; both protected AI mission services
remained inactive in user and system scopes. At the post-check, Windows and WSL
both reported GPU UUID `GPU-7d72b360-33bc-2cee-3ff4-a954474011b5`, driver
595.95, 8151 MiB used / 16011 MiB free, 2 percent utilization and 34 C. These are
separate point-in-time counters, not simultaneous or future launch admission.

Next is the separately fenced native owner/receiver preparation above. The
positive force-changing GPU test, broader solver window, plant/motor stability
and curriculum training are still outstanding. No learned capability is added
by this preparation.
