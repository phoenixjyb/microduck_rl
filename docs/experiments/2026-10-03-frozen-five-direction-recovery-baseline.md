# D1-M: frozen five-direction baseline predeclaration

Declared before matrix outcomes. This is measurement of the existing frozen
seed-577 / iteration-255 actor, **not a new training job or recovery admission**.
Use the unchanged [D1 policy, pulse, gates and evidence contract](2026-10-03-frozen-push-recovery-baseline.md).
Protocol `football-b1d-frozen-five-case-baseline-v1`; ordered fresh one-world
cases **zero-wrench, +x, -x, +y, -y**, fixed evaluation seed 619. Each starts
from the same reset and freshly restored CPU actor. No optimizer, checkpoint
selection, observation expansion, perception training or physical motion.

## Matched causal comparison

Authenticate each complete case's bytes before CPU tensor loading; independently
verify every actor action, plant/state/control accounting and scheduled force
phase under the unchanged numerical scorer. Compare typed bit-exact hashes of
the entire initial state, first 50 unforced ticks/control records, and step-500
pre-force state, actor input/action, initial/after-action control and first
motor proposal. **Do not compare post-push actions or states**: reactive policy
responses are the outcome. All five prefixes must match. Require complete
2,500-step cases and all ten pulse-window phase checks (zero-wrench still has
ten explicitly zero-force window steps), plus a passing zero control. All five
passing gives `frozen-five-case-baseline-passed`; any numerical failure gives
`frozen-five-case-baseline-rejected`; malformed/partial/mismatched evidence
gives a failed diagnostic, never acceptance. CPU scoring is not fresh complete
physics re-simulation; BAM outputs are not independently recomputed.

## Measured bounded budget

Use only the successful R1 report at source
`a44c1d70803012acf791de995a3d384c51162955`, SHA256
`36a4473f2dded6be55e1402b63efd4377f2b0b53127fd1af5ea4d45e664d7e3a`,
with its launch and complete capture bytes separately pinned. Recheck its
full-duration score and all admission flags before deriving any budget.

Repeating case C = construction + collection + serialization =
**166.40493231918664 s**. Fixed child entry E = recorded elapsed - C =
**24.82244214182718 s**. Wrapper W = supervised child elapsed - recorded
elapsed = **2.71480189403518 s**. Five-case predicted child = E + W + 5C =
**859.5619056317956 s**. At 1.25 multiplier, round up to **1,075 s**.
The existing fixed **1,352-second lean-evaluation wrapper** is the smallest
existing watchdog among 120 / 900 / 1,352 / 1,693 seconds that covers this
bound. Reuse it unchanged; do not widen the 120- or 900-second wrappers.

Each fresh case has a separate **250-second cooperative collection/retention
budget** (ceil(1.5C)); successful retention must verify its elapsed time below
that limit. The **1,352-second child watchdog** supplies hard native-stall
protection; the per-case checks do not independently interrupt a stuck native
call. Reserve 60 seconds for serialization and 120 seconds at the overall child end. Refuse
another case unless 250 seconds remain. Retain every normally returned partial
or failed capture, then stop without resets, retries or cap increases. A collector
exception before return cannot expose its local prefix: retain an explicit
`case-N-failure.json` with `current_case_trace_retained=false`, the child log and
all earlier completed cases; do not claim a trace for that failed case. A native
stall killed by the hard watchdog may leave only the log and earlier durable
cases. Release closed case Python/native
references and collect garbage only between completed cases, never during an
active simulation. Use one CUDA child/context, one world at a time; no overlapping
GPU job. Each case remains bounded by the existing 128-MiB capture limit.

Measured parent overhead = 244.01035961904563 - 193.942176355049 =
50.06818326399663 s. Five independent CPU rescores add four times
6.111677116947249 s; predicted parent 74.51489173178563 s. Add a separately
declared 60-second prefix-comparison reserve and 1.5 multiplier, round up:
**202 seconds**. Service **1,560 seconds / 26 minutes** covers the fixed
1,352-second child plus 208 parent seconds. Preserve **2 GiB / 200% CPU /
Nice 10 / control-group**, plus 180-second external closeout and 60-second
deadline margin: require more than **1,800 seconds** before the fixed
**2026-10-03 08:00 Shanghai** cutoff. No automatic retry or larger cgroup.

## Qualification, ownership and closeout

Before worktree installation, exact-source default-math integrated CPU regression
uses the unchanged **150-second / 6-GiB** cap. Each actual portable-profile
suite runs in a fresh **120-second / 2-GiB** service: the existing four D1
suites, the new matrix checker and its new campaign runner. CUDA hidden, initial
exec CPU settings verified, no package/driver change. All services must succeed.
Mac tests and shallow synthetic seams are source checks, not GPU evidence.

Fast-forward the clean exact branch only under the shared lease with two idle
samples and unchanged FilmBrain/protected states. Fresh CPU preparation uses
the unchanged **180-second / 2-GiB** service and 52-tick / 520-step qualification.
Its retained launch binds the exact archive, selected checkpoint, compiled plant,
profile, nominal closeout and D0 physical replay. The matrix launch separately
binds that preparation and measured R1 budget. Every entry rechecks these before
CUDA initialization. The CUDA-hidden parent retains the shared lease until its
sole child and descendants drain, then independently reads/scores one complete
case at a time and compares the authenticated prefixes. Source, FilmBrain,
protected-service, GPU ownership/memory/temperature and fixed-cutoff guards
remain live throughout. No protected service is restored.

Retain immutable launch, five raw captures and metadata, five independent CPU
receipts, deterministic comparison, child log and report with exact hashes.
Report transient/settling/displacement/torque/mechanical-power/soft-limit metrics;
no thermal-model claim. All eight skill/training/checkpoint/physical/attestation
admission flags remain **false**, even if this fixed matrix passes. Randomized
held-out recovery, new training and ball support need separate declarations.
No learner follows automatically. At 08:00 start no new Duck work, verify owned
jobs terminal and evidence durable; leave unrelated/protected workloads unchanged.

## Source review before WSL outcomes

The Mac CUDA-hidden integrated suite passed **403 tests in 33.08 s**. After
adding failure-file inventory hashing, the final focused matrix/campaign suite
passed **91 tests in 7.26 s** (54 checker and 37 runner checks); Python
compilation and whitespace checks passed. Two runner tests exercise collection
exceptions in cases zero and one and verify honest failure receipts, no retry
and preservation of earlier completed cases. Synthetic orchestration seams are
not runtime/GPU qualification. Independent read-only review confirmed the
narrowed retention contract and found no remaining launch blocker. Exact-source
WSL qualifications, CPU preparation and the matrix remain pending at this
predeclaration commit.
