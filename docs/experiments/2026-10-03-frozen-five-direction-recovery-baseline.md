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

## Retained WSL closeout, October 3 approximately 06:16 Shanghai

Exact source **e9b9d498363714deda1214026724aa41382170a9** completed as
`microduck-wsl-d1-frozen-five-case-e9b9d4983637.service`, invocation
`e2d6a2fc349c4334885df7cf995c5f5d`. The service finished successfully in
**952.0004726229236 s**; its sole CUDA child PID 2099988 returned zero after
**858.8091502389871 s**. All five fresh cases completed 250 policy ticks /
2,500 physics steps. The independent parent score took **33.45811812905595 s**
and returned **`frozen-five-case-baseline-passed`**. All pre-push state/action/
control prefixes matched exactly, including the first pre-force motor proposal;
reactive post-push actions were deliberately not required to match.

| Case | Maximum displacement (mm) | Final-second speed p95 (m/s) | Final-second tilt p95 (rad) |
| --- | ---: | ---: | ---: |
| zero | 2.404384 | 0.000166343 | 0.019774206 |
| +x | 2.404384 | 0.000140866 | 0.020894026 |
| -x | 3.144949 | 0.000011691 | 0.020241946 |
| +y | 2.585765 | 0.000078983 | 0.021109734 |
| -y | 2.404384 | 0.000691307 | 0.019984104 |

All nine unchanged numerical gates passed in every case: complete first attempt,
full duration, no hard failure, displacement, final height/speed/tilt, foot
support and soft-limit exposure. Both-foot support fraction was **1.0**, soft
limit exposure **0.0**, actor replay maximum absolute error **0.0**, and final
minimum heights ranged **0.117257–0.117372 m**. Across the complete trajectories,
maximum planar speed was **0.035144307 m/s**, tilt **0.024127312 rad**, motor
torque **0.115199760 N m**, and absolute joint mechanical power **0.037765842 W**.
Torque/power are recorded mechanical metrics, not thermal predictions or new
numerical acceptance limits.

The 676 ownership samples showed only the recorded child or no CUDA process,
maximum GPU temperature **46°C**, maximum memory used **1,104 MiB** and minimum
free **23,058 MiB**. Both protected services stayed inactive in both namespaces;
FilmBrain service state/PIDs/restart counts matched the retained launch. Two
post-child idle samples were 0% utilization, no compute PID, 663 MiB used and
37/36°C. The shared lease was retained through independent scoring and drain.

Before installation, exact-source WSL default-math regression passed **403 tests
in 62.54 s**, invocation `67cef5acead34934ad83fcf2463e0f00`, under the declared
150-second / 6-GiB cap. Six fresh actual portable-profile suites passed
**51 / 24 / 16 / 36 / 54 / 37 tests** respectively (218 total), each under
120 seconds / 2 GiB. Their invocation IDs in contract/runtime/trace/probe/matrix/
campaign order were `adfcd63e2e4d495687952c9b74c45558`,
`9a76f19bfa82452b93d3078f4ead5009`, `1519e67dfd824e13be0028ca647431b6`,
`8ef2ffa1ba9a46539b672aa9a06e6cb6`, `4a2785400bca43fe89152ddc173b02d2`,
`c8deed295f484e38b7b31dcded9a0846`. CPU preparation invocation
`e0ad7a6ca5b841f7928eec2e71bbd6b8` completed the 52-tick / 520-step
qualification with exact actor replay in **7.654890833888203 s**. Some short
service memory summaries were implausibly small; they are not used as measured
RSS or to justify changing a cap.

Retained root on 100.98:
`/home/yanbo/work/microduck_rl-stance-replication-20260930/artifacts/evaluations/stance-wsl-d1-frozen-five-case-e9b9d4983637`.
The separate CPU-prefix root is
`artifacts/evaluations/stance-wsl-d1-frozen-recovery-probe-e9b9d4983637`;
that directory is preparation only, not another successful +x GPU probe.

| Artifact | SHA256 |
| --- | --- |
| matrix launch | `a2e4baae4c052c42d5feffa6e03052dab5828a5e0960e7b601fb32a0718d8d90` |
| comparison | `ffb3d64c07c3ed502afb00555b116509b85922ac927a2803adb38d8d330b0715` |
| report | `f9b659da15f53f276f0dedbe61e9afcaf98b8c9f3602ca5171a5ffb8ee798a84` |
| independent closeout | `a44d87779cca18b3e4df899095126d85d9958472469b2fe23c7c99ad73e38f15` |
| common pre-push prefix | `8312886d127dcccfb59e04f6eef1ff292c6a4dfae87c29d736d0d55f3098c207` |
| case-0.pt, 37,597,000 bytes | `d05163fa57687f7f72ed505b1733e60b40239573fa4281856151669e7cd64d98` |
| case-1.pt, 37,607,688 bytes | `80ba86b9411c239099f1945e7f661b05d9ffbf82aae33d2f7569f7df7774f91f` |
| case-2.pt, 37,603,080 bytes | `cbbb6b8f2c88cc903b1ce226f9eee9ce7bef7f6f6267b5223ff23054899b3bca` |
| case-3.pt, 37,604,616 bytes | `30ee05b26905eeada6a443b6769a57b810da064398d06662cf07848db6c30211` |
| case-4.pt, 37,606,152 bytes | `f191fb3b630bd19fc4422afc3811f5e9ab1c1a32d2e33f8a03356a8fd76f795f` |
| prefix launch | `eaeb009e25b4c1937f01d4f86775ec55b8e6f9dcc8a316c62094412d63dfe026` |
| cpu-prefix.pt | `df2eaabe3fe14f9a5be818f04c8ba6b8f3d389216a058aeb587feed942aaa6b2` |
| CPU qualification | `1ed8530be07b3a8890eda805ad4a60f31bedfe26b873291b856b2183104dc71b` |
| selected checkpoint | `2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5` |

A separate CUDA-hidden whole-case rescore rehashed all **19 original matrix
files** (including the report) and **four CPU-prefix files**, recomputed all five
receipts and the deterministic comparison, and found exact equality. Its actual
portable-profile service `microduck-d1-matrix-independent-cpu-r2-e9b9d4983637.service`,
invocation `585c23c8af13443592ea791573283e02`, finished successfully in 40.608 s;
the retained check itself took **33.39823420206085 s**. Its
`independent-closeout.json` contains the full size/hash inventory and idle/state
evidence. Two earlier verification-wrapper attempts remain in the journal:
invocation `655d43c62f2148d1af88ef31bbdc94cb` exited 127 before Python because
the relative interpreter was resolved outside the repository; invocation
`f390f43970f244b2935b85ff8c1c5477` refused the absent historical-default-host
evidence path because the explicit WSL execution profile was omitted. Neither
ran a GPU child or changed matrix evidence. The final attempt specified both
working directory and WSL profile; no source, resource cap or numerical gate
was relaxed.

This closes only a **fixed, small-push baseline**, not randomized held-out
recovery, a newly learned skill, B1 graduation, ball support, complete binary
equivalence, independent GPU attestation or physical motion. All eight admission
flags remain false. The [source-only next lesson design](2026-10-03-progressive-recovery-lesson-design.md)
screens new timing/dose cells before deciding whether further training is needed.
