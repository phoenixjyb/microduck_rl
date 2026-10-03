# CUDA64 private-state shadow sampler predeclaration

Status: **native fixed-input capture/replay independently closed, non-admitting**.
The original predeclaration below was executed at exact source
`1b96ccaac1326d6f0a1e00cdff8950b91a2916be`; its separately retained outcome is
recorded at the end of this document. No environment rollout or weight update
was performed.
This is a wiring diagnostic after the independently closed
[CUDA64 preparation](2026-10-03-cuda64-policy-preparation-probe.md), not Duck
training, a rollout, a curriculum promotion or physical permission. Authorization
ends October 3 at **20:00 Asia/Shanghai**. The old 08:00 window and all old closed
or failed namespaces stay immutable. FilmBrain is preserved; protected AI Mission
services are not restored.

## Fixed question and scope

Does the pinned stock RSL 5.0.1 CUDA sampler consume the installed private CUDA0
state reproducibly while leaving both caller streams and the frozen D1 weights,
empty Adam and zero rollout storage unchanged? Merely creating a separate
generator did not wire the stock sampler. The reviewed
[RNG boundary](2026-10-03-private-cuda-sampler-boundary.md) installs its state only
inside an isolated, process-global non-reentrant scope.

Use the exact D1 parent checkpoint
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`,
transferred actor/critic state
`e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329`,
64 rows, 28 calls, actor `[64,44]`, critic `[64,50]`, actions `[64,10]`.
All observations are typed finite **synthetic zeros**. There is no environment,
physics step, reward, fall/thermal motor measurement, terminal/reset bridge,
storage write, return calculation, optimizer update or export.

Four **fresh Python CUDA0 children**, in this fixed order:

1. Seed 653 capture.
2. Seed 653 independent fresh replay.
3. Seed 659 capture.
4. Seed 659 independent fresh replay.

Each child invokes the unchanged stock bound `PPO.act` exactly 28 times. Its
prepared generator object stays separate and unadvanced; a cloned private state
is installed around those calls and advances at every actual sampling boundary.
No inherited learner, stale transition or reused Python process is accepted.
Call clearing only discards the transient RSL transition after retaining it;
`process_env_step`, rollout insertion and optimization are forbidden.

## Source, startup and resource gates

The native target is only the frozen clean branch
`feat/athletics-obstacle-curriculum` in
`/home/yanbo/work/microduck_rl-stance-replication-20260930`, via `gw98-direct`.
Require exact machine, GPU UUID, driver 595.95, frozen package/runtime and robot
asset checks. No package or driver installation is authorized by this protocol.
The new experiment source must first be committed, pushed and independently
matched to the fork. Fast-forward the idle native checkout only under the shared
lease and preserved-context checks; never modify an active job.

The new [closed preparation reader](../../src/mjlab_microduck/stance_recovery_cuda_preparation_evidence.py)
rehashes the old whole verification, journals, raw tensors and successful
terminal units, then freshly authenticates the unchanged provenance chain.
Only its two explicitly separate evaluator source IDs may differ. The old failed
preflight wrapper and all six failed-service bindings remain retained unchanged.
The old successful Boolean is not new execution admission.

All CPU parent construction and consistency rescoring start with CUDA hidden.
Native CUDA children start with exactly `CUDA_VISIBLE_DEVICES=0`,
`MICRODUCK_STANCE_PROFILE=wsl-10098-20260930`, `OMP_NUM_THREADS=1`,
`ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`, `PYTHONUNBUFFERED=1` and the
frozen venv path. The child environment uses the existing credential-free
allowlist; inherited preload, Python-path and connector credentials are excluded.
Each child proves the inherited shared lease **before** CUDA access. No unrelated
Python RNG caller or second Python thread is supported in a sampling scope.

Hard service limits, all `CPUQuota=200%`, `Nice=10`, `KillMode=control-group`:

| Phase | Runtime cap | MemoryMax |
| --- | --- | --- |
| Actual hidden-CPU preflight | 180 s | 2 GiB |
| Exact declared native source tests | 240 s | 2 GiB |
| Four sequential CUDA captures/replays | 600 s | 3 GiB |
| Independent hidden-CPU closeout | 180 s | 2 GiB |

Each CUDA child has its own **120-second** wall-clock cap. Stream its entire log
under **1 MiB**; each weights-only raw payload is at most **8 MiB**, each JSON
record at most **2 MiB**. On timeout, oversized output or failure, kill only that
owned child process group, retain the failed/partial record, and diagnose
read-only. There are no automatic retries, success relabelling or failed-unit
resets. The independently capped service is another boundary, not a substitute
for the child cap.

Require **840 seconds** remaining before launching the CUDA run (600 + 180 + 60
margin). Preflight additionally reserves the complete 180 + 240 + 840 seconds.
These are hard budgets, not throughput estimates. Start nothing that cannot
finish and close before 20:00; do not extend an expired protocol or run GPU jobs
concurrently. The shared FilmBrain lease is never deleted or replaced.

Require two actual idle GPU samples before preflight, the run, every child and
closeout, and again afterward. While a child runs, only its PID may own compute;
sample GPU temperature below **75°C**, total memory at most **5 GiB**, free memory
at least **6 GiB**. Preserve the exact FilmBrain service states/PIDs/restarts and
both system/user protected-service states across every source/context gate.
Sampled temperature/memory is not a continuously measured peak or motor heat.

## Retained records and deterministic decision

The new output namespace is
`artifacts/evaluations/stance-wsl-cuda64-shadow-<source12>`; require it unused.
The actual CPU preflight and declared test receipt have separate tools namespaces.
Source binds both unchanged old gate leaves and every new reader, RNG, sampler,
scorer, probe, focused test and this predeclaration. The source suite is the exact
25-file list in
[stance_recovery_cuda_shadow_probe.py](../../src/mjlab_microduck/stance_recovery_cuda_shadow_probe.py),
with **679 passes**, zero failures and zero skips required; retain its whole log
and original terminal invocation before launching CUDA.

Each child retains a whole before-sampling preparation subpayload and the full
sampler output: all actual inputs, raw unclipped actions, values, log probabilities,
Gaussian mean/std, 29 private CUDA state boundaries, four **new actual** caller
RNG snapshots, before/after model tensors, empty optimizer/storage snapshots and
scope receipt. Persist raw bytes before a scored success summary. No child may
claim that it authenticated historical CPU provenance.

The pure [CPU scorer](../../src/mjlab_microduck/stance_recovery_cuda_shadow_evidence.py)
checks typed finite tensors, zero inputs, positive Gaussian scales, fixed-input
output stability, actual parent equality, empty Adam/storage, source/seed/receipt
bindings and every private/caller boundary. Gaussian log-probability algebra is
cross-checked in CPU float64 under a fixed **2e-4 absolute** tolerance; this is
not replay of CUDA arithmetic. Each preparation subpayload is also scored by
the unchanged earlier preparation scorer, not a metadata-only substitute.

For each seed, the paired sampling payloads must match exactly in every field
except `caller_rng_states` and `receipt.caller_rng_state_sha256`. Fresh processes'
OS-initialized caller starting bytes may differ, but **each is separately proven
unchanged**. No other output, model/optimizer/storage tensor, scope receipt or
private sampling state is excluded. The private initial state is tied to the
actual fresh preparation and all 28 advances must be distinct.

After all four children finish, retain the complete fourteen-file raw/JSON/log
inventory and run report. A separate hidden-CPU closeout service rehashes all
fifteen files, freshly checks source/provenance/tests/actual CPU construction,
rescans all raw tensors and repeats both exact pair decisions. Its own record
is the sixteenth file. Independently retain original process journal invocation
IDs and whole journals, current terminal unit evidence, source/test/preflight
hashes and final idle/service state before any native sampler qualification claim.
Pure record equality alone is not native replay authentication.

All eight capability flags remain false, even after a successful sampler result.
A pass proves only fixed-input native CUDA sampler/RNG wiring under this source,
host and input protocol. It does not prove 64-world physical feedback, GAE,
optimizer replay, learner speed, a new balance skill or the whole curriculum.
No-deficit frozen-parent screens still preserve that parent without updates.
The next transition-aware learner and lesson-assignment gates require separate
source review, actual measured deficits and their own predeclarations.

## Source checks

The complete owner 25-file regression passed **672 tests in 77.73 seconds**,
CUDA hidden, no skips. The owner sampler/scorer focused suite passed **74 tests
in 7.99 seconds**; the final capped-probe focused suite passed **41 tests in
11.71 seconds**. The bounded Luna sampler refinement passed **23 tests in
7.27 seconds**; its read-only harness review found the two issues below, then
confirmed the fixes and predeclaration with no remaining hard blocker. Ruff
0.15.7 check/format, compilation, relative-link and whitespace checks passed.
Positive tests
use explicitly synthetic CPU sampler envelopes and mocked CUDA APIs, even where
they include authentic old parent/storage tensors. They do not admit this native
experiment. Review corrected explicit child startup math settings and retained
closeout failure handling before integration.

## First external source-sync wrapper failure, retained unchanged

The source-only wrapper aimed at `5eca76e0b85decfd7c7adfaf520355ba8c35df69`
failed before changing the frozen `48653c58121dec9a2b6ae2db059fc4d0911410cf`
checkout. It omitted systemd's working-directory property; the actual unit started
in `!/home/yanbo`. `git rev-parse HEAD` exited **128**, with the whole journal's
exact message `fatal: not a git repository (or any of the parent directories): .git`.
No CUDA allocation, optimizer step, source fast-forward or protected-service
change occurred. Its measured body elapsed 0.033342347 seconds; systemd reported
11.148 seconds including import/startup, 1.0 GiB memory peak and zero swap.

The failed original unit
`microduck-cuda64-shadow-source-sync-5eca76e0b85d.service` stays failed, exit 1,
PID 0, restarts 0, invocation `c6e0d08a618244c5badd9d4bf6333643`. Original receipt
SHA256 `d88fcfe9a5e29bb1483a1c953c9cd8a7e1897dc00bdda952eefe296bc0a0bc78`
(887 bytes); whole retained journal
`a123208f263b4f79782965f2a2e0211deadfd64e49988ed4ff619ce582a55e51`
(100,804 bytes). The independent failure-retention receipt is
`44c2b12a3553da6fa177b496672f7c4f0454e031684bcd36c31154a77bea3abb`.

The diagnosis was read-only before any correction. The smallest correction is a
new source-sync namespace/service with **explicit `WorkingDirectory` equal to
the exact frozen checkout** and an early actual-directory/property guard, plus
the same 60-second/2-GiB/leased/idle/fast-forward-only limits. Do not restart,
reset, overwrite or relabel the failed original. All subsequent shadow source
bindings authenticate the whole failed receipt/journal and unchanged original
terminal unit, separate from the older closed preparation's six failure bindings.
Every later preflight/test/run/closeout command must likewise pass the explicit
working-directory property. The initial 672-test source result remains true for
its original revision; the new seven failure-binding cases require a new complete
679-test source result before this corrected revision may launch CUDA.
That complete owner regression subsequently passed **679 tests in 80.18 seconds**,
CUDA hidden, no skips; the final 48-case probe suite passed in **7.24 seconds**.
Ruff 0.15.7, compilation, relative links and whitespace checks passed. A bounded
read-only Luna review found no remaining hard blocker in the new failure binding.

## Native outcome and independent retention

All four fresh CUDA0 children completed in the declared order, PIDs
2373842 / 2374104 / 2374366 / 2374628. Each retained 28 stock `PPO.act` calls,
29 private-state boundaries and its own preserved actual caller streams.
Both seed pairs matched exactly except the two predeclared caller-state fields;
there were no additional exclusions. The unchanged D1 weights, empty Adam and
zero storage were authenticated before and after. There were **zero simulation
steps, zero optimizer steps and no new checkpoint or accepted Duck skill**.

The hidden-CPU native 25-file source suite passed **679 tests in 41.57 seconds**,
with no skips. All five new original services finished successfully, exit 0,
PID 0, restart count 0. Their whole retained process journals authenticate the
original invocation IDs even after transient-unit garbage collection:

| Phase / unit suffix `1b96ccaac132.service` | Original invocation | Body elapsed | Systemd elapsed / peak memory / swap |
| --- | --- | --- | --- |
| `microduck-cuda64-shadow-source-sync-` | `d1264b46097446d7b9ce618f2cba74ba` | 3.968 s | 12.369 s / 936.8 MiB / 0 |
| `microduck-cuda64-shadow-preflight-` | `c2846242d34349d6b5d8290838e9d278` | 8.652 s | 16.223 s / 1.7 GiB / 0 |
| `microduck-cuda64-shadow-tests-` | `0e5ec4bd16e340eda5d0725b08212939` | 58.353 s | 65.803 s / 2.0 GiB / **2.1 GiB** |
| `microduck-cuda64-shadow-run-` | `a4e6238c17a142e6b2d3fb97ad0aa169` | 54.463 s | 62.300 s / 1.6 GiB / 0 |
| `microduck-cuda64-shadow-closeout-` | `888087ec299742ed90f66b0e652baf4f` | 8.083 s | 15.764 s / 605.7 MiB / 0 |

The test phase did use swap; the successful CUDA run did not. Body and systemd
startup-inclusive times are distinct. All stayed inside their predeclared
service caps. Sampled GPU maxima were **32°C** and **992 MiB total occupancy**;
afterward it was idle at 0%, 691 MiB, 31°C with no compute PID. These are sampled
GPU observations, not continuous peaks or motor-thermal measurements.

The independent closeout rescored all raw preparation and sampler tensors and
both exact pairs. A separate CUDA-hidden external verifier then repeated whole
inventory, original journals/invocations, current prerequisites, source bundle,
native test log, failed-unit preservation and raw decision checks in **8.626 s**
under its 180-second cap. Decision:
`fixed-input-cuda64-shadow-closed-non-admitting`.
Only `native_fixed_input_cuda_capture_replay_verified` is true;
`cuda_math_replayed_by_cpu_verifier`, transition/optimizer qualification and all
eight capability flags remain false. CPU algebra/equality checks are not an
independent replay of CUDA mathematics.

Native namespace `artifacts/evaluations/stance-wsl-cuda64-shadow-1b96ccaac132`
contains **16 files / 17,246,164 bytes**. Whole-file SHA256 anchors:

| Retained record | SHA256 |
| --- | --- |
| `launch.json` | `73c27d905023b19a7713dedf2fc4637f16f6eee7f123107533e611b0b480acf6` |
| `report.json` | `553787df5fa3db6260b7b6d417d781309b5d6578d37475652719d3532dfb99ad` |
| `independent-closeout.json` | `5893050238944cc006fea386549d7aa5975b747a1a537020043bd1794ccddc83` |
| External verification `receipt.json` | `38c3714d23b1b3c217027a623960e0232654109f4b87d85754edeec1e1bb7460` |
| Corrected source-sync `receipt.json` | `8141d50e0f95545ae29abc4f44301745e3a23b2e6cbdff5178446d86bc750395` |
| Actual native preflight `receipt.json` | `66502426ebe18eb2e955a36cb7c1a0594c04474d06e6fefb22c309f604ec1ebd` |
| Native test `receipt.json` | `708e6fbde967c2a6ab6ff315dbaea71fc90ed012302d72a6448089942f0ed726` |
| Whole native `pytest.log` | `09fc1049541edc76753e63d41cba06b71a3770cc137df1772c419ee56f2d41c8` |

The external receipt pins every raw/JSON/log file, six source/preflight/test
files and five whole original journals. Both native originals and a separately
copied Mac mirror are durable. The mirror at
`artifacts/retained/cuda64-shadow-closed-1b96ccaac132.GZesKH` was independently
rehashed against those pins and the unchanged failed original source-sync
receipt/bundle/failure journal/receipt: **all 32 files / 18,319,761 bytes match**.
The original failed unit remains failed and has never been reset or retried.
FilmBrain's two original services/PIDs/restart counts stayed unchanged; both
system/user AI Mission services stayed inactive.

The next open boundary is terminal-aware CUDA transition/storage integration,
including pre-reset critic observations, exactly-once timeout bootstrap,
raw-action storage and store-before-reset ordering. Fixed-input sampling does
not close that boundary or justify updating a passing no-deficit parent.
