# CUDA64 private-state shadow sampler predeclaration

Status: **owner-reviewed and source-tested predeclaration; no native job launched**.
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
with **672 passes**, zero failures and zero skips required; retain its whole log
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
