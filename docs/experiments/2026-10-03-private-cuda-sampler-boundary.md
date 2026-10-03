# Private CUDA0 sampler boundary

Status: **owner-reviewed source implementation; native qualification not
performed**. This is the next small integration boundary after the independently
closed [CUDA64 preparation probe](2026-10-03-cuda64-policy-preparation-probe.md).
It does not predeclare a GPU job, a rollout, an optimizer update or a curriculum
promotion. Work remains inside the October 3 **20:00 Shanghai** authorization.

## Why a separate generator is not enough

In pinned RSL 5.0.1, `PPO.act` calls the actor with `stochastic_output=True`.
`MLPModel.forward` updates its Gaussian distribution and calls `sample()`;
`GaussianDistribution.sample` delegates to `Normal.sample`, whose implementation
uses `torch.normal(loc, scale)` without a generator argument. CUDA parameters
therefore sample from the device's **default** generator. Merely constructing
and retaining a private `torch.Generator(device="cuda:0")` does not connect it
to the stock sampler.

These statements come from the frozen installed primary source, not a proposal
to modify RSL. The reviewed PPO, MLPModel and distribution SHA256 pins remain:

| Installed source | SHA256 |
| --- | --- |
| `rsl_rl/algorithms/ppo.py` | `a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c` |
| `rsl_rl/models/mlp_model.py` | `6219ebb3ed4df036dae7ff1c30d0fbb169eaaa659e44a659dd757a292124476b` |
| `rsl_rl/modules/distribution.py` | `4631eae1939dcd6b065d79c3c0128a88ba2c6126d08f46944c8795682a208a1b` |

For the prepared parent, actor/critic inputs are `[64,44]` / `[64,50]`;
actions, Gaussian mean and expanded standard deviation are `[64,10]`;
critic values are `[64,1]`; action log-probabilities are `[64]` before storage
reshapes them. The existing CPU observation helper explicitly requires CPU
tensors. A later CUDA sampler must use a separate typed CUDA observation path,
not relax that helper or relabel a CPU tensor as CUDA.

## Small source callable, not a learner

The new sibling
[stance_recovery_cuda_rng_scope.py](../../src/mjlab_microduck/stance_recovery_cuda_rng_scope.py)
accepts only seeds 653/659 and a fresh seed-bound CPU uint8 snapshot of an actual
CUDA0 generator. It requires the inherited shared GPU lease before CUDA access,
the exact clean WSL worktree/profile/host and an **already initialized** single
CUDA0 device. It must not lazily initialize a device, widen visibility, reseed
global generators, create an environment or touch PPO/storage/Adam itself.

Its context temporarily installs the retained private state into CUDA0's default
generator within `fork_rng(devices=[0])`. It captures the advanced private state
even when the body raises, then restores caller CPU and CUDA0 streams. State
properties return clones; receipts bind seed/source/state hashes and remain
non-admitting. Any error faults the object; a fault cannot be retried as success.

Default generators are process-global, not per-object. Cooperating scope
instances require one shared nonblocking process lock, and reentry/concurrent
use is rejected. A swallowed same-object reentry refusal must still fault the
outer attempt; another object's refused attempt cannot borrow that success.
This is not a general thread-safe Torch RNG API: no unrelated concurrent RNG
caller is supported. A native consumer must be a fresh isolated Python child
with one Python thread and the retained OMP setting. The shared GPU lease is
also still required; a Python lock is not external GPU ownership.

Source tests in
[test_stance_recovery_cuda_rng_scope.py](../../tests/test_stance_recovery_cuda_rng_scope.py)
use an explicitly **synthetic mocked CUDA backend**. They can demonstrate state
handoff, refusal ordering, clone ownership, error restoration and scope locking;
they cannot qualify CUDA generator layout, the stock sampler, physical feedback
or numerical CUDA replay. All eight capability/admission flags remain false.

## Next native gate must be independently declared

Before a real CUDA scope or sampler claim, retain tested exact source, a clean
native test receipt, the closed preparation's immutable bytes and successful
terminal invocations, fresh actual CPU provenance/parent construction, idle GPU
samples and the unchanged FilmBrain/protected-service context. Preserve the
original failed wrapper and prior protocols. A new namespace and retained capped
service are required; do not reopen a closed preparation directory.

The smallest useful future sampler experiment is an exact-parent, no-environment
fixed-observation diagnostic using stock `PPO.act`: 28 calls on 64 CUDA rows,
without `process_env_step`, storage writes, return computation or optimization.
Retain actual input tensors, unclipped actions, values, log-probabilities,
distribution mean/std, each private state boundary, caller state boundaries,
unchanged model/empty optimizer/storage state, whole child logs and raw hashes.
These synthetic observations test wiring, not a robot attempt or a rollout.

Reproducibility requires a separate fresh native CUDA replay of the same declared
source/parent/input/seed and complete output/state comparisons for each seed.
A CPU raw-record consistency check may validate schemas, hashes, flags and
distribution algebra; it must not be labelled native CUDA random-sampling replay.
Use separately fixed budgets and enough time for all captures, independent
closeout and retention before 20:00. No such runtime service is admitted by this
source-boundary document alone.

A passed sampler gate would still leave physical transition capture, genuine
terminal/storage/reset ordering, finite CUDA GAE, optimizer replay and per-attempt
lesson assignment open. The no-deficit frozen-parent screens justify retaining
that passing parent, not unnecessary updates. Hopping, obstacles, football
balance and physical deployment keep their separate capability ledgers.

## Reviewed source evidence

A bounded Luna implementation and owner review tightened the process-global
locking, swallowed-reentry fault, constructor stream preservation and isolated
Python-thread contracts before integration. The worker's final focused suite
passed **25 tests in 6.41 seconds**; the owner's independent focused run passed
**25 tests in 6.88 seconds**. The affected 21-file recovery/preparation/RNG
regression passed **517 tests in 52.55 seconds**, with CUDA hidden and no skips.
Ruff 0.15.7 check/format, compilation and document-link/whitespace checks passed.
All positive CUDA API behavior in this suite is mocked and labelled synthetic;
none of these results is a native CUDA sampler receipt. The WSL source remains
frozen at the completed preparation revision until a separately guarded source
transition and tested native predeclaration are ready.
