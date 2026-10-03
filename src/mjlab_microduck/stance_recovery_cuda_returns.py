"""Finite CUDA64 GAE source sibling: no optimizer, storage clear or admission."""

from contextlib import contextmanager

import torch

from mjlab_microduck import stance_recovery_cuda_shadow_sampling as sampling
from mjlab_microduck import stance_recovery_cuda_transition as transition
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cuda64-finite-returns-source-contract-v1"
FALSE_FLAGS = dict(transition.FALSE_FLAGS)


@contextmanager
def _math_rng_scope():
    """Restore caller streams even if a noncanonical critic unexpectedly draws."""
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = sampling._caller_cuda_state()
    with torch.random.fork_rng(devices=[0]):
        yield
        require(
            torch.equal(before_cpu, torch.random.get_rng_state())
            and torch.equal(before_cuda, sampling._caller_cuda_state()),
            "finite returns must consume no caller random draws",
        )


def _targets(rewards, values, dones, last, *, gamma, lam):
    """Compute in input float32; reject nonfinite intermediates without repair."""
    shape = (transition.HORIZON, transition.WORLDS, 1)
    returns = torch.zeros_like(values)
    advantage = torch.zeros_like(last)
    for step in reversed(range(transition.HORIZON)):
        following = last if step == transition.HORIZON - 1 else values[step + 1]
        continuation = 1.0 - dones[step].to(torch.float32)
        delta = rewards[step] + gamma * continuation * following - values[step]
        advantage = delta + gamma * lam * continuation * advantage
        transition._tensor(advantage, (transition.WORLDS, 1), "GAE intermediate")
        returns[step] = advantage + values[step]
    transition._tensor(returns, shape, "finite returns")
    raw = returns - values
    transition._tensor(raw, shape, "raw GAE advantages")
    mean, std = raw.mean(), raw.std(correction=1)
    transition._tensor(mean, (), "advantage mean")
    transition._tensor(std, (), "advantage sample standard deviation")
    normalized = (raw - mean) / (std + 1e-8)
    transition._tensor(normalized, shape, "normalized GAE advantages")
    return returns, raw, normalized


@torch.no_grad()
def compute_finite_returns(collector):
    """Compute once for a full exact collector, retaining inputs and targets.

    A successful result is a source-contract record, not native qualification,
    learning, a policy update or permission for a training launch.
    """
    require(
        type(collector) is transition.CudaTransitionCollector, "exact CUDA collector"
    )
    require(not collector.faulted, "faulted CUDA collector cannot compute returns")
    try:
        # Includes the inherited lease before every CUDA query or operation.
        collector._healthy()
        storage = collector.storage
        require(
            collector.phase == "full"
            and collector.transitions == storage.step == transition.HORIZON,
            "full 28-transition collector required before finite GAE",
        )
        shape = (transition.HORIZON, transition.WORLDS, 1)
        for name in ("rewards", "values", "returns", "advantages"):
            transition._tensor(getattr(storage, name), shape, "stored " + name)
        transition._tensor(storage.dones, shape, "stored dones", dtype=torch.uint8)
        require(
            ((storage.dones == 0) | (storage.dones == 1)).all(), "binary stored dones"
        )
        require(
            not storage.returns.any() and not storage.advantages.any(),
            "fresh unwritten returns and advantages",
        )
        rewards, values, dones = (
            getattr(storage, name).detach().clone()
            for name in ("rewards", "values", "dones")
        )
        private_before = collector.scope.state
        next_obs = transition._observations(collector.env.observations())
        with _math_rng_scope():
            last = collector.critic(next_obs).detach()
            transition._tensor(last, (transition.WORLDS, 1), "last critic value")
            returns, raw, advantages = _targets(
                rewards,
                values,
                dones,
                last,
                gamma=collector.algorithm.gamma,
                lam=collector.algorithm.lam,
            )
        require(
            torch.equal(private_before, collector.scope.state)
            and torch.equal(storage.rewards, rewards)
            and torch.equal(storage.values, values)
            and torch.equal(storage.dones, dones),
            "finite math preserves private RNG and stored transition inputs",
        )
        collector._healthy()
        # Validate the entire result before either write. A copy failure latches
        # the collector; no duplicate computation, repair or resume is allowed.
        storage.returns.copy_(returns)
        storage.advantages.copy_(advantages)
        require(
            torch.equal(storage.returns, returns)
            and torch.equal(storage.advantages, advantages)
            and storage.step == transition.HORIZON
            and not collector.algorithm.optimizer.state,
            "exact target writes without storage clear or optimizer state",
        )
        collector.phase = "returns-computed"
        return {
            "receipt": {
                "protocol": PROTOCOL,
                "source": collector.source,
                "learner_seed": collector.seed,
                "device": transition.DEVICE,
                "worlds": transition.WORLDS,
                "horizon": transition.HORIZON,
                "gamma": collector.algorithm.gamma,
                "lambda": collector.algorithm.lam,
                "normalization_correction": 1,
                "normalization_epsilon": 1e-8,
                "optimizer_steps": 0,
                "storage_cleared": False,
                "native_finite_gae_qualified": False,
                **FALSE_FLAGS,
            },
            "rewards": rewards,
            "pre_action_values": values,
            "dones": dones,
            "next_observations": {
                key: value.clone() for key, value in next_obs.items()
            },
            "last_critic_values": last.clone(),
            "returns": returns.clone(),
            "raw_advantages": raw.clone(),
            "normalized_advantages": advantages.clone(),
            "private_rng_state": private_before,
        }
    except BaseException:
        collector.faulted, collector.phase = True, "faulted"
        raise
