"""CPU-only integration checks for the v8 actor initialization boundary."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_actor_smoke as smoke
from mjlab_microduck.v8_actor_seed import FrozenV8Actor


def _actor(obs=None):
    observations = obs or TensorDict({"actor": torch.zeros(2, 61)}, batch_size=[2])
    return FrozenV8Actor(
        observations, {"actor": ["actor"]}, "actor", 14,
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": smoke.INITIAL_STD,
            "std_type": "scalar",
        },
    )


def _critic():
    observations = TensorDict({"critic": torch.zeros(2, 61)}, batch_size=[2])
    return MLPModel(observations, {"critic": ["critic"]}, "critic", 1,
                    hidden_dims=(512, 256, 128), activation="elu",
                    obs_normalization=True)


def _runner_env(*, actor=None):
    actor = _actor() if actor is None else actor
    critic = _critic()
    optimizer = torch.optim.Adam((*actor.parameters(), *critic.parameters()), lr=1e-3)
    alg = SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer, learning_rate=1e-3)
    raw = torch.linspace(-1.2, 1.8, 64 * 61, dtype=torch.float32).reshape(64, 61)
    robot = SimpleNamespace(find_joints=lambda *_args, **_kwargs: (
        tuple(range(14)), tuple(f"joint_{i}" for i in range(14))))
    env = SimpleNamespace(common_step_counter=0, device="cpu", scene={"robot": robot},
                          get_observations=lambda: TensorDict({"actor": raw.clone()}, batch_size=[64]))
    runner = SimpleNamespace(alg=alg, env=env, current_learning_iteration=0)
    return runner, env


def _state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_state(model, prior):
    assert set(model.state_dict()) == set(prior)
    assert all(torch.equal(value, prior[name]) for name, value in model.state_dict().items())


def test_prepare_config_is_exact_smoke_recipe_with_only_v8_specific_overrides():
    expected_cfg, expected_agent = base.prepare_config("smoke")
    cfg, agent = smoke.prepare_config()

    expected_cfg.seed = smoke.SEED
    expected_agent.seed = smoke.SEED
    expected_agent.actor.class_name = "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
    expected_agent.actor.distribution_cfg = {
        "class_name": "rsl_rl.modules:GaussianDistribution",
        "init_std": smoke.INITIAL_STD, "std_type": "scalar",
    }
    expected_agent.experiment_name = smoke.PROTOCOL
    assert asdict(cfg) == asdict(expected_cfg)
    assert asdict(agent) == asdict(expected_agent)
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 64
    assert agent.max_iterations == 5 and agent.seed == cfg.seed == 947
    assert agent.clip_actions is None
    assert agent.actor.class_name == "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
    assert agent.actor.obs_normalization is True
    assert agent.actor.distribution_cfg["init_std"] == .3
    assert agent.actor.distribution_cfg["std_type"] == "scalar"


def test_initialize_imports_only_actor_preserving_fresh_target_state(monkeypatch, tmp_path):
    runner, env = _runner_env()
    actor, critic, optimizer = runner.alg.actor, runner.alg.critic, runner.alg.optimizer
    actor_ids = tuple(id(p) for p in actor.parameters())
    optimizer_params = tuple(p for group in optimizer.param_groups for p in group["params"])
    optimizer_ids = tuple(id(p) for p in optimizer_params)
    critic_before = _state(critic)
    std_before = _state(actor.distribution)
    seen = {}
    writes = []

    class FakeReference:
        def __init__(self, path, digest):
            seen["reference"] = (path, digest)

        def validate_robot(self, robot, joint_ids, names):
            seen["robot_validation"] = (robot, joint_ids, names)

    def fake_import(import_actor, path, probe):
        seen["import"] = (import_actor, path, probe.detach().clone())
        with torch.no_grad():
            import_actor.mlp[-1].bias.add_(0.125)
        return {"protocol": "test-import", "maximum_actor_error": 0.0}

    monkeypatch.setattr(smoke, "NormalizedOnnxReference", FakeReference)
    monkeypatch.setattr(smoke, "import_v8_actor", fake_import)
    monkeypatch.setattr(smoke.base, "write_new", lambda path, value: writes.append((path, value)))

    report = smoke.initialize(runner, env, tmp_path)

    assert seen["reference"] == (smoke.pinned.V8.path, smoke.pinned.V8.sha256)
    assert seen["import"][0] is actor and seen["import"][1] == smoke.pinned.V8.path
    assert torch.equal(seen["import"][2], env.get_observations()["actor"])
    assert seen["robot_validation"][1] == tuple(range(14))
    assert tuple(id(p) for p in actor.parameters()) == actor_ids
    assert tuple(id(p) for group in optimizer.param_groups for p in group["params"]) == optimizer_ids
    assert all(before is after for before, after in zip(
        optimizer_params, (p for group in optimizer.param_groups for p in group["params"])))
    assert not optimizer.state
    _assert_state(critic, critic_before)
    _assert_state(actor.distribution, std_before)
    assert runner.current_learning_iteration == env.common_step_counter == 0
    assert report["initial_std"] == .3 and report["target_iteration"] == 0
    assert report["target_common_step_counter"] == 0
    assert report["critic_normalizer_count"] == 0 and report["adam_state_entries"] == 0
    assert len(writes) == 1 and writes[0][0] == tmp_path / "actor-import.json"
    assert writes[0][1] is report


@pytest.mark.parametrize("invalid_state", ["wrong_actor", "dirty_optimizer", "iteration", "env_clock",
                                            "actor_count", "critic_count"])
def test_initialize_refuses_nonfresh_runner_before_reference_or_import(monkeypatch, tmp_path,
                                                                       invalid_state):
    runner, env = _runner_env()
    if invalid_state == "wrong_actor":
        wrong = MLPModel(TensorDict({"actor": torch.zeros(2, 61)}, batch_size=[2]),
                         {"actor": ["actor"]}, "actor", 14,
                         hidden_dims=(512, 256, 128), activation="elu",
                         obs_normalization=True,
                         distribution_cfg={"class_name": "GaussianDistribution",
                                           "init_std": .3, "std_type": "scalar"})
        runner.alg.actor = wrong
        runner.alg.optimizer = torch.optim.Adam(
            (*wrong.parameters(), *runner.alg.critic.parameters()), lr=1e-3)
    elif invalid_state == "dirty_optimizer":
        parameter = runner.alg.actor.mlp[0].weight
        runner.alg.optimizer.state[parameter] = {"exp_avg": torch.ones_like(parameter)}
    elif invalid_state == "iteration":
        runner.current_learning_iteration = 1
    elif invalid_state == "env_clock":
        env.common_step_counter = 24
    elif invalid_state == "actor_count":
        runner.alg.actor.obs_normalizer.count.fill_(1)
    elif invalid_state == "critic_count":
        runner.alg.critic.obs_normalizer.count.fill_(1)

    calls = []
    monkeypatch.setattr(smoke, "NormalizedOnnxReference",
                        lambda *_args, **_kwargs: calls.append("reference"))
    monkeypatch.setattr(smoke, "import_v8_actor",
                        lambda *_args, **_kwargs: calls.append("import"))
    monkeypatch.setattr(smoke.base, "write_new", lambda *_args, **_kwargs: calls.append("write"))

    with pytest.raises(ValueError):
        smoke.initialize(runner, env, tmp_path)

    assert calls == []
    assert bool(runner.alg.optimizer.state) is (invalid_state == "dirty_optimizer")
