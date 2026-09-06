"""Adversarial CPU state-load checks, not actor performance evaluation."""

import copy

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microduck.checkpoint_inference_audit import audit_actor_state
from mjlab_microduck.speed_response_control import prepare_config


@pytest.fixture
def model_state():
    _, agent = prepare_config()
    cfg = agent.actor
    model = MLPModel(TensorDict({"actor": torch.zeros(2, 61)}, batch_size=[2]), {"actor": ["actor"]}, "actor", 14,
                     hidden_dims=cfg.hidden_dims, activation=cfg.activation, obs_normalization=True,
                     distribution_cfg=copy.deepcopy(cfg.distribution_cfg))
    state = dict(model.state_dict())
    state["obs_normalizer.count"].fill_(10)
    return state, cfg


def test_strict_load_inference_preserves_state_rng_and_admission(model_state):
    state, cfg = model_state
    before = {k: v.clone() for k, v in state.items()}
    random_before = torch.get_rng_state().clone()
    config_before = copy.deepcopy(cfg)
    report = audit_actor_state(state, cfg)
    assert torch.equal(random_before, torch.get_rng_state()) and cfg == config_before
    assert all(torch.equal(v, before[k]) for k, v in state.items())
    assert report["model_eval"] and report["normalizer_eval"] and report["repeated_outputs_equal"]
    assert report["synthetic_output_shape"] == [2, 14]
    assert not any(report[k] for k in ("policy_acceptance", "physical_motion_authorized",
        "complete_runtime_equivalence_validated", "command_tracking_validated", "optimizer_step_executed"))


@pytest.mark.parametrize("change", [
    lambda s: s["obs_normalizer._mean"].fill_(float("nan")),
    lambda s: s["obs_normalizer._std"].fill_(0),
    lambda s: s["obs_normalizer._var"].fill_(-1),
    lambda s: s["obs_normalizer._var"].fill_(4),
    lambda s: s["obs_normalizer.count"].fill_(0),
    lambda s: s.update(**{"obs_normalizer._mean": torch.zeros(1, 62)}),
    lambda s: s.update(**{"mlp.0.weight": torch.zeros(512, 62)}),
    lambda s: s["mlp.0.weight"].fill_(float("inf")),
    lambda s: s.pop("mlp.6.bias"),
    lambda s: s.update(extra_tensor=torch.zeros(1)),
])
def test_bad_state_cannot_pass_as_a_successful_load(model_state, change):
    state, cfg = model_state
    change(state)
    with pytest.raises((ValueError, RuntimeError)): audit_actor_state(state, cfg)


def test_disabled_normalizer_is_refused(model_state):
    state, cfg = model_state
    cfg.obs_normalization = False
    with pytest.raises(ValueError): audit_actor_state(state, cfg)
