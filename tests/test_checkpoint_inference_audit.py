"""Adversarial CPU state-load checks, not actor performance evaluation."""

import copy
import hashlib
import json
from pathlib import Path

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


def retained(local_name, remote_path, digest):
    root = Path(__file__).resolve().parents[1]
    paths = [root / "artifacts/diagnostics/frozen-actor-input-audit-v1" / local_name,
             root / remote_path]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        pytest.skip("separately retained diagnostic artifact is unavailable")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def test_retained_real_actor_loading_does_not_admit_behavior():
    report = retained("inference-audit.json",
        "artifacts/evaluations/frozen-actor-input-audit-v1/inference-audit.json",
        "3cbc27cf51c6f858d6f9a3f9cc868b084d042331d00d110b9b5c7032e952c91f")
    assert report["source"] == "1ee0f7b6c5ee12ec10dabbf83ab64731bc516361"
    assert report["saved_iteration"] == 7998
    assert report["strict_load"] == "<All keys matched successfully>"
    assert report["normalizer"]["obs_normalizer.count"]["minimum"] == 786432000
    assert report["state_unchanged"] and report["normalizer_eval"]
    assert not any(report[k] for k in ("simulation_executed", "optimizer_step_executed",
        "command_tracking_validated", "complete_runtime_equivalence_validated",
        "policy_acceptance", "physical_motion_authorized"))


def test_historical_command_is_not_achieved_speed_or_new_motor_admission():
    hc0 = retained("hc0-envelope.json",
        "artifacts/evaluations/hc0-command-envelope-a1b3611-s41/checkpoint-evaluation.json",
        "d3c0ce5775d0f8109e2f449b00ff29fdb7ef4e40b9e259dad895e09702cbcd14")
    case, = [c for c in hc0["cases"] if c["commanded_speed_mps"] == .3
             and c["commanded_yaw_rate_rps"] == 0]
    assert case["applied_command_speed_mean_mps"] == pytest.approx(.3)
    assert case["observed_speed_mean_mps"] == pytest.approx(.2114190012216568)
    assert case["observed_speed_mean_mps"] < .27
    assert case["motor_torque_utilization_p99"] < .60
    stage2 = retained("stage2-envelope.json",
        "artifacts/evaluations/run-stage2-motor-envelope-5speed-3seed-36667ee/checkpoint-evaluation.json",
        "9053a929e5d0faf92b9150f223486ae2bf5766b92ba79ff868326ac169f61a47")
    cases = [c for c in stage2["cases"] if c["commanded_speed_mps"] in (.5, .8)]
    assert len(cases) == 6
    assert all(c["observed_speed_mean_mps"] < c["commanded_speed_mps"] - .03 for c in cases)
    assert all(c["motor_torque_utilization_p99"] > .60 for c in cases)
    assert all("applied_command_speed_mean_mps" not in c for c in cases)
