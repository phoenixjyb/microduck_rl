"""Synthetic action-manager checks do not constitute a stable robot rollout."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import action_pipeline_audit as audit


@pytest.fixture
def robot_config():
    from mjlab.entity import Entity
    from mjlab_microduck.foundation_yaw_experiment import prepare_config
    cfg, agent = prepare_config("pilot", arm="yaw")
    return Entity(cfg.scene.entities["robot"]), cfg, agent


def test_actual_manager_transform_and_reset_semantics(robot_config):
    result = audit.probe_actions(*robot_config)
    assert result["target_ids"] == list(range(14))
    assert len(result["one_hot_probes"]) == 14
    assert result["zero_action_target"] != [[0.] * 14] * 2
    assert result["after_reset_only_target"] == result["unclipped_probe_target"]
    assert result["after_fresh_zero_target"] == result["zero_action_target"]
    assert result["after_encoder_bias_change_target"] != result["zero_action_target"]
    for key in ("reset_clears_selected_raw_and_history", "full_reset_clears_raw_and_history",
                "reset_preserves_processed_cache", "fresh_process_required_before_apply",
                "encoder_bias_is_live_per_apply", "default_offset_is_construction_clone",
                "caller_actions_unchanged", "torch_rng_preserved", "synthetic_buffers_only"):
        assert result[key] is True
    assert result["safe_stop_validated"] is result["motor_control_executed"] is False


@pytest.mark.parametrize("field,value", [("scale", 2.), ("scale", True), ("offset", .1),
                                      ("clip", {".*": (-1., 1.)}), ("use_default_offset", False),
                                      ("preserve_order", True), ("actuator_names", ("left_.*",))])
def test_changed_action_contract_rejected_before_probe(robot_config, field, value):
    robot, cfg, agent = robot_config
    setattr(cfg.actions["joint_pos"], field, value)
    with pytest.raises(ValueError, match="unmodified"):
        audit.probe_actions(robot, cfg, agent)
    assert not hasattr(robot, "_data")


def test_runner_clipping_cannot_be_ignored(robot_config):
    robot, cfg, agent = robot_config
    agent.clip_actions = 1.
    with pytest.raises(ValueError, match="unmodified"):
        audit.probe_actions(robot, cfg, agent)


def test_initialized_entity_is_never_repurposed(robot_config):
    robot, cfg, agent = robot_config
    marker = SimpleNamespace()
    robot._data = marker
    with pytest.raises(ValueError, match="isolated"):
        audit.probe_actions(robot, cfg, agent)
    assert robot._data is marker


def test_nonmatching_probe_dtype_is_rejected(robot_config, monkeypatch):
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    with pytest.raises(ValueError, match="CPU float32"):
        audit.probe_actions(*robot_config)


def test_wrong_target_delivery_is_detected(robot_config, monkeypatch):
    robot, cfg, agent = robot_config
    original = robot.set_joint_position_target
    def corrupt(position, joint_ids):
        original(position.roll(1, dims=1), joint_ids)
    monkeypatch.setattr(robot, "set_joint_position_target", corrupt)
    with pytest.raises(ValueError, match="target delivery"):
        audit.probe_actions(robot, cfg, agent)


def normal_step(self, action):
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()


def reversed_step(self, action):
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()
    self.action_manager.process_action(action)


def conditional_process(self, action):
    if self.condition:
        self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.action_manager.apply_action()
        self.sim.step()


def reversed_physics(self, action):
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
        self.sim.step()
        self.action_manager.apply_action()


def test_source_order_check_is_explicitly_not_full_environment_execution():
    result = audit.step_order_source_check(normal_step)
    assert result["process_before_apply_loop"] is result["apply_before_physics"] is True
    assert result["full_environment_executed"] is False


@pytest.mark.parametrize("method", [reversed_step, conditional_process, reversed_physics])
def test_invalid_processing_order_fails_source_check(method):
    with pytest.raises(ValueError):
        audit.step_order_source_check(method)


def test_cuda_guard_precedes_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA hidden"):
        audit.audit(tmp_path, tmp_path / "missing")


def test_cli_protects_closed_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit", str(tmp_path), str(tmp_path / "binding.json"),
                                    "--output", str(tmp_path / "output.json")])
    monkeypatch.setattr(audit, "audit", lambda *_: pytest.fail("must reject before probes"))
    with pytest.raises(ValueError, match="outside closed"):
        audit.main()


def test_real_retained_binding_when_available(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    evidence = next((repo / "artifacts" / prefix / "f1y-yaw-support-paired-s499-v1"
                     for prefix in ("diagnostics", "experiments")
                     if (repo / "artifacts" / prefix / "f1y-yaw-support-paired-s499-v1/manifest.json").is_file()), None)
    binding = repo / "artifacts/audits/f1y-retained-skill-binding-v1.json"
    if evidence is None or not binding.exists():
        pytest.skip("optional retained artifact unavailable")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = audit.audit(evidence, binding)
    assert report["declaration_match"] is True
    assert report["historical_source_files_verified"] == 136
    for key in ("historical_randomized_actuator_state_verified", "full_environment_reset_verified",
                "closed_loop_retention_verified", "transition_authorized", "policy_acceptance",
                "physical_motion_authorized"):
        assert report[key] is False
