import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import normalized_onnx_reference as reference
from test_head_yaw_evaluator_hook import install_fake_evaluator


def _fixture_action(fake, monkeypatch):
    import mjlab.rl

    wrapper = mjlab.rl.RslRlVecEnvWrapper
    closure = dict(zip(wrapper.step.__code__.co_freevars,
                       (cell.cell_contents for cell in wrapper.step.__closure__)))
    action = closure["action"]

    class CaptureWrapper(wrapper):
        def step(self, raw):
            assert raw is action
            return super().step(raw)

    monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", CaptureWrapper)
    return action


def _typed_reference(monkeypatch, artifact_sha256, action, calls):
    class FakeReference(reference.NormalizedOnnxReference):
        def __new__(cls):
            instance = object.__new__(cls)
            instance._artifact_sha256 = artifact_sha256
            return instance

        def __call__(self, observations):
            calls.append(("policy", observations))
            return action

        def validate_robot(self, robot, joint_ids, names):
            calls.append(("robot", robot, tuple(joint_ids), tuple(names)))

        def provenance(self):
            return dict(backend="test-cpu-onnx", artifact_sha256=self.artifact_sha256,
                        provider="CPUExecutionProvider")

    monkeypatch.setattr(reference, "validate_actor_contract",
                        lambda cfg, **kwargs: calls.append(("actor_contract", cfg, kwargs)
                                                           if kwargs else ("actor_contract", cfg)))
    return FakeReference.__new__(FakeReference)


@pytest.mark.parametrize(
    ("terminal_at", "expected_steps", "expected_complete", "expected_terminals", "expected_health"),
    [(None, 240, True, [], 2), (3, 3, False, [2], 1)],
)
def test_reference_branch_skips_runner_and_preserves_ordered_step_metrics(
        monkeypatch, tmp_path, terminal_at, expected_steps, expected_complete,
        expected_terminals, expected_health):
    import mjlab.envs
    import mjlab.tasks.registry as registry

    fake = install_fake_evaluator(monkeypatch, terminal_at=terminal_at)
    checkpoint = tmp_path / "official-reference.onnx"
    checkpoint.write_bytes(b"fixture-only pinned ONNX identity")
    digest = base.sha256(checkpoint)
    action = _fixture_action(fake, monkeypatch)
    calls = []
    adapter = _typed_reference(monkeypatch, digest, action, calls)
    health = []

    def forbidden_runner(*_args, **_kwargs):
        raise AssertionError("ONNX reference branch must not construct or load PPO")

    monkeypatch.setattr(registry, "load_runner_cls", forbidden_runner)
    original_env_factory = mjlab.envs.ManagerBasedRlEnv

    def checked_env(cfg, device):
        assert calls[0] == ("actor_contract", cfg)
        return original_env_factory(cfg, device)

    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", checked_env)
    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: health.append("sample"),
                                onnx_reference=adapter)

    assert len(health) == expected_health
    assert fake.calls == {"policy": 0, "step": expected_steps, "gait": 0, "yaw": 0}
    assert fake.events == ["step"] * expected_steps + ["close"]
    assert [call[0] for call in calls] == ["actor_contract", "robot"] + ["policy"] * expected_steps
    robot_call = calls[1]
    assert robot_call[1] is fake.robot
    assert robot_call[2] == tuple(range(14))
    assert robot_call[3] == tuple(fake.robot.joint_names[:14])
    assert result["steps"] == expected_steps and result["complete"] is expected_complete
    assert result["terminal_worlds"] == expected_terminals
    assert result["checkpoint_sha256"] == digest
    assert result["inference_backend"] == adapter.provenance()
    assert result["motor_stream"] == {"test": "read-only CPU fixture"}


def test_default_none_keeps_legacy_ppo_runner_path(monkeypatch, tmp_path):
    fake = install_fake_evaluator(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture-only PPO checkpoint")

    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: None)

    assert fake.calls == {"policy": 240, "step": 240, "gait": 0, "yaw": 0}
    assert fake.events == ["policy", "step"] * 240 + ["close"]
    assert result["complete"] is True
    assert "inference_backend" not in result


def test_explicit_v8_backlash_override_is_bound_without_touching_default_loader(monkeypatch, tmp_path):
    import mjlab.tasks.registry as registry

    fake = install_fake_evaluator(monkeypatch)
    path = tmp_path / "fixture-reference.onnx"
    path.write_bytes(b"fixture only; artifact hash function explicitly stubbed")
    monkeypatch.setattr(base, "sha256", lambda _path: reference.OFFICIAL_V8_SHA256)
    calls, selected = [], []
    adapter = _typed_reference(monkeypatch, reference.OFFICIAL_V8_SHA256,
                               _fixture_action(fake, monkeypatch), calls)
    original_cfg = registry.load_env_cfg
    def cfg(task, **kwargs):
        selected.append(task)
        return original_cfg(task, **kwargs)
    monkeypatch.setattr(registry, "load_env_cfg", cfg)
    monkeypatch.setattr(registry, "load_runner_cls",
                        lambda *_a, **_k: pytest.fail("must not construct PPO"))
    result = base.evaluate_case(path, 839, .3, 0., lambda: None, onnx_reference=adapter,
                               evaluation_task_override="Mjlab-Velocity-Flat-Backlash-MicroDuck")
    assert selected == ["Mjlab-Velocity-Flat-Backlash-MicroDuck"]
    assert calls[0] == ("actor_contract", fake.cfg, {"expected_actuator": "BacklashEncoderBamActuatorCfg"})
    assert result["complete"] and fake.calls["step"] == 240
    assert result["evaluation_task"] == selected[0]
    assert result["position_measurement_view"] == "named motor-hinge qpos; passive joints excluded"


@pytest.mark.parametrize("task,sha,use_reference", [
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", reference.OFFICIAL_V8_SHA256, False),
    ("Mjlab-Velocity-Flat-MicroDuck", reference.OFFICIAL_V8_SHA256, True),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", reference.OFFICIAL_V8_SHA256, True),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", reference.OFFICIAL_VELSTAND_SHA256, True),
])
def test_plant_override_cannot_change_ppo_or_expand_to_unbound_tasks(monkeypatch, tmp_path, task, sha, use_reference):
    import mjlab.envs
    fake = install_fake_evaluator(monkeypatch)
    calls = []
    adapter = _typed_reference(monkeypatch, sha, _fixture_action(fake, monkeypatch), calls) if use_reference else None
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv",
                        lambda *_a, **_k: pytest.fail("invalid override reached environment"))
    with pytest.raises(ValueError, match="pinned-v8 flat-backlash"):
        base.evaluate_case(tmp_path/"no-file-needed", 839, .3, 0., lambda: None,
                           onnx_reference=adapter, evaluation_task_override=task)
    assert not calls


@pytest.mark.parametrize("reference_value", ["not-an-adapter", object()])
def test_wrong_reference_type_is_rejected_before_environment_construction(
        monkeypatch, tmp_path, reference_value):
    import mjlab.envs

    fake = install_fake_evaluator(monkeypatch)
    constructed = []
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv",
                        lambda *_args, **_kwargs: constructed.append(True))
    checkpoint = tmp_path / "official-reference.onnx"
    checkpoint.write_bytes(b"fixture-only ONNX bytes")

    with pytest.raises(ValueError, match="typed exact-byte ONNX reference"):
        base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
                           onnx_reference=reference_value)

    assert constructed == []
    assert fake.events == []


def test_reference_hash_mismatch_is_rejected_before_environment_construction(monkeypatch, tmp_path):
    import mjlab.envs

    fake = install_fake_evaluator(monkeypatch)
    constructed = []
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv",
                        lambda *_args, **_kwargs: constructed.append(True))
    checkpoint = tmp_path / "official-reference.onnx"
    checkpoint.write_bytes(b"fixture-only ONNX bytes")
    action = _fixture_action(fake, monkeypatch)
    calls = []
    adapter = _typed_reference(monkeypatch, "0" * 64, action, calls)

    with pytest.raises(ValueError, match="typed exact-byte ONNX reference"):
        base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
                           onnx_reference=adapter)

    assert constructed == []
    assert calls == []
    assert fake.events == []
