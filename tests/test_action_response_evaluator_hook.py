"""CPU spies for the optional hook: one policy and one step per sample."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import action_response_capture as capture


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_capture_preserves_single_policy_step_and_action_delivery(monkeypatch, tmp_path, enabled):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry as registry

    @dataclass
    class Agent:
        seed: int = 0
        clip_actions: object = None

    events, delivered, captured = [], [], []
    action = torch.full((8, 14), 1.25)
    command = torch.tensor([.2, 0., -.2]).repeat(8, 1)
    data = SimpleNamespace(
        joint_pos=torch.zeros(8, 14), joint_pos_target=torch.zeros(8, 14),
        joint_pos_limits=torch.tensor([-1., 1.]).repeat(8, 14, 1),
        indexing=SimpleNamespace(joint_v_adr=torch.arange(14)),
        data=SimpleNamespace(qfrc_constraint=torch.zeros(8, 14)),
        root_link_lin_vel_b=torch.tensor([.2, 0., 0.]).repeat(8, 1),
        root_link_ang_vel_b=torch.tensor([0., 0., -.2]).repeat(8, 1),
        root_link_ang_vel_w=torch.tensor([0., 0., -.2]).repeat(8, 1),
        root_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(8, 1))
    robot = SimpleNamespace(data=data)
    env = SimpleNamespace(step_dt=.02, device="cpu", scene={"robot": robot},
        command_manager=SimpleNamespace(get_command=lambda name: command),
        close=lambda: events.append("close"))
    actor = torch.zeros(8, 61)
    actor[:, 48:51] = command
    obs = {"actor": actor}

    class Wrapper:
        def __init__(self, env, clip_actions):
            self.clip_actions = clip_actions

        def get_observations(self):
            return obs

        def step(self, raw):
            events.append("step")
            assert raw is action and raw.eq(1.25).all()
            delivered.append(raw.clone())
            data.joint_pos_target.copy_(raw)
            actor[:, 34:48] = raw
            return obs, torch.zeros(8), torch.zeros(8, dtype=torch.bool), {}

    class Runner:
        def __init__(self, *a, **kw): pass
        def load(self, *a, **kw): pass
        def get_inference_policy(self, **kw):
            def policy(observations):
                events.append("policy")
                assert observations is obs
                return action
            return policy

    class Stream:
        names = capture.JOINTS
        joint_ids = tuple(range(14))
        @classmethod
        def from_robot(cls, *a, **kw): return cls()
        def begin(self, *a): pass
        def consume(self, dones):
            return SimpleNamespace(force_nm=torch.zeros(8, 14), speed_rad_s=torch.zeros(8, 14))
        def provenance(self): return {"test": "CPU-spy"}

    def snapshot(*args):
        assert enabled
        events.append("capture")
        state = {"previous": actor[:, 34:48].clone()}
        captured.append(state)
        return state

    def attach(state, raw):
        events.append("attach")
        assert enabled and raw is action
        return {**state, "raw": raw.clone()}

    def serialize(samples, digest):
        assert enabled and len(samples) == 240
        assert samples[0]["previous"].eq(0.).all()
        assert all(s["previous"].eq(1.25).all() for s in samples[1:])
        assert all(s["raw"].eq(1.25).all() for s in samples)
        return {"test": "unchanged-step-sequence"}

    cfg = SimpleNamespace(seed=0, scene=SimpleNamespace(num_envs=0,
        terrain=SimpleNamespace(num_envs=0)), curriculum={}, events={},
        sim=SimpleNamespace(nan_guard=SimpleNamespace(enabled=False)))
    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", lambda *a, **kw: env)
    monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", Wrapper)
    monkeypatch.setattr(registry, "load_env_cfg", lambda *a, **kw: cfg)
    monkeypatch.setattr(registry, "load_rl_cfg", lambda *a: Agent())
    monkeypatch.setattr(registry, "load_runner_cls", lambda *a: Runner)
    monkeypatch.setattr(base, "MotorStepStream", Stream)
    monkeypatch.setattr(base, "install_metric", lambda cfg: None)
    monkeypatch.setattr(base, "fix_velocity_commands", lambda *a: None)
    monkeypatch.setattr(capture, "capture_state", snapshot)
    monkeypatch.setattr(capture, "add_action", attach)
    monkeypatch.setattr(capture, "trace", serialize)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    result = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None,
        diagnostics=enabled, target_diagnostics=enabled, yaw_trace=enabled, action_trace=enabled)
    expected = ["capture", "policy", "attach", "step"] if enabled else ["policy", "step"]
    assert events == expected * 240 + ["close"]
    assert len(delivered) == 240 and len(captured) == (240 if enabled else 0)
    assert result["complete"] is True
    assert result["speed_mae"] == pytest.approx([0.] * 8, abs=1e-8)
    assert ("action_response_trace" in result) is enabled


def test_action_hook_requires_other_readonly_diagnostics():
    with pytest.raises(ValueError, match="requires position/target/body-yaw"):
        base.evaluate_case(None, 839, .2, -.2, lambda: None, action_trace=True)
