from types import SimpleNamespace
from dataclasses import dataclass

import pytest
import torch

from mjlab_microduck.integrated_turn_foundation import body_yaw_trace_sample


def robot_and_command():
    data = SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(8, 1),
                           root_link_ang_vel_b=torch.zeros(8, 3),
                           root_link_ang_vel_w=torch.zeros(8, 3))
    return SimpleNamespace(data=data), torch.tensor([[.2, 0., -.2]]).repeat(8, 1)


def test_pre_action_capture_is_detached_nonaliasing_and_does_not_modify_input():
    robot, command = robot_and_command()
    actor = command.clone().requires_grad_()
    sample = body_yaw_trace_sample(robot, command, actor)
    assert torch.equal(sample["command"], command)
    assert torch.equal(sample["actor_command"], actor)
    assert not sample["actor_command"].requires_grad
    command.zero_()
    robot.data.root_link_ang_vel_b.fill_(2.)
    robot.data.root_link_quat_w.zero_()
    assert torch.equal(sample["command"], torch.tensor([[.2, 0., -.2]]).repeat(8, 1))
    assert sample["angular_b"].eq(0.).all()
    assert sample["quat_wxyz"][:, 0].eq(1.).all()


@pytest.mark.parametrize("field", ["root_link_quat_w", "root_link_ang_vel_b", "root_link_ang_vel_w"])
@pytest.mark.parametrize("fault", ["short", "nonfinite", "rank"])
def test_trace_rejects_wrong_orientation_rate_layout_or_nonfinite(field, fault):
    robot, command = robot_and_command()
    value = getattr(robot.data, field)
    if fault == "short": value = value[:7]
    elif fault == "nonfinite": value[0, 0] = float("nan")
    else: value = value.unsqueeze(0)
    setattr(robot.data, field, value)
    with pytest.raises(ValueError, match="yaw trace layout"):
        body_yaw_trace_sample(robot, command, command)


@pytest.mark.parametrize("fault", ["command-shape", "actor-shape", "actor-mismatch", "command-nonfinite"])
def test_trace_rejects_command_fault(fault):
    robot, command = robot_and_command()
    actor = command.clone()
    if fault == "command-shape": command = command[:7]
    elif fault == "actor-shape": actor = actor[:7]
    elif fault == "actor-mismatch": actor[2, 2] = .2
    else: command[1, 1] = float("inf")
    with pytest.raises(ValueError):
        body_yaw_trace_sample(robot, command, actor)


def test_optional_trace_keeps_original_case_evidence_and_reads_before_step(monkeypatch, tmp_path):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry as registry
    from mjlab_microduck import integrated_turn_foundation as base

    @dataclass
    class Agent:
        seed: int = 0
        clip_actions: float | None = None

    closed = []

    class Env:
        def __init__(self, cfg, device):
            self.num_envs, self.device, self.step_dt = 8, "cpu", .02
            robot, self.command = robot_and_command()
            robot.data.root_link_lin_vel_b = torch.zeros(8, 3)
            robot.data.joint_pos = torch.zeros(8, 14)
            robot.data.joint_pos_limits = torch.tensor([-1., 1.]).repeat(8, 14, 1)
            self.scene = {"robot": robot}
            self.command_manager = SimpleNamespace(get_command=lambda _: self.command)
            self.count = 0

        def close(self):
            closed.append(self.count)

    class Wrapped:
        def __init__(self, env, clip_actions):
            self.env = env

        def get_observations(self):
            obs = torch.zeros(8, 61)
            obs[:, 48:51] = self.env.command
            return {"actor": obs}

        def step(self, actions):
            self.env.count += 1
            data = self.env.scene["robot"].data
            data.root_link_lin_vel_b[:, 0] = self.env.count / 1000
            data.root_link_ang_vel_b[:, 2] = self.env.count / 1000
            data.root_link_ang_vel_w.copy_(data.root_link_ang_vel_b)
            return self.get_observations(), torch.zeros(8), torch.zeros(8, dtype=torch.bool), {}

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

        def get_inference_policy(self, device):
            return lambda obs: torch.zeros(8, 14)

    class Stream:
        joint_ids = tuple(range(14))
        names = tuple(f"joint{i}" for i in range(14))

        def begin(self, *args):
            pass

        def consume(self, dones):
            return SimpleNamespace(force_nm=torch.zeros(8, 14), speed_rad_s=torch.zeros(8, 14))

        def provenance(self):
            return {"test": "unchanged"}

    def config(*args, **kwargs):
        return SimpleNamespace(curriculum={}, events={}, scene=SimpleNamespace(terrain=SimpleNamespace()),
                               sim=SimpleNamespace(nan_guard=SimpleNamespace(enabled=False)))

    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", Env)
    monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", Wrapped)
    monkeypatch.setattr(registry, "load_env_cfg", config)
    monkeypatch.setattr(registry, "load_rl_cfg", lambda *a: Agent())
    monkeypatch.setattr(registry, "load_runner_cls", lambda *a: Runner)
    monkeypatch.setattr(base, "fix_velocity_commands", lambda *a: None)
    monkeypatch.setattr(base, "install_metric", lambda *a: None)
    monkeypatch.setattr(base.MotorStepStream, "from_robot", lambda *a, **kw: Stream())
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    default = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None, diagnostics=True)
    traced = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None, diagnostics=True, yaw_trace=True)
    trace = traced.pop("body_yaw_trace")
    assert traced == default
    assert closed == [240, 240]
    assert trace["steps"] == 240 and trace["startup_steps"] == 60
    assert trace["timing"] == "pre-action state before policy/step"
    assert trace["angular_b"][0][0] == [0., 0., 0.]
    assert trace["angular_b"][-1][0][2] == pytest.approx(.239)
    assert trace["angular_w"] == trace["angular_b"]
    assert trace["actor_command"] == trace["command"]
    assert trace["checkpoint_sha256"] == default["checkpoint_sha256"]
    assert trace["policy_acceptance"] is trace["physical_motion_authorized"] is False
    from mjlab_microduck.body_yaw_trace import summarize_trace
    audit = summarize_trace(default, trace)
    assert audit["worlds"][0]["body_yaw_mae"] == pytest.approx(default["yaw_mae"][0])

    # A terminal episode retains its short raw trace but cannot be summarized
    # as a complete held-out case or have startup/terminal samples hidden.
    step = Wrapped.step

    def terminating_step(self, actions):
        obs, rewards, dones, extras = step(self, actions)
        if self.env.count == 61:
            dones[0] = True
        return obs, rewards, dones, extras

    monkeypatch.setattr(Wrapped, "step", terminating_step)
    short = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None, diagnostics=True, yaw_trace=True)
    assert short["steps"] == short["body_yaw_trace"]["steps"] == 61
    assert short["complete"] is False and short["terminal_worlds"] == [0]
    assert len(short["body_yaw_trace"]["angular_b"]) == 61
    with pytest.raises(ValueError, match="case must be complete"):
        summarize_trace(short, short["body_yaw_trace"])
