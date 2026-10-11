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

    class Scene(dict):
        pass

    class Env:
        def __init__(self, cfg, device):
            self.num_envs, self.device, self.step_dt = 8, "cpu", .02
            robot, self.command = robot_and_command()
            robot.data.root_link_lin_vel_b = torch.zeros(8, 3)
            robot.data.joint_pos = torch.zeros(8, 14)
            robot.data.default_joint_pos = torch.zeros(8, 14)
            robot.data.joint_vel = torch.zeros(8, 14)
            robot.data.site_lin_vel_w = torch.zeros(8, 2, 3)
            robot.data.data = SimpleNamespace(time=torch.zeros(8, dtype=torch.float64))
            from mjlab_microduck import gait_yaw_capture as gait
            robot.find_sites = lambda *a, **kw: ([0, 1], list(gait.SITES))
            robot.find_joints = lambda *a, **kw: ([5, 6, 7, 8], list(gait.HEAD_JOINTS))
            robot.data.joint_pos_limits = torch.tensor([-1., 1.]).repeat(8, 14, 1)
            self.scene = Scene(robot=robot)
            cfg = SimpleNamespace(name="feet_ground_contact", num_slots=1, fields=("found", "force"),
                reduce="netforce", track_air_time=True, history_length=0,
                primary=SimpleNamespace(entity="robot", mode="geom"),
                secondary=SimpleNamespace(entity=None, mode="body", pattern="terrain"))
            contact = SimpleNamespace(found=torch.ones(8, 2), force=torch.ones(8, 2, 3),
                current_air_time=torch.zeros(8, 2), current_contact_time=torch.ones(8, 2)*.02)
            self.scene.sensors = {"feet_ground_contact": SimpleNamespace(cfg=cfg, data=contact,
                _slots=[SimpleNamespace(primary_name=name, field_name=field)
                        for name in gait.CONTACTS for field in ("found", "force")])}
            self.episode_length_buf = torch.zeros(8, dtype=torch.long)
            self.command_manager = SimpleNamespace(get_command=lambda name:
                self.command if name == "twist" else torch.zeros(8, 4))
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
            self.env.episode_length_buf += 1
            data = self.env.scene["robot"].data
            data.data.time.fill_(self.env.count * .02)
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

    with pytest.raises(ValueError, match="requires unchanged body-yaw"):
        base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None, gait_trace=True)
    gait_case = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None,
                                   diagnostics=True, yaw_trace=True, gait_trace=True)
    gait_trace = gait_case.pop("gait_yaw_trace")
    gait_body_trace = gait_case.pop("body_yaw_trace")
    assert gait_case == default and gait_body_trace == trace
    assert all(gait_trace[key] == trace[key] for key in
               ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w"))
    assert closed == [240, 240, 240]
    assert gait_trace["episode_length"][0] == [0]*8
    assert gait_trace["episode_length"][-1] == [239]*8
    assert type(gait_trace["episode_length"][0][0]) is int
    assert gait_trace["physics_time"][-1] == pytest.approx([4.78]*8)
    assert type(gait_trace["contact_found"][0][0][0]) is float
    from mjlab_microduck.gait_yaw_trace import summarize_trace as summarize_gait
    gait_summary = summarize_gait(gait_case, gait_trace)
    assert gait_summary["recomputed_yaw_mae_per_world"] == pytest.approx(default["yaw_mae"])
    assert gait_summary["contact_pattern_count_per_world"]["two-contacts"] == [180]*8

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

    short_gait = base.evaluate_case(checkpoint, 839, .2, -.2, lambda: None,
                                    diagnostics=True, yaw_trace=True, gait_trace=True)
    assert short_gait["steps"] == short_gait["gait_yaw_trace"]["steps"] == 61
    assert short_gait["gait_yaw_trace"]["episode_length"][-1] == [60]*8
    with pytest.raises(ValueError, match="complete terminal-free"):
        summarize_gait(short_gait, short_gait["gait_yaw_trace"])
