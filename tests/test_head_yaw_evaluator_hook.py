from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import gait_yaw_capture


HEAD_RANGES = ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))
BODY_RANGES = ((0., 0.),) * 6


def install_fake_evaluator(monkeypatch, *, terminal_at=None, passive=False, mutate=None):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry as registry

    events, configs, calls = [], [], {"policy": 0, "step": 0, "gait": 0, "yaw": 0}
    action = torch.full((8, 14), .25)
    twist = torch.tensor([.3, 0., 0.]).repeat(8, 1)
    head = torch.zeros(8, 4)
    body = torch.zeros(8, 6)
    data = NS(
        joint_pos=torch.zeros(8, 14), joint_pos_target=torch.zeros(8, 14),
        joint_pos_limits=torch.tensor([-1., 1.]).repeat(8, 14, 1),
        root_link_lin_vel_b=torch.tensor([.3, 0., 0.]).repeat(8, 1),
        root_link_ang_vel_b=torch.zeros(8, 3), root_link_ang_vel_w=torch.zeros(8, 3),
        root_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(8, 1),
    )
    robot = NS(data=data, joint_names=(
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll", "right_hip_yaw",
        "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ))
    if passive:
        robot.joint_names = robot.joint_names + ("passive_head_yaw",)
    command_manager = NS(get_command=lambda name: {
        "twist": twist, "head_pose": head, "body_pose": body,
    }[name])
    actor = torch.zeros(8, 61)
    actor[:, 48:51].copy_(twist)
    actor[:, 51:55].copy_(head)
    actor[:, 55:61].copy_(body)
    obs = {"actor": actor}
    env = NS(
        step_dt=.02, device="cpu", num_envs=8, scene={"robot": robot},
        command_manager=command_manager, episode_length_buf=torch.full((8,), 20, dtype=torch.long),
        common_step_counter=0, close=lambda: events.append("close"),
    )

    class Wrapper:
        clip_actions = None

        def __init__(self, _env, clip_actions=None):
            self.clip_actions = clip_actions

        def get_observations(self):
            return obs

        def step(self, raw):
            calls["step"] += 1
            events.append("step")
            assert raw is action
            done = torch.zeros(8, dtype=torch.bool)
            if terminal_at == calls["step"]:
                done[2] = True
            if mutate is not None and calls["step"] == 1:
                if mutate == "head":
                    head[0, 2] = .1
                elif mutate == "body":
                    body[0, 1] = .1
                elif mutate == "actor-head":
                    actor[0, 52] = .1
                elif mutate == "actor-body":
                    actor[0, 56] = .1
            env.common_step_counter += 1
            return obs, torch.zeros(8), done, {}

    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self, *_args, **_kwargs):
            pass

        def get_inference_policy(self, **_kwargs):
            def policy(observations):
                assert observations is obs
                calls["policy"] += 1
                events.append("policy")
                return action
            return policy

    class Stream:
        names = tuple(robot.joint_names[:14])
        joint_ids = tuple(range(14))

        @classmethod
        def from_robot(cls, *_args, **_kwargs):
            return cls()

        def begin(self, *_args):
            pass

        def consume(self, _dones):
            return NS(force_nm=torch.zeros(8, 14), speed_rad_s=torch.zeros(8, 14))

        def provenance(self):
            return {"test": "read-only CPU fixture"}

    @dataclass
    class Agent:
        seed: int = 0
        clip_actions: object = None

    head_cfg = NS(ranges=HEAD_RANGES, zero_command_prob=.25)
    body_cfg = NS(ranges=BODY_RANGES, zero_command_prob=.25)
    cfg = NS(
        seed=0, scene=NS(num_envs=0, terrain=NS(num_envs=0)), curriculum={}, events={},
        sim=NS(nan_guard=NS(enabled=False)), commands={"head_pose": head_cfg, "body_pose": body_cfg},
    )

    def make_env(received_cfg, device):
        assert received_cfg is cfg and device == "cuda:0"
        configs.append((tuple(head_cfg.ranges), head_cfg.zero_command_prob,
                        tuple(body_cfg.ranges), body_cfg.zero_command_prob))
        if head_cfg.zero_command_prob == 0.:
            assert head_cfg.ranges == ((0., 0.), (0., 0.), (head_cfg.ranges[2][0], head_cfg.ranges[2][0]), (0., 0.))
            assert head_cfg.ranges[2][0] in (-.2, .2)
            assert body_cfg.ranges == BODY_RANGES and body_cfg.zero_command_prob == .25
            head[:, 2] = head_cfg.ranges[2][0]
        else:
            assert tuple(head_cfg.ranges) == HEAD_RANGES and head_cfg.zero_command_prob == .25
        actor[:, 51:55].copy_(head)
        actor[:, 55:61].copy_(body)
        return env

    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", make_env)
    monkeypatch.setattr(mjlab.rl, "RslRlVecEnvWrapper", Wrapper)
    monkeypatch.setattr(registry, "load_env_cfg", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(registry, "load_rl_cfg", lambda *_args, **_kwargs: Agent())
    monkeypatch.setattr(registry, "load_runner_cls", lambda *_args, **_kwargs: Runner)
    monkeypatch.setattr(base, "MotorStepStream", Stream)
    monkeypatch.setattr(base, "install_metric", lambda _cfg: None)
    monkeypatch.setattr(base, "fix_velocity_commands", lambda *_args: None)

    def yaw_sample(_robot, command, actor_command):
        calls["yaw"] += 1
        events.append("yaw")
        assert torch.equal(command, twist) and torch.equal(actor_command, actor[:, 48:51])
        return dict(command=command.clone(), actor_command=actor_command.clone(),
                    quat_wxyz=robot.data.root_link_quat_w.clone(),
                    angular_b=robot.data.root_link_ang_vel_b.clone(),
                    angular_w=robot.data.root_link_ang_vel_w.clone())

    def gait_sample(received_env, received_robot, actor_obs):
        calls["gait"] += 1
        events.append("gait")
        before = actor_obs.clone()
        assert torch.equal(actor_obs[:, 48:51], twist)
        assert torch.equal(actor_obs[:, 51:55], head)
        assert torch.equal(actor_obs[:, 55:61], body)
        result = {"head": actor_obs[:, 51:55].clone(), "body": actor_obs[:, 55:61].clone(),
                  "step": received_env.common_step_counter}
        assert torch.equal(actor_obs, before)
        assert received_robot is robot
        return result

    monkeypatch.setattr(base, "body_yaw_trace_sample", yaw_sample)
    monkeypatch.setattr(gait_yaw_capture, "capture", gait_sample)
    monkeypatch.setattr(gait_yaw_capture, "trace", lambda samples, _digest: {
        "steps": len(samples), "samples": list(samples),
    })
    monkeypatch.setattr(base, "response_diagnostics", lambda *_args, **_kwargs: {"spy": True})
    return NS(events=events, configs=configs, calls=calls, env=env, robot=robot,
              head=head, body=body, actor=actor, twist=twist, cfg=cfg)


@pytest.mark.parametrize("head_yaw", [-.2, .2])
def test_head_yaw_command_is_pinned_before_construction_and_traces_one_policy_step_per_sample(
        monkeypatch, tmp_path, head_yaw):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry as registry

    fake = install_fake_evaluator(monkeypatch)
    # Ensure config mutations are complete before the environment is built.
    original_factory = mjlab.envs.ManagerBasedRlEnv

    def checked_factory(cfg, device):
        assert cfg.commands["head_pose"].ranges == (
            (0., 0.), (0., 0.), (head_yaw, head_yaw), (0., 0.))
        assert cfg.commands["head_pose"].zero_command_prob == 0.
        assert cfg.commands["body_pose"].ranges == BODY_RANGES
        result = original_factory(cfg, device)
        assert torch.equal(fake.actor[:, 51:55], fake.head)
        assert torch.equal(fake.actor[:, 55:61], fake.body)
        return result

    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", checked_factory)
    # The fixture's factory reads values during construction; the wrapper checks
    # that the exact diagnostic command reached actor slots before stepping.
    old_make = registry.load_env_cfg
    def cfg_with_command(*args, **kwargs):
        cfg = old_make(*args, **kwargs)
        return cfg

    monkeypatch.setattr(registry, "load_env_cfg", cfg_with_command)
    from mjlab_microduck import body_yaw_trace
    monkeypatch.setattr(body_yaw_trace, "summarize_trace", lambda *_args, **_kwargs: {"spy": True})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cpu test checkpoint")
    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
        diagnostics=True, yaw_trace=True, gait_trace=True, head_yaw_command=head_yaw)
    assert fake.calls == {"policy": 240, "step": 240, "gait": 240, "yaw": 240}
    assert fake.events == ["yaw", "gait", "policy", "step"] * 240 + ["close"]
    assert fake.events.count("close") == 1
    assert result["head_yaw_command"] == head_yaw
    assert result["head_measurement_view"] == "named plain-servo qpos minus HOME"
    assert result["response_diagnostics"] == {"spy": True}
    assert result["gait_yaw_trace"]["steps"] == result["body_yaw_trace"]["steps"] == 240
    assert result["gait_yaw_trace"]["samples"][-1]["head"][0, 2] == head_yaw
    assert result["gait_yaw_trace"]["samples"][-1]["body"].eq(0.).all()
    assert result["complete"] is True


def test_default_none_keeps_original_config_and_single_policy_step_path(monkeypatch, tmp_path):
    fake = install_fake_evaluator(monkeypatch)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cpu test checkpoint")
    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: None)
    assert fake.configs == [(HEAD_RANGES, .25, BODY_RANGES, .25)]
    assert fake.calls == {"policy": 240, "step": 240, "gait": 0, "yaw": 0}
    assert fake.events == ["policy", "step"] * 240 + ["close"]
    assert result["complete"] is True
    assert "head_yaw_command" not in result and "gait_yaw_trace" not in result


def test_terminal_keeps_partial_raw_gait_and_body_yaw_traces(monkeypatch, tmp_path):
    fake = install_fake_evaluator(monkeypatch, terminal_at=65)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cpu test checkpoint")
    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
        diagnostics=True, yaw_trace=True, gait_trace=True, head_yaw_command=.2)
    assert result["complete"] is False and result["steps"] == 65
    assert result["terminal_worlds"] == [2]
    assert fake.calls == {"policy": 65, "step": 65, "gait": 65, "yaw": 65}
    assert result["gait_yaw_trace"]["steps"] == 65
    assert len(result["gait_yaw_trace"]["samples"]) == 65
    assert result["gait_yaw_trace"]["samples"][-1]["head"][0, 2] == .2
    assert result["body_yaw_trace"]["steps"] == 65
    assert len(result["body_yaw_trace"]["command"]) == 65


@pytest.mark.parametrize("command", [True, float("nan"), .1])
def test_invalid_head_yaw_command_rejected_before_environment_construction(monkeypatch, command):
    fake = install_fake_evaluator(monkeypatch)
    with pytest.raises(ValueError):
        base.evaluate_case(None, 839, .3, 0., lambda: None,
            diagnostics=True, yaw_trace=True, gait_trace=True, head_yaw_command=command)
    assert fake.configs == [] and fake.calls["step"] == fake.calls["policy"] == 0


@pytest.mark.parametrize("flags", [
    {}, {"diagnostics": True}, {"yaw_trace": True}, {"gait_trace": True},
    {"diagnostics": True, "yaw_trace": True},
])
def test_head_yaw_command_requires_full_position_body_yaw_and_gait_capture_flags(flags):
    with pytest.raises(ValueError):
        base.evaluate_case(None, 839, .3, 0., lambda: None, head_yaw_command=.2, **flags)


def test_head_yaw_command_rejects_passive_output_joints_and_closes_env(monkeypatch, tmp_path):
    fake = install_fake_evaluator(monkeypatch, passive=True)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cpu test checkpoint")
    with pytest.raises(ValueError, match="plain servo"):
        base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
            diagnostics=True, yaw_trace=True, gait_trace=True, head_yaw_command=.2)
    assert fake.events == ["close"]
    assert fake.calls["policy"] == fake.calls["step"] == fake.calls["gait"] == 0


@pytest.mark.parametrize("mutation", ["head", "body", "actor-head", "actor-body"])
def test_head_yaw_live_manager_and_actor_slots_are_revalidated_each_step(monkeypatch, tmp_path, mutation):
    fake = install_fake_evaluator(monkeypatch, mutate=mutation)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cpu test checkpoint")
    with pytest.raises(ValueError, match="fixed diagnostic head/body"):
        base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
            diagnostics=True, yaw_trace=True, gait_trace=True, head_yaw_command=.2)
    # The changed command is produced after one valid step and rejected at the
    # next pre-policy boundary; no second policy action is issued.
    assert fake.calls["policy"] == fake.calls["step"] == 1
    assert fake.calls["gait"] == fake.calls["yaw"] == 1
    assert fake.events[-1] == "close"
