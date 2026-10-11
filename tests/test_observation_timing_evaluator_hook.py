"""One inference/step per sample; no sensor resampling in the optional hook."""

import pytest
import torch

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import observation_timing_capture as capture
from mjlab_microduck import angular_reward_diagnostics as angular


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("terminal_at", [None, 3])
def test_hook_preserves_inference_steps_and_first_terminal(monkeypatch, tmp_path, enabled, terminal_at):
    import mjlab.rl
    from test_head_yaw_evaluator_hook import install_fake_evaluator

    fake = install_fake_evaluator(monkeypatch, terminal_at=terminal_at)
    fake.cfg.metrics = {}
    captured, paired = [], []
    original_step = mjlab.rl.RslRlVecEnvWrapper.step

    def step(self, actions):
        result = original_step(self, actions)
        fake.env._microduck_angular_reward_snapshot = torch.zeros(8, 5)
        if enabled:
            fake.env._microduck_observation_reward_snapshot = {"counter": fake.env.common_step_counter}
        return result

    def pre(env, robot, actor):
        assert enabled and robot is fake.robot and actor is fake.actor
        fake.events.append("capture-pre")
        value = {"counter": env.common_step_counter, "actor": actor.clone()}
        captured.append(value)
        return value

    def post(env, value):
        assert enabled and env.common_step_counter == value["counter"] + 1
        assert env._microduck_angular_reward_snapshot is None
        assert env._microduck_observation_reward_snapshot["counter"] == env.common_step_counter
        fake.events.append("attach-post")
        row = {"pre": value, "post": env._microduck_observation_reward_snapshot}
        env._microduck_observation_reward_snapshot = None
        paired.append(row)
        return row

    def serialize(samples, digest):
        assert enabled and samples == paired and digest == base.sha256(checkpoint)
        assert all(torch.equal(row["pre"]["actor"], fake.actor) for row in samples)
        return {"steps": len(samples), "counter": samples[-1]["post"]["counter"]}

    monkeypatch.setattr(mjlab.rl.RslRlVecEnvWrapper, "step", step)
    monkeypatch.setattr(capture, "capture_pre", pre)
    monkeypatch.setattr(capture, "attach_post", post)
    monkeypatch.setattr(capture, "serialize", serialize)
    monkeypatch.setattr(angular, "summarize", lambda samples, **_kw: {"steps": len(samples)})
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"CPU-only evaluator hook fixture")
    result = base.evaluate_case(checkpoint, 839, .3, 0., lambda: None,
        angular_diagnostics=True, observation_trace=enabled)
    steps = terminal_at or 240
    expected = ["capture-pre", "policy", "step", "attach-post"] if enabled else ["policy", "step"]
    assert fake.events == expected * steps + ["close"]
    assert fake.calls["policy"] == fake.calls["step"] == result["steps"] == steps
    assert len(captured) == len(paired) == (steps if enabled else 0)
    assert result["complete"] is (terminal_at is None)
    assert result["terminal_worlds"] == ([] if terminal_at is None else [2])
    assert ("observation_timing_trace" in result) is enabled
    assert fake.cfg.metrics["microduck_angular_reward_audit"].func is (
        capture.capture_reward if enabled else angular.capture)
    if enabled:
        assert result["observation_timing_trace"] == {"steps": steps, "counter": steps}


def test_optional_observation_trace_requires_post_reward_capture():
    with pytest.raises(ValueError, match="requires post-reward angular diagnostics"):
        base.evaluate_case(None, 839, .3, 0., lambda: None, observation_trace=True)


def test_actual_play_factory_keeps_declared_corrupted_angular_view():
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import mdp

    cfg = load_env_cfg(base.TASK, play=True)
    group = cfg.observations["actor"]
    assert tuple(group.terms) == ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                                 "actions", "command", "head_command", "body_command")
    assert group.enable_corruption and group.concatenate_terms
    term = group.terms["base_ang_vel"]
    assert term.func is mdp.base_ang_vel_imu_misaligned
    assert term.params == {"max_angle_deg": 6.}
    assert term.scale is term.clip is None and term.history_length == 0
    assert (term.delay_min_lag, term.delay_max_lag, term.delay_update_period) == (0, 1, 64)
    assert (term.noise.n_min, term.noise.n_max, term.noise.operation) == (-.03, .03, "add")
