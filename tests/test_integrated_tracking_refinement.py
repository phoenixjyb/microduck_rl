import copy
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_tracking_refinement as tracking
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck.tasks.mdp import body_twist_tracking_cost


def fake_env(command, planar, angular):
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor(planar),
                           root_link_ang_vel_b=torch.tensor(angular))
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(command)),
        num_envs=len(command), device="cpu")


def test_cost_tracks_command_not_absolute_motion_and_ignores_gait_axes():
    env = fake_env([[.3, 0., .2], [.3, 0., -.2]], [[.3, 0., 10.], [0., .1, -10.]],
                   [[5., 6., .2], [-5., -6., .1]])
    assert body_twist_tracking_cost(env).tolist() == pytest.approx([0., .19])
    env.scene["robot"].data.root_link_ang_vel_b[1, 2] = -.2
    assert body_twist_tracking_cost(env).tolist() == pytest.approx([0., .10])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cost_fails_closed_without_sanitizing_raw_state(value):
    env = fake_env([[.3, 0., .2]], [[value, 0., 0.]], [[0., 0., 0.]])
    with pytest.raises(FloatingPointError, match="raw body twist"):
        body_twist_tracking_cost(env)


def test_matched_configs_have_only_command_reward_weight_difference():
    control, ca = tracking.prepare_config("control")
    candidate, ta = tracking.prepare_config("tracking")
    assert control.rewards["command_error"].weight == 0.
    assert candidate.rewards["command_error"].weight == -.5
    candidate.rewards["command_error"].weight = 0.
    assert repr(candidate) == repr(control)
    ta.run_name = ca.run_name
    assert ca == ta
    assert control.curriculum == {} and control.rewards["motor_load"].weight == -1.
    assert control.rewards["action_rate_l2"].weight == -.8
    assert control.rewards["track_angular_velocity"].weight == 2.
    assert ca.seed == control.seed == 881 and ca.max_iterations == 1000
    assert "microduck_tracking_cost" in control.metrics
    assert control.commands["twist"].ranges.lin_vel_x == (.2, .4)
    assert control.commands["twist"].ranges.ang_vel_z == (-.5, .5)


def test_every_preflight_uses_same_actual_penalty_and_learned_motor_stage():
    for mode, (worlds, updates, seed) in tracking.MODES.items():
        cfg, agent = tracking.prepare_config(mode)
        observer = tracking.MotorObserver(mode)
        assert cfg.scene.num_envs == worlds and agent.max_iterations == updates and agent.seed == seed
        assert observer.schedule == ((0, -1.),)
        assert cfg.rewards["command_error"].weight == observer.tracking_weight
        assert observer.tracking_weight == (-.5 if mode.endswith("tracking") else 0.)


@pytest.mark.parametrize("mode", ["control", "tracking"])
def test_metric_and_observer_bind_reward_even_after_terminal_reset(mode):
    env = fake_env([[.3, 0., .2]], [[0., 0., 0.]], [[0., 0., 0.]])
    terms = {"motor_load": SimpleNamespace(weight=0.), "command_error": SimpleNamespace(weight=0.)}
    env.reward_manager = SimpleNamespace(active_terms=list(terms), get_term_cfg=lambda name: terms[name],
                                         _step_reward=torch.zeros(1, 2))
    env.common_step_counter = 0
    observer = tracking.MotorObserver(mode)
    sample = SimpleNamespace(force_nm=torch.zeros(1, 14), mean_cost=torch.zeros(1))
    for _ in range(24):
        observer.before(env)
        assert terms["motor_load"].weight == -1.
        tracking.capture_tracking_metric(env)
        env.reward_manager._step_reward[:, 1] = env._microduck_tracking_cost*observer.tracking_weight
        # A terminal resets the live robot, not the retained metric snapshot.
        env.scene["robot"].data.root_link_lin_vel_b.fill_(100.)
        observer.step(env, sample, {})
        assert env._microduck_tracking_cost is None
        env.scene["robot"].data.root_link_lin_vel_b.zero_()
    row = observer.update(env)
    assert row["command_cost_mean"] == pytest.approx(.13)
    assert row["weighted_command_reward_mean"] == pytest.approx(.13*observer.tracking_weight)
    assert not observer.tracking_costs and not observer.tracking_rewards and not observer.tracking_maxima
    observer.before(env)
    with pytest.raises(ValueError, match="pre-reset command cost"):
        observer.step(env, sample, {})
    tracking.capture_tracking_metric(env)
    env.reward_manager._step_reward[:, 1] = 1.
    with pytest.raises(ValueError, match="reward matches"):
        observer.step(env, sample, {})


def test_parent_wrapper_uses_exact_motor_checkpoint_not_original_foundation(monkeypatch):
    calls = []
    monkeypatch.setattr(motor, "warm_load", lambda *a, **k: calls.append((a, k)) or {"ok": True})
    assert tracking.warm_load("runner", "env") == {"ok": True}
    assert calls == [(("runner", "env", tracking.PARENT), dict(parent_sha=tracking.PARENT_SHA,
                     parent_iteration=999, parent_steps=24000))]


def test_pair_delta_label_and_decision_protocol_are_not_relaxed():
    rows = [dict(seed=seed, speed=speed, yaw=yaw, complete=True, torque_p99=.5,
        soft_limit_fraction=0., rated_speed_exceed_fraction=0., mean_abs_mechanical_power_w=1.,
        thermal_load_proxy=.1, speed_mae=[.02]*8, yaw_mae=[.09]*8, worlds=8,
        steps=240, terminal_worlds=[]) for seed in tracking.base.SEEDS for speed, yaw in tracking.base.CASES]
    candidate = copy.deepcopy(rows)
    candidate[0]["yaw_mae"][0] = .11
    diff = motor.paired_differences(rows, candidate, label="tracking_minus_control")
    assert "tracking_minus_control" in diff[0] and "motor_minus_control" not in diff[0]
    assert tracking.base.decision(candidate)["decision"] == "foundation-not-ready"
