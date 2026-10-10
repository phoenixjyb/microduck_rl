import copy
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import integrated_motor_refinement as refinement
from mjlab_microduck import integrated_turn_foundation as foundation


def test_freeze_materializes_source_stage_before_manager_construction():
    cfg, agent = refinement.prepare_config("motor")
    assert cfg.curriculum == {}
    assert cfg.rewards["action_rate_l2"].weight == -.8
    assert cfg.rewards["head_pose_bias"].weight == 2.
    assert cfg.commands["twist"].rel_standing_envs == .15
    assert cfg.commands["head_pose"].ranges == ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))
    assert cfg.commands["body_pose"].ranges == ((-.005, .005),)*3 + ((-.05, .05),)*3
    assert cfg.events["randomize_com"].params["ranges"] == (-.01, .01)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-.01, .01)
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 256
    assert cfg.seed == agent.seed == 863
    assert agent.max_iterations == 1000 and agent.save_interval == 250
    assert cfg.rewards["motor_load"].weight == 0.
    assert cfg.rewards["motor_load"].params == dict(rated_stall_torque_nm=.6, soft_limit_fraction=.7, over_limit_gain=4.)


def test_matched_configs_identical_except_labels_and_external_schedule():
    control, control_agent = refinement.prepare_config("control")
    motor, motor_agent = refinement.prepare_config("motor")
    # No curriculum/command/reward confound hidden in the two configurations.
    assert repr(control) == repr(motor)
    control_agent.run_name = motor_agent.run_name
    assert control_agent == motor_agent
    assert refinement.stages("control") == ((0, 0.),)
    assert refinement.stages("motor") == refinement.MOTOR_STAGES
    original, _ = foundation.prepare_config("foundation")
    assert original.rewards["action_rate_l2"].weight != -.8
    assert original.commands["twist"].rel_standing_envs == .02


@pytest.mark.parametrize("iteration,expected", [(0, 0.), (100, -.25), (300, -.5), (600, -1.)])
def test_live_schedule_strict_boundary(iteration, expected):
    schedule = refinement.MOTOR_STAGES
    previous = {0: 0., 100: 0., 300: -.25, 600: -.5}[iteration]
    assert refinement.motor_weight(iteration*24, schedule) == previous
    assert refinement.motor_weight(iteration*24+1, schedule) == expected


def test_smoke_exercises_penalty_and_benchmark_times_full_penalty():
    assert refinement.stages("smoke-motor") == ((0, 0.), (1, -.25), (2, -.5), (3, -1.))
    assert refinement.stages("benchmark-motor") == ((0, -1.),)
    assert refinement.MODES["smoke-motor"] == (64, 5, 859)
    assert refinement.MODES["benchmark-motor"] == (256, 10, 861)


def test_observer_validates_actual_reward_and_clears_bounded_update_state():
    term = SimpleNamespace(weight=0.)
    manager = SimpleNamespace(active_terms=["motor_load"], get_term_cfg=lambda name: term,
                              _step_reward=torch.zeros(2, 1))
    env = SimpleNamespace(common_step_counter=600*24+1, reward_manager=manager)
    observer = refinement.MotorObserver("motor")
    sample = SimpleNamespace(force_nm=torch.full((2, 14), .6), mean_cost=torch.ones(2))
    observer.before(env)
    assert term.weight == -1.
    # raw positive cost=1+4*(1-.7)^2=1.36; negative weighted term.
    manager._step_reward.fill_(-1.36)
    for _ in range(24):
        observer.step(env, sample, {})
    row = observer.update(env)
    assert row["weighted_motor_reward_mean"] == pytest.approx(-1.36)
    assert not observer.weights and not observer.costs and not observer.weighted
    manager._step_reward.fill_(1.)
    with pytest.raises(ValueError, match="nonpositive"):
        observer.step(env, sample, {})
    manager._step_reward.fill_(0.)
    with pytest.raises(ValueError, match="matches pre-reset"):
        observer.step(env, sample, {})


def test_exact_state_equality_includes_normalizers_and_adam():
    payload = dict(actor_state_dict={"obs_normalizer._mean": torch.zeros(61)},
                   optimizer_state_dict={"state": {0: {"step": torch.tensor(30000.)}}})
    assert refinement.equal_tree(payload, copy.deepcopy(payload))
    changed = copy.deepcopy(payload)
    changed["actor_state_dict"]["obs_normalizer._mean"][0] = 1.
    assert not refinement.equal_tree(payload, changed)
    assert not refinement.equal_tree(payload, {})


def test_unknown_source_curriculum_fails_closed():
    cfg, _ = foundation.prepare_config("foundation")
    cfg.curriculum["unexpected"] = cfg.curriculum["standing_envs"]
    with pytest.raises(ValueError, match="known source curricula"):
        refinement.freeze_source(cfg)


def test_warm_load_restores_every_learned_state_resets_clocks_and_matches_lr(monkeypatch):
    state = dict(actor_state_dict={"normalizer": torch.zeros(61)},
                 critic_state_dict={"normalizer": torch.zeros(80)},
                 optimizer_state_dict={"state": {0: {"step": torch.tensor(30000.)}},
                                       "param_groups": [{"lr": 2.25e-5}]},
                 iter=1499, infos={"env_state": {"common_step_counter": 36000}})
    env = SimpleNamespace(common_step_counter=36000, device="cpu")
    env.reset = lambda: None
    runner = SimpleNamespace(current_learning_iteration=1499)
    def load(path, **kwargs):
        assert kwargs == dict(strict=True, map_location="cpu")
        assert refinement.os.environ["MICRODUCK_WARM_START"] == "1"
        env.common_step_counter = runner.current_learning_iteration = 0
    runner.load = load
    runner.alg = SimpleNamespace(save=lambda: copy.deepcopy(state), learning_rate=1e-3)
    monkeypatch.setattr(refinement.base, "sha256", lambda path: refinement.PARENT_SHA)
    monkeypatch.setattr(refinement.torch, "load", lambda *args, **kwargs: copy.deepcopy(state))
    monkeypatch.setenv("MICRODUCK_WARM_START", "previous")
    receipt = refinement.warm_load(runner, env)
    assert receipt["learning_rate"] == runner.alg.learning_rate == 2.25e-5
    assert env.common_step_counter == runner.current_learning_iteration == 0
    assert refinement.os.environ["MICRODUCK_WARM_START"] == "previous"
    runner.alg.save = lambda: {**state, "actor_state_dict": {"normalizer": torch.ones(61)}}
    with pytest.raises(ValueError, match="exact restored actor"):
        refinement.warm_load(runner, env)


def test_paired_report_never_compares_tracking_from_partial_reset_case():
    left = [dict(seed=seed, speed=speed, yaw=yaw, complete=True, torque_p99=1.,
                 soft_limit_fraction=.2, rated_speed_exceed_fraction=0.,
                 mean_abs_mechanical_power_w=4., thermal_load_proxy=.15,
                 speed_mae=[.1]*8, yaw_mae=[.3]*8)
            for seed in foundation.SEEDS for speed, yaw in foundation.CASES]
    right = copy.deepcopy(left)
    right[0]["torque_p99"] = .8
    right[1].update(complete=False, speed_mae=None, yaw_mae=None)
    rows = refinement.paired_differences(left, right)
    assert rows[0]["motor_minus_control"]["torque_p99"] == pytest.approx(-.2)
    assert rows[0]["motor_minus_control"]["yaw_mae_mean"] == 0.
    assert not rows[1]["both_complete"] and rows[1]["motor_minus_control"]["yaw_mae_mean"] is None
    right.reverse()
    with pytest.raises(ValueError, match="matched case"):
        refinement.paired_differences(left, right)
