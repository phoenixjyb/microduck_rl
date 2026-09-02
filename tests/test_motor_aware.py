"""Unit and configuration tests for motor-aware Run fine-tuning."""

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.motor_aware import (
    MOTOR_AWARE_VELOCITY_STAGES,
    MOTOR_COST_WEIGHT_STAGES,
    MOTOR_OVER_LIMIT_GAIN,
    MOTOR_SOFT_LIMIT_FRACTION,
    STAGE1_CHECKPOINT_ITERATION,
    make_motor_aware_run_variant,
)
from mjlab_microduck.tasks.run import make_run_variant


class _Data:
    def __init__(self, forces):
        self.actuator_force = torch.tensor(forces, dtype=torch.float32)


class _Asset:
    def __init__(self, forces):
        self.data = _Data(forces)


class _Scene:
    def __init__(self, forces):
        self.asset = _Asset(forces)

    def __getitem__(self, _name):
        return self.asset


class _Env:
    def __init__(self, forces):
        self.num_envs = len(forces)
        self.device = "cpu"
        self.scene = _Scene(forces)
        self.extras = {"log": {}}


def _cost(env):
    return microduck_mdp.motor_torque_load_cost(
        env,
        rated_stall_torque_nm=0.60,
        soft_limit_fraction=0.70,
        over_limit_gain=4.0,
    )


def test_motor_cost_is_normalized_thermal_proxy_below_soft_limit():
    env = _Env([[0.30, 0.0]])
    out = _cost(env)
    assert float(out[0]) == torch.tensor((0.5**2 + 0.0) / 2.0).item()


def test_motor_cost_adds_squared_hinge_above_soft_limit():
    env = _Env([[0.60]])
    out = _cost(env)
    expected = 1.0 + 4.0 * (1.0 - 0.70) ** 2
    assert abs(float(out[0]) - expected) < 1e-6


def test_motor_cost_is_per_environment_and_nan_safe():
    env = _Env([[0.0, float("nan")], [0.30, float("inf")]])
    out = _cost(env)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert float(out[1]) > float(out[0])


def test_motor_cost_logs_training_metrics():
    env = _Env([[0.30, 0.60]])
    _cost(env)
    log = env.extras["log"]
    assert "Metrics/motor_training_cost_mean" in log
    assert "Metrics/motor_training_thermal_proxy_mean" in log
    assert "Metrics/motor_training_soft_limit_exceed_fraction" in log


def _cfg():
    return make_motor_aware_run_variant(
        make_run_variant(make_microduck_velocity_env_cfg())
    )


def test_motor_aware_reward_and_weight_curriculum_are_registered():
    cfg = _cfg()
    reward = cfg.rewards["motor_torque_load"]
    assert reward.func is microduck_mdp.motor_torque_load_cost
    assert reward.weight == MOTOR_COST_WEIGHT_STAGES[0]["weight"]
    assert reward.params["soft_limit_fraction"] == MOTOR_SOFT_LIMIT_FRACTION
    assert reward.params["over_limit_gain"] == MOTOR_OVER_LIMIT_GAIN
    curriculum = cfg.curriculum["motor_torque_load_weight"]
    assert curriculum.func is microduck_mdp.reward_weight
    assert curriculum.params["weight_stages"] == MOTOR_COST_WEIGHT_STAGES


def test_motor_aware_velocity_ladder_is_forward_only_and_capped():
    params = _cfg().curriculum["velocity_command_ranges"].params
    assert params["velocity_stages"] == MOTOR_AWARE_VELOCITY_STAGES
    assert params["forward_only"] is True
    assert params["update_lin_vel_y"] is False
    assert params["velocity_stages"][-1]["lin_vel_range"] == 0.80


def test_stage2_transitions_are_offset_from_resumed_checkpoint_iteration():
    velocity_steps = [stage["step"] for stage in MOTOR_AWARE_VELOCITY_STAGES]
    weight_steps = [stage["step"] for stage in MOTOR_COST_WEIGHT_STAGES]
    assert velocity_steps == [
        0,
        (STAGE1_CHECKPOINT_ITERATION + 750) * 24,
        (STAGE1_CHECKPOINT_ITERATION + 1500) * 24,
    ]
    assert weight_steps == [
        0,
        (STAGE1_CHECKPOINT_ITERATION + 250) * 24,
        (STAGE1_CHECKPOINT_ITERATION + 750) * 24,
        (STAGE1_CHECKPOINT_ITERATION + 1500) * 24,
    ]


def test_other_step_curricula_are_frozen_at_their_final_stage():
    cfg = _cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages == [{"step": 0, "weight": -1.0}]
    standing = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert standing == [{"step": 0, "rel_standing_envs": 0.25}]


def test_motor_aware_task_is_registered_with_distinct_run_identity():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_rl_cfg

    task_id = "Mjlab-Run-MotorAware-Flat-MicroDuck"
    assert task_id in list_tasks()
    rl_cfg = load_rl_cfg(task_id)
    assert rl_cfg.experiment_name == "run_motor_aware"
    assert rl_cfg.run_name == "stage2_motor_envelope"
