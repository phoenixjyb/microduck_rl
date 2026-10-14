from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.selected_motor_load import selected_motor_torque_load_cost


ACTUATORS = ("left_knee", "right_knee", "left_ankle", "right_ankle")


class FakeAsset:
    def __init__(self, actuator_force, ids=(8, 3, 11, 1), names=ACTUATORS):
        self.data = SimpleNamespace(actuator_force=actuator_force)
        self.ids = ids
        self.names = names
        self.find_calls = []

    def find_actuators(self, names, preserve_order=False):
        self.find_calls.append((names, preserve_order))
        return list(self.ids), list(self.names)


class FakeEnv:
    def __init__(self, force, **asset_kwargs):
        self.num_envs = force.shape[0]
        self.device = force.device
        self.scene = {"robot": FakeAsset(force, **asset_kwargs)}


def test_selects_exact_named_columns_across_interleaved_joint_layout():
    force = torch.zeros((2, 13), dtype=torch.float64)
    force[0, [8, 3, 11, 1]] = torch.tensor(
        [0.12, -0.30, 0.42, -0.06], dtype=torch.float64
    )
    force[1, [8, 3, 11, 1]] = torch.tensor(
        [-0.60, 0.00, 0.15, 0.24], dtype=torch.float64
    )
    force[:, [0, 2, 4, 5, 6, 7, 9, 10, 12]] = 1000.0  # passive/unselected columns
    env = FakeEnv(force)
    before = force.clone()

    actual = selected_motor_torque_load_cost(env, ACTUATORS)
    selected = before[:, [8, 3, 11, 1]].abs() / 0.60
    expected = (selected.square() + 4.0 * torch.relu(selected - 0.70).square()).mean(1)

    assert torch.allclose(actual, expected)
    assert actual.dtype == torch.float64
    assert actual.device == force.device
    assert env.scene["robot"].find_calls == [(ACTUATORS, True)]
    assert torch.equal(force, before)


def test_cost_is_sign_symmetric_per_environment_and_ignores_other_actuators():
    force = torch.zeros((2, 6), dtype=torch.float32)
    force[0, [1, 4]] = torch.tensor([0.30, 0.60])
    force[1, [1, 4]] = torch.tensor([-0.30, -0.60])
    env = FakeEnv(force, ids=(1, 4), names=("left_knee", "right_knee"))
    first = selected_motor_torque_load_cost(env, ("left_knee", "right_knee"))
    force[:, [0, 2, 3, 5]] = float("-10000")
    second = selected_motor_torque_load_cost(env, ("left_knee", "right_knee"))

    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize(
    ("ids", "names", "requested", "message"),
    [
        (
            (0, 1),
            ("left_knee", "right_knee"),
            ("right_knee", "left_knee"),
            "exactly match",
        ),
        ((0,), ("left_knee",), ("left_knee", "right_knee"), "count"),
        ((0, 0), ("left_knee", "right_knee"), ("left_knee", "right_knee"), "unique"),
        (
            (0, 9),
            ("left_knee", "right_knee"),
            ("left_knee", "right_knee"),
            "out of bounds",
        ),
    ],
)
def test_invalid_actuator_resolution_fails_closed(ids, names, requested, message):
    env = FakeEnv(torch.zeros((1, 4)), ids=ids, names=names)
    with pytest.raises(ValueError, match=message):
        selected_motor_torque_load_cost(env, requested)


def test_duplicate_or_empty_requested_names_fail_closed():
    env = FakeEnv(torch.zeros((1, 4)))
    for names in ((), ("left_knee", "left_knee")):
        with pytest.raises(ValueError, match="actuator_names"):
            selected_motor_torque_load_cost(env, names)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rated_stall_torque_nm": 0.0}, "positive"),
        ({"rated_stall_torque_nm": float("nan")}, "finite"),
        ({"rated_stall_torque_nm": float("inf")}, "finite"),
        ({"soft_limit_fraction": 0.0}, "between"),
        ({"soft_limit_fraction": 1.0}, "between"),
        ({"soft_limit_fraction": float("nan")}, "finite"),
        ({"over_limit_gain": -1.0}, "non-negative"),
        ({"over_limit_gain": float("inf")}, "finite"),
        ({"over_limit_gain": True}, "finite"),
    ],
)
def test_invalid_scalars_fail_closed(kwargs, message):
    env = FakeEnv(torch.zeros((1, 12)))
    with pytest.raises(ValueError, match=message):
        selected_motor_torque_load_cost(env, ACTUATORS, **kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_force_fails_closed(bad):
    force = torch.zeros((1, 12))
    force[0, 8] = bad
    with pytest.raises(ValueError, match="non-finite"):
        selected_motor_torque_load_cost(FakeEnv(force), ACTUATORS)


def test_force_shape_type_and_integer_dtype_fail_closed():
    with pytest.raises(ValueError, match="rank-2"):
        selected_motor_torque_load_cost(FakeEnv(torch.zeros(4)), ACTUATORS)
    wrong_env_count = FakeEnv(torch.zeros((1, 12)))
    wrong_env_count.num_envs = 2
    with pytest.raises(ValueError, match="shape"):
        selected_motor_torque_load_cost(wrong_env_count, ACTUATORS)
    with pytest.raises(ValueError, match="floating-point"):
        selected_motor_torque_load_cost(
            FakeEnv(torch.zeros((1, 12), dtype=torch.int64)), ACTUATORS
        )


def test_call_does_not_mutate_environment_or_force_tensor():
    force = torch.tensor([[0.2] * 12, [-0.1] * 12])
    env = FakeEnv(force)
    before = force.clone()
    state = (env.num_envs, env.device, tuple(env.scene.keys()))

    result = selected_motor_torque_load_cost(env, ACTUATORS)

    assert torch.equal(force, before)
    assert (env.num_envs, env.device, tuple(env.scene.keys())) == state
    assert result.shape == (2,)
