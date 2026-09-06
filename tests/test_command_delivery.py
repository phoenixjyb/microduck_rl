"""Real installed observation manager, synthetic CPU sensors; no simulation."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager

from mjlab_microduck.command_delivery import DIMENSIONS, TERMS, fresh_actor_twist
from mjlab_microduck.command_delivery import (
    LEGACY_PROTOCOL, PROTOCOL, prepare_actor_command_input, require_matching_delivery,
)
from mjlab_microduck.speed_response_control import prepare_config
from mjlab_microduck.tasks import mdp


def read_sensor(env, name):
    return env.sensors[name]


@pytest.fixture
def cpu_manager():
    cfg, _ = prepare_config()
    group = copy.deepcopy(cfg.observations["actor"])
    env = NS(num_envs=2, device="cpu")
    env.sensors = {name: torch.full((2, shape[0]), float(i))
                   for i, (name, shape) in enumerate(zip(TERMS, DIMENSIONS))}
    env.commands = {"twist": torch.tensor([[.3, 0., 0.], [.3, 0., 0.]]),
                    "head_pose": torch.zeros(2, 4), "body_pose": torch.zeros(2, 6)}
    env.command_manager = NS(get_command=lambda name: env.commands[name])
    for name in TERMS[:5]:
        group.terms[name].func = read_sensor
        group.terms[name].params = {"name": name}
    return env, ObservationManager({"actor": group}, env)


def test_command_write_after_observation_does_not_reach_same_step_actor(cpu_manager):
    env, manager = cpu_manager
    original = manager.compute(update_history=True)["actor"]
    command = env.commands["twist"]
    torch.testing.assert_close(original[:, 48:51], command)
    command[:] = torch.tensor([.1, 0., -.4])
    cached = manager.compute()["actor"]
    assert cached is original  # get_observations() also uses this cached path
    assert not torch.equal(cached[:, 48:51], command)
    assert torch.equal(cached[:, 48:51], torch.tensor([.3, 0., 0.]).expand(2, 3))
    next_step = manager.compute(update_history=True)["actor"]
    torch.testing.assert_close(next_step[:, 48:51], command)


def test_constant_command_has_no_cached_command_deficit(cpu_manager):
    env, manager = cpu_manager
    for _ in range(4):
        obs = manager.compute(update_history=True)["actor"]
        torch.testing.assert_close(obs[:, 48:51], env.commands["twist"])
        torch.testing.assert_close(manager.compute()["actor"][:, 48:51], env.commands["twist"])


def test_refresh_only_changes_twist_not_cache_sensors_rng_or_command(cpu_manager):
    env, manager = cpu_manager
    original = manager.compute(update_history=True)["actor"]
    before = original.clone()
    command = env.commands["twist"]
    command[:] = torch.tensor([.1, 0., .4])
    command_before = command.clone()
    rng_before = torch.get_rng_state().clone()
    refreshed = fresh_actor_twist(original, command, manager)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(command, command_before) and torch.equal(original, before)
    assert manager.compute()["actor"] is original
    torch.testing.assert_close(refreshed[:, :48], before[:, :48], rtol=0, atol=0)
    torch.testing.assert_close(refreshed[:, 51:], before[:, 51:], rtol=0, atol=0)
    torch.testing.assert_close(refreshed[:, 48:51], command, rtol=0, atol=0)
    refreshed.fill_(999)
    assert torch.equal(original, before) and torch.equal(command, command_before)


def test_direct_group_recompute_is_not_a_safe_command_refresh(cpu_manager):
    env, manager = cpu_manager
    manager.compute(update_history=True)
    env.sensors["joint_vel"].fill_(10)
    once = manager.compute(update_history=True)["actor"]
    rng_before = torch.get_rng_state().clone()
    twice = manager.compute_group("actor", update_history=False)
    # Even update_history=False unconditionally appends to the delay buffer.
    assert bool((once[:, 20:34] < 4).all())
    assert bool((twice[:, 20:34] > 9).all())
    assert not torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("change", [
    lambda m: m.active_terms["actor"].reverse(),
    lambda m: m.group_obs_term_dim["actor"].__setitem__(5, (4,)),
    lambda m: setattr(m.cfg["actor"], "concatenate_terms", False),
    lambda m: setattr(m.cfg["actor"], "history_length", 2),
    lambda m: setattr(m.get_term_cfg("actor", "command"), "scale", 2.),
    lambda m: setattr(m.get_term_cfg("actor", "command"), "clip", (-1., 1.)),
    lambda m: setattr(m.get_term_cfg("actor", "command"), "delay_max_lag", 1),
    lambda m: setattr(m.get_term_cfg("actor", "command"), "noise", object()),
    lambda m: setattr(m.get_term_cfg("actor", "command"), "params", {"command_name": "head_pose"}),
])
def test_layout_or_command_transform_drift_refuses_refresh(cpu_manager, change):
    env, manager = cpu_manager
    obs = manager.compute(update_history=True)["actor"]
    before = obs.clone()
    change(manager)
    with pytest.raises(ValueError): fresh_actor_twist(obs, env.commands["twist"], manager)
    assert torch.equal(obs, before)


@pytest.mark.parametrize("bad", ["nan", "batch", "width", "dtype"])
def test_bad_command_rejected(cpu_manager, bad):
    env, manager = cpu_manager
    obs = manager.compute(update_history=True)["actor"]
    command = env.commands["twist"].clone()
    if bad == "nan": command[0, 0] = float("nan")
    if bad == "batch": command = command[:1]
    if bad == "width": command = command[:, :2]
    if bad == "dtype": command = command.double()
    with pytest.raises(ValueError): fresh_actor_twist(obs, command, manager)


def test_play_noise_is_active_and_imu_bias_shared_fixed_not_step_noise():
    cfg, _ = prepare_config()
    actor = cfg.observations["actor"]
    assert actor.enable_corruption
    assert actor.terms["command"].noise is None
    assert actor.terms["joint_vel"].noise.n_min == -.25
    assert actor.terms["base_ang_vel"].params == {"max_angle_deg": 6.}
    assert actor.terms["projected_gravity"].params == {"max_angle_deg": 6.}
    env = NS(num_envs=16, device="cpu")
    env.scene = {"robot": NS(data=NS(projected_gravity_b=torch.tensor([0., 0., -1.]).expand(16, 3),
                                    root_link_ang_vel_b=torch.tensor([.1, .2, .3]).expand(16, 3)))}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(389)
        gravity = mdp.projected_gravity_imu_misaligned(env, max_angle_deg=6.)
        q = env._imu_misalign_quat
        rng = torch.get_rng_state().clone()
        angular = mdp.base_ang_vel_imu_misaligned(env, max_angle_deg=6.)
        assert env._imu_misalign_quat is q
        assert torch.equal(torch.get_rng_state(), rng)
        torch.testing.assert_close(gravity, mdp.quat_apply(q, env.scene["robot"].data.projected_gravity_b))
        torch.testing.assert_close(angular, mdp.quat_apply(q, env.scene["robot"].data.root_link_ang_vel_b))
        torch.testing.assert_close(gravity, mdp.projected_gravity_imu_misaligned(env, max_angle_deg=6.))


@pytest.mark.parametrize("protocol", [LEGACY_PROTOCOL, PROTOCOL])
def test_shared_adapter_trace_matches_actor_received_tensor_and_is_isolated(cpu_manager, protocol):
    from tensordict import TensorDict

    env, manager = cpu_manager
    raw = TensorDict(manager.compute(update_history=True), batch_size=[2])
    raw["critic"] = torch.zeros(2, 7)
    original = raw.clone()
    env.commands["twist"][:] = torch.tensor([.1, 0., -.4])
    rng = torch.get_rng_state().clone()
    delivery = prepare_actor_command_input(raw, env.commands["twist"], manager, protocol=protocol, step=0)
    assert torch.equal(torch.get_rng_state(), rng)
    # A spy at the inference boundary, not an executed trained actor.
    observed_by_policy = delivery.observations["actor"][:, 48:51].clone()
    torch.testing.assert_close(observed_by_policy, delivery.consumed)
    torch.testing.assert_close(raw["actor"], original["actor"])
    report = delivery.report()
    assert report["issued_input_equal_per_env"] == [protocol == PROTOCOL] * 2
    assert not any(report[k] for k in ("actor_inference_executed", "simulation_executed",
                                      "training_admitted", "policy_acceptance", "physical_motion_authorized"))
    # Later reset/command changes and caller edits cannot rewrite the snapshot.
    env.commands["twist"].zero_()
    raw["actor"].fill_(123)
    raw["critic"].fill_(123)
    assert delivery.report() == report
    assert bool((delivery.observations["critic"] == 0).all())
    delivery.observations["actor"].zero_()
    assert delivery.report() == report


def fake_lifecycle(env, manager, protocol, *, nested_training_loop):
    """Two loop shapes with identical command changes, including a reset row."""
    from tensordict import TensorDict

    raw = TensorDict(manager.compute(update_history=True), batch_size=[2])
    reports = []

    def tick(step):
        nonlocal raw
        if step in (0, 5):
            env.commands["twist"][:] = torch.tensor([.1 if step == 0 else .2, 0., -.4])
        delivery = prepare_actor_command_input(raw, env.commands["twist"], manager,
                                               protocol=protocol, step=step)
        reports.append(delivery.report())
        # Synthetic next-step observation acquisition. No dynamics or reward.
        raw = TensorDict(manager.compute(update_history=True), batch_size=[2])
        if step == 2:
            # Existing trainer resets the nominal command after obtaining the
            # reset observation. Only the reset row changes at this boundary.
            env.commands["twist"][1] = torch.tensor([.3, 0., 0.])

    if nested_training_loop:
        for macro in range(2):
            for micro in range(5): tick(macro * 5 + micro)
    else:
        for step in range(10): tick(step)
    return reports


@pytest.mark.parametrize("protocol", [LEGACY_PROTOCOL, PROTOCOL])
def test_flat_eval_and_five_step_training_lifecycles_share_timing(cpu_manager, protocol):
    env, manager = cpu_manager
    # Independent real managers use the same synthetic sensor and command setup.
    other_env = NS(num_envs=2, device="cpu", sensors=copy.deepcopy(env.sensors),
                   commands=copy.deepcopy(env.commands))
    other_env.command_manager = NS(get_command=lambda name: other_env.commands[name])
    other_manager = ObservationManager(copy.deepcopy(manager.cfg), other_env)
    flat = fake_lifecycle(env, manager, protocol, nested_training_loop=False)
    nested = fake_lifecycle(other_env, other_manager, protocol, nested_training_loop=True)
    assert flat == nested  # snapshots contain command fields, not sampled sensors
    mismatches = [(r["step"], i) for r in flat for i, equal in
                  enumerate(r["issued_input_equal_per_env"]) if not equal]
    assert mismatches == ([] if protocol == PROTOCOL else [(0, 0), (0, 1), (3, 1), (5, 0), (5, 1)])


@pytest.mark.parametrize("protocol,step", [("unknown", 0), (None, 0), (PROTOCOL, -1),
                                         (PROTOCOL, True), (PROTOCOL, .5)])
def test_adapter_rejects_ambiguous_identity_or_step(cpu_manager, protocol, step):
    from tensordict import TensorDict

    env, manager = cpu_manager
    raw = TensorDict(manager.compute(update_history=True), batch_size=[2])
    with pytest.raises(ValueError):
        prepare_actor_command_input(raw, env.commands["twist"], manager, protocol=protocol, step=step)


@pytest.mark.parametrize("training,evaluation", [(PROTOCOL, LEGACY_PROTOCOL),
    (LEGACY_PROTOCOL, PROTOCOL), (None, PROTOCOL), (LEGACY_PROTOCOL, None),
    ("unknown", PROTOCOL), (None, None)])
def test_changed_or_missing_delivery_metadata_cannot_claim_compatibility(training, evaluation):
    with pytest.raises(ValueError): require_matching_delivery(training, evaluation)


@pytest.mark.parametrize("protocol", [PROTOCOL, LEGACY_PROTOCOL])
def test_matching_delivery_metadata_is_only_compatibility(protocol):
    assert require_matching_delivery(protocol, protocol) is None  # no admission decision
