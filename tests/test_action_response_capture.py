from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import action_response_capture as capture


@pytest.fixture
def runtime():
    from mjlab.entity import Entity
    from mjlab.managers.action_manager import ActionManager
    from mjlab_microduck.foundation_yaw_experiment import prepare_config

    cfg, _agent = prepare_config("pilot", arm="yaw")
    robot = Entity(cfg.scene.entities["robot"])
    assert tuple(robot.joint_names) == capture.JOINTS
    worlds, joints = capture.WORLDS, len(capture.JOINTS)
    position = torch.linspace(-.1, .1, joints).repeat(worlds, 1)
    bias = torch.linspace(-.01, .01, joints).repeat(worlds, 1)
    robot._data = SimpleNamespace(
        default_joint_pos=position.clone(),
        encoder_bias=bias.clone(),
        joint_pos_target=torch.zeros(worlds, joints),
        joint_pos_limits=torch.tensor([-1., 1.]).repeat(worlds, joints, 1),
        joint_pos=position.clone(),
        joint_vel=torch.zeros(worlds, joints),
        indexing=SimpleNamespace(joint_v_adr=torch.arange(joints)),
        data=SimpleNamespace(qfrc_constraint=torch.zeros(worlds, joints),
                             time=torch.zeros(worlds, dtype=torch.float64)),
        root_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(worlds, 1),
        root_link_ang_vel_b=torch.zeros(worlds, 3),
        root_link_ang_vel_w=torch.zeros(worlds, 3),
    )
    command = torch.tensor([.2, 0., -.2]).repeat(worlds, 1)
    env = SimpleNamespace(num_envs=worlds, step_dt=capture.STEP_DT, device="cpu", cfg=cfg,
                          scene={"robot": robot}, episode_length_buf=torch.zeros(worlds, dtype=torch.long),
                          common_step_counter=0,
                          command_manager=SimpleNamespace(get_command=lambda name: command))
    env.action_manager = ActionManager(cfg.actions, env)
    term = env.action_manager.get_term("joint_pos")
    stream = SimpleNamespace(names=tuple(term.target_names),
                             joint_ids=tuple(int(i) for i in term.target_ids.tolist()))
    actor_obs = torch.zeros(worlds, capture.ACTOR_WIDTH)
    actor_obs[:, 48:51] = command
    return env, robot, actor_obs, stream, command


def advance_one(env, robot, actor_obs, actions):
    term = env.action_manager.get_term("joint_pos")
    env.action_manager.process_action(actions)
    env.action_manager.apply_action()
    env.common_step_counter += 1
    env.episode_length_buf += 1
    robot.data.data.time += capture.STEP_DT
    robot.data.joint_pos.add_(.001)
    robot.data.joint_vel.add_(.01)
    actor_obs[:, 34:48] = actions


def test_capture_clones_named_preaction_state_and_raw_action_without_mutation(runtime):
    env, robot, actor_obs, stream, command = runtime
    rng = torch.random.get_rng_state().clone()
    action_before = env.action_manager.action.clone()
    target_before = robot.data.joint_pos_target.clone()
    obs_before = actor_obs.clone()
    state = capture.capture_state(env, robot, actor_obs, stream)
    assert state["command"].shape == (8, 3)
    assert state["actor_previous_action"].shape == (8, 14)
    assert state["actor_previous_action"].eq(0.).all()
    assert state["raw_previous_action"].equal(state["actor_previous_action"])
    assert state["joint_position"].shape == (8, 14)
    assert state["hard_limits"].shape == (8, 14, 2)
    assert state["action_offset"].shape == (8, 14)
    assert state["generalized_constraint_force"].shape == (8, 14)
    assert state["episode_length"].dtype == torch.int64
    assert state["common_step_counter"].dtype == torch.int64
    assert state["common_step_counter"].shape == ()
    assert tuple(robot.joint_names[i] for i in state["target_ids"]) == capture.JOINTS
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(env.action_manager.action, action_before)
    assert torch.equal(robot.data.joint_pos_target, target_before)
    assert torch.equal(actor_obs, obs_before)

    raw = torch.full((8, 14), 1.25)
    paired = capture.add_action(state, raw)
    assert paired["raw_action"].shape == (8, 14)
    assert paired["raw_action"].eq(1.25).all()  # raw output is not clipped or called saturated
    raw.zero_()
    actor_obs.zero_()
    robot.data.joint_pos.zero_()
    assert paired["raw_action"].eq(1.25).all()
    assert state["joint_position"].abs().sum() > 0
    assert state["actor_command"].equal(command)
    capture.validate_wrapper(SimpleNamespace(clip_actions=None))
    with pytest.raises(ValueError, match="clipping is disabled"):
        capture.validate_wrapper(SimpleNamespace(clip_actions=1.0))


def test_trace_aligns_current_action_to_next_state_target_and_keeps_clocks_integer(runtime):
    env, robot, actor_obs, stream, _command = runtime
    # A restored evaluation runner may begin at a nonzero global clock.
    env.common_step_counter = 24000
    env.episode_length_buf.fill_(17)
    robot.data.data.time.fill_(480.)
    actor_obs[:, 34:48] = env.action_manager.action
    first = capture.capture_state(env, robot, actor_obs, stream)
    raw = torch.linspace(-1.2, 1.2, 14).repeat(8, 1)
    first = capture.add_action(first, raw)
    advance_one(env, robot, actor_obs, raw)
    second = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), raw + .1)

    before = deepcopy((first, second))
    result = capture.trace([first, second], "a" * 64)
    assert first["raw_action"].equal(before[0]["raw_action"])
    assert second["raw_action"].equal(before[1]["raw_action"])
    assert result["protocol"] == capture.PROTOCOL
    assert result["joint_columns"] == list(capture.JOINTS)
    assert result["steps"] == 2 and result["worlds"] == 8
    assert result["episode_length"][0] == [17] * 8
    assert result["episode_length"][1] == [18] * 8
    assert result["common_step_counter"] == [24000, 24001]
    assert result["physics_time"][1] == pytest.approx([480.02] * 8)
    assert result["raw_action"][0][0] == pytest.approx(raw[0].tolist())
    assert result["timing"] == capture.STATE_TIMING
    assert result["action_timing"] == "raw policy output after inference before step"
    assert result["target_timing"] == "previous applied target at pre-action boundary"
    assert result["constraint_timing"] == "last-physics-derived generalized constraint at pre-action boundary"
    assert result["target_ids"] == list(stream.joint_ids)
    assert result["startup_steps"] == 60
    assert result["simulator_qualified"] is False
    assert result["constraint_includes_contacts_and_limits"] is True
    assert result["hard_stop_force_isolated"] is False
    assert result["policy_acceptance"] is result["physical_motion_authorized"] is False


@pytest.mark.parametrize("fault", ["action-order", "action-ids", "active-terms", "scale", "clip"])
def test_capture_fails_closed_on_passive_reordered_or_changed_action_binding(runtime, fault):
    env, robot, actor_obs, stream, _ = runtime
    if fault == "action-order":
        stream.names = tuple(reversed(stream.names))
    elif fault == "action-ids":
        stream.joint_ids = tuple(reversed(stream.joint_ids))
    elif fault == "active-terms":
        env.action_manager._term_names = ["other"]
    elif fault == "scale":
        env.action_manager.get_term("joint_pos").cfg.scale = 2.0
    else:
        env.action_manager.get_term("joint_pos").cfg.clip = {".*": (-1., 1.)}
    with pytest.raises(ValueError):
        capture.capture_state(env, robot, actor_obs, stream)


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "actor-history", "clock", "limits", "negative-time"])
def test_state_capture_rejects_malformed_inputs(runtime, fault):
    env, robot, actor_obs, stream, _ = runtime
    if fault == "shape":
        actor_obs = actor_obs[:, :-1]
    elif fault == "nonfinite":
        robot.data.joint_vel[0, 0] = float("nan")
    elif fault == "actor-history":
        actor_obs[0, 34] = 1.
    elif fault == "clock":
        env.episode_length_buf = torch.zeros(8, dtype=torch.bool)
    elif fault == "limits":
        robot.data.joint_pos_limits[0, 0] = torch.tensor([1., -1.])
    else:
        robot.data.data.time[0] = -.02
    with pytest.raises(ValueError):
        capture.capture_state(env, robot, actor_obs, stream)


@pytest.mark.parametrize("fault", ["missing-action", "reset", "time-gap", "action-gap", "target-mismatch",
                                    "command-change", "clock-gap", "target-ids"])
def test_trace_rejects_partial_or_misaligned_action_response_pairs(runtime, fault):
    env, robot, actor_obs, stream, _ = runtime
    first = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), torch.zeros(8, 14))
    advance_one(env, robot, actor_obs, torch.zeros(8, 14))
    second = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), torch.ones(8, 14))
    rows = [first, second]
    if fault == "missing-action":
        rows[0] = dict(rows[0])
        rows[0].pop("raw_action")
    elif fault == "reset":
        rows[1]["episode_length"].zero_()
    elif fault == "time-gap":
        rows[1]["physics_time"].add_(.02)
    elif fault == "action-gap":
        rows[1]["actor_previous_action"].fill_(.5)
    elif fault == "command-change":
        rows[1]["command"][:, 2] = .2
        rows[1]["actor_command"][:, 2] = .2
    elif fault == "clock-gap":
        rows[1]["common_step_counter"].add_(1)
    elif fault == "target-ids":
        rows[1]["target_ids"] = tuple(reversed(rows[1]["target_ids"]))
    else:
        rows[1]["previous_applied_target"].add_(.25)
    with pytest.raises(ValueError):
        capture.trace(rows, "a" * 64)


def test_action_trace_rejects_bad_checkpoint_identity_and_action_magnitude_is_not_bounded(runtime):
    env, robot, actor_obs, stream, _ = runtime
    state = capture.capture_state(env, robot, actor_obs, stream)
    raw = torch.full((8, 14), 4.0)
    assert capture.add_action(state, raw)["raw_action"].eq(4.).all()
    with pytest.raises(ValueError, match="SHA256"):
        capture.trace([capture.add_action(state, raw)], "A" * 64)


def test_constraint_capture_slices_cpu_warp_storage_before_tensor_validation(runtime):
    import warp as wp
    from mjlab.sim.sim_data import TorchArray

    env, robot, actor_obs, stream, _ = runtime
    qfrc = torch.arange(8 * 14, dtype=torch.float32).reshape(8, 14)
    qfrc_view = TorchArray(wp.array(qfrc.tolist(), dtype=wp.float32, device="cpu"))
    clock = TorchArray(wp.array([0.02 * i for i in range(8)], dtype=wp.float32, device="cpu"))
    robot.data.data.qfrc_constraint = qfrc_view
    robot.data.data.time = clock
    state = capture.capture_state(env, robot, actor_obs, stream)
    assert state["generalized_constraint_force"].shape == (8, 14)
    assert state["generalized_constraint_force"].tolist() == qfrc.tolist()
    assert state["physics_time"].tolist() == pytest.approx(clock[:].tolist())
    assert state["generalized_constraint_force"].data_ptr() != qfrc_view._tensor.data_ptr()
    assert state["physics_time"].data_ptr() != clock._tensor.data_ptr()
    qfrc_view[:].fill_(-99.)
    clock[:].fill_(-1.)
    assert state["generalized_constraint_force"].tolist() == qfrc.tolist()
    assert state["physics_time"].tolist() == pytest.approx([0.02 * i for i in range(8)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_ids_reject_noninteger_dtypes(dtype):
    values = torch.arange(14, dtype=torch.int64).to(dtype)
    with pytest.raises(ValueError, match="integer"):
        capture._validated_ids(values, length=14, label="ids")


@pytest.mark.parametrize("values", [
    [0] * 14,
    [-1, *range(1, 14)],
])
def test_ids_reject_duplicates_and_negative_indices(values):
    with pytest.raises(ValueError, match="unique nonnegative"):
        capture._validated_ids(torch.tensor(values, dtype=torch.int64), length=14, label="ids")


def test_ids_preserve_a_valid_noncontiguous_named_mapping():
    values = torch.tensor([30, 2, 18, 4, 27, 6, 22, 8, 25, 10, 20, 12, 16, 14], dtype=torch.int32)
    assert capture._validated_ids(values, length=14, label="ids") == tuple(values.tolist())


@pytest.mark.parametrize("addresses", [
    torch.arange(14, dtype=torch.float32),
    torch.tensor([0] * 14, dtype=torch.int64),
    torch.tensor([-1, *range(1, 14)], dtype=torch.int64),
])
def test_capture_rejects_invalid_source_dof_address_dtype_or_values(runtime, addresses):
    env, robot, actor_obs, stream, _ = runtime
    robot.data.indexing.joint_v_adr = addresses
    with pytest.raises(ValueError):
        capture.capture_state(env, robot, actor_obs, stream)


@pytest.mark.parametrize("field", ["actor_previous_action", "raw_previous_action"])
def test_trace_requires_both_previous_action_histories_to_match_current_raw_action(runtime, field):
    env, robot, actor_obs, stream, _ = runtime
    first = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), torch.zeros(8, 14))
    advance_one(env, robot, actor_obs, torch.zeros(8, 14))
    second = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), torch.ones(8, 14))
    second[field][:, 0] = 1.
    with pytest.raises(ValueError, match="previous-action histories|history equals"):
        capture.trace([first, second], "a" * 64)


@pytest.mark.parametrize("field", ["action_offset", "encoder_bias", "hard_limits"])
def test_trace_rejects_changed_action_transform_or_limits(runtime, field):
    env, robot, actor_obs, stream, _ = runtime
    action = torch.zeros(8, 14)
    first = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), action)
    advance_one(env, robot, actor_obs, action)
    second = capture.add_action(capture.capture_state(env, robot, actor_obs, stream), action)
    second[field][0, 0] += .001
    with pytest.raises(ValueError, match=f"constant {field}"):
        capture.trace([first, second], "a" * 64)
