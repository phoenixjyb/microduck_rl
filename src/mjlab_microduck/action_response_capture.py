"""Optional read-only aligned action/servo snapshots for the walking evaluator.

This module captures values at policy-step boundaries only. It does not run a
policy, step physics, alter action history, or make an admission decision.
"""

from __future__ import annotations

import torch

from mjlab_microduck import integrated_turn_foundation as base

PROTOCOL = "microduck-action-response-trace-v1"
STEP_DT = 0.02
WORLDS = 8
ACTOR_WIDTH = 61
ACTION_WIDTH = 14
MAX_STEPS = 240
STARTUP_STEPS = 60
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

STATE_TIMING = "pre-action state before policy/step"
ACTION_TIMING = "raw policy output after inference before step"
TARGET_TIMING = "previous applied target at pre-action boundary"
CONSTRAINT_TIMING = "last-physics-derived generalized constraint at pre-action boundary"
_MISSING = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, shape, label, *, floating=True, finite=True):
    _require(isinstance(value, torch.Tensor) and value.shape == shape, f"{label} shape")
    if floating:
        _require(value.dtype.is_floating_point, f"floating {label}")
    if finite:
        _require(bool(torch.isfinite(value).all()), f"finite {label}")
    return value


def _snapshot(values):
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def _validated_ids(value, *, length, upper_bound=None, label):
    _require(isinstance(value, torch.Tensor) and value.ndim == 1
             and value.dtype in (torch.int32, torch.int64) and value.numel() == length,
             f"integer {label} layout")
    ids = tuple(int(item) for item in value.detach().cpu().tolist())
    _require(all(item >= 0 for item in ids) and len(set(ids)) == length,
             f"unique nonnegative {label}")
    if upper_bound is not None:
        _require(all(item < upper_bound for item in ids), f"in-range {label}")
    return ids


def validate_wrapper(wrapped):
    """Reject runner-side clipping so raw output and delivered action coincide."""
    _require(getattr(wrapped, "clip_actions", _MISSING) is None,
             "RslRlVecEnvWrapper action clipping is disabled")


def _bind_action_layout(env, robot, stream):
    from mjlab.envs.mdp.actions.actions import JointPositionAction

    manager = env.action_manager
    _require(manager.active_terms == ["joint_pos"] and manager.total_action_dim == ACTION_WIDTH,
             "single 14D joint-position action term")
    term = manager.get_term("joint_pos")
    _require(type(term) is JointPositionAction, "actual JointPositionAction term")
    cfg = term.cfg
    _require(tuple(cfg.actuator_names) == (".*",) and cfg.preserve_order is False
             and type(cfg.scale) is float and cfg.scale == 1.0
             and type(cfg.offset) is float and cfg.offset == 0.0
             and cfg.use_default_offset is True and cfg.clip is None,
             "unchanged unclipped affine servo action config")
    _require(tuple(term.target_names) == JOINTS, "exact named 14-servo action column order")
    target_ids = _validated_ids(term.target_ids, length=ACTION_WIDTH,
                                upper_bound=len(robot.joint_names), label="action target ids")
    _require(tuple(stream.names) == JOINTS and tuple(stream.joint_ids) == target_ids,
             "action columns equal named motor stream columns")
    _require(tuple(robot.joint_names[i] for i in target_ids) == JOINTS,
             "action target ids resolve to expected robot joints")
    return manager, term, target_ids


def capture_state(env, robot, actor_obs, stream):
    """Clone one pre-policy state and raw previous-action snapshot."""
    _require(env.num_envs == WORLDS and env.step_dt == STEP_DT,
             "fixed eight-world 50Hz action-response capture")
    _tensor(actor_obs, (WORLDS, ACTOR_WIDTH), "actor observation")
    manager, term, target_ids = _bind_action_layout(env, robot, stream)
    manager_action = _tensor(manager.action, (WORLDS, ACTION_WIDTH), "manager current raw action")
    term_raw_action = _tensor(term.raw_action, (WORLDS, ACTION_WIDTH), "term raw action")
    actor_previous = actor_obs[:, 34:48]
    _require(torch.equal(actor_previous, manager_action) and torch.equal(term_raw_action, manager_action),
             "actor previous-action slice equals current action history")

    command = env.command_manager.get_command("twist")
    _tensor(command, (WORLDS, 3), "twist command")
    _require(torch.equal(actor_obs[:, 48:51], command), "actor command slice equals manager command")

    data = robot.data
    joint_pos = _tensor(data.joint_pos[:, target_ids], (WORLDS, ACTION_WIDTH), "named joint position")
    joint_vel = _tensor(data.joint_vel[:, target_ids], (WORLDS, ACTION_WIDTH), "named joint velocity")
    previous_target = _tensor(data.joint_pos_target[:, target_ids], (WORLDS, ACTION_WIDTH),
                               "previous applied position target")
    encoder_bias = _tensor(data.encoder_bias[:, target_ids], (WORLDS, ACTION_WIDTH), "encoder bias")
    offset = _tensor(term.offset, (WORLDS, ACTION_WIDTH), "cloned default-position offset")
    limits = _tensor(data.joint_pos_limits[:, target_ids], (WORLDS, ACTION_WIDTH, 2),
                     "named joint limits")
    _require(bool((limits[..., 1] > limits[..., 0]).all()), "ordered named joint limits")

    dof_addresses = data.indexing.joint_v_adr
    _require(isinstance(dof_addresses, torch.Tensor) and dof_addresses.ndim == 1
             and dof_addresses.dtype in (torch.int32, torch.int64)
             and dof_addresses.numel() > max(target_ids), "joint velocity address layout")
    dofs = _validated_ids(dof_addresses[list(target_ids)], length=ACTION_WIDTH,
                          label="joint velocity addresses")
    qfrc_all = data.data.qfrc_constraint
    # Warp-backed views implement slicing before exposing their Torch tensor.
    # Validate the selected result, not the outer WarpTorchArray wrapper.
    constraint = _tensor(qfrc_all[:, dofs], (WORLDS, ACTION_WIDTH),
                         "lagged generalized constraint force")

    episode_length = env.episode_length_buf
    _require(isinstance(episode_length, torch.Tensor) and episode_length.shape == (WORLDS,)
             and episode_length.dtype in (torch.int32, torch.int64)
             and bool((episode_length >= 0).all()), "nonnegative integer episode clock")
    _require(type(env.common_step_counter) is int and env.common_step_counter >= 0,
             "nonnegative integer common-step clock")
    physics_time = robot.data.data.time[:]
    _tensor(physics_time, (WORLDS,), "physics clock")
    _require(bool((physics_time >= 0).all()), "nonnegative physics clock")

    yaw = base.body_yaw_trace_sample(robot, command, actor_obs[:, 48:51])
    values = {
        **yaw,
        "actor_previous_action": actor_previous,
        "raw_previous_action": manager_action,
        "joint_position": joint_pos,
        "joint_velocity": joint_vel,
        "previous_applied_target": previous_target,
        "encoder_bias": encoder_bias,
        "action_offset": offset,
        "hard_limits": limits,
        "generalized_constraint_force": constraint,
        "episode_length": episode_length,
        "physics_time": physics_time,
    }
    # Keep the integer clocks as integer tensors through serialization.
    return _snapshot(values) | {
        "target_ids": target_ids,
        "common_step_counter": torch.tensor(env.common_step_counter, dtype=torch.int64),
    }


def add_action(sample, actions):
    """Attach the current raw actor output after inference, before env.step."""
    _require(isinstance(sample, dict) and "raw_action" not in sample,
             "unmodified pre-policy sample required")
    _tensor(actions, (WORLDS, ACTION_WIDTH), "current raw actor action")
    # No [-1, 1] restriction: this recipe has no runner or action-term clipping.
    return dict(sample) | {"raw_action": actions.detach().cpu().clone()}


def trace(samples, checkpoint_sha256):
    """Serialize a bounded list of paired pre-action/current-action samples."""
    _require(type(checkpoint_sha256) is str and len(checkpoint_sha256) == 64
             and all(c in "0123456789abcdef" for c in checkpoint_sha256), "lowercase checkpoint SHA256")
    _require(type(samples) is list and 0 < len(samples) <= MAX_STEPS
             and all(isinstance(sample, dict) and "raw_action" in sample for sample in samples),
             "bounded complete action-response samples")
    keys = set(samples[0])
    _require(all(set(sample) == keys for sample in samples), "consistent action-response sample fields")
    expected = {
        "command": (WORLDS, 3), "actor_command": (WORLDS, 3),
        "quat_wxyz": (WORLDS, 4), "angular_b": (WORLDS, 3), "angular_w": (WORLDS, 3),
        "actor_previous_action": (WORLDS, ACTION_WIDTH),
        "raw_previous_action": (WORLDS, ACTION_WIDTH),
        "raw_action": (WORLDS, ACTION_WIDTH),
        "joint_position": (WORLDS, ACTION_WIDTH), "joint_velocity": (WORLDS, ACTION_WIDTH),
        "previous_applied_target": (WORLDS, ACTION_WIDTH), "encoder_bias": (WORLDS, ACTION_WIDTH),
        "action_offset": (WORLDS, ACTION_WIDTH),
        "hard_limits": (WORLDS, ACTION_WIDTH, 2),
        "generalized_constraint_force": (WORLDS, ACTION_WIDTH),
        "episode_length": (WORLDS,), "physics_time": (WORLDS,),
    }
    _require(keys == set(expected) | {"common_step_counter", "target_ids"},
             "exact action-response sample schema")
    target_ids = samples[0]["target_ids"]
    _require(type(target_ids) is tuple and len(target_ids) == ACTION_WIDTH
             and len(set(target_ids)) == ACTION_WIDTH
             and all(type(value) is int and value >= 0 for value in target_ids),
             "ordered target id metadata")
    for sample in samples:
        for key, shape in expected.items():
            _tensor(sample[key], shape, key, floating=key != "episode_length")
        _require(sample["episode_length"].dtype in (torch.int32, torch.int64)
                 and bool((sample["episode_length"] >= 0).all()), "integer episode clocks")
        _require(sample["physics_time"].dtype.is_floating_point and bool((sample["physics_time"] >= 0).all()),
                 "nonnegative floating physics clocks")
        counter = sample["common_step_counter"]
        _require(isinstance(counter, torch.Tensor) and counter.shape == ()
                 and counter.dtype in (torch.int32, torch.int64) and int(counter.item()) >= 0,
                 "scalar nonnegative integer common-step clock")
        _require(sample["target_ids"] == target_ids, "consistent action target ids")
        _require(torch.equal(sample["command"], sample["actor_command"]), "shared body-trace command")
        _require(torch.equal(sample["actor_previous_action"], sample["raw_previous_action"]),
                 "actor and manager previous-action histories agree")
        _require(bool((sample["hard_limits"][..., 1] > sample["hard_limits"][..., 0]).all()),
                 "ordered joint limits")

    for index in range(len(samples) - 1):
        current, following = samples[index], samples[index + 1]
        _require(bool(torch.equal(following["command"], current["command"])),
                 "fixed command throughout action-response trace")
        _require(int(following["common_step_counter"].item())
                 == int(current["common_step_counter"].item()) + 1,
                 "contiguous common-step clock")
        _require(bool(torch.equal(following["episode_length"], current["episode_length"] + 1)),
                 "reset-free contiguous per-world episode clocks")
        _require(bool(torch.allclose(following["physics_time"] - current["physics_time"],
                                     torch.full_like(current["physics_time"], STEP_DT), atol=1e-6, rtol=0)),
                 "contiguous physics clock at control dt")
        _require(bool(torch.equal(following["actor_previous_action"], current["raw_action"])),
                 "next actor history equals preceding current action")
        _require(bool(torch.equal(following["raw_previous_action"], current["raw_action"])),
                 "next manager history equals preceding current action")
        for key in ("action_offset", "encoder_bias", "hard_limits"):
            _require(bool(torch.equal(following[key], current[key])),
                     f"constant {key} during reset-free trace")
        expected_target = current["raw_action"] + current["action_offset"] - current["encoder_bias"]
        _require(bool(torch.allclose(following["previous_applied_target"], expected_target,
                                     atol=1e-6, rtol=0)),
                 "next target equals preceding action transform")
    return dict(
        protocol=PROTOCOL, step_dt=STEP_DT, worlds=WORLDS, steps=len(samples),
        startup_steps=STARTUP_STEPS, simulator_qualified=False,
        checkpoint_sha256=checkpoint_sha256, joint_columns=list(JOINTS),
        target_ids=list(target_ids),
        timing=STATE_TIMING, action_timing=ACTION_TIMING, target_timing=TARGET_TIMING,
        action_history_timing=("actor_previous_action and raw_previous_action are the pre-policy "
                               "previous-step action history; first sample may follow restored/reset state"),
        constraint_timing=CONSTRAINT_TIMING,
        action_interpretation="raw unclipped actor output; magnitude alone is not saturation",
        constraint_includes_contacts_and_limits=True, hard_stop_force_isolated=False,
        policy_acceptance=False, physical_motion_authorized=False,
        **{key: torch.stack([sample[key] for sample in samples]).tolist()
           if key in ("episode_length", "common_step_counter") else
           torch.stack([sample[key] for sample in samples]).double().tolist()
           for key in (*expected, "common_step_counter")},
    )
