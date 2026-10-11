"""Read-only pre-action gait/head channels for a bounded flat-ground audit."""

import torch

from mjlab_microduck import integrated_turn_foundation as base

CONTACTS = ("left_foot_collision", "right_foot_collision")
SITES = ("left_foot", "right_foot")
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
PROTOCOL = "microduck-gait-yaw-trace-v1"


def capture(env, robot, actor_obs):
    """Clone named channels without stepping sensors, physics, policy or RNG."""
    worlds = env.num_envs
    base.require(worlds == 8 and env.step_dt == .02
                 and isinstance(actor_obs, torch.Tensor) and actor_obs.shape == (worlds, 61)
                 and base.finite_tree(actor_obs), "fixed eight-world gait audit input")
    sensor = env.scene.sensors["feet_ground_contact"]
    cfg = sensor.cfg
    # Pin the compiled sensor columns, not an assumed geometry index. This
    # inspection is intentionally tied to the retained mjlab sensor runtime.
    slots = [(slot.primary_name, slot.field_name) for slot in sensor._slots]
    base.require(slots == [(name, field) for name in CONTACTS for field in ("found", "force")]
                 and cfg.name == "feet_ground_contact" and cfg.num_slots == 1
                 and cfg.fields == ("found", "force") and cfg.reduce == "netforce"
                 and cfg.track_air_time and cfg.history_length == 0
                 and cfg.primary.entity == "robot" and cfg.primary.mode == "geom"
                 and cfg.secondary.entity is None and cfg.secondary.mode == "body"
                 and cfg.secondary.pattern == "terrain",
                 "known compiled left/right flat-ground contact sensor")
    contact = sensor.data  # cached extraction only; never sensor.update/reset
    site_ids, site_names = robot.find_sites(SITES, preserve_order=True)
    joint_ids, joint_names = robot.find_joints(HEAD_JOINTS, preserve_order=True)
    base.require(tuple(site_names) == SITES and len(set(site_ids)) == 2
                 and tuple(joint_names) == HEAD_JOINTS and len(set(joint_ids)) == 4,
                 "named foot sites and actuated head joint columns")
    command = env.command_manager.get_command("twist")
    head = env.command_manager.get_command("head_pose")
    values = {
        **base.body_yaw_trace_sample(robot, command, actor_obs[:, 48:51]),
        "episode_length": env.episode_length_buf,
        "physics_time": robot.data.data.time,
        "contact_found": contact.found,
        "contact_force_w": contact.force,
        "current_air_time": contact.current_air_time,
        "current_contact_time": contact.current_contact_time,
        "foot_velocity_w": robot.data.site_lin_vel_w[:, site_ids],
        "head_position_offset": robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids],
        "head_velocity": robot.data.joint_vel[:, joint_ids],
        "head_command": head,
        "actor_head_command": actor_obs[:, 51:55],
    }
    shapes = {"command": (worlds, 3), "actor_command": (worlds, 3),
              "quat_wxyz": (worlds, 4), "angular_b": (worlds, 3), "angular_w": (worlds, 3),
              "episode_length": (worlds,), "physics_time": (worlds,),
              "contact_found": (worlds, 2), "contact_force_w": (worlds, 2, 3),
              "current_air_time": (worlds, 2), "current_contact_time": (worlds, 2),
              "foot_velocity_w": (worlds, 2, 3), "head_position_offset": (worlds, 4),
              "head_velocity": (worlds, 4), "head_command": (worlds, 4),
              "actor_head_command": (worlds, 4)}
    base.require(all(isinstance(value, torch.Tensor) and value.shape == shapes[key]
                     for key, value in values.items()) and base.finite_tree(values),
                 "finite pre-action gait channel layouts")
    base.require(values["episode_length"].dtype in (torch.int32, torch.int64)
                 and bool((values["episode_length"] >= 0).all())
                 and all(bool((values[key] >= 0).all()) for key in
                         ("physics_time", "contact_found", "current_air_time", "current_contact_time"))
                 and torch.equal(values["contact_found"], values["contact_found"].round())
                 and torch.equal(head, actor_obs[:, 51:55]),
                 "nonnegative contact clocks and exact actor head command")
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def trace(samples, checkpoint_sha256):
    """Retain integer reset clocks and explicit timing/frame, without admission."""
    base.require(0 < len(samples) <= 240 and all(set(x) == set(samples[0]) for x in samples),
                 "bounded consistent gait samples")
    return dict(protocol=PROTOCOL, step_dt=.02, startup_steps=60, steps=len(samples), worlds=8,
                checkpoint_sha256=checkpoint_sha256, timing="pre-action state before policy/step",
                contact_columns=list(CONTACTS), foot_sites=list(SITES), head_joints=list(HEAD_JOINTS),
                force_timing="last-physics-derived-contact-at-pre-action", force_frame="world",
                terrain="stationary-flat", policy_acceptance=False, physical_motion_authorized=False,
                **{key: torch.stack([sample[key] for sample in samples]).tolist()
                   if key == "episode_length" else
                   torch.stack([sample[key] for sample in samples]).double().tolist()
                   for key in samples[0]})
