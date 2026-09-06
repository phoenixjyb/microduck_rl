"""Opt-in raw actor-input refresh primitive; not wired into retained rollouts.

Use only before the actor's existing normalizer and after external command
bounds. This changes command timing, not sensor sampling or motor authority.
"""

import torch
from mjlab.envs.mdp.observations import generated_commands

from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "frozen-actor-fresh-twist-v1"
TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
         "actions", "command", "head_command", "body_command")
DIMENSIONS = ((3,), (3,), (14,), (14,), (14,), (3,), (4,), (6,))


def fresh_actor_twist(actor_observation, command, manager):
    """Return an independent raw 61D tensor with only its twist term refreshed.

Never recompute the observation group: that would advance delayed sensors and
resample noise inside a control step. Preserve the manager's cache unchanged.
The caller must not pass an already normalized tensor or infer safety admission.
"""
    group = manager.cfg["actor"]
    require(tuple(manager.active_terms["actor"]) == TERMS
            and tuple(manager.group_obs_term_dim["actor"]) == DIMENSIONS
            and manager.group_obs_dim["actor"] == (61,), "exact frozen actor layout")
    require(group.concatenate_terms and group.concatenate_dim == -1
            and group.history_length in (None, 0), "flat raw actor group")
    term = manager.get_term_cfg("actor", "command")
    require(term.func is generated_commands and term.params == {"command_name": "twist"}
            and term.noise is None and term.scale is None and term.clip is None
            and term.delay_min_lag == term.delay_max_lag == term.history_length == 0,
            "untransformed undelayed raw twist term")
    require(isinstance(actor_observation, torch.Tensor) and isinstance(command, torch.Tensor),
            "tensor command and observation")
    require(actor_observation.shape == (manager.num_envs, 61)
            and command.shape == (manager.num_envs, 3)
            and actor_observation.dtype == command.dtype
            and actor_observation.device == command.device
            and actor_observation.is_floating_point(), "matching actor and command tensors")
    require(bool(torch.isfinite(actor_observation).all()) and bool(torch.isfinite(command).all()),
            "finite raw inputs")
    result = actor_observation.clone()
    result[:, 48:51] = command
    return result
