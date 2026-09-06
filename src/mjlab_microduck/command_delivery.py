"""Opt-in raw actor-input refresh primitive; not wired into retained rollouts.

Use only before the actor's existing normalizer and after external command
bounds. This changes command timing, not sensor sampling or motor authority.
"""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.observations import generated_commands

from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "frozen-actor-fresh-twist-v1"
LEGACY_PROTOCOL = "frozen-actor-cached-twist-v1"
TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
         "actions", "command", "head_command", "body_command")
DIMENSIONS = ((3,), (3,), (14,), (14,), (14,), (3,), (4,), (6,))


def require_matching_delivery(training_protocol, evaluation_protocol):
    """Future artifact compatibility only; missing metadata is not guessed."""
    allowed = (LEGACY_PROTOCOL, PROTOCOL)
    require(training_protocol in allowed and evaluation_protocol in allowed,
            "explicit known training and evaluation delivery identities")
    require(training_protocol == evaluation_protocol, "training/evaluation command timing mismatch")


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


@dataclass(frozen=True)
class ActorCommandInput:
    """One inference-boundary snapshot, not evidence that physics executed."""

    observations: object
    protocol: str
    step: int
    issued: torch.Tensor
    cached: torch.Tensor
    consumed: torch.Tensor

    def report(self):
        """JSON-safe raw inputs; no measured speed or policy-admission claim."""
        error = (self.issued - self.consumed).abs().amax(dim=-1)
        return dict(protocol=self.protocol, step=self.step,
                    boundary="prepared-raw-input-before-actor-normalization",
                    issued_twist=self.issued.cpu().tolist(),
                    cached_twist=self.cached.cpu().tolist(),
                    actor_input_twist=self.consumed.cpu().tolist(),
                    issued_input_max_abs_error_per_env=error.cpu().tolist(),
                    issued_input_equal_per_env=(error == 0).cpu().tolist(),
                    actor_inference_executed=False, simulation_executed=False,
                    training_admitted=False, policy_acceptance=False,
                    physical_motion_authorized=False)


def prepare_actor_command_input(observations, command, manager, *, protocol, step):
    """Shared opt-in boundary for future evaluators and frozen-gait trainers.

No default protocol: an adopter must explicitly select fresh or cached timing
and retain that identity. All observations and trace tensors are independent
snapshots. This function neither invokes the actor nor advances an environment.
It remains unwired into the historical rollout/training/recording entrypoints.
"""
    from tensordict import TensorDictBase

    require(protocol in (PROTOCOL, LEGACY_PROTOCOL), "explicit command delivery protocol")
    require(type(step) is int and step >= 0, "nonnegative integer control step")
    require(isinstance(observations, TensorDictBase)
            and observations.batch_size == torch.Size([manager.num_envs]), "batched raw observations")
    actor = observations["actor"]
    refreshed = fresh_actor_twist(actor, command, manager)
    inputs = observations.clone()
    if protocol == PROTOCOL:
        inputs["actor"] = refreshed
    return ActorCommandInput(
        inputs, protocol, step, command.detach().clone(),
        actor[:, 48:51].detach().clone(), inputs["actor"][:, 48:51].detach().clone(),
    )
