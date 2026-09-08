"""Batched B1-N substep rewards/stops, independent of simulator integration.

The caller must supply refreshed post-step state and skip/freeze closed worlds.
This module does not step MuJoCo, reset worlds or admit a training launch.
"""

from dataclasses import dataclass

import torch

DT = .002
DECIMATION = 10
EPISODE_STEPS = 2500


@dataclass
class PhysicsState:
    tilt: torch.Tensor
    root_velocity: torch.Tensor
    height: torch.Tensor
    support: torch.Tensor
    torque: torch.Tensor
    joint_velocity: torch.Tensor
    hard_limit: torch.Tensor
    forbidden_contact: torch.Tensor
    warning: torch.Tensor

    def validate(self, count, device):
        for key, shape in (
            ('tilt', (count,)), ('root_velocity', (count, 3)), ('height', (count,)),
            ('support', (count, 2)), ('torque', (count, 14)), ('joint_velocity', (count, 14)),
        ):
            value = getattr(self, key)
            if value.shape != shape or value.device != device or not value.is_floating_point():
                raise ValueError('state shape/device/dtype: '+key)
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite physics state: '+key)
        if (self.tilt < 0).any() or (self.support < 0).any():
            raise ValueError('nonnegative tilt and normal support required')
        for key in ('hard_limit', 'forbidden_contact', 'warning'):
            value = getattr(self, key)
            if value.shape != (count,) or value.dtype != torch.bool or value.device != device:
                raise ValueError('state boolean shape/device: '+key)
        if self.warning.any():
            raise ValueError('MuJoCo warning: stop the job, not just an episode')


def dense_terms(state, correction, correction_change):
    """Dimensionless declared terms; dt and the one-time failure cost are separate."""
    return dict(
        upright=2*torch.exp(-(state.tilt/.10).square()),
        stillness=torch.exp(-(torch.linalg.vector_norm(state.root_velocity[:, :2], dim=1)/.03).square()),
        height=torch.exp(-((state.height-.12)/.015).square()),
        support=.5*(state.support > .01).to(state.tilt.dtype).mean(dim=1),
        motor=-.2*(state.torque/.36).square().mean(dim=1),
        joint_speed=-.05*(state.joint_velocity/10).square().mean(dim=1),
        correction=-.05*(correction/.2).square().mean(dim=1),
        correction_change=-.1*(correction_change/.02).square().mean(dim=1),
    )


def physical_failures(state, episode_steps):
    """Hold stops on validated states; integer steps remove grace-period drift."""
    failure = (state.tilt > .35) | (state.height < .08)
    failure |= torch.linalg.vector_norm(state.root_velocity, dim=1) > 1.
    failure |= state.torque.abs().amax(dim=1) > .36
    failure |= state.joint_velocity.abs().amax(dim=1) > 10.
    failure |= state.hard_limit | state.forbidden_contact
    failure |= (episode_steps >= 50) & (state.support.amin(dim=1) <= .01)
    return failure


class StanceTransition:
    """Owns exactly one policy tick's accounting across up to ten physics calls."""

    def __init__(self, episode_steps, correction, correction_change, *, initial_live=None):
        if episode_steps.ndim != 1 or episode_steps.dtype != torch.long:
            raise ValueError('episode counters must be one-dimensional int64')
        n = episode_steps.numel(); self.device = episode_steps.device
        if n == 0: raise ValueError('nonempty batch required')
        if initial_live is None:
            initial_live = torch.ones(n, dtype=torch.bool, device=self.device)
        if (initial_live.shape != (n,) or initial_live.dtype != torch.bool
                or initial_live.device != self.device):
            raise ValueError('initial live mask must match episode counters')
        if ((episode_steps < 0) | (episode_steps > EPISODE_STEPS)
                | (initial_live & (episode_steps == EPISODE_STEPS))).any():
            raise ValueError('episode counters require unfinished live episodes')
        self.participating = initial_live.detach().clone()
        for name, value, bound in (('correction', correction, .2),
                                   ('correction change', correction_change, .02)):
            if (value.shape != (n, 10) or value.device != self.device
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError('finite shaped '+name)
            # Compare in the tensor's own dtype (e.g. float32 representation of .2).
            if (value.abs() > bound).any(): raise ValueError('bounded '+name)
        self.episode_steps = episode_steps.clone()
        self.correction = correction.detach().clone(); self.change = correction_change.detach().clone()
        self.reward = torch.zeros(n, device=self.device, dtype=correction.dtype)
        self.terminated = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.timed_out = torch.zeros_like(self.terminated)
        self.executed_steps = torch.zeros_like(episode_steps)
        self.applied_calls = 0
        self.term_sums = {}

    @property
    def live(self):
        return self.participating & ~(self.terminated | self.timed_out)

    @torch.no_grad()
    def reject_pre_step_state(self, state):
        """Fail an already-invalid boundary without counting an unexecuted step."""
        if self.applied_calls >= DECIMATION:
            raise ValueError('closed policy tick cannot inspect another boundary')
        state.validate(self.reward.numel(), self.device)
        new_failure = self.live & physical_failures(state, self.episode_steps)
        self.reward -= 2*new_failure.to(self.reward.dtype)
        self.terminated |= new_failure
        return new_failure.clone()

    @torch.no_grad()
    def reject_proposed_torque(self, proposed):
        """Before physics: reject excessive proposed motor torque without a substep."""
        if self.applied_calls >= DECIMATION:
            raise ValueError('closed policy tick cannot receive another torque proposal')
        if (proposed.shape != (self.reward.numel(), 14) or proposed.device != self.device
                or not proposed.is_floating_point() or not torch.isfinite(proposed).all()):
            raise ValueError('finite shaped proposed torque required')
        new_failure = self.live & (proposed.abs().amax(dim=1) > .36)
        self.reward -= 2*new_failure.to(self.reward.dtype)
        self.terminated |= new_failure
        return new_failure.clone()

    @torch.no_grad()
    def advance(self, state):
        """Account one refreshed post-step snapshot of all still-live worlds.

        Includes the executed terminal substep's dense reward, then failure -2
        once. Never counts further dense rewards, failures or time for that world.
        Caller integration must obey this same live mask; bookkeeping alone does
        not prove a simulator stopped or froze a terminal world.
        """
        if self.applied_calls >= DECIMATION:
            raise ValueError('more than ten physics boundaries in one policy tick')
        state.validate(self.reward.numel(), self.device)
        live = self.live
        next_steps = self.episode_steps + live.to(torch.long)
        failure = physical_failures(state, next_steps)
        new_failure = live & failure
        terms = dense_terms(state, self.correction, self.change)
        # Validate arithmetic too: finite inputs can overflow squares in float32.
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise ValueError('nonfinite reward terms: stop the job')
        active_weight = live.to(self.reward.dtype)*DT
        for key, value in terms.items():
            weighted = value*active_weight
            self.term_sums[key] = self.term_sums.get(key, torch.zeros_like(self.reward))+weighted
            self.reward += weighted
        self.reward -= 2*new_failure.to(self.reward.dtype)
        self.terminated |= new_failure
        self.timed_out |= live & ~new_failure & (next_steps >= EPISODE_STEPS)
        self.episode_steps = next_steps
        self.executed_steps += live.to(torch.long)
        self.applied_calls += 1

    def result(self):
        """Independent tensors so a consumer cannot mutate accounting history."""
        return dict(reward=self.reward.clone(), terminated=self.terminated.clone(),
            timed_out=self.timed_out.clone(), episode_steps=self.episode_steps.clone(),
            executed_steps=self.executed_steps.clone(), live=self.live.clone(),
            term_sums={k: v.clone() for k, v in self.term_sums.items()})
