"""Bounded native CPU B1 runtime, not a GPU environment or training entrypoint."""

from copy import deepcopy

import mujoco
import numpy as np
import torch

from mjlab_microduck import football_bam_probe as bam
from mjlab_microduck import football_flat_hold as hold
from mjlab_microduck import stance_lesson_contract as lesson
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_transition import PhysicsState, StanceTransition, DECIMATION


def physics_state(frame):
    """Exact rigid-free-root frame mapping; all fields owned on CPU."""
    def floating(value): return torch.tensor([value], dtype=torch.float64, device='cpu')
    def boolean(value): return torch.tensor([bool(value)], device='cpu')
    return PhysicsState(floating(frame['tilt_rad']), floating(frame['qvel'][:3]),
        floating(frame['root_height_m']),
        floating([frame['support_normal_n'][n] for n in hold.FEET]),
        floating(frame['applied_torque_nm']), floating(frame['hinge_speed_rad_s']),
        boolean(frame['outside_hard_limit']), boolean(frame['forbidden_contacts']),
        boolean(frame['warning_count']))


class NativeStanceRuntime:
    """One CPU-only audit world, explicit reset and at most 2500 physics steps.

    This is not registered as a mjlab/PPO environment. No automatic episode reset,
    rollout loop, checkpoint loading, randomization, GPU or physical-robot API.
    """

    def __init__(self):
        self.model, self.data, self.actuator = bam.build_component_fixture(flat_floor=True)
        require(tuple(self.actuator.target_names) == lesson.JOINTS, 'exact stance joint order')
        root = self.model.joint('trunk_base_freejoint')
        require(int(root.qposadr[0]) == 0 and int(root.dofadr[0]) == 0,
                'declared rigid free-root addresses')
        self.root_body_id = int(self.model.jnt_bodyid[root.id])
        self.joint_ids = np.array([self.model.joint(n).id for n in lesson.JOINTS])
        self.dofs = self.model.jnt_dofadr[self.joint_ids].copy()
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids].copy()
        self.control_ids = []
        for j in self.joint_ids:
            matches = np.flatnonzero((self.model.actuator_trnid[:, 0] == j)
                & (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            require(len(matches) == 1, 'unique stance motor control')
            self.control_ids.append(int(matches[0]))
        require(self.model.opt.timestep == .002 and self.model.neq == 0 and self.model.nmocap == 0,
                'unassisted declared CPU plant')
        hold.place_on_floor(self.model, self.data)
        self.initial_qpos = self.data.qpos.copy()
        self.nominal = self.initial_qpos[self.qpos_ids].copy()
        self.delay = hold.FixedDelay(self.nominal)
        self.faulted = False
        self.reset()

    def reset(self):
        if self.faulted:
            raise RuntimeError('faulted audit runtime requires job closeout, not episode reset')
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.model.dof_frictionloss[self.dofs] = 0.
        self.model.dof_damping[self.dofs] = 0.
        self.actuator.reset(); self.delay.reset()
        self.actuator._mjwarp_model.dof_frictionloss.zero_()
        self.actuator._mjwarp_model.dof_damping.zero_()
        self.correction = np.zeros(10)
        self.steps = 0; self.done = False
        self.terminated = False; self.timed_out = False
        self.commands = []
        mujoco.mj_forward(self.model, self.data)
        frame = hold.observe(self.model, self.data, self.actuator)
        require(not hold.stop_reasons(frame), 'valid stance reset')
        self.frames = [self._annotate(frame)]
        return self.observations()

    def _annotate(self, frame):
        q = np.asarray(frame['qpos'])[self.qpos_ids]
        ranges = self.model.jnt_range[self.joint_ids]
        soft = np.abs(q-ranges.mean(1)) > .45*(ranges[:, 1]-ranges[:, 0])
        return dict(physics_step=self.steps, soft_limit_mask=soft.tolist(), **frame)

    def observations(self):
        """No simulation calls: terminal states stay untouched until explicit reset."""
        frame = self.frames[-1]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity_body = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
        local_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.root_body_id, local_velocity, 1)
        actor = lesson.actor_observation(gravity_body, local_velocity[:3],
            self.data.qpos[self.qpos_ids]-self.nominal, self.data.qvel[self.dofs], self.correction)
        critic = lesson.critic_observation(actor, self.data.qvel[:3], frame['root_height_m'],
                                           [frame['support_normal_n'][n] for n in hold.FEET])
        return dict(actor=actor, critic=critic)

    def step(self, action):
        if self.faulted or self.done:
            raise RuntimeError('closed stance world cannot advance before an explicit reset')
        try:
            target, realized = lesson.limited_targets(action, self.correction, self.nominal,
                                                      self.model.jnt_range[self.joint_ids])
            change = realized-self.correction
            # Reference limiter arithmetic can produce an ulp beyond .02; pin only
            # that arithmetic representation, never relax motor or physical gates.
            require(np.abs(change).max() <= .02+1e-14, 'declared target slew')
            tick = StanceTransition(torch.tensor([self.steps], dtype=torch.long),
                torch.tensor(realized[None], dtype=torch.float64),
                torch.tensor(np.clip(change, -.02, .02)[None], dtype=torch.float64))
            self.correction = realized.copy()
            for _ in range(DECIMATION):
                # reset() and the preceding post-step refresh already ran forward;
                # do not solve the same boundary twice or mutate terminal evidence.
                frame = hold.observe(self.model, self.data, self.actuator)
                # Preserve the refreshed current boundary without adding a duplicate.
                self.frames[-1] = self._annotate(frame)
                tick.reject_pre_step_state(physics_state(frame))
                if not bool(tick.live[0]): break
                delayed_target = self.delay.push(target)
                proposal = bam.compute_snapshot(self.model, self.data, self.actuator, delayed_target)
                command = dict(physics_step=self.steps, delayed_target_rad=delayed_target.tolist(),
                               applied=False, **proposal)
                self.commands.append(command)
                tick.reject_proposed_torque(torch.tensor([proposal['torque_nm']], dtype=torch.float64))
                if not bool(tick.live[0]): break
                self.data.ctrl[self.control_ids] = proposal['torque_nm']
                self.model.dof_frictionloss[self.dofs] = proposal['frictionloss_nm']
                self.model.dof_damping[self.dofs] = proposal['damping_nm_s_per_rad']
                mujoco.mj_step(self.model, self.data)
                self.steps += 1; command['applied'] = True
                mujoco.mj_forward(self.model, self.data)
                frame = hold.observe(self.model, self.data, self.actuator)
                self.frames.append(self._annotate(frame))
                tick.advance(physics_state(frame))
                if not bool(tick.live[0]): break
            result = tick.result()
            require(int(result['episode_steps'][0]) == self.steps, 'physical/accounting step agreement')
            self.terminated = bool(result['terminated'][0])
            self.timed_out = bool(result['timed_out'][0])
            self.done = self.terminated or self.timed_out
            return dict(observation=self.observations(), reward=float(result['reward'][0]),
                terminated=self.terminated, timed_out=self.timed_out,
                executed_physics_steps=int(result['executed_steps'][0]),
                terminal_frame=deepcopy(self.frames[-1]) if self.done else None)
        except Exception:
            self.faulted = True
            raise

    def trace(self):
        return deepcopy(dict(backend='native-cpu-audit', physics_steps=self.steps,
            frames=self.frames, commands=self.commands, faulted=self.faulted,
            terminated=self.terminated, timed_out=self.timed_out,
            policy_training=False, gpu_parity=False, learned_stance_accepted=False))
