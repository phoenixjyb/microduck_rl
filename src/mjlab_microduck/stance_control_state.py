"""Per-world stance actions, delay and staged BAM state; no physics/training loop."""

from copy import copy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
import bam.actuator as bam_motor
import bam.mjlab as bam_bridge
import bam.dynamixel.actuator as dynamixel_motor

from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
from mjlab_microduck.stance_lesson_contract import LEG_IDS, JOINTS


def finite(value, shape, device, name):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('finite shaped '+name)


def mask_check(mask, n, device):
    if not isinstance(mask, torch.Tensor) or mask.shape != (n,) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError('boolean world mask on matching device required')


class StanceActionDelay:
    """Owned batched rate-limited targets and a three-physics-step FIFO.

    Set actions once per policy tick. Peek the delayed target before proposing
    BAM torque; advance only rows whose physical step will actually execute.
    Reset is explicit and selective. The caller owns episode/live counters.
    """

    def __init__(self, nominal, ranges, nworld):
        if type(nworld) is not int or nworld < 1:
            raise ValueError('positive world count')
        self.n = nworld; self.device = nominal.device
        finite(nominal, (14,), self.device, 'nominal')
        finite(ranges, (14, 2), self.device, 'joint ranges')
        if ranges.dtype != nominal.dtype or not (ranges[:, 1] > ranges[:, 0]).all():
            raise ValueError('ordered matching-dtype joint ranges')
        center = ranges.mean(1); half = .45*(ranges[:, 1]-ranges[:, 0])
        lo, hi = center-half, center+half
        if not ((nominal >= lo) & (nominal <= hi)).all():
            raise ValueError('nominal inside soft range')
        ids = list(LEG_IDS)
        self.lower = (lo[ids]-nominal[ids]).clamp(min=-.2)
        self.upper = (hi[ids]-nominal[ids]).clamp(max=.2)
        self.nominal = nominal.detach().clone()
        self.correction = nominal.new_zeros((nworld, 10))
        self.target = nominal.expand(nworld, 14).clone()
        self.queue = nominal.expand(nworld, 3, 14).clone()
        self.faulted = False

    def _mask(self, mask):
        if self.faulted: raise RuntimeError('faulted control state requires job closeout')
        mask_check(mask, self.n, self.device)

    @torch.no_grad()
    def set_actions(self, actions, live):
        try:
            self._mask(live); finite(actions, (self.n, 10), self.device, 'actions')
            if actions.dtype != self.nominal.dtype: raise ValueError('matching action dtype')
            desired = (.2*actions.clamp(-1, 1)).clamp(self.lower, self.upper)
            realized = (self.correction+(desired-self.correction).clamp(-.02, .02)).clamp(self.lower, self.upper)
            change = (realized-self.correction).clamp(-.02, .02)
            self.correction.copy_(torch.where(live[:, None], realized, self.correction))
            candidate = self.nominal.expand(self.n, 14).clone()
            candidate[:, list(LEG_IDS)] += self.correction
            self.target.copy_(torch.where(live[:, None], candidate, self.target))
            return self.correction.clone(), torch.where(live[:, None], change, 0.)
        except Exception:
            self.faulted = True
            raise

    def peek(self):
        if self.faulted: raise RuntimeError('faulted control state requires job closeout')
        return self.queue[:, 0].clone()

    @torch.no_grad()
    def advance(self, accepted):
        try:
            self._mask(accepted)
            shifted = torch.cat((self.queue[:, 1:], self.target[:, None]), dim=1)
            self.queue.copy_(torch.where(accepted[:, None, None], shifted, self.queue))
        except Exception:
            self.faulted = True
            raise

    @torch.no_grad()
    def reset(self, rows):
        try:
            self._mask(rows)
            self.correction[rows] = 0
            self.target[rows] = self.nominal
            self.queue[rows] = self.nominal
        except Exception:
            self.faulted = True
            raise


class BamStateCommit:
    """Stock BAM computes into owned candidate state; commit safe live rows.

    Must own this actuator exclusively. No calls to the source compute/reset or
    model-field expansion while bound. It does not write native/Warp ctrl, step
    physics, shift delay queues, or supply a termination/terminal-contact store.
    """

    def __init__(self, actuator):
        pins = ((bam_bridge, 'af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440'),
                (bam_motor, '6927d7b2341cbaca0e5d872fae4ddce6200056885310d9ee5acaba1f34e09c13'),
                (dynamixel_motor, '8d3ed39d68758981204901dd18dcce11dcb107b842067e036d2b8ac7b1411cec'))
        if version('better-actuator-models') != '1.0.1' or any(
                sha256(Path(module.__file__).read_bytes()).hexdigest() != expected for module, expected in pins):
            raise ValueError('BAM state write-set audit mismatch')
        if (type(actuator) is not FrictionDRBamActuator or tuple(actuator.target_names) != JOINTS
                or actuator.cfg.motor_name != 'xl330' or actuator.cfg.model != 'm6'
                or type(actuator._bam_model.actuator) is not dynamixel_motor.XL330Actuator
                or type(actuator._bam_model.actuator.backend) is not bam_motor.TorchBackend):
            raise ValueError('exact XL330 m6 stance adapter required')
        self.actuator = actuator; self.n = actuator._num_envs
        self.previous = actuator._prev_motor_torque
        self.device = self.previous.device
        finite(self.previous, (self.n, 14), self.device, 'motor history')
        if actuator._dt != .002 or actuator._base_kp != 200. or actuator.cfg.vin_min != 6.:
            raise ValueError('declared nominal motor timing/gain/voltage floor')
        self.nominal_parameters = {}
        for name, expected in (('vin_tensor', 7.5), ('vin_drop_gain', .1),
                               ('kp_scale', 1.), ('kd_scale', 1.), ('friction_scale', 1.)):
            parameter = getattr(actuator, name)
            finite(parameter, (self.n, 1), self.device, 'nominal '+name)
            if not (parameter == expected).all(): raise ValueError('nominal '+name+' required')
            self.nominal_parameters[name] = parameter.detach().clone()
        self.ids = actuator._dof_ids.clone()
        if self.ids.shape != (14,) or self.ids.unique().numel() != 14:
            raise ValueError('unique controlled DOFs')
        self.fields = {name: actuator._as_tensor(getattr(actuator._mjwarp_model, name))
                       for name in ('dof_frictionloss', 'dof_damping')}
        for field in self.fields.values():
            if field.ndim != 2 or field.shape[0] != self.n or not field.is_contiguous():
                raise ValueError('expanded owned per-world friction fields required')
            finite(field, field.shape, self.device, 'friction field')
            if (field < 0).any() or field.dtype != self.previous.dtype:
                raise ValueError('nonnegative matching-dtype friction fields')
            if (self.ids < 0).any() or (self.ids >= field.shape[1]).any():
                raise ValueError('controlled DOFs within model fields')
        self.voltage = self._parameter(actuator._bam_model.actuator.vin)
        self.kp = self._parameter(actuator._bam_model.actuator.kp)
        self.faulted = False

    def _parameter(self, value):
        tensor = torch.as_tensor(value, dtype=self.previous.dtype, device=self.device)
        result = tensor.expand(self.n, 1).detach().clone()
        finite(result, (self.n, 1), self.device, 'firmware parameter')
        return result

    def _validate(self, mask):
        if self.faulted: raise RuntimeError('faulted BAM state requires job closeout')
        mask_check(mask, self.n, self.device)
        if self.actuator._prev_motor_torque is not self.previous:
            raise ValueError('motor history replaced outside owned adapter')
        finite(self.previous, (self.n, 14), self.device, 'motor history')
        finite(self.voltage, (self.n, 1), self.device, 'effective voltage')
        finite(self.kp, (self.n, 1), self.device, 'effective gain')
        if self.actuator._dt != .002 or self.actuator._base_kp != 200. or self.actuator.cfg.vin_min != 6.:
            raise ValueError('nominal motor configuration changed after binding')
        for name, expected in self.nominal_parameters.items():
            current = getattr(self.actuator, name)
            finite(current, (self.n, 1), self.device, 'nominal '+name)
            if not torch.equal(current, expected): raise ValueError('nominal parameter changed: '+name)
        for name, value in self.fields.items():
            current = self.actuator._as_tensor(getattr(self.actuator._mjwarp_model, name))
            if current.data_ptr() != value.data_ptr() or current.shape != value.shape or current.stride() != value.stride():
                raise ValueError('model fields replaced outside owned adapter')
            finite(value, value.shape, self.device, name)
            if (value < 0).any(): raise ValueError('nonnegative committed friction fields')

    @torch.no_grad()
    def compute(self, command, live):
        try:
            self._validate(live)
            for name in ('position_target', 'velocity_target', 'effort_target', 'pos', 'vel'):
                finite(getattr(command, name), (self.n, 14), self.device, 'motor command '+name)
                if getattr(command, name).dtype != self.previous.dtype:
                    raise ValueError('matching motor command/history dtype')
            if command.velocity_target.any() or command.effort_target.any():
                raise ValueError('position-only stance commands')
            if not live.any():
                return dict(torque_nm=torch.zeros_like(self.previous), accepted=live.clone(), rejected=live.clone())
            staged = copy(self.actuator)
            staged._bam_model = copy(self.actuator._bam_model)
            staged._bam_model.actuator = copy(self.actuator._bam_model.actuator)
            staged._bam_model.actuator.model = staged._bam_model
            staged._prev_motor_torque = self.previous.clone()
            staged._mjwarp_model = SimpleNamespace(**{name: value.clone() for name, value in self.fields.items()})
            staged._friction_fields_checked = False
            torque = staged.compute(command)
            finite(torque, (self.n, 14), self.device, 'proposed torque')
            voltage = self._parameter(staged._bam_model.actuator.vin)
            kp = self._parameter(staged._bam_model.actuator.kp)
            for name, value in self.fields.items():
                candidate = getattr(staged._mjwarp_model, name)
                finite(candidate, value.shape, self.device, 'proposed '+name)
                if (candidate < 0).any(): raise ValueError('nonnegative proposed friction/damping')
            rejected = live & (torque.abs().amax(1) > .36)
            accepted = live & ~rejected
            # No write to original history, model friction or firmware fields
            # until every candidate has been validated. Inactive rows stay exact.
            self.previous.copy_(torch.where(accepted[:, None], torque, self.previous))
            for name, value in self.fields.items():
                value.copy_(torch.where(accepted[:, None], getattr(staged._mjwarp_model, name), value))
            self.voltage.copy_(torch.where(accepted[:, None], voltage, self.voltage))
            self.kp.copy_(torch.where(accepted[:, None], kp, self.kp))
            self.actuator._bam_model.actuator.vin = self.voltage
            self.actuator._bam_model.actuator.kp = self.kp
            return dict(torque_nm=torque.detach().clone(), accepted=accepted.clone(), rejected=rejected.clone())
        except Exception:
            self.faulted = True
            raise

    @torch.no_grad()
    def reset(self, rows):
        try:
            self._validate(rows)
            voltage = self._parameter(self.actuator.vin_tensor)
            kp = self._parameter(self.actuator._base_kp*self.actuator.kp_scale)
            self.previous[rows] = 0
            for value in self.fields.values():
                # Preserve unrelated DOFs and sibling worlds.
                value[rows[:, None] & torch.isin(torch.arange(value.shape[1], device=self.device), self.ids)[None]] = 0
            self.voltage.copy_(torch.where(rows[:, None], voltage, self.voltage))
            self.kp.copy_(torch.where(rows[:, None], kp, self.kp))
            self.actuator._bam_model.actuator.vin = self.voltage
            self.actuator._bam_model.actuator.kp = self.kp
        except Exception:
            self.faulted = True
            raise
