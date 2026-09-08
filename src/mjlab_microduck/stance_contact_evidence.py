"""Owned first-terminal contact evidence, not a stance environment or admission."""

from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import torch
import warp as wp
from mujoco_warp._src import support
from mujoco_warp._src.types import ConeType

from mjlab_microduck.stance_control_state import mask_check
from mjlab_microduck.stance_transition import EPISODE_STEPS, physical_failures

SUPPORT_HASH = '3ac8475d5e41318d8289601ad41493245d95d2138023d47e0462f04b6c67b94a'


def validate_contacts(table, nworld, device=None):
    worlds = table['worldid']; count = len(worlds)
    device = worlds.device if device is None else device
    for name, shape in (('worldid', (count,)), ('geom', (count, 2)), ('dim', (count,))):
        value = table[name]
        if value.shape != shape or value.device != device or value.dtype not in (torch.int32, torch.int64):
            raise ValueError('integer contact layout: '+name)
    for name, shape in (('dist', (count,)), ('pos', (count, 3)), ('frame', (count, 3, 3)),
                        ('friction', (count, 5)), ('force', (count, 6))):
        value = table[name]
        if value.shape != shape or value.device != device or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('finite contact layout: '+name)
    address = table['efc_address']
    if (address.ndim != 2 or address.shape[0] != count or address.shape[1] < 1
            or address.device != device or address.dtype not in (torch.int32, torch.int64)):
        raise ValueError('integer contact constraint addresses required')
    if ((worlds < 0) | (worlds >= nworld)).any() or (table['geom'] < 0).any():
        raise ValueError('rigid contact world/geom outside allowed range')
    if (table['friction'] < 0).any(): raise ValueError('nonnegative contact friction')


def _synchronize(device):
    if device.is_cuda: torch.cuda.synchronize(device.alias)
    wp.synchronize_device(device)


def read_contacts(model, data):
    """Read only the active prefix after a caller-owned fresh forward solve.

    Does not run forward or integrate. Returns owned tensors, including decoded
    force:torque in contact coordinates. Never exposes a mutable global slot as
    durable contact identity. Capacity/row errors fail before the decoding kernel.
    """
    if (version('mujoco-warp') != '3.8.1'
            or sha256(Path(support.__file__).read_bytes()).hexdigest() != SUPPORT_HASH):
        raise ValueError('contact decoder source audit mismatch')
    device = data.qpos.device
    _synchronize(device)
    count = int(wp.to_torch(data.nacon)[0])
    collision_count = int(wp.to_torch(data.ncollision)[0])
    nefc = wp.to_torch(data.nefc)
    if (not 0 <= count <= data.naconmax or not 0 <= collision_count <= data.naconmax
            or ((nefc < 0) | (nefc > data.njmax)).any()):
        raise ValueError('contact/constraint capacity overflow or invalid count')
    table = {name: wp.to_torch(getattr(data.contact, name))[:count].detach().clone()
             for name in ('worldid', 'geom', 'dist', 'pos', 'frame', 'friction', 'dim', 'efc_address')}
    worlds = table['worldid'].long(); dim = table['dim']; address = table['efc_address']
    if ((worlds < 0) | (worlds >= data.nworld)).any():
        raise ValueError('contact world ID outside batch')
    if ((table['geom'] < 0) | (table['geom'] >= model.ngeom)).any():
        raise ValueError('rigid contact geom IDs required')
    if not torch.isin(dim, torch.tensor([1, 3, 4, 6], device=worlds.device)).all():
        raise ValueError('supported contact dimensionality required')
    for name in ('dist', 'pos', 'frame', 'friction'):
        if not torch.isfinite(table[name]).all(): raise ValueError('nonfinite contact '+name)
    if (table['friction'] < 0).any(): raise ValueError('nonnegative contact friction')
    # Friction-only active constraints also matter when there are no contacts.
    efc_force = wp.to_torch(data.efc.force)
    active = torch.arange(efc_force.shape[1], device=worlds.device) < nefc[:, None]
    if not torch.isfinite(efc_force[active]).all(): raise ValueError('nonfinite active constraint force')
    if count:
        first = address[:, 0]; included = first >= 0
        if (first < -1).any(): raise ValueError('invalid excluded contact address')
        if model.opt.cone == ConeType.PYRAMIDAL:
            rows = torch.where(dim == 1, 1, 2*(dim-1))
            if (included & (first+rows > nefc[worlds])).any():
                raise ValueError('pyramid rows outside active constraints')
        elif model.opt.cone == ConeType.ELLIPTIC:
            if address.shape[1] < int(dim.max()): raise ValueError('short elliptic address layout')
            active = included[:, None] & (torch.arange(address.shape[1], device=worlds.device) < dim[:, None])
            if (active & ((address < 0) | (address >= nefc[worlds, None]))).any():
                raise ValueError('elliptic rows outside active constraints')
        else:
            raise ValueError('unsupported contact cone')
        ids = torch.arange(count, dtype=torch.int32, device=worlds.device)
        with wp.ScopedDevice(device):
            forces = wp.zeros(count, dtype=wp.spatial_vector)
            support.contact_force(model, data, wp.from_torch(ids), False, forces)
        _synchronize(device)
        table['force'] = wp.to_torch(forces).detach().clone()
    else:
        table['force'] = table['pos'].new_zeros((0, 6))
    if not torch.isfinite(table['force']).all(): raise ValueError('nonfinite decoded contact force')
    validate_contacts(table, data.nworld)
    return table


def contact_summary(table, nworld, *, floor_id, foot_ids):
    """Native-hold semantics: foot normal load and non-foot penetrating pairs."""
    if len(foot_ids) != 2 or len({floor_id, *foot_ids}) != 3:
        raise ValueError('distinct floor and two foot geometries required')
    validate_contacts(table, nworld)
    worlds = table['worldid'].long(); pair = table['geom']
    if ((worlds < 0) | (worlds >= nworld)).any(): raise ValueError('contact world outside batch')
    normal = table['force'][:, 0].clamp(min=0)
    result = normal.new_zeros((nworld, 2)); allowed = torch.zeros_like(worlds, dtype=torch.bool)
    for index, foot in enumerate(foot_ids):
        matches = ((pair[:, 0] == floor_id) & (pair[:, 1] == foot)) | ((pair[:, 1] == floor_id) & (pair[:, 0] == foot))
        result[:, index].index_add_(0, worlds, normal*matches)
        allowed |= matches
    counts = torch.zeros(nworld, dtype=torch.long, device=worlds.device)
    counts.index_add_(0, worlds, ((table['dist'] <= 0) & ~allowed).long())
    return result, counts > 0


class FirstTerminalContacts:
    """One immutable first-attempt terminal record per world, with no reset API.

    This ledger does not prove the preceding trajectory was continuous or bind
    policy/source hashes. The full evaluator must supply those independent gates.
    """

    def __init__(self, nworld):
        if type(nworld) is not int or nworld < 1: raise ValueError('positive world count')
        self.n = nworld
        self._records = [None]*nworld

    @torch.no_grad()
    def capture(self, *, state, steps, qpos, qvel, contacts, terminated, timed_out, proposed_torque=None):
        device = steps.device
        if steps.shape != (self.n,) or steps.dtype != torch.long or ((steps < 0) | (steps > EPISODE_STEPS)).any():
            raise ValueError('bounded per-world physics counters required')
        state.validate(self.n, device)
        for value, shape in ((qpos, (self.n, 21)), (qvel, (self.n, 20))):
            if value.shape != shape or value.device != device or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('finite rigid stance state required')
        mask_check(terminated, self.n, device); mask_check(timed_out, self.n, device)
        if (terminated & timed_out).any(): raise ValueError('failure and timeout are distinct')
        excess = torch.zeros(self.n, dtype=torch.bool, device=device)
        if proposed_torque is not None:
            if (proposed_torque.shape != (self.n, 14) or proposed_torque.device != device
                    or not proposed_torque.is_floating_point() or not torch.isfinite(proposed_torque).all()):
                raise ValueError('finite proposed torque evidence required')
            excess = proposed_torque.abs().amax(1) > .36
        failure = physical_failures(state, steps) | excess
        if (terminated & ~failure).any() or (timed_out & (failure | (steps != EPISODE_STEPS))).any():
            raise ValueError('terminal flags require actual failure or exact timeout evidence')
        selected = (terminated | timed_out).nonzero().flatten().tolist()
        if any(self._records[i] is not None for i in selected):
            raise ValueError('first terminal record cannot be overwritten')
        worlds = contacts['worldid']
        validate_contacts(contacts, self.n, device)
        if ((worlds < 0) | (worlds >= self.n)).any(): raise ValueError('contact world outside ledger')
        # Construct all records before storing any: a malformed later row must
        # not leave a partly updated first-terminal ledger.
        pending = {}
        for i in selected:
            rows = worlds == i
            owned_contacts = {name: value[rows].detach().cpu().tolist() for name, value in contacts.items()}
            pending[i] = dict(world_id=i, physics_step=int(steps[i]),
                terminated=bool(terminated[i]), timed_out=bool(timed_out[i]),
                qpos=qpos[i].detach().cpu().tolist(), qvel=qvel[i].detach().cpu().tolist(),
                state={f.name: getattr(state, f.name)[i].detach().cpu().tolist() for f in fields(state)},
                contacts=owned_contacts,
                rejected_proposed_torque_nm=proposed_torque[i].detach().cpu().tolist() if bool(excess[i]) else None,
                trajectory_continuity_validated=False, checkpoint_admitted=False)
        for i, record in pending.items(): self._records[i] = record

    def records(self):
        return deepcopy(self._records)
