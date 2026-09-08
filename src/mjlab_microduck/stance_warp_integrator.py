"""Audited Euler candidate/commit boundary, not a complete stance environment.

Forward dynamics and contact work still run for the batch. Only committed
integration state is frozen for closed worlds; terminal observations/contacts
must be retained separately by the caller. No global package monkeypatching.
"""

from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import torch
import warp as wp
from mujoco_warp._src import forward, smooth
from mujoco_warp._src.types import IntegratorType


# Private Euler write-set audit: these exact implementations write only the
# following Data integration arrays (act has zero columns in our admitted plant).
AUDITED_SOURCE = {
    'forward': 'c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3',
    'smooth': '63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f',
}
STATE_FIELDS = ('qpos', 'qvel', 'time', 'qacc_warmstart')


def check_runtime():
    actual = {name: sha256(Path(module.__file__).read_bytes()).hexdigest()
              for name, module in (('forward', forward), ('smooth', smooth))}
    if version('mujoco-warp') != '3.8.1' or actual != AUDITED_SOURCE:
        raise ValueError('Euler write-set audit does not match installed runtime')
    return actual


class EulerCandidateCommit:
    """Stage stock Euler outputs and commit live rows only.

    Supports the no-activation, rigid Euler plant only. Calls are synchronized
    audit boundaries, not graph-capture/performance-ready operations. A caller
    must solve fresh forward dynamics first, mask its BAM/delay/control updates,
    and refresh derived fields after committing. This object never resets data,
    performs forward dynamics, chooses live worlds, or admits a training job.
    """

    def __init__(self, model, data):
        self.runtime_hashes = check_runtime()
        if (model.opt.integrator != IntegratorType.EULER or model.na != 0
                or model.nflex != 0 or model.ntendon != 0 or model.nmocap != 0
                or model.neq != 0):
            raise ValueError('only declared rigid no-activation Euler plant supported')
        self.model = model
        self.data = data
        self.device = data.qpos.device
        self.faulted = False
        self.arrays = {name: getattr(data, name) for name in STATE_FIELDS}
        with wp.ScopedDevice(self.device):
            self.candidate = replace(data, **{name: wp.clone(array)
                                             for name, array in self.arrays.items()})
        self.views = {name: wp.to_torch(array) for name, array in self.arrays.items()}
        self.proposals = {name: wp.to_torch(getattr(self.candidate, name))
                          for name in STATE_FIELDS}
        self.torch_device = self.views['qpos'].device
        if any(array.device != self.device or array.shape[0] != data.nworld
               for array in self.arrays.values()):
            raise ValueError('integration state requires consistent world/device layout')

    def _synchronize(self):
        # Explicitly cover both Torch and Warp streams; correctness first. A
        # graph implementation must establish its own equivalent stream ordering.
        if self.torch_device.type == 'cuda':
            torch.cuda.synchronize(self.torch_device)
        wp.synchronize_device(self.device)

    @torch.no_grad()
    def integrate(self, live):
        if self.faulted:
            raise RuntimeError('faulted Euler commit requires job closeout')
        try:
            if (live.dtype != torch.bool or live.shape != (self.data.nworld,)
                    or live.device != self.torch_device):
                raise ValueError('live mask must match world count and device')
            if any(getattr(self.data, name) is not array for name, array in self.arrays.items()):
                raise ValueError('integration arrays replaced after binding')
            self._synchronize()
            if not all(torch.isfinite(value).all() for value in self.views.values()):
                raise ValueError('nonfinite committed integration state')
            mask = live.detach().clone()
            if not mask.any():
                return  # No integrator call at all when the batch is closed.
            with wp.ScopedDevice(self.device):
                for name, array in self.arrays.items():
                    wp.copy(getattr(self.candidate, name), array)
                forward.euler(self.model, self.candidate)
            self._synchronize()
            # Validate all candidates before committing any field. Invalid
            # inactive proposals are also job failures, never silently ignored.
            if not all(torch.isfinite(value).all() for value in self.proposals.values()):
                raise ValueError('nonfinite Euler candidate; nothing committed')
            for name, dest in self.views.items():
                rows = mask.reshape((self.data.nworld,) + (1,)*(dest.ndim-1))
                dest.copy_(torch.where(rows, self.proposals[name], dest))
            self._synchronize()
        except Exception:
            self.faulted = True
            raise
