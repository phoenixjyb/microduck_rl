"""Opt-in captured forward only; no changes to integration or physical stops."""
from dataclasses import fields, is_dataclass
from enum import Enum
import ctypes
import numpy as np

import mujoco_warp as mjwarp
import warp as wp
from mjlab.sim.sim import _suspend_gc


def binding(value):
    """Bind every model/data array and static value, not its changing contents."""
    if isinstance(value, wp.array):
        return ('array', id(value), value.ptr, str(value.dtype), str(value.device), value.shape, value.strides)
    if is_dataclass(value):
        return (type(value), tuple((f.name, binding(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, Enum): return (type(value), value.value)
    if isinstance(value, np.generic): return (type(value), value.item())
    if isinstance(value, np.ndarray): return ('numpy', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, ctypes.Array): return (type(value), tuple(binding(v) for v in value))
    if type(value) in (str, int, float, bool, type(None)): return (type(value), value)
    if isinstance(value, (list, tuple)): return (type(value), tuple(binding(v) for v in value))
    if isinstance(value, dict): return ('dict', tuple((k, binding(v)) for k, v in sorted(value.items())))
    raise ValueError('unsupported captured binding: '+str(type(value)))


class ForwardGraph:
    def __init__(self, model, data, device):
        if not device.is_cuda or not wp.is_mempool_enabled(device) or not wp.is_conditional_graph_supported():
            raise ValueError('forward graph requires CUDA memory pool and conditional graph support')
        self.model = model; self.data = data; self.device = device
        before = binding((model, data))
        # Match the pinned mjlab capture lifetime discipline. Captured forward
        # includes collisions/solver; nothing is replaced by approximate dynamics.
        with _suspend_gc(), wp.ScopedDevice(device):
            with wp.ScopedCapture() as capture:
                mjwarp.forward(model, data)
        self.graph = capture.graph
        if before != binding((model, data)):
            raise ValueError('forward capture changed model/data bindings')
        self.signature = before

    def run(self, model, data):
        if model is not self.model or data is not self.data or binding((model, data)) != self.signature:
            raise ValueError('captured forward array or static configuration replaced')
        with wp.ScopedDevice(self.device): wp.capture_launch(self.graph)
