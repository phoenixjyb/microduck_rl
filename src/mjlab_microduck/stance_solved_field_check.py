"""Standalone finite-check candidate; not installed into the stance runtime.

Both checkers read caller-owned quiescent, same-device float32 tensors. Packing
is fresh on every call. It performs no physics/motor operation and changes no
input. This module deliberately supplies no runtime switch or training gate.
"""

from __future__ import annotations

from collections.abc import Mapping


FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl",
          "qfrc_bias", "qfrc_constraint", "qfrc_actuator", "cvel", "xquat")
MAX_BYTES = 64 * 1024 * 1024


def _inputs(values):
    import torch

    if not isinstance(values, Mapping) or tuple(values) != FIELDS:
        raise ValueError("exact ordered solved-field inventory required")
    tensors = tuple(values[name] for name in FIELDS)
    if not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
               and value.layout == torch.strided for value in tensors):
        raise ValueError("strided float32 solved fields required")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("one solved-field device required")
    if sum(value.numel() * value.element_size() for value in tensors) > MAX_BYTES:
        raise ValueError("bounded solved-field bytes required")
    return tensors


def _legacy(tensors):
    import torch

    for name, value in zip(FIELDS, tensors):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite solved stance field: " + name)


def legacy_check(values):
    """The existing ordered finite predicate, within the declared input scope."""
    _legacy(_inputs(values))


def packed_check(values):
    """One fresh aggregate predicate, preserving the original first error.

    Concatenation copies values without casting or arithmetic. A failing check
    replays the original field order. Allocation/backend errors propagate; they
    cannot turn into a pass. Concurrent mutation is unsupported in both paths.
    """
    import torch

    tensors = _inputs(values)
    packed = torch.cat(tuple(value.reshape(-1) for value in tensors))
    if not torch.isfinite(packed).all():
        _legacy(tensors)
        # The aggregate cannot fail while all unchanged inputs pass. Do not
        # silently accept an inconsistent/backend or concurrent-mutation case.
        raise RuntimeError("inconsistent packed solved-field result")
