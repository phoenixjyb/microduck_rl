"""Bounded CPU recurrence for supplied bytes, never entry-provenance admission."""

from hashlib import sha256
import re

import numpy as np

from mjlab_microduck import stance_forward_reduction_plan as planner

PROTOCOL = "football-b1d-forward-entry-reference-v1"
WORLDS, BODIES = 64, 16
MAX_SNAPSHOT_BYTES = 24576
_COMPONENTS = {"subtree_com": 3, "rne_backward": 6}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def reference(raw, raw_sha256, topology, topology_sha256, reduction):
    """Return a detached float32 candidate; a byte hash cannot prove its stage."""
    _need(
        type(reduction) is str
        and 0 < len(reduction) <= 32
        and reduction in _COMPONENTS,
        "fixed entry-reference reduction",
    )
    components = _COMPONENTS[reduction]
    byte_count = WORLDS * BODIES * components * 4
    _need(
        type(raw) is bytes and len(raw) == byte_count <= MAX_SNAPSHOT_BYTES,
        "plain exact-sized entry-candidate bytes",
    )
    _need(
        type(raw_sha256) is str
        and len(raw_sha256) == 64
        and _SHA256.fullmatch(raw_sha256) is not None,
        "lowercase input SHA-256",
    )
    _need(sha256(raw).hexdigest() == raw_sha256, "whole input bytes authenticated")
    declared_plan = planner.plan(topology, topology_sha256, reduction)
    shape = (WORLDS, BODIES, components)
    expected = np.frombuffer(raw, dtype="<f4").reshape(shape).copy(order="C")
    _need(bool(np.isfinite(expected).all()), "finite entry-candidate float32 input")
    with np.errstate(over="ignore", invalid="ignore"):
        for group in declared_plan["launch_groups"]:
            for body in group["active_body_ids"]:
                parent = topology["body_parentid"][body]
                np.add(
                    expected[:, parent, :],
                    expected[:, body, :],
                    out=expected[:, parent, :],
                )
                # Do not conceal an intermediate overflow with a later cancel.
                _need(
                    bool(np.isfinite(expected[:, parent, :]).all()),
                    "finite intermediate entry-reference prediction",
                )
    expected_raw = expected.tobytes(order="C")
    return {
        "protocol": PROTOCOL,
        "expected_raw": expected_raw,
        "metadata": {
            "reduction": reduction,
            "input_role": "caller-supplied-unqualified-entry-candidate",
            "input_sha256": raw_sha256,
            "expected_sha256": sha256(expected_raw).hexdigest(),
            "bytes_per_snapshot": byte_count,
            "shape": list(shape),
            "dtype": "little-endian-float32",
            "recurrence": "float32-in-place-child-to-parent-nine-serial-groups",
            "whole_input_bytes_bound": True,
        },
        "plan": declared_plan,
        "flags": {
            "actual_launch_inputs_captured": False,
            "input_stage_provenance_authenticated": False,
            "fresh_installed_source_authenticated": False,
            "actual_kernel_launch_performed": False,
            "actual_kernel_order_observed": False,
            "cause_proven": False,
            "original_pair_accepted": False,
            "full_window_qualified": False,
            "training_authorized": False,
            "physical_result_accepted": False,
        },
    }
