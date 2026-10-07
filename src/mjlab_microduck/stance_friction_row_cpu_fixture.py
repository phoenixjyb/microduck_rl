# Copyright 2025 The Newton Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone CPU fixture for dense MuJoCo-Warp DOF-friction row creation.

The private ascending-DOF kernel is a diagnostic fixture, not a runtime
controller or a simulator path. Its dense row construction follows
MuJoCo-Warp's Apache-2.0-licensed ``_friction_dof`` implementation.
Copyright 2025 The Newton Developers. Licensed under Apache-2.0.
"""

from hashlib import sha256
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import numpy as np
import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src.types import ConstraintType, vec5

PROTOCOL = "microduck-dense-friction-row-cpu-fixture-oct7-v1"
TREE_SHA256 = "188d58bfac6ab54e51adfc80a6c504d782aec43856cbb15e54b4109f0ce30a1d"
CONSTRAINT_SHA256 = "b69f15e5c7206b30bfe1af12b5ca6c0bdf3e37398116846643df73a2e8f8ef53"
WARP_VERSION = "1.12.0"
MUJOCO_WARP_VERSION = "3.8.1"
CPU_PLATFORMS = {
    ("Darwin", "arm64", "3.12.12"),
    ("Linux", "x86_64", "3.12.13"),
}
NV = 20
WORLD_MIN, WORLD_MAX = 1, 4
NJMAX_MIN, NJMAX_MAX = 1, 32
FLOAT_LIMIT = np.float32(1_000_000.0)
OUTPUT_CAP = 1024 * 1024

_SOURCE_PATH = Path(constraint.__file__).resolve()
_PACKAGE_ROOT = _SOURCE_PATH.parent.parent
_ORIGINAL_KERNEL = constraint._friction_dof
_ORIGINAL_KERNEL_FUNC = _ORIGINAL_KERNEL.func
_ORIGINAL_KERNEL_CODE = _ORIGINAL_KERNEL_FUNC.__code__
_ORIGINAL_EFC_ROW = constraint._efc_row
_ORIGINAL_EFC_FUNC = _ORIGINAL_EFC_ROW.func
_ORIGINAL_EFC_CODE = _ORIGINAL_EFC_FUNC.__code__
_ORIGINAL_LAUNCH = wp.launch
_ORIGINAL_LAUNCH_CODE = wp.launch.__code__
_F32 = np.dtype("<f4")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _tree_sha256():
    files = {
        str(path.relative_to(_PACKAGE_ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
    }
    _need(len(files) == 69, "complete frozen 69-file package tree")
    canonical = (
        json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return sha256(canonical.encode()).hexdigest()


def _source_binding():
    _need(
        (platform.system(), platform.machine(), platform.python_version())
        in CPU_PLATFORMS
        and importlib.metadata.version("warp-lang") == WARP_VERSION
        and importlib.metadata.version("mujoco-warp") == MUJOCO_WARP_VERSION,
        "frozen CPU Python/Warp/MuJoCo-Warp versions",
    )
    _need(
        _tree_sha256() == TREE_SHA256
        and sha256(_SOURCE_PATH.read_bytes()).hexdigest() == CONSTRAINT_SHA256,
        "exact installed MuJoCo-Warp source tree and constraint file",
    )
    _need(
        constraint._friction_dof is _ORIGINAL_KERNEL
        and constraint._friction_dof.func is _ORIGINAL_KERNEL_FUNC
        and _ORIGINAL_KERNEL_FUNC.__code__ is _ORIGINAL_KERNEL_CODE
        and constraint._efc_row is _ORIGINAL_EFC_ROW
        and _ORIGINAL_EFC_ROW.func is _ORIGINAL_EFC_FUNC
        and _ORIGINAL_EFC_FUNC.__code__ is _ORIGINAL_EFC_CODE
        and _ascending_dense_friction is _CANDIDATE_KERNEL
        and _CANDIDATE_KERNEL.func is _CANDIDATE_FUNC
        and _CANDIDATE_FUNC.__code__ is _CANDIDATE_CODE
        and wp.launch is _ORIGINAL_LAUNCH
        and wp.launch.__code__ is _ORIGINAL_LAUNCH_CODE,
        "pinned original friction kernel and row helper identity/code",
    )
    return {
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "warp_lang": importlib.metadata.version("warp-lang"),
        "mujoco_warp": importlib.metadata.version("mujoco-warp"),
        "python_tree_sha256": TREE_SHA256,
        "python_tree_files": 69,
        "constraint_sha256": CONSTRAINT_SHA256,
        "fixture_module_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "friction_kernel": {
            "name": _ORIGINAL_KERNEL.__name__,
            "code_name": _ORIGINAL_KERNEL_CODE.co_qualname,
            "first_line": _ORIGINAL_KERNEL_CODE.co_firstlineno,
            "object_id": id(_ORIGINAL_KERNEL),
            "code_id": id(_ORIGINAL_KERNEL_CODE),
        },
        "efc_row": {
            "name": _ORIGINAL_EFC_ROW.__name__,
            "code_name": _ORIGINAL_EFC_CODE.co_qualname,
            "first_line": _ORIGINAL_EFC_CODE.co_firstlineno,
            "object_id": id(_ORIGINAL_EFC_ROW),
            "code_id": id(_ORIGINAL_EFC_CODE),
        },
        "candidate": {
            "code_name": _CANDIDATE_CODE.co_qualname,
            "object_id": id(_CANDIDATE_KERNEL),
            "code_id": id(_CANDIDATE_CODE),
        },
        "launch": {
            "code_name": _ORIGINAL_LAUNCH_CODE.co_qualname,
            "object_id": id(_ORIGINAL_LAUNCH),
            "code_id": id(_ORIGINAL_LAUNCH_CODE),
        },
    }


def _validate_array(value, name, shape, *, positive=False):
    _need(type(value) is np.ndarray, name + " must be a plain NumPy array")
    _need(value.dtype == _F32 and value.shape == shape, name + " exact f32 shape")
    _need(value.flags.c_contiguous, name + " C-contiguous fixture")
    _need(bool(np.isfinite(value).all()), name + " finite full array")
    _need(bool((np.abs(value) <= FLOAT_LIMIT).all()), name + " bounded values")
    if positive:
        _need(bool((value > 0).all()), name + " strictly positive")


def _validate_fixture(frictionloss, qvel, invweight, solref, solimp, timestep, njmax):
    _need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "CUDA must be hidden for the CPU-only fixture",
    )
    _need(type(njmax) is int and NJMAX_MIN <= njmax <= NJMAX_MAX, "bounded njmax")
    _need(
        type(qvel) is np.ndarray
        and qvel.ndim == 2
        and qvel.shape[1] == NV
        and WORLD_MIN <= qvel.shape[0] <= WORLD_MAX,
        "qvel literal nworld by 20 fixture",
    )
    nworld = qvel.shape[0]
    _validate_array(qvel, "qvel", (nworld, NV))
    _need(
        type(frictionloss) is np.ndarray
        and frictionloss.ndim == 2
        and frictionloss.shape[1] == NV
        and frictionloss.shape[0] in (1, nworld),
        "frictionloss exact broadcastable fixture shape",
    )
    _validate_array(frictionloss, "frictionloss", (frictionloss.shape[0], NV))
    for name, value, tail in (
        ("invweight", invweight, (NV,)),
        ("solref", solref, (NV, 2)),
        ("solimp", solimp, (NV, 5)),
        ("timestep", timestep, ()),
    ):
        _need(
            type(value) is np.ndarray
            and value.ndim == len(tail) + 1
            and value.shape[0] in (1, nworld)
            and value.shape[1:] == tail,
            name + " exact per-world or broadcast fixture shape",
        )
    _validate_array(invweight, "invweight", (invweight.shape[0], NV), positive=True)
    _validate_array(solref, "solref", (solref.shape[0], NV, 2))
    _validate_array(solimp, "solimp", (solimp.shape[0], NV, 5))
    _validate_array(timestep, "timestep", (timestep.shape[0],), positive=True)
    return nworld, frictionloss.shape[0]


@wp.kernel(enable_backward=False)
def _ascending_dense_friction(
    nv: int,
    opt_timestep: wp.array(dtype=wp.float32),
    opt_disableflags: int,
    dof_solref: wp.array2d(dtype=wp.vec2),
    dof_solimp: wp.array2d(dtype=vec5),
    dof_frictionloss: wp.array2d(dtype=wp.float32),
    dof_invweight0: wp.array2d(dtype=wp.float32),
    qvel_in: wp.array2d(dtype=wp.float32),
    njmax_in: int,
    nf_out: wp.array(dtype=wp.int32),
    nefc_out: wp.array(dtype=wp.int32),
    efc_type_out: wp.array2d(dtype=wp.int32),
    efc_id_out: wp.array2d(dtype=wp.int32),
    efc_J_out: wp.array3d(dtype=wp.float32),
    efc_pos_out: wp.array2d(dtype=wp.float32),
    efc_margin_out: wp.array2d(dtype=wp.float32),
    efc_D_out: wp.array2d(dtype=wp.float32),
    efc_vel_out: wp.array2d(dtype=wp.float32),
    efc_aref_out: wp.array2d(dtype=wp.float32),
    efc_frictionloss_out: wp.array2d(dtype=wp.float32),
):
    worldid = wp.tid()
    frictionloss_id = worldid % dof_frictionloss.shape[0]
    invweight_id = worldid % dof_invweight0.shape[0]
    solref_id = worldid % dof_solref.shape[0]
    solimp_id = worldid % dof_solimp.shape[0]
    timestep_id = worldid % opt_timestep.shape[0]

    for dofid in range(nv):
        if dof_frictionloss[frictionloss_id, dofid] <= 0.0:
            continue

        wp.atomic_add(nf_out, worldid, 1)
        efcid = wp.atomic_add(nefc_out, worldid, 1)
        if efcid >= njmax_in:
            continue

        for index in range(nv):
            efc_J_out[worldid, efcid, index] = 0.0
        efc_J_out[worldid, efcid, dofid] = 1.0
        qvel = qvel_in[worldid, dofid]
        _ORIGINAL_EFC_ROW(
            opt_disableflags,
            worldid,
            opt_timestep[timestep_id],
            efcid,
            0.0,
            0.0,
            dof_invweight0[invweight_id, dofid],
            dof_solref[solref_id, dofid],
            dof_solimp[solimp_id, dofid],
            0.0,
            qvel,
            dof_frictionloss[frictionloss_id, dofid],
            ConstraintType.FRICTION_DOF,
            dofid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
        )


_CANDIDATE_KERNEL = _ascending_dense_friction
_CANDIDATE_FUNC = _CANDIDATE_KERNEL.func
_CANDIDATE_CODE = _CANDIDATE_FUNC.__code__


def _wp_inputs(wp, frictionloss, qvel, invweight, solref, solimp, timestep):
    return (
        wp.array(frictionloss, dtype=wp.float32, device="cpu"),
        wp.array(qvel, dtype=wp.float32, device="cpu"),
        wp.array(invweight, dtype=wp.float32, device="cpu"),
        wp.array(solref, dtype=wp.vec2, device="cpu"),
        wp.array(solimp, dtype=vec5, device="cpu"),
        wp.array(timestep, dtype=wp.float32, device="cpu"),
    )


def _copies(values):
    return tuple(np.array(value, dtype=_F32, order="C", copy=True) for value in values)


def _wp_dtype(wp, dtype):
    return wp.int32 if np.dtype(dtype).kind == "i" else wp.float32


def _fresh_outputs(wp, nworld, njmax):
    def array(shape, dtype, fill):
        return wp.array(
            np.full(shape, fill, dtype=dtype), dtype=_wp_dtype(wp, dtype), device="cpu"
        )

    return {
        "nf": array((nworld,), np.int32, 0),
        "nefc": array((nworld,), np.int32, 0),
        "type": array((nworld, njmax), np.int32, -7),
        "id": array((nworld, njmax), np.int32, -1),
        "J": array((nworld, njmax, NV), np.float32, np.nan),
        "pos": array((nworld, njmax), np.float32, np.nan),
        "margin": array((nworld, njmax), np.float32, np.nan),
        "D": array((nworld, njmax), np.float32, np.nan),
        "vel": array((nworld, njmax), np.float32, np.nan),
        "aref": array((nworld, njmax), np.float32, np.nan),
        "frictionloss": array((nworld, njmax), np.float32, np.nan),
    }


def _array_values(value):
    return value.numpy()


def _float_bits(value):
    return [int(v) for v in np.asarray(value, dtype="<f4").view("<u4").reshape(-1)]


def _capture(outputs, nworld, njmax):
    values = {name: _array_values(value) for name, value in outputs.items()}
    counts = []
    worlds = []
    for world in range(nworld):
        nf = int(values["nf"][world])
        nefc = int(values["nefc"][world])
        _need(nf == nefc and 0 <= nf <= NV, "bounded dense friction proposal counts")
        visible = min(nefc, njmax)
        rows = []
        for row_index in range(visible):
            _need(
                int(values["type"][world, row_index])
                == int(ConstraintType.FRICTION_DOF),
                "active friction row type",
            )
            dof = int(values["id"][world, row_index])
            _need(0 <= dof < NV, "active friction row DOF address")
            _need(
                all(
                    bool(np.isfinite(values[name][world, row_index]).all())
                    for name in (
                        "J",
                        "pos",
                        "margin",
                        "D",
                        "vel",
                        "aref",
                        "frictionloss",
                    )
                ),
                "finite complete active dense row fields",
            )
            rows.append(
                {
                    "row": row_index,
                    "dof": dof,
                    "type": int(values["type"][world, row_index]),
                    "J_u32": _float_bits(values["J"][world, row_index]),
                    "pos_u32": _float_bits(values["pos"][world, row_index]),
                    "margin_u32": _float_bits(values["margin"][world, row_index]),
                    "D_u32": _float_bits(values["D"][world, row_index]),
                    "vel_u32": _float_bits(values["vel"][world, row_index]),
                    "aref_u32": _float_bits(values["aref"][world, row_index]),
                    "frictionloss_u32": _float_bits(
                        values["frictionloss"][world, row_index]
                    ),
                }
            )
        counts.append({"nf": nf, "nefc": nefc, "stored_rows": visible})
        worlds.append(rows)
    return counts, worlds


_ADDRESS_FIELDS = (
    "type",
    "J_u32",
    "pos_u32",
    "margin_u32",
    "D_u32",
    "vel_u32",
    "aref_u32",
    "frictionloss_u32",
)


def _addressed_field(world_rows, field):
    return sorted(
        (row["dof"], tuple(row[field]) if type(row[field]) is list else row[field])
        for row in world_rows
    )


def run_cpu_fixture(frictionloss, qvel, invweight, solref, solimp, timestep, njmax):
    """Run original and ascending dense row constructors on fresh CPU arrays."""
    nworld, _broadcast_rows = _validate_fixture(
        frictionloss, qvel, invweight, solref, solimp, timestep, njmax
    )
    binding_before = _source_binding()
    device = wp.get_device("cpu")
    _need(str(device) == "cpu", "literal Warp CPU device")
    with wp.ScopedDevice("cpu"):
        # One caller-independent snapshot supplies both executions; no second
        # read of potentially externally owned mutable NumPy storage.
        fixture_values = _copies(
            (frictionloss, qvel, invweight, solref, solimp, timestep)
        )
        _validate_fixture(*fixture_values, njmax)
        baseline_inputs = _wp_inputs(wp, *_copies(fixture_values))
        candidate_inputs = _wp_inputs(wp, *_copies(fixture_values))
        original = _fresh_outputs(wp, nworld, njmax)
        candidate = _fresh_outputs(wp, nworld, njmax)
        frictionloss_wp, qvel_wp, invweight_wp, solref_wp, solimp_wp, timestep_wp = (
            baseline_inputs
        )
        (
            candidate_frictionloss,
            candidate_qvel,
            candidate_invweight,
            candidate_solref,
            candidate_solimp,
            candidate_timestep,
        ) = candidate_inputs
        wp.launch(
            _ORIGINAL_KERNEL,
            dim=(nworld, NV),
            inputs=[
                NV,
                timestep_wp,
                0,
                solref_wp,
                solimp_wp,
                frictionloss_wp,
                invweight_wp,
                False,
                qvel_wp,
                njmax,
                njmax * NV,
            ],
            outputs=[
                original["nf"],
                original["nefc"],
                original["type"],
                original["id"],
                wp.zeros((nworld, njmax), dtype=wp.int32, device="cpu"),
                wp.zeros((nworld, njmax), dtype=wp.int32, device="cpu"),
                wp.zeros((nworld, 1, njmax * NV), dtype=wp.int32, device="cpu"),
                original["J"],
                original["pos"],
                original["margin"],
                original["D"],
                original["vel"],
                original["aref"],
                original["frictionloss"],
                wp.zeros((nworld,), dtype=wp.int32, device="cpu"),
            ],
            device="cpu",
        )
        wp.launch(
            _ascending_dense_friction,
            dim=nworld,
            inputs=[
                NV,
                candidate_timestep,
                0,
                candidate_solref,
                candidate_solimp,
                candidate_frictionloss,
                candidate_invweight,
                candidate_qvel,
                njmax,
            ],
            outputs=[
                candidate["nf"],
                candidate["nefc"],
                candidate["type"],
                candidate["id"],
                candidate["J"],
                candidate["pos"],
                candidate["margin"],
                candidate["D"],
                candidate["vel"],
                candidate["aref"],
                candidate["frictionloss"],
            ],
            device="cpu",
        )
        original_counts, original_rows = _capture(original, nworld, njmax)
        candidate_counts, candidate_rows = _capture(candidate, nworld, njmax)
    binding_after = _source_binding()
    _need(
        binding_before == binding_after,
        "source/functions unchanged through CPU fixture",
    )

    per_world = []
    overflow = False
    all_addressed_exact = True
    all_candidate_ascending = True
    all_counts_complete = True
    for world in range(nworld):
        before_ids = [row["dof"] for row in original_rows[world]]
        after_ids = [row["dof"] for row in candidate_rows[world]]
        world_overflow = (
            original_counts[world]["nefc"] > njmax
            or candidate_counts[world]["nefc"] > njmax
        )
        overflow |= world_overflow
        candidate_ascending = after_ids == sorted(after_ids)
        all_candidate_ascending &= candidate_ascending
        losses = fixture_values[0][world % fixture_values[0].shape[0]]
        expected_dofs = np.flatnonzero(losses > 0).astype(int).tolist()
        expected_counts = {
            "nf": len(expected_dofs),
            "nefc": len(expected_dofs),
            "stored_rows": min(len(expected_dofs), njmax),
        }
        counts_complete = (
            original_counts[world] == candidate_counts[world] == expected_counts
            and len(set(before_ids)) == len(before_ids)
            and all(dof in expected_dofs for dof in before_ids)
            and after_ids == expected_dofs[:njmax]
        )
        all_counts_complete &= counts_complete
        addressed_field_exact = {
            field: _addressed_field(original_rows[world], field)
            == _addressed_field(candidate_rows[world], field)
            for field in _ADDRESS_FIELDS
        }
        addressed_exact = all(addressed_field_exact.values())
        all_addressed_exact &= addressed_exact
        per_world.append(
            {
                "world": world,
                "expected_active_dofs": expected_dofs,
                "expected_counts": expected_counts,
                "counts_and_addresses_complete": counts_complete,
                "original_counts": original_counts[world],
                "candidate_counts": candidate_counts[world],
                "original_rows": original_rows[world],
                "candidate_rows": candidate_rows[world],
                "candidate_dof_order_ascending": candidate_ascending,
                "overflow": world_overflow,
                "baseline_candidate_addressed_rows_equal": addressed_exact,
                "baseline_candidate_addressed_field_exact": addressed_field_exact,
                "baseline_candidate_address_multiset_equal": sorted(before_ids)
                == sorted(after_ids),
            }
        )
    source_bound_exact = (
        all_addressed_exact
        and all_candidate_ascending
        and all_counts_complete
        and not overflow
    )
    result = {
        "protocol": PROTOCOL,
        "device": "cpu",
        "source_binding_before": binding_before,
        "source_binding_after": binding_after,
        "launches": {"original_friction_dof": 1, "ascending_dense_candidate": 1},
        "fixture_inputs": {
            name: {
                "shape": list(value.shape),
                "dtype": "<f4",
                "u32": _float_bits(value),
            }
            for name, value in zip(
                ("frictionloss", "qvel", "invweight", "solref", "solimp", "timestep"),
                fixture_values,
                strict=True,
            )
        },
        "worlds": nworld,
        "njmax": njmax,
        "candidate_dof_order_ascending": all_candidate_ascending,
        "counts_and_addresses_complete": all_counts_complete,
        "active_addressed_rows_exact_without_overflow": source_bound_exact,
        "overflow_negative": overflow,
        "overflow_decision": "overflow-negative-no-qualification"
        if overflow
        else "no-overflow",
        "qualification": False,
        "fixture_decision": "dense-rows-by-address-exact"
        if source_bound_exact
        else "dense-rows-negative-or-overflow",
        "per_world": per_world,
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
        "flags": {
            "runtime_cause_proven": False,
            "native_qualified": False,
            "full_window_qualified": False,
            "training_authorized": False,
            "physical_acceptance": False,
        },
    }
    from mjlab_microduck.stance_com_entry_receiver import canonical

    _need(len(canonical(result)) <= OUTPUT_CAP, "bounded complete fixture report")
    return result


def predeclared_fixture_values():
    """Eight tiny, freshly owned input cases; these are not a robot model."""

    def values(worlds=1, rows=1, capacity=32):
        return {
            "frictionloss": np.zeros((rows, NV), dtype="<f4"),
            "qvel": np.arange(worlds * NV, dtype="<f4").reshape(worlds, NV) / 100,
            "invweight": np.full((rows, NV), 0.75, dtype="<f4"),
            "solref": np.broadcast_to(
                np.array([0.02, 1.0], dtype="<f4"), (rows, NV, 2)
            ).copy(),
            "solimp": np.broadcast_to(
                np.array([0.9, 0.95, 0.01, 0.5, 2.0], dtype="<f4"), (rows, NV, 5)
            ).copy(),
            "timestep": np.full(rows, 0.002, dtype="<f4"),
            "njmax": capacity,
        }

    cases = {"empty": values()}
    cases["broadcast"] = values(3)
    cases["broadcast"]["frictionloss"][0, [1, 6, 19]] = [0.3, 0.8, 1.1]
    cases["per-world"] = values(4, 4)
    for world, dofs in enumerate(([0, 4], [1, 5, 9], [2, 6, 10, 14], [19])):
        cases["per-world"]["frictionloss"][world, dofs] = (world + 1) / 10
    cases["mixed-broadcast"] = values(4)
    mixed = cases["mixed-broadcast"]
    mixed["frictionloss"][0, [2, 8]] = [0.3, 0.9]
    mixed["invweight"] = np.full((4, NV), 0.75, dtype="<f4")
    mixed["invweight"][1] *= 1.2
    mixed["solimp"] = np.repeat(mixed["solimp"], 4, axis=0)
    mixed["solimp"][2, :, 0] = 0.85
    mixed["solref"] = np.repeat(mixed["solref"], 4, axis=0)
    mixed["solref"][3, :, 0] = 0.003  # exercise the original REFSAFE clamp
    mixed["timestep"] = np.array([0.002, 0.003, 0.004, 0.005], dtype="<f4")
    cases["all-dofs"] = values(1, capacity=NV)
    cases["all-dofs"]["frictionloss"][:] = np.linspace(0.01, 1.0, NV, dtype="<f4")
    cases["signed-zero"] = values()
    cases["signed-zero"]["frictionloss"][0, 6] = 0.5
    cases["signed-zero"]["qvel"][0, 6] = np.array([0x80000000], dtype="<u4").view(
        "<f4"
    )[0]
    cases["direct-solref"] = values(2)
    cases["direct-solref"]["frictionloss"][0, [0, 6, 19]] = 0.4
    cases["direct-solref"]["solref"][..., 0] = -100.0
    cases["direct-solref"]["solref"][..., 1] = -2.0
    cases["overflow"] = values(2, 2, capacity=2)
    cases["overflow"]["frictionloss"][:, [1, 3, 7, 11, 18]] = 0.5
    return cases


def run_predeclared_cpu_matrix():
    """Complete CPU component evidence, including a deliberately negative overflow."""
    cases = {
        name: run_cpu_fixture(**values)
        for name, values in predeclared_fixture_values().items()
    }
    expectations = {
        name: report["fixture_decision"]
        == (
            "dense-rows-negative-or-overflow"
            if name == "overflow"
            else "dense-rows-by-address-exact"
        )
        and report["overflow_negative"] == (name == "overflow")
        and report["counts_and_addresses_complete"]
        and report["candidate_dof_order_ascending"]
        and all(value is False for value in report["flags"].values())
        for name, report in cases.items()
    }
    bindings = [report["source_binding_before"] for report in cases.values()]
    _need(
        all(binding == bindings[0] for binding in bindings),
        "one unchanged matrix source binding",
    )
    return {
        "protocol": PROTOCOL + ":eight-case-matrix",
        "device": "cpu",
        "cases": cases,
        "case_expectations_met": expectations,
        "component_expectations_met": all(expectations.values()),
        "flags": {name: False for name in next(iter(cases.values()))["flags"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.output)
    _need(
        ".." not in path.parts and path.name not in ("", ".", ".."),
        "literal output path without parent traversal",
    )
    path = path.absolute()
    parent_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Anchor every parent component with directory descriptors. Neither an
        # ancestor symlink nor a replacement of the path while computing can
        # redirect the exclusive write into a newly selected directory.
        for part in path.parent.parts[1:]:
            child_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("new exclusive fixture output")
        result = run_predeclared_cpu_matrix()
        from mjlab_microduck.stance_com_entry_receiver import canonical

        payload = canonical(result)
        _need(0 < len(payload) <= OUTPUT_CAP, "bounded complete CPU matrix")
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                _need(written > 0, "progressing exclusive CPU report write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    _need(
        result["component_expectations_met"],
        "CPU component matrix expectation failure; retained report",
    )


if __name__ == "__main__":
    main()
