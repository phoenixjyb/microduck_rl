"""Complete new measured-boundary packet primitives; no executable GPU CLI.

Not a historical Ada boundary, simulator admission, or training permission.
Runtime stage/restore helpers must be wrapped by a separately reviewed launcher.
Importing this module initializes no simulator runtime.
"""
import ctypes
from dataclasses import fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math

from mjlab_microduck import ada_saved_pose_response as response

need = response.need
PROTOCOL = "microduck-new-measured-boundary-packet-primitives-v1"
STATIC_CONVENTION = "Complete portable scalar/container topology; logical arrays bound separately; no addresses or device identity"
DATA_CAPACITIES = ("nworld", "naconmax", "naccdmax", "njmax", "njmax_pad", "njmax_nnz")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def portable_static(value, array_type):
    """Bind every dataclass/container/scalar; array contents/layouts separately.

    No selected scalar subset substitutes for this complete walk. In-process
    pointer/device stability is a separate binding, never a portable identity.
    """
    import numpy as np
    rows, arrays = {}, []
    def typename(v): return type(v).__module__ + "." + type(v).__qualname__
    def visit(v, path):
        if isinstance(v, array_type): arrays.append(path); return
        if is_dataclass(v):
            children = [(f.name, getattr(v, f.name)) for f in fields(v)]
            row = dict(kind="dataclass", type=typename(v), fields=[k for k, _ in children])
        elif isinstance(v, Enum):
            need(type(v.value) is int, "integer frozen enum static")
            rows[path] = dict(kind="enum", type=typename(v), value=v.value); return
        elif isinstance(v, np.generic):
            rows[path] = dict(kind="numpy-scalar", dtype=str(v.dtype), bytes=v.tobytes().hex())
            need(np.isfinite(v), "finite numpy static scalar"); return
        elif isinstance(v, np.ndarray):
            need(not v.dtype.hasobject and np.isfinite(v).all() and v.nbytes <= 256 * 1024,
                 "bounded plain finite static numpy bytes")
            rows[path] = dict(kind="numpy", dtype=v.dtype.str, shape=list(v.shape), bytes=v.tobytes().hex()); return
        elif isinstance(v, ctypes.Array):
            children = [(str(i), x) for i, x in enumerate(v)]
            row = dict(kind="ctypes-array", type=typename(v), size=len(v))
        elif type(v) in (list, tuple):
            children = [(str(i), x) for i, x in enumerate(v)]
            row = dict(kind=type(v).__name__, size=len(v))
        elif type(v) is dict:
            need(all(type(k) is str and "/" not in k for k in v), "unambiguous static dictionary keys")
            children = sorted(v.items()); row = dict(kind="dict", keys=[k for k, _ in children])
        elif type(v) is float:
            need(math.isfinite(v), "finite portable static float")
            rows[path] = dict(kind="float64", hex=v.hex()); return
        elif type(v) in (int, bool, str, type(None)):
            rows[path] = dict(kind=type(v).__name__, value=v); return
        else: raise ValueError("unsupported portable static: " + typename(v))
        rows[path] = row
        for key, child in children: visit(child, path + "/" + key)
    visit(value, "")
    need(len(arrays) == len(set(arrays)) and len(rows) <= 4096
         and len(json.dumps(rows, sort_keys=True, allow_nan=False).encode()) <= 256 * 1024,
         "bounded complete portable static inventory")
    result = dict(convention=STATIC_CONVENTION, arrays=sorted(arrays), scalars=rows)
    validate_static_structure(result)
    return result


def validate_static_structure(manifest):
    """Every declared child exists and every retained node is reachable."""
    import numpy as np
    need(type(manifest) is dict and set(manifest) == {"convention", "arrays", "scalars"}
         and manifest["convention"] == STATIC_CONVENTION
         and type(manifest["arrays"]) is list and type(manifest["scalars"]) is dict
         and len(manifest["arrays"]) == len(set(manifest["arrays"]))
         and all(type(k) is str for k in manifest["arrays"]), "closed portable static envelope")
    rows, arrays = manifest["scalars"], set(manifest["arrays"])
    need(len(rows) <= 4096 and not arrays & rows.keys() and "" in rows, "disjoint complete static node inventory")
    seen = set()
    def visit(path):
        need(path not in seen, "unique reachable portable static path"); seen.add(path)
        if path in arrays: return
        need(path in rows, "every declared portable child present: " + path)
        row = rows[path]
        need(type(row) is dict and type(row.get("kind")) is str, "closed static node")
        kind = row["kind"]; children = []
        if kind in ("dataclass", "dict"):
            key = "fields" if kind == "dataclass" else "keys"
            need(set(row) == {"kind", key} | ({"type"} if kind == "dataclass" else set())
                 and type(row[key]) is list and len(row[key]) == len(set(row[key]))
                 and all(type(k) is str and k and "/" not in k for k in row[key]), "complete static child names")
            if kind == "dataclass": need(type(row["type"]) is str and bool(row["type"]), "dataclass type identity")
            children = row[key]
        elif kind in ("list", "tuple", "ctypes-array"):
            need(set(row) == {"kind", "size"} | ({"type"} if kind == "ctypes-array" else set())
                 and type(row["size"]) is int and 0 <= row["size"] <= 4096, "bounded static sequence")
            if kind == "ctypes-array": need(type(row["type"]) is str and bool(row["type"]), "ctypes static identity")
            children = [str(i) for i in range(row["size"])]
        elif kind == "enum":
            need(set(row) == {"kind", "type", "value"} and type(row["type"]) is str
                 and type(row["value"]) is int, "integer static enum")
        elif kind == "float64":
            need(set(row) == {"kind", "hex"} and type(row["hex"]) is str
                 and math.isfinite(float.fromhex(row["hex"])) and float.fromhex(row["hex"]).hex() == row["hex"],
                 "exact finite signed static float")
        elif kind in ("int", "bool", "str", "NoneType"):
            need(set(row) == {"kind", "value"} and type(row["value"]) is
                 {"int": int, "bool": bool, "str": str, "NoneType": type(None)}[kind], "strict static scalar type")
        elif kind in ("numpy", "numpy-scalar"):
            need(set(row) == {"kind", "dtype", "bytes"} | ({"shape"} if kind == "numpy" else set())
                 and type(row["bytes"]) is str and len(row["bytes"]) <= 512 * 1024, "bounded numpy static leaf")
            dtype = np.dtype(row["dtype"]); need(not dtype.hasobject, "plain numpy static dtype")
            raw = bytes.fromhex(row["bytes"])
            value = np.frombuffer(raw, dtype=dtype)
            if kind == "numpy":
                shape = row["shape"]
                need(type(shape) is list and len(shape) <= 4 and all(type(n) is int and 0 <= n <= 256 * 1024 for n in shape)
                     and value.size == int(np.prod(shape)), "complete static numpy shape")
            else: need(value.size == 1, "one complete static numpy scalar")
            need(np.isfinite(value).all(), "finite numpy static bytes")
        else: raise ValueError("unsupported portable static node")
        for child in children: visit(path + "/" + child)
    visit("")
    need(seen == set(rows) | arrays, "no orphaned or omitted portable static fields")


def bind_static_inventory(manifest, report, *, value, array_type):
    validate_static_structure(manifest)
    need(manifest == portable_static(value, array_type),
         "exact complete actual Model/Data static topology and values")
    expected = report["child"]["input_manifest"]
    need(manifest["convention"] == STATIC_CONVENTION and manifest["arrays"] == sorted(expected)
         and len([k for k in expected if k.startswith("/model/")]) == 347
         and len([k for k in expected if k.startswith("/data/")]) == 114,
         "complete same347-model/114-Data portable array inventory")
    rows = manifest["scalars"]
    need(all("/data/" + name in rows and rows["/data/" + name]["kind"] == "int"
             and type(rows["/data/" + name]["value"]) is int for name in DATA_CAPACITIES),
         "every actual Data capacity retained")
    need(rows["/data/nworld"]["value"] == 2 and rows["/data/naconmax"]["value"] == 256
         and rows["/data/naccdmax"]["value"] == 256 and rows["/data/njmax"]["value"] == 512
         and rows["/data/njmax_pad"]["value"] == 512 and rows["/data/njmax_nnz"]["value"] == 10240,
         "unchanged two-world complete contact/constraint capacities")
    return digest(manifest)


def measured_staged_calls(model, data, modules, capture):
    """One NEW measured kinematics call then frozen forward order through solve.

    Caller must retain all347 model bytes/statics and114 Data fields at each
    capture and enforce topology/callback/device/lease/source/launch guards.
    This helper deliberately provides no launcher or resource authority.
    """
    modules["smooth"].kinematics(model, data)
    response.staged_calls(model, data, modules, capture)


def measured_reference_banks(arrays, layout, report, banks):
    """New measured poses, never guessed from after-solve or old history.

    Reuse the exact-layout/five-boundary/state checker with the actual new
    before-collision eleven-pose bank as its only pose reference. This changes
    no saved bytes or runtime state; old prepared seven-state bytes stay fixed.
    """
    need(set(arrays) == {stage + k for stage in response.STAGES for k in layout},
         "complete newly measured five-stage inventory")
    prepared = dict(banks["prepared-inputs.npz"])
    for name in response.POSES:
        key = "/data/" + name
        prepared[key] = arrays["before_collision" + key].copy()
    reference = dict(banks, **{"prepared-inputs.npz": prepared})
    response.decode_stages(arrays, layout, report, reference)
    return reference


def analyze_measured(arrays, layout, report, banks):
    reference = measured_reference_banks(arrays, layout, report, banks)
    value = response.analyze_stages(arrays, layout, report, reference)
    value.update(pose_reference="Actual complete eleven-pose before-collision bank of this new measured run",
                 historical_live_gpu_boundary_established=False,
                 same_bank_cpu_solver_control_executed=False, simulator_qualified=False,
                 training_authorized=False, physical_motion_authorized=False)
    return value


def restore_complete_data(data, raw, layout, report, wp, *, model, static_manifest):
    """Restore ALL114 fields into fresh CPU Data, then prove exact bytes.

    No kinematics, collision, construction, factorization, sensors or forces
    are recomputed here. Complete actual Model/Data statics are preflighted
    before copying. CPU allocation/model-byte/topology/source checks belong
    to the future closed solver-control launcher.
    """
    import numpy as np
    bind_static_inventory(static_manifest, report, value=dict(model=model, data=data), array_type=wp.array)
    actual = response.saved.p.p.all_arrays(data, "/data")
    expected = {k: v for k, v in report["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(len(actual) == 114 and set(actual) == set(raw) == set(layout) == set(expected),
         "complete unmodified114-field restore inventory")
    decoded = {}
    # Validate EVERY source/target layout and leaf before the first write.
    for key, array in actual.items():
        row = expected[key]; value = raw[key]
        need(array.device.is_cpu and not array.device.is_cuda
             and str(array.dtype) == row["dtype"] and list(array.shape) == row["shape"]
             and list(array.strides) == row["strides"] and layout[key] == response.expanded_layout(row),
             "exact CPU restore logical layout: " + key)
        need(value.dtype == np.uint8 and value.shape == (row["bytes"],), "complete original raw restore bytes")
        expanded = np.frombuffer(value.tobytes(), dtype=layout[key]["numpy_dtype"]).reshape(layout[key]["numpy_shape"]).copy()
        need(np.isfinite(expanded).all(), "finite complete restore field")
        decoded[key] = expanded
    for key, array in actual.items():
        wp.copy(array, wp.array(decoded[key], dtype=array.dtype, device="cpu"))
    wp.synchronize_device("cpu")
    captured, actual_layout = response.snapshot_data(data, report)
    response.exact_values(captured, raw, "complete restored pre-solver Data bank")
    need(actual_layout == layout, "same full actual post-copy Data layout")
    return {k: sha256(v.tobytes()).hexdigest() for k, v in captured.items()}


def compare_solver_banks(before, gpu_after, cpu_after, layout, report):
    """Descriptive same-bank outputs; never an admission or tolerance gate."""
    import numpy as np
    expected = {k: v for k, v in report["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(set(before) == set(gpu_after) == set(cpu_after) == set(layout) == set(expected) and len(before) == 114,
         "complete same-bank solver branch inventory")
    for key, row in expected.items():
        need(layout[key] == response.expanded_layout(row), "canonical complete solver branch layout")
        for bank in (before, gpu_after, cpu_after):
            raw = bank[key]
            need(raw.dtype == np.uint8 and raw.shape == (row["bytes"],), "complete solver branch leaf")
            value = np.frombuffer(raw.tobytes(), dtype=layout[key]["numpy_dtype"]).reshape(layout[key]["numpy_shape"])
            need(np.isfinite(value).all(), "finite complete solver branch")
    changed = {"gpu": response.solver_write_fence(before, gpu_after),
               "cpu": response.solver_write_fence(before, cpu_after)}
    fields = {}
    for key in sorted(response.SOLVER_OUTPUTS):
        row = layout[key]
        values = []
        for bank in (gpu_after, cpu_after):
            raw = bank[key]
            need(raw.dtype == np.uint8 and raw.shape == (row["bytes"],), "complete solver-output leaf")
            value = np.frombuffer(raw.tobytes(), dtype=row["numpy_dtype"]).reshape(row["numpy_shape"])
            need(np.isfinite(value).all(), "finite solver-output branch")
            values.append(value)
        gpu, cpu = values
        # f32/i32 operands widen exactly. No RMS backend reduction is needed.
        deltas = [float(b) - float(a) for a, b in zip(gpu.flat, cpu.flat)]
        fields[key] = dict(shape=list(gpu.shape), bytes_equal=gpu.tobytes() == cpu.tobytes(),
            gpu_sha256=sha256(gpu.tobytes()).hexdigest(), cpu_sha256=sha256(cpu.tobytes()).hexdigest(),
            cpu_minus_gpu_max_abs=max(map(abs, deltas), default=0.),
            differing_scalar_elements=sum(a.tobytes() != b.tobytes() for a, b in zip(gpu.flat, cpu.flat)))
    return dict(protocol=PROTOCOL, solver_changed_data_fields=changed, complete_solver_outputs=fields,
        comparison_convention="CPU minus GPU over all six complete solver outputs; no tolerance or manifold reconstruction",
        same_input_bank_claim_requires_external_restore_receipt=True, identical_kernel_execution_claimed=False,
        solver_qualified=False, simulator_qualified=False, training_authorized=False, physical_motion_authorized=False)
