"""Synthetic complete boundary/restore metadata; no GPU or physics execution."""
import ast
import ctypes
from dataclasses import dataclass, make_dataclass
from enum import IntEnum
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mjlab_microduck import ada_measured_boundary_packet as p
from test_ada_saved_pose_response_receiver import packet


class Array:
    def __init__(self, value, row=None, *, dtype=None, device=None):
        self.value = value.copy(); self.dtype = row["dtype"] if row else dtype
        self.shape = tuple(row["shape"]) if row else value.shape
        self.strides = tuple(row["strides"]) if row else value.strides
        self.device = NS(is_cpu=True, is_cuda=False)
    def numpy(self): return self.value.copy()


class Choice(IntEnum): VALUE = 3


@dataclass
class Static:
    number: int
    array: object
    option: object


def test_import_inert_no_executable_launcher():
    code = "import sys;from mjlab_microduck import ada_measured_boundary_packet as p;assert not {'numpy','warp','torch','mujoco','mujoco_warp'} & sys.modules.keys();assert not hasattr(p,'main')"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_complete_portable_statics_and_explicit_signed_zero():
    a = Array(np.zeros(1, np.float32), dict(dtype="f32", shape=[1], strides=[4]))
    value = dict(model=Static(20, a, dict(callback=None, solver=Choice.VALUE, flag=True, value=-0.,
                 numpy=np.array([2], np.int32), scalar=np.int32(4), c=(ctypes.c_int * 2)(5, 6), sequence=(1, "x"))))
    r = p.portable_static(value, Array)
    assert r["arrays"] == ["/model/array"]
    rows = r["scalars"]
    assert rows["/model/option/value"] == dict(kind="float64", hex="-0x0.0p+0")
    assert rows["/model/option/solver"]["value"] == 3
    assert rows["/model/option/scalar"]["bytes"] == np.int32(4).tobytes().hex()
    assert rows["/model/option/numpy"]["bytes"] == np.array([2], np.int32).tobytes().hex()
    assert rows["/model/option/c/1"]["value"] == 6
    assert "ptr" not in str(rows) and "device" not in str(rows)
    a.value[:] = 1
    assert p.portable_static(value, Array) == r  # Separate byte binder is mandatory.
    value["model"].number = 21
    assert p.digest(p.portable_static(value, Array)) != p.digest(r)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float32("nan"),
    np.array([np.inf]), np.array([object()]), object(), lambda: None, {"a/b": 1}, {1: 2}])
def test_unsupported_or_nonfinite_statics_refuse(value):
    with pytest.raises(ValueError): p.portable_static(value, Array)


def static_inventory():
    capacities = dict(nworld=2, naconmax=256, naccdmax=256, njmax=512, njmax_pad=512, njmax_nnz=10240)
    def arrays(n): return {"array" + str(i): Array(np.zeros(1), dtype="f64") for i in range(n)}
    value = dict(model=dict(arrays(347), nv=20, nv_pad=20, is_sparse=False,
                           block_dim=32, opt=dict(cone=0), callback=None), data=dict(arrays(114), **capacities))
    result = p.portable_static(value, Array)
    return result, dict(child=dict(input_manifest={k: {} for k in result["arrays"]})), value


def test_all_array_and_six_capacity_binding():
    r, report, value = static_inventory()
    assert len(p.bind_static_inventory(r, report, value=value, array_type=Array)) == 64


@pytest.mark.parametrize("damage", ["model", "data", "extra", "capacity", "type", "world", "contact", "ccd", "constraint", "padding", "nnz", "convention"])
def test_static_inventory_or_capacity_refusals(damage):
    r, report, value = static_inventory()
    if damage == "model": r["arrays"].remove("/model/array0")
    if damage == "data": r["arrays"].remove("/data/array0")
    if damage == "extra": r["arrays"].append("/data/extra")
    if damage == "capacity": r["scalars"].pop("/data/njmax_pad")
    if damage == "type": r["scalars"]["/data/njmax_nnz"]["value"] = True
    if damage in {"world", "contact", "ccd", "constraint"}:
        key = dict(world="nworld", contact="naconmax", ccd="naccdmax", constraint="njmax")[damage]
        r["scalars"]["/data/" + key]["value"] += 1
    if damage == "convention": r["convention"] = "pointer or historical identity"
    if damage == "padding": r["scalars"]["/data/njmax_pad"]["value"] += 1
    if damage == "nnz": r["scalars"]["/data/njmax_nnz"]["value"] += 1
    with pytest.raises(ValueError): p.bind_static_inventory(r, report, value=value, array_type=Array)


@pytest.mark.parametrize("damage", ["missing", "orphan", "extra_field", "duplicate", "unsupported", "nonfinite", "boolean_int"])
def test_complete_static_tree_refusals(damage):
    r, report, value = static_inventory()
    if damage == "missing": r["scalars"].pop("/data/nworld")
    if damage == "orphan": r["scalars"]["/orphan"] = dict(kind="int", value=1)
    if damage == "extra_field": r["scalars"]["/data/nworld"]["extra"] = True
    if damage == "duplicate": r["scalars"][""]["keys"].append("data")
    if damage == "unsupported": r["scalars"]["/data/nworld"] = dict(kind="callback", value="x")
    if damage == "nonfinite": r["scalars"]["/data/nworld"] = dict(kind="float64", hex="nan")
    if damage == "boolean_int": r["scalars"]["/data/nworld"]["value"] = True
    with pytest.raises(ValueError): p.bind_static_inventory(r, report, value=value, array_type=Array)


@pytest.mark.parametrize("name", ["nv", "nv_pad", "is_sparse", "block_dim", "opt", "callback"])
def test_consistent_omission_of_actual_model_statics_refused(name):
    r, report, value = static_inventory()
    r["scalars"]["/model"]["keys"].remove(name)
    for key in list(r["scalars"]):
        if key == "/model/" + name or key.startswith("/model/" + name + "/"):
            del r["scalars"][key]
    p.validate_static_structure(r)  # A well-formed but incomplete tree is insufficient.
    with pytest.raises(ValueError, match="complete actual"):
        p.bind_static_inventory(r, report, value=value, array_type=Array)


def test_one_new_kinematics_then_unmodified_frozen_stage_order(monkeypatch):
    log = []
    modules = dict(smooth=NS(kinematics=lambda m, d: log.append(("kinematics", m, d))))
    monkeypatch.setattr(p.response, "staged_calls", lambda m, d, mods, c: log.append(("staged", m, d, mods, c)))
    capture = object(); p.measured_staged_calls("model", "data", modules, capture)
    assert log == [("kinematics", "model", "data"), ("staged", "model", "data", modules, capture)]


def measured():
    a, l, r, b, s = packet()
    for name in p.response.POSES:
        for stage in p.response.STAGES:
            a[stage + "/data/" + name].view(np.float32)[0] = .125
    return a, l, r, b, s


def test_actual_new_eleven_pose_bundle_not_historical_intervention():
    a, l, r, b, _ = measured()
    original = {k: v.tobytes() for k, v in b["prepared-inputs.npz"].items()}
    reference = p.measured_reference_banks(a, l, r, b)
    assert all(v.tobytes() == original[k] for k, v in b["prepared-inputs.npz"].items())
    assert reference["prepared-inputs.npz"]["/data/site_xpos"].tobytes() != original["/data/site_xpos"]
    result = p.analyze_measured(a, l, r, b)
    assert result["historical_live_gpu_boundary_established"] is False
    assert result["same_bank_cpu_solver_control_executed"] is False and result["training_authorized"] is False
    with pytest.raises(ValueError): p.response.decode_stages(a, l, r, b)


@pytest.mark.parametrize("name", p.response.POSES + p.response.saved.p.p.base.UNCHANGED)
def test_every_measured_pose_and_original_state_remains_exact(name):
    a, l, r, b, _ = measured(); a["after_solve/data/" + name][0] ^= 1
    with pytest.raises(ValueError): p.analyze_measured(a, l, r, b)


def restore_fixture(monkeypatch):
    a, l, r, b, stages = packet()
    rows = r["child"]["input_manifest"]
    actual = {k: Array(np.zeros_like(v), rows[k]) for k, v in stages["before_solve"].items()}
    monkeypatch.setattr(p.response.saved.p.p, "all_arrays", lambda *args: actual)
    tree = {}
    for key, array in actual.items():
        node = tree
        path = key.split("/")[2:]
        for part in path[:-1]: node = node.setdefault(part, {})
        node[path[-1]] = array
    tree.update(nworld=2, naconmax=256, naccdmax=256, njmax=512, njmax_pad=512, njmax_nnz=10240)
    def datatree(node, name):
        cls = make_dataclass(name, [(k, object) for k in node])
        return cls(**{k: datatree(v, k.title()) if type(v) is dict else v for k, v in node.items()})
    data = datatree(tree, "Data")
    model = dict(nv=20, nv_pad=20, is_sparse=False, block_dim=32, opt=dict(cone=0), callback=None)
    for i in range(347):
        key = "/model/array" + str(i)
        model["array" + str(i)] = Array(np.zeros(1), dtype="f64")
        rows[key] = {}
    statics = p.portable_static(dict(model=model, data=data), Array)
    copied = []
    def copy(dest, source):
        dest.value[...] = source.value; copied.append(dest)
    wp = NS(copy=copy, array=Array, synchronize_device=lambda device: None)
    raw = {k: a["before_solve" + k] for k in l}
    return actual, raw, l, r, wp, copied, data, dict(model=model, static_manifest=statics)


def test_all114_fields_restore_and_byte_proof_before_solve(monkeypatch):
    actual, raw, layout, report, wp, copied, data, kw = restore_fixture(monkeypatch)
    hashes = p.restore_complete_data(data, raw, layout, report, wp, **kw)
    assert len(hashes) == len(copied) == 114
    assert all(actual[k].numpy().tobytes() == v.tobytes() for k, v in raw.items())


@pytest.mark.parametrize("damage", ["missing", "extra", "dtype", "shape", "stride", "gpu", "rawdtype", "rawbytes", "nan", "layout"])
def test_every_restore_leaf_validated_before_first_write(monkeypatch, damage):
    actual, raw, layout, report, wp, copied, data, kw = restore_fixture(monkeypatch); key = "/data/qacc"
    if damage == "missing": raw.pop(key)
    if damage == "extra": actual["/data/extra"] = actual[key]
    if damage == "dtype": actual[key].dtype = "wrong"
    if damage == "shape": actual[key].shape = (40,)
    if damage == "stride": actual[key].strides = (0, 0)
    if damage == "gpu": actual[key].device.is_cpu = False; actual[key].device.is_cuda = True
    if damage == "rawdtype": raw[key] = raw[key].astype(np.int32)
    if damage == "rawbytes": raw[key] = raw[key][:-1]
    if damage == "nan": raw[key].view(np.float32)[0] = np.nan
    if damage == "layout": layout[key]["numpy_shape"] = [40]
    with pytest.raises(ValueError): p.restore_complete_data(data, raw, layout, report, wp, **kw)
    assert not copied


def test_actual_post_copy_bytes_proved_without_tolerance(monkeypatch):
    actual, raw, layout, report, wp, _, data, kw = restore_fixture(monkeypatch)
    original = wp.copy
    def damaged(dest, source):
        original(dest, source)
        if dest is actual["/data/qacc"]: dest.value.view(np.uint32).flat[0] ^= 1
    wp.copy = damaged
    with pytest.raises(ValueError, match="complete restored"): p.restore_complete_data(data, raw, layout, report, wp, **kw)


@pytest.mark.parametrize("name", p.DATA_CAPACITIES + ("nv", "nv_pad", "is_sparse", "block_dim", "opt", "callback"))
def test_restore_static_preflight_before_any_write(monkeypatch, name):
    _, raw, layout, report, wp, copied, data, kw = restore_fixture(monkeypatch)
    if name in p.DATA_CAPACITIES: setattr(data, name, getattr(data, name) + 1)
    else: kw["model"][name] = None if name != "callback" else 1
    with pytest.raises(ValueError, match="complete actual"):
        p.restore_complete_data(data, raw, layout, report, wp, **kw)
    assert not copied


def solver_banks():
    a, l, r, *_ = packet()
    return ({k: a["before_solve" + k].copy() for k in l},
            {k: a["after_solve" + k].copy() for k in l}, l, r)


def test_all_six_complete_solver_outputs_compared_no_claimed_acceptance():
    before, gpu, layout, report = solver_banks(); cpu = {k: v.copy() for k, v in gpu.items()}
    cpu["/data/qacc"].view(np.float32)[0] = .5
    result = p.compare_solver_banks(before, gpu, cpu, layout, report)
    assert set(result["complete_solver_outputs"]) == p.response.SOLVER_OUTPUTS
    r = result["complete_solver_outputs"]["/data/qacc"]
    assert r["cpu_minus_gpu_max_abs"] == .5 and r["differing_scalar_elements"] == 1
    assert not r["bytes_equal"] and result["identical_kernel_execution_claimed"] is False
    assert result["solver_qualified"] is result["training_authorized"] is False


@pytest.mark.parametrize("damage", ["missing", "input", "shape", "dtype", "nonfinite", "inactive_nonfinite", "before_nonfinite", "layout"])
def test_complete_solver_branch_refusals(damage):
    before, gpu, layout, report = solver_banks(); cpu = {k: v.copy() for k, v in gpu.items()}
    if damage == "missing": cpu.pop("/data/qacc")
    if damage == "input": cpu["/data/efc/J"][0] ^= 1
    if damage == "shape": cpu["/data/qacc"] = cpu["/data/qacc"][:-1]
    if damage == "dtype": cpu["/data/qacc"] = cpu["/data/qacc"].astype(np.int32)
    if damage == "nonfinite": cpu["/data/qacc"].view(np.float32)[0] = np.nan
    if damage == "inactive_nonfinite":
        for bank in (before, gpu, cpu): bank["/data/efc/J"].view(np.float32)[-1] = np.nan
    if damage == "before_nonfinite": before["/data/qacc"].view(np.float32)[0] = np.nan
    if damage == "layout": layout["/data/qacc"]["numpy_shape"] = [40]
    with pytest.raises(ValueError): p.compare_solver_banks(before, gpu, cpu, layout, report)


def test_no_allocation_launcher_or_integrator_and_no_reconstruction_of_inputs():
    tree = ast.parse(Path(p.__file__).read_text())
    names = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not names & {"make_data", "put_model", "MjData", "Popen", "systemd_run", "forward", "step", "mj_step", "sensor_acc"}
    assert "kinematics" in names and "staged_calls" in names
