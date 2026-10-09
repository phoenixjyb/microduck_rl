"""Pure guarded-boundary/stage fixtures; never execute response physics."""
import ast
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from mjlab_microduck import ada_saved_pose_response as p


def array(value, *, dtype="synthetic", logical_shape=None, strides=None):
    return NS(dtype=dtype, shape=value.shape if logical_shape is None else logical_shape,
              strides=value.strides if strides is None else strides, numpy=lambda: value.copy(),
              device=NS(is_cpu=True, is_cuda=False))


def topology():
    basic = dict(nflex=0, neq=0, nmocap=0, na=0, ntendon=0, nq=21, nv=20, nu=14, nsite=7)
    native = NS(**basic, actuator_trntype=np.zeros(14, np.int32))
    model = NS(**basic, is_sparse=False, nacttrnbody=0,
               callback=NS(**{k: None for k in p.CALLBACKS}),
               actuator_trntype=array(np.zeros(14, np.int32)),
               opt=NS(run_collision_detection=True, graph_conditional=True))
    data = NS(qpos=array(np.zeros((2, 21), np.float32)), act=array(np.empty((2, 0), np.float32)),
              eq_active=array(np.empty((2, 0), bool)), mocap_pos=array(np.empty((2, 0, 3), np.float32)),
              mocap_quat=array(np.empty((2, 0, 4), np.float32)))
    return native, model, data


def test_import_inert_and_no_standalone_cli():
    code = "import sys; from mjlab_microduck import ada_saved_pose_response; assert not {'numpy','torch','warp','mujoco','mujoco_warp'} & sys.modules.keys(); assert not hasattr(ada_saved_pose_response,'main')"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


@pytest.mark.parametrize("damage", ["none", "flex", "equality", "mocap", "activation", "tendon", "sparse",
                                   "body", "site", "transmission", "empty", "gpu", "collision", "graph"])
def test_topology_refusals_before_dispatch(monkeypatch, damage):
    native, model, data = topology()
    called = []
    monkeypatch.setattr(p.saved, "dispatch_guard", lambda *args: called.append(True) or {"unchanged": True})
    if damage in {"flex", "equality", "mocap", "activation", "tendon"}:
        setattr(model, dict(flex="nflex", equality="neq", mocap="nmocap", activation="na", tendon="ntendon")[damage], 1)
    if damage == "sparse": model.is_sparse = True
    if damage == "body": model.nacttrnbody = 1
    if damage == "site": native.nsite = 8
    if damage == "transmission": model.actuator_trntype = array(np.ones(14, np.int32))
    if damage == "empty": data.act = array(np.zeros((2, 1), np.float32))
    if damage == "gpu": data.qpos.device.is_cuda = True
    if damage == "collision": model.opt.run_collision_detection = False
    if damage == "graph": model.opt.graph_conditional = False
    if damage == "none": assert p.topology_guard(native, model, data) == {"unchanged": True} and called == [True]
    else:
        with pytest.raises(ValueError): p.topology_guard(native, model, data)
        assert not called


@pytest.mark.parametrize("callback", p.CALLBACKS)
def test_every_callback_refuses_before_dispatch(callback, monkeypatch):
    native, model, data = topology()
    setattr(model.callback, callback, lambda *args: None)
    monkeypatch.setattr(p.saved, "dispatch_guard", lambda *args: pytest.fail("must reject earlier"))
    with pytest.raises(ValueError, match="no custom"): p.topology_guard(native, model, data)


def test_staged_order_factorization_and_boundaries_without_physics():
    log = []
    def module(names):
        return NS(**{name: (lambda *args, n=name, **kwargs: log.append((n, kwargs))) for name in names})
    modules = {"smooth": module(("com_pos", "camlight", "flex", "tendon", "crb", "tendon_armature", "transmission")),
               "forward": module(("fwd_velocity", "fwd_actuation", "fwd_acceleration")),
               "collision_driver": module(("collision",)), "constraint": module(("make_constraint",)),
               "sensor": module(("sensor_pos", "sensor_vel")), "solver": module(("solve",))}
    data = NS(sensordata=NS(zero_=lambda: log.append(("zero_sensors", {}))),
              energy=NS(zero_=lambda: log.append(("zero_energy", {}))))
    p.staged_calls(NS(), data, modules, lambda s: log.append((s, {})))
    assert [k for k, _ in log] == ["com_pos", "camlight", "flex", "tendon", "crb", "tendon_armature",
        "before_collision", "collision", "after_collision", "make_constraint", "transmission", "after_construction",
        "zero_sensors", "sensor_pos", "zero_energy", "fwd_velocity", "sensor_vel", "fwd_actuation",
        "fwd_acceleration", "before_solve", "solve", "after_solve"]
    assert next(v for k, v in log if k == "fwd_acceleration") == {"factorize": True}
    assert [k for k, _ in log if k in p.STAGES] == list(p.STAGES)


def test_exact_values_reject_signed_zero_one_bit_or_partial():
    expected = {"x": np.zeros((2, 3), np.float32)}
    p.exact_values(expected, expected, "state")
    for kind in ("bit", "signed_zero", "dtype", "shape", "missing"):
        actual = {"x": expected["x"].copy()}
        if kind == "bit": actual["x"].view(np.uint32).flat[0] ^= 1
        if kind == "signed_zero": actual["x"].flat[0] = -0.
        if kind == "dtype": actual["x"] = actual["x"].astype(np.float64)
        if kind == "shape": actual["x"] = actual["x"].reshape(3, 2)
        if kind == "missing": actual = {}
        with pytest.raises(ValueError): p.exact_values(actual, expected, "state")


def site_fixture():
    values, arrays, rows, raw = {}, {}, {}, {}
    for name in ("site_xpos", "site_xmat"):
        tail = (3, 3) if name.endswith("mat") else (3,)
        value = np.arange(2 * 7 * np.prod(tail), dtype=np.float32).reshape((2, 7) + tail)
        values[name] = value
        arrays[name] = array(value, logical_shape=(2, 7), strides=(value.strides[0], value.strides[1]))
        key = "/data/" + name
        rows[key] = dict(dtype="synthetic", shape=[2, 7], strides=list(arrays[name].strides),
                         bytes=value.nbytes, sha256=sha256(value.tobytes()).hexdigest())
        raw[key] = np.frombuffer(value.tobytes(), np.uint8).copy()
    return values, NS(**arrays), {"child": {"input_manifest": rows}}, {"prepared-inputs.npz": raw}


def test_extra_sites_bound_without_recomputing_kinematics(monkeypatch):
    values, data, report, banks = site_fixture()
    nine = {k: np.zeros((2, 1, 3), np.float32) for k in p.saved.p.KINEMATIC}
    monkeypatch.setattr(p.saved, "supplied_poses", lambda *args: dict(nine))
    poses = p.supplied_poses(data, report, banks)
    assert tuple(poses) == p.POSES
    assert all(poses[k].tobytes() == v.tobytes() and not np.shares_memory(poses[k], v) for k, v in values.items())


@pytest.mark.parametrize("name", ["site_xpos", "site_xmat"])
@pytest.mark.parametrize("damage", ["dtype", "shape", "stride", "bytes", "hash", "nonfinite"])
def test_each_site_pose_bound_layout_and_bytes_refuse(name, damage, monkeypatch):
    values, data, report, banks = site_fixture()
    monkeypatch.setattr(p.saved, "supplied_poses", lambda *args: {})
    key = "/data/" + name; row = report["child"]["input_manifest"][key]
    if damage == "dtype": row["dtype"] = "different"
    if damage == "shape": row["shape"] = [14]
    if damage == "stride": row["strides"][0] += 4
    if damage == "bytes": row["bytes"] -= 4
    if damage == "hash": row["sha256"] = "0" * 64
    if damage == "nonfinite":
        values[name].flat[0] = np.nan
        banks["prepared-inputs.npz"][key] = np.frombuffer(values[name].tobytes(), np.uint8)
        row["sha256"] = sha256(values[name].tobytes()).hexdigest()
    with pytest.raises(ValueError): p.supplied_poses(data, report, banks)


def test_candidate_addresses_are_stage_dependent_but_other_bytes_frozen():
    before = {k: np.zeros((2,) + shape, np.float32) for k, shape in p.saved.CONTACT_TAILS.items()}
    before["efc_address"] = np.full((2, 4), -1, np.int32)
    after = {k: v.copy() for k, v in before.items()}; after["efc_address"][:] = 14
    p.candidate_stability(before, after)
    after["pos"].flat[0] = 1
    with pytest.raises(ValueError, match="generated candidates"): p.candidate_stability(before, after)


def test_solver_output_whitelist_never_allows_input_mutation():
    before = {k: np.zeros(8, np.uint8) for k in p.SOLVER_OUTPUTS | {"/data/qacc_warmstart", "/data/efc/J"}}
    after = {k: v.copy() for k, v in before.items()}
    after["/data/qacc"][0] = 1
    assert p.solver_write_fence(before, after) == ["/data/qacc"]
    for key in ("/data/qacc_warmstart", "/data/efc/J"):
        damaged = {k: v.copy() for k, v in after.items()}; damaged[key][0] = 1
        with pytest.raises(ValueError): p.solver_write_fence(before, damaged)
    after.pop("/data/efc/state")
    with pytest.raises(ValueError): p.solver_write_fence(before, after)


def complete_arrays():
    values = {"/data/field" + str(i): np.asarray([[i, i+1]], np.float32) for i in range(114)}
    arrays = {k: array(v) for k, v in values.items()}
    manifest = {k: dict(dtype=str(a.dtype), shape=list(a.shape), strides=list(a.strides), bytes=values[k].nbytes)
                for k, a in arrays.items()}
    return values, arrays, {"child": {"input_manifest": manifest}}


def test_snapshot_retains_every_raw_byte_and_expanded_layout(monkeypatch):
    values, arrays, report = complete_arrays()
    monkeypatch.setattr(p.saved.p.p, "all_arrays", lambda *args: arrays)
    raw, layout = p.snapshot_data(NS(), report)
    assert len(raw) == len(layout) == 114
    for k, v in values.items():
        assert raw[k].dtype == np.uint8 and raw[k].tobytes() == v.tobytes()
        assert layout[k]["numpy_shape"] == [1, 2] and layout[k]["numpy_dtype"] == "float32"
        assert not np.shares_memory(raw[k], v)


@pytest.mark.parametrize("damage", ["missing", "extra", "dtype", "shape", "stride", "bytes", "nonfinite"])
def test_complete_snapshot_layout_or_value_refusals(damage, monkeypatch):
    values, arrays, report = complete_arrays(); key = next(iter(arrays))
    if damage == "missing": arrays.pop(key)
    if damage == "extra": arrays["/data/additional"] = arrays[key]
    if damage in {"dtype", "shape", "stride", "bytes"}:
        field = {"dtype": "dtype", "shape": "shape", "stride": "strides", "bytes": "bytes"}[damage]
        report["child"]["input_manifest"][key][field] = "bad" if field == "dtype" else [] if field in {"shape", "strides"} else 0
    if damage == "nonfinite": values[key].flat[0] = np.nan
    monkeypatch.setattr(p.saved.p.p, "all_arrays", lambda *args: arrays)
    with pytest.raises(ValueError): p.snapshot_data(NS(), report)


def test_no_integration_ordinary_forward_or_fixture_kinematics():
    tree = ast.parse(Path(p.__file__).read_text())
    names = {n.func.attr if isinstance(n.func, ast.Attribute) else p.source.dotted(n.func)
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not names & {"kinematics", "mj_kinematics", "mj_forward", "forward", "step", "mj_step", "_advance", "sensor_acc"}
    assert "make_constraint" in names and "solve" in names and "collision" in names
