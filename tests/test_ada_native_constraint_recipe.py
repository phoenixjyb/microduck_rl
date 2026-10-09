"""Pure scalar and serialized-input tests; no native Model/Data or physics."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mjlab_microduck import ada_native_constraint_recipe as p
from test_ada_native_constraint_capture import fixture as capture_fixture


@pytest.mark.parametrize("position,expected,derivative", [
    (0., .9, 0.), (.0005, .925, 100.), (-.0005, .925, -100.),
    (.001, .95, 0.), (-.002, .95, 0.)])
def test_native_impedance_endpoints_midpoint_and_sign(position, expected, derivative):
    I, P, dmax = p.impedance([.9, .95, .001, .5, 2.], position)
    assert I == pytest.approx(expected) and P == pytest.approx(derivative) and dmax == .95


@pytest.mark.parametrize("solimp,expected", [([.9, .9, .001, .5, 2.], .9),
    ([.9, .95, 0., .5, 2.], .925), ([0., 0., 0., .5, 2.], .0001)])
def test_native_flat_or_zero_width_impedance(solimp, expected):
    I, P, _ = p.impedance(solimp, -.0002)
    assert I == pytest.approx(expected) and P == 0.


def args(kind=6):
    return dict(row_type=kind, timestep=.002, impratio=1., inverse_weight=2.,
        solref=[.02, 1.], solimp=[.9, .95, .001, .5, 2.], position=-.0002 if kind == 6 else 0.,
        margin=0., velocity=0., frictionloss=0. if kind == 6 else .02,
        friction=[.7, .7, .005, .0001, .0001] if kind == 6 else None)


def test_native_positive_pyramid_recipe_from_independent_scalar_equations():
    value = args(); r = p.scalar_recipe(**value)
    I = .9 + 2 * (.2**2) * (.95 - .9)
    approximate = 2. * (1. + .7**2)
    expected_R = 2. * .7**2 * (1. - I) * approximate / I
    assert r["K"] == pytest.approx(1. / (.95**2 * .02**2))
    assert r["B"] == pytest.approx(2. / (.95 * .02))
    assert r["I"] == I and r["P"] == pytest.approx(-40.)
    assert r["R"] == pytest.approx(expected_R) and r["D"] == pytest.approx(1. / expected_R)
    assert r["diagA"] == pytest.approx(expected_R * I / (1. - I))
    assert r["aref"] == pytest.approx(r["K"] * I * .0002)


@pytest.mark.parametrize("kind", [1, 6])
def test_direct_reference_and_dof_friction_has_zero_K(kind):
    value = args(kind); value["solref"] = [-50000., -200.]
    r = p.scalar_recipe(**value)
    assert r["K"] == (0. if kind == 1 else pytest.approx(50000. / .95**2))
    assert r["B"] == pytest.approx(200. / .95)
    if kind == 1:
        assert r["mu"] is None and r["aref"] == 0. and r["frictionloss"] == .02


def test_native_refsafe_clamp_and_impratio_are_source_ordered():
    a = args(); a["solref"] = [.0001, 1.]; a["impratio"] = 4.
    r = p.scalar_recipe(**a)
    assert r["K"] == pytest.approx(1. / (.95**2 * .004**2))
    assert r["mu"] == pytest.approx(.35)
    baseline = deepcopy(a); baseline["impratio"] = 1.
    assert r["R"] == pytest.approx(p.scalar_recipe(**baseline)["R"] / 4.)


def test_positive_reference_dof_friction_K_stays_zero():
    value = args(1); value["velocity"] = .01
    r = p.scalar_recipe(**value)
    assert r["K"] == 0. and r["aref"] == -r["B"] * .01
    assert r["pre_impedance_diagonal"] == value["inverse_weight"]


@pytest.mark.parametrize("damage", ["none", "onefield", "topology"])
def test_selected_model_precision_brackets_are_cast_descriptors_not_identity(monkeypatch, damage):
    arrays, _, _ = capture_fixture()
    prior = dict(child=dict(input_manifest={}))
    banks = {"prepared-inputs.npz": {}}
    layouts = {}
    bound = []
    monkeypatch.setattr(p.audit, "bound_model_values", lambda *a: bound.append(True))
    monkeypatch.setattr(p.audit.response, "expanded_layout", lambda row: layouts[row["key"]])
    for name in p.MODEL_COMPARE:
        integer = name == "geom_bodyid"
        actual = np.arange(6, dtype=np.int32 if integer else np.float64).reshape((6,) if integer else (2,3))
        gpu = actual.astype(np.int32 if integer else np.float32)
        if not integer: gpu = gpu.reshape((1,) + gpu.shape)
        key = "/model/" + name
        layouts[key] = dict(numpy_dtype=gpu.dtype, numpy_shape=gpu.shape)
        prior["child"]["input_manifest"][key] = dict(key=key)
        banks["prepared-inputs.npz"][key] = np.frombuffer(gpu.tobytes(), np.uint8)
        for w in range(2): arrays[f"model/{w}/model/{name}"] = actual.copy()
    if damage == "onefield": arrays["model/1/model/geom_friction"][0,0] = 1.
    if damage == "topology": arrays["model/1/model/geom_friction"] = np.zeros((1,3))
    if damage == "topology":
        with pytest.raises(ValueError): p.model_precision_brackets(arrays, prior, banks, {})
    else:
        result = p.model_precision_brackets(arrays, prior, banks, {})
        assert len(result) == 16 and len(bound) == 1
        assert sum(not r["native_cast_to_gpu_dtype_bytes_equal"] for r in result) == (1 if damage == "onefield" else 0)


@pytest.mark.parametrize("damage", ["type", "time", "ratio", "inv", "ref", "power", "order", "friction", "anisotropic", "nan", "dofpos"])
def test_unsupported_or_nonfinite_scalar_recipe_refused(damage):
    a = args()
    if damage == "type": a["row_type"] = 7
    if damage == "time": a["timestep"] = 0.
    if damage == "ratio": a["impratio"] = 0.
    if damage == "inv": a["inverse_weight"] = 0.
    if damage == "ref": a["solref"] = [.02, -1.]
    if damage == "power": a["solimp"][4] = 3.
    if damage == "order": a["solimp"][:2] = [.95, .9]
    if damage == "friction": a["friction"] = [1., 1.]
    if damage == "anisotropic": a["friction"][1] = .8
    if damage == "nan": a["velocity"] = math.nan
    if damage == "dofpos": a = args(1); a["position"] = .01
    with pytest.raises(ValueError): p.scalar_recipe(**a)


def native_fixture():
    arrays, _, _ = capture_fixture()
    scalars = {"option/" + k:dict(value=v) for k,v in dict(disableflags=0, enableflags=0, cone=0, solver=2,
        noslip_iterations=0, timestep=.002, impratio=1.).items()}
    capture = dict(model_static=[dict(scalars=deepcopy(scalars)) for _ in range(2)])
    for w in range(2):
        for name, value in dict(dof_invweight0=np.ones(20), dof_solref=np.tile([-.05, -.2], (20, 1)),
            dof_solimp=np.tile([.99, .9999, .001, .5, 2.], (20, 1)), dof_frictionloss=np.ones(20),
            geom_bodyid=np.ones(82, np.int32), body_invweight0=np.ones((16, 2))).items():
            arrays[f"model/{w}/model/{name}"] = value
        arrays[f"model/{w}/model/geom_bodyid"][0] = 0
        arrays[f"model/{w}/model/body_invweight0"][0] = 0.
        arrays[f"efc/{w}/id"][:14] = np.arange(6, 20, dtype=np.int32)
        ncon = len(arrays[f"contact/{w}/dist"])
        arrays[f"contact/{w}/geom"] = np.array([[0,29], [0,29], [0,79], [0,79]][:ncon], np.int32).reshape(ncon, 2)
        arrays[f"contact/{w}/flex"][:] = -1
        arrays[f"contact/{w}/dim"][:] = 3
        arrays[f"contact/{w}/solref"][:] = [.02, 1.]
        arrays[f"contact/{w}/solimp"][:] = [.9, .95, .001, .5, 2.]
        arrays[f"contact/{w}/friction"][:] = [.7, .7, .005, .0001, .0001]
    return arrays, capture


def test_complete_native_44row_partition_is_descriptive_not_a_gate():
    rows, contacts = p.native_recipes(*native_fixture())
    assert len(rows) == 44 and len(contacts) == 4
    assert sum(r["row_type"] == 1 for r in rows) == 28
    assert sum(r["row_type"] == 6 for r in rows) == 16
    assert all(r["recipe"]["K"] == 0. for r in rows if r["row_type"] == 1)
    assert any(r["recipe_minus_captured"]["D"] != 0. for r in rows)
    assert all(set(r["recipe_fp64_bytes_equal"]) == set(r["captured"]) for r in rows)


@pytest.mark.parametrize("damage", ["branch", "count", "rowtype", "duplicateid", "id", "address", "flex", "body", "modeldtype", "modelshape"])
def test_complete_native_recipe_refuses_other_branches_or_partial_linkage(damage):
    arrays, capture = native_fixture()
    if damage == "branch": capture["model_static"][0]["scalars"]["option/enableflags"]["value"] = 1
    if damage == "count": arrays["efc/1/type"] = np.ones(29, np.int32)
    if damage == "rowtype": arrays["efc/1/type"][0] = 2
    if damage == "duplicateid": arrays["efc/1/id"][1] = arrays["efc/1/id"][0]
    if damage == "id": arrays["efc/1/id"][14] = 9
    if damage == "address": arrays["contact/1/efc_address"][0] = 13
    if damage == "flex": arrays["contact/1/flex"][0,0] = 0
    if damage == "body": arrays["model/1/model/geom_bodyid"][29] = 99
    if damage == "modeldtype": arrays["model/1/model/dof_solref"] = arrays["model/1/model/dof_solref"].astype(np.float32)
    if damage == "modelshape": arrays["model/1/model/dof_solref"] = np.zeros(20)
    with pytest.raises(ValueError): p.native_recipes(arrays, capture)


@pytest.mark.parametrize("damage", ["none", "wrongbytes", "size", "symlink"])
def test_c_source_binding_is_complete_not_version_label(tmp_path, monkeypatch, damage):
    path = tmp_path / "source.c"; raw = b"synthetic C source recipe, not a binary"
    path.write_bytes(raw); monkeypatch.setattr(p, "C_BYTES", len(raw)); monkeypatch.setattr(p, "C_SHA", sha256(raw).hexdigest())
    if damage == "wrongbytes": path.write_bytes(raw.replace(b"recipe", b"damage"))
    if damage == "size": path.write_bytes(raw + b"extra")
    if damage == "symlink": target = tmp_path / "linked.c"; target.symlink_to(path); path = target
    if damage == "none":
        result = p.bind_c_source(path)
        assert result["wheel_binary_correspondence_established"] is False and result["source_recipe_only"] is True
    else:
        with pytest.raises(ValueError): p.bind_c_source(path)


@pytest.mark.parametrize("damage", ["none", "value", "flag", "module", "extra"])
def test_pure_receiver_recomputes_every_result_and_rejects_forged_scope(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(p, "source_check", lambda source: None)
    expected = dict(protocol=p.PROTOCOL, result=[1.,2.], **deepcopy(p.CONTRACT))
    monkeypatch.setattr(p, "analyze", lambda *args: deepcopy(expected))
    actual = deepcopy(expected); actual.update(source="a"*40, module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    if damage == "value": actual["result"][0] = 0.
    if damage == "flag": actual["training_authorized"] = True
    if damage == "module": actual["module_sha256"] = "0"*64
    if damage == "extra": actual["unbound"] = 1
    path = tmp_path / "result.json"; path.write_text(json.dumps(actual))
    if damage == "none": assert p.receive(path,"a"*40,None,None,None,None)["result"] == [1.,2.]
    else:
        with pytest.raises(ValueError): p.receive(path,"a"*40,None,None,None,None)


def test_native_recipe_module_is_runtime_inert_in_fresh_process():
    code = "import sys; from mjlab_microduck import ada_native_constraint_recipe as p; assert not any(n in sys.modules for n in ('mujoco','mujoco_warp','warp','torch')); assert not p.CONTRACT['training_authorized']"
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=10,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
    assert result.returncode == 0, result.stderr
