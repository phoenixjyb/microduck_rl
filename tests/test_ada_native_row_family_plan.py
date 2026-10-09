"""Synthetic complete hybrid encoding/receipts; no simulator runtime."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mjlab_microduck import ada_native_row_family_plan as p
from test_ada_native_constraint_recipe import native_fixture
from test_ada_saved_pose_response_receiver import packet


def fixture():
    native, capture = native_fixture()
    raw, layout, prior, _, _ = packet()
    before = {k:raw["before_solve"+k].copy() for k in layout}
    missing = set(p.OVERRIDES)-set(before)
    for key in sorted(missing):
        dummy = next(k for k in before if "/padding" in k)
        before.pop(dummy); layout.pop(dummy); prior["child"]["input_manifest"].pop(dummy)
        value = np.zeros((2,512),np.float32)
        row = dict(dtype="<class 'warp._src.types.float32'>",shape=[2,512],strides=[2048,4],bytes=value.nbytes,sha256=sha256(value.tobytes()).hexdigest())
        prior["child"]["input_manifest"][key] = row
        before[key] = np.frombuffer(value.tobytes(),np.uint8).copy()
        layout[key] = p.response.expanded_layout(row)
    for w in range(2):
        native[f"efc/{w}/D"][:] = 1.123456789
        native[f"efc/{w}/frictionloss"][:14] = .1
        native[f"efc/{w}/J"][:14*20].reshape(14,20)[np.arange(14),np.arange(6,20)] = 1.
        native[f"contact/{w}/vert"][:] = -1
        native[f"contact/{w}/dist"][:] = -.0001
        native[f"contact/{w}/pos"][:] = .0123456789
    return native,capture,before,layout,prior


def test_all114_leaves_exactly_partitioned_without_mutating_inputs():
    native,capture,before,layout,prior = fixture()
    inputs = {k:v.tobytes() for k,v in before.items()}; ninputs = {k:v.tobytes() for k,v in native.items()}
    encoded, report = p.encode(native,capture,before,layout,prior)
    assert len(encoded) == len(report["data_leaves"]) == 114
    assert len(report["declared_override_fields"]) == 31 and len(report["held_context_fields"]) == 83
    assert set(report["declared_override_fields"]).isdisjoint(report["held_context_fields"])
    assert set(report["declared_override_fields"])|set(report["held_context_fields"]) == set(encoded)
    assert all(encoded[k].tobytes()==before[k].tobytes() for k in report["held_context_fields"])
    assert {k:v.tobytes() for k,v in before.items()} == inputs
    assert {k:v.tobytes() for k,v in native.items()} == ninputs
    assert report["counts"] == dict(native_ncon=[0,4],ne=[0,0],nf=[14,14],nl=[0,0],nefc=[14,30],nacon=[4])
    d = p.full_bank(encoded,layout,prior)
    assert d["/data/efc/D"][1,:30].tobytes() == native["efc/1/D"].astype(np.float32).tobytes()
    assert not np.any(d["/data/efc/J"][0,14:]) and not np.any(d["/data/efc/J"][1,30:])
    assert d["/data/contact/efc_address"][:4].tolist() == [list(range(i,i+4)) for i in (14,18,22,26)]
    assert (d["/data/contact/efc_address"][4:] == -1).all()
    assert (d["/data/contact/geomcollisionid"] == -1).all()
    assert d["/data/contact/type"][:4].tolist() == [1]*4
    assert all(r["world"] == 1 and r["native_slot"] == r["hybrid_global_slot"] for r in report["own_contact_links"])


@pytest.mark.parametrize("key", ["/data/qacc","/data/qacc_warmstart","/data/qfrc_constraint","/data/solver_niter","/data/efc/force","/data/efc/state","/data/efc/Ma"])
def test_gpu_solver_state_is_held_not_replaced_with_native_final_results(key):
    native,capture,before,layout,prior = fixture()
    before[key][0] ^= 1
    for w in range(2):
        native[f"efc/{w}/force"][:] = np.nan
        native[f"efc/{w}/state"][:] = 123456
        native[f"efc/{w}/b"][:] = np.inf
        native["fields/qacc"][:] = np.nan
    encoded,report = p.encode(native,capture,before,layout,prior)
    assert encoded[key].tobytes() == before[key].tobytes() and key in report["held_context_fields"]


@pytest.mark.parametrize("key", list(p.OVERRIDES))
def test_each_declared_target_layout_is_checked(key):
    native,capture,before,layout,prior = fixture()
    # Keep source bytes, but break one canonical logical descriptor.
    layout[key]["numpy_shape"] = [layout[key]["numpy_shape"][0]+1]
    with pytest.raises(ValueError): p.encode(native,capture,before,layout,prior)


@pytest.mark.parametrize("damage", ["missing","extra","rawdtype","rawshape","nan","bytecount","dirtystride","counts","countdtype","countshape","dofid","contactid","address","dim","geometry","flex","vert","excluded","D","frictionloss","castoverflow"])
def test_partial_or_incoherent_family_refuses(damage):
    native,capture,before,layout,prior = fixture()
    key = "/data/qpos"
    if damage == "missing": before.pop(key)
    if damage == "extra": before["extra"] = np.zeros(1,np.uint8)
    if damage == "rawdtype": before[key] = before[key].astype(np.int32)
    if damage == "rawshape": before[key] = before[key].reshape(2,-1)
    if damage == "nan": before[key].view(np.float32)[0] = np.nan
    if damage == "bytecount": before[key] = before[key][:-1]
    if damage == "dirtystride": prior["child"]["input_manifest"][key]["strides"][0] += 4
    if damage == "counts": native["data/1/counters"][0] = 3
    if damage == "countdtype": native["data/1/counters"] = native["data/1/counters"].astype(np.float64)
    if damage == "countshape": native["data/1/counters"] = native["data/1/counters"][:-1]
    if damage == "dofid": native["efc/1/id"][0] = 5
    if damage == "contactid": native["efc/1/id"][14] = 3
    if damage == "address": native["contact/1/efc_address"][0] = 13
    if damage == "dim": native["contact/1/dim"][0] = 4
    if damage == "geometry": native["contact/1/geom"][0,1] = 81
    if damage == "flex": native["contact/1/flex"][0,0] = 0
    if damage == "vert": native["contact/1/vert"][0,0] = 0
    if damage == "excluded": native["contact/1/dist"][0] = .0001
    if damage == "D": native["efc/1/D"][14] = 0.
    if damage == "frictionloss": native["efc/1/frictionloss"][0] = -1.
    if damage == "castoverflow": native["contact/1/pos"][0,0] = 1e100
    if damage == "castoverflow":
        with pytest.warns(RuntimeWarning),pytest.raises(ValueError): p.encode(native,capture,before,layout,prior)
    else:
        with pytest.raises(ValueError): p.encode(native,capture,before,layout,prior)


@pytest.mark.parametrize("field", ["source","module","descriptor","flag","extra","payload","leaf","missing_leaf","duplicate_archive"])
def test_receiver_rebuilds_every_descriptor_and_full_bank(tmp_path,monkeypatch,field):
    native,capture,before,layout,prior = fixture()
    encoded,description = p.encode(native,capture,before,layout,prior)
    result = dict(protocol=p.PROTOCOL,decision=p.DECISION,**description,**deepcopy(p.CONTRACT))
    monkeypatch.setattr(p,"source_check",lambda source:None)
    monkeypatch.setattr(p,"analyze",lambda *args:(deepcopy(result),{k:v.copy() for k,v in encoded.items()}))
    path = tmp_path/"packet.json"; arrays = {k:v.copy() for k,v in encoded.items()}
    actual = deepcopy(result); actual.update(source="a"*40,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    if field == "source": actual["source"] = "b"*40
    if field == "module": actual["module_sha256"] = "0"*64
    if field == "descriptor": actual["own_contact_links"][0]["rows"][0] = 13
    if field == "flag": actual["training_authorized"] = True
    if field == "extra": actual["unbound"] = 1
    if field == "leaf": arrays["/data/efc/J"][0] ^= 1
    if field == "missing_leaf": arrays.pop("/data/efc/J")
    with path.with_suffix(".npz").open("xb") as stream: np.savez_compressed(stream,**arrays)
    if field == "duplicate_archive":
        import zipfile
        with zipfile.ZipFile(path.with_suffix(".npz"),"a") as z:
            with pytest.warns(UserWarning): z.writestr(z.namelist()[0],z.read(z.namelist()[0]))
    raw = path.with_suffix(".npz").read_bytes()
    actual["payload"] = dict(file=path.with_suffix(".npz").name,bytes=len(raw),sha256=sha256(raw).hexdigest())
    if field == "payload": actual["payload"]["sha256"] = "0"*64
    path.write_text(json.dumps(actual))
    with pytest.raises(ValueError): p.receive(path,"a"*40,None,None,None,None,None)


def test_successful_receiver_is_complete_and_deterministic(tmp_path,monkeypatch):
    native,capture,before,layout,prior = fixture()
    encoded,description = p.encode(native,capture,before,layout,prior)
    result = dict(protocol=p.PROTOCOL,decision=p.DECISION,**description,**deepcopy(p.CONTRACT))
    monkeypatch.setattr(p,"source_check",lambda source:None)
    monkeypatch.setattr(p,"analyze",lambda *args:(deepcopy(result),encoded))
    path = tmp_path/"packet.json"
    with path.with_suffix(".npz").open("xb") as stream: np.savez_compressed(stream,**encoded)
    raw = path.with_suffix(".npz").read_bytes()
    result.update(source="a"*40,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest(),payload=dict(file="packet.npz",bytes=len(raw),sha256=sha256(raw).hexdigest()))
    path.write_text(json.dumps(result))
    actual,arrays = p.receive(path,"a"*40,None,None,None,None,None)
    assert actual == result and set(arrays) == set(encoded)


def test_import_and_help_inert():
    code = "import sys;from mjlab_microduck import ada_native_row_family_plan as p;assert not {'numpy','mujoco','warp','mujoco_warp','torch'}&sys.modules.keys();assert not p.CONTRACT['subsequent_solver_execution_authorized']"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
    subprocess.run([sys.executable,"-m",p.__name__,"--help"],capture_output=True,check=True,timeout=10)


@pytest.mark.parametrize("damage", [False,True])
def test_exact_encoder_source_bytes_are_bound(monkeypatch,damage):
    called = []
    monkeypatch.setattr(p.recipe,"source_check",lambda source:called.append(source))
    monkeypatch.setattr(p.host,"read",lambda *args,**kwargs:b"different" if damage else Path(p.__file__).read_bytes())
    if damage:
        with pytest.raises(ValueError): p.source_check("a"*40)
    else: p.source_check("a"*40)
    assert called == ["a"*40]
