"""Pure algebra/hypothesis coverage; no actual contact constructor or solve."""
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
from mjlab_microduck import ada_own_manifold_recipe as p
from test_ada_native_constraint_recipe import args,native_fixture


def inputs():
    row=args();row.pop("row_type");row.pop("frictionloss");row["invsqrt"]=1.
    return row


def test_positive_like_input_algebra_is_not_a_tolerance_gate():
    a=inputs();r=p.scalar_compare(a)
    assert set(r["warp_minus_C"])==set(p.FIELDS)
    assert r["warp_source_recipe"]["D"]==pytest.approx(r["native_C_recipe"]["D"])
    assert r["warp_source_recipe"]["aref"]==pytest.approx(r["native_C_recipe"]["aref"])
    assert r["ratio_minus_inverse_sqrt_impratio"]==0
    assert "solver_qualified" not in r and not p.CONTRACT["solver_qualified"]


@pytest.mark.parametrize("position",[0.,-.0005,.0005,-.001,.002])
@pytest.mark.parametrize("ratio",[1.,.5,.31622776601683794])
def test_saturated_signed_and_ratio_cases_are_retained(position,ratio):
    a=inputs();a.update(position=position,invsqrt=ratio,impratio=1./(ratio*ratio),velocity=-.2)
    r=p.scalar_compare(a)
    assert all(math.isfinite(v) for v in r["warp_minus_C"].values())
    assert r["native_C_recipe"]["aref"]==pytest.approx(r["warp_source_recipe"]["aref"])


def test_explicit_parameter_conversion_not_silent_native_identity():
    a=inputs();a["invsqrt"]=.5
    r=p.scalar_compare(a)
    assert r["ratio_minus_inverse_sqrt_impratio"]==-.5
    assert r["warp_minus_C"]["D"]!=0


def test_zero_width_source_branches_may_disagree_without_acceptance():
    a=inputs();a["solimp"][2]=0.
    r=p.scalar_compare(a)
    assert r["branch"]["native_flat_width"]
    assert r["native_C_recipe"]["I"]==pytest.approx(.925)
    assert r["warp_source_recipe"]["impedance"]==.95
    assert r["warp_minus_C"]["D"]!=0


@pytest.mark.parametrize("field",sorted(p.INPUTS))
def test_each_input_is_required(field):
    a=inputs();a.pop(field)
    with pytest.raises(ValueError):p.scalar_compare(a)


@pytest.mark.parametrize("damage",["extra","bool","nan","ratio","impratio","inv","time","negative_ref","mixed_ref","anisotropy","negative_friction","power","width_order","short_ref","short_imp","short_friction","tuple_ref"])
def test_closed_source_domain_refuses(damage):
    a=inputs()
    if damage=="extra":a["unknown"]=0.
    if damage=="bool":a["timestep"]=True
    if damage=="nan":a["velocity"]=math.nan
    if damage=="ratio":a["invsqrt"]=0.
    if damage=="impratio":a["impratio"]=0.
    if damage=="inv":a["inverse_weight"]=0.
    if damage=="time":a["timestep"]=0.
    if damage=="negative_ref":a["solref"]=[-.1,-.2]
    if damage=="mixed_ref":a["solref"][1]=-.2
    if damage=="anisotropy":a["friction"][1]=.8
    if damage=="negative_friction":a["friction"][4]=0.
    if damage=="power":a["solimp"][4]=3.
    if damage=="width_order":a["solimp"][:2]=[.95,.9]
    if damage=="short_ref":a["solref"].pop()
    if damage=="short_imp":a["solimp"].pop()
    if damage=="short_friction":a["friction"].pop()
    if damage=="tuple_ref":a["solref"]=tuple(a["solref"])
    with pytest.raises(ValueError):p.scalar_compare(a)


def fixture():
    native,capture=native_fixture();nrows,contacts=p.recipe.native_recipes(native,capture)
    models={"/model/opt/timestep":np.array([.002],np.float32),"/model/opt/impratio_invsqrt":np.ones(1,np.float32),
        "/model/body_invweight0":np.ones((1,16,2),np.float32),"/model/geom_bodyid":np.ones(82,np.int32)}
    models["/model/body_invweight0"][0,0]=0.;models["/model/geom_bodyid"][0]=0
    gcontacts=[];grows=[]
    for slot in range(8):
        c=dict(slot=slot,world=1,geom=[0,29 if slot<4 else 79],friction=[.7,.7,.005,.0001,.0001],solref=[.02,1.],solimp=[.9,.95,.001,.5,2.],dist=-.0002,includemargin=0.,efc_address=list(range(14+4*slot,18+4*slot)))
        gcontacts.append(c)
        for addr in c["efc_address"]:
            a=inputs();a.update(timestep=float(models["/model/opt/timestep"][0]),inverse_weight=1.)
            result=p.scalar_compare(a)["warp_source_recipe"]
            grows.append(dict(slot=slot,world=1,efc_address=addr,geom=c["geom"],bodies=[0,1],captured={k:float(np.float32(result[k])) for k in p.FIELDS}))
    description=dict(native_row_recipes=nrows,measured_gpu_row_recipes=grows,native_contact_inputs=contacts,measured_gpu_contact_inputs=gcontacts)
    return description,[r["scalars"] for r in capture["model_static"]],models


def test_all48_own_rows_without_cross_point_pairing_or_input_mutation():
    d,s,m=fixture();before=json.dumps(d,sort_keys=True)
    result=p.compare_rows(d,s,m)
    assert len(result["rows"])==48 and json.dumps(d,sort_keys=True)==before
    assert result["own_covered_rows"]==dict(native=list(range(14,30)),measured_gpu=list(range(14,46)))
    assert all(set(r["captured_scalar_sha256"])==set(p.FIELDS) for r in result["rows"])
    assert {r["captured_dtype"] for r in result["rows"]}=={"float32","float64"}
    assert all("hypothetical" in r["input_conversion"] for r in result["rows"])


@pytest.mark.parametrize("damage",["native_missing","native_extra","gpu_missing","gpu_extra","native_duplicate","gpu_duplicate","native_world","gpu_world","native_addr","gpu_addr","slot","contact_addr","body","geom","ratio","native_ratio","nonfinite","bool"])
def test_incomplete_or_mislinked_own_partitions_refuse(damage):
    d,s,m=fixture()
    if damage=="native_missing":d["native_row_recipes"].pop()
    if damage=="native_extra":d["native_row_recipes"].append(deepcopy(d["native_row_recipes"][-1]))
    if damage=="gpu_missing":d["measured_gpu_row_recipes"].pop()
    if damage=="gpu_extra":d["measured_gpu_row_recipes"].append(deepcopy(d["measured_gpu_row_recipes"][-1]))
    if damage=="native_duplicate":d["native_row_recipes"][-1]=deepcopy(d["native_row_recipes"][-2])
    if damage=="gpu_duplicate":d["measured_gpu_row_recipes"][-1]=deepcopy(d["measured_gpu_row_recipes"][-2])
    if damage=="native_world":d["native_row_recipes"][-1]["world"]=0
    if damage=="gpu_world":d["measured_gpu_row_recipes"][0]["world"]=0
    if damage=="native_addr":d["native_row_recipes"][-1]["row"]=13
    if damage=="gpu_addr":d["measured_gpu_row_recipes"][0]["efc_address"]=13
    if damage=="slot":d["measured_gpu_row_recipes"][0]["slot"]=-1
    if damage=="contact_addr":d["measured_gpu_contact_inputs"][0]["efc_address"][0]=13
    if damage=="body":d["measured_gpu_row_recipes"][0]["bodies"]=[1,1]
    if damage=="geom":d["measured_gpu_row_recipes"][0]["geom"]=[0,79]
    if damage=="ratio":m["/model/opt/impratio_invsqrt"][0]=np.inf
    if damage=="native_ratio":s[1]["option/impratio"]["value"]=0.
    if damage=="nonfinite":d["native_row_recipes"][-1]["captured"]["D"]=math.inf
    if damage=="bool":d["native_row_recipes"][-1]["world"]=True
    with pytest.raises(ValueError):p.compare_rows(d,s,m)


@pytest.mark.parametrize("damage",["none","source","module","field","flag","extra","missing","row","conversion"])
def test_receiver_regenerates_every_complete_descriptor(tmp_path,monkeypatch,damage):
    d,s,m=fixture();result=dict(protocol=p.PROTOCOL,decision=p.DECISION,**p.compare_rows(d,s,m),**deepcopy(p.CONTRACT))
    monkeypatch.setattr(p,"source_check",lambda source:None)
    monkeypatch.setattr(p,"analyze",lambda *args:deepcopy(result))
    value=deepcopy(result);value.update(source="a"*40,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    if damage=="source":value["source"]="b"*40
    if damage=="module":value["module_sha256"]="0"*64
    if damage=="field":value["rows"][0]["warp_minus_C"]["D"]+=1
    if damage=="flag":value["training_authorized"]=True
    if damage=="extra":value["unbound"]=True
    if damage=="missing":value["rows"].pop()
    if damage=="row":value["rows"][0]["own_slot"]=2
    if damage=="conversion":value["rows"][0]["input_conversion"]="native ABI identity"
    path=tmp_path/"packet.json";path.write_text(json.dumps(value,allow_nan=False))
    if damage=="none":assert p.receive(path,"a"*40,None,None,None)==value
    else:
        with pytest.raises(ValueError):p.receive(path,"a"*40,None,None,None)


def test_import_and_help_inert():
    code="import sys;from mjlab_microduck import ada_own_manifold_recipe;assert not {'numpy','mujoco','mujoco_warp','warp','torch'}&sys.modules.keys()"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))
    subprocess.run([sys.executable,"-m",p.__name__,"--help"],capture_output=True,check=True,timeout=10)


@pytest.mark.parametrize("damage",[False,True])
def test_module_source_bytes_bound(monkeypatch,damage):
    monkeypatch.setattr(p.recipe,"source_check",lambda source:None)
    monkeypatch.setattr(p.host,"read",lambda *args,**kw:b"changed" if damage else Path(p.__file__).read_bytes())
    if damage:
        with pytest.raises(ValueError):p.source_check("a"*40)
    else:p.source_check("a"*40)
