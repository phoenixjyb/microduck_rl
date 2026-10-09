"""Synthetic-only numeric projection contract tests; never create native Data."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from mjlab_microduck import ada_native_phase_projection as p


def fixture(world):
    arrays={k:np.zeros(shape,dtype=dtype) for k,(dtype,shape) in p.layout(world).items()}
    ncon,nefc=(0,14) if world==0 else (4,30)
    counts=dict(ncon=ncon,ne=0,nf=14,nl=0,nefc=nefc,nJ=20*nefc,nA=0,nisland=0)
    arrays["data/counters"][:]=[counts[k] for k in p.native.COUNTERS]
    arrays["efc/type"][:14]=1;arrays["efc/id"][:14]=np.arange(6,20)
    for slot in range(ncon):
        addr=14+4*slot
        arrays["contact/dim"][slot]=3;arrays["contact/efc_address"][slot]=addr
        arrays["contact/geom"][slot]=[0,29 if slot<2 else 79]
        arrays["efc/type"][addr:addr+4]=6;arrays["efc/id"][addr:addr+4]=slot
    states={k:arrays["fields/"+k].copy() for k in p.native.STATE}
    return arrays,states


@pytest.mark.parametrize("world",[0,1])
def test_complete_all_leaf_projection_no_mutation_or_authentication(world):
    arrays,states=fixture(world);copy=deepcopy(arrays)
    r=p.validate(arrays,world,states)
    assert r["projected_fields"]==len(arrays)==len(p.layout(world))
    assert not r["producer_authenticated"] and not r["native_phase_identity_established"]
    assert all(a.tobytes()==copy[k].tobytes() for k,a in arrays.items())
    assert all(k in r["arrays"] for k in p.layout(world))


@pytest.mark.parametrize("world",[0,1])
@pytest.mark.parametrize("key",sorted(p.layout(1)))
def test_every_leaf_is_required(world,key):
    a,s=fixture(world);a.pop(key)
    with pytest.raises(ValueError):p.validate(a,world,s)


@pytest.mark.parametrize("key",sorted(p.native.STATE))
def test_every_input_state_byte_is_required(key):
    a,s=fixture(1);a["fields/"+key][0]=1.
    with pytest.raises(ValueError):p.validate(a,1,s)


@pytest.mark.parametrize("damage",["extra","dtype","shape","nan","infinite","counter","nefc","nJ","nA","island","warning","warning_info","iteration","negative_iter","negative_nnz","type","id","duplicate_id","dim","addr","contact_type","contact_id","geom","state_missing","state_bool","state_shape","state_nan"])
def test_closed_domain_invalid_leaves_refuse(damage):
    a,s=fixture(1)
    if damage=="extra":a["unknown"]=np.zeros(1)
    if damage=="dtype":a["efc/D"]=a["efc/D"].astype(np.float32)
    if damage=="shape":a["efc/J"]=a["efc/J"].reshape(30,20)
    if damage=="nan":a["fields/qacc"][0]=np.nan
    if damage=="infinite":a["efc/force"][0]=np.inf
    if damage in ("counter","nefc","nJ","nA","island"):
        key={"counter":"nf","nefc":"nefc","nJ":"nJ","nA":"nA","island":"nisland"}[damage]
        a["data/counters"][p.native.COUNTERS.index(key)]={"nA":901,"nisland":21}.get(key,99)
    if damage=="warning":a["data/warning/number"][0]=1
    if damage=="warning_info":a["data/warning/lastinfo"][0]=-1
    if damage=="iteration":a["data/solver_niter"][0]=201
    if damage=="negative_iter":a["data/solver_niter"][0]=-1
    if damage=="negative_nnz":a["data/solver_nnz"][0]=-1
    if damage=="type":a["efc/type"][0]=6
    if damage=="id":a["efc/id"][0]=20
    if damage=="duplicate_id":a["efc/id"][0]=a["efc/id"][1]
    if damage=="dim":a["contact/dim"][0]=4
    if damage=="addr":a["contact/efc_address"][0]=15
    if damage=="contact_type":a["efc/type"][14]=1
    if damage=="contact_id":a["efc/id"][14]=1
    if damage=="geom":a["contact/geom"][0]=[29,0]
    if damage=="state_missing":s.pop("time")
    if damage=="state_bool":s["time"]=np.zeros(1,dtype=bool)
    if damage=="state_shape":s["qpos"]=np.zeros((1,21))
    if damage=="state_nan":s["time"][0]=np.nan
    with pytest.raises(ValueError):p.validate(a,1,s)


@pytest.mark.parametrize("world",[False,1.,-1,2,"1"])
def test_world_is_plain_two_world_index(world):
    with pytest.raises(ValueError):p.layout(world)


def test_all_leaf_comparison_keeps_changes_without_tolerance_or_promotion():
    before,s=fixture(1);after=deepcopy(before)
    after["efc/D"][0]=1e-15;after["efc/force"][14]=2.
    after["data/solver_niter"][0]=4
    r=p.compare(before,after,1,s)
    assert set(r["differences"])==set(p.layout(1))
    assert r["differences"]["efc/D"]["changed_values"]==1
    assert r["differences"]["efc/force"]["max_abs"]==2.
    assert not r["training_authorized"] and not r["execution_ready"]
    assert r["no_assumption_inputs_survive_solve"]


def test_zero_contact_world_retains_every_empty_leaf():
    a,s=fixture(0);r=p.compare(a,deepcopy(a),0,s)
    assert all(r["differences"]["contact/"+k]["max_abs"]==0 and a["contact/"+k].size==0 for k in p.native.CONTACT)


@pytest.mark.parametrize("key",["J_rownnz","J_rowadr","J_rowsuper","J_colind"])
def test_dense_unused_sparse_metadata_is_retained_not_reinterpreted(key):
    a,s=fixture(1);b=deepcopy(a)
    b["efc/"+key][0]=19
    result=p.compare(a,b,1,s)
    assert result["differences"]["efc/"+key]["changed_values"]==1
    assert not result["before"]["sparse_metadata_interpreted"]


def test_import_and_fixture_checks_have_no_simulator_runtime():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    code='import sys; from mjlab_microduck import ada_native_phase_projection as p; assert not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(); assert not p.CONTRACT["actual_native_capture_executed"]'
    r=subprocess.run([sys.executable,"-c",code],env=env,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr


def test_retained_adapter_keeps_exact_every_leaf_owned_bytes():
    a,s=fixture(1);bank={}
    for k,v in a.items():
        category,name=k.split("/",1)
        path=k if category=="fields" else category+"/1/"+name
        bank[path]=np.stack([v,v]) if category=="fields" else v.copy()
    decoded=p.retained_post_forward_projection(bank,1)
    assert set(decoded)==set(a)
    assert all(v.tobytes()==a[k].tobytes() and v.flags.owndata and not np.shares_memory(v,bank[k if k.startswith("fields/") else k.split("/",1)[0]+"/1/"+k.split("/",1)[1]]) for k,v in decoded.items())
    assert not p.validate(decoded,1,s)["producer_authenticated"]
    bank.pop("efc/1/J")
    with pytest.raises(ValueError):p.retained_post_forward_projection(bank,1)


def test_retained_analysis_refuses_runtime_before_any_file_read(monkeypatch):
    monkeypatch.setitem(sys.modules,"warp",object())
    with pytest.raises(ValueError,match="fresh pure"):
        p.analyze_retained("absent","absent","absent")


@pytest.mark.parametrize("damage",["admission","stage","missing","extra","source","module"])
def test_whole_receiver_refuses_partial_or_relabelled_projection(tmp_path,monkeypatch,damage):
    reference=dict(protocol=p.PROTOCOL,decision=p.DECISION,stage="retained_post_forward_only",records=[],**p.CONTRACT)
    monkeypatch.setattr(p,"source_check",lambda source:None)
    monkeypatch.setattr(p,"analyze_retained",lambda *args:deepcopy(reference))
    value=deepcopy(reference);value.update(source="a"*40,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    f=tmp_path/"report.json";f.write_text(json.dumps(value))
    assert p.receive(f,"a"*40,"i","m","t")==value
    if damage=="admission":value["native_phase_identity_established"]=True
    if damage=="stage":value["stage"]="before_solve"
    if damage=="missing":value.pop("records")
    if damage=="extra":value["unknown"]=0
    if damage=="source":value["source"]="b"*40
    if damage=="module":value["module_sha256"]="0"*64
    f.write_text(json.dumps(value))
    with pytest.raises(ValueError):p.receive(f,"a"*40,"i","m","t")
