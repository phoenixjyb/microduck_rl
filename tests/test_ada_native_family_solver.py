"""Synthetic native-family solver contracts; no actual simulator execution."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from mjlab_microduck import ada_native_family_solver as p
from test_ada_native_row_family_plan import fixture as family_fixture
from test_ada_same_bank_solver import reference
from test_ada_saved_pose_response_receiver import SOURCE, service


def hybrid_fixture():
    native,capture,before,layout,prior=family_fixture()
    for name,shape in (("J_rownnz",(2,0)),("J_rowadr",(2,0)),("J_colind",(2,0,0))):
        key="/data/efc/"+name
        dummy=next(k for k in before if "/padding" in k)
        before.pop(dummy);layout.pop(dummy);prior["child"]["input_manifest"].pop(dummy)
        value=np.zeros(shape,np.int32)
        width=4;strides=[]
        for n in reversed(shape): strides.insert(0,width);width*=n
        row=dict(dtype="<class 'warp._src.types.int32'>",shape=list(shape),strides=strides,bytes=0,sha256=sha256(b"").hexdigest())
        prior["child"]["input_manifest"][key]=row;layout[key]=p.response.expanded_layout(row);before[key]=np.zeros(0,np.uint8)
    hybrid,description=p.plan.encode(native,capture,before,layout,prior)
    return native,hybrid,layout,prior


def checked_service():
    row=service();row["unit"]=p.PREFIX+SOURCE[:8]+".service";return row


def test_import_and_help_are_inert():
    code="import sys;from mjlab_microduck import ada_native_family_solver;assert not {'numpy','warp','torch','mujoco','mujoco_warp'}&sys.modules.keys()"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=15)
    subprocess.run([sys.executable,"-m",p.__name__,"--help"],capture_output=True,check=True,timeout=15)


@pytest.mark.parametrize("field",list(p.response.SERVICE_CAPS)+["unit","MainPID","InvocationID","Environment","extra"])
def test_all_service_fields_are_closed(field):
    row=checked_service();p.validate_service(row,SOURCE,pid=123)
    if field=="unit": row[field]="old.service"
    else: row["properties"][field]="wrong"
    with pytest.raises(ValueError):p.validate_service(row,SOURCE,pid=123)


@pytest.mark.parametrize("damage",["none","sparse","solver","cone","disable","enable","modeldisable","modelenable","noslip","capacity","type","count","nonfinite","sparsemetadata"])
def test_branch_guard_excludes_other_solver_paths(damage):
    _,hybrid,layout,prior=hybrid_fixture()
    native=SimpleNamespace(opt=SimpleNamespace(solver=2,cone=0,disableflags=0,enableflags=0,noslip_iterations=0))
    model=SimpleNamespace(is_sparse=False,opt=SimpleNamespace(solver=2,cone=0,disableflags=0,enableflags=0))
    data=SimpleNamespace(nworld=2,naconmax=256,njmax=512)
    if damage=="sparse":model.is_sparse=True
    if damage=="solver":model.opt.solver=1
    if damage=="cone":model.opt.cone=1
    if damage=="disable":native.opt.disableflags=1
    if damage=="enable":native.opt.enableflags=1
    if damage=="modeldisable":model.opt.disableflags=1
    if damage=="modelenable":model.opt.enableflags=1
    if damage=="noslip":native.opt.noslip_iterations=1
    if damage=="capacity":data.njmax=513
    if damage=="type":hybrid["/data/efc/type"].view(np.int32)[0]=7
    if damage=="count":hybrid["/data/nefc"].view(np.int32)[1]=31
    if damage=="nonfinite":hybrid["/data/qacc"].view(np.float32)[0]=np.nan
    if damage=="sparsemetadata":layout["/data/efc/J_rownnz"]["numpy_shape"]=[0,2]
    if damage=="none":assert p.branch_guard(native,model,data,hybrid,layout,prior)["source_path_only"]
    else:
        with pytest.raises(ValueError):p.branch_guard(native,model,data,hybrid,layout,prior)


def fixture(tmp_path,monkeypatch):
    native,hybrid,layout,prior=hybrid_fixture()
    after={k:v.copy() for k,v in hybrid.items()};after["/data/qacc"].view(np.float32)[0]=.5
    raw={"after_solve"+k:v.copy() for k,v in hybrid.items()}
    child=dict(plant=prior["child"]["plant"],options={"synthetic":True},model_array_sha256={"model/"+str(i):"a"*64 for i in range(347)})
    manifest={"synthetic":"all184"};ref=reference();h={"synthetic":"complete114"}
    monkeypatch.setattr(p,"source_check",lambda source:None)
    monkeypatch.setattr(p,"references",lambda *args:(h,hybrid,{"child":child},raw,layout,manifest,prior,{},ref,{},hybrid,native))
    audit={"frozen":True};expansion={"frozen":"expansion"}
    monkeypatch.setattr(p.response.source,"audit",lambda:audit)
    monkeypatch.setattr(p.allocation,"expansion_source",lambda:expansion)
    arrays={b+k:v.copy() for b,bank in zip(p.BRANCHES,(hybrid,after)) for k,v in bank.items()}
    value=dict(deepcopy(p.CONTRACT),protocol=p.PROTOCOL,decision=p.DECISION,source=SOURCE,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest(),versions=deepcopy(p.same.VERSIONS),source_audit_sha256=p.packet.digest(audit),
        plan_source=p.PLAN_SOURCE,plan_report_sha256=p.PLAN_SHA,plan_payload_sha256=p.PLAN_NPZ_SHA,plan_descriptor_sha256=p.packet.digest(h),baseline_source=p.BASE_SOURCE,baseline_report_sha256=p.BASE_SHA,baseline_payload_sha256=p.BASE_NPZ_SHA,gpu_report_sha256=p.same.GPU_REPORT_SHA,provenance_reference_sha256=p.same.CPU_RESPONSE_SHA,
        private_cache_was_absent=True,private_cache_files={"module.o":dict(bytes=4,sha256="a"*64)},existing_cpu_executables=[deepcopy(ref[k]) for k in ("repeat_array_kernel_39317a34","mujoco_warp._src.solver")],running_service=checked_service(),protected_services={s+":"+n:"inactive" for s in ("system","user") for n in p.host.SERVICES},foreign_compute_processes=deepcopy(p.same.measured.FOREIGN),
        comparison=p.compare_outputs(after,dict(measured_gpu=hybrid,original_gpu_bank_cpu_reference=hybrid),layout,prior,native),plant=child["plant"],options=child["options"],model_array_sha256=child["model_array_sha256"],static_manifest=manifest,static_manifest_sha256=p.same.measured.STATIC_SHA,data_layout=layout,restore_sha256={k:sha256(v.tobytes()).hexdigest() for k,v in hybrid.items()},branch_guard=dict(model_is_sparse=False,solver=2,cone=0,disableflags=0,enableflags=0,noslip_iterations=0,empty_dense_sparse_metadata=True,active_row_types=[1,6],source_path_only=True,machine_code_read_set_proved=False),solver_changed_data_fields=p.response.solver_write_fence(hybrid,after),expansion_source=expansion,pre_solve_executables=[dict(deepcopy(p.allocation.EXPECTED_REPEAT),opaque_handle="wp_repeat_array_kernel_39317a34_1")],in_process_binding_sha256="c"*64,in_process_binding_convention=p.response.STATIC_CONVENTION,cpu_counters={k:p.plan.full_bank(after,layout,prior)["/data/"+k].tolist() for k in p.saved.p.COUNTERS+("nacon","ncollision")})
    path=tmp_path/"response.json"
    def write():
        payload=path.with_suffix(".npz")
        if payload.exists():payload.unlink() # Only test-owned temporary file.
        value["payload"]=p.saved.p.retain(payload,arrays)
        path.write_text(json.dumps(value,allow_nan=False))
    write();return path,value,arrays,write


def test_complete_portable_receiver_and_private_cache(tmp_path,monkeypatch):
    path,value,arrays,_=fixture(tmp_path,monkeypatch)
    received,actual=p.receive(path,SOURCE,tmp_path,tmp_path,tmp_path)
    assert received==value and len(actual)==228
    assert value["comparison"]["complete_solver_outputs"]["measured_gpu"]["/data/qacc"]["hybrid_minus_reference_max_abs"]==.5
    monkeypatch.setattr(p.saved,"cache_files",lambda root:value["private_cache_files"])
    p.receive(path,SOURCE,tmp_path,tmp_path,tmp_path,cache=tmp_path)
    monkeypatch.setattr(p.saved,"cache_files",lambda root:{})
    with pytest.raises(ValueError):p.receive(path,SOURCE,tmp_path,tmp_path,tmp_path,cache=tmp_path)


@pytest.mark.parametrize("field",["extra","source","module_sha256","versions","source_audit_sha256","plan_source","plan_report_sha256","plan_payload_sha256","plan_descriptor_sha256","baseline_source","baseline_report_sha256","baseline_payload_sha256","gpu_report_sha256","provenance_reference_sha256","plant","options","model_array_sha256","static_manifest","static_manifest_sha256","data_layout","restore_sha256","branch_guard","solver_changed_data_fields","expansion_source","cpu_counters","protected_services","foreign_compute_processes","flags","comparison","payload","leaf","missing","input","nonsolver","dtype","nonfinite"])
def test_complete_receiver_refuses_damage(tmp_path,monkeypatch,field):
    path,v,a,write=fixture(tmp_path,monkeypatch)
    if field=="extra":v[field]=True
    elif field=="payload":v["payload"]["sha256"]="0"*64
    elif field=="leaf":v["payload"]["arrays"]["cpu_after_solve/data/qacc"]["sha256"]="0"*64
    elif field=="missing":a.pop("cpu_after_solve/data/qacc")
    elif field=="input":a["restored_hybrid/data/qacc"][0]^=1
    elif field=="nonsolver":a["cpu_after_solve/data/qpos"][0]^=1
    elif field=="dtype":a["cpu_after_solve/data/qacc"]=a["cpu_after_solve/data/qacc"].astype(np.int32)
    elif field=="nonfinite":
        a["cpu_after_solve/data/qacc"].view(np.float32)[0]=np.nan
        payload=path.with_suffix(".npz");payload.unlink();np.savez(payload,**a)
        raw=payload.read_bytes();v["payload"].update(bytes=len(raw),sha256=sha256(raw).hexdigest())
        v["payload"]["arrays"]["cpu_after_solve/data/qacc"]["sha256"]=sha256(a["cpu_after_solve/data/qacc"].tobytes()).hexdigest()
    elif field in {"versions","plant","options","model_array_sha256","static_manifest","data_layout","restore_sha256","branch_guard","expansion_source","cpu_counters","protected_services","flags","comparison"}:v[field]={}
    elif field in {"foreign_compute_processes","solver_changed_data_fields"}:v[field]=["wrong"]
    else:v[field]="wrong"
    if field in {"missing","input","nonsolver","dtype"}:write()
    path.write_text(json.dumps(v,allow_nan=False))
    with pytest.raises(ValueError):p.receive(path,SOURCE,tmp_path,tmp_path,tmp_path)


@pytest.mark.parametrize("damage",["duplicate","missing","extra","oversized"])
def test_raw_archive_preflight(tmp_path,damage):
    path=tmp_path/"bank.npz"
    np.savez(path,one=np.zeros(1,np.uint8))
    if damage=="duplicate":
        with zipfile.ZipFile(path,"a") as z:
            with pytest.warns(UserWarning):z.writestr("one.npy",z.read("one.npy"))
    if damage=="extra":
        with zipfile.ZipFile(path,"a") as z:z.writestr("extra.npy",b"extra")
    if damage=="oversized":np.savez(path,one=np.zeros(2*1024**2,np.uint8))
    with pytest.raises(ValueError):p.raw_npz(path,{"missing"} if damage=="missing" else {"one"},4*1024**2)


def test_runtime_single_solve_no_other_fixture_call_and_fresh_receiver():
    tree=ast.parse(Path(p.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
    calls=[ast.unparse(n.func) for n in sorted((n for n in ast.walk(run) if isinstance(n,ast.Call)),key=lambda n:n.lineno)]
    assert calls.count("solver.solve")==calls.count("packet.restore_complete_data")==1
    assert calls.index("packet.restore_complete_data")<calls.index("solver.solve")
    assert calls.count("branch_guard")==2
    assert not set(calls)&{"mjwarp.forward","mujoco.mj_forward","smooth.kinematics","smooth.factor_m","collision_driver.collision","constraint.make_constraint","solver.create_solver_context"}
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    assert "receive" not in [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n,ast.Call)]


@pytest.mark.parametrize("damage",[False,True])
def test_archival_module_bytes_not_just_old_revision_label(monkeypatch,damage):
    monkeypatch.setattr(p.host,"read",lambda *args,**kw:b"changed" if damage else Path(p.plan.__file__).read_bytes())
    if damage:
        with pytest.raises(ValueError):p.archived_module(p.plan,p.PLAN_SOURCE)
    else:assert p.archived_module(p.plan,p.PLAN_SOURCE)==sha256(Path(p.plan.__file__).read_bytes()).hexdigest()
