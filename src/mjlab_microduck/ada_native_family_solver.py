"""One bounded CPU solve of the native row-family hybrid context.

Not a native phase replay, point-matched manifold, cause isolation, simulator
admission, or training. Older exact-source receivers remain unchanged; the
explicit archival bindings below recompute the two immutable reference banks.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

from mjlab_microduck import ada_native_row_family_plan as plan
from mjlab_microduck import ada_same_bank_solver as same

need,host,packet,response,saved,allocation = same.need,same.host,same.packet,same.response,same.saved,same.allocation
PROTOCOL = "microduck-current-native-family-cpu-solver-only-v1"
DECISION = "native-family-cpu-hybrid-response-collected-not-cause-isolation-or-admission"
PREFIX = "microduck-native-family-solver-"
PLAN_SOURCE = "c9b37b5496fc4b809c65fb832263c78f3f17bbdd"
PLAN_SHA = "585d728f420cc1ef7bc4a9bcf3ab9d9b3682cfeb8c8f96d97b4d0df26b6bfcb2"
PLAN_NPZ_SHA = "9a885cd219762de5345139438cb7f3b7e9581bbcbecefaa3f386bb9272023285"
BASE_SOURCE = "d75ef0cde0be3198e2f0372c4d5f9fe0aa17825d"
BASE_SHA = "b0287a679ad297dfb40a71a6c03a2e99a3227d298da109edcdde3b9e131c0339"
BASE_NPZ_SHA = "8193be85083cb50e031a3fa95238daa3d6f7eeb2fa36f60c93e5074e3406cb40"
BRANCHES = ("restored_hybrid", "cpu_after_solve")
CONTRACT = dict(allocation_native_kinematics_calls=1,source_expected_model_expansion_launches=2,
    fixture_native_kinematics_calls=0,fixture_warp_kinematics_calls=0,fixture_native_forward_calls=0,
    collision_calls=0,constraint_construction_calls=0,pre_solve_mass_factorization_calls=0,
    sensor_stage_calls=0,force_decoder_calls=0,ordinary_forward_calls=0,public_solver_calls=1,
    integration_steps=0,optimizer_steps=0,restored_actual_data_arrays=114,
    input_family_overrides=31,input_context_held=83,actual_model_arrays_bound=347,actual_static_fields_bound=184,
    native_final_solver_outputs_transplanted=False,native_capture_is_before_solve=False,
    native_phase_replay_executed=False,native_end_to_end_context_identity=False,
    physical_contact_point_matching=False,synthetic_collision_metadata=True,
    source_branch_guard_checked=True,machine_code_read_set_proved=False,
    exclusive_cause_established=False,identical_kernel_execution_claimed=False,
    loaded_binary_bytes_bound=False,solver_qualified=False,simulator_qualified=False,
    training_authorized=False,physical_motion_authorized=False,flags=host.FLAGS)


def archived_module(module, source):
    raw = host.read("git","show",source+":src/mjlab_microduck/"+Path(module.__file__).name,binary=True)
    need(Path(module.__file__).read_bytes()==raw,"unchanged immutable archival module bytes")
    return sha256(raw).hexdigest()


def raw_npz(path, keys, limit):
    import numpy as np
    raw=Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows=archive.infolist()
        need(len(rows)==len({r.filename for r in rows})==len(keys) and all(0<=r.file_size<2*1024**2 for r in rows) and sum(r.file_size for r in rows)<limit+len(keys)*1024,"bounded complete unique archive before decoding")
    with np.load(io.BytesIO(raw),allow_pickle=False) as bank:
        need(len(bank.files)==len(keys) and set(bank.files)==set(keys),"closed complete archival payload keys")
        return {k:bank[k].copy() for k in keys}


def archived_plan(tools,input_root,measured_root):
    """Explicit immutable-source binding; no bypass of the old live receiver."""
    path=Path(tools)/"c9b37b54-linux-native-family.json"
    same.exact_file(path,PLAN_SHA,256*1024);same.exact_file(path.with_suffix(".npz"),PLAN_NPZ_SHA,2*1024**2)
    module=archived_module(plan,PLAN_SOURCE)
    expected,arrays=plan.analyze(input_root,measured_root,Path(tools)/"f8540785-linux-native-constraint.json",
        Path(tools)/"mujoco-28009f91-engine-core-constraint.c",Path(tools)/"ec81e2bb-linux-native-recipe.json")
    raw=path.with_suffix(".npz").read_bytes()
    expected.update(source=PLAN_SOURCE,module_sha256=module,payload=dict(file=path.with_suffix(".npz").name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    actual=json.loads(path.read_bytes())
    need(packet.digest(actual)==packet.digest(expected),"every immutable hybrid descriptor completely recomputed")
    payload=raw_npz(path.with_suffix(".npz"),arrays,2*1024**2)
    response.exact_values(payload,arrays,"every immutable114-field hybrid payload leaf")
    return actual,arrays


def archived_baseline(tools,gpu,gpu_arrays,layout,manifest,prior,reference):
    import numpy as np
    path=Path(tools)/"d75ef0cd-linux-same-bank-solver.json"
    same.exact_file(path,BASE_SHA,256*1024);same.exact_file(path.with_suffix(".npz"),BASE_NPZ_SHA,2*1024**2)
    module=archived_module(same,BASE_SOURCE);v=json.loads(path.read_bytes())
    need(v["source"]==BASE_SOURCE and v["module_sha256"]==module and v["protocol"]==same.PROTOCOL and v["decision"]==same.DECISION and packet.digest({k:v[k] for k in same.CONTRACT})==packet.digest(same.CONTRACT),"immutable unqualified baseline source/contract")
    need(v["source_audit_sha256"]==packet.digest(response.source.audit()) and v["model_array_sha256"]==gpu["child"]["model_array_sha256"] and v["plant"]==gpu["child"]["plant"] and v["options"]==gpu["child"]["options"] and v["static_manifest"]==manifest and v["static_manifest_sha256"]==same.measured.STATIC_SHA and v["data_layout"]==layout,"complete baseline source/model/static/layout matches this context")
    same.validate_executables(v["existing_cpu_executables"],reference);allocation.repeat_executables(v["pre_solve_executables"]);same.validate_service(v["running_service"],BASE_SOURCE)
    keys={branch+k for branch in same.BRANCHES for k in layout}
    arrays=raw_npz(path.with_suffix(".npz"),keys,4*1024**2)
    need(set(v["payload"]["arrays"])==keys,"complete baseline leaf manifest")
    for k,a in arrays.items():
        need(v["payload"]["arrays"][k]==dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha256(a.tobytes()).hexdigest()),"all228 baseline leaf bytes bound")
    before,after=[{k:arrays[b+k] for k in layout} for b in same.BRANCHES]
    expected={k:gpu_arrays["before_solve"+k] for k in layout};response.exact_values(before,expected,"all114 baseline measured-context inputs")
    gpu_after={k:gpu_arrays["after_solve"+k] for k in layout}
    need(packet.compare_solver_banks(before,gpu_after,after,layout,prior)==v["comparison"] and v["restore_sha256"]=={k:sha256(a.tobytes()).hexdigest() for k,a in before.items()},"complete baseline outputs/fences independently recomputed")
    counters={k:np.frombuffer(after["/data/"+k].tobytes(),np.int32).tolist() for k in saved.p.COUNTERS+("nacon","ncollision")}
    need(counters==v["cpu_counters"],"complete baseline counter receipt")
    return v,after


def branch_guard(native,model,data,raw,layout,prior):
    """Frozen dense/Newton/pyramidal source path only, not binary proof."""
    import numpy as np
    need(not model.is_sparse and int(model.opt.solver)==2 and int(model.opt.cone)==0 and
        int(model.opt.disableflags)==0 and int(model.opt.enableflags)==0 and
        int(native.opt.solver)==2 and int(native.opt.cone)==0 and int(native.opt.disableflags)==0 and
        int(native.opt.enableflags)==0 and native.opt.noslip_iterations==0,"dense Newton pyramidal no-override/no-noslip native-family branch")
    bank=plan.full_bank(raw,layout,prior)
    for name,shape in (("J_rownnz",(2,0)),("J_rowadr",(2,0)),("J_colind",(2,0,0))):
        a=bank["/data/efc/"+name];need(a.dtype==np.int32 and a.shape==shape and a.nbytes==0,"authentic empty dense sparse metadata")
    need(bank["/data/ne"].tolist()==[0,0] and bank["/data/nf"].tolist()==[14,14] and bank["/data/nl"].tolist()==[0,0] and bank["/data/nefc"].tolist()==[14,30] and bank["/data/nacon"].tolist()==[4],"actual fixed coupled hybrid count family")
    for w,n in enumerate((14,30)):
        need(bank["/data/efc/type"][w,:n].tolist()==[1]*14+[6]*(n-14),"only expected active friction/pyramid kinds")
    need(data.nworld==2 and data.naconmax==256 and data.njmax==512,"unchanged actual hybrid capacities")
    return dict(model_is_sparse=False,solver=2,cone=0,disableflags=0,enableflags=0,noslip_iterations=0,
        empty_dense_sparse_metadata=True,active_row_types=[1,6],source_path_only=True,machine_code_read_set_proved=False)


def compare_outputs(after,references,layout,prior,native_arrays):
    """Full allocated outputs, then native generalized fields; no tolerance."""
    decoded=plan.full_bank(after,layout,prior);result={}
    for name,bank in references.items():
        other=plan.full_bank(bank,layout,prior);fields={}
        for k in sorted(response.SOLVER_OUTPUTS):
            left,right=decoded[k],other[k]
            deltas=[float(a)-float(b) for a,b in zip(left.flat,right.flat)]
            fields[k]=dict(shape=list(left.shape),hybrid_sha256=sha256(left.tobytes()).hexdigest(),reference_sha256=sha256(right.tobytes()).hexdigest(),bytes_equal=left.tobytes()==right.tobytes(),hybrid_minus_reference_max_abs=max(map(abs,deltas),default=0.),differing_scalar_elements=sum(a.tobytes()!=b.tobytes() for a,b in zip(left.flat,right.flat)))
        result[name]=fields
    native={}
    for name in ("qacc","qfrc_constraint"):
        left,right=decoded["/data/"+name],native_arrays["fields/"+name]
        need(left.shape==right.shape==(2,20),"complete native generalized output topology")
        native[name]=dict(hybrid_minus_current_native_max_abs=max(abs(float(a)-float(b)) for a,b in zip(left.flat,right.flat)),per_world_max_abs=[max(abs(float(a)-float(b)) for a,b in zip(left[w],right[w])) for w in range(2)],native_sha256=sha256(right.tobytes()).hexdigest(),native_dtype=str(right.dtype),hybrid_dtype=str(left.dtype))
    return dict(complete_solver_outputs=result,current_native_generalized_fields=native,
        convention="Hybrid minus each reference, full allocated six outputs; native fields current F64 versus hybrid F32; no point matching/tolerance/cause gate")


def validate_service(service,source,pid=None):
    need(service["unit"]==PREFIX+source[:8]+".service","closed native-family solver unit")
    response.validate_service(dict(service,unit="microduck-ada-response-"+source[:8]+".service"),source,pid=pid)


def unit_identity(source):
    props=tuple(response.SERVICE_CAPS)+("MainPID","InvocationID","Environment")
    unit=PREFIX+source[:8]+".service"
    row=dict(s.split("=",1) for s in host.read("systemctl","--user","show",unit,*["--property="+k for k in props]).splitlines())
    result=dict(unit=unit,properties=row);validate_service(result,source,os.getpid());return result


def run(prior,banks,hybrid,layout,manifest,gpu_child):
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp
    from mujoco_warp._src import solver
    from mjlab.sim import randomization
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_forward_graph import binding
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and all(d.is_cpu for d in wp.get_devices()) and not saved.passive_executables(),"fresh CPU-only native-family solver allocation")
    expansion=allocation.expansion_callable(randomization.expand_model_fields,randomization.__file__)
    native=build_entity().compile();plant=saved.p.bind_plant(native,prior["child"]["plant"])
    for name in ("dof_frictionloss","dof_damping"): getattr(native,name)[:]=banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice("cpu"):
        model=mjwarp.put_model(native);data=mjwarp.make_data(native,nworld=2,nconmax=128,naconmax=256,naccdmax=256,njmax=512,njmax_nnz=10240)
        options=response.topology_guard(native,model,data)
        randomization.expand_model_fields(model,2,["dof_frictionloss","dof_damping"])
        for name in ("dof_frictionloss","dof_damping"): wp.copy(getattr(model,name),wp.array(banks["motor.npz"][name],dtype=wp.float32,device="cpu"))
        wp.synchronize_device("cpu");same.cpu_device(model,data)
        model_hashes=saved.p.model_binding(model,prior,banks)
        need(model_hashes==gpu_child["model_array_sha256"] and plant==gpu_child["plant"] and options==gpu_child["options"],"all347 actual model/plant/options match measured GPU context")
        static_sha=packet.bind_static_inventory(manifest,prior,value=dict(model=model,data=data),array_type=wp.array)
        allocation.frozen_static_schema(manifest);need(static_sha==same.measured.STATIC_SHA,"all184 actual hybrid statics equal measured GPU")
        allocation.repeat_executables(saved.passive_executables());pointers=binding((model,data))
        restore=packet.restore_complete_data(data,hybrid,layout,prior,wp,model=model,static_manifest=manifest)
        restored,actual_layout=response.snapshot_data(data,prior);response.exact_values(restored,hybrid,"all114 actual restored hybrid fields")
        branch=branch_guard(native,model,data,restored,layout,prior)
        need(actual_layout==layout and response.topology_guard(native,model,data)==options and saved.p.model_binding(model,prior,banks)==model_hashes and binding((model,data))==pointers,"complete actual restored hybrid boundary unchanged")
        before_executables=saved.passive_executables();allocation.repeat_executables(before_executables)
        solver.solve(model,data)
        wp.synchronize_device("cpu");same.cpu_device(model,data)
        after,actual_layout=response.snapshot_data(data,prior)
        need(actual_layout==layout and response.topology_guard(native,model,data)==options and saved.p.model_binding(model,prior,banks)==model_hashes and binding((model,data))==pointers and packet.bind_static_inventory(manifest,prior,value=dict(model=model,data=data),array_type=wp.array)==static_sha and allocation.expansion_source()==expansion,"complete actual hybrid bindings unchanged after single solve")
        branch_guard(native,model,data,after,layout,prior);plan.full_bank(after,layout,prior)
        changed=response.solver_write_fence(restored,after)
        need(((data.solver_niter.numpy()>=0)&(data.solver_niter.numpy()<=native.opt.iterations)).all(),"bounded unchanged actual hybrid iterations")
    return dict(plant=plant,options=options,model_array_sha256=model_hashes,static_manifest=manifest,static_manifest_sha256=static_sha,data_layout=layout,restore_sha256=restore,branch_guard=branch,solver_changed_data_fields=changed,expansion_source=expansion,pre_solve_executables=before_executables,in_process_binding_sha256=sha256(repr(pointers).encode()).hexdigest(),in_process_binding_convention=response.STATIC_CONVENTION,cpu_counters={k:getattr(data,k).numpy().tolist() for k in saved.p.COUNTERS+("nacon","ncollision")}),{b+k:v for b,bank in zip(BRANCHES,(restored,after)) for k,v in bank.items()}


def references(tools,input_root,measured_root):
    hybrid_report,hybrid=archived_plan(tools,input_root,measured_root)
    gpu,gpu_arrays,layout,manifest,_=same.gpu_packet(measured_root,input_root)
    prior,banks=saved.p.prior.authenticated_banks(input_root);ref=same.provenance_reference(input_root)
    baseline,old_after=archived_baseline(tools,gpu,gpu_arrays,layout,manifest,prior,ref)
    capture,native_arrays=plan.recipe.native.receive(Path(tools)/"f8540785-linux-native-constraint.json",plan.recipe.NATIVE_SOURCE,input_root)
    return hybrid_report,hybrid,gpu,gpu_arrays,layout,manifest,prior,banks,ref,baseline,old_after,native_arrays


def source_check(source):
    same.receiver_source_check(source);plan.source_check(source)
    need(Path(__file__).read_bytes()==host.read("git","show",source+":src/mjlab_microduck/ada_native_family_solver.py",binary=True),"exact committed native-family solver module")


def receive(output,source,tools,input_root,measured_root,cache=None):
    import numpy as np
    source_check(source);output=Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size<=256*1024,"plain bounded complete hybrid solver report")
    value=json.loads(output.read_bytes())
    h,hybrid,gpu,raw,layout,manifest,prior,_,ref,_,old_after,native_arrays=references(tools,input_root,measured_root)
    keys={"protocol","decision","source","module_sha256","versions","source_audit_sha256","plan_source","plan_report_sha256","plan_payload_sha256","plan_descriptor_sha256","baseline_source","baseline_report_sha256","baseline_payload_sha256","gpu_report_sha256","provenance_reference_sha256","private_cache_was_absent","private_cache_files","existing_cpu_executables","payload","running_service","protected_services","foreign_compute_processes","comparison","plant","options","model_array_sha256","static_manifest","static_manifest_sha256","data_layout","restore_sha256","branch_guard","solver_changed_data_fields","expansion_source","pre_solve_executables","in_process_binding_sha256","in_process_binding_convention","cpu_counters"}|set(CONTRACT)
    need(set(value)==keys and value["source"]==source and value["protocol"]==PROTOCOL and value["decision"]==DECISION and packet.digest({k:value[k] for k in CONTRACT})==packet.digest(CONTRACT),"closed unqualified hybrid solver contract")
    need(value["module_sha256"]==sha256(Path(__file__).read_bytes()).hexdigest() and value["source_audit_sha256"]==packet.digest(response.source.audit()) and value["expansion_source"]==allocation.expansion_source() and set(value["versions"])==set(same.VERSIONS) and all(value["versions"][k].split("+")[0]==v for k,v in same.VERSIONS.items()),"exact executed module/frozen packages and sources")
    need(value["plan_source"]==PLAN_SOURCE and value["plan_report_sha256"]==PLAN_SHA and value["plan_payload_sha256"]==PLAN_NPZ_SHA and value["plan_descriptor_sha256"]==packet.digest(h) and value["baseline_source"]==BASE_SOURCE and value["baseline_report_sha256"]==BASE_SHA and value["baseline_payload_sha256"]==BASE_NPZ_SHA and value["gpu_report_sha256"]==same.GPU_REPORT_SHA and value["provenance_reference_sha256"]==same.CPU_RESPONSE_SHA,"all exact immutable intervention/control references")
    need(value["plant"]==gpu["child"]["plant"] and value["options"]==gpu["child"]["options"] and value["model_array_sha256"]==gpu["child"]["model_array_sha256"] and value["static_manifest"]==manifest and value["static_manifest_sha256"]==same.measured.STATIC_SHA and value["data_layout"]==layout and value["in_process_binding_convention"]==response.STATIC_CONVENTION and re.fullmatch(r"[0-9a-f]{64}",value["in_process_binding_sha256"]),"all actual complete measured context bindings retained")
    saved.provenance_layout(value);same.validate_executables(value["existing_cpu_executables"],ref);allocation.repeat_executables(value["pre_solve_executables"]);validate_service(value["running_service"],source)
    need(value["protected_services"]=={s+":"+n:"inactive" for s in ("system","user") for n in host.SERVICES} and value["foreign_compute_processes"]==same.measured.FOREIGN,"protected/foreign owners preserved")
    path=output.with_suffix(".npz");payload=value["payload"]
    same.exact_file(path,payload["sha256"],2*1024**2)
    need(payload["file"]==path.name and path.stat().st_size==payload["bytes"],"complete hybrid response payload envelope")
    arrays=raw_npz(path,{b+k for b in BRANCHES for k in layout},4*1024**2)
    need(set(payload["arrays"])==set(arrays),"complete228-field response manifest")
    for k,a in arrays.items(): need(payload["arrays"][k]==dict(shape=list(a.shape),dtype=str(a.dtype),sha256=sha256(a.tobytes()).hexdigest()),"every hybrid response leaf byte")
    before,after=[{k:arrays[b+k] for k in layout} for b in BRANCHES]
    response.exact_values(before,hybrid,"all114 actual hybrid inputs independently rebuilt");decoded=plan.full_bank(after,layout,prior)
    need(value["restore_sha256"]=={k:sha256(a.tobytes()).hexdigest() for k,a in before.items()} and value["solver_changed_data_fields"]==response.solver_write_fence(before,after),"full hybrid restore/write fences")
    # Pure validation of the retained branch descriptor; runtime guard itself
    # was executed before and after solve and is not a machine-code proof.
    expected_guard=dict(model_is_sparse=False,solver=2,cone=0,disableflags=0,enableflags=0,noslip_iterations=0,empty_dense_sparse_metadata=True,active_row_types=[1,6],source_path_only=True,machine_code_read_set_proved=False)
    need(value["branch_guard"]==expected_guard,"closed actually checked source-only branch descriptor")
    refs=dict(measured_gpu={k:raw["after_solve"+k] for k in layout},original_gpu_bank_cpu_reference=old_after)
    need(value["comparison"]==compare_outputs(after,refs,layout,prior,native_arrays),"all full-capacity reference/output differences recomputed")
    counters={k:decoded["/data/"+k].tolist() for k in saved.p.COUNTERS+("nacon","ncollision")}
    need(counters==value["cpu_counters"] and all(0<=v<=prior["child"]["plant"]["options"]["iterations"] for v in counters["solver_niter"]),"complete bounded hybrid counters")
    if cache is not None: need(saved.cache_files(Path(cache))==value["private_cache_files"],"complete private CPU cache bytes, not loaded binary identity")
    return value,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("input","output","cache"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source",required=True);args=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and datetime.now(timezone.utc)<datetime(2026,10,9,22,50,tzinfo=timezone.utc),"CPU-hidden hybrid launch before closeout reserve")
    identity=host.source_check(args.source);source_check(args.source)
    tools=host.ROOT/"artifacts/tools/ada-measured-boundary";input_root=host.ROOT/"artifacts/evaluations/ada-duck-contact-0ce8c9d0a820"
    need(args.input==host.ROOT/"artifacts/evaluations/ada-measured-boundary-daebeb6ee97b" and args.input.resolve(strict=True)==args.input and args.output.parent==args.cache.parent==tools and tools.resolve(strict=True)==tools and args.output.name==args.source[:8]+"-linux-native-family-solver.json" and args.cache.name==args.source[:8]+"-cpu-native-family-cache" and not args.output.exists() and not args.output.with_suffix(".npz").exists() and not args.cache.exists(),"fresh canonical source-bound native-family outputs/cache/input")
    running,services,foreign=unit_identity(args.source),host.service_snapshot(),host.foreign_processes();need(foreign==same.measured.FOREIGN,"only unchanged Dino GPU owner")
    versions={k:version(k) for k in same.VERSIONS};need(versions==same.VERSIONS,"unchanged exact frozen packages including BAM")
    audit=response.source.audit();need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"pure complete reference reception before runtime")
    h,hybrid,gpu,raw,layout,manifest,prior,banks,ref,_,old_after,native_arrays=references(tools,input_root,args.input)
    import warp as wp
    import torch
    args.cache.mkdir();saved.configure_private_cpu_cache(wp,args.cache);wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(),"CPU-only hybrid runtime")
    value,arrays=run(prior,banks,hybrid,layout,manifest,gpu["child"])
    need(host.source_check(args.source)==identity and response.source.audit()==audit and host.service_snapshot()==services and host.foreign_processes()==foreign and not torch.cuda.is_initialized(),"unchanged source/packages/services/foreign owners and no CUDA initialization")
    executables=saved.passive_executables();same.validate_executables(executables,ref)
    after={k:arrays["cpu_after_solve"+k] for k in layout}
    value.update(**CONTRACT,protocol=PROTOCOL,decision=DECISION,source=args.source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),versions=versions,source_audit_sha256=packet.digest(audit),plan_source=PLAN_SOURCE,plan_report_sha256=PLAN_SHA,plan_payload_sha256=PLAN_NPZ_SHA,plan_descriptor_sha256=packet.digest(h),baseline_source=BASE_SOURCE,baseline_report_sha256=BASE_SHA,baseline_payload_sha256=BASE_NPZ_SHA,gpu_report_sha256=same.GPU_REPORT_SHA,provenance_reference_sha256=same.CPU_RESPONSE_SHA,private_cache_was_absent=True,private_cache_files=saved.cache_files(args.cache),existing_cpu_executables=executables,payload=saved.p.retain(args.output.with_suffix(".npz"),arrays),running_service=running,protected_services=services,foreign_compute_processes=foreign,comparison=compare_outputs(after,dict(measured_gpu={k:raw["after_solve"+k] for k in layout},original_gpu_bank_cpu_reference=old_after),layout,prior,native_arrays))
    # The complete independent receiver must run in a separate fresh process:
    # immutable plan recomputation intentionally refuses imported runtimes.
    host.write_json(args.output,value)
    print(DECISION,value["cpu_counters"],value["comparison"]["current_native_generalized_fields"])


if __name__=="__main__": main()
