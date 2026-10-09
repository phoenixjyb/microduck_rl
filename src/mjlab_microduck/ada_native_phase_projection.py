"""Pure closed native phase projection checks, not a capture producer.

Synthetic arrays can satisfy this schema. Provenance, real native calls, full
Model/MJB fences and before-solve identity require the separate future collector.
Never turn these shape/content checks into simulator or training admission.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import numpy as np
from mjlab_microduck import ada_native_phase_plan as plan

need,native=plan.need,plan.native
PROTOCOL="microduck-native-phase-projection-contract-v1"
DECISION="retained-post-forward-projection-compatible-not-before-solve-capture"
PLAN_SOURCE="8382487e5dea6142d9300e617781cfa20b3d7b62"
PLAN_SHA="1c1980250956120ad9589b706bf71053e8d72fd14d03ac2fea0819f7856eeb85"
CONTRACT=dict(runtime_imported=False,actual_native_capture_executed=False,
    new_model_allocations=0,new_data_allocations=0,new_forward_calls=0,new_solver_calls=0,
    integration_steps=0,optimizer_steps=0,full_native_Data_inventory=False,
    compiled_binary_identity_established=False,native_phase_identity_established=False,
    simulator_qualified=False,solver_qualified=False,training_authorized=False,
    physical_motion_authorized=False,execution_ready=False,flags=plan.host.FLAGS)
INT_EFC={"type","id","J_rownnz","J_rowadr","J_rowsuper","J_colind","state"}
INT_SOLVER={"nactive","nchange","neval","nupdate"}


def layout(world):
    """The declared saved fixture's complete projected public numeric leaves."""
    need(type(world) is int and world in (0,1),"plain two-world index")
    ncon,nefc=(0,14) if world==0 else (4,30)
    fields={"fields/"+k:(np.dtype("float64"),(native.base.WIDTHS[k],)) for k in native.base.FIELDS}
    for k in native.EFC:
        shape=(nefc,4) if k=="KBIP" else (nefc*20,) if k in ("J","J_colind") else (nefc,)
        fields["efc/"+k]=(np.dtype("int32" if k in INT_EFC else "float64"),shape)
    for k in native.CONTACT:
        fields["contact/"+k]=(np.dtype("int32" if k in native.CONTACT_INTS else "float64"),(ncon,)+native.CONTACT_TAILS[k])
    fields["data/counters"]=(np.dtype("int64"),(len(native.COUNTERS),))
    for k in native.DATA[:-1]:
        fields["data/"+k]=(np.dtype("int32"),(native.NWARNING if k.startswith("warning/") else native.NISLAND,))
    for k in native.SOLVER:
        fields["solver/"+k]=(np.dtype("int32" if k in INT_SOLVER else "float64"),(native.NISLAND,native.NSOLVER))
    return fields


def validate(arrays,world,seven_states):
    """Owned numeric projection schema/own rows; does not authenticate a producer."""
    schema=layout(world)
    need(type(arrays) is dict and set(arrays)==set(schema),"complete closed projected native inventory")
    for k,(dtype,shape) in schema.items():
        a=arrays[k]
        need(type(a) is np.ndarray and a.dtype==dtype and a.shape==shape and a.flags.c_contiguous,"exact numeric projection dtype/shape/contiguity: "+k)
        need(np.isfinite(a).all(),"finite complete projected leaf: "+k)
    need(sum(a.nbytes for a in arrays.values())<1024**2,"bounded entire projected world")
    need(type(seven_states) is dict and set(seven_states)==set(native.STATE),"all seven separately supplied input states")
    for k,a in seven_states.items():
        need(type(a) is np.ndarray and a.dtype==np.float64 and a.shape==(native.base.WIDTHS[k],) and np.isfinite(a).all(),"exact finite supplied native state layout")
        need(arrays["fields/"+k].tobytes()==a.tobytes(),"each supplied state byte unchanged: "+k)
    counters=dict(zip(native.COUNTERS,map(int,arrays["data/counters"])))
    ncon,nefc=(0,14) if world==0 else (4,30)
    need({k:counters[k] for k in ("ncon","ne","nf","nl","nefc","nJ")}==dict(ncon=ncon,ne=0,nf=14,nl=0,nefc=nefc,nJ=nefc*20),"complete declared dense own counter domain")
    need(0<=counters["nA"]<=nefc*nefc and 0<=counters["nisland"]<=native.NISLAND,"bounded matrix/island counters")
    need((arrays["data/warning/number"]==0).all() and (arrays["data/warning/lastinfo"]>=0).all(),"no warnings; retain all statistic slots")
    need((arrays["data/solver_niter"]>=0).all() and (arrays["data/solver_niter"]<=native.NSOLVER).all() and (arrays["data/solver_nnz"]>=0).all(),"bounded all solver iteration statistics")
    need((arrays["efc/type"][:14]==1).all() and (arrays["efc/id"][:14]>=0).all() and (arrays["efc/id"][:14]<20).all() and len(set(map(int,arrays["efc/id"][:14])))==14,"complete own DOF-friction row IDs")
    # Dense J uses the full flat numeric array. Sparse metadata getters remain
    # retained leaves, but are not interpreted as populated sparse row tables.
    # Empty world0 contacts are retained as every correctly shaped zero-size leaf.
    for slot in range(ncon):
        addr=14+4*slot
        need(int(arrays["contact/dim"][slot])==3 and int(arrays["contact/efc_address"][slot])==addr,"own dim3 complete row partition")
        need((arrays["efc/type"][addr:addr+4]==6).all() and (arrays["efc/id"][addr:addr+4]==slot).all(),"every own contact row type/id/address")
        geom=arrays["contact/geom"][slot]
        need(geom.tolist() in ([0,29],[0,79]),"declared ordered floor/foot geom domain; not physical point matching")
    return dict(world=world,counters=counters,arrays=native.manifest(arrays),
        projected_fields=len(schema),raw_bytes=sum(a.nbytes for a in arrays.values()),
        supplied_seven_state_bytes_unchanged=True,producer_authenticated=False,
        native_phase_identity_established=False,sparse_metadata_interpreted=False)


def compare(before,after,world,seven_states):
    """All leaves, including solver-written ones; no masking or equality gate."""
    b=validate(before,world,seven_states);a=validate(after,world,seven_states)
    differences={}
    for k in layout(world):
        left,right=before[k],after[k]
        delta=right.astype(np.float64)-left.astype(np.float64)
        differences[k]=dict(bytes_equal=left.tobytes()==right.tobytes(),
            changed_values=int(np.count_nonzero(left!=right)),
            max_abs=float(np.abs(delta).max()) if delta.size else 0.)
    return dict(protocol=PROTOCOL,world=world,before=b,after=a,
        differences=differences,no_assumption_inputs_survive_solve=True,
        no_assumption_outputs_equal_previous_forward=True,**CONTRACT)


def retained_post_forward_projection(arrays,world):
    """Decode already-authenticated f854 arrays, never manufacture a pre-boundary.

    Caller must first authenticate the entire old capture and its references.
    This pure adapter does not establish provenance by itself.
    """
    schema=layout(world);result={}
    need(type(arrays) is dict,"plain already-authenticated logical array bank")
    for k in schema:
        category,name=k.split("/",1)
        path=k if category=="fields" else category+"/"+str(world)+"/"+name
        need(path in arrays and type(arrays[path]) is np.ndarray,"every retained projected source leaf")
        a=arrays[path][world] if category=="fields" else arrays[path]
        result[k]=np.array(a,copy=True,order="C")
    return result


def analyze_retained(input_root,measured_root,tools):
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"fresh pure retained projection check")
    tools=Path(tools);old=tools/"8382487e-linux-native-phase-plan.json"
    plan.own.hybrid.same.exact_file(old,PLAN_SHA,256*1024)
    module=plan.own.hybrid.archived_module(plan,PLAN_SOURCE)
    rebuilt=plan.analyze(input_root,measured_root,tools)
    rebuilt.update(source=PLAN_SOURCE,module_sha256=module)
    need(plan.own.audit.packet.digest(rebuilt)==plan.own.audit.packet.digest(json.loads(old.read_bytes())),"entire predecessor phase plan independently regenerated")
    capture,bank=native.receive(tools/"f8540785-linux-native-constraint.json",plan.own.recipe.NATIVE_SOURCE,input_root)
    previous,banks=native.prior.authenticated_banks(input_root)
    state=native.replay.state_inputs(previous,banks)
    records=[]
    for world in (0,1):
        projection=retained_post_forward_projection(bank,world)
        seven={k:state[k][world].astype(np.float64).reshape(-1) for k in native.STATE}
        records.append(dict(stage="retained_post_forward_only",**validate(projection,world,seven)))
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"projection check stayed runtime-inert")
    return dict(protocol=PROTOCOL,decision=DECISION,previous_source=PLAN_SOURCE,previous_plan_sha256=PLAN_SHA,
        native_source=plan.own.recipe.NATIVE_SOURCE,native_report_sha256=plan.own.recipe.NATIVE_REPORT_SHA,
        native_payload_sha256=capture["payload"]["sha256"],retained_mjb_files=capture["mjb"],
        references_full_logical_model_array_count=len(rebuilt["full_native_model_array_manifest"]),
        stage="retained_post_forward_only",records=records,
        no_before_after_phase_comparison_executed=True,**CONTRACT)


def source_check(source):
    plan.source_check(source)
    need(Path(__file__).read_bytes()==plan.host.read("git","show",source+":src/mjlab_microduck/ada_native_phase_projection.py",binary=True),"exact committed projection module")


def receive(path,source,input_root,measured_root,tools):
    source_check(source);path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=256*1024,"plain bounded full retained projection report")
    expected=analyze_retained(input_root,measured_root,tools)
    expected.update(source=source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    value=json.loads(path.read_bytes())
    need(plan.own.audit.packet.digest(value)==plan.own.audit.packet.digest(expected),"every retained projection descriptor regenerated")
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("input","measured","tools","output"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--source",required=True);a=p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and all(os.environ.get(n+"_NUM_THREADS")=="1" for n in ("OMP","MKL","OPENBLAS","NUMEXPR")),"CPU-hidden four literal thread1 assignments")
    need(datetime.now(timezone.utc)<datetime(2026,10,9,22,50,tzinfo=timezone.utc),"launch before closeout reserve")
    need(a.tools.is_absolute() and a.tools.resolve(strict=True)==a.tools and a.output.parent==a.tools and a.output.name in {a.source[:8]+"-"+h+"-native-phase-projection.json" for h in ("mac","linux")} and not a.output.exists(),"fresh canonical source-prefix projection report")
    source_check(a.source);value=analyze_retained(a.input,a.measured,a.tools)
    value.update(source=a.source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source_check(a.source);plan.host.write_json(a.output,value)
    receive(a.output,a.source,a.input,a.measured,a.tools)
    print(DECISION,[r["projected_fields"] for r in value["records"]])


if __name__=="__main__":main()
