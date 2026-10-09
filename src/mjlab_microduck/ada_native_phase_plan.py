"""Pure source-bound native phase preflight, not an executed phase capture.

Do not reinterpret post-forward inputs as before-solve bytes. This preparation
binds a prospective explicit call order and its limits without loading a Model,
allocating Data, importing a simulator, or changing any admission gate.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from mjlab_microduck import ada_own_manifold_recipe as own

need, host, native = own.need, own.host, own.recipe.native
SOURCE_SHA = "cba19332e7dc7b110e158cb12c0ff9ff247b1b7e9338b2bbb88cedbf1b9887ed"
SOURCE_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco/28009f9105cd92784b7b0b30c0605a5e29107a77/src/engine/engine_forward.c"
OWN_SOURCE = "f960d5905550a280e4350ce740c4f0a53c51f54f"
OWN_SHA = "060e392640e9a5d33dd42c218e63d016dcf0ef79d7c96ec08e1cd3226f5d51d6"
PROTOCOL = "microduck-native-before-solve-phase-preflight-v1"
DECISION = "source-bound-native-phase-plan-not-executed-or-admitted"
BEFORE = ("mj_fwdPosition", "mj_sensorPos", "mj_fwdVelocity", "mj_sensorVel",
          "mj_fwdActuation", "mj_fwdAcceleration")
AFTER = ("mj_fwdConstraint",)
SOURCE_FUNCTIONS = ("mj_forward", "mj_forwardSkip", "mj_fwdPosition", "mj_fwdVelocity",
                    "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint")
FORWARD_ORDER = ("mj_fwdPosition", "mj_sensorPos", "mj_energyPos", "mj_fwdVelocity",
                 "mj_sensorVel", "mj_energyVel", "mj_fwdActuation", "mj_fwdAcceleration",
                 "mj_fwdConstraint", "mj_sensorAcc")
GUARDS = {"model/nq":21,"model/nv":20,"model/nu":14,"model/na":0,
          "model/nmocap":0,"model/nplugin":0,"model/nuserdata":0,"model/neq":0,
          "model/nflex":0,"option/enableflags":0,"option/disableflags":0,
          "option/cone":0,"option/jacobian":2,"option/solver":2,
          "option/noslip_iterations":0}
CONTRACT = dict(runtime_imported=False,new_model_allocations=0,new_data_allocations=0,
    new_forward_calls=0,new_collision_calls=0,new_constraint_calls=0,new_solver_calls=0,
    integration_steps=0,optimizer_steps=0,native_before_solve_capture_executed=False,
    full_native_Data_inventory_claimed=False,actual_compiled_phase_identity_established=False,
    full_historical_model_identity_established=False,contact_point_matching=False,
    exclusive_cause_established=False,compiled_binary_identity_established=False,
    simulator_qualified=False,solver_qualified=False,training_authorized=False,
    physical_motion_authorized=False,execution_ready=False,flags=host.FLAGS)


def c_body(text, name):
    """Balanced lexical body; no C semantics or transitive read-set claim."""
    need(type(text) is str and re.fullmatch(r"mj_[A-Za-z]+",name),"plain source/name")
    heads=list(re.finditer(r"^void "+re.escape(name)+r"\([^;]*?\)\s*\{",text,re.M))
    need(len(heads)==1,"one named void function declaration")
    start=heads[0].end()-1;depth=0
    tokens=re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|[{}]',text[start:])
    for match in tokens:
        token=match.group()
        if token=="{":depth+=1
        elif token=="}":
            depth-=1
            if depth==0:return text[start:start+match.end()]
    raise ValueError("unterminated source function")


def source_audit(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==64281,"plain bounded exact forward source")
    raw=path.read_bytes();need(sha256(raw).hexdigest()==SOURCE_SHA,"frozen upstream forward source SHA")
    text=raw.decode();bodies={n:c_body(text,n) for n in SOURCE_FUNCTIONS}
    # Frozen whole-file hash is authoritative. This is lexical call-order evidence.
    calls=re.findall(r"\b(mj_[A-Za-z0-9_]+)\s*\(",bodies["mj_forwardSkip"])
    need(tuple(calls)==FORWARD_ORDER,"all frozen forwardSkip lexical mj_ calls in order")
    need(re.findall(r"\b(mj_[A-Za-z0-9_]+)\s*\(",bodies["mj_forward"])==["mj_forwardSkip"],"forward wraps forwardSkip")
    need("mjSTAGE_NONE, 0" in bodies["mj_forward"],"no skip/no sensor skip native forward")
    return dict(url=SOURCE_URL,bytes=len(raw),sha256=SOURCE_SHA,
        named_function_body_sha256={n:sha256(b.encode()).hexdigest() for n,b in bodies.items()},
        lexical_forward_call_order=list(FORWARD_ORDER),transitive_machine_read_set_proven=False)


def model_guards(static):
    need(type(static) is list and len(static)==2 and static[0]==static[1],"two complete equal scalar/exclusion inventories")
    values=[]
    for meta in static:
        need(type(meta) is dict and set(meta)=={"scalars","exclusions"},"closed native static descriptor")
        row={}
        for k,v in GUARDS.items():
            item=meta["scalars"].get(k)
            need(type(item) is dict and item==dict(type="int",value=v) and type(item["value"]) is int,"exact supported native phase option/state guard: "+k)
            row[k]=item
        values.append(row)
    return values


def analyze(input_root, measured_root, tools):
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"fresh pure phase preflight")
    tools=Path(tools)
    old=tools/"f960d590-linux-own-manifold-recipe.json"
    own.hybrid.same.exact_file(old,OWN_SHA,256*1024)
    module=own.hybrid.archived_module(own,OWN_SOURCE)
    rebuilt=own.analyze(input_root,measured_root,tools)
    rebuilt.update(source=OWN_SOURCE,module_sha256=module)
    need(own.audit.packet.digest(rebuilt)==own.audit.packet.digest(json.loads(old.read_bytes())),"whole preceding own-manifold packet regenerated")
    source=source_audit(tools/"mujoco-28009f91-engine-forward.c")
    value,arrays=native.receive(tools/"f8540785-linux-native-constraint.json",own.recipe.NATIVE_SOURCE,input_root)
    guards=model_guards(value["capture"]["model_static"])
    models={k:a for k,a in arrays.items() if k.startswith("model/")}
    state={k:a for k,a in arrays.items() if k.startswith("fields/") and k.removeprefix("fields/") in native.STATE}
    need(len(state)==7 and len(models)>900,"complete authenticated logical native model and seven restored state fields")
    proposed=dict(model_input="Exact retained current two-world MJBs; fresh Data default plus all seven declared native F64 states, not recovered historical Data",
        ordered_calls_before_capture=list(BEFORE),ordered_calls_after_capture=list(AFTER),
        capture_boundaries=["immediately_before_mj_fwdConstraint","immediately_after_mj_fwdConstraint"],
        omitted_after_boundary=["mj_sensorAcc"],energy_calls_omitted_because_enableflags_zero=True,
        callbacks_required_absent=list(native.CALLBACKS),
        projected_numeric_capture=dict(fields=list(native.base.FIELDS),efc=list(native.EFC),
            contact=list(native.CONTACT),counters=list(native.COUNTERS),data=list(native.DATA[:-1]),solver=list(native.SOLVER)),
        full_native_Data_inventory=False,requires_all_runtime_shapes_dtypes_bytes_and_bounds_preflight=True,
        requires_full627_public_model_and_MJB_before_after_fences=True,
        requires_seven_state_bytes_unchanged=True,
        no_assumption_inputs_survive_solve=True,
        no_assumption_outputs_equal_prior_forward=True,
        no_assumption_full_sensor_state_equals_forward=True,
        no_cross_backend_contact_pairing=True,
        prospective_native_models=2,prospective_native_datas=2,
        prospective_explicit_call_counts={n:2 for n in BEFORE+AFTER},
        prospective_constraint_phase_runs_solver=True,
        prospective_solver_worlds=2,internal_solver_dispatch_count_requires_runtime_evidence=True,
        prospective_execution_is_not_solver_only=True,
        separate_execution_source_review_commit_required=True,
        proposed_runtime_caps=dict(seconds=180,memory_bytes=6*1024**3,cpu_percent=200,tasks=64,
            output_file_bytes=16*1024**2,integration_steps=0,optimizer_steps=0),
        launch_not_authorized_by_this_plan=True)
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"preflight stayed runtime-inert")
    return dict(protocol=PROTOCOL,decision=DECISION,previous_source=OWN_SOURCE,previous_report_sha256=OWN_SHA,
        native_report_sha256=own.recipe.NATIVE_REPORT_SHA,
        native_payload={k:value["payload"][k] for k in ("file","bytes","sha256")},
        native_payload_complete_manifest_sha256=own.audit.packet.digest(value["payload"]["arrays"]),
        retained_mjb_files=value["mjb"],full_native_model_array_manifest=native.manifest(models),
        seven_native_state_array_manifest=native.manifest(state),supported_model_guards=guards,
        frozen_source=source,prospective_execution=proposed,**CONTRACT)


def source_check(source):
    own.source_check(source)
    need(Path(__file__).read_bytes()==host.read("git","show",source+":src/mjlab_microduck/ada_native_phase_plan.py",binary=True),"exact current committed phase preflight")


def receive(path,source,input_root,measured_root,tools):
    source_check(source);path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=256*1024,"plain bounded complete preflight JSON")
    expected=analyze(input_root,measured_root,tools)
    expected.update(source=source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    value=json.loads(path.read_bytes())
    need(own.audit.packet.digest(value)==own.audit.packet.digest(expected),"whole phase plan independently regenerated")
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input","measured","tools","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--source",required=True);a=p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and all(os.environ.get(n+"_NUM_THREADS")=="1" for n in ("OMP","MKL","OPENBLAS","NUMEXPR")),"CPU-hidden four thread1 settings")
    need(datetime.now(timezone.utc)<datetime(2026,10,9,22,50,tzinfo=timezone.utc),"before closeout launch reserve")
    need(a.tools.is_absolute() and a.tools.resolve(strict=True)==a.tools and a.output.parent==a.tools and a.output.name in {a.source[:8]+"-"+h+"-native-phase-plan.json" for h in ("mac","linux")} and not a.output.exists(),"fresh canonical phase preflight report")
    source_check(a.source);value=analyze(a.input,a.measured,a.tools)
    value.update(source=a.source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source_check(a.source);host.write_json(a.output,value)
    receive(a.output,a.source,a.input,a.measured,a.tools)
    print(DECISION,len(value["full_native_model_array_manifest"]))


if __name__=="__main__":main()
