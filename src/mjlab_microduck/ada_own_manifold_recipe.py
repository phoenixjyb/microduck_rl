"""Pure like-input contact recipe algebra, not compiled construction proof.

Keep each generator's own slots/rows. Ratio conversions are explicit numerical
hypotheses; do not pair unlike contact points, emulate FP32 kernels, or admit
physics/training from descriptive arithmetic agreement.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from mjlab_microduck import ada_native_constraint_recipe as recipe
from mjlab_microduck import ada_native_family_solver as hybrid

need,host,audit=recipe.need,recipe.host,recipe.audit
PROTOCOL="microduck-own-manifold-like-input-recipe-algebra-v1"
DECISION="own-manifold-like-input-source-algebra-not-compiled-construction-or-admission"
RECIPE_SOURCE=hybrid.plan.RECIPE_SOURCE
RECIPE_SHA=hybrid.plan.RECIPE_SHA
HYBRID_SOURCE="baab71808c2bca04892c4b61907e675408cbb0d6"
HYBRID_SHA="bdc34b3f800187145eca2e0078644f6c44ba3bc009883e77e6bcd04e13d3825a"
FIELDS=("D","aref","vel","pos","margin","frictionloss")
INPUTS={"timestep","impratio","invsqrt","inverse_weight","friction","solref","solimp","position","margin","velocity"}
CONTRACT=dict(runtime_imported=False,new_model_data_allocations=0,new_forward_calls=0,
    new_collision_calls=0,new_constraint_calls=0,new_solver_calls=0,integration_steps=0,optimizer_steps=0,
    own_contact_rows=48,native_contact_rows=16,measured_gpu_contact_rows=32,
    friction_rows_compared=False,contact_point_matching=False,row_count_normalization=False,
    scalar_input_conversions_explicit=True,actual_cross_backend_construction_executed=False,
    fp32_machine_emulation=False,compiled_binary_identity_established=False,
    native_input_phase_replay_executed=False,exclusive_cause_established=False,
    solver_qualified=False,simulator_qualified=False,training_authorized=False,
    physical_motion_authorized=False,flags=host.FLAGS)


def scalar_compare(inputs):
    """Both ordered F64 recipes over declared like-input scalar parameters."""
    need(type(inputs) is dict and set(inputs)==INPUTS,"closed complete like-input scalar schema")
    for key,size in (("friction",5),("solref",2),("solimp",5)):
        need(type(inputs[key]) is list and len(inputs[key])==size,"complete source parameter vector")
    flat=[v for k,v in inputs.items() if k not in {"friction","solref","solimp"}]+sum((inputs[k] for k in ("friction","solref","solimp")),[])
    need(all(type(v) in (int,float) and math.isfinite(v) for v in flat),"finite numeric scalars, not booleans")
    need(inputs["impratio"]>0 and inputs["invsqrt"]>0 and inputs["friction"][0]==inputs["friction"][1] and min(inputs["friction"])>0 and min(inputs["solref"])>0 and inputs["solimp"][4]==2,"positive standard isotropic dim3/power2 source domain")
    c=recipe.scalar_recipe(row_type=6,frictionloss=0.,**{k:inputs[k] for k in INPUTS-{"invsqrt"}})
    w=audit.row_recipe(timestep=inputs["timestep"],invsqrt=inputs["invsqrt"],inverse_weight=inputs["inverse_weight"],friction=inputs["friction"][0],solref=inputs["solref"],solimp=inputs["solimp"],distance=inputs["position"],margin=inputs["margin"],velocity=inputs["velocity"])
    result=dict(native_C_recipe=c,warp_source_recipe=w,
        warp_minus_C={k:w[k]-c[k] for k in FIELDS},
        ratio_minus_inverse_sqrt_impratio=inputs["invsqrt"]-1./math.sqrt(inputs["impratio"]),
        branch=dict(native_flat_width=inputs["solimp"][2]<=audit.NUMERIC["minval"],
            native_flat_impedance=inputs["solimp"][0]==inputs["solimp"][1],
            timeconst_clipped=inputs["solref"][0]<2.*inputs["timestep"]),
        convention="Ordered Python F64 algebra; not actual cross-backend construction or FP32 rounding/fusion")
    need(all(math.isfinite(v) for v in result["warp_minus_C"].values()) and math.isfinite(result["ratio_minus_inverse_sqrt_impratio"]),"finite unqualified algebra differences")
    return result


def compare_rows(description,native_scalars,models):
    """Cover own16 native + own32 measured GPU rows, never a cross pairing."""
    import numpy as np
    rows=[];covered={"native":[],"measured_gpu":[]}
    nrows=[r for r in description["native_row_recipes"] if r["row_type"]==6]
    grows=description["measured_gpu_row_recipes"]
    need(len(nrows)==16 and len(grows)==32 and len(description["native_contact_inputs"])==4 and len(description["measured_gpu_contact_inputs"])==8,"fixed complete own contact row/contact inventories")
    for backend,source_rows in (("native",nrows),("measured_gpu",grows)):
        for r in source_rows:
            world=r["world"];slot=r["own_id"] if backend=="native" else r["slot"];addr=r["row"] if backend=="native" else r["efc_address"]
            need(type(world) is int and world==1 and type(slot) is int and type(addr) is int and addr in range(14+4*slot,18+4*slot),"every own world/slot/address link, no row pairing")
            if backend=="native":
                original=r["inputs"];scalars=native_scalars[world]
                timestep=scalars["option/timestep"]["value"];impratio=scalars["option/impratio"]["value"]
                need(type(impratio) in (int,float) and math.isfinite(impratio) and impratio>0,"actual native positive impratio")
                ratio=1./math.sqrt(impratio);velocity=r["recipe"]["vel"]
                inputs=dict(timestep=timestep,impratio=impratio,invsqrt=ratio,velocity=velocity,**{k:original[k] for k in ("inverse_weight","friction","solref","solimp","position","margin")})
                conversion="Actual native F64 impratio -> ordered F64 inverse sqrt for hypothetical Warp source algebra"
                captured_dtype="float64"
            else:
                contact=description["measured_gpu_contact_inputs"][slot]
                need(contact["slot"]==slot and contact["world"]==world and contact["efc_address"]==list(range(14+4*slot,18+4*slot)),"every own complete GPU contact partition")
                bodies=[int(models["/model/geom_bodyid"][g]) for g in contact["geom"]]
                need(bodies==r["bodies"] and r["geom"]==contact["geom"],"same own bound Model/body/contact linkage")
                inverse_weight=sum(float(models["/model/body_invweight0"][0,b,0]) for b in bodies)
                ratio=float(models["/model/opt/impratio_invsqrt"][0]);square=ratio*ratio
                need(math.isfinite(square) and square>0,"finite explicit GPU ratio inverse conversion")
                inputs=dict(timestep=float(models["/model/opt/timestep"][0]),impratio=1./square,invsqrt=ratio,inverse_weight=inverse_weight,friction=contact["friction"],solref=contact["solref"],solimp=contact["solimp"],position=contact["dist"],margin=contact["includemargin"],velocity=r["captured"]["vel"])
                conversion="Actual GPU F32 values widened -> hypothetical C impratio = 1/(invsqrt*invsqrt); not an actual native option"
                captured_dtype="float32"
            result=scalar_compare(inputs);captured={k:r["captured"][k] for k in FIELDS}
            need(all(type(v) in (float,int) and math.isfinite(v) for v in captured.values()),"finite full own captured comparable scalars")
            rows.append(dict(backend=backend,world=world,own_slot=slot,own_row=addr,inputs=inputs,input_conversion=conversion,
                captured=captured,captured_dtype=captured_dtype,
                captured_scalar_sha256={k:sha256(np.array(v,dtype=captured_dtype).tobytes()).hexdigest() for k,v in captured.items()},
                C_minus_own_captured={k:result["native_C_recipe"][k]-captured[k] for k in FIELDS},
                warp_minus_own_captured={k:result["warp_source_recipe"][k]-captured[k] for k in FIELDS},**result))
            covered[backend].append(addr)
    need(covered==dict(native=list(range(14,30)),measured_gpu=list(range(14,46))),"every own contact row exactly once with no masking/normalization")
    return dict(rows=rows,own_covered_rows=covered,
        max_abs_warp_minus_C={b:{k:max(abs(r["warp_minus_C"][k]) for r in rows if r["backend"]==b) for k in FIELDS} for b in covered})


def analyze(input_root,measured_root,tools):
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"fresh pure own-manifold algebra")
    tools=Path(tools);path=tools/"ec81e2bb-linux-native-recipe.json"
    hybrid.same.exact_file(path,RECIPE_SHA,256*1024);module=hybrid.archived_module(recipe,RECIPE_SOURCE)
    description=recipe.analyze(input_root,measured_root,tools/"f8540785-linux-native-constraint.json",tools/"mujoco-28009f91-engine-core-constraint.c")
    description.update(source=RECIPE_SOURCE,module_sha256=module)
    need(audit.packet.digest(description)==audit.packet.digest(json.loads(path.read_bytes())),"all original own recipe descriptors independently regenerated")
    motive=tools/"baab7180-linux-native-family-solver.json"
    hybrid.same.exact_file(motive,HYBRID_SHA,256*1024);hybrid.archived_module(hybrid,HYBRID_SOURCE)
    native=json.loads((tools/"f8540785-linux-native-constraint.json").read_bytes())
    gpu,_,_,_,_=hybrid.same.gpu_packet(measured_root,input_root)
    prior,banks=recipe.native.prior.authenticated_banks(input_root)
    models=audit.bound_model_values(prior,banks,gpu["child"])
    result=compare_rows(description,[r["scalars"] for r in native["capture"]["model_static"]],models)
    need(not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(),"pure algebra stayed runtime-inert")
    return dict(protocol=PROTOCOL,decision=DECISION,recipe_source=RECIPE_SOURCE,recipe_report_sha256=RECIPE_SHA,
        native_report_sha256=recipe.NATIVE_REPORT_SHA,native_C_source=description["native_recipe_source"],
        frozen_warp_source_audit_sha256=description["gpu_stage_source_audit_sha256"],
        measured_gpu_report_sha256=audit.GPU_REPORT_SHA,all347_model_array_sha256=gpu["child"]["model_array_sha256"],
        motive_hybrid_source=HYBRID_SOURCE,motive_hybrid_report_sha256=HYBRID_SHA,motive_is_context_only=True,
        comparable_fields=list(FIELDS),input_semantics="Own native F64 or widened measured GPU F32; explicit ratio conversions, not an ABI identity",
        **result,**CONTRACT)


def source_check(source):
    recipe.source_check(source)
    need(Path(__file__).read_bytes()==host.read("git","show",source+":src/mjlab_microduck/ada_own_manifold_recipe.py",binary=True),"exact committed own-manifold algebra source")


def receive(path,source,input_root,measured_root,tools):
    source_check(source);path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=256*1024,"plain bounded complete algebra report")
    expected=analyze(input_root,measured_root,tools)
    expected.update(source=source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    actual=json.loads(path.read_bytes())
    need(audit.packet.digest(actual)==audit.packet.digest(expected),"every own-manifold algebra descriptor independently regenerated")
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("input","measured","tools","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source",required=True);args=parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and datetime.now(timezone.utc)<datetime(2026,10,9,22,50,tzinfo=timezone.utc),"CPU-hidden pure algebra before closeout reserve")
    need(all(os.environ.get(n+"_NUM_THREADS")=="1" for n in ("OMP","MKL","OPENBLAS","NUMEXPR")),"four literal single-thread settings")
    need(args.tools.is_absolute() and args.tools.resolve(strict=True)==args.tools and args.output.parent==args.tools and args.output.name in {args.source[:8]+"-"+h+"-own-manifold-recipe.json" for h in ("mac","linux")} and not args.output.exists(),"fresh canonical source-prefix own-manifold report")
    source_check(args.source);result=analyze(args.input,args.measured,args.tools)
    result.update(source=args.source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source_check(args.source);host.write_json(args.output,result)
    receive(args.output,args.source,args.input,args.measured,args.tools)
    print(DECISION,result["max_abs_warp_minus_C"])


if __name__=="__main__":main()
