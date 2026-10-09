"""Read-only getter-to-projection adapter; not a native capture producer.

No simulator is imported or called here. Mock objects can satisfy these checks.
The future collector must separately bind installed provenance, current Model/
MJBs, absent callbacks, real call order, resource bounds and observation phases.
"""
import numpy as np
from mjlab_microduck import ada_native_phase_projection as projection

need,native=projection.need,projection.native
PROTOCOL="microduck-native-phase-getter-adapter-v1"
CONTRACT=dict(producer_authenticated=False,adapter_simulator_imports=0,
    adapter_physics_calls=0,new_model_allocations=0,new_data_allocations=0,
    integration_steps=0,optimizer_steps=0,native_phase_identity_established=False,
    compiled_binary_identity_established=False,solver_qualified=False,
    simulator_qualified=False,training_authorized=False,physical_motion_authorized=False,
    execution_ready=False,flags=projection.CONTRACT["flags"],
    caller_supplied_object_is_not_native_identity_proof=True,
    caller_runtime_process_and_phase_not_attested=True,
    scope="Getter adapter operations only; caller runtime/process/phase not attested")
FIELD_GETTER_SHAPES={"xfrc_applied":(16,6),"xpos":(16,3),"xquat":(16,4),
    "xipos":(16,3),"ximat":(16,9),"subtree_com":(16,3),"cinert":(16,10),
    "crb":(16,10),"cvel":(16,6)}


def attribute(owner,name):
    try:return getattr(owner,name)
    except AttributeError as exc:raise ValueError("missing projected getter: "+name) from exc


def copy_leaf(value,dtype,shape,aliases=()):
    """Exact raw dtype, explicit getter shapes, owned C-order byte-preserving copy."""
    need(type(value) is np.ndarray and value.dtype==dtype and value.flags.c_contiguous,
         "plain exact contiguous numeric getter")
    need(value.shape in (shape,)+tuple(aliases),"explicit complete getter shape, not inferred size")
    need(np.isfinite(value).all(),"finite complete getter before copying")
    result=value.reshape(shape).copy(order="C")
    need(result.flags.owndata and not np.shares_memory(value,result)
         and result.tobytes()==value.tobytes(),"owned unchanged getter bytes")
    return result


def snapshot(data,world,seven_states):
    """Read all68 leaves after validating counts; does not authenticate native Data."""
    schema=projection.layout(world);counts={}
    for k in native.COUNTERS:
        value=attribute(data,k)
        need(type(value) is int,"plain native counter getter, not coerced bool/float")
        counts[k]=value
    ncon,nefc=(0,14) if world==0 else (4,30)
    need({k:counts[k] for k in ("ncon","ne","nf","nl","nefc","nJ")}
         ==dict(ncon=ncon,ne=0,nf=14,nl=0,nefc=nefc,nJ=20*nefc)
         and 0<=counts["nA"]<=nefc*nefc and 0<=counts["nisland"]<=native.NISLAND,
         "declared bounded counters before every array getter")
    out={"data/counters":np.array([counts[k] for k in native.COUNTERS],np.int64)}
    for key,(dtype,shape) in schema.items():
        category,name=key.split("/",1)
        if key=="data/counters":continue
        aliases=()
        if category=="fields":
            value=attribute(data,name)
            if name=="time":
                need(type(value) is float and np.isfinite(value),"plain finite native time getter")
                value=np.array([value],np.float64)
            if name in FIELD_GETTER_SHAPES:aliases=(FIELD_GETTER_SHAPES[name],)
        elif category=="efc":value=attribute(data,"efc_"+name)
        elif category=="contact":
            value=attribute(attribute(data,"contact"),name)
            if name=="frame":aliases=((ncon,9),)
            if name=="H":aliases=((ncon,36),)
        elif category=="solver":
            value=attribute(attribute(data,"solver"),name)
            aliases=((native.NISLAND*native.NSOLVER,),)
        else:
            if name.startswith("warning/"):value=attribute(attribute(data,"warning"),name.split("/")[1])
            else:value=attribute(data,name)
        out[key]=copy_leaf(value,dtype,shape,aliases)
    check=projection.validate(out,world,seven_states)
    need(check["projected_fields"]==68 and not check["producer_authenticated"],"complete nonauthenticating projection")
    return out,dict(protocol=PROTOCOL,projection=check,**CONTRACT)
