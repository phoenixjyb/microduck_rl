"""Mock getter objects only: no native Model/Data or physics execution."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from mjlab_microduck import ada_native_phase_getters as p
from test_ada_native_phase_projection import fixture


def mock(world):
    arrays,states=fixture(world);data=SimpleNamespace(contact=SimpleNamespace(),solver=SimpleNamespace(),warning=SimpleNamespace())
    counts=dict(zip(p.native.COUNTERS,map(int,arrays["data/counters"])))
    for k,v in counts.items():setattr(data,k,v)
    for key,value in arrays.items():
        category,name=key.split("/",1)
        if key=="data/counters":continue
        owner=data;attr=name
        if category=="fields":
            if name=="time":value=float(value[0])
            elif name in p.FIELD_GETTER_SHAPES:value=value.reshape(p.FIELD_GETTER_SHAPES[name])
        elif category=="efc":attr="efc_"+name
        elif category=="contact":
            owner=data.contact
            if name in ("frame","H"):value=value.reshape(counts["ncon"],9 if name=="frame" else 36)
        elif category=="solver":owner=data.solver;value=value.reshape(-1)
        elif name.startswith("warning/"):owner=data.warning;attr=name.split("/")[1]
        setattr(owner,attr,value)
    return data,arrays,states


@pytest.mark.parametrize("world",[0,1])
def test_every_getter_exact_owned_bytes_without_native_identity(world):
    data,original,states=mock(world)
    arrays,meta=p.snapshot(data,world,states)
    assert set(arrays)==set(original)
    assert all(a.tobytes()==original[k].tobytes() and a.flags.owndata for k,a in arrays.items())
    assert not meta["producer_authenticated"] and meta["adapter_physics_calls"]==0
    assert meta["caller_runtime_process_and_phase_not_attested"]
    assert not meta["execution_ready"] and meta["caller_supplied_object_is_not_native_identity_proof"]
    data.qpos[0]=1.;assert arrays["fields/qpos"][0]==0.


@pytest.mark.parametrize("key",sorted(p.projection.layout(1)))
def test_each_projected_getter_is_required(key):
    data,_,states=mock(1);category,name=key.split("/",1)
    if key=="data/counters":owner,attr=data,"nJ"
    elif category=="efc":owner,attr=data,"efc_"+name
    elif category=="contact":owner,attr=data.contact,name
    elif category=="solver":owner,attr=data.solver,name
    elif name.startswith("warning/"):owner,attr=data.warning,name.split("/")[1]
    else:owner,attr=data,name
    delattr(owner,attr)
    with pytest.raises(ValueError):p.snapshot(data,1,states)


@pytest.mark.parametrize("bad",[True,30.,np.int64(30),999,-1])
def test_counter_preflight_refuses_before_reading_any_array(bad):
    data,_,states=mock(1);data.nefc=bad
    delattr(data,"qpos")
    with pytest.raises(ValueError,match="counter"):
        p.snapshot(data,1,states)


@pytest.mark.parametrize("bad",[True,0,np.float32(0),float("nan")])
def test_time_kind_and_finiteness_refused(bad):
    data,_,states=mock(1);data.time=bad
    with pytest.raises(ValueError):p.snapshot(data,1,states)


@pytest.mark.parametrize("damage",["dtype","size_only","nan","strided","object","tuple"])
def test_numeric_getter_does_not_silently_cast_or_infer_shape(damage):
    value=np.zeros((2,3),np.float64)
    if damage=="dtype":value=value.astype(np.float32)
    if damage=="size_only":value=value.reshape(3,2)
    if damage=="nan":value[0,0]=np.nan
    if damage=="strided":value=np.zeros((2,6))[:,::2]
    if damage=="object":value=value.astype(object)
    if damage=="tuple":value=tuple(value)
    with pytest.raises(ValueError):p.copy_leaf(value,np.dtype("float64"),(6,),((2,3),))


def test_complete_boundary_comparison_keeps_tiny_getter_changes():
    data,_,states=mock(1);before,_=p.snapshot(data,1,states)
    data.efc_D[0]=1e-15;data.efc_force[14]=3.
    after,_=p.snapshot(data,1,states)
    r=p.projection.compare(before,after,1,states)
    assert len(r["differences"])==68
    assert r["differences"]["efc/D"]["max_abs"]==1e-15
    assert r["differences"]["efc/force"]["max_abs"]==3.
    assert not r["native_phase_identity_established"] and not r["training_authorized"]
