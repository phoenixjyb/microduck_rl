"""CPU-only typed staging and independent positive native envelope fixtures."""
from copy import deepcopy
from hashlib import sha256
import ast
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_stance_solver_efc_transition import packet, prospective
from test_stance_solver_cost_probe import bundle as legacy_bundle, _refresh
from mjlab_microduck import stance_solver_cost_probe as old_probe
from mjlab_microduck import stance_solver_efc_probe as probe
from mjlab_microduck import stance_solver_efc_receiver as receiver
from mjlab_microduck import stance_solver_efc_transition as transition
from mjlab_microduck import stance_solver_cost_stages as stages


@pytest.fixture(scope="module")
def example(packet, prospective):
    root = Path(transition.__file__).resolve().parents[2]
    directory = root / "artifacts/evaluations" / ("caller-cost-" + transition.PRIOR_SOURCE[:12])
    # Authenticated retained reference, paired with an independent CPU fixture.
    transition.audit_retained(root)
    reference = {p: probe.read_plain(directory / ("reference-" + p + ".bin")) for p in stages.PHASES}
    arms = dict(reference=reference, control=prospective.banks)
    anchors = {a: {p: dict(bytes=len(raw), sha256=sha256(raw).hexdigest()) for p,raw in banks.items()}
               for a,banks in arms.items()}
    return NS(packet=packet, arms=arms, anchors=anchors)


@pytest.fixture
def bundle(legacy_bundle):
    b = legacy_bundle
    source = b.receipt["source"]
    old_path, new_path = str(old_probe.output(source)), str(probe.output(source))
    old_unit, new_unit = old_probe.unit(source), probe.unit(source)
    def rewrite(v):
        if type(v) is str: return v.replace(old_path,new_path).replace(old_unit,new_unit)
        if type(v) is dict: return {k:rewrite(x) for k,x in v.items()}
        if type(v) is list: return [rewrite(x) for x in v]
        return v
    records = rewrite(deepcopy(b.records))
    dcl, receipt = records["declaration"], records["receipt"]
    dcl["protocol"] = receipt["protocol"] = probe.PROTOCOL
    dcl["control"] = transition.control_manifest(b.packet)
    dcl["source_binding"]["leaves"] = [dict(path=n,bytes=0,git_blob="0"*40,sha256=sha256(b"").hexdigest())
                                         for n in sorted(probe.OWN)]
    receipt["declaration_sha256"] = sha256(probe.canonical(dcl)).hexdigest()
    for arm, value in receipt["arms"].items():
        value["staging"] = None if arm == "reference" else dict(protocol=transition.CONTROL_PROTOCOL,
            overrides=transition.control_manifest(b.packet)["overrides"], fields=list(transition.control_fields(b.packet)),
            preflight_complete=True, same_stream_copy=True, gpu_numpy_readback=False,
            timing_changed_by_staging=True, qualification=dict(stages.FLAGS))
    for name,v in records.items(): (b.root/(name+".json")).write_bytes(probe.canonical(v))
    inventory = _refresh(b.root)
    close = rewrite(deepcopy(b.closeout))
    close["declaration_sha256"] = receipt["declaration_sha256"]
    close["inventory_sha256"] = sha256(probe.canonical(inventory)).hexdigest()
    return NS(root=b.root, inventory=inventory, closeout=close, packet=b.packet, receipt=receipt, records=records)


def test_independent_positive_pair_not_old_output_equality(example):
    r = receiver.receive_banks(example.arms, example.anchors, example.packet)
    assert r["control"]["efc"]["force_changes"] == r["control"]["efc"]["state_changes"] == 2944
    assert r["reference"]["efc"]["force_changes"] == r["reference"]["efc"]["state_changes"] == 0
    assert not r["cross_arm_output_equality_required"] and not any(r["qualification"].values())
    for n,p in (("efc.force","efc.after"),("efc.state","efc.after"),("data.qfrc_constraint","dense.after")):
        assert stages.unpack_bank(example.arms["reference"][p])[n] != stages.unpack_bank(example.arms["control"][p])[n]


@pytest.mark.parametrize("arm", ("reference","control"))
@pytest.mark.parametrize("phase", stages.PHASES)
def test_all_sixteen_anchors_precede_any_decode(example, monkeypatch, arm, phase):
    arms = {a:dict(v) for a,v in example.arms.items()}
    arms[arm][phase] = b"x" + arms[arm][phase][1:]
    monkeypatch.setattr(stages,"unpack_bank",lambda *_:pytest.fail("premature bank decode"))
    with pytest.raises(ValueError, match="sixteen-bank anchors"):
        receiver.receive_banks(arms, example.anchors, example.packet)


def test_complete_native_envelope_fixture_is_not_native_proof(bundle):
    r = receiver.receive(bundle.root,bundle.inventory,bundle.closeout,bundle.packet)
    assert r["external_retirement_evidence_checked"]
    assert not any(r["qualification"].values()) and not r["driver_loaded_bytes_observed"]


@pytest.mark.parametrize("fault", ("pid","exit","invocation","restart","cgroup","inventory","declaration"))
def test_external_retirement_cannot_be_substituted(bundle,fault):
    c = deepcopy(bundle.closeout)
    if fault=="pid": c["service"]["MainPID"]="42"
    elif fault=="exit": c["service"]["ExecMainStatus"]="1"
    elif fault=="invocation": c["service"]["InvocationID"]="0"*32
    elif fault=="restart": c["service"]["NRestarts"]="1"
    elif fault=="cgroup": c["cgroup"]["absent"]=False
    else: c[fault+"_sha256"]="0"*64
    with pytest.raises(ValueError): receiver.receive(bundle.root,bundle.inventory,c,bundle.packet)


@pytest.mark.parametrize("fault", ("staging-missing","state-field","dtype-claim","reference-staging","gpu","binding","site","flags","old-protocol"))
def test_reanchored_receipt_cannot_bypass_new_contract(bundle,fault):
    r = deepcopy(bundle.receipt)
    if fault=="staging-missing": del r["arms"]["control"]["staging"]
    elif fault=="state-field": r["arms"]["control"]["staging"]["fields"].remove("efc.state")
    elif fault=="dtype-claim": r["arms"]["control"]["staging"]["gpu_numpy_readback"]=True
    elif fault=="reference-staging": r["arms"]["reference"]["staging"]={}
    elif fault=="gpu": r["device"]["gpu_uuid"]="other"
    elif fault=="binding": r["modules"]["efc"]["bindings"]["efc"]["context"]+=1
    elif fault=="site": r["arms"]["control"]["observer"]["call_site_lines"][1]+=1
    elif fault=="old-protocol": r["protocol"]=old_probe.PROTOCOL
    else: r["flags"]["native_qualified"]=True
    (bundle.root/"receipt.json").write_bytes(probe.canonical(r))
    inv = _refresh(bundle.root)
    c = dict(bundle.closeout,inventory_sha256=sha256(probe.canonical(inv)).hexdigest())
    with pytest.raises(ValueError): receiver.receive(bundle.root,inv,c,bundle.packet)


@pytest.mark.parametrize("fault", ("tree","leaf-duplicate","leaf-missing","path","dot","control-char","bytes","blob","sha","leaf-extra"))
def test_reanchored_source_declaration_cannot_bypass_closed_inventory(bundle,fault):
    d = deepcopy(bundle.records["declaration"])
    s = d["source_binding"]
    if fault=="tree": s["tree"]="wrong"
    elif fault=="leaf-duplicate": s["leaves"].append(dict(s["leaves"][0]))
    elif fault=="leaf-missing": s["leaves"].pop()
    elif fault=="path": s["leaves"][0]["path"]="../outside"
    elif fault=="dot": s["leaves"][0]["path"]="."
    elif fault=="control-char": s["leaves"][0]["path"]="bad\nname.py"
    elif fault=="bytes": s["leaves"][0]["bytes"]=True
    elif fault=="blob": s["leaves"][0]["git_blob"]="A"*40
    elif fault=="sha": s["leaves"][0]["sha256"]="0"*63
    else: s["leaves"][0]["unknown"]=0
    (bundle.root/"declaration.json").write_bytes(probe.canonical(d))
    r = deepcopy(bundle.receipt)
    r["declaration_sha256"] = sha256(probe.canonical(d)).hexdigest()
    (bundle.root/"receipt.json").write_bytes(probe.canonical(r))
    inv=_refresh(bundle.root)
    close=dict(bundle.closeout,declaration_sha256=r["declaration_sha256"],inventory_sha256=sha256(probe.canonical(inv)).hexdigest())
    with pytest.raises(ValueError): receiver.receive(bundle.root,inv,close,bundle.packet)


@pytest.mark.parametrize("index,value", ((2,0),(3,False),(4,-1)))
def test_generated_identity_size_and_time_schema_bound_to_authenticated_bytes(bundle,index,value):
    r=deepcopy(bundle.receipt)
    r["modules"]["efc"]["generated"]["binary"]["identity"][index]=value
    (bundle.root/"receipt.json").write_bytes(probe.canonical(r))
    inv=_refresh(bundle.root)
    close=dict(bundle.closeout,inventory_sha256=sha256(probe.canonical(inv)).hexdigest())
    with pytest.raises(ValueError): receiver.receive(bundle.root,inv,close,bundle.packet)


@pytest.mark.parametrize("fault", (None,"unpinned-state","state-dtype","state-bytes","shape","source-gpu","dst-alias","host-alias","host-dst-alias","entry-change","stream-change"))
def test_typed_preflight_all_fields_before_first_copy(packet,fault):
    gpu, cpu = NS(is_cuda=True,is_cpu=False), NS(is_cuda=False,is_cpu=True)
    dtypes = {n:object() for n in ("float32","int32","bool")}
    stream, events = NS(device=gpu,cuda_stream=55), []
    specs={n:(logical,host,dtype) for n,logical,host,dtype in stages.SPECS}
    values=transition.control_fields(packet)
    state_index=list(values).index("efc.state")
    class Array:
        is_contiguous, requires_grad = True, False
        def __init__(self, host, *, dtype, device, pinned, requires_grad):
            index=counter[0]; counter[0]+=1
            self.host,self.dtype,self.device,self.pinned=host,dtype,cpu,pinned
            self.shape,self.ptr=host.shape,100_000_000+index*4_000_000
            if index==state_index:
                if fault=="unpinned-state": self.pinned=False
                elif fault=="state-dtype": self.dtype=dtypes["float32"]
                elif fault=="state-bytes": self.host=np.zeros_like(host)
                elif fault=="shape": self.shape=(host.size,)
                elif fault=="source-gpu": self.device=gpu
                elif fault=="host-alias": self.ptr=100_000_000
                elif fault=="host-dst-alias": self.ptr=10_000_000
                elif fault=="entry-change": wp.copy=lambda *a,**k:None
                elif fault=="stream-change": stream.cuda_stream=56
        def numpy(self):
            assert self.device is cpu, "GPU numpy access forbidden"
            return self.host
    counter=[0]
    arrays={}
    for i,(n,(logical,host,dtype)) in enumerate(specs.items()):
        a=object.__new__(Array)
        a.shape,a.dtype,a.device,a.ptr,a.pinned=logical,dtypes[dtype],gpu,10_000_000+i*4_000_000,False
        arrays[n]=a
    if fault=="dst-alias": arrays["efc.state"].ptr=arrays["efc.force"].ptr
    def copy(dst,src,*,stream):
        assert src.device is cpu and stream.device is gpu
        events.append(("copy",dst.dtype,src.dtype))
    def sync(s):
        assert s is stream
        events.append(("sync",))
    wp=NS(array=Array,copy=copy,synchronize_stream=sync,**dtypes)
    if fault is None:
        r=probe.copy_control(wp,arrays,stream,lambda:None,packet)
        assert len(events)==26 and sum(e[0]=="copy" for e in events)==13
        assert all(e[1] is e[2] for e in events if e[0]=="copy")
        assert sum(e[1] is dtypes["int32"] for e in events if e[0]=="copy")==1
        assert r["preflight_complete"] and not r["gpu_numpy_readback"]
    else:
        with pytest.raises(ValueError): probe.copy_control(wp,arrays,stream,lambda:None,packet)
        assert events==[]


def test_owner_refuses_cuda_and_missing_cpu_evidence_first(monkeypatch):
    monkeypatch.setattr(probe,"source_binding",lambda *_:pytest.fail("premature source"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CPU-only"): probe.owner(NS(child=False))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    with pytest.raises(ValueError,match="explicit paired"): probe.owner(NS(child=False,cpu_evidence=None))


def test_child_refuses_wrong_parent_before_read(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    monkeypatch.setattr(probe,"read_plain",lambda *_:pytest.fail("premature read"))
    with pytest.raises(ValueError,match="fresh inherited"):
        probe.child(NS(source="a"*40,child=True,owner_pid=-1))


@pytest.mark.parametrize("fault", (None,"source","files","flags","hash"))
def test_exact_source_nineteen_file_cpu_prerequisite(monkeypatch,fault):
    source="a"*40
    path=probe.ROOT/"artifacts/tools/mock-cpu-evidence.json"
    xml=b'<testsuites><testsuite tests="900" errors="0" failures="0" skipped="0"/></testsuites>'
    anchors={p:dict(path=str(probe.ROOT/("artifacts/tools/mock-"+p+".xml")),bytes=len(xml),
                   sha256=sha256(xml).hexdigest(),tests=900) for p in ("mac","native")}
    value=dict(source=source,test_files=list(probe.TESTS),flags=dict(stages.FLAGS),**anchors)
    if fault=="source": value["source"]="b"*40
    elif fault=="files": value["test_files"].pop()
    elif fault=="flags": value["flags"]["training_authorized"]=True
    elif fault=="hash": value["native"]["sha256"]="0"*64
    raw=probe.canonical(value)
    monkeypatch.setattr(probe,"read_plain",lambda p:raw if p==path else xml)
    monkeypatch.setattr(probe,"read_json",lambda _:value)
    if fault is None:
        assert probe.tests_record(path,source)["evidence"]==value
    else:
        with pytest.raises(ValueError): probe.tests_record(path,source)


@pytest.mark.parametrize("name", ("CostModuleExecutable","CostDispatchObserver"))
def test_actual_immutable_constructor_signature(name):
    constructor=getattr(probe.executable,name) if name=="CostModuleExecutable" else probe.executable.dispatch.CostDispatchObserver
    calls=[n for n in ast.walk(ast.parse(inspect.getsource(probe.child))) if isinstance(n,ast.Call)
           and (getattr(n.func,"id",None)==name or getattr(n.func,"attr",None)==name)]
    assert len(calls)==1 and not calls[0].args
    inspect.signature(constructor).bind(**{k.arg:object() for k in calls[0].keywords})


def test_new_fence_and_no_runtime_monkeypatches():
    assert len(probe.OWN)==4 and len(probe.TESTS)==19 and probe.TESTS[:-1]==transition.TESTS
    assert old_probe.BASE=="98aa8a1d1a413f76cadc397492eddb8cb953135e"
    assert probe.BASE=="a6d65d9b90650900a6c736913a967caf99ae5638"
    for module in (probe,receiver):
        tree=ast.parse(inspect.getsource(module))
        writes=[target for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))
                for target in (n.targets if isinstance(n,ast.Assign) else [n.target])]
        assert not any(isinstance(t,ast.Attribute) and getattr(t.value,"id",None)
                       in {"old_probe","stages","transition","retained","executable"} for t in writes)
    code="from mjlab_microduck import stance_solver_efc_probe,stance_solver_efc_receiver; import sys; assert not any(n.split('.')[0] in {'warp','torch','mujoco','mujoco_warp','numpy'} for n in sys.modules)"
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10)
