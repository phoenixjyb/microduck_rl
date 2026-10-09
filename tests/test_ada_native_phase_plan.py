"""Pure phase blueprint tests; do not execute the prospective native pipeline."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from mjlab_microduck import ada_native_phase_plan as p


def static():
    m=dict(scalars={k:dict(type="int",value=v) for k,v in p.GUARDS.items()},exclusions=[])
    return [deepcopy(m),deepcopy(m)]


def test_lexical_body_ignores_comment_string_and_nested_braces():
    s='void mj_fake(int x) { /* } */ if (x) { char *s="}"; } // {\n return; }\nvoid mj_next() {}'
    body=p.c_body(s,"mj_fake")
    assert body.endswith("return; }") and "mj_next" not in body


@pytest.mark.parametrize("text",["void mj_fake() {", "int mj_fake() {}", "void mj_other() {}", "void mj_fake() {}\nvoid mj_fake() {}"])
def test_missing_duplicate_nonvoid_unterminated_source_refuses(text):
    with pytest.raises(ValueError):p.c_body(text,"mj_fake")


@pytest.mark.parametrize("name",["mj_fake;", "not_mj", "mj_fake()", "mj_fake_name", "mj_fake1"])
def test_nonplain_function_name_refuses(name):
    with pytest.raises(ValueError):p.c_body("void mj_fake() {}",name)


def test_exact_model_guards_no_input_mutation():
    s=static();b=deepcopy(s)
    assert p.model_guards(s)==[s[0]["scalars"],s[1]["scalars"]] and s==b


@pytest.mark.parametrize("field",sorted(p.GUARDS))
def test_each_native_model_guard_is_required(field):
    s=static()
    for m in s:m["scalars"].pop(field)
    with pytest.raises(ValueError):p.model_guards(s)


@pytest.mark.parametrize("damage",["bool","float","wrong","kind","extra_item","missing_world","extra_world","different_world","meta_extra","meta_missing","tuple"])
def test_phase_domain_and_descriptor_refuse(damage):
    s=static();k="model/nplugin"
    if damage=="bool":
        for m in s:m["scalars"][k]["value"]=False
    if damage=="float":
        for m in s:m["scalars"][k]["value"]=0.
    if damage=="wrong":
        for m in s:m["scalars"][k]["value"]=1
    if damage=="kind":
        for m in s:m["scalars"][k]["type"]="bool"
    if damage=="extra_item":
        for m in s:m["scalars"][k]["extra"]=0
    if damage=="missing_world":s.pop()
    if damage=="extra_world":s.append(deepcopy(s[0]))
    if damage=="different_world":s[1]["scalars"][k]["value"]=1
    if damage=="meta_extra":
        for m in s:m["extra"]=0
    if damage=="meta_missing":
        for m in s:m.pop("exclusions")
    if damage=="tuple":s=tuple(s)
    with pytest.raises(ValueError):p.model_guards(s)


def test_source_audit_exact_hash_and_all_calls(tmp_path,monkeypatch):
    bodies=[]
    for n in p.SOURCE_FUNCTIONS:
        inner=""
        if n=="mj_forwardSkip":inner="".join(c+"(m,d);" for c in p.FORWARD_ORDER)
        if n=="mj_forward":inner="mj_forwardSkip(m,d,mjSTAGE_NONE, 0);"
        bodies.append("void "+n+"() {"+inner+"}\n")
    raw="".join(bodies).encode().ljust(64281,b" ");f=tmp_path/"forward.c";f.write_bytes(raw)
    monkeypatch.setattr(p,"SOURCE_SHA",sha256(raw).hexdigest())
    r=p.source_audit(f)
    assert r["lexical_forward_call_order"]==list(p.FORWARD_ORDER)
    assert set(r["named_function_body_sha256"])==set(p.SOURCE_FUNCTIONS)
    assert not r["transitive_machine_read_set_proven"]
    f.write_bytes(raw[:-1]+b"x")
    with pytest.raises(ValueError):p.source_audit(f)


def test_source_symlink_and_wrong_size_refused(tmp_path):
    f=tmp_path/"source";f.write_bytes(b"x");link=tmp_path/"link";link.symlink_to(f)
    for path in (f,link,tmp_path/"absent"):
        with pytest.raises(ValueError):p.source_audit(path)


def test_contract_stays_unexecuted_not_a_promotion():
    assert p.BEFORE[-1]=="mj_fwdAcceleration" and p.AFTER==("mj_fwdConstraint",)
    assert "mj_sensorAcc" not in p.BEFORE+p.AFTER
    assert p.CONTRACT["new_model_allocations"]==p.CONTRACT["new_solver_calls"]==0
    assert all(p.CONTRACT[k] is False for k in ("runtime_imported","execution_ready","training_authorized","native_before_solve_capture_executed","simulator_qualified","physical_motion_authorized"))


def test_module_import_stays_pure_in_fresh_process():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    code='import sys; from mjlab_microduck import ada_native_phase_plan as p; assert not {"warp","torch","mujoco","mujoco_warp"}&sys.modules.keys(); assert not p.CONTRACT["execution_ready"]'
    result=subprocess.run([sys.executable,"-c",code],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_whole_receiver_refuses_modified_descriptor(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"source_check",lambda source:None)
    expected=dict(protocol=p.PROTOCOL,decision=p.DECISION,**p.CONTRACT)
    reference=deepcopy(expected)
    monkeypatch.setattr(p,"analyze",lambda *args:deepcopy(reference))
    expected.update(source="a"*40,module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    f=tmp_path/"report.json";f.write_text(json.dumps(expected))
    assert p.receive(f,"a"*40,"input","measured","tools")==expected
    expected["execution_ready"]=True;f.write_text(json.dumps(expected))
    with pytest.raises(ValueError):p.receive(f,"a"*40,"input","measured","tools")
