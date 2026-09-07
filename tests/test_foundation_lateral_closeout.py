"""Only the never-started evaluation may run; no training or original mutation."""

import datetime as dt
import json
import sys
from types import SimpleNamespace as NS

import pytest
from mjlab_microduck import foundation_lateral_closeout as closeout


@pytest.mark.parametrize("failure",[None,"gate","busy","child","retained"])
def test_one_child_no_training_replay_and_closed_artifact(tmp_path,monkeypatch,failure):
    output=tmp_path/"closeout";calls=[]
    monkeypatch.setattr(closeout,"OUTPUT",output)
    monkeypatch.setattr(closeout,"LAST_START",dt.datetime.max.replace(tzinfo=dt.timezone.utc))
    monkeypatch.setattr(closeout,"verify_source",lambda s:None)
    monkeypatch.setattr(closeout.exp.training,"runtime_identity",lambda:{})
    verifies=[]
    def verify():
        verifies.append(True)
        if failure=="retained" and len(verifies)>1:raise ValueError("changed original evidence")
        return {}
    monkeypatch.setattr(closeout,"verify_retained",verify)
    def idle():
        if failure=="busy":raise ValueError("busy")
        return {"samples":[{},{}]}
    monkeypatch.setattr(closeout,"wait_idle",idle)
    monkeypatch.setattr(closeout,"paired",lambda r:dict(failures=["gate"] if failure=="gate" else [],policy_acceptance=False))
    monkeypatch.setattr(sys,"argv",["closeout","--source","b"*40])
    def child(args,**kwargs):
        calls.append(args)
        assert kwargs["timeout"]==90 and args[1:3]==["-m","mjlab_microduck.foundation_lateral_closeout"]
        closeout.write_new(output/"lateral-s467.json",{"synthetic":True})
        return NS(returncode=1 if failure=="child" else 0)
    monkeypatch.setattr(closeout.subprocess,"run",child)
    if failure in ("busy","child","retained"):
        with pytest.raises(ValueError):closeout.main()
    else:closeout.main()
    assert len(calls)==(0 if failure=="busy" else 1)
    d=json.loads((output/"decision.json").read_text())
    assert d["decision"]==({None:"paired-case-support-only","gate":"numerical-gate-stop"}.get(failure,"runtime-failure-stop"))
    assert not d["policy_acceptance"] and not d["hop_validated"]
    m=json.loads((output/"manifest.json").read_text())
    for name,row in m["files"].items():assert closeout.sha256(output/name)==row["sha256"]
    monkeypatch.setattr(closeout,"verify_retained",lambda:{})
    with pytest.raises(FileExistsError):closeout.main()


def test_changed_original_manifest_refuses_before_payload_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(closeout,"ORIGINAL",tmp_path)
    (tmp_path/"manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="immutable original evidence"):closeout.verify_retained()
