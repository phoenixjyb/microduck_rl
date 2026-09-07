"""CPU evidence reconciliation cannot silently re-execute a policy."""

import json
import pytest
from mjlab_microduck import foundation_lateral_reconcile as reconcile


def test_requires_explicit_cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CPU-only"):reconcile.reconcile("a"*40)


def test_changed_manifest_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(reconcile,"RETAINED",tmp_path)
    (tmp_path/"manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="immutable closeout manifest"):reconcile.read_report()


def test_retained_rejection_zero_execution_and_no_overwrite(tmp_path,monkeypatch):
    output=tmp_path/"cpu"
    monkeypatch.setattr(reconcile,"OUTPUT",output)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(reconcile,"verify_source",lambda s:None)
    monkeypatch.setattr(reconcile.closeout.exp.training,"runtime_identity",lambda:{})
    monkeypatch.setattr(reconcile.closeout,"verify_retained",lambda:{})
    monkeypatch.setattr(reconcile,"read_report",lambda:{})
    monkeypatch.setattr(reconcile.closeout,"paired",lambda r:dict(failures=["settled-legacy-torque"]))
    def forbidden(*a,**k):raise AssertionError("must not simulate or train")
    monkeypatch.setattr(reconcile.closeout.exp,"run_control",forbidden)
    monkeypatch.setattr(reconcile.closeout.exp.training,"train",forbidden)
    result=reconcile.reconcile("a"*40)
    assert result["decision"]=="numerical-gate-stop" and result["simulation_steps_executed"]==0
    assert result["optimizer_updates"]==0 and not result["policy_acceptance"]
    assert json.loads((output/"decision.json").read_text())==result
    with pytest.raises(FileExistsError):reconcile.reconcile("a"*40)
