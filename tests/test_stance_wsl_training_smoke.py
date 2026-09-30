"""Disposable WSL smoke stays separate from failed long-run qualification."""
import pytest
from mjlab_microduck import stance_wsl_training_smoke as w


def test_declared_short_training_plan_never_admits_replication(monkeypatch):
    monkeypatch.setattr(w.execution,'PROFILE',w.execution.select(w.execution.WSL))
    monkeypatch.setattr(w.host,'identity',lambda source:dict(source=source))
    monkeypatch.setattr(w.q,'validate_timing_basis',lambda source:dict(recorded_timing_rejection=True))
    value=w.plan('a'*40,1,'b'*64)
    assert (value['worlds'],value['updates'],value['seed'],value['steps_per_update'])==(64,16,523,24)
    assert (value['child_seconds'],value['service_seconds'])==(900,960)
    assert not value['long_replication_authorized']
    assert not value['pilot_parent_authorized']
    assert not value['learned_stance']
    assert not value['physical_motion_authorized']
    assert value['purpose']=='smoke' and value['initialization']=='fresh-not-parent-weights'


def test_unknown_or_wrong_host_cannot_prepare_a_smoke(monkeypatch):
    monkeypatch.setattr(w.execution,'PROFILE',w.execution.select(w.execution.DEFAULT))
    with pytest.raises(ValueError,match='fixed authorized WSL'):
        w.plan('a'*40,1,'b'*64)


def test_seed_and_exports_remain_disposable_not_replication():
    assert w.smoke.SEED not in w.q.lean.REPLICATION['seeds']
    assert w.output_path('a'*40).name=='stance-wsl-training-smoke-aaaaaaaaaaaa'
    assert w.service_name('a'*40)=='microduck-wsl-training-smoke-aaaaaaaaaaaa.service'
    assert w.smoke.UPDATES==16 and w.files.LEAN_LESSON_CHILD_SECONDS==1693
