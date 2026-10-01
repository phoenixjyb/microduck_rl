"""Packed qualification wiring, not CUDA timing or learner admission."""

from copy import deepcopy
import pytest

from mjlab_microduck import stance_wsl_qualification as q
from mjlab_microduck import stance_solved_field_integration as integration


@pytest.mark.parametrize('mode', [None, True, 1, '', 'PACKED', 'shadow'])
def test_unknown_checker_mode_is_refused(mode):
    with pytest.raises(ValueError, match='explicit qualification checker mode'):
        q.output_path('a'*40, mode)


def test_packed_paths_do_not_replace_old_qualification():
    assert q.output_path('a'*40).name == 'stance-wsl-qualification-aaaaaaaaaaaa'
    assert q.output_path('a'*40, 'packed').name == 'stance-wsl-packed-qualification-aaaaaaaaaaaa'
    assert q.service_name('a'*40, 'packed') == 'microduck-wsl-packed-qualification-aaaaaaaaaaaa.service'


def test_new_plan_binds_checker_and_original_budget(monkeypatch):
    monkeypatch.setattr(q.execution, 'PROFILE', q.execution.select(q.execution.WSL))
    monkeypatch.setattr(q.host, 'identity', lambda source: dict(source=source))
    monkeypatch.setattr(q, 'validate_timing_basis', lambda source: dict(pinned=True))
    legacy = q.plan('a'*40, 42)
    packed = q.plan('a'*40, 42, 'packed')
    assert legacy['protocol'] == q.PROTOCOL
    assert 'solved_field_check' not in legacy
    assert packed['protocol'] == q.PACKED_PROTOCOL
    assert packed['solved_field_check'] == 'packed'
    assert packed['same_input_integration_protocol'] == integration.PROTOCOL
    assert not packed['separate_trajectory_equivalence_claimed']
    assert len(packed['checker_sha256']) == 64
    assert packed['child_seconds'] == 900 and packed['service_seconds'] == 960
    assert packed['training_budget']['child_seconds'] == 4320
    assert packed['training_budget']['service_seconds'] == 4380
    assert packed['memory_max_bytes'] == 6*1024**3
    assert not packed['acceptance_gates_changed'] and not packed['checkpoint_exported']


@pytest.mark.parametrize('deadline', [True, 42.0, q.PACKED_CUTOFF+1])
def test_packed_plan_cannot_exceed_authorized_cutoff(monkeypatch, deadline):
    monkeypatch.setattr(q.execution, 'PROFILE', q.execution.select(q.execution.WSL))
    monkeypatch.setattr(q.host, 'identity', lambda source: dict(source=source))
    monkeypatch.setattr(q, 'validate_timing_basis', lambda source: dict(pinned=True))
    with pytest.raises(ValueError, match='authorized October 1 cutoff'):
        q.plan('a'*40, deadline, 'packed')


@pytest.mark.parametrize('qualified', [False, True])
@pytest.mark.parametrize('damage', [None, 'decision', 'unexpected-file', 'hash', 'result'])
def test_replay_retains_rejection_but_learner_verify_refuses(tmp_path, monkeypatch, qualified, damage):
    names = {'parent.pt', 'runtime.json', 'launch.json', 'integration.json',
             'checker-integration.json', 'collection.json', 'optimizer.json', 'child.log'}
    decision = 'qualified-for-bounded-replication' if qualified else 'timing-rejected-no-training'
    derived = dict(decision=decision, caps=dict(child_seconds=1))
    report = dict(protocol=q.PACKED_PROTOCOL, launch_sha256='a'*64,
                  decision=decision, result=deepcopy(derived), files={name:'b'*64 for name in names})
    if damage == 'decision': report['decision'] = 'timing-rejected-no-training' if qualified else 'qualified-for-bounded-replication'
    elif damage == 'unexpected-file': report['files']['unknown.json'] = 'b'*64
    elif damage == 'hash': report['files']['collection.json'] = 'c'*64
    elif damage == 'result': report['result']['caps']['child_seconds'] = 2
    for name in report['files']:
        (tmp_path/name).touch()
    (tmp_path/'report.json').touch()
    monkeypatch.setattr(q, 'output_path', lambda *_: tmp_path)
    monkeypatch.setattr(q.files, 'file_bytes', lambda path: q.canonical(report if path.name == 'report.json' else {}).encode())
    monkeypatch.setattr(q.host, 'digest', lambda _: 'b'*64)
    monkeypatch.setattr(q, 'checked', lambda *_: dict(protocol=q.PACKED_PROTOCOL, source='d'*40))
    monkeypatch.setattr(q.host, 'validate_payload', lambda *_: None)
    monkeypatch.setattr(q, 'verify_packed', lambda *_: None)
    monkeypatch.setattr(q, 'derive', lambda *_: derived)
    if damage is not None:
        with pytest.raises(ValueError): q.replay('d'*40, 'packed', require_qualified=False)
    else:
        assert q.replay('d'*40, 'packed', require_qualified=False)['result']['decision'] == decision
        if qualified:
            assert q.verify('d'*40, 'packed')['result']['decision'] == decision
        else:
            with pytest.raises(ValueError, match='WSL training timing gate'):
                q.verify('d'*40, 'packed')


@pytest.mark.parametrize('damage', [None, 'source', 'checker', 'mode', 'runtime-mode', 'collection-checker'])
def test_source_bound_predicates_and_collection_required(tmp_path, monkeypatch, damage):
    launch = dict(checker_sha256='b'*64)
    receipt = dict(launch_sha256='a'*64, checker_sha256='b'*64)
    collection = dict(solved_field_check='packed', runtime_solved_field_check='packed', checker_sha256='b'*64)
    if damage == 'source': receipt['launch_sha256'] = 'c'*64
    elif damage == 'checker': receipt['checker_sha256'] = 'c'*64
    elif damage == 'mode': collection['solved_field_check'] = 'legacy'
    elif damage == 'runtime-mode': collection['runtime_solved_field_check'] = 'legacy'
    elif damage == 'collection-checker': collection['checker_sha256'] = 'c'*64
    rows = {'checker-integration.json': receipt, 'collection.json': collection}
    monkeypatch.setattr(q.files, 'file_bytes', lambda path: q.canonical(rows[path.name]).encode())
    monkeypatch.setattr(q.host, 'digest', lambda _: 'a'*64)
    checked = []
    monkeypatch.setattr(integration, 'validate', lambda value, device: checked.append((deepcopy(value),device)))
    if damage is not None:
        with pytest.raises(ValueError): q.verify_packed(tmp_path, launch)
    else:
        q.verify_packed(tmp_path, launch)
        assert checked == [(receipt,'cuda:0')]
