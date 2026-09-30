"""CPU fixtures test timing decisions, never claim actual WSL qualification."""
from copy import deepcopy
import pytest
from mjlab_microduck import stance_wsl_qualification as q


def measurements(seconds=4.0):
    return (dict(physics_device='cuda:0', optimizer_steps=0, worlds=64, ticks_per_update=24,
                 summarize=q.throughput.summarize([seconds]*8), setup_seconds=1.0,
                 qualification_setup_seconds=20.0),
            dict(device='cpu', stand_in=True, physics_evidence=False,
                 summarize=q.throughput.summarize([0.1]*8)))


def test_faster_measurement_fits_without_widening_original_caps():
    result = q.derive(*measurements())
    assert result['decision'] == 'qualified-for-bounded-replication'
    assert (result['existing_child_seconds'], result['existing_service_seconds']) == (1693,1753)
    assert result['caps']['setup_seconds'] == 21.0
    assert not result['full_evaluation_timing_qualified']
    assert not result['learned_stance_accepted']
    assert not result['football_balance_accepted']
    assert not result['physical_motion_authorized']


def test_slower_host_is_rejected_not_given_more_time():
    result = q.derive(*measurements(7.0))
    assert result['decision'] == 'timing-rejected-no-training'
    assert result['existing_child_seconds'] == 1693


@pytest.mark.parametrize('damage', ['device','worlds','count','summary','optimizer','finite','setup'])
def test_wrong_or_nonfinite_measurement_cannot_qualify(damage):
    collection, optimizer = deepcopy(measurements())
    if damage == 'device': collection['physics_device'] = 'cpu'
    elif damage == 'worlds': collection['worlds'] = 128
    elif damage == 'count': collection['summarize'] = q.throughput.summarize([1.0]*7)
    elif damage == 'summary': collection['summarize']['max'] = 0.1
    elif damage == 'optimizer': optimizer['physics_evidence'] = True
    elif damage == 'finite': collection['summarize']['series'][0] = float('nan')
    else: collection['qualification_setup_seconds'] = -100.0
    with pytest.raises(ValueError): q.derive(collection, optimizer)


def test_qualification_does_not_reuse_the_expired_cuda_probe_window():
    assert q.PROTOCOL == 'football-b1n-wsl-qualification-v1'
    assert q.service_name('a'*40) == 'microduck-wsl-qualification-aaaaaaaaaaaa.service'
    assert q.output_path('a'*40).name == 'stance-wsl-qualification-aaaaaaaaaaaa'
    with pytest.raises(ValueError): q.prepare('a'*40, int(q.host.CUTOFF))


def test_wsl_training_binds_qualification_and_only_allows_replication(monkeypatch):
    monkeypatch.setattr(q.execution, 'PROFILE', q.execution.select(q.execution.WSL))
    evidence = dict(report_sha256='b'*64, source='a'*40)
    monkeypatch.setattr(q, 'verify', lambda source: evidence)
    launch = q.lean.plan('a'*40, {}, 'c'*64, 1, q.lean.REPLICATION, 577)
    assert launch['host_qualification'] == evidence
    with pytest.raises(ValueError, match='only the declared replication'):
        q.lean.plan('a'*40, {}, 'c'*64, 1)


def test_wsl_evaluation_cannot_borrow_4090_measurement(monkeypatch):
    from mjlab_microduck import stance_lean_evaluation as ev
    monkeypatch.setattr(q.execution, 'PROFILE', q.execution.select(q.execution.WSL))
    with pytest.raises(ValueError, match='separately measured timing gate'):
        ev.prepare('a'*40, 1, 'evaluate')
