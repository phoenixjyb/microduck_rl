"""Pure runner-contract checks, not native PPO or capability evidence."""
from copy import deepcopy

import pytest

from mjlab_microduck import stance_recovery_ppo_probe as probe

SOURCE = 'a' * 40
DIGEST = 'b' * 64


def run_properties():
    return dict(MainPID='1234', ActiveState='active', RuntimeMaxUSec='3min',
        MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group')


def test_fixed_training_pair_caps_and_deadline_reserve():
    declaration = probe.declaration(SOURCE)
    assert declaration['split'] == 'training'
    assert declaration['cell_ids'] == ['zero-wrench', '+x-2n-20steps-t250']
    assert declaration['worlds'] == 2
    assert (probe.SERVICE_SECONDS, probe.CLOSEOUT_SECONDS, probe.LAUNCH_RESERVE) == (180, 120, 360)
    assert probe.evidence.HORIZON == 28 and probe.evidence.SEED == 653
    assert probe.PRE_REPORT_FILES | {'report.json'} == probe.COMPLETE_FILES
    probe.window.check(now=probe.window.START, reserve_seconds=probe.LAUNCH_RESERVE)
    with pytest.raises(ValueError):
        probe.window.check(now=probe.window.CUTOFF-probe.LAUNCH_RESERVE,
                           reserve_seconds=probe.LAUNCH_RESERVE)


def test_exact_service_caps_and_no_owned_overlap(monkeypatch):
    name = probe.service_name(SOURCE, 'run')
    properties = dict(MainPID=str(probe.os.getpid()), ActiveState='active',
        RuntimeMaxUSec='3min', MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s',
        Nice='10', KillMode='control-group')
    running = [name + ' loaded active running fake-test-description']
    monkeypatch.setattr(probe.base.host, 'read',
        lambda *args: properties[args[-2]] if args[2] == 'show' else '\n'.join(running))
    assert probe.service_properties(SOURCE, 'run') == properties
    properties['RuntimeMaxUSec'] = '4min'
    with pytest.raises(ValueError, match='exact independently capped'):
        probe.service_properties(SOURCE, 'run')
    properties['RuntimeMaxUSec'] = '3min'
    running.append('microduck-other.service loaded active running unrelated')
    with pytest.raises(ValueError, match='only this owned Duck service'):
        probe.service_properties(SOURCE, 'run')


@pytest.mark.parametrize('accepted', [False, True])
def test_complete_retained_capture_is_not_yet_independently_qualified(accepted):
    capture = dict(collection=dict(accepted_complete=accepted))
    report = dict(protocol=probe.PROTOCOL, source=SOURCE, launch_sha256=DIGEST,
        capture=capture, decision=probe._capture_decision(capture['collection']),
        optimizer_steps=0, training_update_performed=False, cuda_initialized=False,
        service_properties=run_properties(), postchecks_passed=True, elapsed_seconds=4.0,
        source_unchanged=True, filmbrain_unchanged=True, protected_services_inactive=True,
        files={name: 'f'*64 for name in probe.PRE_REPORT_FILES}, **probe.base.baseline.FALSE_FLAGS)
    assert probe._check_report(report, capture, SOURCE, DIGEST, run_properties()) is True
    assert 'qualified' not in report['decision']
    for key, damaged in [('training_update_performed', True), ('optimizer_steps', 1),
                         ('cuda_initialized', True), ('training_admitted', True),
                         ('decision', 'cpu-ppo-transition-integration-qualified')]:
        changed = deepcopy(report); changed[key] = damaged
        with pytest.raises(ValueError, match='complete unupdated capture report'):
            probe._check_report(changed, capture, SOURCE, DIGEST, run_properties())


def test_post_retention_failure_preserves_linkage_but_cannot_qualify():
    capture = dict(collection=dict(accepted_complete=True))
    report = dict(protocol=probe.PROTOCOL, source=SOURCE, launch_sha256=DIGEST,
        capture=capture, decision='cpu-ppo-transition-postcheck-failed-retained',
        optimizer_steps=0, training_update_performed=False, cuda_initialized=False,
        service_properties=run_properties(), postchecks_passed=False, elapsed_seconds=4.0,
        error_type='ValueError', error='postcheck failed',
        files={name: 'f'*64 for name in probe.PRE_REPORT_FILES}, **probe.base.baseline.FALSE_FLAGS)
    assert probe._check_report(report, capture, SOURCE, DIGEST, run_properties()) is False
    changed = deepcopy(report); changed['postchecks_passed'] = True
    changed['decision'] = probe._capture_decision(capture['collection'])
    with pytest.raises(ValueError, match='hidden failure'):
        probe._check_report(changed, capture, SOURCE, DIGEST, run_properties())


@pytest.mark.parametrize('damage', ['cap', 'properties', 'elapsed-nan', 'elapsed-negative'])
def test_closeout_rejects_malformed_recorded_service_and_elapsed_receipts(damage):
    capture = dict(collection=dict(accepted_complete=True))
    props = run_properties()
    report = dict(protocol=probe.PROTOCOL, source=SOURCE, launch_sha256=DIGEST,
        capture=capture, decision=probe._capture_decision(capture['collection']),
        optimizer_steps=0, training_update_performed=False, cuda_initialized=False,
        service_properties=deepcopy(props), postchecks_passed=True, elapsed_seconds=4.0,
        source_unchanged=True, filmbrain_unchanged=True, protected_services_inactive=True,
        files={name: 'f'*64 for name in probe.PRE_REPORT_FILES}, **probe.base.baseline.FALSE_FLAGS)
    if damage == 'cap':
        props['RuntimeMaxUSec'] = '4min'
    elif damage == 'properties':
        report['service_properties']['KillMode'] = 'process'
    elif damage == 'elapsed-nan':
        report['elapsed_seconds'] = float('nan')
    else:
        report['elapsed_seconds'] = -1.0
    with pytest.raises(ValueError):
        probe._check_report(report, capture, SOURCE, DIGEST, props)
    report['elapsed_seconds'] = 180.0
    report['service_properties'] = run_properties()
    assert probe._check_report(report, capture, SOURCE, DIGEST, run_properties()) is False


def test_malformed_capture_status_fails_closed():
    for value in (None, {}, {'accepted_complete': 1}):
        with pytest.raises(ValueError, match='typed retained collection outcome'):
            probe._capture_decision(value)
