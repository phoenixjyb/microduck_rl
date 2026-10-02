"""R2 CPU contracts and synthetic replay guards, never a live GPU acceptance."""
from copy import deepcopy
from hashlib import sha256
import pytest

from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_packed_evaluation_probe as probe
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.first_attempt_smoke import canonical
from test_stance_packed_evaluation_probe import (
    SOURCE, TRAINING_SOURCE, RUNTIME_SHA, retained_training, select_wsl,
    _retained_probe_fixture, portable_retained_training, historical_auth_fixture,
)


def portable_plan(window=probe.PORTABLE_WINDOW):
    return probe.plan(SOURCE, {'execution_profile': probe.host.execution.select(probe.host.execution.WSL)},
        RUNTIME_SHA, portable_retained_training(), probe.WINDOWS[window]['cutoff'],
        window=window, training_auth=historical_auth_fixture())


def test_pure_portable_plan_records_distinct_profile_bound_protocol_without_current_wsl_claim(monkeypatch):
    monkeypatch.setattr(probe.host.execution, 'PROFILE', probe.host.execution.select(probe.host.execution.DEFAULT))
    launch = portable_plan()
    binding = launch['cases'][0]['binding']
    assert launch['protocol'] == evaluation.PORTABLE_PROBE['protocol'] != probe.PROTOCOL
    assert binding['protocol'] == trace.PORTABLE_PROBE_PROTOCOL
    assert launch['cpu_math_profile'] == binding['cpu_math_profile'] == profile.expected_receipt()
    assert launch['cpu_math_profile_sha256'] == sha256((canonical(profile.expected_receipt())+'\n').encode()).hexdigest()
    assert launch['attempts_required'] == 128 and launch['optimizer_steps'] == 0
    assert launch['child_timeout_seconds'] == 600 and launch['service_timeout_seconds'] == 960
    assert not launch['full_evaluation_enabled'] and not launch['learned_stance_accepted']
    auth = launch['historical_training_authentication']
    assert auth == historical_auth_fixture()
    assert launch['historical_training_authentication_sha256'] == sha256((canonical(auth)+'\n').encode()).hexdigest()
    assert evaluation.PORTABLE_PROBE not in evaluation.EVALUATIONS.values()
    with pytest.raises(ValueError, match='only declared judged continuations'):
        evaluation.plan(SOURCE, {}, RUNTIME_SHA, retained_training(), 1234, 'evaluate', evaluation.PORTABLE_PROBE)


@pytest.mark.parametrize('damage', ['missing', 'setting', 'legacy_protocol', 'wrong_worlds', 'wrong_seed'])
def test_portable_trace_profile_cannot_be_missing_changed_or_borrowed_by_legacy(damage):
    binding = deepcopy(portable_plan()['cases'][0]['binding'])
    trace.validate_binding(binding)
    if damage == 'missing': binding.pop('cpu_math_profile')
    elif damage == 'setting': binding['cpu_math_profile']['settings']['MKL_CBWR'] = 'AUTO'
    elif damage == 'legacy_protocol': binding['protocol'] = trace.PACKED_PROBE_PROTOCOL
    elif damage == 'wrong_worlds': binding['worlds'] = 64
    else: binding['evaluation_seed'] = 547
    with pytest.raises(ValueError): trace.validate_binding(binding)


def test_portable_bundle_refuses_wrong_actual_profile_before_checkpoint_or_inference(monkeypatch):
    binding = portable_plan()['cases'][0]['binding']
    monkeypatch.setattr(profile, 'checked_receipt', lambda: {})
    monkeypatch.setitem(bundle.LOADERS, trace.PORTABLE_PROBE_PROTOCOL,
        lambda *_: pytest.fail('wrong profile must precede checkpoint loading'))
    with pytest.raises(ValueError, match='actual recorded CPU math profile'):
        bundle.checked_inputs(binding, b'not tensors', {}, b'not runtime', b'not launch')
    with pytest.raises(ValueError, match='actual recorded CPU math profile'):
        bundle.actor_replay({'binding': binding, 'ticks': []}, None, {})


def test_portable_child_exec_environment_is_explicit_and_does_not_leak_other_variables(monkeypatch):
    monkeypatch.setattr(profile, 'checked_receipt', profile.expected_receipt)
    monkeypatch.setattr(probe.files, 'child_environment', lambda: {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1'})
    monkeypatch.setenv('UNRELATED_PRIVATE_KEY', 'not-in-child')
    result = probe.child_environment(portable_plan())
    assert result == {'CUDA_VISIBLE_DEVICES': '0', **profile.settings()}
    assert probe.child_environment({'execution_window': probe.REPAIRED_WINDOW}) == {
        'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1'}
    with pytest.raises(ValueError, match='explicit declared packed probe window'):
        probe.child_environment({'execution_window': 'not-declared'})
    with pytest.raises(ValueError, match='explicit declared packed probe window'):
        probe.check_cpu_profile('not-declared')


def test_fresh_window_cannot_borrow_old_authority_or_outlast_cutoff(monkeypatch):
    select_wsl(monkeypatch)
    limits = probe.WINDOWS[probe.PORTABLE_WINDOW]
    assert limits['cutoff'] - limits['not_before'] == 3600
    monkeypatch.setattr(probe.time, 'time', lambda: limits['not_before']-1)
    with pytest.raises(ValueError, match='before its declared start'):
        probe.check_window(limits['cutoff'], window=probe.PORTABLE_WINDOW)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['cutoff']-1620)
    with pytest.raises(ValueError, match='whole packed probe plus closeout'):
        probe.check_window(limits['cutoff'], launching=True, window=probe.PORTABLE_WINDOW)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['cutoff']+1)
    with pytest.raises(ValueError): probe.check_window(limits['cutoff'], window=probe.PORTABLE_WINDOW)
    with pytest.raises(ValueError): probe.checked_deadline(limits['cutoff']+1, probe.PORTABLE_WINDOW)
    with pytest.raises(ValueError): probe.checked_deadline(limits['cutoff'], probe.REPAIRED_WINDOW)


def test_portable_prepare_missing_exec_profile_refuses_before_identity_or_allocation(monkeypatch):
    select_wsl(monkeypatch)
    limits = probe.WINDOWS[probe.PORTABLE_WINDOW]
    monkeypatch.setattr(probe.time, 'time', lambda: limits['not_before'])
    monkeypatch.setattr(profile, 'checked_receipt', lambda: {})
    monkeypatch.setattr(probe.host, 'identity', lambda *_: pytest.fail('profile refusal before identity'))
    with pytest.raises(ValueError, match='actual recorded CPU math profile'):
        probe.prepare(SOURCE, TRAINING_SOURCE, 577, limits['cutoff'], window=probe.PORTABLE_WINDOW)


def test_offline_portable_receipt_rederives_on_non_wsl_runtime_without_gpu_qualification(tmp_path, monkeypatch):
    root, launch, report = _retained_probe_fixture(tmp_path, monkeypatch, window=probe.PORTABLE_WINDOW)
    monkeypatch.setattr(probe.host.execution, 'PROFILE', probe.host.execution.select(probe.host.execution.DEFAULT))
    result = probe.verify_retained(root, report['launch_sha256'], probe.host.digest(root/'report.json'))
    assert launch['protocol'] == evaluation.PORTABLE_PROBE['protocol']
    assert not result['full_evaluation_enabled'] and not result['physical_motion_authorized']


def test_offline_portable_profile_refusal_precedes_case_replay(tmp_path, monkeypatch):
    root, _, report = _retained_probe_fixture(tmp_path, monkeypatch, window=probe.PORTABLE_WINDOW)
    monkeypatch.setattr(profile, 'checked_receipt', lambda: {})
    monkeypatch.setattr(evaluation, 'verify_probe', lambda *_: pytest.fail('profile refusal before bundle replay'))
    with pytest.raises(ValueError, match='actual recorded CPU math profile'):
        probe.verify_retained(root, report['launch_sha256'], probe.host.digest(root/'report.json'))


def test_r3_is_separate_window_and_cannot_reopen_expired_r2(monkeypatch):
    select_wsl(monkeypatch)
    limits = probe.WINDOWS[probe.AUTH_WINDOW]
    assert limits['cutoff']-limits['not_before'] == 3600
    assert probe.WINDOWS[probe.PORTABLE_WINDOW]['cutoff'] < limits['not_before']
    monkeypatch.setattr(probe.time, 'time', lambda: limits['not_before'])
    probe.check_window(limits['cutoff'], launching=True, window=probe.AUTH_WINDOW)
    with pytest.raises(ValueError):
        probe.check_window(probe.WINDOWS[probe.PORTABLE_WINDOW]['cutoff'], launching=True,
                           window=probe.PORTABLE_WINDOW)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['cutoff']-1620)
    with pytest.raises(ValueError, match='whole packed probe plus closeout'):
        probe.check_window(limits['cutoff'], launching=True, window=probe.AUTH_WINDOW)
    launch = portable_plan(probe.AUTH_WINDOW)
    assert launch['execution_window'] == probe.AUTH_WINDOW
    assert launch['historical_training_authentication'] == historical_auth_fixture()
    assert probe.declaration_of_window(probe.AUTH_WINDOW) is evaluation.PORTABLE_PROBE


def test_r3_offline_rederivation_keeps_profile_and_auth_without_live_gpu(tmp_path, monkeypatch):
    root, _, report = _retained_probe_fixture(tmp_path, monkeypatch, window=probe.AUTH_WINDOW)
    monkeypatch.setattr(probe.host.execution, 'PROFILE', probe.host.execution.select(probe.host.execution.DEFAULT))
    assert not probe.verify_retained(root, report['launch_sha256'],
                                    probe.host.digest(root/'report.json'))['full_evaluation_enabled']


def test_r4_renewal_is_additive_bounded_and_does_not_reopen_old_windows(monkeypatch):
    select_wsl(monkeypatch)
    limits = probe.WINDOWS[probe.OCT2_WINDOW]
    assert limits['cutoff']-limits['not_before'] == 3600
    assert probe.WINDOWS[probe.AUTH_WINDOW]['cutoff'] < limits['not_before']
    monkeypatch.setattr(probe.time, 'time', lambda: limits['not_before']-1)
    with pytest.raises(ValueError, match='before its declared start'):
        probe.check_window(limits['cutoff'], launching=True, window=probe.OCT2_WINDOW)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['not_before'])
    probe.check_window(limits['cutoff'], launching=True, window=probe.OCT2_WINDOW)
    for old in (probe.PORTABLE_WINDOW, probe.AUTH_WINDOW, probe.REPAIRED_WINDOW):
        with pytest.raises(ValueError):
            probe.check_window(probe.WINDOWS[old]['cutoff'], launching=True, window=old)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['cutoff']-1620)
    with pytest.raises(ValueError, match='whole packed probe plus closeout'):
        probe.check_window(limits['cutoff'], launching=True, window=probe.OCT2_WINDOW)
    monkeypatch.setattr(probe.time, 'time', lambda: limits['cutoff']+1)
    with pytest.raises(ValueError):
        probe.check_window(limits['cutoff'], window=probe.OCT2_WINDOW)
    launch = portable_plan(probe.OCT2_WINDOW)
    assert launch['execution_window'] == probe.OCT2_WINDOW
    assert launch['historical_training_authentication'] == historical_auth_fixture()
    assert launch['child_timeout_seconds'] == 600 and launch['service_timeout_seconds'] == 960


def test_r4_retained_plan_rederives_on_independent_cpu_runtime(tmp_path, monkeypatch):
    root, _, report = _retained_probe_fixture(tmp_path, monkeypatch, window=probe.OCT2_WINDOW)
    monkeypatch.setattr(probe.host.execution, 'PROFILE', probe.host.execution.select(probe.host.execution.DEFAULT))
    assert not probe.verify_retained(root, report['launch_sha256'],
                                    probe.host.digest(root/'report.json'))['learned_stance_accepted']
