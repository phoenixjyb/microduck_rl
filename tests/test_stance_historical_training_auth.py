"""Isolated historical archive contracts; no actual subprocess/GPU/service."""
from copy import deepcopy
import subprocess
import pytest
from mjlab_microduck import stance_historical_training_auth as auth
from mjlab_microduck import stance_packed_evaluation_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical
from test_stance_packed_evaluation_probe import SOURCE, historical_auth_fixture, portable_retained_training


@pytest.mark.parametrize('damage', ['source', 'seed_bool', 'runtime_bool', 'cuda', 'attestation',
                                  'missing_file', 'changed_checkpoint', 'report', 'extra'])
def test_historical_receipt_fail_closed(damage):
    receipt = historical_auth_fixture()
    if damage == 'source': receipt['source'] = '0'*40
    elif damage == 'seed_bool': receipt['training_seed'] = True
    elif damage == 'runtime_bool': receipt['cpu_runtime']['torch_num_threads'] = True
    elif damage == 'cuda': receipt['gpu_execution_performed'] = True
    elif damage == 'attestation': receipt['independent_attestation'] = True
    elif damage == 'missing_file': receipt['archive_files'].pop('parent.pt')
    elif damage == 'changed_checkpoint': receipt['archive_files']['model_255.pt'] = '0'*64
    elif damage == 'report': receipt['retained_training']['report_sha256'] = '0'*64
    else: receipt['extra'] = True
    with pytest.raises(ValueError): auth.validate(receipt, SOURCE, portable_retained_training())


def test_isolated_exec_is_sanitized_bounded_and_source_pinned(monkeypatch):
    receipt = historical_auth_fixture()
    env = {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '4'}
    monkeypatch.setattr(auth.files, 'child_environment', lambda: deepcopy(env))
    def execute(command, **kwargs):
        assert command == [str(auth.host.ROOT/'.venv/bin/python'), '-c', auth.EXEC_WRAPPER,
                           SOURCE, auth.TRAINING_SOURCE, '577']
        assert kwargs['cwd'] == auth.host.ROOT
        assert kwargs['env'] == {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}
        return (canonical(receipt)+'\n').encode()
    monkeypatch.setattr(auth, 'bounded_output', execute)
    assert auth.run(SOURCE, auth.TRAINING_SOURCE, 577) == receipt
    assert env['CUDA_VISIBLE_DEVICES'] == '0'  # Parent settings never changed.
    with pytest.raises(ValueError): auth.run(SOURCE, '0'*40, 577)


@pytest.mark.parametrize('damage', ['extra_output', 'noncanonical', 'oversize', 'timeout', 'failed'])
def test_exec_failure_never_falls_back_to_parent(monkeypatch, damage):
    raw = (canonical(historical_auth_fixture())+'\n').encode()
    def execute(*args, **kwargs):
        if damage == 'timeout': raise subprocess.TimeoutExpired(args[0], 60)
        if damage == 'failed': raise subprocess.CalledProcessError(1, args[0])
        if damage == 'extra_output': return b'unexpected\n'+raw
        if damage == 'noncanonical': return b' '+raw
        return b'x'*(auth.OUTPUT_LIMIT+1)
    monkeypatch.setattr(auth, 'bounded_output', execute)
    monkeypatch.setattr(auth.evaluation, 'training_inputs', lambda *_: pytest.fail('no parent fallback'))
    with pytest.raises((ValueError, subprocess.TimeoutExpired, subprocess.CalledProcessError)):
        auth.run(SOURCE, auth.TRAINING_SOURCE, 577)


def test_wrong_historical_exec_profile_refuses_before_archive(monkeypatch):
    monkeypatch.setattr(auth.profile, 'inspect_exec_environment', lambda: {
        'OMP_NUM_THREADS': '1', 'MKL_CBWR': 'COMPATIBLE'})
    monkeypatch.setattr(auth.host, 'identity', lambda *_: pytest.fail('profile before identity/archive'))
    with pytest.raises(ValueError, match='original historical CPU profile'):
        auth.authenticate(SOURCE, auth.TRAINING_SOURCE, 577)


def test_portable_plan_requires_historical_receipt_and_legacy_forbids_it(monkeypatch):
    from test_stance_packed_evaluation_probe import RUNTIME_SHA, retained_training, select_wsl
    select_wsl(monkeypatch)
    with pytest.raises(ValueError, match='exact historical authentication receipt'):
        probe.plan(SOURCE, {'execution_profile': probe.host.execution.PROFILE}, RUNTIME_SHA,
            portable_retained_training(), probe.WINDOWS[probe.PORTABLE_WINDOW]['cutoff'],
            window=probe.PORTABLE_WINDOW)
    with pytest.raises(ValueError, match='no historical auth field'):
        probe.plan(SOURCE, {}, RUNTIME_SHA, retained_training(), probe.WINDOWS[probe.LEGACY_WINDOW]['cutoff'],
                   training_auth=historical_auth_fixture())


def test_live_portable_checked_reauthenticates_original_archive_without_parent_fallback(tmp_path, monkeypatch):
    from hashlib import sha256
    from test_stance_packed_evaluation_probe import select_wsl
    select_wsl(monkeypatch)
    monkeypatch.setattr(probe.cpu_profile, 'checked_receipt', probe.cpu_profile.expected_receipt)
    monkeypatch.setattr(probe, 'output_path', lambda *_: tmp_path)
    inputs = {'execution_profile': probe.host.execution.PROFILE}
    monkeypatch.setattr(probe.host, 'identity', lambda *_: inputs)
    monkeypatch.setattr(probe.plant, 'checked_runtime', lambda *_: None)
    runtime = b'{"fixture":true}'
    (tmp_path/'runtime.json').write_bytes(runtime)
    receipt = historical_auth_fixture()
    retained = portable_retained_training()
    launch = probe.plan(SOURCE, inputs, sha256(runtime).hexdigest(), retained,
        probe.WINDOWS[probe.PORTABLE_WINDOW]['cutoff'], window=probe.PORTABLE_WINDOW,
        training_auth=receipt)
    (tmp_path/'launch.json').write_text(canonical(launch)+'\n')
    calls = []
    def isolated(source, training_source, seed):
        calls.append((source, training_source, seed))
        return deepcopy(receipt)
    monkeypatch.setattr(auth, 'run', isolated)
    monkeypatch.setattr(auth.evaluation, 'training_inputs', lambda *_: pytest.fail('no portable parent validation'))
    assert probe.checked(SOURCE, 577, probe.host.digest(tmp_path/'launch.json')) == launch
    assert calls == [(SOURCE, auth.TRAINING_SOURCE, 577)]
    receipt['archive_files']['parent.pt'] = '0'*64
    with pytest.raises(ValueError, match='unchanged packed probe source/host/archive/plan'):
        probe.checked(SOURCE, 577, probe.host.digest(tmp_path/'launch.json'))


def test_streamed_output_limit_and_timeout_reap_only_owned_cpu_child(tmp_path, monkeypatch):
    import sys
    env = {'PATH': '/usr/bin:/bin'}
    assert auth.bounded_output([sys.executable, '-c', 'print("one receipt")'], cwd=tmp_path, env=env) == b'one receipt\n'
    monkeypatch.setattr(auth, 'OUTPUT_LIMIT', 128)
    with pytest.raises(ValueError, match='bounded historical authentication output'):
        auth.bounded_output([sys.executable, '-c', 'print("x"*256)'], cwd=tmp_path, env=env)
    monkeypatch.setattr(auth, 'SECONDS', .2)
    with pytest.raises(subprocess.TimeoutExpired):
        auth.bounded_output([sys.executable, '-c', 'import time; time.sleep(5)'], cwd=tmp_path, env=env)


def test_streamed_nonzero_exit_retains_safe_failure_note(tmp_path):
    import sys
    with pytest.raises(subprocess.CalledProcessError) as saved:
        auth.bounded_output([sys.executable, '-c', 'import sys; print("failed identity", file=sys.stderr); sys.exit(2)'],
                            cwd=tmp_path, env={'PATH': '/usr/bin:/bin'})
    assert saved.value.returncode == 2
    assert 'failed identity' in saved.value.__notes__[0]


def test_startup_diagnostics_are_separate_from_the_canonical_receipt(tmp_path):
    import sys
    script = ('import sys; print("one receipt"); '+
              'print('+repr(auth.host.STARTUP_INFO)+', file=sys.stderr); '+
              'print("registered task", file=sys.stderr)')
    assert auth.bounded_output([sys.executable, '-c', script], cwd=tmp_path,
                               env={'PATH': '/usr/bin:/bin'}) == b'one receipt\n'
    with pytest.raises(ValueError, match='no numerical/backend warning'):
        auth.bounded_output([sys.executable, '-c', 'import sys; print("warning: nonfinite", file=sys.stderr)'],
                            cwd=tmp_path, env={'PATH': '/usr/bin:/bin'})
