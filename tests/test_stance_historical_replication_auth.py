"""Contracts for the fixed three-seed historical archive authenticator."""
from copy import deepcopy
import subprocess

import pytest

from mjlab_microduck import stance_historical_replication_auth as auth
from mjlab_microduck import stance_historical_training_auth as legacy
from mjlab_microduck.first_attempt_smoke import canonical
from test_stance_packed_evaluation_probe import SOURCE, portable_retained_training


def retained(seed):
    result = portable_retained_training()
    result['report_sha256'] = auth.REPORT_SHA256[seed]
    for saved in result['checkpoints']:
        saved['identity']['training_seed'] = seed
    return result


def receipt(seed):
    result = dict(protocol=auth.PROTOCOL, source=SOURCE, training_seed=seed,
        cpu_runtime=auth.expected_runtime(),
        settings=dict(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=''),
        timeout_seconds=60,
        archive_files={name: 'f'*64 for name in auth.inventory_names()},
        retained_training=retained(seed), optimizer_steps=0,
        gpu_execution_performed=False, independent_attestation=False)
    return result


@pytest.mark.parametrize('seed', auth.SEEDS)
def test_each_fixed_seed_has_its_own_exact_receipt(seed):
    value = receipt(seed)
    auth.validate(value, SOURCE, value['retained_training'])
    assert value['training_seed'] == seed
    assert value['retained_training']['report_sha256'] == auth.REPORT_SHA256[seed]


@pytest.mark.parametrize('damage', [
    'seed_bool', 'unknown_seed', 'wrong_report', 'wrong_checkpoint_seed',
    'wrong_checkpoint_iteration', 'wrong_checkpoint_hash', 'missing_checkpoint',
    'missing_archive_file', 'gpu', 'extra_field',
])
def test_replication_receipts_fail_closed(damage):
    value = receipt(587)
    if damage == 'seed_bool':
        value['training_seed'] = True
    elif damage == 'unknown_seed':
        value['training_seed'] = 577
    elif damage == 'wrong_report':
        value['retained_training']['report_sha256'] = '0'*64
    elif damage == 'wrong_checkpoint_seed':
        value['retained_training']['checkpoints'][1]['identity']['training_seed'] = 593
    elif damage == 'wrong_checkpoint_iteration':
        value['retained_training']['checkpoints'][1]['identity']['iteration'] = 193
    elif damage == 'wrong_checkpoint_hash':
        value['archive_files']['model_128.pt'] = '0'*64
    elif damage == 'missing_checkpoint':
        value['retained_training']['checkpoints'].pop()
    elif damage == 'missing_archive_file':
        value['archive_files'].pop('parent.pt')
    elif damage == 'gpu':
        value['gpu_execution_performed'] = True
    else:
        value['unexpected'] = True
    with pytest.raises(ValueError):
        auth.validate(value, SOURCE, value['retained_training'])


@pytest.mark.parametrize('seed', auth.SEEDS)
def test_isolated_run_uses_original_profile_and_fixed_seed(monkeypatch, seed):
    value = receipt(seed)
    original_env = {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '8',
                    'ATEN_CPU_CAPABILITY': 'default', 'MKL_CBWR': 'COMPATIBLE'}
    monkeypatch.setattr(auth.files, 'child_environment', lambda: deepcopy(original_env))

    def execute(command, **kwargs):
        assert command == [str(auth.host.ROOT/'.venv/bin/python'), '-c', auth.EXEC_WRAPPER,
                           SOURCE, auth.TRAINING_SOURCE, str(seed)]
        assert kwargs['cwd'] == auth.host.ROOT
        assert kwargs['env'] == {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'}
        return (canonical(value)+'\n').encode()

    monkeypatch.setattr(auth, 'bounded_output', execute)
    assert auth.run(SOURCE, auth.TRAINING_SOURCE, seed) == value
    assert original_env['CUDA_VISIBLE_DEVICES'] == '0'
    with pytest.raises(ValueError):
        auth.run(SOURCE, auth.TRAINING_SOURCE, True)


def test_pinned_report_is_checked_before_original_archive_loader(monkeypatch, tmp_path):
    seed = 587
    report_raw = canonical({'files': {name: 'f'*64 for name in auth.inventory_names()}}).encode()
    events = []
    monkeypatch.setattr(auth, 'check_process', lambda: auth.expected_runtime())
    monkeypatch.setattr(auth.host, 'identity', lambda *_: events.append('identity'))
    monkeypatch.setattr(auth.evaluation.lean, 'output_path', lambda *_: tmp_path)
    monkeypatch.setattr(auth.files, 'file_bytes', lambda _: report_raw)

    class FakeDigest:
        def hexdigest(self):
            events.append('report-hash')
            return auth.REPORT_SHA256[seed]

    monkeypatch.setattr(auth, 'sha256', lambda _: FakeDigest())
    expected_retained = retained(seed)

    def training_inputs(declaration, requested_seed, source):
        events.append('archive-loader')
        assert declaration is auth.evaluation.PORTABLE_PROBE
        assert requested_seed == seed and source == auth.TRAINING_SOURCE
        return deepcopy(expected_retained)

    monkeypatch.setattr(auth.evaluation, 'training_inputs', training_inputs)
    monkeypatch.setattr(auth, 'check_process', lambda: (events.append('runtime-check') or auth.expected_runtime()))
    value = auth.authenticate(SOURCE, auth.TRAINING_SOURCE, seed)
    assert value['retained_training'] == expected_retained
    assert events.index('report-hash') < events.index('archive-loader')


def test_wrong_pinned_report_refuses_before_tensor_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, 'check_process', lambda: auth.expected_runtime())
    monkeypatch.setattr(auth.host, 'identity', lambda *_: None)
    monkeypatch.setattr(auth.evaluation.lean, 'output_path', lambda *_: tmp_path)
    monkeypatch.setattr(auth.files, 'file_bytes', lambda _: b'wrong report')
    monkeypatch.setattr(auth.evaluation, 'training_inputs',
                        lambda *_: pytest.fail('tensor loader called before pinned report check'))
    with pytest.raises(ValueError, match='predeclared original packed report hash'):
        auth.authenticate(SOURCE, auth.TRAINING_SOURCE, 593)


def test_legacy_seed_577_authenticator_remains_unchanged_and_rejects_new_protocol():
    assert legacy.SEED == 577
    new_receipt = receipt(587)
    with pytest.raises(ValueError, match='declared historical authentication provenance'):
        legacy.validate(new_receipt, SOURCE, new_receipt['retained_training'])


def test_invalid_training_sources_and_seeds_are_rejected_before_exec(monkeypatch):
    monkeypatch.setattr(auth, 'bounded_output', lambda *_args, **_kwargs: pytest.fail('child launched'))
    for seed in (True, 578, 587.0):
        with pytest.raises(ValueError):
            auth.run(SOURCE, auth.TRAINING_SOURCE, seed)
    with pytest.raises(ValueError):
        auth.run(SOURCE, '0'*40, 577)


def test_child_timeout_and_failure_are_not_parent_fallback(monkeypatch):
    monkeypatch.setattr(auth, 'bounded_output',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired('child', 60)))
    with pytest.raises(subprocess.TimeoutExpired):
        auth.run(SOURCE, auth.TRAINING_SOURCE, 577)
    monkeypatch.setattr(auth, 'bounded_output',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(2, 'child')))
    with pytest.raises(subprocess.CalledProcessError):
        auth.run(SOURCE, auth.TRAINING_SOURCE, 587)
