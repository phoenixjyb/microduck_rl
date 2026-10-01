"""Portable CPU profile checks use synthetic inspection, never a replay run."""

import os
import pytest

from mjlab_microduck import stance_cpu_replay_profile as profile


def expected_runtime():
    return {
        'platform': 'linux',
        'architecture': 'x86_64',
        'torch_version': '2.9.1',
        'cpu_capability': 'DEFAULT',
        'torch_num_threads': 1,
        'libtorch_cpu_sha256': profile.LIBTORCH_CPU_SHA256,
    }


def selected_env():
    return profile.settings()


def check(runtime=None, *, exec_env=None, current_env=None):
    return profile.validate_runtime(
        expected_runtime() if runtime is None else runtime,
        exec_env=selected_env() if exec_env is None else exec_env,
        current_env=selected_env() if current_env is None else current_env,
    )


def test_settings_are_exact_fresh_and_do_not_mutate_process_environment():
    before = {key: os.environ.get(key) for key in profile.settings()}
    first = profile.settings()
    assert first == {
        'ATEN_CPU_CAPABILITY': 'default',
        'MKL_CBWR': 'COMPATIBLE',
        'OMP_NUM_THREADS': '1',
    }
    first['MKL_CBWR'] = 'AUTO'
    assert profile.settings()['MKL_CBWR'] == 'COMPATIBLE'
    assert {key: os.environ.get(key) for key in before} == before


@pytest.mark.parametrize('snapshot', ['exec_env', 'current_env'])
@pytest.mark.parametrize('damage', ['missing', 'wrong'])
def test_missing_or_changed_profile_environment_refuses(snapshot, damage):
    values = selected_env()
    if damage == 'missing':
        del values['MKL_CBWR']
    else:
        values['ATEN_CPU_CAPABILITY'] = 'avx2'
    kwargs = {snapshot: values}
    with pytest.raises(ValueError, match='ATEN_CPU_CAPABILITY|MKL_CBWR'):
        check(**kwargs)


@pytest.mark.parametrize(('key', 'value', 'message'), [
    ('platform', 'darwin', 'Linux CPU replay platform'),
    ('architecture', 'arm64', 'x86_64 CPU replay architecture'),
    ('torch_version', '2.9.2', 'pinned PyTorch metadata version'),
    ('cpu_capability', 'AVX2', 'ATen default CPU capability'),
    ('torch_num_threads', 2, 'single PyTorch CPU thread'),
    ('libtorch_cpu_sha256', '0'*64, 'pinned libtorch_cpu.so hash'),
])
def test_runtime_mismatch_refuses(key, value, message):
    runtime = expected_runtime()
    runtime[key] = value
    with pytest.raises(ValueError, match=message):
        check(runtime)


def test_runtime_schema_and_thread_type_are_strict():
    runtime = expected_runtime()
    runtime['extra'] = 'ignored?'
    with pytest.raises(ValueError, match='exact CPU runtime inspection'):
        check(runtime)
    runtime = expected_runtime()
    runtime['torch_num_threads'] = True
    with pytest.raises(ValueError, match='single PyTorch CPU thread'):
        check(runtime)


def test_receipt_is_narrow_and_does_not_claim_replay_or_acceptance(monkeypatch):
    monkeypatch.setattr(profile, 'inspect_exec_environment', selected_env)
    for key, value in selected_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(profile, 'inspect_runtime', expected_runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError('profile checking must not call CUDA APIs')

    monkeypatch.setattr(profile.torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(profile.torch.cuda, 'init', forbidden)
    before = {key: os.environ.get(key) for key in selected_env()}
    receipt = profile.checked_receipt()
    assert receipt['protocol'] == profile.PROTOCOL
    assert receipt['status'] == 'portable-cpu-math-profile-checked'
    assert receipt['settings'] == selected_env()
    assert receipt['full_replay_performed'] is False
    assert receipt['actor_output_parity_verified'] is False
    assert receipt['capability_accepted'] is False
    assert receipt['binary_equivalence_verified'] is False
    assert {key: os.environ.get(key) for key in before} == before


def test_late_environment_edit_cannot_retrofit_exec_settings(monkeypatch):
    # Torch is already imported, but the default verifier reads exec-time state.
    monkeypatch.setattr(profile, 'inspect_exec_environment', lambda: {})
    for key, value in selected_env().items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='ATEN_CPU_CAPABILITY.*at exec'):
        profile.validate_runtime(expected_runtime(), current_env=selected_env())


@pytest.mark.parametrize('trailing_null', [b'', b'\0'])
def test_exec_environment_parser_selects_only_profile_keys(trailing_null):
    raw = b'PRIVATE_VALUE=not-returned\0' + b'\0'.join(
        f'{key}={value}'.encode() for key, value in selected_env().items())
    assert profile._parse_exec_environment(raw + trailing_null) == selected_env()
    assert profile._parse_exec_environment(b'') == {}


@pytest.mark.parametrize(('raw', 'message'), [
    (b'MKL_CBWR=COMPATIBLE\0MKL_CBWR=AUTO', 'unique profile key'),
    (b'MKL_CBWR=\xff', 'ASCII profile value'),
    (b'NOT_AN_ASSIGNMENT', 'well-formed exec environment'),
    (b'X=' + b'0' * (1024 * 1024), 'bounded exec environment'),
])
def test_exec_environment_parser_refuses_ambiguous_or_unbounded_input(raw, message):
    with pytest.raises(ValueError, match=message):
        profile._parse_exec_environment(raw)


def test_missing_proc_environment_fails_closed(monkeypatch):
    monkeypatch.setattr(profile.sys, 'platform', 'linux')

    def missing(*args, **kwargs):
        raise FileNotFoundError('synthetic unavailable proc')

    monkeypatch.setattr(profile.Path, 'open', missing)
    with pytest.raises(ValueError, match='readable exec environment required'):
        profile.inspect_exec_environment()


def test_exec_environment_read_is_bounded_and_private(monkeypatch):
    import io
    monkeypatch.setattr(profile.sys, 'platform', 'linux')
    raw = b'PRIVATE_VALUE=not-returned\0' + b'\0'.join(
        f'{key}={value}'.encode() for key, value in selected_env().items())

    class BoundedRead(io.BytesIO):
        def read(self, size):
            assert size == 1024 * 1024 + 1
            return super().read(size)

    monkeypatch.setattr(profile.Path, 'open', lambda *args, **kwargs: BoundedRead(raw))
    assert profile.inspect_exec_environment() == selected_env()


def test_non_linux_inspection_refuses_before_library_or_proc_read(monkeypatch):
    monkeypatch.setattr(profile.sys, 'platform', 'darwin')

    def forbidden(*args, **kwargs):
        raise AssertionError('no Linux file read on a non-Linux host')

    monkeypatch.setattr(profile.Path, 'open', forbidden)
    for inspect in (profile.inspect_exec_environment, profile.inspect_runtime):
        with pytest.raises(ValueError, match='Linux CPU replay platform'):
            inspect()


def test_runtime_inspection_uses_metadata_not_version_label(monkeypatch):
    import io
    monkeypatch.setattr(profile.sys, 'platform', 'linux')
    monkeypatch.setattr(profile.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(profile.Path, 'open', lambda *args, **kwargs: io.BytesIO(b'fixture'))
    monkeypatch.setattr(profile, 'version', lambda name: '2.9.2' if name == 'torch' else None)
    monkeypatch.setattr(profile.torch, '__version__', '2.9.1+synthetic')
    runtime = profile.inspect_runtime()
    assert runtime['torch_version'] == '2.9.2'
    with pytest.raises(ValueError, match='pinned PyTorch metadata version'):
        check(runtime)


def test_expected_receipt_is_pure_fresh_and_matches_validated_runtime():
    assert profile.expected_receipt() == check()
    changed = profile.expected_receipt()
    changed['settings']['MKL_CBWR'] = 'AUTO'
    assert profile.expected_receipt()['settings']['MKL_CBWR'] == 'COMPATIBLE'


@pytest.mark.parametrize('damage', ['extra', 'missing', 'wrong_setting', 'bool_thread', 'int_flag'])
def test_archived_receipt_requires_exact_schema_values_and_json_types(damage):
    receipt = profile.expected_receipt()
    if damage == 'extra': receipt['ignored'] = True
    elif damage == 'missing': receipt.pop('settings')
    elif damage == 'wrong_setting': receipt['settings']['MKL_CBWR'] = 'AUTO'
    elif damage == 'bool_thread': receipt['torch_num_threads'] = True
    else: receipt['capability_accepted'] = 0
    with pytest.raises(ValueError, match='exact recorded CPU math profile'):
        profile.validate_receipt(receipt)


def test_archived_profile_requires_actual_runtime_before_inference(monkeypatch):
    monkeypatch.setattr(profile, 'checked_receipt', lambda: {})
    with pytest.raises(ValueError, match='actual recorded CPU math profile'):
        profile.check_recorded(profile.expected_receipt())
