import itertools
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_evaluation_worker as worker


def binding(**changes):
    value = dict(protocol=trace.PORTABLE_FULL_PROTOCOL, source='a'*40,
        runtime_sha256='b'*64, checkpoint_sha256='c'*64, launch_sha256='d'*64,
        checkpoint_iteration=64, evaluation_seed=541, worlds=128,
        capture_device='cuda:0', solved_field_check='packed', checker_sha256='e'*64,
        cpu_math_profile=profile.expected_receipt(), training_seed=577)
    value.update(changes)
    return value


def test_full_protocol_admits_only_declared_three_by_four_by_three_matrix():
    for training_seed, iteration, evaluation_seed in itertools.product(
            trace.PORTABLE_FULL_TRAINING_SEEDS, trace.LEAN_REPLICATION_CHECKPOINTS, trace.SEEDS):
        trace.validate_binding(binding(training_seed=training_seed,
            checkpoint_iteration=iteration, evaluation_seed=evaluation_seed))


@pytest.mark.parametrize('changes', [
    {'training_seed': True}, {'training_seed': 521}, {'training_seed': '577'},
    {'evaluation_seed': True}, {'evaluation_seed': 5470}, {'checkpoint_iteration': True},
    {'checkpoint_iteration': 256}, {'worlds': True}, {'worlds': 127},
    {'capture_device': 'cpu'}, {'solved_field_check': 'scalar'},
    {'checker_sha256': 'bad'}, {'cpu_math_profile': {}},
    {'extra': 'field'}, {'checker_sha256': None},
])
def test_full_protocol_rejects_malformed_or_unbound_matrix_rows(changes):
    with pytest.raises((ValueError, KeyError)):
        trace.validate_binding(binding(**changes))


@pytest.mark.parametrize('key', ['checker_sha256', 'cpu_math_profile', 'training_seed'])
def test_full_protocol_requires_checker_profile_and_training_seed(key):
    selected = binding()
    del selected[key]
    with pytest.raises(ValueError, match='exact trace binding fields'):
        trace.validate_binding(selected)


@pytest.mark.parametrize('changes', [
    {'checkpoint_iteration': 64}, {'evaluation_seed': 547},
])
def test_old_probe_protocol_remains_final_541_only(changes):
    selected = dict(protocol=trace.PORTABLE_PROBE_PROTOCOL, source='a'*40,
        runtime_sha256='b'*64, checkpoint_sha256='c'*64, launch_sha256='d'*64,
        checkpoint_iteration=255, evaluation_seed=541, worlds=128,
        capture_device='cuda:0', solved_field_check='packed', checker_sha256='e'*64,
        cpu_math_profile=profile.expected_receipt())
    selected.update(changes)
    with pytest.raises(ValueError):
        trace.validate_binding(selected)


def test_legacy_probe_allowlist_and_bindings_are_unchanged():
    assert trace.PACKED_PROBE_PROTOCOLS == (
        trace.PACKED_PROBE_PROTOCOL, trace.PORTABLE_PROBE_PROTOCOL)
    assert trace.ITERATIONS[trace.PORTABLE_PROBE_PROTOCOL] == (255,)
    probe = dict(protocol=trace.PORTABLE_PROBE_PROTOCOL, source='a'*40,
        runtime_sha256='b'*64, checkpoint_sha256='c'*64, launch_sha256='d'*64,
        checkpoint_iteration=255, evaluation_seed=541, worlds=128,
        capture_device='cuda:0', solved_field_check='packed', checker_sha256='e'*64,
        cpu_math_profile=profile.expected_receipt())
    trace.validate_binding(probe)


def test_packed_full_runtime_requires_live_eager_cuda_and_exact_checker_source():
    import mjlab_microduck.stance_solved_field_check as checker

    selected = binding(checker_sha256=sha256(Path(checker.__file__).read_bytes()).hexdigest())
    cpu_mock = SimpleNamespace(solved_field_check='packed', forward_graph=None,
        wp_device=SimpleNamespace(is_cuda=False))
    with pytest.raises(ValueError, match='actual eager packed probe runtime'):
        worker.check_packed_runtime(cpu_mock, selected)

    cuda_mock = SimpleNamespace(solved_field_check='packed', forward_graph=None,
        wp_device=SimpleNamespace(is_cuda=True))
    worker.check_packed_runtime(cuda_mock, selected)
    with pytest.raises(ValueError, match='actual packed probe checker source'):
        worker.check_packed_runtime(cuda_mock, binding(checker_sha256='0'*64))


def test_full_bundle_input_and_actor_replay_require_recorded_profile(monkeypatch):
    monkeypatch.setattr(profile, 'check_recorded',
        lambda _receipt: (_ for _ in ()).throw(ValueError('recorded portable profile required')))
    selected = binding()
    with pytest.raises(ValueError, match='recorded portable profile required'):
        bundle.checked_inputs(selected, b'x', {'training_seed': 577}, b'x', b'x')

    actor, _ = checkpoint.fresh_models(521)
    with pytest.raises(ValueError, match='recorded portable profile required'):
        bundle.actor_replay({'binding': selected, 'ticks': []}, actor, {})
