"""Source and replay contract checks for the separately bound CUDA protocol.

CPU fixtures in this file exercise constructors and pure declarations only;
they are never labeled or scored as native CUDA capture evidence.
"""
import pytest

from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_runtime as runtime
from mjlab_microduck import stance_recovery_schedule_trace as cpu_trace
from mjlab_microduck import stance_recovery_cuda_schedule_trace as cuda_trace


SOURCE = 'a' * 40
SYNTHETIC_PROFILE = profile.expected_receipt()


def cpu_shape_fixture(cell='zero-wrench'):
    """Actual CPU runtime state used only to check CUDA replay tensor ownership."""
    declaration = schedule.declaration(SOURCE, 'dose', 'held-out', [cell])
    env = runtime.ScheduledRecoveryRuntime(declaration, device='cpu')
    binding = cuda_trace.binding(declaration, env.binding, SYNTHETIC_PROFILE)
    return env, declaration, binding


def test_protocol_and_binding_pin_only_the_declared_short_heldout_cell():
    env, declaration, value = cpu_shape_fixture()
    assert cuda_trace.PROTOCOL == 'football-b1d-cuda-scheduled-integration-trace-v1'
    assert cuda_trace.PROTOCOL not in cpu_trace.trace.ITERATIONS
    assert value['source'] == SOURCE and value['worlds'] == 1
    assert value['capture_device'] == 'cuda:0'
    assert value['schedule_sha256'] == schedule.binding_sha256(declaration)
    assert value['plant_sha256']
    assert value['checkpoint_sha256'] == '2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5'
    assert value['checkpoint_identity'] == cuda_trace.parent.expected_identity()
    assert value['evaluation_seed'] == 671
    assert value['cpu_math_profile'] == SYNTHETIC_PROFILE
    assert baseline.CHECKPOINT_SHA256 == value['checkpoint_sha256']
    assert env.device.type == 'cpu'  # This is a CPU fixture, not capture evidence.


@pytest.mark.parametrize('split,stage,cell', [
    ('training', 'dose', 'zero-wrench'),
    ('held-out', 'gentle', 'zero-wrench'),
    ('held-out', 'dose', '+x-2n-10steps-t500'),
    ('held-out', 'dose', 'diagonal-++-2n-10steps-t375'),
])
def test_binding_rejects_unselected_split_stage_or_cell(split, stage, cell):
    declaration = schedule.declaration(SOURCE, stage, split, [cell])
    plant = {'selected_plant': {}, 'nbody': 2, 'body_names': ['world', 'trunk_base'],
             'body_name': 'trunk_base', 'body_id': 1}
    with pytest.raises(ValueError, match='one declared held-out CUDA integration cell'):
        cuda_trace.binding(declaration, plant, SYNTHETIC_PROFILE)


def test_replay_owns_cpu_tensors_without_relabeling_capture_provenance():
    env, declaration, binding = cpu_shape_fixture()
    replay = cuda_trace.CudaScheduledTrace(binding, env.snapshot(), declaration, env.binding,
                                           replay=True)
    assert replay._tensor_device == 'cpu'
    assert replay.binding['capture_device'] == 'cuda:0'
    assert replay.binding['protocol'] == cuda_trace.PROTOCOL
    with pytest.raises(ValueError, match='actual CUDA scheduled capture device'):
        cuda_trace.CudaScheduledTrace(binding, env.snapshot(), declaration, env.binding)


def test_capture_refuses_cpu_runtime_before_any_synthetic_backend_can_be_emitted(monkeypatch):
    env, _, _ = cpu_shape_fixture()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES=0 capture'):
        cuda_trace.collect(env, b'fixture-only', deadline_monotonic=10**20)


def test_shared_prefix_helper_is_the_existing_scheduled_prefix_codec():
    value = dict(payload=dict(initial={}, ticks=[]), control_evidence=dict(ticks=[]))
    # The shared helper remains the owner of range/schema semantics; malformed
    # input fails before any raw artifact or CUDA provenance is involved.
    with pytest.raises(ValueError, match='bounded scheduled prefix step'):
        cuda_trace.prefix_hash(value, 2500)


def test_verify_hashes_and_checks_cpu_scorer_before_deserialization(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(cuda_trace.torch, 'load',
                        lambda *_a, **_k: pytest.fail('whole bytes and CPU scorer precede load'))
    with pytest.raises(ValueError, match='whole-byte hash before tensor loading'):
        cuda_trace.verify(b'fixture-only', '0' * 64, b'', {}, {})
