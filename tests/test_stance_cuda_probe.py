"""CPU checks of the exact probe cases and refusal gates; no GPU allocation."""

from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_cuda_probe as probe


@pytest.fixture(scope='module')
def cpu_payload():
    return probe.cases('cpu')


def test_reviewed_dependency_python_trees_match_local_runtime():
    assert probe.python_trees() == probe.supervisor.parse(probe.PIN_FILE.read_bytes())


def test_cpu_cases_are_finite_owned_and_not_cuda_admission(cpu_payload):
    probe.canonical(cpu_payload)
    assert cpu_payload['backend']['torch_device'] == 'cpu'
    assert not cpu_payload['backend']['warp_is_cuda']
    assert cpu_payload['normal']['steps'] == [20, 20]
    assert cpu_payload['isolation']['before_reset_steps'] == [1, 20]
    assert not torch.cuda.is_initialized()
    value = deepcopy(cpu_payload); value['launch_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='actual CUDA'): probe.validate_payload(value, 'a'*64)


@pytest.fixture
def synthetic_cuda_schema(cpu_payload):
    # Schema fixture only: CPU-produced trajectories must never be published as CUDA.
    value = deepcopy(cpu_payload)
    value['launch_sha256'] = 'a'*64
    value['backend'] = dict(torch_device='cuda:0', warp_device='cuda:0',
                            warp_is_cuda=True, torch_cuda_initialized=True)
    return value


def test_probe_schema_fixture_and_source_binding(synthetic_cuda_schema):
    probe.validate_payload(synthetic_cuda_schema, 'a'*64)
    with pytest.raises(ValueError, match='source-bound'): probe.validate_payload(synthetic_cuda_schema, 'b'*64)


@pytest.mark.parametrize('damage', ['counter', 'capability', 'terminal', 'reset', 'finite', 'isolation_counter'])
def test_probe_validator_refuses_changed_evidence(synthetic_cuda_schema, damage):
    v = synthetic_cuda_schema
    if damage == 'counter': v['normal']['boundaries'][3]['physics_steps'][0] = 4
    elif damage == 'capability': v['learned_stance'] = True
    elif damage == 'terminal': v['isolation']['first_terminal']['physics_step'] = 2
    elif damage == 'reset': v['isolation']['reset_sibling_fields_equal'] = False
    elif damage == 'isolation_counter': v['isolation']['next_boundaries'][1]['physics_steps'][0] = 2
    else: v['normal']['rewards'][0][0] = float('nan')
    with pytest.raises(ValueError): probe.validate_payload(v, 'a'*64)


@pytest.mark.parametrize('message', ['WARNING: CUDA issue', '[WARN] task registration',
    'CCD overflow', 'value NaN', 'value -Inf', 'Infinity', 'nonfinite solved state'])
def test_warning_log_closes_probe(tmp_path, message):
    path = tmp_path/'child.log'; path.write_text(message)
    with pytest.raises(ValueError, match='retained child log'): probe.check_log(path)


def test_normal_log_and_missing_initial_log_are_allowed(tmp_path):
    path = tmp_path/'child.log'; probe.check_log(path)
    path.write_text("Warp 1.12.0 initialized\nModule kernel loaded\nCUDA integration cases complete\n")
    probe.check_log(path)


@pytest.mark.parametrize('extra', ['', '\nvalue NaN', ' WARNING: failure', '\nnonfinite state'])
def test_exact_startup_declaration_does_not_mask_failures(tmp_path, extra):
    path = tmp_path/'child.log'
    path.write_text(probe.STARTUP_INFO+extra+'\n')
    if extra:
        with pytest.raises(ValueError, match='retained child log'): probe.check_log(path)
    else:
        probe.check_log(path)


def test_launch_requires_full_service_budget_plus_closeout():
    probe.check_window(probe.CUTOFF-probe.SERVICE_SECONDS-probe.CLOSEOUT_SECONDS-1)
    with pytest.raises(ValueError, match='closeout window'):
        probe.check_window(probe.CUTOFF-probe.SERVICE_SECONDS-probe.CLOSEOUT_SECONDS)


def test_plan_has_no_optimizer_checkpoint_or_motion_authority():
    p = probe.plan('a'*40, {'fixture': True})
    assert p['optimizer_steps'] == 0 and p['checkpoint_files'] == []
    assert not p['physical_motion_authorized']
    assert p['child_timeout_seconds'] <= probe.supervisor.CELL_SECONDS
    assert p['service_timeout_seconds'] == 180
    assert probe.output_path('a'*40).parent == probe.ROOT/'artifacts/evaluations'
    with pytest.raises(ValueError): probe.output_path('../elsewhere')


@pytest.mark.parametrize('change', [None, 'platform', 'cwd', 'venv', 'host', 'branch',
                                    'source', 'dirty', 'gpu', 'version', 'runtime'])
def test_identity_refuses_drift_before_cuda(tmp_path, monkeypatch, change):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path.parent if change == 'cwd' else tmp_path)
    monkeypatch.setattr(probe.sys, 'platform', 'darwin' if change == 'platform' else 'linux')
    monkeypatch.setattr(probe.sys, 'prefix', str(tmp_path/('other' if change == 'venv' else '.venv')))
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) == '/etc/machine-id': return 'b'*32 if change == 'host' else probe.MACHINE
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    def read(*args):
        return {
            ('git', 'branch', '--show-current'): 'main' if change == 'branch' else probe.BRANCH,
            ('git', 'rev-parse', 'HEAD'): 'b'*40 if change == 'source' else 'a'*40,
            ('git', 'status', '--porcelain'): ' M x' if change == 'dirty' else '',
            ('nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'):
                'other, driver' if change == 'gpu' else probe.GPU+', '+probe.DRIVER,
        }[args]
    monkeypatch.setattr(probe, 'read', read)
    monkeypatch.setattr(probe, 'version', lambda name: 'wrong' if change == 'version' else probe.VERSIONS[name])
    expected = probe.supervisor.parse(probe.PIN_FILE.read_bytes())
    monkeypatch.setattr(probe, 'python_trees', lambda: {} if change == 'runtime' else expected)
    monkeypatch.setattr(probe, 'asset_hashes', lambda: {'synthetic': 'a'*64})
    monkeypatch.setattr(probe, 'digest', lambda _: 'a'*64)
    if change is not None:
        with pytest.raises(ValueError): probe.identity('a'*40)
    else:
        value = probe.identity('a'*40)
        assert value['robot_assets'] == {'synthetic': 'a'*64}
        assert value['complete_binary_runtime_equivalence_verified'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad_log', [False, True])
def test_supervisor_retains_pass_or_failure_without_real_gpu(tmp_path, monkeypatch, synthetic_cuda_schema, bad_log):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    monkeypatch.setattr(probe, 'identity', lambda _: {'fixture': True})
    monkeypatch.setattr(probe, 'check_window', lambda: None)
    monkeypatch.setattr(probe.time, 'time', lambda: probe.CUTOFF-10000)
    monkeypatch.setattr(probe, 'wait_idle', lambda: {'fixture_idle': True})
    output = probe.output_path('a'*40); output.mkdir(parents=True)
    probe.supervisor.write_json(output/'launch.json', probe.plan('a'*40, {'fixture': True}))
    @contextmanager
    def lease(): yield 19
    monkeypatch.setattr(probe.supervisor, 'gpu_lease', lease)
    def process(command, log, **kw):
        assert kw['lock_fd'] == 19 and kw['timeout'] == 120 and kw['cwd'] == tmp_path
        assert kw['env']['CUDA_VISIBLE_DEVICES'] == '0'
        assert command[-3:] == ['child', '--source', 'a'*40]
        log.write_text('WARNING: synthetic failure' if bad_log else 'synthetic normal child')
        kw['guard']()
        payload = deepcopy(synthetic_cuda_schema); payload['launch_sha256'] = probe.digest(output/'launch.json')
        probe.supervisor.write_json(output/'probe.json', payload)
        return {'synthetic_process': True}
    monkeypatch.setattr(probe.supervisor, 'supervised_process', process)
    if bad_log:
        with pytest.raises(ValueError, match='retained child log'): probe.supervise('a'*40)
    else: probe.supervise('a'*40)
    report = probe.supervisor.parse((output/'report.json').read_bytes())
    assert report['decision'] == ('failed' if bad_log else 'cuda-integration-only-passed')
    assert report['retained_files']['child.log'] == probe.digest(output/'child.log')
    with pytest.raises(ValueError, match='no overwrite'): probe.supervise('a'*40)
