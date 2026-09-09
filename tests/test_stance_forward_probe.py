"""CPU and synthetic service tests; fake graph dispatch is not CUDA evidence."""
from contextlib import contextmanager
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from mjlab_microduck import stance_forward_probe as probe
from mjlab_microduck import stance_forward_graph as graph


def test_full_cpu_snapshot_restores_contents_without_replacing_arrays():
    env = probe.WarpStanceRuntime(2, device='cpu')
    snapshot = probe.InputSnapshot(env)
    assert len(snapshot.dest) > 400 and snapshot.nbytes > 10*1024*1024
    assert snapshot.payload['arrays']['/model/body_mass']['strides'] == list(env.model.body_mass.strides)
    assert snapshot.backup['/model/body_mass'].strides != env.model.body_mass.strides
    env._view('qacc_warmstart')[0] = 7
    env._view('ctrl')[0] = .1
    wp.to_torch(env.model.dof_damping)[0] += 1
    assert snapshot.restore() == snapshot.identity
    assert probe.binding((env.model, env.data)) == snapshot.signature
    env.model.opt.iterations += 1
    with pytest.raises(ValueError, match='static configuration'): snapshot.restore()


def test_snapshot_rejects_array_replacement_and_size_before_backup(monkeypatch):
    env = probe.WarpStanceRuntime(2, device='cpu')
    snapshot = probe.InputSnapshot(env)
    env.data.ctrl = wp.clone(env.data.ctrl)
    with pytest.raises(ValueError, match='allocation'): snapshot.restore()
    monkeypatch.setattr(probe, 'SNAPSHOT_LIMIT', 1)
    monkeypatch.setattr(probe.wp, 'clone', lambda *a, **k: pytest.fail('oversized snapshot cloned'))
    with pytest.raises(ValueError, match='bounded complete'): probe.InputSnapshot(env)


@pytest.fixture
def cpu_batch(tmp_path, monkeypatch):
    def fake_enable(env):
        # Deliberately mimic a capture with dynamic side effects: each measured
        # call must restore them. This dispatch remains eager CPU, NOT a graph.
        env._view('qvel')[0] += .1
        env.forward_graph = SimpleNamespace(run=lambda m, d: graph.mjwarp.forward(m, d))
    monkeypatch.setattr(probe.WarpStanceRuntime, 'enable_forward_graph', fake_enable)
    monkeypatch.setattr(probe.WarpStanceRuntime, 'step', lambda *a, **k: pytest.fail('policy tick called'))
    root = tmp_path/'run'; root.mkdir()
    result = probe.batch(2, 'cpu', lambda n, v: probe.retain(root, n, v))
    return root, result


def test_eight_cpu_calls_restore_capture_side_effects_and_preserve_no_integration(cpu_batch):
    root, result = cpu_batch
    assert [c['graph'] for c in result['calls']] == list(probe.ORDER)
    assert len({c['input_sha256'] for c in result['calls']}) == 1
    assert result['comparison']['decision'] == 'same-input-forward-repeatable-in-this-sample'
    assert not result['training_admitted'] and result['integration_steps'] == 0
    assert len(list(root.iterdir())) == 9
    pairs = result['comparison']['pairs']
    assert {k: sum(p['kind'] == k for p in pairs) for k in ('eager', 'graph', 'cross')} == {
        'eager': 6, 'graph': 6, 'cross': 16}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('arm,decision', [(0, 'forward-baseline-nonrepeatable'),
                                        (1, 'candidate-difference-requires-diagnosis')])
def test_nonrepeatability_classification_does_not_admit_graph(cpu_batch, arm, decision):
    root, _ = cpu_batch
    values = [torch.load(root/f'output-2-{i}.pt', map_location='cpu', weights_only=True) for i in range(8)]
    values[arm]['dynamics']['qacc'][0, 0] += .001
    result = probe.comparison(values)
    assert result['decision'] == decision
    assert not result['training_admitted'] and not result['graph_equivalence_established']


def synthetic_files(root, result):
    result = deepcopy(result); result['device'] = result['warp_device'] = 'cuda:0'
    probe.host.supervisor.write_json(root/'batch-2.json', result)
    probe.host.supervisor.write_json(root/'launch.json', {'synthetic': True})
    (root/'child.log').write_text('Synthetic CPU fixture only\n')


def test_cpu_raw_file_revalidation_and_no_overwrite(cpu_batch, monkeypatch):
    root, result = cpu_batch
    monkeypatch.setattr(probe, 'WORLDS', (2,))
    synthetic_files(root, result)
    checked = probe.verify(root)
    assert checked['decision'] == 'forward-diagnostic-complete' and not checked['training_admitted']
    assert checked == probe.verify(root)
    with pytest.raises(FileExistsError): probe.retain(root, 'input-2.pt', {})


def test_raw_tampering_fails_before_deserialization(cpu_batch, monkeypatch):
    root, result = cpu_batch; monkeypatch.setattr(probe, 'WORLDS', (2,))
    synthetic_files(root, result)
    path = root/'input-2.pt'; path.write_bytes(path.read_bytes()+b' ')
    monkeypatch.setattr(probe.torch, 'load', lambda *a, **k: pytest.fail('unverified load'))
    with pytest.raises(ValueError, match='raw forward evidence hash'): probe.verify(root)


def test_raw_active_nonfinite_field_fails(cpu_batch):
    root, _ = cpu_batch
    value = torch.load(root/'output-2-0.pt', map_location='cpu', weights_only=True)
    value['constraints'][0]['force'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'): probe.validate_output(value, 2)


def test_empty_friction_is_not_an_informative_forward_trial(cpu_batch):
    root, _ = cpu_batch
    value = torch.load(root/'output-2-0.pt', map_location='cpu', weights_only=True)
    value['solver']['nf'].zero_()
    with pytest.raises(ValueError, match='friction constraints'): probe.validate_output(value, 2)


@pytest.mark.parametrize('damage', ['input', 'order', 'missing'])
def test_changed_replay_controls_are_rejected(cpu_batch, monkeypatch, damage):
    root, result = cpu_batch; monkeypatch.setattr(probe, 'WORLDS', (2,))
    if damage == 'input': result['calls'][0]['input_sha256'] = '0'*64
    if damage == 'order': result['calls'][0]['graph'] = True
    if damage == 'missing': (root/'output-2-0.pt').unlink()
    synthetic_files(root, result)
    with pytest.raises(ValueError): probe.verify(root)


def test_retained_quota_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'TOTAL_LIMIT', 16*1024*1024)
    with pytest.raises(ValueError, match='quota'): probe.retain(tmp_path, 'too-big.pt', {})
    assert not list(tmp_path.iterdir())


def test_new_14_00_deadline_includes_service_and_closeout(monkeypatch):
    monkeypatch.setattr(probe.host, 'check_window', lambda: None)
    monkeypatch.setattr(probe.time, 'time', lambda: probe.DEADLINE-781)
    probe.check_window()
    monkeypatch.setattr(probe.time, 'time', lambda: probe.DEADLINE-780)
    with pytest.raises(ValueError, match='14:00'): probe.check_window()


@pytest.mark.parametrize('bad', [False, True])
def test_service_supervisor_guards_command_and_failed_closeout(tmp_path, monkeypatch, bad):
    root = tmp_path/'artifacts/evaluations'/('stance-forward-'+'a'*12); root.mkdir(parents=True)
    monkeypatch.setattr(probe.host, 'ROOT', tmp_path)
    monkeypatch.setattr(probe.host, 'check_window', lambda: None)
    monkeypatch.setattr(probe, 'checked', lambda *a: dict(inputs={'synthetic': True}))
    monkeypatch.setattr(probe.host, 'identity', lambda *a: {'synthetic': True})
    monkeypatch.setattr(probe.time, 'time', lambda: probe.DEADLINE-10000)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    states = dict(MainPID=str(os.getpid()), RuntimeMaxUSec='3min', KillMode='control-group', ActiveState='active')
    monkeypatch.setattr(probe.host, 'read', lambda *a: states[a[-2]])
    monkeypatch.setattr(probe.host, 'wait_idle', lambda: {'synthetic': True})
    @contextmanager
    def lease(): yield 4
    monkeypatch.setattr(probe.host.supervisor, 'gpu_lease', lease)
    def child(command, log, **kw):
        assert command[1:4] == ['-m', probe.MODULE, 'child']
        assert kw['timeout'] == 120 and kw['lock_fd'] == 4
        log.write_text('WARNING: synthetic failure' if bad else 'Synthetic normal')
        kw['guard']()
        return dict(returncode=0)
    monkeypatch.setattr(probe.host.supervisor, 'supervised_process', child)
    monkeypatch.setattr(probe, 'verify', lambda _: dict(decision='forward-diagnostic-complete'))
    probe.host.supervisor.write_json(root/'launch.json', {'synthetic': True})
    if bad:
        with pytest.raises(ValueError, match='warning'): probe.supervise('a'*40, 'b'*64)
    else: probe.supervise('a'*40, 'b'*64)
    report = probe.host.supervisor.parse((root/'report.json').read_bytes())
    assert report['decision'] == ('failed' if bad else 'forward-diagnostic-complete')
    with pytest.raises(ValueError, match='one fresh'): probe.supervise('a'*40, 'b'*64)
