"""CPU test seams only: no actual CUDA job or learned-skill acceptance."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
import time

import pytest
import torch

from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_learner_checkpoint as codec
from mjlab_microduck.stance_ppo import CpuStanceLearner, STEPS
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


class Synthetic64:
    synthetic_ppo_fixture = True
    def __init__(self):
        self.n = 64; self.device = torch.device('cpu'); self.tick = 0
        self.live = torch.ones(64, dtype=torch.bool); self.obs = torch.zeros(64, 50)
        self.terminals = [None]*64

    def observations(self): return dict(actor=self.obs[:, :44].clone(), critic=self.obs.clone())

    def step(self, action):
        self.tick += 1; self.obs[:, 0] = self.tick/1000
        terminated = torch.zeros(64, dtype=torch.bool); terminated[0] = self.tick % 3 == 0
        timed_out = torch.zeros(64, dtype=torch.bool); timed_out[1] = self.tick % 7 == 0
        self.live = ~(terminated | timed_out)
        self.terminals = [None if bool(self.live[i]) else dict(tick=self.tick, world=i) for i in range(64)]
        return dict(observation=self.observations(), reward=torch.full((64,), .03),
            terminated=terminated, timed_out=timed_out, live=self.live.clone(),
            executed_steps=torch.full((64,), 10, dtype=torch.long),
            terminal_records=deepcopy(self.terminals), boundaries=[{'synthetic': True}])

    def reset(self, rows):
        retained = deepcopy(self.terminals)
        self.obs[rows, 0] = 0; self.live[rows] = True
        self.terminals = [None]*64
        return retained


def test_smoke_has_fixed_dimensions_no_fixture_resume_or_pilot():
    learner = smoke.SmokeStanceLearner()
    assert learner.n == 64 and learner.seed == 523 and learner.UPDATE_LIMIT == 16
    assert all(p.device.type == 'cpu' for p in learner.actor.parameters())
    with pytest.raises(TypeError): smoke.SmokeStanceLearner(512)
    with pytest.raises(ValueError): CpuStanceLearner(64)
    with pytest.raises(ValueError): codec.encode(learner, dict(source='a'*40, runtime_sha256='b'*64, fixture_launch_sha256='c'*64))
    learner.updates = 16
    with pytest.raises(ValueError, match='collecting'): learner.collect_one(Synthetic64())
    assert not torch.cuda.is_initialized()


def test_actual_short_cpu_bridge_preserves_actions_and_pre_reset_data():
    env = WarpStanceRuntime(2, device='cpu'); bridge = smoke.PhysicsBridge(env)
    learner = CpuStanceLearner()
    result = learner.collect_one(bridge)
    assert bridge.physics_device == 'cpu' and learner.storage.step == 1
    assert result['executed_steps'].tolist() == [10, 10]
    assert result['boundaries'][-1]['physics_steps'].tolist() == [10, 10]
    evidence = smoke.tick_evidence(result, 0)
    assert evidence['executed_joint_samples'] == 280
    assert evidence['max_applied_torque_nm'] <= .36 and evidence['minimum_height_m'] > .08
    assert not evidence['thermal_model_available'] and not evidence['spring_bottoming_applicable']
    result['observation']['actor'].zero_()
    assert not torch.equal(result['observation']['actor'], env.observations()['actor'])
    with pytest.raises(ValueError): bridge.step(torch.full((2, 10), float('nan')))
    assert not learner.algorithm.optimizer.state


def test_bridge_timeout_uses_terminal_not_reset_value(monkeypatch):
    learner = smoke.SmokeStanceLearner(); env = Synthetic64(); env.tick = 6
    monkeypatch.setattr(learner.critic, 'forward', lambda obs, **kw: obs['critic'][:, :1])
    result = learner.collect_one(smoke.PhysicsBridge(env))
    assert learner.storage.rewards[0, 1, 0] == pytest.approx(.03+.99*.007)
    assert result['observation']['critic'][1, 0] == pytest.approx(.007)
    assert learner.next_obs['critic'][1, 0] == 0


@pytest.fixture(scope='module')
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-smoke')
    learner = smoke.SmokeStanceLearner(); env = smoke.PhysicsBridge(Synthetic64())
    exports = smoke.run_updates(learner, env, root, 'a'*40, 'b'*64, 'c'*64, deadline=time.monotonic()+60)
    # This schema stand-in explicitly does NOT prove a CUDA backend.
    smoke.supervisor.write_json(root/'completed.json', dict(protocol=smoke.PROTOCOL,
        launch_sha256='b'*64, completed_updates=16, checkpoints=exports,
        physics_device='cuda:0', learner_device='cpu', seed=523, worlds=64,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False))
    return root, learner


def test_real_synthetic_optimizer_runs_exact_16_updates_and_retains_all(synthetic_run):
    root, learner = synthetic_run
    assert learner.updates == 16 and len(list(root.glob('tick-*.json'))) == 384
    assert len(list(root.glob('*.pt'))) == 17
    assert all(s['step'] == 320 for s in learner.algorithm.optimizer.state.values())
    assert checkpoint.state_hash(checkpoint.states_of(learner.actor, learner.critic)) != learner.initial_hash
    result = smoke.verify_completed(root, 'a'*40, 'b'*64, {'runtime_sha256': 'c'*64})
    assert not result['pilot_parent_authorized']
    with pytest.raises(ValueError, match='only declared pilot'):
        checkpoint.load_evaluation((root/'model_15.pt').read_bytes(), result['checkpoints'][-1]['sha256'], result['checkpoints'][-1]['identity'])


@pytest.mark.parametrize('damage', ['count', 'promotion', 'file_hash', 'checkpoint_identity', 'missing_tick'])
def test_completed_verifier_refuses_corruption(synthetic_run, monkeypatch, damage):
    root, _ = synthetic_run; original = smoke.supervisor.file_bytes
    def damaged(path, **kwargs):
        raw = original(path, **kwargs)
        if path.name == 'completed.json':
            value = json.loads(raw)
            if damage == 'count': value['completed_updates'] = 15
            if damage == 'promotion': value['pilot_parent_authorized'] = True
            if damage == 'file_hash': value['checkpoints'][-1]['sha256'] = 'f'*64
            if damage == 'checkpoint_identity': value['checkpoints'][-1]['identity']['purpose'] = 'pilot'
            return json.dumps(value).encode()
        if damage == 'missing_tick' and path.name == 'tick-00-00.json': raise ValueError('missing tick')
        return raw
    monkeypatch.setattr(smoke.supervisor, 'file_bytes', damaged)
    with pytest.raises(ValueError): smoke.verify_completed(root, 'a'*40, 'b'*64, {'runtime_sha256': 'c'*64})


def test_early_budget_retains_only_initial_checkpoint(tmp_path):
    learner = smoke.SmokeStanceLearner()
    with pytest.raises(ValueError, match='collection deadline'):
        smoke.run_updates(learner, smoke.PhysicsBridge(Synthetic64()), tmp_path,
            'a'*40, 'b'*64, 'c'*64, deadline=time.monotonic()-1)
    assert {p.name for p in tmp_path.iterdir()} == {'initial.pt'}
    assert learner.updates == 0 and not learner.algorithm.optimizer.state
    with pytest.raises(FileExistsError):
        smoke.run_updates(learner, smoke.PhysicsBridge(Synthetic64()), tmp_path,
            'a'*40, 'b'*64, 'c'*64, deadline=time.monotonic()+60)


@pytest.mark.parametrize('key', [None, 'MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState'])
def test_independent_service_gate(monkeypatch, key):
    values = dict(MainPID=str(os.getpid()), RuntimeMaxUSec='16min', KillMode='control-group', ActiveState='active')
    if key: values[key] = 'wrong'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(smoke.host, 'read', lambda *args: values[args[-2]])
    if key:
        with pytest.raises(ValueError, match='timed smoke service'): smoke.check_service('a'*40)
    else: smoke.check_service('a'*40)


def test_window_and_fixed_timeout_are_separate_from_old_map(monkeypatch):
    monkeypatch.setattr(smoke.time, 'time', lambda: smoke.host.CUTOFF-1561)
    smoke.check_window()
    monkeypatch.setattr(smoke.time, 'time', lambda: smoke.host.CUTOFF-1560)
    with pytest.raises(ValueError, match='fit window'): smoke.check_window()
    seen = []
    monkeypatch.setattr(smoke.supervisor, '_timed_process', lambda *a, **kw: seen.append(kw))
    smoke.supervisor.supervised_stance_smoke([], None, cwd=None, env={}, lock_fd=4)
    assert seen[-1]['timeout'] == 900
    with pytest.raises(ValueError):
        smoke.supervisor.supervised_process([], None, cwd=None, env={}, lock_fd=4, timeout=121)


@pytest.mark.parametrize('bad', [False, True])
def test_supervisor_guard_retains_failure_and_does_not_retry(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(smoke.host, 'ROOT', tmp_path)
    monkeypatch.setattr(smoke, 'check_window', lambda: None)
    monkeypatch.setattr(smoke, 'check_service', lambda _: None)
    monkeypatch.setattr(smoke.time, 'time', lambda: smoke.host.CUTOFF-10000)
    monkeypatch.setattr(smoke.host, 'identity', lambda _: {'synthetic': True})
    monkeypatch.setattr(smoke.host, 'wait_idle', lambda: {'synthetic': True})
    launch = smoke.plan('a'*40, {'synthetic': True}, 'c'*64)
    monkeypatch.setattr(smoke, 'inputs_check', lambda *a: launch)
    root = smoke.output_path('a'*40); root.mkdir(parents=True)
    smoke.supervisor.write_json(root/'launch.json', launch)
    smoke.supervisor.write_json(root/'runtime.json', {'synthetic': True})
    @contextmanager
    def lease(): yield 4
    monkeypatch.setattr(smoke.supervisor, 'gpu_lease', lease)
    def process(command, log, **kw):
        assert kw['lock_fd'] == 4 and kw['env']['CUDA_VISIBLE_DEVICES'] == '0'
        assert command[-2:] == ['--lock-fd', '4']
        log.write_text('WARNING: synthetic failure' if bad else 'synthetic normal')
        kw['guard']()
        return {'synthetic_process': True}
    monkeypatch.setattr(smoke.supervisor, 'supervised_stance_smoke', process)
    monkeypatch.setattr(smoke, 'verify_completed', lambda *a: {'synthetic': True})
    if bad:
        with pytest.raises(ValueError): smoke.supervise('a'*40, 'b'*64)
    else: smoke.supervise('a'*40, 'b'*64)
    report = json.loads((root/'report.json').read_text())
    assert report['decision'] == ('failed' if bad else 'disposable-training-smoke-complete-not-capability')
    assert 'child.log' in report['files']
    with pytest.raises(ValueError, match='one fresh'): smoke.supervise('a'*40, 'b'*64)


def test_inherited_lock_must_really_be_held(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.supervisor, 'LOCK', tmp_path/'gpu.lock')
    with smoke.supervisor.gpu_lease() as fd: smoke.inherited_lease(fd)
    with (tmp_path/'gpu.lock').open('r') as f:
        with pytest.raises(ValueError, match='not held'): smoke.inherited_lease(f.fileno())
