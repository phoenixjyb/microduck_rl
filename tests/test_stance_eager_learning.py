"""Bounded CPU fixtures do not establish a real GPU run or learned balance."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import os
import subprocess
import sys
import time

import pytest
import torch

from mjlab_microduck import stance_eager_learning as eager
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_learner_checkpoint as codec


class Synthetic:
    n = 64
    device = torch.device('cpu')
    def __init__(self):
        self.live = torch.ones(self.n, dtype=torch.bool); self.tick = 0
        self.obs = torch.zeros(self.n, 50)

    def observations(self): return dict(actor=self.obs[:, :44].clone(), critic=self.obs.clone())

    def step(self, action):
        self.tick += 1; self.obs[:, 0] = self.tick/10000
        terminated = torch.zeros(self.n, dtype=torch.bool); terminated[0] = self.tick % 3 == 0
        timed_out = torch.zeros(self.n, dtype=torch.bool); timed_out[1] = self.tick % 7 == 0
        self.live = ~(terminated | timed_out)
        self.records = [None if live else dict(tick=self.tick, world=i) for i, live in enumerate(self.live)]
        return dict(observation=self.observations(), reward=torch.full((self.n,), .03),
            terminated=terminated, timed_out=timed_out, live=self.live.clone(),
            executed_steps=torch.full((self.n,), 10, dtype=torch.long),
            terminal_records=deepcopy(self.records), boundaries=[{'synthetic': True}])

    def reset(self, rows):
        self.live[rows] = True; self.obs[rows, 0] = 0
        return deepcopy(self.records)


def test_distinct_seed_exact_bounds_no_resume_and_no_pilot_admission():
    learner = eager.EagerLearner()
    assert (learner.n, learner.seed, learner.UPDATE_LIMIT) == (64, 563, 128)
    assert learner.initial_hash != eager.smoke.SmokeStanceLearner().initial_hash
    meta = eager.identity('a'*40, 'b'*64, 'c'*64, learner, 127)
    raw = cp.encode(learner.actor, learner.critic, meta)
    with pytest.raises(ValueError, match='only declared pilot'):
        cp.load_evaluation(raw, sha256(raw).hexdigest(), meta)
    for key, value in [('iteration', 128), ('training_seed', 523), ('worlds', 512), ('purpose', 'pilot')]:
        changed = deepcopy(meta); changed[key] = value
        with pytest.raises(ValueError): cp.encode(learner.actor, learner.critic, changed)
    with pytest.raises(ValueError): codec.encode(learner, dict(source='a'*40, runtime_sha256='b'*64, fixture_launch_sha256='c'*64))
    with pytest.raises(TypeError): eager.EagerLearner(512)
    learner.updates = 128
    with pytest.raises(ValueError): learner.collect_one(Synthetic())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('offset,accepted', [(1560, False), (1561, True), (3600, True), (3601, False), (-1, False)])
def test_explicit_fresh_window_not_old_campaign_cutoff(monkeypatch, offset, accepted):
    monkeypatch.setattr(eager.time, 'time', lambda: 10000)
    monkeypatch.setattr(eager.host, 'CUTOFF', 1)
    if accepted: eager.check_window(10000+offset, launching=True)
    else:
        with pytest.raises(ValueError): eager.check_window(10000+offset, launching=True)
    with pytest.raises(ValueError): eager.check_window(True)
    with pytest.raises(ValueError): eager.check_window(float('nan'))
    with pytest.raises(ValueError): eager.check_window(10600)


@pytest.mark.parametrize('key', [None, 'MainPID', 'ActiveState', 'RuntimeMaxUSec', 'KillMode'])
def test_exact_service_and_fixed_independent_watchdog(monkeypatch, key):
    fields = dict(MainPID=str(os.getpid()), ActiveState='active', RuntimeMaxUSec='16min', KillMode='control-group')
    if key: fields[key] = 'wrong'
    monkeypatch.setattr(eager.host, 'read', lambda *a: '\n'.join(k+'='+v for k, v in fields.items()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    if key:
        with pytest.raises(ValueError): eager.check_service('a'*40)
    else: eager.check_service('a'*40)
    assert eager.CHILD_SECONDS == eager.smoke.CHILD_SECONDS == 900
    assert eager.SERVICE_SECONDS == eager.smoke.SERVICE_SECONDS == 960


@pytest.fixture(scope='module')
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('eager-synthetic'); learner = eager.EagerLearner()
    launch = eager.plan('a'*40, {}, 'c'*64, 123456)
    for name, value in [('launch.json', launch), ('runtime.json', {})]: eager.supervisor.write_json(root/name, value)
    eager.smoke.write_bytes(root/'child.log', b'Synthetic fixture only\n')
    exports = eager.run_updates(learner, eager.smoke.PhysicsBridge(Synthetic()), root,
        'a'*40, 'b'*64, 'c'*64, deadline=time.monotonic()+120)
    # Explicit schema stand-in. This is NOT a real CUDA completion receipt.
    eager.supervisor.write_json(root/'completed.json', dict(protocol=eager.PROTOCOL,
        launch_sha256='b'*64, completed_updates=128, checkpoints=exports, physics_device='cuda:0',
        learner_device='cpu', forward_graph=False, seed=563, worlds=64,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False))
    return root, learner, launch


def test_exact_128_real_cpu_optimizer_updates_and_exports(synthetic_run):
    root, learner, launch = synthetic_run
    result = eager.verify_completed(root, 'a'*40, 'b'*64, launch)
    assert len(result['checkpoints']) == 129 and learner.updates == 128
    assert len(list(root.glob('tick-*.json'))) == 3072
    assert len(list(root.glob('update-*.json'))) == 128
    assert cp.state_hash(cp.states_of(learner.actor, learner.critic)) != learner.initial_hash
    assert not torch.cuda.is_initialized()


def test_verify_rejects_modified_last_weights_before_load(synthetic_run, monkeypatch):
    root, _, launch = synthetic_run; original = eager.supervisor.file_bytes
    def damaged(path, **kwargs):
        data = original(path, **kwargs)
        return data+b'damaged' if path.name == 'model_127.pt' else data
    monkeypatch.setattr(eager.supervisor, 'file_bytes', damaged)
    with pytest.raises(ValueError, match='hash/identity'): eager.verify_completed(root, 'a'*40, 'b'*64, launch)


def test_expired_collection_retains_initializer_only(tmp_path):
    with pytest.raises(ValueError, match='collection deadline'):
        eager.run_updates(eager.EagerLearner(), eager.smoke.PhysicsBridge(Synthetic()), tmp_path,
            'a'*40, 'b'*64, 'c'*64, deadline=time.monotonic()-1)
    assert {p.name for p in tmp_path.iterdir()} == {'initial.pt'}


def test_launch_hash_checked_before_host_or_deserialization(tmp_path, monkeypatch):
    monkeypatch.setattr(eager, 'output_path', lambda source: tmp_path)
    eager.supervisor.write_json(tmp_path/'launch.json', {})
    def forbidden(*a, **kw): pytest.fail('hash must be checked first')
    monkeypatch.setattr(eager.host, 'identity', forbidden)
    with pytest.raises(ValueError, match='launch hash'): eager.inputs_check('a'*40, 'b'*64)


def test_failed_supervision_closes_once_without_retry(tmp_path, monkeypatch):
    launch = eager.plan('a'*40, {}, 'c'*64, int(time.time())+3500)
    for name in ('launch.json', 'runtime.json'): eager.supervisor.write_json(tmp_path/name, {})
    monkeypatch.setattr(eager, 'check_service', lambda source: None)
    monkeypatch.setattr(eager, 'inputs_check', lambda *a: launch)
    monkeypatch.setattr(eager, 'output_path', lambda source: tmp_path)
    monkeypatch.setattr(eager.host, 'wait_idle', lambda: {})
    @contextmanager
    def lease(): yield 4
    monkeypatch.setattr(eager.supervisor, 'gpu_lease', lease)
    calls = []
    def child(command, *a, **kw):
        calls.append(command)
        assert command[1:4] == ['-m', eager.MODULE, 'child']
        raise ValueError('synthetic child failure')
    monkeypatch.setattr(eager.supervisor, 'supervised_stance_smoke', child)
    with pytest.raises(ValueError, match='synthetic child'): eager.supervise('a'*40, 'b'*64)
    report = eager.supervisor.parse((tmp_path/'report.json').read_bytes())
    assert report['decision'] == 'failed' and report['learned_stance'] is False and len(calls) == 1
    with pytest.raises(ValueError, match='one fresh'): eager.supervise('a'*40, 'b'*64)


def test_cli_has_no_resume_graph_pilot_or_evaluation_mode():
    result = subprocess.run([sys.executable, '-m', eager.MODULE, '--help'],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and '{prepare,supervise,child}' in result.stdout
    for option in ('--resume', '--graph', '--worlds', '--updates'): assert option not in result.stdout
