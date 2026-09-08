"""CPU loader and lifecycle/evidence tests; synthetic sessions are not physics proof."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mjlab_microduck import foundation_command_session as native
from mjlab_microduck import foundation_command_capture as core
from mjlab_microduck import foundation_command_map as mapping
from test_checkpoint_inference_audit import model_state
from test_foundation_command_capture import session

CELL = mapping.Cell(503,.1,'original')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoint(tmp_path,monkeypatch,model_state):
    state,cfg = model_state
    payload = dict(iter=7998,infos=dict(env_state=dict(common_step_counter=192000)),
                   actor_state_dict=state,optimizer_state_dict={'ignored':torch.tensor(7)})
    path = tmp_path/'model.pt'
    def save():
        torch.save(payload,path)
        monkeypatch.setitem(mapping.CHECKPOINTS,'original',sha(path))
    save()
    return path,cfg,payload,save


def test_load_hashes_exact_deserialized_bytes_and_preserves_rng(checkpoint,monkeypatch):
    path,cfg,payload,_ = checkpoint
    real_load = torch.load
    calls = []
    def checked_load(source,**kwargs):
        assert hashlib.sha256(source.getvalue()).hexdigest() == mapping.CHECKPOINTS['original']
        assert kwargs == dict(map_location='cpu',weights_only=True)
        calls.append(1)
        return real_load(source,**kwargs)
    monkeypatch.setattr(torch,'load',checked_load)
    rng = torch.get_rng_state().clone()
    actor,report = native.load_actor(CELL,path,cfg)
    assert calls == [1] and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,payload['actor_state_dict'][k]) for k,v in actor.state_dict().items())
    assert not actor.training and report['strict_actor_restore']
    assert not any(report[k] for k in ('optimizer_restored','critic_restored','policy_acceptance'))


@pytest.mark.parametrize('change',['bytes','iteration','time','state','normalizer','dtype'])
def test_invalid_checkpoint_cannot_be_loaded(checkpoint,monkeypatch,change):
    path,cfg,payload,save = checkpoint
    if change == 'bytes':
        path.write_bytes(b'changed bytes')
        monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('deserialization before hash check'))
    else:
        if change == 'iteration': payload['iter'] = 8498
        elif change == 'time': payload['infos']['env_state']['common_step_counter'] = 0
        elif change == 'state': payload['actor_state_dict'].pop('mlp.6.bias')
        elif change == 'dtype': payload['actor_state_dict']['mlp.6.bias'] = payload['actor_state_dict']['mlp.6.bias'].double()
        else: payload['actor_state_dict']['obs_normalizer._std'].zero_()
        save()
    with pytest.raises((ValueError,RuntimeError)):
        native.load_actor(CELL,path,cfg)


@pytest.mark.parametrize('policy,path',[
    ('original','artifacts/retained/recovery-seed379-v1/logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt'),
    ('narrow','artifacts/diagnostics/f1r-width-paired-s421-v1/narrow-pilot/model_8498.pt'),
])
def test_retained_real_checkpoint_strict_cpu_restore(policy,path):
    path = ROOT/path
    if not path.exists(): pytest.skip('separately retained checkpoint is unavailable')
    cell = mapping.Cell(503,.1,policy)
    _,cfg = core.prepare_config(cell)
    actor,report = native.load_actor(cell,path,cfg.actor)
    assert report['checkpoint_sha256'] == mapping.CHECKPOINTS[policy]
    assert report['saved_iteration'] == native.ITERATIONS[policy]
    assert report['common_step'] == native.COMMON_STEPS[policy]
    assert report['actor_state_sha256'] == core.actor_digest(actor)
    assert report['actor_audit']['state_unchanged'] and not report['policy_acceptance']


@pytest.fixture
def native_stubs(session,tmp_path,monkeypatch):
    wrapped,actor,stream = session
    env = wrapped.unwrapped
    calls = []
    path = tmp_path/'fixture.pt'; path.write_bytes(b'fixture')
    loading = dict(common_step=192000,actor_state_sha256=core.actor_digest(actor),checkpoint_sha256=sha(path))
    def make_env(**kwargs):
        assert kwargs['device'] == 'cpu' and kwargs['cfg'].scene.num_envs == 8
        calls.append('env'); return env
    def make_wrapper(actual,**kwargs):
        assert actual is env and kwargs['clip_actions'] is None
        calls.append('wrapper-reset'); env.common_step_counter = 0; return wrapped
    env.close = lambda:calls.append('close')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(native,'runtime_identity',lambda:{'selected-test-pin':'fixture'})
    monkeypatch.setattr(native,'load_actor',lambda *a:(actor,loading))
    monkeypatch.setattr(native,'_native_types',lambda:(make_env,make_wrapper))
    monkeypatch.setattr(native.MotorStepStream,'from_robot',lambda *a,**k:stream)
    return env,path,calls


def test_native_factory_order_snapshot_and_cleanup(native_stubs):
    env,path,calls = native_stubs
    with native.native_session(CELL,path,device='cpu') as result:
        assert calls == ['env','wrapper-reset'] and env.common_step_counter == 192000
        assert result['stream'] is env._microduck_motor_step_stream
        assert set(result['config_yaml']) == {'env','agent'}
        assert all(isinstance(v,str) and len(v)>100 for v in result['config_yaml'].values())
        assert not result['runtime']['complete_runtime_equivalence_verified']
    assert calls == ['env','wrapper-reset','close']


@pytest.mark.parametrize('failure',['wrapper','capture','runtime','checkpoint','stream'])
def test_native_factory_closes_on_setup_capture_or_postcheck_failure(native_stubs,monkeypatch,failure):
    env,path,calls = native_stubs
    def bad(*a,**k): raise RuntimeError('injected setup failure')
    if failure == 'wrapper':
        make_env,_ = native._native_types()
        monkeypatch.setattr(native,'_native_types',lambda:(make_env,bad))
    elif failure == 'stream': monkeypatch.setattr(native.MotorStepStream,'from_robot',bad)
    with pytest.raises((ValueError,RuntimeError)):
        with native.native_session(CELL,path,device='cpu'):
            if failure == 'capture': raise RuntimeError('injected capture failure')
            elif failure == 'runtime': monkeypatch.setattr(native,'runtime_identity',lambda:{'changed':True})
            elif failure == 'checkpoint': path.write_bytes(b'changed')
    assert calls[-1] == 'close' and calls.count('close') == 1


def test_native_device_isolation_checked_before_loading(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(native,'runtime_identity',lambda:pytest.fail('runtime touched before isolation check'))
    with pytest.raises(ValueError,match='isolated device'):
        with native.native_session(CELL,'unused',device='cpu'): pass


def assert_manifest(path,status):
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['status'] == status
    assert set(manifest['files']) == {p.name for p in path.iterdir()}-{'manifest.json'}
    for name,record in manifest['files'].items():
        assert record == dict(sha256=sha(path/name),bytes=(path/name).stat().st_size)
    assert manifest['policy_acceptance'] is manifest['physical_motion_authorized'] is False
    return manifest


def test_exclusive_durable_evidence_and_frame_order(tmp_path):
    path = tmp_path/'cell'
    out = native.Evidence(path)
    with pytest.raises(FileExistsError): native.Evidence(path)
    with pytest.raises(ValueError): out.write('../escape',b'no')
    with pytest.raises(ValueError): out.frame(1,{'x':torch.zeros(1)})
    out.frame(0,{'x':torch.ones(2)})
    with pytest.raises(ValueError): out.frame(0,{'x':torch.zeros(1)})
    with pytest.raises(ValueError): out.frame(1,{'x':torch.tensor(float('nan'))})
    out.json('a.json',{'fixture':True})
    with pytest.raises(FileExistsError): out.json('a.json',{})
    out.seal('runtime-failure-stop')
    assert_manifest(path,'runtime-failure-stop')
    assert len((path/'frames.jsonl').read_text().splitlines()) == 1
    with pytest.raises(ValueError): out.seal('runtime-failure-stop')


def factory_for(session,closed,*,post_error=False,missing_config=False):
    @contextmanager
    def factory(*a,**k):
        wrapped,actor,stream = session
        try:
            yield dict(wrapped=wrapped,actor=actor,stream=stream,
                       loading={'actor_state_sha256':core.actor_digest(actor)},runtime={'synthetic':True},
                       config_yaml={'env':'fixture: true\n',**({} if missing_config else {'agent':'fixture: true\n'})})
            if post_error: raise RuntimeError('post-capture identity failure')
        finally: closed.append(True)
    return factory


@pytest.mark.parametrize('terminal',[None,2])
def test_retained_cell_is_diagnostic_only_with_journal_and_hashes(session,tmp_path,terminal):
    closed = []; path = tmp_path/'cell'
    session[0].terminal_at = terminal
    score = native.retain_cell(CELL,'fixture',path,device='cpu',session_factory=factory_for(session,closed),clock=lambda:1.)
    expected = 'descriptive-cell-within-checks' if terminal is None else 'safety-or-coverage-stop'
    assert closed == [True] and score['classification'] == expected
    assert_manifest(path,'captured-diagnostic-only')
    rows = [json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(400 if terminal is None else 3))
    raw = torch.load(path/'trace.pt',weights_only=True)
    assert bool(raw['dones'][-1,2]) == (terminal is not None)
    assert torch.all(raw['pre_force'][-1] == .12)
    decision = json.loads((path/'decision.json').read_text())
    assert not any(decision[k] for k in ('native_factory_used','policy_acceptance','training_admitted'))


@pytest.mark.parametrize('failure',['physics','actor','postcheck','snapshot','expired','clock'])
def test_failure_is_closed_and_durable_without_passing_result(session,tmp_path,failure):
    wrapped,actor,_ = session; wrapped.terminal_at = 2
    if failure == 'physics': wrapped.raise_step = True
    elif failure == 'actor': actor.nan = True
    factory = factory_for(session,closed:=[],post_error=failure=='postcheck',missing_config=failure=='snapshot')
    clock = (iter([1.,122.]).__next__ if failure == 'expired'
             else iter([2.,1.]).__next__ if failure == 'clock' else lambda:1.)
    path = tmp_path/'cell'
    with pytest.raises((ValueError,core.CaptureFailure,RuntimeError)):
        native.retain_cell(CELL,'fixture',path,device='cpu',session_factory=factory,clock=clock)
    assert closed == [True]
    assert_manifest(path,'runtime-failure-stop')
    assert not (path/'decision.json').exists()
    receipt = json.loads((path/'failure.json').read_text())
    assert receipt['journaled_complete_steps'] == (3 if failure == 'postcheck' else 0)
    if failure in ('physics','actor'):
        frames = torch.load(path/'partial-failure.pt',weights_only=True)
        assert len(frames) == 1 and 'pre_force' not in frames[0]


def test_retention_budget_closes_failure_even_after_capture(session,tmp_path,monkeypatch):
    session[0].terminal_at = 0
    now = [1.]; closed = []
    save = native._tensor_bytes
    def slow(value):
        raw = save(value); now[0] = 122.; return raw
    monkeypatch.setattr(native,'_tensor_bytes',slow)
    path = tmp_path/'cell'
    with pytest.raises(ValueError,match='budget'):
        native.retain_cell(CELL,'fixture',path,device='cpu',session_factory=factory_for(session,closed),clock=lambda:now[0])
    assert closed == [True]
    assert_manifest(path,'runtime-failure-stop')  # Manifest, not provisional decision, is authoritative.
    assert not (path/'decision.json').exists()


def test_journal_sink_receives_clones_and_sink_failure_keeps_completed_step(session):
    session[0].terminal_at = 0
    def sink(step,row):
        row['pre_force'].zero_()
        raise OSError('synthetic journal failure')
    with pytest.raises(core.CaptureFailure,match='journal failure') as error:
        core.capture(CELL,*session,budget_seconds=120,clock=lambda:1.,on_frame=sink)
    assert error.value.simulation_steps == 1
    assert torch.all(error.value.frames[0]['pre_force'] == .12)


def test_storage_failure_keeps_causal_exception_without_claiming_a_receipt(session,tmp_path,monkeypatch):
    session[0].raise_step = True
    write = native.Evidence.write
    def broken(self,name,raw):
        if name == 'partial-failure.pt': raise OSError('injected storage failure')
        return write(self,name,raw)
    monkeypatch.setattr(native.Evidence,'write',broken)
    closed = []; path = tmp_path/'cell'
    with pytest.raises(core.CaptureFailure,match='synthetic physics failure') as error:
        native.retain_cell(CELL,'fixture',path,device='cpu',session_factory=factory_for(session,closed),clock=lambda:1.)
    assert closed == [True]
    assert 'storage failure' in error.value.__notes__[0]
    assert (path/'frames.jsonl').exists() and not (path/'manifest.json').exists()
