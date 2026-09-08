"""CPU-only orchestration and real disposable CPU subprocess tests, never GPU jobs."""

from contextlib import contextmanager
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import yaml

from mjlab_microduck import foundation_command_campaign as campaign
from mjlab_microduck import foundation_command_session as native
from mjlab_microduck import foundation_command_capture as core
from mjlab_microduck import foundation_command_map as mapping
from test_foundation_command_capture import session
from test_foundation_command_map import trace
from test_foundation_reset_evidence import synthetic_report

CPU_ENV = {'PATH':os.defpath,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}


@pytest.fixture
def plan():
    return campaign.LaunchPlan('a'*40,'b'*32,1.,5000.,{'original':'models/a.pt','narrow':'models/b.pt'},
        {'.venv/lib/python3.12/site-packages/'+name:'c'*64 for name in campaign.REQUIRED_RUNTIME_SUFFIXES})


@pytest.mark.parametrize('field,value',[
    ('source','latest'),('machine_id','100.100'),('not_before_unix',True),('deadline_unix',float('nan')),
    ('deadline_unix',2000.),('runtime_scope','complete'),('checkpoints',{'original':'a.pt'}),
    ('checkpoints',{'original':'../a.pt','narrow':'b.pt'}),('runtime_files',{}),
    ('runtime_files',{'/outside':'c'*64}),
])
def test_plan_rejects_unbound_or_unsafe_identity(plan,field,value):
    with pytest.raises(ValueError): replace(plan,**{field:value}).validate()


def test_json_duplicate_keys_and_nan_are_refused():
    for raw in ('{"a":1,"a":2}','{"nested":[NaN]}'):
        with pytest.raises(ValueError): campaign.parse(raw)


def test_empty_runtime_source_can_be_pinned_but_empty_evidence_is_refused(tmp_path):
    path = tmp_path/'__init__.py'; path.touch()
    with pytest.raises(ValueError): campaign.file_bytes(path)
    assert campaign.file_bytes(path,allow_empty=True) == b''
    assert campaign.digest(campaign.file_bytes(path,allow_empty=True)) == campaign.digest(b'')
    path.write_text('changed = True\n')
    assert campaign.digest(campaign.file_bytes(path,allow_empty=True)) != campaign.digest(b'')


def test_child_environment_does_not_inherit_secrets_or_python_injection(monkeypatch):
    for name in ('FIXTURE_API_SECRET','PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH'):
        monkeypatch.setenv(name,'synthetic-not-a-secret')
    env = campaign.child_environment()
    assert not any(name in env for name in ('FIXTURE_API_SECRET','PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH'))
    assert set(env) <= {'HOME','USER','LOGNAME','XDG_CACHE_HOME','XDG_RUNTIME_DIR','PATH',
                        'CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONUNBUFFERED'}


@pytest.mark.parametrize('change',[None,'host','platform','cwd','branch','head','dirty','runtime','checkpoint'])
def test_preflight_checks_exact_host_source_and_file_bytes(plan,tmp_path,monkeypatch,change):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path.parent if change == 'cwd' else tmp_path)
    monkeypatch.setattr(campaign.sys,'platform','darwin' if change == 'platform' else 'linux')
    original = Path.read_text
    def read(path,*a,**k):
        if str(path) == '/etc/machine-id': return 'd'*32 if change == 'host' else plan.machine_id
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',read)
    def git(args,**kwargs):
        assert kwargs['timeout'] == 5
        return {('git','branch','--show-current'):'main' if change == 'branch' else campaign.BRANCH,
                ('git','rev-parse','HEAD'):'d'*40 if change == 'head' else plan.source,
                ('git','status','--porcelain'):' M file' if change == 'dirty' else ''}[tuple(args)]
    monkeypatch.setattr(campaign.subprocess,'check_output',git)
    monkeypatch.setattr(native,'runtime_identity',lambda:{'fixture':'selected-pins'})
    pins = {}
    for name in plan.runtime_files:
        path = tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(name.encode())
        pins[name] = campaign.digest(path.read_bytes())
    plan = replace(plan,runtime_files=pins)
    for policy,name in plan.checkpoints.items():
        path = tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(policy.encode())
        monkeypatch.setitem(mapping.CHECKPOINTS,policy,campaign.digest(path.read_bytes()))
    if change == 'runtime': (tmp_path/next(iter(pins))).write_bytes(b'changed')
    if change == 'checkpoint': (tmp_path/plan.checkpoints['original']).write_bytes(b'changed')
    if change is None:
        result = campaign.preflight(plan)
        assert result['source'] == plan.source and result['runtime_files'] == pins
        assert result['complete_runtime_equivalence_verified'] is False
    else:
        with pytest.raises(ValueError): campaign.preflight(plan)


def test_gpu_lease_is_exclusive_and_never_unlinked(tmp_path):
    path = tmp_path/'lease'
    with campaign.gpu_lease(path) as fd:
        inode = os.fstat(fd).st_ino
        with pytest.raises(BlockingIOError):
            with campaign.gpu_lease(path): pytest.fail('concurrent lease')
    with campaign.gpu_lease(path) as fd: assert os.fstat(fd).st_ino == inode
    link = tmp_path/'link'; link.symlink_to(path)
    with pytest.raises(ValueError):
        with campaign.gpu_lease(link): pass


@pytest.mark.parametrize('mode',['clean','foreign','hot','service','invalid'])
def test_live_gpu_probe_never_mutates_services_or_other_workloads(monkeypatch,mode):
    calls = []
    def read(args,**kwargs):
        calls.append(args); assert kwargs['timeout'] == 2
        if args[0] == 'systemctl': return 'active' if mode == 'service' else 'inactive'
        if '--query-compute-apps=pid' in args: return '123\n999' if mode == 'foreign' else '123'
        return 'N/A' if mode == 'invalid' else '80' if mode == 'hot' else '45'
    monkeypatch.setattr(campaign.subprocess,'check_output',read)
    if mode == 'clean': assert campaign.live_gpu(123)['compute_pids'] == [123]
    else:
        with pytest.raises(ValueError): campaign.live_gpu(123)
    assert all(args[0] == 'nvidia-smi' or args[:2] == ('systemctl','show') for args in calls)


@pytest.fixture
def cell_output(session,tmp_path,monkeypatch):
    cell = mapping.schedule()[0]; session[0].terminal_at = 2
    cfg,agent = core.prepare_config(cell)
    @contextmanager
    def fixture_factory(*a,**k):
        wrapped,actor,stream = session
        yield dict(wrapped=wrapped,actor=actor,stream=stream,
            loading=dict(checkpoint_sha256=mapping.CHECKPOINTS[cell.policy],saved_iteration=7998,common_step=192000,
                         strict_actor_restore=True,optimizer_restored=False,critic_restored=False,policy_acceptance=False,
                         actor_state_sha256=core.actor_digest(actor)),
            runtime=dict(selected_pins=native.runtime_identity(),complete_runtime_equivalence_verified=False,
                         reset_evidence=synthetic_report(wrapped.unwrapped,cell,device='cuda:0')),
            config_yaml={k:yaml.dump(asdict(v),sort_keys=False) for k,v in (('env',cfg),('agent',agent))})
    # Deliberately label synthetic evidence as the factory for adversarial reader
    # tests. The reader still cannot independently prove actual native execution.
    monkeypatch.setattr(native,'native_session',fixture_factory)
    path = tmp_path/'cell'
    native.retain_cell(cell,'fixture',path,device='cuda:0',session_factory=fixture_factory,clock=lambda:1.)
    return path,cell


def rehash(path):
    manifest = json.loads((path/'manifest.json').read_text())
    for name in manifest['files']:
        raw = (path/name).read_bytes()
        manifest['files'][name] = dict(sha256=campaign.digest(raw),bytes=len(raw))
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_reconciled_cell_rescores_raw_and_does_not_admit_native_execution(cell_output):
    path,cell = cell_output
    raw,report = campaign.verify_cell(path,cell)
    assert raw.velocity.shape == (3,8,4)
    assert report['score']['classification'] == 'safety-or-coverage-stop'
    assert report['evidence_reconciled'] and not report['policy_acceptance']
    assert report['native_execution_independently_verified'] is False


@pytest.mark.parametrize('change',['missing','hash','identity','score','journal','config','native','runtime','reset','count','extra','failed'])
def test_cell_tampering_fails_even_when_attacker_rehashes_metadata(cell_output,change):
    path,cell = cell_output
    if change == 'missing': (path/'manifest.json').unlink()
    elif change == 'extra': (path/'unexpected').write_text('extra')
    else:
        name = {'hash':'score.json','identity':'launch.json','score':'score.json','journal':'frames.jsonl',
                'config':'env.yaml','native':'decision.json','runtime':'runtime.json','count':'capture.json',
                'failed':'manifest.json','reset':'runtime.json'}[change]
        if change == 'config': (path/name).write_text('changed: true')
        elif change == 'journal':
            rows = (path/name).read_text().splitlines(); row = json.loads(rows[0]); row['step'] = 1
            rows[0] = json.dumps(row); (path/name).write_text('\n'.join(rows)+'\n')
        else:
            value = json.loads((path/name).read_text())
            if change in ('hash','score'): value['classification'] = 'descriptive-cell-within-checks'
            elif change == 'identity': value['cell']['seed'] = 509
            elif change == 'native': value['native_factory_used'] = False
            elif change == 'runtime': value['complete_runtime_equivalence_verified'] = True
            elif change == 'reset': value['reset_evidence']['common_step_before_action'] = 0
            elif change == 'count': value['step_calls_completed'] = 2
            elif change == 'failed': value['status'] = 'runtime-failure-stop'
            (path/name).write_text(json.dumps(value))
        if change not in ('hash','failed'): rehash(path)
    with pytest.raises((ValueError,FileNotFoundError)): campaign.verify_cell(path,cell)


@pytest.fixture
def harness(plan,tmp_path,monkeypatch):
    output = tmp_path/'campaign'; monkeypatch.setattr(campaign,'OUTPUT',output)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    events = []; state = {'clock':0.,'wall':100.,'stop_at':2,'performance_miss':True}
    @contextmanager
    def lease():
        events.append('lock')
        try: yield 17
        finally: events.append('unlock')
    def identity(plan): events.append('identity'); return {'fixture':'identity'}
    def idle(): events.append('idle'); return {'fixture':'idle'}
    def child(command,log,**kwargs):
        index = int(command[command.index('--cell')+1]); events.append(('child',index))
        assert kwargs['lock_fd'] == 17 and kwargs['timeout'] == 120
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0'
        kwargs['guard']()
        return {'returncode':0,'fixture':True}
    def verify(path,cell):
        index = int(path.name.split('-')[-1]); assert mapping.schedule()[index] == cell
        events.append(('verify',index))
        raw = trace(cell.speed_mps,3 if index == state['stop_at'] else 400)
        if index == state['stop_at']: raw.dones[-1,0] = True
        if state['performance_miss'] and index == 0: raw.velocity[:,:,:2] = 0.
        return raw,{'fixture':True,'manifest_sha256':'a'*64}
    kwargs = dict(identity=identity,idle=idle,lease=lease,child=child,verify=verify,
                  clock=lambda:state['clock'],utc=lambda:state['wall'])
    return output,events,state,kwargs


def test_fixed_order_continues_performance_miss_but_stops_safety(plan,harness):
    output,events,state,kwargs = harness
    result = campaign.run_campaign(plan,**kwargs)
    assert [e for e in events if isinstance(e,tuple) and e[0]=='child'] == [('child',0),('child',1),('child',2)]
    assert result['cells'][0]['classification'] == 'descriptive-performance-miss'
    assert result['decision'] == 'safety-or-coverage-stop' and len(result['unexecuted']) == 15
    assert events[0] == 'lock' and events[-1] == 'unlock'
    assert events.count('idle') == 6
    assert len(list(output.glob('receipt-*.json'))) == 3
    assert not (output/'campaign-failure.json').exists()


def test_full_fixed_map_still_grants_no_policy_admission(plan,harness):
    output,events,state,kwargs = harness; state.update(stop_at=None,performance_miss=False)
    result = campaign.run_campaign(plan,**kwargs)
    assert result['decision'] == 'complete-descriptive-map' and len(result['cells']) == 18
    assert result['policy_acceptance'] is result['training_admitted'] is False
    assert len(list(output.glob('receipt-*.json'))) == 18


@pytest.mark.parametrize('kind',['busy','child','evidence','identity','reserve','clock'])
def test_failure_closes_without_retry_or_next_cell(plan,harness,kind):
    output,events,state,kwargs = harness
    def fail(*a,**k): raise ValueError('injected failure')
    if kind == 'busy': kwargs['idle'] = fail
    elif kind in ('child','evidence'): kwargs['child' if kind == 'child' else 'verify'] = fail
    elif kind == 'identity':
        sequence = iter([{'fixture':'identity'},{'changed':True}])
        kwargs['identity'] = lambda plan:next(sequence)
    else:
        original = kwargs['child']
        def changed(*a,**k):
            result = original(*a,**k); state['clock'] = 2250. if kind == 'reserve' else -1.
            return result
        kwargs['child'] = changed
    with pytest.raises(ValueError): campaign.run_campaign(plan,**kwargs)
    assert events[-1] == 'unlock'
    assert sum(isinstance(e,tuple) and e[0]=='child' for e in events) <= 1
    failure = json.loads((output/'campaign-failure.json').read_text())
    assert failure['status'] == 'runtime-failure-stop' and not failure['policy_acceptance']
    assert failure['verified_cell_count'] == 0 and not (output/'campaign-result.json').exists()


def test_exclusive_campaign_output_preserves_existing_work(plan,harness):
    output,events,_,kwargs = harness
    output.mkdir(); (output/'keep').write_text('owned by previous run')
    with pytest.raises(FileExistsError): campaign.run_campaign(plan,**kwargs)
    assert (output/'keep').read_text() == 'owned by previous run'
    assert {p.name for p in output.iterdir()} == {'keep'} and events[-1] == 'unlock'


@pytest.mark.parametrize('wall',[0.,4900.,5001.])
def test_window_gate_precedes_lock_or_output(plan,harness,wall):
    output,events,state,kwargs = harness; state['wall'] = wall
    with pytest.raises(ValueError): campaign.run_campaign(plan,**kwargs)
    assert not events and not output.exists()


def test_real_cpu_child_success_and_timeout_do_not_touch_unrelated_process(tmp_path):
    env = CPU_ENV
    unrelated = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],env=env)
    try:
        with campaign.gpu_lease(tmp_path/'lease') as fd:
            result = campaign.supervised_process([sys.executable,'-c','print("CPU fixture")'],tmp_path/'ok.log',
                     cwd=tmp_path,env=env,lock_fd=fd,timeout=5,monitor=lambda pid:{'fixture':pid})
            assert result['returncode'] == 0 and (tmp_path/'ok.log').read_text().strip() == 'CPU fixture'
            started = time.monotonic()
            with pytest.raises(ValueError,match='deadline'):
                campaign.supervised_process([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'timeout.log',
                     cwd=tmp_path,env=env,lock_fd=fd,timeout=.2,monitor=lambda pid:{'fixture':pid})
            assert time.monotonic()-started < 5 and unrelated.poll() is None
    finally:
        unrelated.terminate(); unrelated.wait(timeout=5)


def test_real_cpu_child_nonzero_exit_is_not_a_completed_cell(tmp_path):
    with campaign.gpu_lease(tmp_path/'lease') as fd:
        with pytest.raises(ValueError,match='unsuccessfully'):
            campaign.supervised_process([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'fail.log',
                cwd=tmp_path,env=CPU_ENV,lock_fd=fd,timeout=5,monitor=lambda pid:{})


def test_surviving_cpu_descendant_closes_the_cell_instead_of_advancing(tmp_path):
    code = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])'
    with campaign.gpu_lease(tmp_path/'lease') as fd:
        with pytest.raises(ValueError,match='surviving process-group'):
            campaign.supervised_process([sys.executable,'-c',code],tmp_path/'descendant.log',
                cwd=tmp_path,env=CPU_ENV,lock_fd=fd,timeout=5,monitor=lambda pid:{})


def test_runtime_monitor_conflict_stops_only_supervised_cpu_group(tmp_path):
    def conflict(pid): raise ValueError('synthetic foreign GPU owner')
    with campaign.gpu_lease(tmp_path/'lease') as fd:
        with pytest.raises(ValueError,match='foreign GPU') as error:
            campaign.supervised_process([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'conflict.log',
                cwd=tmp_path,env=CPU_ENV,lock_fd=fd,timeout=5,monitor=conflict)
    assert 'child_pid' in error.value.__notes__[0]


def test_watchdog_start_failure_does_not_orphan_an_already_started_cpu_child(tmp_path,monkeypatch):
    created = []; original = subprocess.Popen
    def recorded(*a,**k):
        process = original(*a,**k); created.append(process); return process
    def broken_start(self): raise RuntimeError('synthetic watchdog start failure')
    monkeypatch.setattr(campaign.subprocess,'Popen',recorded)
    monkeypatch.setattr(campaign.threading.Thread,'start',broken_start)
    with campaign.gpu_lease(tmp_path/'lease') as fd:
        with pytest.raises(RuntimeError,match='watchdog start failure'):
            campaign.supervised_process([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'startup.log',
                cwd=tmp_path,env=CPU_ENV,lock_fd=fd,timeout=5,monitor=lambda pid:{})
    assert len(created) == 1 and created[0].poll() is not None


@pytest.mark.parametrize('persistent',[False,True])
def test_cleanup_reaps_racing_zombie_but_does_not_hide_real_permission_error(monkeypatch,persistent):
    from types import SimpleNamespace
    events = []
    proc = SimpleNamespace(pid=12345,wait=lambda **k:events.append('reap'))
    def signal_group(pid,sig):
        assert pid == 12345
        events.append('signal')
        if len(events) == 1 or persistent: raise PermissionError('fixture')
        raise ProcessLookupError('reaped fixture')
    monkeypatch.setattr(campaign.os,'killpg',signal_group)
    if persistent:
        with pytest.raises(PermissionError): campaign._signal_owned_group(proc,campaign.signal.SIGTERM)
    else: campaign._signal_owned_group(proc,campaign.signal.SIGTERM)
    assert events == ['signal','reap','signal']


def test_watchdog_kills_cpu_child_while_telemetry_callback_is_blocked(tmp_path):
    observed = []
    def slow_monitor(pid):
        time.sleep(1.)
        # The watchdog should have fired at 1.2 s, not waited for this callback.
        status = subprocess.check_output(['ps','-o','stat=','-p',str(pid)],text=True).strip()
        observed.append(status)
        return {'fixture':True}
    with campaign.gpu_lease(tmp_path/'lease') as fd:
        with pytest.raises(ValueError,match='deadline') as error:
            campaign.supervised_process([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'slow.log',
                cwd=tmp_path,env=CPU_ENV,lock_fd=fd,timeout=1.2,monitor=slow_monitor)
    assert observed and observed[0].startswith('Z')  # Dead child awaits supervisor reap.
    assert 'child_pid' in error.value.__notes__[0]


@pytest.mark.parametrize('bad',['none','hash','lease','window','source'])
def test_worker_boundary_is_read_only_until_all_launch_guards_pass(plan,tmp_path,monkeypatch,bad):
    output = tmp_path/'campaign'; output.mkdir()
    lease = tmp_path/'lease'
    monkeypatch.setattr(campaign,'OUTPUT',output); monkeypatch.setattr(campaign,'LOCK',lease)
    document = dict(protocol=campaign.PROTOCOL,plan=asdict(plan),identity={'fixture':True})
    campaign.write_json(output/'plan.json',document)
    sha = campaign.digest((output/'plan.json').read_bytes())
    monkeypatch.setattr(campaign.time,'time',lambda:4999. if bad == 'window' else 100.)
    monkeypatch.setattr(campaign,'preflight',lambda plan:{'changed':True} if bad == 'source' else {'fixture':True})
    events = []
    monkeypatch.setattr(campaign,'wait_idle',lambda:events.append('idle'))
    monkeypatch.setattr(native,'retain_cell',lambda *a,**k:events.append(('capture',a,k)))
    with campaign.gpu_lease(lease) as fd:
        # Closing a separate unrelated descriptor never weakens the held lease.
        other = os.open(tmp_path/'other',os.O_RDWR|os.O_CREAT,0o600)
        try:
            monkeypatch.setattr(sys,'argv',['worker','--cell','0','--plan',str(output/'plan.json'),
                '--plan-sha256','0'*64 if bad == 'hash' else sha,'--lock-fd',str(other if bad == 'lease' else fd)])
            if bad == 'none':
                campaign.worker_main()
                assert events[0] == 'idle' and events[1][0] == 'capture'
                assert events[1][2] == {'device':'cuda:0'}
                assert events[1][1][0] == mapping.schedule()[0]
            else:
                with pytest.raises(ValueError): campaign.worker_main()
                assert not events
        finally: os.close(other)
