"""Sequential frozen-map supervisor; no scheduler or service installation.

The public API needs a separately reviewed launch plan and a service-level hard
timeout. The module CLI is only the supervised child, not a campaign launcher.
All output remains descriptive research, never training or physical admission.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

import torch
import yaml

from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck import foundation_command_session as native
from mjlab_microduck import foundation_command_capture as core
from mjlab_microduck import foundation_reset_evidence as reset_evidence
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.gpu_idle_gate import SERVICES, wait_idle

PROTOCOL = 'foundation-command-map-supervisor-v1'
ROOT = Path('/home/converge/work/microduck_rl-athletics-obstacle-curriculum')
OUTPUT = ROOT/'artifacts/evaluations/foundation-command-map-v1'
LOCK = Path('/home/converge/.local/state/microduck-gpu0.lock')
BRANCH = 'feat/athletics-obstacle-curriculum'
SERVICE_SECONDS, CELL_SECONDS, CLOSEOUT_SECONDS = 2400, 120, 180
RUNTIME_SCOPE = 'reviewed-file-set-not-complete-runtime-equivalence'
REQUIRED_RUNTIME_SUFFIXES = (
    'mjlab/envs/manager_based_rl_env.py', 'mjlab/managers/metrics_manager.py',
    'mjlab/managers/observation_manager.py', 'mjlab/entity/data.py',
    'rsl_rl/models/mlp_model.py',
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hex_id(value,n):
    require(type(value) is str and len(value) == n and all(c in '0123456789abcdef' for c in value),
            'exact hexadecimal identity')


def parse(raw):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result,'duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(raw,object_pairs_hook=unique)
    canonical(result)  # Also rejects NaN/Infinity nested in otherwise valid JSON.
    return result


def file_bytes(path, *, limit=64*1024*1024, allow_empty=False):
    path = native._plain_path(path)
    require(path.is_file() and (0 if allow_empty else 1) <= path.stat().st_size <= limit,'bounded regular evidence file')
    with path.open('rb') as source: raw = source.read(limit+1)
    require((0 if allow_empty else 1) <= len(raw) <= limit,'bounded file read')
    return raw


def write_json(path,value):
    path = native._plain_path(path)
    with path.open('xb') as target:
        target.write((canonical(value)+'\n').encode()); target.flush(); os.fsync(target.fileno())
    native._fsync_dir(path.parent)


@dataclass(frozen=True)
class LaunchPlan:
    source: str
    machine_id: str
    not_before_unix: float
    deadline_unix: float
    checkpoints: dict[str,str]  # Paths relative to the one authorized worktree.
    runtime_files: dict[str,str]  # Reviewed relative .venv file paths -> SHA256.
    runtime_scope: str = RUNTIME_SCOPE

    def validate(self):
        hex_id(self.source,40); hex_id(self.machine_id,32)
        require(all(type(v) in (int,float) and math.isfinite(v) and v > 0
                    for v in (self.not_before_unix,self.deadline_unix)),'finite explicit UTC window')
        require(self.deadline_unix-self.not_before_unix >= SERVICE_SECONDS,'full service window')
        require(self.runtime_scope == RUNTIME_SCOPE,'honest selected runtime scope')
        require(set(self.checkpoints) == set(mapping.CHECKPOINTS),'both fixed checkpoint paths')
        require(type(self.runtime_files) is dict and bool(self.runtime_files),'reviewed runtime pins')
        for name in (*self.checkpoints.values(),*self.runtime_files):
            require(type(name) is str and name and not Path(name).is_absolute()
                    and all(p not in ('.','..') for p in name.split('/')),'contained relative path')
        for name,sha in self.runtime_files.items():
            require(name.startswith('.venv/'),'environment-local runtime file')
            hex_id(sha,64)
        require(all(any(name.endswith('/'+suffix) for name in self.runtime_files)
                    for suffix in REQUIRED_RUNTIME_SUFFIXES),'required runtime pin coverage')


def preflight(plan):
    """Read-only identity check, no GPU allocation or workload changes."""
    plan.validate()
    require(sys.platform == 'linux' and Path.cwd().resolve() == ROOT,'exact Linux training worktree')
    require(Path('/etc/machine-id').read_text().strip() == plan.machine_id,'exact GPU host identity')
    def git(*args):
        return subprocess.check_output(['git',*args],text=True,timeout=5).strip()
    require(git('branch','--show-current') == BRANCH and git('rev-parse','HEAD') == plan.source
            and not git('status','--porcelain'),'clean exact feature source')
    # Empty package __init__.py files are real runtime inputs: pin them too so
    # replacing an empty initializer with executable code invalidates the plan.
    pins = {name:digest(file_bytes(ROOT/name,allow_empty=True)) for name in plan.runtime_files}
    require(pins == plan.runtime_files,'reviewed runtime file hashes')
    for policy,path in plan.checkpoints.items():
        require(digest(file_bytes(ROOT/path)) == mapping.CHECKPOINTS[policy],'frozen checkpoint pin')
    return dict(source=plan.source,machine_id=plan.machine_id,runtime_files=pins,
                selected_runtime=native.runtime_identity(),complete_runtime_equivalence_verified=False)


@contextmanager
def gpu_lease(path=None):
    """Advisory lock shared by cooperating Duck jobs; never unlink its inode."""
    path = native._plain_path(LOCK if path is None else path)
    require(path.parent.is_dir(),'existing explicit lock parent')
    fd = os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_uid == os.getuid(),'owned regular GPU lock')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield fd
    finally:
        # Closing rather than LOCK_UN preserves the inherited lease if a child
        # survives a supervisor failure. The parent must reap it before returning.
        os.close(fd)


def live_gpu(pid):
    """Abort our child if another compute owner or protected service appears."""
    def read(*args):
        return subprocess.check_output(args,text=True,timeout=2).strip()
    states = {s:read('systemctl','show',s,'-p','ActiveState','--value') for s in SERVICES}
    raw = read('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits')
    pids = [int(p) for p in raw.splitlines()]
    temperature = int(read('nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'))
    sample = dict(services=states,compute_pids=pids,temperature_c=temperature)
    require(all(s == 'inactive' for s in states.values()) and set(pids) <= {pid}
            and 0 <= temperature < 80,'GPU ownership/service/temperature conflict: '+canonical(sample))
    return sample


def _signal_owned_group(proc,sig):
    try:
        os.killpg(proc.pid,sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin can expose a just-killed, unreaped leader as EPERM. A poll
        # before killpg still races with its death. Reap, then retry the group
        # signal so surviving descendants are not silently ignored. Persistent
        # permission errors or a still-live leader remain real failures.
        proc.wait(timeout=2)
        try: os.killpg(proc.pid,sig)
        except ProcessLookupError: pass


def _kill_owned_group(proc):
    """Only the group created for this child; never other GPU PIDs or services."""
    # Reap an already dead leader first. On macOS, signaling an unreaped killed
    # process group can return EPERM rather than ESRCH; do not mask the cause.
    proc.poll()
    _signal_owned_group(proc,signal.SIGTERM)
    try: proc.wait(timeout=2)
    except subprocess.TimeoutExpired: pass
    # Also clear surviving descendants when the group leader has already exited.
    _signal_owned_group(proc,signal.SIGKILL)
    proc.wait(timeout=2)


def child_environment():
    # Do not leak connector/API credentials into a simulator, its crash report,
    # or pytest's argument display. No PYTHONPATH/preload/library-path injection.
    allowed = ('HOME','USER','LOGNAME','XDG_CACHE_HOME','XDG_RUNTIME_DIR')
    return {**{key:os.environ[key] for key in allowed if key in os.environ},
            'PATH':str(ROOT/'.venv/bin')+':/usr/local/bin:/usr/bin:/bin',
            'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}


def supervised_process(command,log,*,cwd,env,lock_fd,timeout=120,monitor=live_gpu,guard=lambda:None):
    require(type(timeout) in (int,float) and 0 < timeout <= CELL_SECONDS,'bounded child timeout')
    return _timed_process(command,log,cwd=cwd,env=env,lock_fd=lock_fd,timeout=timeout,monitor=monitor,guard=guard)


def supervised_stance_smoke(command,log,*,cwd,env,lock_fd,monitor=live_gpu,guard=lambda:None):
    """Separate fixed 900-second stance smoke; frozen-map bounds stay unchanged."""
    return _timed_process(command,log,cwd=cwd,env=env,lock_fd=lock_fd,timeout=900,monitor=monitor,guard=guard)


def _timed_process(command,log,*,cwd,env,lock_fd,timeout,monitor,guard):
    started = time.monotonic(); samples = []
    finished,expired = threading.Event(),threading.Event()
    with native._plain_path(log).open('xb') as output:
        proc = subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=output,
                                stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
        def watchdog():
            if not finished.wait(max(0,timeout-(time.monotonic()-started))):
                expired.set()
                # A separate watchdog can terminate a stuck native child even
                # while the supervisor's telemetry subprocess is blocked.
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
        timer = None
        try:
            timer = threading.Thread(target=watchdog,daemon=True); timer.start()
            while True:
                guard()
                remaining = timeout-(time.monotonic()-started)
                require(remaining > 0,'child hard deadline exceeded')
                try:
                    code = proc.wait(timeout=min(1,remaining))
                    require(not expired.is_set() and time.monotonic()-started <= timeout,'child hard deadline exceeded')
                    require(code == 0,'child exited unsuccessfully: '+str(code))
                    try: os.killpg(proc.pid,0)
                    except ProcessLookupError: pass
                    else: raise ValueError('child left surviving process-group members')
                    return dict(pid=proc.pid,returncode=code,elapsed_s=time.monotonic()-started,samples=samples)
                except subprocess.TimeoutExpired:
                    samples.append(monitor(proc.pid))
        except Exception as exc:
            exc.add_note(canonical(dict(child_pid=proc.pid,elapsed_s=time.monotonic()-started,samples=samples)))
            raise
        finally:
            finished.set()
            if timer is not None and timer.ident is not None: timer.join(timeout=2)
            _kill_owned_group(proc)
            output.flush(); os.fsync(output.fileno())


def verify_cell(path,cell):
    """Rehash exact deserialized bytes, reconcile journal and rescore raw tensors."""
    path = native._plain_path(path)
    manifest_raw = file_bytes(path/'manifest.json')
    manifest = parse(manifest_raw)
    require(manifest['protocol'] == native.PROTOCOL and manifest['status'] == 'captured-diagnostic-only'
            and manifest['policy_acceptance'] is False and manifest['physical_motion_authorized'] is False,
            'closed diagnostic cell manifest')
    names = {'launch.json','loading.json','runtime.json','env.yaml','agent.yaml','frames.jsonl',
             'trace.pt','capture.json','score.json','decision.json'}
    require(set(manifest['files']) == names and {p.name for p in path.iterdir()} == names|{'manifest.json'},
            'exact successful cell files')
    raw = {name:file_bytes(path/name) for name in names}
    require(all(manifest['files'][name] == dict(sha256=digest(value),bytes=len(value))
                for name,value in raw.items()),'cell evidence hash/length mismatch')
    launch,loading,runtime,capture,decision = [parse(raw[n+'.json']) for n in
                                             ('launch','loading','runtime','capture','decision')]
    require(launch['protocol'] == native.PROTOCOL and launch['cell'] == cell.identity()
            and launch['device'] == 'cuda:0' and launch['policy_acceptance'] is False
            and type(launch['budget_seconds']) in (float,int) and 0 < launch['budget_seconds'] <= 120,'cell launch identity')
    require(loading['checkpoint_sha256'] == mapping.CHECKPOINTS[cell.policy]
            and loading['saved_iteration'] == native.ITERATIONS[cell.policy]
            and loading['common_step'] == native.COMMON_STEPS[cell.policy]
            and loading['strict_actor_restore'] is True and loading['optimizer_restored'] is False
            and loading['critic_restored'] is False and loading['policy_acceptance'] is False,'strict frozen loading evidence')
    hex_id(loading['actor_state_sha256'],64)
    require(runtime['complete_runtime_equivalence_verified'] is False
            and runtime['selected_pins'] == native.runtime_identity(),'selected runtime evidence')
    reset_evidence.validate(runtime['reset_evidence'],cell,device='cuda:0',common_step=loading['common_step'])
    cfg,agent = core.prepare_config(cell)
    for name,value in (('env',cfg),('agent',agent)):
        require(raw[name+'.yaml'].decode() == yaml.dump(asdict(value),sort_keys=False),'parameterized configuration evidence')
    value = torch.load(io.BytesIO(raw['trace.pt']),map_location='cpu',weights_only=True)
    require(type(value) is dict and set(value) == set(mapping.Trace.__dataclass_fields__)|{'cached_commands'},'exact trace fields')
    trace = mapping.Trace(**{k:v for k,v in value.items() if k != 'cached_commands'})
    score = mapping.score(cell,trace)
    cached = value['cached_commands']; n = score['sample_steps']
    require(isinstance(cached,torch.Tensor) and cached.shape == (n,8,3)
            and cached.dtype in (torch.float32,torch.float64) and bool(torch.isfinite(cached).all()),'cached command coverage')
    rows = raw['frames.jsonl'].splitlines()
    require(len(rows) == n,'journal complete-step count')
    keys = set(mapping.Trace.__dataclass_fields__)-{'joint_names'}|{'cached'}
    for step,line in enumerate(rows):
        row = parse(line)
        require(type(row['step']) is int and row['step'] == step and set(row['tensors']) == keys,'ordered exact journal fields')
        for key,sample in row['tensors'].items():
            expected = cached[step] if key == 'cached' else getattr(trace,key)[step]
            require(canonical(sample) == canonical(dict(dtype=str(expected.dtype),shape=list(expected.shape),value=expected.tolist())),
                    'journal/trace mismatch')
    require(capture['protocol'] == core.PROTOCOL and capture['cell'] == cell.identity()
            and type(capture['inference_calls_completed']) is type(capture['step_calls_completed']) is int
            and capture['inference_calls_completed'] == capture['step_calls_completed'] == n
            and capture['actor_state_sha256'] == loading['actor_state_sha256']
            and capture['actor_state_unchanged'] is True,'capture identity and call counts')
    require(all(capture[k] is False for k in ('checkpoint_loaded_verified','runtime_verified','physics_execution_verified',
                                            'policy_acceptance','training_admitted','physical_motion_authorized')),'capture scope')
    require(parse(raw['score.json']) == score and decision == dict(status='captured-diagnostic-only',
            classification=score['classification'],native_factory_used=True,policy_acceptance=False,
            training_admitted=False,physical_motion_authorized=False),'recomputed decision')
    return trace,dict(manifest_sha256=digest(manifest_raw),score=score,evidence_reconciled=True,
                      native_execution_independently_verified=False,policy_acceptance=False)


def run_campaign(plan,*,identity=preflight,idle=wait_idle,lease=gpu_lease,
                 child=supervised_process,verify=verify_cell,utc=time.time,clock=time.monotonic):
    """Run the fixed prefix once. Requires an external 2400s whole-service timeout.

Injection points are CPU-test seams, not a way to establish native execution.
No retry, resume, seed selection, optimizer, service restore or video path exists.
"""
    plan.validate()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '','CPU-only supervisor')
    started = last = clock()
    def remaining(reserve=0):
        nonlocal last
        now,wall = clock(),utc()
        require(all(math.isfinite(t) for t in (started,now,wall)) and now >= last,'monotonic supervisor clock')
        last = now
        require(plan.not_before_unix <= wall < plan.deadline_unix,'explicit authorized window')
        available = min(SERVICE_SECONDS-(now-started),plan.deadline_unix-wall)
        require(available > reserve,'insufficient campaign time/reserve')
        return available
    remaining()
    require(plan.deadline_unix-utc() >= SERVICE_SECONDS,'full predeclared service fits window')
    with lease() as lock_fd:
        before = identity(plan)
        output = native._plain_path(OUTPUT); output.mkdir()
        native._fsync_dir(output.parent)
        traces,receipts = [],[]
        try:
            write_json(output/'plan.json',dict(protocol=PROTOCOL,plan=asdict(plan),identity=before,
                       policy_acceptance=False,physical_motion_authorized=False))
            plan_hash = digest(file_bytes(output/'plan.json'))
            for index,cell in enumerate(mapping.schedule()):
                remaining(CELL_SECONDS+CLOSEOUT_SECONDS+10)
                require(identity(plan) == before,'source/runtime changed between cells')
                pre = idle()
                remaining(CELL_SECONDS+CLOSEOUT_SECONDS)
                command = [str(ROOT/'.venv/bin/python'),'-m',__name__,'--cell',str(index),
                           '--plan',str(output/'plan.json'),'--plan-sha256',plan_hash,'--lock-fd',str(lock_fd)]
                child_result = child(command,output/f'{index:02d}.log',cwd=ROOT,
                    env=child_environment(),
                    lock_fd=lock_fd,timeout=CELL_SECONDS,guard=remaining)
                require(child_result['returncode'] == 0,'successful supervised child exit')
                post = idle()
                require(identity(plan) == before,'source/runtime changed during cell')
                remaining(CLOSEOUT_SECONDS)
                trace,record = verify(output/f'cell-{index:02d}',cell)
                traces.append((cell,trace))
                summary = mapping.summarize_prefix(traces)
                receipt = dict(index=index,cell=cell.identity(),preflight=pre,postflight=post,child=child_result,**record)
                write_json(output/f'receipt-{index:02d}.json',receipt); receipts.append(receipt)
                if summary['decision'] == 'safety-or-coverage-stop': break
            remaining()
            require(identity(plan) == before,'source/runtime changed during closeout')
            write_json(output/'campaign-result.json',dict(protocol=PROTOCOL,summary=summary,
                       verified_cell_count=len(receipts),policy_acceptance=False,training_admitted=False,
                       physical_motion_authorized=False))
            return summary
        except Exception as exc:
            write_json(output/'campaign-failure.json',dict(protocol=PROTOCOL,status='runtime-failure-stop',
                       error_type=type(exc).__name__,error=str(exc),verified_cell_count=len(receipts),
                       error_notes=getattr(exc,'__notes__',[]),
                       unexecuted_or_unverified=[c.identity() for c in mapping.schedule()[len(receipts):]],
                       policy_acceptance=False,training_admitted=False,physical_motion_authorized=False))
            raise


def worker_main():
    """Internal child requires the inherited lease and byte-pinned parent plan."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',type=int,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--lock-fd',type=int,required=True)
    args = parser.parse_args()
    require(0 <= args.cell < len(mapping.schedule()),'fixed child index')
    require(args.plan == OUTPUT/'plan.json','exact retained launch plan path')
    raw = file_bytes(args.plan); require(digest(raw) == args.plan_sha256,'retained launch plan bytes')
    document = parse(raw); require(document['protocol'] == PROTOCOL,'parent protocol')
    plan = LaunchPlan(**document['plan']); plan.validate()
    inherited,lock = os.fstat(args.lock_fd),native._plain_path(LOCK).stat()
    require((inherited.st_dev,inherited.st_ino) == (lock.st_dev,lock.st_ino),'inherited common GPU lease')
    # A second open must fail to take the parent's existing lease.
    try:
        with gpu_lease(): raise ValueError('parent GPU lease was not held')
    except BlockingIOError: pass
    require(plan.not_before_unix <= time.time() and time.time()+CELL_SECONDS < plan.deadline_unix,'child fits window')
    require(preflight(plan) == document['identity'],'child source/runtime identity')
    wait_idle()
    require(time.time()+CELL_SECONDS < plan.deadline_unix,'child setup consumed its window')
    cell = mapping.schedule()[args.cell]
    native.retain_cell(cell,ROOT/plan.checkpoints[cell.policy],OUTPUT/f'cell-{args.cell:02d}',device='cuda:0')


if __name__ == '__main__': worker_main()
