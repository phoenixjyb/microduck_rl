"""Leased same-input forward-only diagnostic. No integration or training."""
import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
from itertools import combinations
import os
import random
import time

import numpy as np
import torch
import warp as wp

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_throughput_probe as throughput
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_forward_graph import binding
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime, ActuatorCmd
from mjlab_microduck.first_attempt_smoke import canonical, require

MODULE = 'mjlab_microduck.stance_forward_probe'
PROTOCOL = 'football-b1n-same-input-forward-v2'
ORDER = (False, True, True, False, True, False, False, True)
WORLDS = (64, 512)
SNAPSHOT_LIMIT = 256*1024*1024
TOTAL_LIMIT = 1024*1024*1024
DEADLINE = datetime(2026, 9, 9, 6, 0, tzinfo=timezone.utc).timestamp()
KINEMATICS = ('qpos', 'qvel', 'time', 'qacc_warmstart')
DYNAMICS = ('qacc', 'qacc_smooth', 'qfrc_bias', 'qfrc_actuator', 'qfrc_constraint')
CONSTRAINTS = ('type', 'id', 'J', 'D', 'aref', 'force', 'state')


def check_window():
    host.check_window()
    require(time.time()+180+600 < DEADLINE, 'insufficient window before September 9 14:00 Shanghai')


def arrays(value, path=''):
    if isinstance(value, wp.array): return {path: value}
    result = {}
    if is_dataclass(value): children = [(f.name, getattr(value, f.name)) for f in fields(value)]
    elif isinstance(value, (tuple, list)): children = [(str(i), v) for i, v in enumerate(value)]
    elif isinstance(value, dict): children = sorted(value.items())
    else: children = []  # binding() separately validates all static types.
    for key, child in children: result.update(arrays(child, path+'/'+key))
    return result


class InputSnapshot:
    """All owned model/data arrays, including inactive scratch, restored in place."""
    def __init__(self, env):
        self.env = env; env._sync()
        self.signature = binding((env.model, env.data))
        self.dest = arrays(dict(model=env.model, data=env.data))
        require(self.dest and all(a.is_contiguous for a in self.dest.values()), 'contiguous complete input arrays')
        self.nbytes = sum(a.size*wp.types.type_size_in_bytes(a.dtype) for a in self.dest.values())
        require(self.nbytes <= SNAPSHOT_LIMIT, 'bounded complete input snapshot')
        self.backup = {p: wp.clone(a, device='cpu') for p, a in self.dest.items()}
        env._sync(); wp.synchronize_device('cpu')
        self.payload = self.capture(self.backup)
        self.identity = throughput.tree_hash(self.payload)

    def capture(self, collection):
        content = {}
        for path, array in collection.items():
            owner = self.dest[path]
            require(array.shape == owner.shape and array.dtype == owner.dtype, 'backup logical array layout')
            raw = array.numpy().tobytes()
            # CPU clones pack singleton broadcast axes. Their storage strides
            # are not the simulator's strides; bind the original allocation's
            # layout while hashing the clone's exact logical content bytes.
            content[path] = dict(dtype=str(owner.dtype), shape=list(owner.shape), strides=list(owner.strides),
                raw=torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()))
        return dict(allocation_binding_repr=repr(self.signature), arrays=content)

    def restore(self):
        self.env._sync()
        require(binding((self.env.model, self.env.data)) == self.signature, 'input allocation or static configuration drift')
        for path, dest in self.dest.items(): wp.copy(dest, self.backup[path])
        self.env._sync()
        actual = throughput.tree_hash(self.capture(self.dest))
        require(actual == self.identity, 'invalid-input-control')
        return actual


def output(env):
    env._sync()
    nefc = env._view('nefc').detach().cpu().clone()
    require(((nefc >= 0) & (nefc <= env.data.njmax)).all(), 'bounded active constraint rows')
    efc = {k: wp.to_torch(getattr(env.data.efc, k)).detach().cpu().clone() for k in CONSTRAINTS}
    result = dict(kinematics={k: env._view(k).detach().cpu().clone() for k in KINEMATICS},
        dynamics={k: env._view(k).detach().cpu().clone() for k in DYNAMICS},
        solver=dict(nefc=nefc, nf=env._view('nf').detach().cpu().clone(),
                    solver_niter=env._view('solver_niter').detach().cpu().clone()),
        contacts=trace.owned(env.contacts),
        constraints=[{k: v[i, :int(n)].clone() for k, v in efc.items()} for i, n in enumerate(nefc)])
    validate_output(result, env.n)
    return result


def validate_output(value, n):
    require(set(value) == {'kinematics', 'dynamics', 'solver', 'contacts', 'constraints'}, 'complete forward output')
    for group, names in (('kinematics', KINEMATICS), ('dynamics', DYNAMICS)):
        require(set(value[group]) == set(names), 'exact '+group+' fields')
        for k, v in value[group].items():
            shape = (n,) if k == 'time' else (n, 21 if k == 'qpos' else 20)
            trace.tensor(v, shape, torch.float32, k)
    require(set(value['solver']) == {'nefc', 'nf', 'solver_niter'}, 'solver counters')
    for k, v in value['solver'].items():
        require(v.dtype == torch.int32 and v.shape == (n,) and v.device.type == 'cpu'
                and (v >= 0).all(), 'bounded solver counter '+k)
    require((value['solver']['nefc'] <= 512).all() and (value['solver']['solver_niter'] <= 100).all(),
            'solver capacity and iteration ceiling')
    require((value['solver']['nf'] == 14).all() and (value['solver']['nefc'] >= 14).all(),
            'motor friction constraints exercised')
    require(len(value['constraints']) == n, 'one active row table per world')
    for rows, count in zip(value['constraints'], value['solver']['nefc']):
        require(set(rows) == set(CONSTRAINTS), 'complete active row fields')
        for k, v in rows.items():
            trace.tensor(v, (int(count), 20) if k == 'J' else (int(count),),
                         torch.int32 if k in ('type', 'id', 'state') else torch.float32, k)
    from mjlab_microduck.stance_contact_evidence import validate_contacts
    require(set(value['contacts']) == {'worldid', 'geom', 'dist', 'pos', 'frame', 'friction', 'dim', 'efc_address', 'force'},
            'complete ordered contacts')
    validate_contacts(value['contacts'], n, torch.device('cpu'))


def comparison(samples):
    require(len(samples) == 8, 'eight ordered calls')
    hashes = [{k: throughput.tree_hash(v) for k, v in sample.items()} for sample in samples]
    pairs = []
    for i, j in combinations(range(8), 2):
        groups = [k for k in hashes[i] if hashes[i][k] != hashes[j][k]]
        pairs.append(dict(calls=[i, j], kind=('graph' if ORDER[i] else 'eager') if ORDER[i] == ORDER[j] else 'cross',
            exact=not groups, differing_groups=groups,
            dynamics_max_abs={k: float((samples[i]['dynamics'][k].double()-samples[j]['dynamics'][k].double()).abs().max())
                              for k in DYNAMICS}))
    eager = all(p['exact'] for p in pairs if p['kind'] == 'eager')
    decision = ('forward-baseline-nonrepeatable' if not eager else
                'candidate-difference-requires-diagnosis' if any(not p['exact'] for p in pairs) else
                'same-input-forward-repeatable-in-this-sample')
    return dict(decision=decision, pairs=pairs, group_hashes=hashes, training_admitted=False,
                graph_equivalence_established=False, physical_motion_authorized=False)


def batch(n, device, retain):
    random.seed(523); np.random.seed(523); torch.manual_seed(523)
    env = WarpStanceRuntime(n, device=device)
    require(plant.describe(env.native) == plant.reference(), 'actual compiled forward plant')
    # Reset clears BAM friction, so the v1 nominal reset-only design exercised
    # no constraints. Prepare one genuine zero-error BAM command, as the first
    # pre-Euler solve would see, then freeze it for all measured forward calls.
    initial = {k: env._view(k).detach().cpu().clone() for k in KINEMATICS}
    initial_hash = throughput.tree_hash(initial)
    pos = env._view('qpos')[:, env.qids]; vel = env._view('qvel')[:, env.dofs]
    zeros = torch.zeros_like(pos)
    proposal = env.motor.compute(ActuatorCmd(env.delay.peek(), zeros, zeros, pos, vel), env.live.clone())
    require(proposal['accepted'].all() and not proposal['rejected'].any()
            and torch.isfinite(proposal['torque_nm']).all() and not proposal['torque_nm'].any(),
            'one zero-error zero-torque motor preparation')
    env._view('ctrl')[:, env.ctrl_ids] = proposal['torque_nm']
    env._forward()
    require((env._view('nf') == 14).all() and (env._view('nefc') >= 14).all(), 'prepared motor friction rows')
    require(throughput.tree_hash({k: env._view(k) for k in KINEMATICS}) == initial_hash,
            'motor preparation cannot integrate')
    snap = InputSnapshot(env)
    descriptor = retain(f'input-{n}.pt', snap.payload)
    env.enable_forward_graph(); graph = env.forward_graph
    samples = []; receipts = []
    for i, captured in enumerate(ORDER):
        env.forward_graph = graph if captured else None
        input_sha = snap.restore()
        started = time.monotonic(); env._forward(); elapsed = time.monotonic()-started
        value = output(env)
        require(throughput.tree_hash(value['kinematics']) == initial_hash
                and not env.steps.any(), 'forward must not integrate')
        saved = retain(f'output-{n}-{i}.pt', value)
        receipts.append(dict(call=i, graph=captured, input_sha256=input_sha, output=saved, elapsed_s=elapsed))
        samples.append(value)
        print(f'Forward-only worlds={n} call={i} graph={captured} retained', flush=True)
    return dict(worlds=n, device=str(env.device), warp_device=str(env.wp_device),
        snapshot=descriptor, snapshot_value_sha256=snap.identity, snapshot_array_bytes=snap.nbytes,
        initial_kinematics_sha256=initial_hash, calls=receipts, comparison=comparison(samples),
        motor_preparations=1, integration_steps=0, optimizer_steps=0, training_admitted=False, physical_motion_authorized=False)


def retain(root, name, value):
    buffer = io.BytesIO(); torch.save(value, buffer); raw = buffer.getvalue()
    require(len(raw) <= SNAPSHOT_LIMIT, 'bounded serialized snapshot/output')
    require(sum(p.stat().st_size for p in root.iterdir())+len(raw) < TOTAL_LIMIT-16*1024*1024,
            'retained experiment quota and log reserve')
    path = host.supervisor.native._plain_path(root/name)
    with path.open('xb') as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    host.supervisor.native._fsync_dir(root)
    return dict(file=name, sha256=sha256(raw).hexdigest())


def output_path(source):
    host.supervisor.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-forward-'+source[:12])


def plan(source):
    return dict(protocol=PROTOCOL, source=source, inputs=host.identity(source), plant=plant.reference(),
        worlds=list(WORLDS), graph_order=list(ORDER), seed=523, snapshot_limit=SNAPSHOT_LIMIT,
        total_limit=TOTAL_LIMIT, snapshot_hash_binding='before-any-measured-call-and-before-every-replay',
        child_seconds=120, service_seconds=180, cutoff_unix=DEADLINE,
        motor_preparations_per_batch=1, integration_steps=0, optimizer_steps=0, training_admitted=False, physical_motion_authorized=False)


def prepare(source):
    check_window(); require(not torch.cuda.is_initialized(), 'CPU preparation')
    value = plan(source); root = output_path(source); root.mkdir(exist_ok=False)
    host.supervisor.write_json(root/'launch.json', value)
    return dict(output=str(root), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    path = output_path(source)/'launch.json'
    require(host.digest(path) == launch_sha, 'independent forward launch hash')
    value = host.supervisor.parse(host.supervisor.file_bytes(path))
    require(value == plan(source), 'exact source/runtime/plant plan')
    return value


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd); check_window(); checked(source, launch_sha)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'CUDA child only')
    host.wait_idle(); root = output_path(source)
    for n in WORLDS:
        result = batch(n, 'cuda:0', lambda name, value: retain(root, name, value))
        host.supervisor.write_json(root/f'batch-{n}.json', result)
    checked(source, launch_sha)


def verify(root):
    expected = {'launch.json', 'child.log'} | {f'batch-{n}.json' for n in WORLDS} | {
        f'input-{n}.pt' for n in WORLDS} | {f'output-{n}-{i}.pt' for n in WORLDS for i in range(8)}
    require({p.name for p in root.iterdir()} == expected, 'exact complete forward inventory')
    require(sum(p.stat().st_size for p in root.iterdir()) < TOTAL_LIMIT, 'retained forward quota')
    summaries = []
    def load(descriptor, name):
        raw = host.supervisor.file_bytes(root/name, limit=SNAPSHOT_LIMIT)
        require(descriptor == dict(file=name, sha256=sha256(raw).hexdigest()), 'raw forward evidence hash')
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    for n in WORLDS:
        r = host.supervisor.parse(host.supervisor.file_bytes(root/f'batch-{n}.json'))
        require(r['worlds'] == n and r['device'] == r['warp_device'] == 'cuda:0'
                and r['motor_preparations'] == 1
                and r['integration_steps'] == r['optimizer_steps'] == 0
                and r['training_admitted'] is r['physical_motion_authorized'] is False, 'forward-only CUDA batch')
        snapshot = load(r['snapshot'], f'input-{n}.pt')
        require(throughput.tree_hash(snapshot) == r['snapshot_value_sha256'], 'retained complete input values')
        require(sum(v['raw'].numel() for v in snapshot['arrays'].values()) == r['snapshot_array_bytes'] <= SNAPSHOT_LIMIT,
                'complete input byte count')
        require(len(r['calls']) == 8, 'all eight calls retained')
        samples = []
        for i, call in enumerate(r['calls']):
            require(call['call'] == i and call['graph'] is ORDER[i] and call['elapsed_s'] > 0
                    and call['input_sha256'] == r['snapshot_value_sha256'], 'exact replay input control')
            value = load(call['output'], f'output-{n}-{i}.pt'); validate_output(value, n)
            require(throughput.tree_hash(value['kinematics']) == r['initial_kinematics_sha256'], 'no integration receipt')
            samples.append(value)
        decided = comparison(samples)
        require(decided == r['comparison'], 'independently recomputed forward comparison')
        summaries.append(dict(worlds=n, **decided))
    return dict(protocol=PROTOCOL, batches=summaries, decision='forward-diagnostic-complete',
                training_admitted=False, graph_equivalence_established=False, physical_motion_authorized=False)


def supervise(source, launch_sha):
    check_window(); launch = checked(source, launch_sha); root = output_path(source)
    unit = 'microduck-stance-forward-'+source[:12]+'.service'
    actual = {k: host.read('systemctl', '--user', 'show', unit, '-p', k, '--value')
              for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState')}
    require(actual == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='3min', KillMode='control-group', ActiveState='active'),
            'independently timed forward service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only supervisor')
    require({p.name for p in root.iterdir()} == {'launch.json'}, 'one fresh forward attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed', training_admitted=False)
    try:
        with host.supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+600 < DEADLINE, 'closeout boundary')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live forward source drift')
            report['child'] = host.supervisor.supervised_process(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log', cwd=host.ROOT,
                env=host.supervisor.child_environment(), lock_fd=fd, timeout=120, guard=guard)
            host.check_log(root/'child.log')
            report['result'] = verify(root); report['decision'] = report['result']['decision']
            report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error=str(exc), error_notes=getattr(exc, '__notes__', [])); raise
    finally:
        report['files'] = {p.name: sha256(host.supervisor.file_bytes(p, limit=SNAPSHOT_LIMIT, allow_empty=True)).hexdigest()
                           for p in sorted(root.iterdir()) if p.is_file()}
        host.supervisor.write_json(root/'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    p.add_argument('--source', required=True); p.add_argument('--launch-sha256'); p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    if a.mode == 'prepare': print(canonical(prepare(a.source)))
    elif a.mode == 'supervise': supervise(a.source, a.launch_sha256)
    else: child(a.source, a.launch_sha256, a.lock_fd)


if __name__ == '__main__': main()
