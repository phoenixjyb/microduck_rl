"""Short, non-admitting profile over the unchanged retained WSL runtime.

Run an authenticated copy of this file by absolute path from the old checkout.
The tool commit and runtime commit are deliberately separate: no fast-forward,
environment rewrite, learned weights, or relaxation of a rejected timing gate.
"""
import argparse
import cProfile
from hashlib import sha256
import math
import os
from pathlib import Path
import pstats
import random
import subprocess
import time

import numpy as np
import torch

from mjlab_microduck import stance_wsl_qualification as q
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files, execution, throughput = q.host, q.files, q.execution, q.throughput
PROTOCOL = 'football-b1n-wsl-collection-profile-v1'
RUNTIME_SOURCE = '987b452dbfdb4cba7c7fef79f419790bd602ae3a'
TOOL_PATH = 'src/mjlab_microduck/stance_wsl_collection_profile.py'
BATCHES = ('baseline-before', 'python-profile', 'baseline-after')
NO_ADMISSION = dict(training_admitted=False, timing_qualified=False,
                    learned_stance=False, football_balance=False,
                    physical_motion_authorized=False)


def output_path(tool_source):
    files.hex_id(tool_source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-collection-profile-'+tool_source[:12])


def service_name(tool_source):
    files.hex_id(tool_source, 40)
    return 'microduck-wsl-collection-profile-'+tool_source[:12]+'.service'


def tool_identity(tool_source, expected_sha):
    files.hex_id(tool_source, 40); files.hex_id(expected_sha, 64)
    raw = files.file_bytes(Path(__file__).resolve())
    committed = subprocess.check_output(['git', 'show', tool_source+':'+TOOL_PATH],
                                        cwd=host.ROOT, timeout=5)
    require(raw == committed and sha256(raw).hexdigest() == expected_sha,
            'independently pinned tool bytes at exact tool commit')
    return dict(source=tool_source, file=TOOL_PATH, sha256=expected_sha,
                runtime_source=RUNTIME_SOURCE)


def plan(tool_source, tool_sha, deadline):
    require(execution.PROFILE['name'] == execution.WSL, 'fixed authorized WSL profile')
    return dict(protocol=PROTOCOL, tool=tool_identity(tool_source, tool_sha),
        inputs=host.identity(RUNTIME_SOURCE), deadline_unix=deadline,
        child_seconds=900, service_seconds=960, closeout_seconds=600,
        memory_max_bytes=6*1024**3, cpu_quota_per_sec_usec=2_000_000, nice=10,
        worlds=64, ticks_per_update=24, seed=523, warmup_updates=2, measured_updates=8,
        batches=list(BATCHES), optimizer_steps=0, forward_graph=False,
        parent_sha256=throughput.checkpoint.LEAN_PARENT_SHA256,
        timing_basis=q.validate_timing_basis(RUNTIME_SOURCE),
        instrumented_times_excluded_from_qualification=True, **NO_ADMISSION)


def prepare(tool_source, tool_sha, deadline):
    throughput.check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only profile preparation')
    launch = plan(tool_source, tool_sha, deadline)
    runtime = plant.runtime_bytes(RUNTIME_SOURCE, plant.build_entity().compile())
    parent = files.file_bytes(throughput.parent_path())
    require(sha256(parent).hexdigest() == launch['parent_sha256'], 'unchanged frozen parent')
    # Size only this short diagnostic from retained measurement, not a learner.
    basis = launch['timing_basis']['measured_caps']
    estimate = 30*basis['per_update_worst_seconds']+basis['setup_seconds']
    require(estimate*1.5+60 < 900, 'bounded diagnostic with instrumentation reserve')
    root = output_path(tool_source); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runner.py', files.file_bytes(Path(__file__).resolve()))
    smoke.write_bytes(root/'runtime.json', runtime)
    smoke.write_bytes(root/'parent.pt', parent)
    launch['runtime_sha256'] = sha256(runtime).hexdigest()
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(tool_source),
                launch_sha256=host.digest(root/'launch.json'))


def checked(tool_source, tool_sha, launch_sha):
    root = output_path(tool_source)
    files.hex_id(launch_sha, 64)
    require(host.digest(root/'launch.json') == launch_sha, 'independent profile launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    expected = plan(tool_source, tool_sha, launch['deadline_unix'])
    expected['runtime_sha256'] = host.digest(root/'runtime.json')
    require(launch == expected and host.digest(root/'runner.py') == tool_sha,
            'unchanged runtime, profiling tool, launch and host')
    plant.checked_runtime(files.parse(files.file_bytes(root/'runtime.json')), RUNTIME_SOURCE)
    throughput.load_parent(root)
    return launch


def cpu_snapshot():
    """Service-wide cgroup accounting, not process CPU or hardware attribution."""
    entries = Path('/proc/self/cgroup').read_text().splitlines()
    paths = [row[3:] for row in entries if row.startswith('0::')]
    require(len(paths) == 1 and paths[0].startswith('/') and '..' not in Path(paths[0]).parts,
            'one contained unified cgroup')
    root = Path('/sys/fs/cgroup')/paths[0].lstrip('/')
    values = dict((key, int(value)) for key, value in
                  (line.split() for line in (root/'cpu.stat').read_text().splitlines()))
    require(all(key in values for key in ('usage_usec', 'nr_periods', 'nr_throttled', 'throttled_usec'))
            and all(value >= 0 for value in values.values()), 'finite cgroup counters')
    return dict(path=paths[0], cpu_max=(root/'cpu.max').read_text().strip(), counters=values)


def cpu_delta(before, after):
    for snapshot in (before, after):
        require(type(snapshot['counters']) is dict
                and all(key in snapshot['counters'] for key in
                        ('usage_usec','nr_periods','nr_throttled','throttled_usec'))
                and all(type(v) is int and v >= 0 for v in snapshot['counters'].values()),
                'nonnegative integer service CPU counters')
    require(before['path'] == after['path'] and before['cpu_max'] == after['cpu_max']
            and before['counters'].keys() == after['counters'].keys(), 'unchanged service cgroup')
    quota, period = before['cpu_max'].split()
    require(int(quota) == 2*int(period) and int(period) > 0, 'declared 200 percent CPU quota')
    delta = {key: after['counters'][key]-value for key,value in before['counters'].items()}
    require(all(value >= 0 for value in delta.values()), 'monotone cgroup counters')
    return dict(cpu_max=before['cpu_max'], counters=delta, scope='whole-service-not-child-only')


def profile_rows(profiler):
    stats = pstats.Stats(profiler)
    rows = [dict(file=file, line=line, function=name, primitive_calls=cc, calls=nc,
                 self_seconds=tt, cumulative_seconds=ct)
            for (file,line,name),(cc,nc,tt,ct,_) in stats.stats.items()]
    return sorted(rows, key=lambda row: (-row['cumulative_seconds'], row['file'],
                                         row['line'], row['function']))


def validate_rows(rows):
    require(type(rows) is list and len(rows) > 0, 'nonempty Python function profile')
    keys = {'file','line','function','primitive_calls','calls','self_seconds','cumulative_seconds'}
    for row in rows:
        require(set(row) == keys and type(row['file']) is str and type(row['function']) is str,
                'exact profile fields')
        require(all(type(row[key]) is int and row[key] >= 0
                    for key in ('line','primitive_calls','calls'))
                and row['calls'] >= row['primitive_calls'], 'profile call counts')
        require(all(type(row[key]) is float and math.isfinite(row[key]) and row[key] >= 0
                    for key in ('self_seconds','cumulative_seconds'))
                and row['cumulative_seconds']+1e-9 >= row['self_seconds'], 'finite profile times')
    require(len({(r['file'],r['line'],r['function']) for r in rows}) == len(rows), 'unique profile functions')


def validate_collection(value):
    require(value['worlds'] == 64 and value['ticks_per_update'] == 24
            and value['physics_device'] == 'cuda:0' and value['optimizer_steps'] == 0,
            'declared actual collection, never learning')
    updates = value['updates']
    require(type(updates) is list and len(updates) == 10, 'two warmups plus eight measurements')
    for index,row in enumerate(updates):
        require(row['update'] == index and type(row['update']) is int
                and row['warmup'] is (index < 2), 'ordered exact collection updates')
        throughput.summarize(row['tick_seconds'])
        require(len(row['tick_seconds']) == 24 and type(row['seconds']) is float
                and math.isfinite(row['seconds']) and row['seconds'] >= sum(row['tick_seconds'])
                and row['tick_max_seconds'] == max(row['tick_seconds']), 'complete tick timings')
    require(value['summarize'] == throughput.summarize([r['seconds'] for r in updates[2:]])
            and type(value['setup_seconds']) is float and math.isfinite(value['setup_seconds'])
            and value['setup_seconds'] >= 0, 're-derived collection timing')


def summarize_batches(batches):
    require(type(batches) is list and len(batches) == 3, 'all three declared batches required')
    baseline = []
    for name, batch in zip(BATCHES, batches):
        require(batch['name'] == name and batch['instrumented'] is (name == 'python-profile'),
                'fixed baseline/profile/baseline order')
        validate_collection(batch['collection'])
        require(batch['cpu_delta'] == cpu_delta(batch['cpu_before'], batch['cpu_after']),
                're-derived cgroup accounting')
        if not batch['instrumented']:
            baseline.extend(batch['collection']['summarize']['series'])
    return dict(decision='profile-complete-not-training-qualification',
                baseline=throughput.summarize(baseline),
                profiled=throughput.summarize(batches[1]['collection']['summarize']['series']),
                python_profile_scope='inclusive-wall-time-not-GPU-kernel-attribution', **NO_ADMISSION)


def child(tool_source, tool_sha, launch_sha, fd):
    smoke.inherited_lease(fd)
    launch = checked(tool_source, tool_sha, launch_sha)
    throughput.check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'explicit CUDA child')
    host.wait_idle(); root = output_path(tool_source)
    payload = host.cases('cuda:0'); payload['launch_sha256'] = launch_sha
    host.validate_payload(payload, launch_sha); files.write_json(root/'integration.json', payload)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    batches = []
    for name in BATCHES:
        require(time.time()+600 < launch['deadline_unix'], 'profile closeout reserve')
        random.seed(523); np.random.seed(523); torch.manual_seed(523)
        env = WarpStanceRuntime(64, device='cuda:0')
        require(env.forward_graph is None and env.wp_device.is_cuda
                and plant.describe(env.native) == plant.reference(), 'unchanged eager CUDA plant')
        actor = throughput.load_parent(root)['actor'].eval()
        before = cpu_snapshot(); profiler = cProfile.Profile() if name == 'python-profile' else None
        if profiler is not None: profiler.enable()
        try:
            collection = throughput.measure_collection(env, actor)
        finally:
            if profiler is not None: profiler.disable()
        after = cpu_snapshot()
        batch = dict(name=name, instrumented=profiler is not None, collection=collection,
                     cpu_before=before, cpu_after=after, cpu_delta=cpu_delta(before,after))
        validate_collection(collection)
        files.write_json(root/(name+'.json'), batch); batches.append(batch)
        if profiler is not None:
            rows = profile_rows(profiler); validate_rows(rows)
            files.write_json(root/'python-functions.json', rows)
        print(canonical(dict(batch=name, seconds=collection['summarize'],
                             cpu_delta=batch['cpu_delta'])), flush=True)
        del env
    files.write_json(root/'summary.json', summarize_batches(batches))
    checked(tool_source, tool_sha, launch_sha)


def verify_evidence(root, launch_sha, tool_sha):
    """Offline hash/structure replay; does not itself attest a live host."""
    files.hex_id(launch_sha, 64); files.hex_id(tool_sha, 64)
    report = files.parse(files.file_bytes(root/'report.json'))
    require(report['protocol'] == PROTOCOL and report['decision'] == 'profile-complete-not-training-qualification'
            and report['launch_sha256'] == launch_sha
            and all(report[key] is False for key in NO_ADMISSION), 'completed non-admitting diagnostic')
    names = {'runner.py','parent.pt','runtime.json','launch.json','integration.json','child.log',
             'baseline-before.json','python-profile.json','baseline-after.json',
             'python-functions.json','summary.json'}
    require(set(report['files']) == names and {p.name for p in root.iterdir()} == names|{'report.json'},
            'exact profile evidence inventory')
    require(all(host.digest(root/name) == digest for name,digest in report['files'].items()), 'all profile hashes')
    require(host.digest(root/'launch.json') == launch_sha and host.digest(root/'runner.py') == tool_sha,
            'independent launch and tool hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    require(launch['protocol'] == PROTOCOL and launch['tool']['sha256'] == tool_sha
            and launch['tool']['runtime_source'] == launch['inputs']['source'] == RUNTIME_SOURCE
            and all(launch[key] is False for key in NO_ADMISSION), 'honest separate tool/runtime binding')
    require(launch['tool']['file'] == TOOL_PATH and launch['batches'] == list(BATCHES)
            and (launch['worlds'],launch['ticks_per_update'],launch['seed'],
                 launch['warmup_updates'],launch['measured_updates']) == (64,24,523,2,8)
            and (launch['child_seconds'],launch['service_seconds'],launch['closeout_seconds']) == (900,960,600)
            and (launch['memory_max_bytes'],launch['cpu_quota_per_sec_usec'],launch['nice']) ==
                (6*1024**3,2_000_000,10)
            and launch['optimizer_steps'] == 0 and launch['forward_graph'] is False
            and launch['instrumented_times_excluded_from_qualification'] is True,
            'fixed non-learning profiling protocol and resource bounds')
    require(host.digest(root/'parent.pt') == throughput.checkpoint.LEAN_PARENT_SHA256
            and host.digest(root/'runtime.json') == launch['runtime_sha256'], 'parent/runtime hashes')
    require(report['child']['returncode'] == 0, 'successful owned child')
    host.validate_payload(files.parse(files.file_bytes(root/'integration.json')), launch_sha)
    host.check_log(root/'child.log')
    batches = [files.parse(files.file_bytes(root/(name+'.json'))) for name in BATCHES]
    summary = summarize_batches(batches)
    require(files.parse(files.file_bytes(root/'summary.json')) == summary, 'deterministic profile summary')
    validate_rows(files.parse(files.file_bytes(root/'python-functions.json')))
    return dict(report_sha256=host.digest(root/'report.json'), summary=summary,
                live_host_rechecked=False)


def supervise(tool_source, tool_sha, launch_sha):
    launch = checked(tool_source, tool_sha, launch_sha); root = output_path(tool_source)
    throughput.check_window(launch['deadline_unix'], launching=True)
    expected = dict(MainPID=str(os.getpid()),RuntimeMaxUSec='16min',KillMode='control-group',
                    ActiveState='active',MemoryMax=str(6*1024**3),CPUQuotaPerSecUSec='2s',Nice='10')
    actual = {key: host.read('systemctl','--user','show',service_name(tool_source),'-p',key,'--value')
              for key in expected}
    require(actual == expected, 'independently bounded service and declared CPU/RAM limits')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only supervisor')
    require({p.name for p in root.iterdir()} == {'runner.py','parent.pt','runtime.json','launch.json'}, 'fresh profile')
    report = dict(protocol=PROTOCOL,launch_sha256=launch_sha,decision='failed',**NO_ADMISSION)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+600 < launch['deadline_unix'], 'profile closeout reserve')
                host.check_log(root/'child.log')
                require(host.identity(RUNTIME_SOURCE) == launch['inputs'], 'live runtime/host drift')
                tool_identity(tool_source, tool_sha)
            report['child'] = files.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'),str(root/'runner.py'),'child','--tool-source',tool_source,
                 '--tool-sha256',tool_sha,'--launch-sha256',launch_sha,'--lock-fd',str(fd)],
                root/'child.log',cwd=host.ROOT,env=files.child_environment(),lock_fd=fd,guard=guard)
            checked(tool_source,tool_sha,launch_sha)
            report['idle_after'] = host.wait_idle()
            report['decision'] = 'profile-complete-not-training-qualification'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc),error_notes=getattr(exc,'__notes__',[])); raise
    finally:
        report['files'] = {p.name:host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json',report)
    verify_evidence(root,launch_sha,tool_sha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','supervise','child','verify'))
    p.add_argument('--tool-source',required=True); p.add_argument('--tool-sha256',required=True)
    p.add_argument('--deadline-unix',type=int); p.add_argument('--launch-sha256'); p.add_argument('--lock-fd',type=int)
    a = p.parse_args()
    if a.mode == 'prepare': value = prepare(a.tool_source,a.tool_sha256,a.deadline_unix)
    elif a.mode == 'supervise': return supervise(a.tool_source,a.tool_sha256,a.launch_sha256)
    elif a.mode == 'child': return child(a.tool_source,a.tool_sha256,a.launch_sha256,a.lock_fd)
    else:
        checked(a.tool_source,a.tool_sha256,a.launch_sha256)
        value = verify_evidence(output_path(a.tool_source),a.launch_sha256,a.tool_sha256)
    print(canonical(value))


if __name__ == '__main__': main()
