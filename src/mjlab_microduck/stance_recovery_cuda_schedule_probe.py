"""Two short leased scheduled CUDA cases, with genuine CPU preparation/replay.

This is a new renewed-window integration protocol, not a new duration for any
expired job. A 60-tick capture cannot pass the five-second recovery gate.
"""
import argparse
import gc
from hashlib import sha256
import io
import os
import time

import torch

from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_cuda_schedule_trace as evidence
from mjlab_microduck import stance_recovery_dose_screen as prior
from mjlab_microduck import stance_recovery_schedule_probe as base
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1d-cuda-schedule-integration-probe-v1'
MODULE = 'mjlab_microduck.stance_recovery_cuda_schedule_probe'
CELL_IDS = evidence.ALLOWED_CELLS
PREFIX_STEP = 250
PREPARE_SECONDS, SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS = 120, 240, 180, 120
LAUNCH_RESERVE = PREPARE_SECONDS + SERVICE_SECONDS + CLOSEOUT_SECONDS + 60
COLLECTION_SECONDS, SERIALIZATION_RESERVE = 60, 30
LOG_LIMIT = 1024 * 1024
PREPARE_FILES = {'checkpoint.pt', 'launch.json', 'cpu-qualification.json'} | {
    f'cpu-{i}{suffix}' for i in range(2) for suffix in ('.pt', '.json', '-replay.json')}
CAPTURE_FILES = {f'cuda-{i}{suffix}' for i in range(2) for suffix in ('.pt', '.json')}
COMPLETE_FILES = PREPARE_FILES | CAPTURE_FILES | {'child.log'}


def output_path(source):
    base.files.hex_id(source, 40)
    return base.host.ROOT/'artifacts/evaluations'/('stance-wsl-scheduled-cuda-integration-'+source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ('prepare', 'supervise', 'closeout'), 'declared scheduled CUDA service mode')
    return 'microduck-scheduled-cuda-'+mode+'-'+source[:12]+'.service'


def service_properties(source, mode):
    props = {k: base.host.read('systemctl', '--user', 'show', service_name(source, mode), '-p', k, '--value')
             for k in ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    seconds = dict(prepare=PREPARE_SECONDS, supervise=SERVICE_SECONDS, closeout=CLOSEOUT_SECONDS)[mode]
    gib = 3 if mode == 'supervise' else 2
    require(props == dict(MainPID=str(os.getpid()), ActiveState='active', RuntimeMaxUSec=f'{seconds//60}min',
        MemoryMax=str(gib*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'),
        'exact independently capped renewed scheduled CUDA service')
    return props


def context(source):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '0'), 'explicit owned CPU or CUDA process')
    identity = base.host.identity(source)
    cpu = base.profile.checked_receipt()
    services = base.protected_state()
    require(all(v == 'inactive' for v in services.values()), 'protected services remain inactive')
    return dict(source_identity=identity, cpu_math_profile=cpu,
                preserved_filmbrain=base.retained.d0.filmbrain_state(), services_before=services)


def prepare(source):
    started = time.monotonic(); window.check(reserve_seconds=LAUNCH_RESERVE)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only fresh scheduled preparation')
    props = service_properties(source, 'prepare'); inputs = context(source)
    previous, cp = prior.prerequisite()
    declarations = [base.schedule.declaration(source, 'dose', 'held-out', [cell]) for cell in CELL_IDS]
    root = base.files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    base.retained.write_capture(root/'checkpoint.pt', cp)
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    scores, prefixes, captures = [], [], []
    compiled = None
    for index, declaration in enumerate(declarations):
        window.check(reserve_seconds=SERVICE_SECONDS+CLOSEOUT_SECONDS+60)
        require(time.monotonic()-started < PREPARE_SECONDS-40, 'reserve next CPU qualification and retention')
        base.seed_cpu(); env = ScheduledRecoveryRuntime(declaration, device='cpu')
        if compiled is None: compiled = env.binding
        require(env.binding == compiled, 'identical genuine CPU compiled plant')
        value = base.evidence.collect(env, cp, deadline_monotonic=time.monotonic()+30,
                                      policy_tick_limit=evidence.POLICY_TICK_LIMIT)
        raw = base.evidence.encode(value); digest = sha256(raw).hexdigest()
        base.retained.write_capture(root/f'cpu-{index}.pt', raw)
        capture = dict(source=source, declaration=declaration, capture_sha256=digest,
                       capture_bytes=len(raw), collection=value['collection'])
        base.files.write_json(root/f'cpu-{index}.json', capture)
        del env, value; gc.collect()
        score = base.evidence.verify(raw, digest, cp, declaration, compiled)
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        prefix = base.evidence.prefix_hash(value, PREFIX_STEP)
        del raw, value; gc.collect()
        base.files.write_json(root/f'cpu-{index}-replay.json', score)
        require(0 < score['collection']['elapsed_seconds'] < 30
                and score['collection']['policy_ticks'] == evidence.POLICY_TICK_LIMIT
                and score['pulse']['checked_physics_steps'] == evidence.MAX_PHYSICS_STEPS
                and score['pulse']['complete_pulse_delivery'] and score['pulse']['complete_phase_checks']
                and not score['numerical_diagnostic']['complete_first_attempt']
                and score['actor_replay_max_abs_error'] == 0., 'complete short genuine CPU scheduled qualification')
        scores.append(score); prefixes.append(prefix); captures.append(capture)
    require(prefixes[0] == prefixes[1], 'exact CPU matched pre-push prefix')
    require(context(source) == inputs and not torch.cuda.is_initialized()
            and time.monotonic()-started < PREPARE_SECONDS, 'unchanged bounded CPU preparation')
    qualification = dict(protocol=PROTOCOL, source=source, captures=captures, scores=scores,
        prefixes=prefixes, prefixes_identical=True, elapsed_seconds=time.monotonic()-started,
        cpu_initialized_only=True, optimizer_steps=0, **base.baseline.FALSE_FLAGS)
    base.files.write_json(root/'cpu-qualification.json', qualification)
    launch = dict(protocol=PROTOCOL, source=source, **inputs, prerequisite=previous,
        window=window.declaration(), declarations=declarations, compiled_plant=compiled,
        evaluation_seed=evidence.EVALUATION_SEED, policy_ticks=evidence.POLICY_TICK_LIMIT,
        prepare_seconds=PREPARE_SECONDS, service_seconds=SERVICE_SECONDS,
        child_seconds=CHILD_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
        collection_seconds=COLLECTION_SECONDS, serialization_reserve_seconds=SERIALIZATION_RESERVE,
        prefix_step=PREFIX_STEP, qualification_sha256=base.host.digest(root/'cpu-qualification.json'),
        prepare_service_properties=props, optimizer_steps=0, **base.baseline.FALSE_FLAGS)
    base.files.write_json(root/'launch.json', launch)
    return dict(output=str(root), launch_sha256=base.host.digest(root/'launch.json'))


def checked(source, launch_sha, *, replay_cpu=True):
    require(type(replay_cpu) is bool, 'explicit CPU replay or hash-only leased-child entry')
    if not replay_cpu:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and not torch.cuda.is_initialized(),
                'hash-only child checks precede CUDA initialization')
    base.files.hex_id(launch_sha, 64); root = output_path(source)
    require(base.host.digest(root/'launch.json') == launch_sha, 'whole launch bytes before parsing')
    launch = base.files.parse(base.files.file_bytes(root/'launch.json'))
    require(launch['protocol'] == PROTOCOL and launch['source'] == source
            and launch['window'] == window.declaration()
            and launch['declarations'] == [base.schedule.declaration(source, 'dose', 'held-out', [c]) for c in CELL_IDS]
            and all(launch[k] is False for k in base.baseline.FALSE_FLAGS)
            and launch['optimizer_steps'] == 0 and launch['evaluation_seed'] == evidence.EVALUATION_SEED
            and launch['policy_ticks'] == evidence.POLICY_TICK_LIMIT
            and [launch[k] for k in ('prepare_seconds', 'service_seconds', 'child_seconds', 'closeout_seconds',
                    'collection_seconds', 'serialization_reserve_seconds', 'prefix_step')]
                == [PREPARE_SECONDS, SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS,
                    COLLECTION_SECONDS, SERIALIZATION_RESERVE, PREFIX_STEP], 'exact renewed launch and fixed caps')
    current = context(source)
    require({k: launch[k] for k in current} == current, 'unchanged source/profile/protected/unrelated services')
    previous, cp = prior.prerequisite()
    require(previous == launch['prerequisite']
            and base.files.file_bytes(root/'checkpoint.pt', limit=base.retained.evaluation.checkpoint.LIMIT) == cp,
            'unchanged retained prerequisite and real parent bytes')
    require(base.host.digest(root/'cpu-qualification.json') == launch['qualification_sha256'],
            'whole CPU qualification hash before parsing')
    q = base.files.parse(base.files.file_bytes(root/'cpu-qualification.json'))
    require(q['protocol'] == PROTOCOL and q['source'] == source and q['prefixes_identical'] is True
            and q['optimizer_steps'] == 0 and 0 < q['elapsed_seconds'] < PREPARE_SECONDS
            and all(q[k] is False for k in base.baseline.FALSE_FLAGS), 'bounded non-admitting CPU qualification')
    prefixes = []
    for index, declaration in enumerate(launch['declarations']):
        capture = base.files.parse(base.files.file_bytes(root/f'cpu-{index}.json'))
        raw = base.files.file_bytes(root/f'cpu-{index}.pt', limit=base.evidence.LIMIT)
        require(capture == q['captures'][index] and capture['source'] == source
                and capture['declaration'] == declaration and capture['capture_bytes'] == len(raw)
                and sha256(raw).hexdigest() == capture['capture_sha256'],
                'exact retained CPU raw metadata')
        retained_score = base.files.parse(base.files.file_bytes(root/f'cpu-{index}-replay.json'))
        require(retained_score == q['scores'][index], 'exact retained CPU qualification score')
        if replay_cpu:
            scored = base.evidence.verify(raw, capture['capture_sha256'], cp, declaration, launch['compiled_plant'])
            require(scored == retained_score, 'independent CPU qualification rescore identical')
            value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            prefixes.append(base.evidence.prefix_hash(value, PREFIX_STEP))
            del value; gc.collect()
    if replay_cpu:
        require(prefixes == q['prefixes'] and prefixes[0] == prefixes[1], 'independent matched CPU prefix recomputation')
    return launch, cp


def child(source, launch_sha, fd, started):
    window.check(reserve_seconds=CHILD_SECONDS+CLOSEOUT_SECONDS+60)
    require(type(started) is float and 0 < time.monotonic()-started < CHILD_SECONDS-SERIALIZATION_RESERVE,
            'same-clock bounded child entry')
    base.retained.smoke.inherited_lease(fd)
    launch, cp = checked(source, launch_sha, replay_cpu=False)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'leased CUDA0 child only')
    base.host.wait_idle(); root = output_path(source)
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    for index, declaration in enumerate(launch['declarations']):
        require(time.monotonic()-started < CHILD_SECONDS-COLLECTION_SECONDS-SERIALIZATION_RESERVE,
                'reserve complete next CUDA collection and serialization')
        base.seed_cpu(); torch.manual_seed(evidence.EVALUATION_SEED)
        built = time.monotonic(); env = ScheduledRecoveryRuntime(declaration, device='cuda:0')
        construction_seconds = time.monotonic()-built
        require(env.binding == launch['compiled_plant'], 'same declared actual CPU/CUDA compiled plant')
        value = evidence.collect(env, cp, deadline_monotonic=min(time.monotonic()+COLLECTION_SECONDS,
            started+CHILD_SECONDS-SERIALIZATION_RESERVE))
        raw = evidence.encode(value); digest = sha256(raw).hexdigest()
        base.retained.write_capture(root/f'cuda-{index}.pt', raw)
        base.files.write_json(root/f'cuda-{index}.json', dict(protocol=PROTOCOL, source=source,
            launch_sha256=launch_sha, declaration=declaration, capture_sha256=digest,
            capture_bytes=len(raw), collection=value['collection'], backend=value['backend'],
            construction_seconds=construction_seconds, elapsed_seconds=time.monotonic()-started,
            optimizer_steps=0, **base.baseline.FALSE_FLAGS))
        del raw, value, env; gc.collect()
    require(time.monotonic()-started < CHILD_SECONDS-SERIALIZATION_RESERVE,
            'retained captures within fixed child serialization reserve')


def replay(source, launch, cp, launch_sha):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CUDA-hidden scheduled replay')
    root = output_path(source); scores, prefixes = [], []
    for index, declaration in enumerate(launch['declarations']):
        capture = base.files.parse(base.files.file_bytes(root/f'cuda-{index}.json'))
        raw = base.files.file_bytes(root/f'cuda-{index}.pt', limit=evidence.CAPTURE_LIMIT)
        require(capture['protocol'] == PROTOCOL and capture['source'] == source
                and capture['launch_sha256'] == launch_sha and capture['declaration'] == declaration
                and capture['capture_bytes'] == len(raw) and capture['optimizer_steps'] == 0
                and all(capture[k] is False for k in base.baseline.FALSE_FLAGS), 'exact CUDA capture metadata')
        score = evidence.verify(raw, capture['capture_sha256'], cp, declaration, launch['compiled_plant'])
        require(score['collection'] == capture['collection']
                and 0 < score['collection']['elapsed_seconds'] < COLLECTION_SECONDS
                and type(capture['construction_seconds']) is float
                and 0 < capture['construction_seconds'] < CHILD_SECONDS
                and type(capture['elapsed_seconds']) is float
                and 0 < capture['elapsed_seconds'] < CHILD_SECONDS-SERIALIZATION_RESERVE,
                'raw CUDA collection and phase timings within retained fixed caps')
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        prefix = evidence.prefix_hash(value, PREFIX_STEP)
        del raw, value; gc.collect()
        scores.append(score); prefixes.append(prefix)
    matched = prefixes[0] == prefixes[1]
    passed = matched and all(s['integration_qualified'] for s in scores)
    return dict(scores=scores, prefixes=prefixes, prefixes_identical=matched,
        decision='scheduled-cuda-integration-qualified' if passed else 'scheduled-cuda-integration-rejected')


def check_log(root):
    path = root/'child.log'
    if path.exists():
        require(path.stat().st_size <= LOG_LIMIT, 'bounded owned scheduled CUDA child log')
        base.host.check_log(path)


def supervise(source, launch_sha):
    started = time.monotonic(); window.check(reserve_seconds=SERVICE_SECONDS+CLOSEOUT_SECONDS+60)
    props = service_properties(source, 'supervise'); root = output_path(source)
    report = dict(protocol=PROTOCOL, source=source, launch_sha256=launch_sha,
        service_properties=props, decision='scheduled-cuda-integration-failed', optimizer_steps=0,
        **base.baseline.FALSE_FLAGS)
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU supervisor')
        launch, cp = checked(source, launch_sha)
        require({p.name for p in root.iterdir()} == PREPARE_FILES, 'one fresh exact CPU-qualified CUDA attempt')
        with base.files.gpu_lease() as fd:
            report['idle_before'] = base.host.wait_idle()
            child_started = time.monotonic()
            command = [str(base.host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                '--launch-sha256', launch_sha, '--lock-fd', str(fd), '--started-monotonic', repr(child_started)]
            env = base.files.child_environment(); env.update(base.profile.settings())
            def guard():
                window.check(reserve_seconds=CLOSEOUT_SECONDS+60); check_log(root)
                require(base.host.read('git', 'rev-parse', 'HEAD') == source
                        and not base.host.read('git', 'status', '--porcelain'), 'unchanged exact live scheduled source')
                require(base.retained.d0.filmbrain_state() == launch['preserved_filmbrain'], 'preserved FilmBrain state')
            # New separately declared 180-second wrapper. Every old public cap
            # remains unchanged; the lower-level watchdog only owns this child.
            report['child'] = base.files._timed_process(command, root/'child.log', cwd=base.host.ROOT,
                env=env, lock_fd=fd, timeout=CHILD_SECONDS, monitor=base.files.live_gpu, guard=guard)
            check_log(root)
            report.update(replay(source, launch, cp, launch_sha))
            current = context(source)
            require(current == {k: launch[k] for k in current}, 'unchanged source/service closeout')
            report['idle_after'] = base.host.wait_idle()
        require(time.monotonic()-started < SERVICE_SECONDS, 'bounded complete scheduled CUDA supervisor')
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['files'] = {p.name: base.host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        base.files.write_json(root/'report.json', report)


def closeout(source, launch_sha):
    started = time.monotonic(); window.check(reserve_seconds=CLOSEOUT_SECONDS)
    props = service_properties(source, 'closeout'); root = output_path(source)
    launch, cp = checked(source, launch_sha)
    report_raw = base.files.file_bytes(root/'report.json'); report = base.files.parse(report_raw)
    require(report['protocol'] == PROTOCOL and report['source'] == source
            and report['launch_sha256'] == launch_sha and report['child']['returncode'] == 0
            and report['decision'] in ('scheduled-cuda-integration-qualified', 'scheduled-cuda-integration-rejected'),
            'complete qualified or rejected numerical integration report')
    require(set(report['files']) == COMPLETE_FILES
            and {p.name for p in root.iterdir()} == COMPLETE_FILES | {'report.json'},
            'exact static independently closed inventory')
    inventory = {}
    for name, digest in report['files'].items():
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and base.host.digest(root/name) == digest, 'whole retained scheduled file hash '+str(name))
        inventory[name] = dict(sha256=digest, bytes=(root/name).stat().st_size)
    result = replay(source, launch, cp, launch_sha)
    require(all(result[k] == report[k] for k in result), 'fresh whole CPU rescore and decision identical')
    idle = base.host.wait_idle()
    require(time.monotonic()-started < CLOSEOUT_SECONDS and not torch.cuda.is_initialized(), 'bounded independent CPU closeout')
    inventory['report.json'] = dict(sha256=sha256(report_raw).hexdigest(), bytes=len(report_raw))
    result.update(protocol=PROTOCOL, source=source, launch_sha256=launch_sha,
        report_sha256=sha256(report_raw).hexdigest(), service_properties=props,
        files_rehashed=inventory, cases_checked=2, whole_cpu_rescore_identical=True,
        independent_native_attestation=False, whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, idle_after=idle, elapsed_seconds=time.monotonic()-started,
        optimizer_steps=0, **base.baseline.FALSE_FLAGS)
    base.files.write_json(root/'independent-closeout.json', result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'supervise', 'child', 'closeout'))
    p.add_argument('--source', required=True); p.add_argument('--launch-sha256')
    p.add_argument('--lock-fd', type=int); p.add_argument('--started-monotonic', type=float)
    a = p.parse_args(argv)
    if a.mode == 'prepare': print(prepare(a.source))
    elif a.mode == 'supervise': supervise(a.source, a.launch_sha256)
    elif a.mode == 'closeout': closeout(a.source, a.launch_sha256)
    else: child(a.source, a.launch_sha256, a.lock_fd, a.started_monotonic)


if __name__ == '__main__':
    main()
