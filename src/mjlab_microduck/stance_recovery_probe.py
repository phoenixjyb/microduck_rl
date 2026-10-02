"""One leased full-length frozen +x recovery timing probe; never training."""
import argparse
from hashlib import sha256
import io
import json
import math
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as force_fixture
from mjlab_microduck import stance_disturbance_probe as d0
from mjlab_microduck import stance_historical_replication_auth as auth
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_trace as evidence
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import require

host, files = d0.host, d0.files
MODULE = 'mjlab_microduck.stance_recovery_probe'
CPU_QUAL_TICKS, CPU_QUAL_SECONDS = 52, 120
D0_SOURCE = '6d9f950254cc2127361c3384065217c8f19a1b42'
D0_ROOT = host.ROOT/'artifacts/evaluations/stance-wsl-d0-force-fixture-6d9f950254cc'
D0_FILES = {
    'launch.json': 'c80cbc3ac2d63a65ebafd1b5a639cb200f597df67ff12733ccd5392a71c558f2',
    'report.json': '33743033bf292921d09df7ac682c4dd16910af03b10318fabf9137fd72ae05c4',
    'capture.json': '348e9eda64826367f84d02783ed6dad9e062fb70290d206684ec372910468806',
    'capture.pt': '823f2902a5352c7b78ef2c62dcbd117995581644566502f33cfb256b9b7305f9',
    'cpu-capture.pt': '4b82126a4c8fcd2f18948bb266b7fa075c3862749230d44a36583c50bf7ffc5e',
    'cpu-qualification.json': 'eb14762d8dd25a7b54d7a2e2015abfe8fbb78695a94a71d279ebfed51e2be839',
    'child.log': '923756654d29bdd94a5b1ef697ce1964cb465ba77bd2398a94398f92a4622377',
}
D0_CODE = {
    'stance_disturbance_contract.py': 'bc6ab0ccfa2f69b4abf85124458d3d0a2a2c2713d838ed83dfb6ad491c980a21',
    'stance_disturbance_fixture.py': '8cf75fdd0d5428415a4a3abef08f6e2b0ffad29cfcbd87133ef2d1729fc975cd',
    'stance_disturbance_probe.py': '5ee377dfb410013f5bd7349ef21e446141efee614d956d0d119aa767ee6f025c',
    'stance_warp_runtime.py': '56d84e7d9c4b84a21244a0706cec074411ce501cd6514cf00845eba2c3a21aaa',
    'stance_warp_integrator.py': '66ef13d3d1a2b76e436d4e4beadd4dac8ace6cf121b78b6ed70a10d8f1bfc799',
    'stance_control_state.py': '7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9',
    'stance_transition.py': 'fa9d809b8c67568bc21137add70706a70b892e838fec1088185301a336edfa8a',
    'stance_forward_probe.py': 'c5502f027121f06643a58299408a8d7a0fcfc77f2181cf9bea3cf1184bc0d63c',
}


def check_window(*, launching=False, now=None):
    remaining = contract.CUTOFF-(time.time() if now is None else now)
    reserve = contract.PROBE_SERVICE_SECONDS+contract.CLOSEOUT_SECONDS+contract.MARGIN_SECONDS if launching else 0
    require(remaining > reserve, 'complete D1 timing probe and closeout before fixed 08:00 Shanghai cutoff')


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-d1-frozen-recovery-probe-'+source[:12])


def service_name(source):
    output_path(source)
    return 'microduck-wsl-d1-recovery-probe-'+source[:12]+'.service'


def force_prerequisite():
    for name, digest in D0_CODE.items():
        require(host.digest(host.ROOT/'src/mjlab_microduck'/name) == digest,
                'unchanged qualified force/nominal primitive '+name)
    retained = {}
    for name, digest in D0_FILES.items():
        raw = files.file_bytes(D0_ROOT/name, limit=16*1024*1024)
        require(sha256(raw).hexdigest() == digest, 'exact retained D0 artifact '+name)
        retained[name] = raw
    report = files.parse(retained['report.json']); launch = files.parse(retained['launch.json'])
    require(report['decision'] == 'single-substep-force-path-replayed'
            and report['source'] == launch['source'] == D0_SOURCE
            and report['launch_sha256'] == D0_FILES['launch.json']
            and report['child']['returncode'] == 0
            and all(report[k] is False for k in contract.FALSE_FLAGS if k in report),
            'completed diagnostic-only D0 prerequisite')
    # Artifact hashes and primitive source pins precede any tensor loading.
    value = torch.load(io.BytesIO(retained['capture.pt']), map_location='cpu', weights_only=True)
    started = time.monotonic(); replay = force_fixture.replay(value)
    require(time.monotonic()-started < d0.contract.CPU_QUAL_SECONDS
            and replay == report['replay'] and replay['cuda_initialized'] is False,
            'fresh unchanged D0 physics replay before D1 GPU entry')
    return dict(source=D0_SOURCE, files=dict(D0_FILES), code=dict(D0_CODE), replay=replay)


def selected_checkpoint(authentication):
    auth.validate(authentication, authentication['source'], authentication['retained_training'])
    require(authentication['training_seed'] == contract.TRAINING_SEED, 'preselected recovery training seed')
    saved = next(c for c in authentication['retained_training']['checkpoints']
                 if c['identity']['iteration'] == contract.ITERATION)
    require(saved['file'] == contract.CHECKPOINT_FILE and saved['sha256'] == contract.CHECKPOINT_SHA256,
            'preselected recovery checkpoint before outcomes')
    root = evaluation.lean.output_path(auth.TRAINING_SOURCE, evaluation.lean.PACKED_REPLICATION,
                                      contract.TRAINING_SEED)
    raw = files.file_bytes(root/saved['file'], limit=evaluation.checkpoint.LIMIT)
    require(sha256(raw).hexdigest() == contract.CHECKPOINT_SHA256, 'original selected checkpoint bytes')
    return saved, raw


def launch_plan(source, authentication, qualification_sha):
    files.hex_id(qualification_sha, 64)
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'explicit unchanged WSL recovery profile')
    profile.checked_receipt()
    inputs = host.identity(source)
    auth.validate(authentication, source, authentication['retained_training'])
    saved, _ = selected_checkpoint(authentication)
    binding = force_fixture.compiled_binding(force_fixture.build_entity().compile())
    return dict(protocol=contract.PROTOCOL, source=source, inputs=inputs,
        declaration=contract.declaration(source, binding), selected_checkpoint=saved,
        historical_training_authentication=authentication,
        nominal_closeout=d0.nominal_closeout(), force_prerequisite=force_prerequisite(),
        cpu_math_profile=profile.expected_receipt(), preserved_filmbrain=d0.filmbrain_state(),
        cpu_qualification_sha256=qualification_sha, cpu_qualification_ticks=CPU_QUAL_TICKS,
        cpu_qualification_seconds=CPU_QUAL_SECONDS, **contract.FALSE_FLAGS)


def restored_actor(raw, identity):
    return evaluation.checkpoint.load_lean_replication_evaluation(
        raw, contract.CHECKPOINT_SHA256, identity)[0]


def seed_reset():
    """Seed only this owned evaluator process, never an unrelated workload."""
    random.seed(contract.EVALUATION_SEED)
    np.random.seed(contract.EVALUATION_SEED)
    torch.manual_seed(contract.EVALUATION_SEED)


def prepare(source):
    check_window(launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only D1 preparation')
    # Host/source/stack and original AVX2 authentication precede portable capture.
    host.identity(source); profile.checked_receipt()
    authentication = auth.run(source, auth.TRAINING_SOURCE, contract.TRAINING_SEED)
    initial = launch_plan(source, authentication, '0'*64)
    saved, checkpoint_raw = selected_checkpoint(authentication)
    from mjlab_microduck.stance_recovery_runtime import RecoveryRuntime
    root = files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    started = time.monotonic()
    seed_reset()
    actor = restored_actor(checkpoint_raw, saved['identity'])
    env = RecoveryRuntime([contract.PROBE_CASE], device='cpu')
    value = evidence.collect(env, actor, initial['declaration'], saved['identity'],
        deadline_monotonic=started+CPU_QUAL_SECONDS, policy_tick_limit=CPU_QUAL_TICKS)
    raw = evidence.encode(value)
    replay = evidence.verify(raw, sha256(raw).hexdigest(), checkpoint_raw, initial['declaration'])
    require(replay['pulse']['complete_pulse_delivery'] and replay['pulse']['checked_physics_steps'] >= 510
            and replay['actor_replay_max_abs_error'] == 0.
            and not torch.cuda.is_initialized(), 'complete CPU pulse and exact actor/control qualification before GPU')
    elapsed = time.monotonic()-started
    require(0 < elapsed < CPU_QUAL_SECONDS, 'bounded actual D1 CPU prefix qualification')
    smoke.write_bytes(root/'checkpoint.pt', checkpoint_raw)
    smoke.write_bytes(root/'cpu-prefix.pt', raw)
    qualification = dict(protocol='football-b1d-cpu-policy-pulse-qualification-v1', source=source,
        prefix_sha256=sha256(raw).hexdigest(), prefix_bytes=len(raw), elapsed_seconds=elapsed,
        checkpoint_sha256=contract.CHECKPOINT_SHA256, cpu_math_profile=profile.checked_receipt(),
        replay=replay, cuda_initialized=False)
    files.write_json(root/'cpu-qualification.json', qualification)
    launch = launch_plan(source, authentication, host.digest(root/'cpu-qualification.json'))
    require({k: v for k, v in launch.items() if k != 'cpu_qualification_sha256'} ==
            {k: v for k, v in initial.items() if k != 'cpu_qualification_sha256'}, 'unchanged CPU-qualified D1 inputs')
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    files.hex_id(launch_sha, 64); root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent D1 launch byte hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    authentication = auth.run(source, auth.TRAINING_SOURCE, contract.TRAINING_SEED)
    qraw = files.file_bytes(root/'cpu-qualification.json')
    require(launch == launch_plan(source, authentication, sha256(qraw).hexdigest()),
            'unchanged exact D1 host/source/archive/plant/profile/force qualification')
    q = files.parse(qraw); raw = files.file_bytes(root/'cpu-prefix.pt', limit=contract.CAPTURE_LIMIT)
    checkpoint_raw = files.file_bytes(root/'checkpoint.pt', limit=evaluation.checkpoint.LIMIT)
    require(set(q) == {'protocol', 'source', 'prefix_sha256', 'prefix_bytes', 'elapsed_seconds',
        'checkpoint_sha256', 'cpu_math_profile', 'replay', 'cuda_initialized'}
        and q['protocol'] == 'football-b1d-cpu-policy-pulse-qualification-v1' and q['source'] == source
        and q['prefix_sha256'] == sha256(raw).hexdigest() and q['prefix_bytes'] == len(raw)
        and q['checkpoint_sha256'] == sha256(checkpoint_raw).hexdigest() == contract.CHECKPOINT_SHA256
        and q['cpu_math_profile'] == profile.expected_receipt() and q['cuda_initialized'] is False
        and type(q['elapsed_seconds']) is float and 0 < q['elapsed_seconds'] < CPU_QUAL_SECONDS,
        'authenticated CPU pulse qualification before tensor loading')
    replay = evidence.verify(raw, q['prefix_sha256'], checkpoint_raw, launch['declaration'])
    require(replay == q['replay'] and replay['pulse']['complete_pulse_delivery']
            and replay['actor_replay_max_abs_error'] == 0., 'reexecuted exact portable actor/control/pulse checks')
    return launch


def child(source, launch_sha, fd, started):
    require(type(started) is float and 0 < time.monotonic()-started < contract.PROBE_CHILD_SECONDS,
            'bounded same-clock D1 child entry')
    check_window(launching=True); smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    require(time.monotonic()-started < 180, 'D1 CPU entry leaves fixed capture and closeout reserves')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'only inherited-lease D1 child may use CUDA0')
    host.wait_idle(); root = output_path(source)
    checkpoint_raw = files.file_bytes(root/'checkpoint.pt', limit=evaluation.checkpoint.LIMIT)
    seed_reset()
    actor = restored_actor(checkpoint_raw, launch['selected_checkpoint']['identity'])
    from mjlab_microduck.stance_recovery_runtime import RecoveryRuntime
    built = time.monotonic(); env = RecoveryRuntime([contract.PROBE_CASE], device='cuda:0')
    construction = time.monotonic()-built
    value = evidence.collect(env, actor, launch['declaration'], launch['selected_checkpoint']['identity'],
        deadline_monotonic=started+contract.PROBE_CHILD_SECONDS-120)
    require(time.monotonic()-started < contract.PROBE_CHILD_SECONDS-60, 'fixed D1 serialization closeout reserve')
    serial_started = time.monotonic(); raw = evidence.encode(value)
    smoke.write_bytes(root/'capture.pt', raw)
    capture = dict(protocol=contract.PROTOCOL, source=source, launch_sha256=launch_sha,
        capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
        backend=value['backend'], construction_seconds=construction,
        collection=value['collection'], serialization_seconds=time.monotonic()-serial_started,
        elapsed_seconds=time.monotonic()-started, optimizer_steps=0, **contract.FALSE_FLAGS)
    files.write_json(root/'capture.json', capture)


def replay_capture(source, launch, launch_sha):
    root = output_path(source); record = files.parse(files.file_bytes(root/'capture.json'))
    raw = files.file_bytes(root/'capture.pt', limit=contract.CAPTURE_LIMIT)
    require(record['protocol'] == contract.PROTOCOL and record['source'] == source
            and record['launch_sha256'] == launch_sha and record['capture_sha256'] == sha256(raw).hexdigest()
            and record['capture_bytes'] == len(raw) and record['backend'] == dict(torch_device='cuda:0', warp_is_cuda=True)
            and record['optimizer_steps'] == 0 and all(record[k] is False for k in contract.FALSE_FLAGS),
            'authenticated actual diagnostic CUDA capture before tensor loading')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'independent CPU-only frozen actor/control/pulse rescore')
    checkpoint_raw = files.file_bytes(root/'checkpoint.pt', limit=evaluation.checkpoint.LIMIT)
    started = time.monotonic()
    score = evidence.verify(raw, record['capture_sha256'], checkpoint_raw, launch['declaration'])
    require(score['case'] == contract.PROBE_CASE and score['collection'] == record['collection'],
            'retained timing matches complete selected case payload')
    for key in ('construction_seconds', 'serialization_seconds', 'elapsed_seconds'):
        require(type(record[key]) is float and math.isfinite(record[key])
                and 0 < record[key] < contract.PROBE_CHILD_SECONDS,
                'bounded retained D1 timing '+key)
    require(record['serialization_seconds'] < 60
            and record['construction_seconds']+record['collection']['elapsed_seconds']+
                record['serialization_seconds'] <= record['elapsed_seconds'],
            'nonoverlapping measured case phases within child elapsed time')
    if score['numerical_diagnostic']['gates']['full_duration'] is not True:
        exc = ValueError('full-length D1 timing probe must complete before timing admission')
        exc.add_note(json.dumps(dict(collection=score['collection'],
            numerical_diagnostic=score['numerical_diagnostic']), sort_keys=True))
        raise exc
    return record, score, time.monotonic()-started


def supervise(source, launch_sha):
    started = time.monotonic(); root = output_path(source)
    report = dict(protocol=contract.PROTOCOL, source=source, launch_sha256=launch_sha,
        decision='frozen-recovery-probe-failed', **contract.FALSE_FLAGS)
    try:
        props = {k: host.read('systemctl', '--user', 'show', service_name(source), '-p', k, '--value')
                 for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState',
                           'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice')}
        require(props == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='16min', KillMode='control-group',
            ActiveState='active', MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10'),
            'independently capped D1 timing probe service')
        report['service_properties'] = props
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
                'CPU-only D1 timing supervisor')
        check_window(launching=True); launch = checked(source, launch_sha)
        require({p.name for p in root.iterdir()} == {'launch.json', 'checkpoint.pt', 'cpu-prefix.pt',
                    'cpu-qualification.json'}, 'one fresh CPU-qualified D1 probe attempt')
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            child_started = time.monotonic()
            command = [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                '--launch-sha256', launch_sha, '--lock-fd', str(fd), '--started-monotonic', repr(child_started)]
            env = files.child_environment(); env.update(profile.settings())
            def guard():
                check_window(); host.check_log(root/'child.log')
                require(host.read('git', 'rev-parse', 'HEAD') == source
                        and not host.read('git', 'status', '--porcelain'), 'frozen exact live D1 source')
                require(d0.filmbrain_state() == launch['preserved_filmbrain'], 'preserved FilmBrain state')
            # Reuse the existing *fixed* 900-second wrapper. The generic
            # supervised_process remains capped at 120 seconds, unchanged.
            require(contract.PROBE_CHILD_SECONDS == 900, 'unchanged fixed stance timing watchdog')
            report['child'] = files.supervised_stance_smoke(command, root/'child.log', cwd=host.ROOT,
                env=env, lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            require(checked(source, launch_sha) == launch, 'unchanged completed D1 source and qualification')
            capture, score, replay_seconds = replay_capture(source, launch, launch_sha)
            report.update(capture=capture, score=score, supervisor_replay_seconds=replay_seconds)
            report['idle_after'] = host.wait_idle()
        elapsed = time.monotonic()-started
        require(elapsed <= contract.PROBE_SERVICE_SECONDS, 'whole D1 timing service within fixed cap')
        report.update(decision='frozen-recovery-timing-probe-replayed', elapsed_seconds=elapsed,
            full_length_probe_completed=score['numerical_diagnostic']['gates']['full_duration'],
            files={p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        files.write_json(root/'report.json', report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    p.add_argument('--source', required=True); p.add_argument('--launch-sha256')
    p.add_argument('--lock-fd', type=int); p.add_argument('--started-monotonic', type=float)
    a = p.parse_args(argv)
    if a.mode == 'prepare': print(prepare(a.source))
    elif a.mode == 'supervise': supervise(a.source, a.launch_sha256)
    else: child(a.source, a.launch_sha256, a.lock_fd, a.started_monotonic)


if __name__ == '__main__':
    main()
