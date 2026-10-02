"""Bounded leased WSL D0 force fixture and fresh CPU solve replay; no training."""
import argparse
from hashlib import sha256
import io
import math
import os
import time

import torch

from mjlab_microduck import stance_disturbance_contract as contract
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_cpu_replay_profile as cpu_profile
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import require, canonical

files = host.supervisor
MODULE = 'mjlab_microduck.stance_disturbance_probe'
CAPTURE_LIMIT = 16*1024*1024
FULL_SOURCE = '0b73855a0dbddc7ae06303398177a706ac812dc6'
PREREQUISITES = {
    'artifacts/evaluations/stance-wsl-portable-full-evaluation-0b73855a0dbd/launch.json':
        '1284cd7e94f381d82e9fc0e7652b132ecf35d4bb53907aa3c2f7d33622f04b2f',
    'artifacts/evaluations/stance-wsl-portable-full-evaluation-0b73855a0dbd/report.json':
        'c6f2719da866d9452e2771fc15e6b5684b56bece2b3acc96b2579ac7af8aec44',
    'artifacts/tools/portable-full-closeout-0b73855a0dbd/replay.json':
        '69547a77cee34e208e366512fe247f194d34e0c78f02ae7321cbb00ddaf89ce2',
    'artifacts/tools/portable-full-closeout-0b73855a0dbd/inventory-1858f359f9b6.json':
        '69a5be8fb1e0863e20e576218992baedbe9219ba49f172fd31a8a95cde851519',
    'artifacts/tools/portable-full-closeout-0b73855a0dbd/inventory-closeout-1858f359f9b6.json':
        '2cbd2b0df5ff5efd4a21bf2616cff7c72dd934d269cbe24b8262cca5772b2299',
}


def nominal_closeout():
    rows = []
    for path, digest in PREREQUISITES.items():
        raw = files.file_bytes(host.ROOT/path, limit=4*1024*1024)
        require(sha256(raw).hexdigest() == digest, 'pinned complete nominal closeout '+path)
        rows.append(files.parse(raw))
    launch, report, replay, inventory, receipt = rows
    require(launch['source'] == FULL_SOURCE and report['decision'] == 'lean-replication-passed'
            and report['launch_sha256'] == PREREQUISITES[next(iter(PREREQUISITES))]
            and report['summary']['cases'] == 36 and report['summary']['complete_attempts'] == 4608,
            'complete nominal replication prerequisite')
    require(replay['source'] == FULL_SOURCE and replay['whole_plan_bundle_score_verified'] is True
            and replay['selected_compiled_plant_verified'] is True and replay['cuda_initialized'] is False
            and replay['summary'] == report['summary'], 'complete exact-source CPU nominal replay')
    require(inventory['source'] == FULL_SOURCE and inventory['file_count'] == 329
            and inventory['total_bytes'] == 22352528834 and inventory['byte_inventory_only'] is True
            and receipt['separate_readonly_rehash_verified'] is True,
            'complete retained byte-only nominal inventory')
    return dict(PREREQUISITES)


def filmbrain_state():
    return {unit: {key: host.read('systemctl', '--user', 'show', unit, '-p', key, '--value')
                  for key in ('ActiveState', 'MainPID', 'NRestarts')}
            for unit in ('recomo-filmbrain-observatory.service', 'recomo-filmbrain-video-playground.service')}


def check_window(*, launching=False, now=None):
    remaining = contract.CUTOFF-(time.time() if now is None else now)
    require(remaining > (contract.SERVICE_SECONDS+contract.CLOSEOUT_SECONDS+60 if launching else 0),
            'complete D0 fixture and closeout before fixed 08:00 Shanghai cutoff')


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-d0-force-fixture-'+source[:12])


def service_name(source):
    output_path(source)
    return 'microduck-wsl-d0-force-fixture-'+source[:12]+'.service'


def launch_plan(source, cpu_qualification_sha256):
    files.hex_id(cpu_qualification_sha256, 64)
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'explicit unchanged WSL profile')
    cpu_profile.checked_receipt()
    result = contract.plan(source, fixture.compiled_binding(fixture.build_entity().compile()))
    result.update(inputs=host.identity(source), nominal_closeout=nominal_closeout(),
                  cpu_math_profile=cpu_profile.expected_receipt(),
                  preserved_filmbrain=filmbrain_state(),
                  cpu_qualification_sha256=cpu_qualification_sha256,
                  independent_gpu_attestation=False, complete_binary_runtime_equivalence_verified=False)
    return result


def prepare(source):
    check_window(launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only D0 preparation')
    # Check real source/host/profile/nominal inputs before even CPU simulation.
    initial_plan = launch_plan(source, '0'*64)
    root = files.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    value = fixture.capture_cases(source, 'cpu')
    require(value['plan'] == {k: initial_plan[k] for k in value['plan']},
            'prequalified actual CPU fixture source/plant')
    replay = fixture.replay(value)
    require(replay['cuda_initialized'] is False and replay['maximum_solved_field_error'] == 0.,
            'exact same-backend CPU five-case qualification before GPU admission')
    buffer = io.BytesIO()
    torch.save(value, buffer)
    raw = buffer.getvalue()
    require(0 < len(raw) <= CAPTURE_LIMIT, 'bounded CPU fixture qualification bytes')
    smoke.write_bytes(root/'cpu-capture.pt', raw)
    elapsed = time.monotonic()-started
    require(0 < elapsed < contract.CPU_QUAL_SECONDS, 'bounded CPU physics qualification')
    qualification = dict(protocol='football-b1d-cpu-force-qualification-v1', source=source,
        capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
        fixture_plan_sha256=sha256(canonical(value['plan']).encode()).hexdigest(),
        cpu_math_profile=cpu_profile.checked_receipt(), replay=replay, elapsed_seconds=elapsed)
    files.write_json(root/'cpu-qualification.json', qualification)
    launch = launch_plan(source, host.digest(root/'cpu-qualification.json'))
    require({k: v for k, v in launch.items() if k != 'cpu_qualification_sha256'} ==
            {k: v for k, v in initial_plan.items() if k != 'cpu_qualification_sha256'},
            'unchanged preparation source/host/plant/nominal inputs')
    require(value['plan'] == {k: launch[k] for k in value['plan']}, 'qualified actual CPU plant')
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    files.hex_id(launch_sha, 64)
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent D0 launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    qualification_sha = host.digest(root/'cpu-qualification.json')
    require(launch == launch_plan(source, qualification_sha), 'unchanged D0 source/host/plant/closeout/plan')
    qualification = files.parse(files.file_bytes(root/'cpu-qualification.json'))
    raw = files.file_bytes(root/'cpu-capture.pt', limit=CAPTURE_LIMIT)
    require(set(qualification) == {'protocol', 'source', 'capture_sha256', 'capture_bytes',
                'fixture_plan_sha256', 'cpu_math_profile', 'replay', 'elapsed_seconds'}
            and qualification['protocol'] == 'football-b1d-cpu-force-qualification-v1'
            and qualification['source'] == source
            and qualification['capture_sha256'] == sha256(raw).hexdigest()
            and qualification['capture_bytes'] == len(raw)
            and qualification['cpu_math_profile'] == cpu_profile.expected_receipt()
            and type(qualification['elapsed_seconds']) is float
            and 0 < qualification['elapsed_seconds'] < contract.CPU_QUAL_SECONDS,
            'authenticated exact CPU physics qualification before tensor loading')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(value['backend'] == dict(torch_device='cpu', warp_is_cuda=False)
            and value['plan'] == {k: launch[k] for k in value['plan']}
            and qualification['fixture_plan_sha256'] == sha256(canonical(value['plan']).encode()).hexdigest(),
            'CPU qualified source/plant fixture')
    started = time.monotonic()
    replay = fixture.replay(value)
    require(time.monotonic()-started < contract.CPU_QUAL_SECONDS, 'bounded repeated CPU qualification')
    require(qualification['replay'] == replay and replay['maximum_solved_field_error'] == 0.
            and replay['cuda_initialized'] is False, 'reexecuted complete CPU qualification')
    return launch


def child(source, launch_sha, fd, started):
    require(type(started) is float and 0 < time.monotonic()-started < contract.CHILD_SECONDS,
            'same-clock bounded child entry')
    check_window(launching=True)
    smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    require(time.monotonic()-started < contract.CHILD_SECONDS-contract.GPU_CAPTURE_RESERVE-
            contract.CHILD_CLOSEOUT_RESERVE, 'CPU entry leaves declared GPU and closeout reserves')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'only leased D0 child may use CUDA0')
    host.wait_idle()
    value = fixture.capture_cases(source, 'cuda:0')
    require(value['plan'] == {k: launch[k] for k in value['plan']}, 'actual declared CUDA plant')
    require(time.monotonic()-started < contract.CHILD_SECONDS-contract.CHILD_CLOSEOUT_RESERVE,
            'D0 fixed child closeout reserve')
    buffer = io.BytesIO()
    torch.save(value, buffer)
    raw = buffer.getvalue()
    require(0 < len(raw) <= CAPTURE_LIMIT, 'bounded full force payload')
    root = output_path(source)
    smoke.write_bytes(root/'capture.pt', raw)
    record = dict(protocol=contract.PROTOCOL, source=source, launch_sha256=launch_sha,
        capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
        elapsed_seconds=time.monotonic()-started, backend=value['backend'],
        integration_steps=5, policy_inferences=0, optimizer_steps=0, **contract.NO_ADMISSION)
    files.write_json(root/'capture.json', record)


def replay_capture(root, launch, launch_sha):
    record = files.parse(files.file_bytes(root/'capture.json'))
    raw = files.file_bytes(root/'capture.pt', limit=CAPTURE_LIMIT)
    require(record['protocol'] == contract.PROTOCOL and record['source'] == launch['source']
            and record['launch_sha256'] == launch_sha
            and record['capture_sha256'] == sha256(raw).hexdigest()
            and record['capture_bytes'] == len(raw), 'authenticated D0 payload before tensor loading')
    require(record['backend'] == dict(torch_device='cuda:0', warp_is_cuda=True)
            and record['integration_steps'] == 5 and record['optimizer_steps'] == 0
            and record['policy_inferences'] == 0
            and all(record[k] is False for k in contract.NO_ADMISSION), 'actual diagnostic-only capture')
    require(type(record['elapsed_seconds']) is float and math.isfinite(record['elapsed_seconds'])
            and 0 < record['elapsed_seconds'] < contract.CHILD_SECONDS-10, 'bounded retained D0 capture time')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(value['plan'] == {k: launch[k] for k in value['plan']}
            and value['backend'] == record['backend'], 'source-bound actual force payload')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'independent CPU-only force solve replay')
    return fixture.replay(value)


def supervise(source, launch_sha):
    service_started = time.monotonic()
    props = {k: host.read('systemctl', '--user', 'show', service_name(source), '-p', k, '--value')
             for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState',
                       'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice')}
    require(props == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='3min', KillMode='control-group',
        ActiveState='active', MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10'),
        'independently bounded 180-second D0 service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only D0 supervisor')
    check_window(launching=True)
    launch = checked(source, launch_sha)
    root = output_path(source)
    require({p.name for p in root.iterdir()} == {'launch.json', 'cpu-capture.pt', 'cpu-qualification.json'},
            'one CPU-qualified fresh D0 attempt only')
    report = dict(protocol=contract.PROTOCOL, source=source, launch_sha256=launch_sha,
        decision='force-fixture-failed', service_properties=props, **contract.NO_ADMISSION,
        independent_gpu_attestation=False, complete_binary_runtime_equivalence_verified=False)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            started = time.monotonic()
            command = [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                '--launch-sha256', launch_sha, '--lock-fd', str(fd), '--started-monotonic', repr(started)]
            env = files.child_environment()
            env.update(cpu_profile.settings())
            def guard():
                check_window()
                host.check_log(root/'child.log')
                require(host.read('git', 'rev-parse', 'HEAD') == source
                        and not host.read('git', 'status', '--porcelain'), 'frozen live D0 source')
                require(filmbrain_state() == launch['preserved_filmbrain'], 'preserved FilmBrain state')
            report['child'] = files.supervised_process(command, root/'child.log', cwd=host.ROOT,
                env=env, lock_fd=fd, timeout=contract.CHILD_SECONDS, guard=guard)
            host.check_log(root/'child.log')
            require(checked(source, launch_sha) == launch, 'unchanged completed force inputs')
            report['replay'] = replay_capture(root, launch, launch_sha)
            report['idle_after'] = host.wait_idle()
        elapsed = time.monotonic()-service_started
        require(elapsed <= contract.SERVICE_SECONDS, 'whole D0 service within predeclared cap')
        report.update(decision='single-substep-force-path-replayed', elapsed_seconds=elapsed,
            files={p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()})
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        files.write_json(root/'report.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--launch-sha256')
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--started-monotonic', type=float)
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        print(prepare(args.source))
    elif args.mode == 'supervise':
        supervise(args.source, args.launch_sha256)
    else:
        child(args.source, args.launch_sha256, args.lock_fd, args.started_monotonic)


if __name__ == '__main__':
    main()
