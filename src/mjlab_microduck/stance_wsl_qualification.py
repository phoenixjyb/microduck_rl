"""Source-bound Blackwell integration and timing gate before replication.

900-second CUDA child, 960-second user service. Reuses the exact integration
cases and frozen-parent 2-warmup/8-measured-update throughput probe. This is not
a training checkpoint or a capability gate. No historical watchdog is widened.
"""
import argparse
from hashlib import sha256
import math
import os
import time

import torch

from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_lean_throughput as throughput
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

files = host.supervisor
MODULE = 'mjlab_microduck.stance_wsl_qualification'
PROTOCOL = 'football-b1n-wsl-qualification-v1'


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-qualification-'+source[:12])


def service_name(source):
    return 'microduck-wsl-qualification-'+source[:12]+'.service'


def plan(source, deadline):
    require(execution.PROFILE['name'] == execution.WSL, 'explicit authorized WSL profile')
    return dict(protocol=PROTOCOL, source=source, inputs=host.identity(source),
        training_budget=execution.training_budget(), timing_basis=validate_timing_basis(source),
        deadline_unix=deadline, child_seconds=900, service_seconds=960,
        collection_worlds=64, policy_ticks=24, warmup_updates=2, measured_updates=8,
        target_updates=256, frozen_parent_sha256=throughput.checkpoint.LEAN_PARENT_SHA256,
        acceptance_gates_changed=False, checkpoint_exported=False,
        physical_motion_authorized=False)


def prepare(source, deadline):
    throughput.check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only WSL preparation')
    launch = plan(source, deadline)
    parent = files.file_bytes(throughput.parent_path())
    require(sha256(parent).hexdigest() == launch['frozen_parent_sha256'], 'pinned parent')
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    root = output_path(source); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'parent.pt', parent)
    smoke.write_bytes(root/'runtime.json', runtime)
    launch['runtime_sha256'] = sha256(runtime).hexdigest()
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent qualification launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    expected = plan(source, launch['deadline_unix'])
    expected['runtime_sha256'] = host.digest(root/'runtime.json')
    require(launch == expected, 'unchanged qualification source/host/runtime')
    plant.checked_runtime(files.parse(files.file_bytes(root/'runtime.json')), source)
    throughput.load_parent(root)
    return launch


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    throughput.check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'explicit CUDA child')
    host.wait_idle()
    started = time.monotonic()
    integration = host.cases('cuda:0')
    integration['launch_sha256'] = launch_sha
    host.validate_payload(integration, launch_sha)
    root = output_path(source)
    files.write_json(root/'integration.json', integration)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    torch.manual_seed(523)
    env = WarpStanceRuntime(64, device='cuda:0')
    require(env.forward_graph is None and env.wp_device.is_cuda, 'actual eager CUDA physics')
    require(plant.describe(env.native) == plant.reference(), 'unchanged actual plant')
    parent = throughput.load_parent(root)
    setup = time.monotonic()-started
    collection = throughput.measure_collection(env, parent['actor'].eval())
    collection['qualification_setup_seconds'] = setup
    files.write_json(root/'collection.json', collection)
    checked(source, launch_sha)


def derive(collection, optimizer):
    require(collection['physics_device'] == 'cuda:0' and collection['optimizer_steps'] == 0
            and collection['worlds'] == 64 and collection['ticks_per_update'] == 24,
            'actual declared collection, not training')
    require(optimizer['device'] == 'cpu' and optimizer['stand_in'] is True
            and optimizer['physics_evidence'] is False, 'honest CPU optimizer timing stand-in')
    for component in (collection, optimizer):
        require(component['summarize'] == throughput.summarize(component['summarize']['series'])
                and component['summarize']['count'] == 8, 're-derived eight measured samples')
    for key in ('setup_seconds', 'qualification_setup_seconds'):
        value = collection[key]
        require(type(value) is float and math.isfinite(value) and value >= 0,
                'finite nonnegative measured setup component: '+key)
    caps = throughput.caps(collection['summarize'], optimizer['summarize'],
        float(collection['setup_seconds']+collection['qualification_setup_seconds']))
    budget = execution.training_budget()
    fits = caps['child_seconds'] <= budget['child_seconds'] and caps['service_seconds'] <= budget['service_seconds']
    return dict(caps=caps, declared_child_seconds=budget['child_seconds'],
        declared_service_seconds=budget['service_seconds'],
        decision='qualified-for-bounded-replication' if fits else 'timing-rejected-no-training',
        full_evaluation_timing_qualified=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)


def validate_timing_basis(source):
    """Authenticate the prior host measurement; never pretend it fitted 4090."""
    root = output_path(execution.WSL_TIMING_SOURCE)
    require(host.digest(root/'report.json') == execution.WSL_TIMING_REPORT, 'pinned original WSL timing report')
    report = files.parse(files.file_bytes(root/'report.json'))
    require(report['protocol'] == PROTOCOL and report['decision'] == 'timing-rejected-no-training',
            'retained original 4090-budget rejection')
    require(set(report['files']) == {p.name for p in root.iterdir()}-{'report.json'}, 'exact timing basis inventory')
    require(all(host.digest(root/name) == digest for name,digest in report['files'].items()), 'timing basis hashes')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    require(host.digest(root/'launch.json') == report['launch_sha256']
            and launch['source'] == execution.WSL_TIMING_SOURCE, 'timing basis source/launch')
    current = host.identity(source)
    require({k:v for k,v in launch['inputs'].items() if k != 'source'} ==
            {k:v for k,v in current.items() if k != 'source'}, 'same measured host/dependencies/assets')
    host.validate_payload(files.parse(files.file_bytes(root/'integration.json')), report['launch_sha256'])
    caps = derive(files.parse(files.file_bytes(root/'collection.json')),
                  files.parse(files.file_bytes(root/'optimizer.json')))['caps']
    require(caps == report['result']['caps'], 're-derived original timing maxima')
    rounded_service = math.ceil(caps['service_seconds']/60)*60
    budget = execution.training_budget()
    require((budget['child_seconds'],budget['service_seconds']) == (rounded_service-60,rounded_service),
            'new WSL cap transcribed from authenticated measurement')
    return dict(source=execution.WSL_TIMING_SOURCE, report_sha256=execution.WSL_TIMING_REPORT,
                measured_caps=caps, rounded_service_seconds=rounded_service)


def verify(source):
    root = output_path(source)
    report = files.parse(files.file_bytes(root/'report.json'))
    launch = checked(source, report['launch_sha256'])
    require(report['protocol'] == PROTOCOL, 'WSL qualification report')
    require(report['decision'] == 'qualified-for-bounded-replication', 'WSL training timing gate')
    require(set(report['files']) == {p.name for p in root.iterdir()}-{'report.json'}, 'exact qualification files')
    require(all(host.digest(root/name) == digest for name,digest in report['files'].items()),
            'qualification evidence hashes')
    host.validate_payload(files.parse(files.file_bytes(root/'integration.json')), report['launch_sha256'])
    derived = derive(files.parse(files.file_bytes(root/'collection.json')),
                     files.parse(files.file_bytes(root/'optimizer.json')))
    require(report['result'] == derived, 're-derived WSL timing decision')
    return dict(report_sha256=host.digest(root/'report.json'), launch_sha256=report['launch_sha256'],
                source=launch['source'], result=derived)


def supervise(source, launch_sha):
    launch = checked(source, launch_sha); root = output_path(source)
    throughput.check_window(launch['deadline_unix'], launching=True)
    actual = throughput.service_state(service_name(source))
    require(actual == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='16min',
                          KillMode='control-group', ActiveState='active'), 'independently timed WSL service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only supervisor')
    require({p.name for p in root.iterdir()} == {'parent.pt','runtime.json','launch.json'}, 'one fresh qualification')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed')
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+600 < launch['deadline_unix'], 'qualification closeout reserve')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live host/source drift')
            report['child'] = files.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log',
                cwd=host.ROOT, env=files.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            optimizer = throughput.measure_optimizer(throughput.load_parent(root))
            files.write_json(root/'optimizer.json', optimizer)
            report['result'] = derive(files.parse(files.file_bytes(root/'collection.json')), optimizer)
            report['decision'] = report['result']['decision']
            report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc,'__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json', report)
    verify(source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','supervise','child','verify'))
    p.add_argument('--source', required=True)
    p.add_argument('--deadline-unix', type=int)
    p.add_argument('--launch-sha256')
    p.add_argument('--lock-fd', type=int)
    args = p.parse_args()
    if args.mode == 'prepare': print(canonical(prepare(args.source, args.deadline_unix)))
    elif args.mode == 'supervise': supervise(args.source, args.launch_sha256)
    elif args.mode == 'child': child(args.source, args.launch_sha256, args.lock_fd)
    else: print(canonical(verify(args.source)))


if __name__ == '__main__': main()
