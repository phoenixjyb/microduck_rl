"""Capped native CPU2 stochastic transition capture and independent closeout.

This probe does not update, export a student, or qualify a full episode/reset,
GPU throughput, general recovery, football balance or physical motion.
"""
import argparse
from hashlib import sha256
import math
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_recovery_broad_screen as prerequisites
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_ppo_bridge as bridge
from mjlab_microduck import stance_recovery_ppo_trace as evidence
from mjlab_microduck.first_attempt_smoke import require

base = prerequisites.base
PROTOCOL = 'football-b1d-cpu-stochastic-ppo-transition-probe-v1'
MODULE = 'mjlab_microduck.stance_recovery_ppo_probe'
SERVICE_SECONDS, CLOSEOUT_SECONDS, MARGIN_SECONDS = 180, 120, 60
LAUNCH_RESERVE = SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN_SECONDS
PRE_REPORT_FILES = {'checkpoint.pt', 'launch.json', 'capture.pt', 'capture.json'}
COMPLETE_FILES = PRE_REPORT_FILES | {'report.json'}


def declaration(source):
    return base.schedule.declaration(source, 'dose', 'training', evidence.WORLD_CELLS)


def output_path(source):
    base.files.hex_id(source, 40)
    return base.host.ROOT / 'artifacts/evaluations' / (
        'stance-wsl-cpu-stochastic-ppo-transition-' + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ('run', 'closeout'), 'declared stochastic CPU probe service mode')
    return f'microduck-cpu-ppo-{mode}-{source[:12]}.service'


def service_properties(source, mode):
    seconds = {'run': SERVICE_SECONDS, 'closeout': CLOSEOUT_SECONDS}[mode]
    name = service_name(source, mode)
    properties = {key: base.host.read('systemctl', '--user', 'show', name, '-p', key, '--value')
                  for key in ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax',
                              'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    require(properties == dict(MainPID=str(os.getpid()), ActiveState='active',
        RuntimeMaxUSec=f'{seconds // 60}min', MemoryMax=str(2 * 1024**3),
        CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'),
        'exact independently capped stochastic CPU probe')
    running = base.host.read('systemctl', '--user', 'list-units', '--state=running',
                             '--no-legend', 'microduck*')
    require({line.split()[0] for line in running.splitlines()} == {name},
            'only this owned Duck service runs during stochastic CPU probe')
    return properties


def _capture_decision(collection):
    require(type(collection) is dict and type(collection.get('accepted_complete')) is bool,
            'typed retained collection outcome')
    return ('cpu-ppo-transition-collected-awaiting-independent-replay'
            if collection['accepted_complete'] else 'cpu-ppo-transition-rejected-prefix-retained')


def _recorded_run_properties(properties):
    require(type(properties) is dict and type(properties.get('MainPID')) is str
            and properties['MainPID'].isdecimal() and int(properties['MainPID']) > 0,
            'recorded owned run PID')
    require(properties == dict(MainPID=properties['MainPID'], ActiveState='active',
        RuntimeMaxUSec='3min', MemoryMax=str(2 * 1024**3), CPUQuotaPerSecUSec='2s',
        Nice='10', KillMode='control-group'), 'exact recorded run service cap')


def run(source):
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE)
    properties = service_properties(source, 'run')
    context = prerequisites._context(source)
    previous, checkpoint_raw = prerequisites.prerequisites()
    schedule = declaration(source)
    root = base.files.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    base.retained.write_capture(root / 'checkpoint.pt', checkpoint_raw)
    report = dict(protocol=PROTOCOL, source=source, decision='cpu-ppo-transition-process-failed',
        optimizer_steps=0, training_update_performed=False, service_properties=properties,
        postchecks_passed=False, **base.baseline.FALSE_FLAGS)
    try:
        from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
        # These are owned training initialization streams, not held-out seed 671.
        random.seed(evidence.SEED)
        np.random.seed(evidence.SEED)
        torch.random.default_generator.manual_seed(evidence.SEED)
        env = ScheduledRecoveryRuntime(schedule, device='cpu')
        learner = bridge.RecoveryPPOBridge(checkpoint_raw, env, schedule, seed=evidence.SEED)
        launch = dict(protocol=PROTOCOL, source=source, **context,
            prerequisites=previous, campaign_window=window.declaration(),
            declaration=schedule, compiled_plant=env.binding,
            trace_binding=evidence.binding(schedule, env.binding, context['cpu_math_profile']),
            parent_checkpoint_sha256=sha256(checkpoint_raw).hexdigest(),
            learner_seed=evidence.SEED, worlds=2, horizon=evidence.HORIZON,
            collection_seconds=evidence.WALL_LIMIT, capture_bytes_limit=evidence.LIMIT,
            service_seconds=SERVICE_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
            launch_reserve_seconds=LAUNCH_RESERVE, service_properties=properties,
            optimizer_steps=0, training_update_performed=False, **base.baseline.FALSE_FLAGS)
        base.files.write_json(root / 'launch.json', launch)
        # Keep retention/serialization room inside the separately enforced service cap.
        require(time.monotonic() - started < SERVICE_SECONDS - evidence.WALL_LIMIT - 20,
                'reserve complete bounded collection plus raw retention')
        value = evidence.collect(learner, checkpoint_raw,
                                 deadline_monotonic=time.monotonic() + evidence.WALL_LIMIT)
        raw = evidence.encode(value)
        base.retained.write_capture(root / 'capture.pt', raw)
        capture = dict(protocol=PROTOCOL, source=source, binding=value['binding'],
            capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
            collection=value['collection'], storage_step=value['optimizer']['storage_step'],
            optimizer_steps=0, training_update_performed=False, **base.baseline.FALSE_FLAGS)
        base.files.write_json(root / 'capture.json', capture)
        # Link durable raw evidence before any later context/deadline judgment.
        report.update(launch_sha256=base.host.digest(root / 'launch.json'), capture=capture,
                      decision=_capture_decision(value['collection']))
        require(prerequisites._context(source) == context
                and service_properties(source, 'run') == properties and not torch.cuda.is_initialized()
                and time.monotonic() - started < SERVICE_SECONDS,
                'unchanged source/profile/services and bounded CPU capture')
        window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
        report.update(postchecks_passed=True,
            source_unchanged=True, filmbrain_unchanged=True, protected_services_inactive=True)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      error_notes=getattr(error, '__notes__', []))
        if 'capture' in report:
            report['decision'] = 'cpu-ppo-transition-postcheck-failed-retained'
        raise
    finally:
        report['cuda_initialized'] = bool(torch.cuda.is_initialized())
        report['elapsed_seconds'] = float(time.monotonic() - started)
        report['files'] = {path.name: base.host.digest(path) for path in sorted(root.iterdir())
                           if path.is_file()}
        base.files.write_json(root / 'report.json', report)
    return dict(output=str(root), launch_sha256=report['launch_sha256'], decision=report['decision'])


def checked(source, launch_sha256):
    base.files.hex_id(launch_sha256, 64)
    root = output_path(source)
    launch_raw = base.files.file_bytes(root / 'launch.json')
    require(sha256(launch_raw).hexdigest() == launch_sha256, 'whole launch hash before parse')
    launch = base.files.parse(launch_raw)
    _recorded_run_properties(launch['service_properties'])
    context = prerequisites._context(source)
    previous, checkpoint_raw = prerequisites.prerequisites()
    require(launch['protocol'] == PROTOCOL and launch['source'] == source
            and {key: launch[key] for key in context} == context
            and launch['prerequisites'] == previous
            and launch['campaign_window'] == window.declaration()
            and launch['declaration'] == declaration(source)
            and launch['trace_binding'] == evidence.binding(declaration(source),
                launch['compiled_plant'], context['cpu_math_profile'])
            and launch['parent_checkpoint_sha256'] == sha256(checkpoint_raw).hexdigest()
            and launch['learner_seed'] == evidence.SEED and launch['worlds'] == 2
            and launch['horizon'] == evidence.HORIZON
            and launch['collection_seconds'] == evidence.WALL_LIMIT
            and launch['capture_bytes_limit'] == evidence.LIMIT
            and launch['service_seconds'] == SERVICE_SECONDS
            and launch['closeout_seconds'] == CLOSEOUT_SECONDS
            and launch['launch_reserve_seconds'] == LAUNCH_RESERVE
            and launch['optimizer_steps'] == 0 and launch['training_update_performed'] is False
            and all(launch[key] is False for key in base.baseline.FALSE_FLAGS)
            and base.files.file_bytes(root / 'checkpoint.pt',
                limit=base.retained.evaluation.checkpoint.LIMIT) == checkpoint_raw,
            'exact pinned parent, schedule, source, CPU profile and unupdated launch')
    return launch, checkpoint_raw


def _check_report(report, capture, source, launch_sha256, run_properties):
    _recorded_run_properties(run_properties)
    postchecks_passed = report.get('postchecks_passed')
    expected_decision = (_capture_decision(capture['collection']) if postchecks_passed
                         else 'cpu-ppo-transition-postcheck-failed-retained')
    require(report.get('protocol') == PROTOCOL and report.get('source') == source
            and report.get('launch_sha256') == launch_sha256
            and report.get('capture') == capture
            and type(postchecks_passed) is bool
            and report.get('decision') == expected_decision
            and report.get('service_properties') == run_properties
            and type(report.get('elapsed_seconds')) is float
            and math.isfinite(report['elapsed_seconds']) and report['elapsed_seconds'] >= 0
            and report.get('optimizer_steps') == 0
            and report.get('training_update_performed') is False
            and report.get('cuda_initialized') is False
            and all(report.get(key) is False for key in base.baseline.FALSE_FLAGS)
            and set(report.get('files', {})) == PRE_REPORT_FILES,
            'complete unupdated capture report, including rejected retained prefixes')
    if postchecks_passed:
        require(not any(key in report for key in ('error', 'error_type', 'error_notes'))
                and all(report.get(key) is True for key in ('source_unchanged',
                    'filmbrain_unchanged', 'protected_services_inactive')),
                'successful run postchecks without hidden failure')
    else:
        require(type(report.get('error_type')) is str and bool(report['error_type'])
                and type(report.get('error')) is str,
                'retained postcheck failure is explicit and cannot qualify')
    return bool(postchecks_passed and report['elapsed_seconds'] < SERVICE_SECONDS)


def closeout(source, launch_sha256):
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS)
    properties = service_properties(source, 'closeout')
    launch, checkpoint_raw = checked(source, launch_sha256)
    root = output_path(source)
    report_raw = base.files.file_bytes(root / 'report.json')
    report = base.files.parse(report_raw)
    capture = base.files.parse(base.files.file_bytes(root / 'capture.json'))
    run_admissible = _check_report(report, capture, source, launch_sha256,
                                   launch['service_properties'])
    require({path.name for path in root.iterdir()} == COMPLETE_FILES,
            'exact static stochastic CPU capture inventory')
    inventory = {}
    for name, digest in report['files'].items():
        path = root / name
        require(base.host.digest(path) == digest, 'whole retained file hash ' + name)
        inventory[name] = dict(sha256=digest, bytes=path.stat().st_size)
    raw = base.files.file_bytes(root / 'capture.pt', limit=evidence.LIMIT)
    require(capture['protocol'] == PROTOCOL and capture['source'] == source
            and capture['binding'] == launch['trace_binding']
            and capture['capture_bytes'] == len(raw)
            and capture['optimizer_steps'] == 0 and capture['training_update_performed'] is False
            and all(capture[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact native CPU2 capture metadata')
    score = evidence.verify(raw, capture['capture_sha256'], checkpoint_raw,
                            launch['declaration'], launch['compiled_plant'])
    context = prerequisites._context(source)
    require(score['collection'] == capture['collection']
            and score['storage_step'] == capture['storage_step']
            and context == {key: launch[key] for key in context}
            and service_properties(source, 'closeout') == properties
            and not torch.cuda.is_initialized() and time.monotonic() - started < CLOSEOUT_SECONDS,
            'independent retained transition score and unchanged bounded CPU closeout')
    window.check()
    inventory['report.json'] = dict(sha256=sha256(report_raw).hexdigest(), bytes=len(report_raw))
    result = closeout_result(source, launch_sha256, report_raw, capture, score, inventory,
        report, run_admissible, properties, float(time.monotonic() - started))
    base.files.write_json(root / 'independent-closeout.json', result)
    return result


def closeout_result(source, launch_sha256, report_raw, capture, score, inventory,
                    report, run_admissible, properties, elapsed):
    """Receipt construction only; no standalone replay or native attestation."""
    return dict(protocol=PROTOCOL, source=source, launch_sha256=launch_sha256,
        report_sha256=sha256(report_raw).hexdigest(), capture_sha256=capture['capture_sha256'],
        decision=('cpu-ppo-transition-integration-qualified' if
            run_admissible and score['complete_two_world_transition_qualification']
            else 'cpu-ppo-transition-integration-rejected'),
        replay=score, files_rehashed=inventory, independent_cpu_score_performed=True,
        run_postchecks_passed=report['postchecks_passed'], run_admissible=run_admissible,
        run_error=({key: report[key] for key in ('error_type', 'error', 'error_notes')
                    if key in report} if not report['postchecks_passed'] else None),
        optimizer_steps=0, training_update_performed=False, cuda_initialized=False,
        full_episode_timeout_reset_qualified=False, finite_optimizer_step_qualified=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, service_properties=properties,
        elapsed_seconds=elapsed, **base.baseline.FALSE_FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'closeout'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--launch-sha256')
    args = parser.parse_args(argv)
    if args.mode == 'run':
        print(run(args.source))
    else:
        require(args.launch_sha256 is not None, 'independent closeout needs launch digest')
        print(closeout(args.source, args.launch_sha256))


if __name__ == '__main__':
    main()
