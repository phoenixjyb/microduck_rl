"""One WSL packed held-out timing case; no full evaluator or skill admission.

Predeclared in docs/experiments/2026-10-01-wsl-packed-evaluation-probe.md.
The 600/960-second probe bounds do not derive from training timing. All twelve
held-out cases remain blocked until a later reviewed measured-cap declaration.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import math
import os
import time

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_solved_field_check as checker
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_wsl_qualification as qualification
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = smoke.host, smoke.supervisor
MODULE = 'mjlab_microduck.stance_packed_evaluation_probe'
DECLARATION = evaluation.PACKED_PROBE
PROTOCOL = DECLARATION['protocol']
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT_SECONDS = 600, 960, 600
MARGIN_SECONDS, MAX_PROBE_WINDOW, MAX_EVALUATION_WINDOW = 60, 3600, 7200
PROBE_TRAINING_SEED = lean.PACKED_REPLICATION['seeds'][0]


def probe_seed(seed):
    require(type(seed) is int and seed == PROBE_TRAINING_SEED, 'fixed first training seed for packed timing probe')
    return seed


def check_window(deadline, *, launching=False):
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'exact packed probe WSL profile')
    require(type(deadline) is int and deadline <= qualification.PACKED_CUTOFF,
            'packed probe wholly before authorized October 1 cutoff')
    remaining = deadline-time.time()
    require(math.isfinite(remaining) and 0 < remaining <= MAX_PROBE_WINDOW, 'fresh bounded probe window')
    if launching:
        require(remaining > SERVICE_SECONDS+CLOSEOUT_SECONDS+MARGIN_SECONDS,
                'whole packed probe plus closeout reserve')


def output_path(source, seed):
    files.hex_id(source, 40); probe_seed(seed)
    return host.ROOT/'artifacts/evaluations'/f'stance-wsl-packed-eval-probe-{source[:12]}-seed-{seed}'


def service_name(source, seed):
    output_path(source, seed)
    return f'microduck-wsl-packed-eval-probe-{source[:12]}-seed-{seed}.service'


def plan(source, inputs, runtime_sha, retained, deadline):
    files.hex_id(source, 40); files.hex_id(runtime_sha, 64)
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'exact packed probe WSL profile')
    require(type(deadline) is int and deadline <= qualification.PACKED_CUTOFF,
            'packed probe wholly before authorized October 1 cutoff')
    files.hex_id(retained['source'], 40); files.hex_id(retained['report_sha256'], 64)
    require([c['identity']['iteration'] for c in retained['checkpoints']] == list(lean.CHECKPOINTS),
            'all four common checkpoints in completed packed archive')
    saved = retained['checkpoints'][-1]; meta = saved['identity']
    lean.checkpoint.validate_identity(meta, evaluation=lean.PACKED_REPLICATION['purpose'])
    require(meta['source'] == retained['source'] and meta['iteration'] == evaluation.PROBE_ITERATION
            and saved['file'] == 'model_255.pt', 'fixed final packed checkpoint')
    probe_seed(meta['training_seed']); files.hex_id(saved['sha256'], 64)
    binding = dict(protocol=trace.PACKED_PROBE_PROTOCOL, source=source, runtime_sha256=runtime_sha,
        checkpoint_sha256=saved['sha256'], checkpoint_iteration=evaluation.PROBE_ITERATION,
        evaluation_seed=evaluation.PROBE_SEED, worlds=evaluation.WORLDS, capture_device='cuda:0',
        solved_field_check='packed', checker_sha256=host.digest(checker.__file__))
    binding['launch_sha256'] = sha256(evaluation.bundle.launch_bytes(binding, meta)).hexdigest()
    return dict(protocol=PROTOCOL, mode='probe', source=source, inputs=inputs,
        runtime_sha256=runtime_sha, retained_training=deepcopy(retained), deadline_unix=deadline,
        cases=[dict(name='packed-probe-255-seed-541', binding=binding, checkpoint=deepcopy(saved))],
        worlds_per_case=evaluation.WORLDS, cases_required=1, attempts_required=evaluation.WORLDS,
        policy_ticks=evaluation.POLICY_TICKS, child_timeout_seconds=CHILD_SECONDS,
        service_timeout_seconds=SERVICE_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
        watchdog_margin_seconds=MARGIN_SECONDS, memory_max_bytes=6*1024**3,
        cpu_quota_per_sec_usec=2_000_000, nice=10, actor_device='cpu', physics_device='cuda:0',
        forward_graph=False, solved_field_check='packed', checker_sha256=binding['checker_sha256'],
        optimizer_steps=0, reset_policy='one-nominal-first-attempt-no-auto-reset',
        seed_interpretation='nominal-repeatability-not-randomized-generalization',
        tilt_gate_rad=evaluation.TILT_GATE_RAD, tilt_gate_relaxed=False,
        full_evaluation_enabled=False, checkpoint_admitted=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)


def prepare(source, training_source, seed, deadline):
    probe_seed(seed)
    check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only packed evaluation probe preparation')
    inputs = host.identity(source)
    retained = evaluation.training_inputs(DECLARATION, seed, training_source)
    training_launch = files.parse(files.file_bytes(lean.output_path(training_source, lean.PACKED_REPLICATION, seed)/'launch.json'))
    require({k: v for k, v in training_launch['inputs'].items() if k != 'source'} ==
            {k: v for k, v in inputs.items() if k != 'source'}
            and training_launch['checker_sha256'] == host.digest(checker.__file__),
            'same frozen host/stack/plant/checker as completed packed training')
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, inputs, sha256(runtime).hexdigest(), retained, deadline)
    root = files.native._plain_path(output_path(source, seed)); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runtime.json', runtime); files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source, seed), launch_sha256=host.digest(root/'launch.json'))


def checked(source, seed, launch_sha):
    probe_seed(seed)
    root = output_path(source, seed)
    require(host.digest(root/'launch.json') == launch_sha, 'independent packed probe launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    retained = evaluation.training_inputs(DECLARATION, seed, launch['retained_training']['source'])
    require(launch == plan(source, host.identity(source), host.digest(root/'runtime.json'),
            retained, launch['deadline_unix']), 'unchanged packed probe source/host/archive/plan')
    plant.checked_runtime(files.parse(files.file_bytes(root/'runtime.json')), source)
    return launch


def derive(measurements, replay_seconds, observed_seconds):
    require(measurements.get('policy_ticks') == evaluation.POLICY_TICKS
            and measurements.get('stop_reason') == evaluation.VALID_STOP_REASON
            and measurements.get('checkpoint_iteration') == evaluation.PROBE_ITERATION
            and measurements.get('evaluation_seed') == evaluation.PROBE_SEED
            and measurements.get('cases_projected') == evaluation.CASES,
            'only the complete fixed full-length probe sizes the matrix')
    evaluation.derive_caps(measurements)  # Original finite-positive component checks.
    for value in (replay_seconds, observed_seconds):
        require(type(value) is float and math.isfinite(value) and value > 0, 'positive observed probe timing')
    prelude, env, case = (measurements[k] for k in ('prelude_seconds', 'env_seconds', 'case_seconds'))
    measured = prelude+env+case+replay_seconds
    require(observed_seconds >= measured, 'nonoverlapping observed timing components')
    overhead = observed_seconds-measured
    # A one-case observation cannot separate repeated loop-tail/receipt/GC work
    # from one-off supervisor setup. Conservatively repeat ALL residual work.
    projected = prelude+evaluation.CASES*(env+case+replay_seconds+overhead)
    require(math.isfinite(projected), 'finite full evaluation projection')
    service = math.ceil(evaluation.PROBE_SAFETY*projected)
    return dict(cases=evaluation.CASES, attempts=evaluation.ATTEMPTS,
        supervisor_replay_seconds=replay_seconds, observed_probe_seconds=observed_seconds,
        unattributed_overhead_seconds=overhead, overhead_projection_count=evaluation.CASES,
        predicted_seconds=projected,
        safety_factor=evaluation.PROBE_SAFETY, service_seconds=service,
        child_seconds=service-MARGIN_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
        watchdog_margin_seconds=MARGIN_SECONDS, max_window_seconds=MAX_EVALUATION_WINDOW,
        fits_declared_wsl_window=service+CLOSEOUT_SECONDS+MARGIN_SECONDS <= MAX_EVALUATION_WINDOW,
        full_evaluation_enabled=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)


def child_command(source, seed, launch_sha, fd, started):
    files.hex_id(source, 40); files.hex_id(launch_sha, 64); probe_seed(seed)
    require(type(fd) is int and fd >= 0 and type(started) is float and math.isfinite(started) and started > 0,
            'owned child lease and monotonic start')
    return [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
        '--seed', str(seed), '--launch-sha256', launch_sha, '--lock-fd', str(fd),
        '--started-monotonic', repr(started)]


def child(source, seed, launch_sha, fd, started):
    entered = time.monotonic()
    require(type(started) is float and math.isfinite(started) and 0 < entered-started < CHILD_SECONDS,
            'same-clock pre-exec monotonic start')
    smoke.inherited_lease(fd); launch = checked(source, seed, launch_sha)
    check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit CUDA0 probe child')
    host.wait_idle(); root = output_path(source, seed)
    _, receipts = evaluation.run_cases(launch, root, files.file_bytes(root/'runtime.json'),
        started+CHILD_SECONDS-30, DECLARATION, started_monotonic=started)
    require(checked(source, seed, launch_sha) == launch, 'unchanged completed probe inputs')
    files.write_json(root/'measurements.json', dict(protocol=PROTOCOL, mode='probe',
        launch_sha256=launch_sha, measurements=evaluation.probe_measurements(launch, receipts)))


def check_service(source, seed):
    values = {k: host.read('systemctl', '--user', 'show', service_name(source, seed), '-p', k, '--value')
        for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice')}
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='16min', KillMode='control-group',
        ActiveState='active', MemoryMax=str(6*1024**3), CPUQuotaPerSecUSec='2s', Nice='10'),
        'independently bounded packed evaluation probe service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only probe supervisor')


def verify_retained(root, launch_sha, report_sha):
    """CPU-only byte/replay/math verification; not independent GPU authentication."""
    require(host.digest(root/'launch.json') == launch_sha and host.digest(root/'report.json') == report_sha,
            'independent retained packed probe hashes')
    launch = files.parse(files.file_bytes(root/'launch.json')); report = files.parse(files.file_bytes(root/'report.json'))
    require(report['protocol'] == PROTOCOL and report['launch_sha256'] == launch_sha
            and report['decision'] in ('probe-measured-full-evaluation-disabled', 'timing-rejected-no-full-evaluation')
            and report['optimizer_steps'] == 0,
            'successful timing-only probe report')
    # Re-hashing a contradictory receipt must not turn a timed-out child into a
    # consistent success archive. The live wrapper enforces the watchdog; this
    # is only an offline schema/bound check, not independent GPU authentication.
    child_receipt = report.get('child')
    require(type(child_receipt) is dict and set(child_receipt) ==
            {'pid', 'returncode', 'elapsed_s', 'samples'}, 'exact retained child success receipt')
    require(type(child_receipt['pid']) is int and child_receipt['pid'] > 0
            and type(child_receipt['returncode']) is int and child_receipt['returncode'] == 0
            and type(child_receipt['elapsed_s']) is float
            and math.isfinite(child_receipt['elapsed_s'])
            and 0 < child_receipt['elapsed_s'] <= CHILD_SECONDS
            and type(child_receipt['samples']) is list, 'bounded retained child success receipt')
    require(all(report[k] is False for k in ('full_evaluation_enabled', 'checkpoint_admitted',
        'learned_stance_accepted', 'football_balance_accepted', 'physical_motion_authorized')),
        'retained probe cannot enable capability')
    require(launch == plan(launch['source'], launch['inputs'], host.digest(root/'runtime.json'),
            launch['retained_training'], launch['deadline_unix']), 'rederived source-bound probe plan')
    require(set(report['files']) == {p.name for p in root.iterdir() if p.is_file()}-{'report.json'}
            and all(host.digest(root/name) == digest for name, digest in report['files'].items()
                    if name == os.path.basename(name) and name not in ('.', '..'))
            and all(name == os.path.basename(name) and name not in ('.', '..') for name in report['files']),
            'exact retained top-level probe hashes')
    replay = evaluation.verify_probe(root, launch, launch_sha, report['measurements_sha256'], DECLARATION)
    require(report['probe'] == dict(measurements=replay['measurements']), 'replayed full-length timing case')
    timing = report['timing']
    observed = timing.get('observed_probe_seconds')
    require(type(observed) is float and math.isfinite(observed)
            and child_receipt['elapsed_s'] <= observed <= SERVICE_SECONDS,
            'consistent retained child and service timing bounds')
    require(timing == derive(replay['measurements'], timing['supervisor_replay_seconds'],
                            timing['observed_probe_seconds']), 'rederived packed evaluation timing')
    require(report['decision'] == ('probe-measured-full-evaluation-disabled' if timing['fits_declared_wsl_window']
                                  else 'timing-rejected-no-full-evaluation'),
            'retained decision agrees with measured timing fit')
    return timing


def supervise(source, seed, launch_sha):
    service_started = time.monotonic()
    check_service(source, seed); launch = checked(source, seed, launch_sha)
    check_window(launch['deadline_unix'], launching=True); root = output_path(source, seed)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'}, 'one fresh packed probe attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed', optimizer_steps=0,
        full_evaluation_enabled=False, checkpoint_admitted=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix']); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live packed probe inputs drift')
            started = time.monotonic()
            require(CHILD_SECONDS == files.PACKED_EVALUATION_PROBE_CHILD_SECONDS,
                    'probe child agrees with independently frozen wrapper')
            report['child'] = files.supervised_packed_evaluation_probe(child_command(source, seed, launch_sha, fd, started),
                root/'child.log', cwd=host.ROOT, env=files.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            report['measurements_sha256'] = host.digest(root/'measurements.json')
            replay_started = time.monotonic()
            probe = evaluation.verify_probe(root, launch, launch_sha, report['measurements_sha256'], DECLARATION)
            replay_seconds = time.monotonic()-replay_started
            report['idle_after'] = host.wait_idle()
        # Include final top-level hashes in observed closeout work; report write
        # itself is still outside the observation and covered by fixed reserve.
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        observed = time.monotonic()-service_started
        timing = derive(probe['measurements'], replay_seconds, observed)
        report.update(probe=dict(measurements=probe['measurements']), timing=timing,
            decision=('probe-measured-full-evaluation-disabled' if timing['fits_declared_wsl_window']
                      else 'timing-rejected-no-full-evaluation'))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        if 'files' not in report:
            report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json', report)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    result.add_argument('--source', required=True); result.add_argument('--training-source')
    result.add_argument('--seed', type=int, required=True); result.add_argument('--deadline-unix', type=int)
    result.add_argument('--launch-sha256'); result.add_argument('--lock-fd', type=int)
    result.add_argument('--started-monotonic', type=float)
    return result


def main():
    args = parser().parse_args()
    if args.mode == 'prepare': print(canonical(prepare(args.source, args.training_source, args.seed, args.deadline_unix)))
    elif args.mode == 'supervise': supervise(args.source, args.seed, args.launch_sha256)
    else: child(args.source, args.seed, args.launch_sha256, args.lock_fd, args.started_monotonic)


if __name__ == '__main__': main()
