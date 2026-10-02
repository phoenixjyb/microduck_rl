"""One bounded 36-case portable packed replication evaluation; never training.

Declared in docs/experiments/2026-10-03-portable-packed-full-evaluation.md.
Historical AVX2 archive authentication, portable CPU actor replay, and actual
leased CUDA capture remain separate. A numerical result never admits a skill.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import gc
import math
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_historical_replication_auth as auth
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck import stance_lean_replication_campaign as campaign
from mjlab_microduck import stance_packed_evaluation_probe as probe
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_solved_field_check as checker
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = evaluation.host, evaluation.files
MODULE = 'mjlab_microduck.stance_portable_packed_evaluation'
PROTOCOL = 'football-b1n-wsl-portable-packed-full-evaluation-v1'
SEEDS, ITERATIONS = auth.SEEDS, trace.LEAN_REPLICATION_CHECKPOINTS
CASES, ATTEMPTS = 36, 4608
NOT_BEFORE = int(datetime(2026, 10, 2, 16, 30, tzinfo=timezone.utc).timestamp())
CUTOFF = int(datetime(2026, 10, 3, 0, 0, tzinfo=timezone.utc).timestamp())
R4_SOURCE = '07466b58021c084dbc2f688effa936d9466ba456'
R4_LAUNCH = 'dc2955e29f9dff489b77b2e8b19ecb596c6a1105f85c2cd200bcdf2e9172376c'
R4_REPORT = 'c75bbc4d43b977d50478840b362982199d3797490c53e68be57d821cfe54facf'
R4_INVENTORY = '1587ad5e90b63ccfe06ff5a588185dab55b19c89ad6a52da06b858a90e8f2b62'
INDEPENDENT_RECEIPT = 'eed529043cb4af240cc922138e73ca383beda9dd79f4805a17ce3fd962100406'
AUTH_CALLS = 12  # supervisor pre/post plus child pre/post, three seeds each.
AUTH_RESERVE = AUTH_CALLS * auth.SECONDS
CLOSEOUT_SECONDS, MARGIN_SECONDS = 600, 60
SERVICE_SECONDS, CHILD_SECONDS = 11728, 10981  # checked transcription below.
DECLARATION = dict(protocol=PROTOCOL, trace_protocol=trace.PORTABLE_FULL_PROTOCOL,
    iterations=ITERATIONS, gate_key=evaluation.REPLICATION['gate_key'],
    passed=evaluation.REPLICATION['passed'], rejected=evaluation.REPLICATION['rejected'])
FALSE_FLAGS = dict(checkpoint_admitted=False, learned_stance_accepted=False,
    football_balance_accepted=False, physical_motion_authorized=False,
    independent_gpu_attestation=False, complete_binary_runtime_equivalence_verified=False)


def runtime_max(seconds):
    require(type(seconds) is int and seconds > 0, 'positive integer runtime cap')
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return ' '.join(part for part in (f'{hours}h' if hours else '',
        f'{minutes}min' if minutes else '', f'{rest}s' if rest else '') if part)


def derive_caps(measurements, replay_seconds, observed_seconds):
    old = probe.derive(measurements, replay_seconds, observed_seconds)
    unit = (measurements['env_seconds'] + measurements['case_seconds'] +
            replay_seconds + old['unattributed_overhead_seconds'])
    prediction = measurements['prelude_seconds'] + CASES * unit + AUTH_RESERVE
    service = math.ceil(evaluation.PROBE_SAFETY * prediction)
    replay_reserve = math.ceil(evaluation.PROBE_SAFETY * CASES * replay_seconds) + MARGIN_SECONDS
    parent_auth = len(SEEDS) * auth.SECONDS
    child = service - replay_reserve - parent_auth - MARGIN_SECONDS
    require(child > 0, 'positive full matrix child cap')
    return dict(cases=CASES, attempts=ATTEMPTS, repeating_unit_seconds=unit,
        unattributed_overhead_seconds=old['unattributed_overhead_seconds'],
        predicted_seconds=prediction, historical_authentication_calls=AUTH_CALLS,
        historical_authentication_reserve_seconds=AUTH_RESERVE,
        safety_factor=evaluation.PROBE_SAFETY, parent_replay_reserve_seconds=replay_reserve,
        parent_authentication_reserve_seconds=parent_auth, child_seconds=child,
        service_seconds=service, runtime_max=runtime_max(service),
        closeout_seconds=CLOSEOUT_SECONDS, watchdog_margin_seconds=MARGIN_SECONDS)


def r4_root():
    return host.ROOT/'artifacts/evaluations'/f'stance-wsl-packed-eval-probe-{R4_SOURCE[:12]}-seed-577'


def independent_receipt_path():
    return host.ROOT/'artifacts/tools/r4-native-independent-replay-07466b58021c.json'


def pinned_caps():
    raw = files.file_bytes(r4_root()/'report.json', limit=1024*1024)
    require(sha256(raw).hexdigest() == R4_REPORT, 'exact completed R4 timing report')
    report = files.parse(raw)
    require(report['launch_sha256'] == R4_LAUNCH and report['child']['returncode'] == 0,
            'successful source-bound R4 capture')
    receipt_raw = files.file_bytes(independent_receipt_path(), limit=16384)
    require(sha256(receipt_raw).hexdigest() == INDEPENDENT_RECEIPT,
            'exact independently retained R4 replay receipt')
    receipt = files.parse(receipt_raw)
    require(receipt['source'] == R4_SOURCE and receipt['report_sha256'] == R4_REPORT
            and receipt['launch_sha256'] == R4_LAUNCH
            and receipt['inventory_sha256'] == R4_INVENTORY
            and receipt['strict_complete_plan_bundle_score_verified'] is True
            and receipt['entire_compiled_plant_verified'] is True
            and receipt['complete_attempts'] == 128 and receipt['numerical_passes'] == 128
            and receipt['actor_replay_max_abs_error'] == 0.0
            and receipt['cuda_initialized'] is False and receipt['service_result'] == 'success',
            'completed independent R4 replay gate')
    measured = report['probe']['measurements']
    timing = report['timing']
    require(probe.derive(measured, timing['supervisor_replay_seconds'],
                        timing['observed_probe_seconds']) == timing,
            'unchanged R4 measured timing derivation')
    caps = derive_caps(measured, timing['supervisor_replay_seconds'], timing['observed_probe_seconds'])
    require((caps['service_seconds'], caps['child_seconds']) == (SERVICE_SECONDS, CHILD_SECONDS),
            'full matrix caps equal checked transcription')
    return caps


def check_window(deadline, *, launching=False, now=None):
    require(type(deadline) is int and NOT_BEFORE < deadline <= CUTOFF,
            'explicit full evaluation deadline before October 3 08:00 Shanghai')
    now = time.time() if now is None else now
    require(type(now) in (int, float) and math.isfinite(now) and NOT_BEFORE <= now < deadline,
            'active separately declared full evaluation window')
    if launching:
        require(deadline-now > SERVICE_SECONDS+CLOSEOUT_SECONDS+MARGIN_SECONDS,
                'whole full matrix plus closeout fits the authorized window')


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/f'stance-wsl-portable-full-evaluation-{source[:12]}'


def service_name(source):
    output_path(source)
    return f'microduck-wsl-portable-full-evaluation-{source[:12]}.service'


def authenticate_inputs(source):
    return {str(seed): auth.run(source, auth.TRAINING_SOURCE, seed) for seed in SEEDS}


def plan(source, inputs, runtime_sha, authentications, deadline, checker_sha, caps):
    files.hex_id(source, 40); files.hex_id(runtime_sha, 64); files.hex_id(checker_sha, 64)
    require(type(deadline) is int and NOT_BEFORE < deadline <= CUTOFF, 'fixed full evaluation cutoff')
    require(inputs.get('source') == source and
            inputs.get('execution_profile') == host.execution.select(host.execution.WSL),
            'recorded exact WSL full capture inputs')
    require(caps == pinned_caps(), 'source-bound measured full matrix budget')
    require(type(authentications) is dict and set(authentications) == {str(s) for s in SEEDS},
            'all three fixed historical archives')
    cases = []
    for seed in SEEDS:
        receipt = authentications[str(seed)]; retained = receipt['retained_training']
        auth.validate(receipt, source, retained)
        require(receipt['training_seed'] == seed, 'ordered historical authentication seed')
        for saved in retained['checkpoints']:
            meta = saved['identity']
            require(meta['training_launch_sha256'] == receipt['archive_files']['launch.json']
                    and meta['runtime_sha256'] == receipt['archive_files']['runtime.json'],
                    'checkpoint binds its exact historical launch and runtime')
            for evaluation_seed in trace.SEEDS:
                binding = dict(protocol=trace.PORTABLE_FULL_PROTOCOL, source=source,
                    runtime_sha256=runtime_sha, checkpoint_sha256=saved['sha256'],
                    checkpoint_iteration=meta['iteration'], training_seed=seed,
                    evaluation_seed=evaluation_seed, worlds=evaluation.WORLDS,
                    capture_device='cuda:0', solved_field_check='packed', checker_sha256=checker_sha,
                    cpu_math_profile=profile.expected_receipt())
                binding['launch_sha256'] = sha256(evaluation.bundle.launch_bytes(binding, meta)).hexdigest()
                cases.append(dict(name=f'packed-full-train-{seed}-cp-{meta["iteration"]}-eval-{evaluation_seed}',
                                  binding=binding, checkpoint=deepcopy(saved)))
    require(len(cases) == CASES and len({c['name'] for c in cases}) == CASES, '36 unique ordered cases')
    return dict(protocol=PROTOCOL, source=source, inputs=deepcopy(inputs), runtime_sha256=runtime_sha,
        historical_training_authentications=deepcopy(authentications),
        authentications_sha256=sha256((canonical(authentications)+'\n').encode()).hexdigest(),
        training_source=auth.TRAINING_SOURCE, training_seeds=list(SEEDS),
        checkpoint_iterations=list(ITERATIONS), evaluation_seeds=list(trace.SEEDS), cases=cases,
        cases_required=CASES, attempts_required=ATTEMPTS, worlds_per_case=evaluation.WORLDS,
        policy_ticks=evaluation.POLICY_TICKS, deadline_unix=deadline, not_before_unix=NOT_BEFORE,
        measured_caps=deepcopy(caps), probe_source=R4_SOURCE, probe_report_sha256=R4_REPORT,
        independent_probe_receipt_sha256=INDEPENDENT_RECEIPT, checker_sha256=checker_sha,
        actor_device='cpu', physics_device='cuda:0', forward_graph=False, optimizer_steps=0,
        reset_policy='one-nominal-first-attempt-no-auto-reset',
        seed_interpretation='nominal-repeatability-not-randomized-generalization',
        tilt_gate_rad=evaluation.TILT_GATE_RAD, tilt_gate_relaxed=False, **FALSE_FLAGS)


def prepare(source, deadline):
    check_window(deadline, launching=True)
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'explicit full evaluation WSL profile')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only full matrix preparation')
    profile.checked_receipt()
    caps = pinned_caps()
    probe.verify_retained(r4_root(), R4_LAUNCH, R4_REPORT)
    inputs = host.identity(source); authentications = authenticate_inputs(source)
    for seed in SEEDS:
        receipt = authentications[str(seed)]
        training_root = evaluation.lean.output_path(auth.TRAINING_SOURCE, evaluation.lean.PACKED_REPLICATION, seed)
        training_launch = files.parse(files.file_bytes(training_root/'launch.json'))
        require({k: v for k, v in training_launch['inputs'].items() if k != 'source'} ==
                {k: v for k, v in inputs.items() if k != 'source'}
                and training_launch['checker_sha256'] == host.digest(checker.__file__)
                and host.digest(training_root/'launch.json') == receipt['archive_files']['launch.json'],
                'same frozen host/stack/plant/checker as all three original trainings')
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, inputs, sha256(runtime).hexdigest(), authentications, deadline,
                  host.digest(checker.__file__), caps)
    check_window(deadline, launching=True)
    root = files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    evaluation.smoke.write_bytes(root/'runtime.json', runtime); files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'), caps=caps)


def checked(source, launch_sha):
    files.hex_id(source, 40); files.hex_id(launch_sha, 64)
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent full matrix launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    profile.checked_receipt()
    require(launch == plan(source, host.identity(source), host.digest(root/'runtime.json'),
        authenticate_inputs(source), launch['deadline_unix'], host.digest(checker.__file__), pinned_caps()),
        'unchanged full matrix source/host/three archives/plan')
    plant.checked_runtime(files.parse(files.file_bytes(root/'runtime.json')), source)
    return launch


def summarize(launch, scores):
    require(launch == plan(launch['source'], launch['inputs'], launch['runtime_sha256'],
        launch['historical_training_authentications'], launch['deadline_unix'], launch['checker_sha256'],
        launch['measured_caps']), 'exact complete full matrix plan')
    require(set(scores) == {c['name'] for c in launch['cases']}, 'all 36 scores required')
    verdicts = {}
    for seed in SEEDS:
        cases = [c for c in launch['cases'] if c['binding']['training_seed'] == seed]
        selected = {c['name']: scores[c['name']] for c in cases}
        verdicts[seed] = evaluation.summarize_cases(cases, selected, DECLARATION)
    decision = campaign.decide(verdicts)
    return dict(protocol=PROTOCOL, decision=decision, complete_attempts=ATTEMPTS,
        cases=CASES, training_seeds=list(SEEDS),
        per_seed={str(s): verdicts[s] for s in SEEDS},
        passing_seeds=[s for s in SEEDS if verdicts[s]['decision'] == DECLARATION['passed']],
        nominal_replication_numerical_gate_passed=decision == 'lean-replication-passed',
        checkpoint_selection='any checkpoint passes all three held-out seeds per training seed; no common-iteration requirement',
        tilt_gate_rad=evaluation.TILT_GATE_RAD, tilt_gate_relaxed=False, **FALSE_FLAGS)


def run_cases(launch, root, runtime, deadline, *, started_monotonic):
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    scores, receipts = {}, []
    for index, case in enumerate(launch['cases']):
        require(time.monotonic() < deadline, 'whole full matrix case budget')
        binding, saved = case['binding'], case['checkpoint']
        seed = binding['evaluation_seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        training_root = evaluation.lean.output_path(auth.TRAINING_SOURCE,
            evaluation.lean.PACKED_REPLICATION, binding['training_seed'])
        cp_raw = files.file_bytes(training_root/saved['file'])
        launch_raw = evaluation.bundle.launch_bytes(
            {k: v for k, v in binding.items() if k != 'launch_sha256'}, saved['identity'])
        before_env = time.monotonic()
        env = WarpStanceRuntime(evaluation.WORLDS, device='cuda:0', solved_field_check='packed')
        after_env = time.monotonic()
        require(env.forward_graph is None and env.wp_device.is_cuda, 'actual eager packed full runtime')
        result = evaluation.worker.evaluate_owned_case(root/case['name'], env, binding=binding,
            checkpoint_raw=cp_raw, checkpoint_identity=saved['identity'], runtime_raw=runtime,
            launch_raw=launch_raw, deadline_monotonic=deadline, policy_tick_limit=evaluation.POLICY_TICKS)
        after_case = time.monotonic()
        require(env.forward_graph is None, 'full matrix graph remains disabled')
        receipt = dict(case=case['name'], manifest_sha256=result['manifest_sha256'], collection=result['collection'],
            timings=dict(prelude_seconds=before_env-started_monotonic if index == 0 else None,
                env_seconds=after_env-before_env, case_seconds=after_case-after_env,
                elapsed_seconds=after_case-started_monotonic))
        files.write_json(root/(case['name']+'.json'), receipt)
        receipts.append(receipt); scores[case['name']] = result['score']
        print(canonical(dict(case=case['name'], complete=result['score']['complete_attempts'],
            numerical_passes=result['score']['numerical_passes'], policy_ticks=result['collection']['policy_ticks'])), flush=True)
        del env; gc.collect()
    return scores, receipts


def child_command(source, launch_sha, fd, started):
    files.hex_id(source, 40); files.hex_id(launch_sha, 64)
    require(type(fd) is int and fd >= 0 and type(started) is float and math.isfinite(started) and started > 0,
            'owned full matrix lease and start')
    return [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
        '--launch-sha256', launch_sha, '--lock-fd', str(fd), '--started-monotonic', repr(started)]


def child(source, launch_sha, fd, started):
    require(type(started) is float and math.isfinite(started) and 0 < time.monotonic()-started < CHILD_SECONDS,
            'same-clock bounded child start')
    evaluation.smoke.inherited_lease(fd)
    launch = checked(source, launch_sha); check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit sole CUDA0 child')
    host.wait_idle(); root = output_path(source)
    scores, receipts = run_cases(launch, root, files.file_bytes(root/'runtime.json'),
        started+CHILD_SECONDS-30, started_monotonic=started)
    require(checked(source, launch_sha) == launch, 'unchanged completed full matrix inputs')
    files.write_json(root/'comparison.json', dict(protocol=PROTOCOL, launch_sha256=launch_sha,
        cases=receipts, summary=summarize(launch, scores)))


def verify(root, launch, comparison_sha):
    raw = files.file_bytes(root/'comparison.json')
    require(sha256(raw).hexdigest() == comparison_sha, 'independent full comparison hash')
    result = files.parse(raw)
    require(result['protocol'] == PROTOCOL and result['launch_sha256'] ==
            sha256((canonical(launch)+'\n').encode()).hexdigest(), 'full comparison launch binding')
    require([r['case'] for r in result['cases']] == [c['name'] for c in launch['cases']], 'ordered 36-case inventory')
    scores = {}
    for case, receipt in zip(launch['cases'], result['cases']):
        require(files.parse(files.file_bytes(root/(case['name']+'.json'))) == receipt, 'independent full case receipt')
        collection = receipt['collection']
        require(type(collection) is dict and set(collection) == {'stop_reason', 'policy_ticks',
            'actor_device', 'physics_device', 'seed_initialization_validated',
            'independent_gpu_supervision_validated', 'checkpoint_admitted'}
            and collection['stop_reason'] == 'all-first-attempts-complete'
            and type(collection['policy_ticks']) is int and 1 <= collection['policy_ticks'] <= evaluation.POLICY_TICKS
            and collection['actor_device'] == 'cpu' and collection['physics_device'] == 'cuda:0'
            and all(collection[k] is False for k in ('seed_initialization_validated',
                'independent_gpu_supervision_validated', 'checkpoint_admitted')),
            'complete first attempts, never a wall-budget prefix')
        scores[case['name']] = evaluation.bundle.verify_bundle(root/case['name'], receipt['manifest_sha256'],
            binding=case['binding'], checkpoint_identity=case['checkpoint']['identity'])
    expected = {'launch.json', 'runtime.json', 'child.log', 'comparison.json'}
    expected.update(c['name'] for c in launch['cases']); expected.update(c['name']+'.json' for c in launch['cases'])
    require({p.name for p in root.iterdir()} in (expected, expected|{'report.json'}), 'exact full comparison inventory')
    summary = summarize(launch, scores)
    require(result['summary'] == summary, 'recomputed full matrix decision')
    return summary


def check_service(source):
    caps = pinned_caps()
    values = {key: host.read('systemctl', '--user', 'show', service_name(source), '-p', key, '--value')
              for key in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState', 'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice')}
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec=caps['runtime_max'], KillMode='control-group',
        ActiveState='active', MemoryMax=str(6*1024**3), CPUQuotaPerSecUSec='2s', Nice='10'), 'bounded full matrix user service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only full supervisor')
    require(host.execution.PROFILE['name'] == host.execution.WSL, 'exact leased WSL full supervisor')


def supervise(source, launch_sha):
    service_started = time.monotonic()
    check_service(source); launch = checked(source, launch_sha)
    check_window(launch['deadline_unix'], launching=True); root = output_path(source)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'}, 'one fresh full evaluation attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='lean-replication-incomplete',
        optimizer_steps=0, tilt_gate_rad=evaluation.TILT_GATE_RAD, tilt_gate_relaxed=False, **FALSE_FLAGS)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix']); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live full matrix host/source drift')
            env = files.child_environment(); env.update(profile.settings())
            started = time.monotonic()
            report['child'] = files._timed_process(child_command(source, launch_sha, fd, started), root/'child.log',
                cwd=host.ROOT, env=env, lock_fd=fd, timeout=CHILD_SECONDS, monitor=files.live_gpu, guard=guard)
            host.check_log(root/'child.log')
            require(checked(source, launch_sha) == launch, 'completed immutable three-archive full inputs')
            report['comparison_sha256'] = host.digest(root/'comparison.json')
            report['summary'] = verify(root, launch, report['comparison_sha256'])
            report['decision'] = report['summary']['decision']; report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        report['observed_service_seconds'] = time.monotonic()-service_started
        files.write_json(root/'report.json', report)


def verify_retained(root, launch_sha, report_sha):
    """CPU-only whole-plan/bundle/score rederivation, never GPU attestation."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only retained full matrix replay')
    require(host.digest(root/'launch.json') == launch_sha and host.digest(root/'report.json') == report_sha,
            'independent full retained hashes')
    launch = files.parse(files.file_bytes(root/'launch.json')); report = files.parse(files.file_bytes(root/'report.json'))
    profile.checked_receipt()
    require(report['protocol'] == PROTOCOL and report['launch_sha256'] == launch_sha
            and report['decision'] in campaign.DECISIONS[:-1] and report['optimizer_steps'] == 0,
            'completed numerical full matrix report')
    require(all(report[k] is False for k in FALSE_FLAGS), 'no full matrix capability admission')
    child_receipt = report['child']
    require(type(child_receipt) is dict and set(child_receipt) == {'pid', 'returncode', 'elapsed_s', 'samples'},
            'exact retained full child receipt')
    require(type(child_receipt['pid']) is int and child_receipt['pid'] > 0
            and type(child_receipt['returncode']) is int and child_receipt['returncode'] == 0
            and type(child_receipt['elapsed_s']) is float and math.isfinite(child_receipt['elapsed_s'])
            and 0 < child_receipt['elapsed_s'] <= CHILD_SECONDS
            and type(child_receipt['samples']) is list and bool(child_receipt['samples']),
            'bounded retained full child success')
    observed = report['observed_service_seconds']
    require(type(observed) is float and math.isfinite(observed)
            and child_receipt['elapsed_s'] <= observed <= SERVICE_SECONDS,
            'consistent retained full service bounds')
    # Recorded WSL namespaces are independent of the replay host's profile.
    service_keys = set(files.SERVICES) | {'user:'+key for key in files.SERVICES}
    for sample in child_receipt['samples']:
        require(type(sample) is dict and set(sample) == {'services', 'compute_pids', 'temperature_c',
                'memory_used_mib', 'memory_free_mib'}
                and type(sample['services']) is dict and set(sample['services']) == service_keys
                and all(state == 'inactive' for state in sample['services'].values())
                and type(sample['compute_pids']) is list
                and all(type(pid) is int and pid == child_receipt['pid'] for pid in sample['compute_pids'])
                and type(sample['temperature_c']) is int
                and 0 <= sample['temperature_c'] < host.execution.select(host.execution.WSL)['temperature_c']
                and type(sample['memory_used_mib']) is int and 0 <= sample['memory_used_mib'] <= 5120
                and type(sample['memory_free_mib']) is int and sample['memory_free_mib'] >= 6144,
                'recorded full child ownership, protected services and GPU reserves')
    require(set(report['files']) == {p.name for p in root.iterdir() if p.is_file()}-{'report.json'},
            'exact retained top-level full inventory')
    for name, digest in report['files'].items():
        require(name == os.path.basename(name) and name not in ('.', '..')
                and host.digest(root/name) == digest, 'unchanged retained full file')
    summary = verify(root, launch, report['comparison_sha256'])
    require(report['summary'] == summary and report['decision'] == summary['decision'], 'rederived complete full decision')
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    result.add_argument('--source', required=True)
    result.add_argument('--deadline-unix', type=int)
    result.add_argument('--launch-sha256')
    result.add_argument('--lock-fd', type=int)
    result.add_argument('--started-monotonic', type=float)
    return result


def main():
    cli = parser(); args = cli.parse_args()
    required = {'prepare': ('deadline_unix',), 'supervise': ('launch_sha256',),
                'child': ('launch_sha256', 'lock_fd', 'started_monotonic')}[args.mode]
    if any(getattr(args, key) is None for key in required):
        cli.error('missing mode arguments: '+', '.join('--'+key.replace('_', '-') for key in required))
    allowed = set(required)
    if any(getattr(args, key) is not None for key in ('deadline_unix', 'launch_sha256', 'lock_fd', 'started_monotonic')
           if key not in allowed):
        cli.error('arguments from another mode are not permitted')
    if args.mode == 'prepare': print(canonical(prepare(args.source, args.deadline_unix)))
    elif args.mode == 'supervise': supervise(args.source, args.launch_sha256)
    else: child(args.source, args.launch_sha256, args.lock_fd, args.started_monotonic)


if __name__ == '__main__': main()
