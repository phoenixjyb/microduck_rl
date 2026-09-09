"""Bounded frozen initializer/final comparison; no optimizer or pilot admission."""
import argparse
from copy import deepcopy
import gc
from hashlib import sha256
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_eager_learning as training
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_evaluation_worker as worker
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = training.host, training.supervisor
MODULE = 'mjlab_microduck.stance_eager_evaluation'
PROTOCOL = 'football-b1n-eager-initial-final-comparison-v1'
TRAINING_SOURCE = 'c8f6b994a2991e400bf6478b967c30e9b618db6a'
TRAINING_REPORT = 'fc7356c3f3763bc3bf6e800e65adb1826cabe5a86b0705e6c3fbf3607dcd4948'
WORLDS, REQUIRED_PASSES = 128, 122
ITERATIONS = (-1, 127)


def training_inputs():
    """Authenticate the immutable archive before any tensor deserialization."""
    root = training.output_path(TRAINING_SOURCE)
    raw = files.file_bytes(root/'report.json')
    require(sha256(raw).hexdigest() == TRAINING_REPORT, 'independent training report hash')
    report = files.parse(raw)
    require(report['decision'] == 'eager-learning-complete-not-capability'
            and report['child']['returncode'] == 0, 'successful completed training')
    require({p.name for p in root.iterdir()} == set(report['files'])|{'report.json'}, 'exact training archive inventory')
    for name, digest in report['files'].items():
        require(name == os.path.basename(name) and name not in ('.', '..'), 'plain training filename')
        require(host.digest(root/name) == digest, 'immutable training archive hash: '+name)
    launch = files.parse(files.file_bytes(root/'launch.json'))
    completed = training.verify_completed(root, TRAINING_SOURCE, report['launch_sha256'], launch)
    selected = [completed['checkpoints'][0], completed['checkpoints'][-1]]
    require([c['identity']['iteration'] for c in selected] == list(ITERATIONS), 'fixed initializer/final selection')
    for saved in selected:
        checkpoint.load_eager_diagnostic(files.file_bytes(root/saved['file']), saved['sha256'], saved['identity'])
    return dict(source=TRAINING_SOURCE, report_sha256=TRAINING_REPORT, checkpoints=selected)


def plan(source, inputs, runtime_sha, retained, deadline):
    files.hex_id(source, 40); files.hex_id(runtime_sha, 64)
    require(retained['source'] == TRAINING_SOURCE and retained['report_sha256'] == TRAINING_REPORT,
            'fixed completed training parent')
    require(len(retained['checkpoints']) == 2 and type(deadline) is int, 'two checkpoints and explicit deadline')
    cases = []
    for iteration, saved in zip(ITERATIONS, retained['checkpoints']):
        meta = saved['identity']; checkpoint.validate_identity(meta, evaluation=False)
        require(meta['purpose'] == 'eager-learning' and meta['source'] == TRAINING_SOURCE
                and meta['iteration'] == iteration, 'distinct eager checkpoint identity')
        files.hex_id(saved['sha256'], 64)
        require(saved['file'] == ('initial.pt' if iteration == -1 else 'model_127.pt'), 'exact checkpoint filename')
        for seed in trace.SEEDS:
            binding = dict(protocol=trace.EAGER_PROTOCOL, source=source, runtime_sha256=runtime_sha,
                checkpoint_sha256=saved['sha256'], checkpoint_iteration=iteration,
                evaluation_seed=seed, worlds=WORLDS, capture_device='cuda:0')
            launch = bundle.launch_bytes(binding, meta); binding['launch_sha256'] = sha256(launch).hexdigest()
            cases.append(dict(name=f'{"initial" if iteration == -1 else "final"}-seed-{seed}',
                              binding=binding, checkpoint=deepcopy(saved)))
    return dict(protocol=PROTOCOL, source=source, inputs=inputs, runtime_sha256=runtime_sha,
        retained_training=deepcopy(retained), deadline_unix=deadline, cases=cases,
        worlds_per_case=128, cases_required=6, attempts_required=768, policy_ticks=250,
        child_timeout_seconds=900, service_timeout_seconds=960, closeout_seconds=600,
        actor_device='cpu', physics_device='cuda:0', forward_graph=False, optimizer_steps=0,
        reset_policy='one-nominal-first-attempt-no-auto-reset',
        seed_interpretation='nominal-repeatability-not-randomized-generalization',
        checkpoint_admitted=False, learned_stance_accepted=False, physical_motion_authorized=False)


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-eager-evaluation-'+source[:12])


def service_name(source):
    files.hex_id(source, 40)
    return 'microduck-stance-eval-'+source[:12]+'.service'


def prepare(source, deadline):
    training.check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only evaluation preparation')
    inputs = host.identity(source); retained = training_inputs()
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, inputs, sha256(runtime).hexdigest(), retained, deadline)
    root = files.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    training.smoke.write_bytes(root/'runtime.json', runtime); files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def inputs_check(source, launch_sha):
    root = output_path(source); raw = files.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'independent evaluation launch hash')
    launch = files.parse(raw); runtime = files.file_bytes(root/'runtime.json')
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest(),
        training_inputs(), launch['deadline_unix']), 'exact evaluation plan and immutable training')
    plant.checked_runtime(files.parse(runtime), source)
    return launch


def check_service(source):
    fields = ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'KillMode')
    state = dict(line.split('=', 1) for line in host.read('systemctl', '--user', 'show',
        service_name(source), *('--property='+k for k in fields)).splitlines())
    require(state == dict(MainPID=str(os.getpid()), ActiveState='active',
        RuntimeMaxUSec='16min', KillMode='control-group'), 'independent evaluation service boundary')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only evaluation supervisor')


def summarize(launch, scores):
    require(launch == plan(launch['source'], launch['inputs'], launch['runtime_sha256'],
        launch['retained_training'], launch['deadline_unix']), 'exact fixed comparison plan')
    require(set(scores) == {c['name'] for c in launch['cases']}, 'all six cases required')
    rows = []
    for case in launch['cases']:
        score = scores[case['name']]; attempts = score['attempts']
        require(score['binding'] == case['binding'] and score['protocol'] == trace.EAGER_PROTOCOL,
                'separate eager trace identity')
        require(len(attempts) == WORLDS and [a['world_id'] for a in attempts] == list(range(WORLDS))
                and all(a['complete_first_attempt'] is True for a in attempts)
                and score['complete_attempts'] == WORLDS, 'all first attempts complete; no prefix promotion')
        for key in ('trajectory_continuity_validated', 'strict_checkpoint_checked',
                    'deterministic_actor_replay_checked', 'compiled_plant_checked', 'nominal_reset_checked',
                    'terminal_contact_summary_checked', 'kinematic_observations_checked',
                    'action_slew_checked', 'delayed_motor_targets_checked', 'motor_commit_masks_checked',
                    'voltage_history_checked'):
            require(score[key] is True, 'required replay check: '+key)
        require(score['terminal_contact_records_checked'] == WORLDS, 'all terminal contacts replayed')
        require(all(score[k] is False for k in ('checkpoint_admitted', 'learned_stance_accepted',
                'physical_motion_authorized', 'provenance_validated')), 'no implicit skill admission')
        passes = sum(a['candidate_pass'] for a in attempts)
        require(score['numerical_passes'] == passes, 'recomputed pass count')
        rows.append(dict(case=case['name'], iteration=case['binding']['checkpoint_iteration'],
            seed=case['binding']['evaluation_seed'], complete_attempts=WORLDS, numerical_passes=passes,
            meets_original_per_seed_threshold=passes >= REQUIRED_PASSES,
            mean_first_attempt_duration_s=sum(a['last_physics_step']*.002 for a in attempts)/WORLDS,
            hard_failures=sum(a['hard_failure'] for a in attempts),
            failed_gates={key: sum(not a['gates'][key] for a in attempts) for key in attempts[0]['gates']}))
    final = [r for r in rows if r['iteration'] == 127]
    passed = len(final) == 3 and all(r['meets_original_per_seed_threshold'] for r in final)
    return dict(protocol=PROTOCOL, rows=rows, complete_attempts=768, final_numerical_gate_passed=passed,
        decision='final-numerical-pass-diagnostic-only' if passed else 'final-numerical-rejected',
        checkpoint_admitted=False, learned_stance_accepted=False, football_balance_accepted=False,
        physical_motion_authorized=False)


def child(source, launch_sha, fd):
    training.smoke.inherited_lease(fd); launch = inputs_check(source, launch_sha)
    training.check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit evaluation CUDA0')
    host.wait_idle(); root = output_path(source); runtime = files.file_bytes(root/'runtime.json')
    end = time.monotonic()+870; scores = {}; receipts = []
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    for case in launch['cases']:
        require(time.monotonic() < end, 'whole comparison time budget')
        seed = case['binding']['evaluation_seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        saved = case['checkpoint']; binding = case['binding']
        cp_raw = files.file_bytes(training.output_path(TRAINING_SOURCE)/saved['file'])
        launch_raw = bundle.launch_bytes({k: v for k, v in binding.items() if k != 'launch_sha256'}, saved['identity'])
        env = WarpStanceRuntime(WORLDS, device='cuda:0')
        require(env.forward_graph is None and env.wp_device.is_cuda, 'eager-only actual CUDA evaluation')
        result = worker.evaluate_owned_case(root/case['name'], env, binding=binding,
            checkpoint_raw=cp_raw, checkpoint_identity=saved['identity'], runtime_raw=runtime,
            launch_raw=launch_raw, deadline_monotonic=end)
        require(env.forward_graph is None, 'graph remained disabled')
        receipts.append(dict(case=case['name'], manifest_sha256=result['manifest_sha256'], collection=result['collection']))
        files.write_json(root/(case['name']+'.json'), receipts[-1])
        scores[case['name']] = result['score']
        print(canonical(dict(case=case['name'], complete=result['score']['complete_attempts'],
            numerical_passes=result['score']['numerical_passes'])), flush=True)
        del env; gc.collect()
    require(inputs_check(source, launch_sha) == launch, 'unchanged comparison inputs')
    files.write_json(root/'comparison.json', dict(launch_sha256=launch_sha, cases=receipts,
                                                summary=summarize(launch, scores)))


def verify(root, launch, comparison_sha):
    raw = files.file_bytes(root/'comparison.json')
    require(sha256(raw).hexdigest() == comparison_sha, 'independent comparison hash')
    result = files.parse(raw)
    require(result['launch_sha256'] == sha256((canonical(launch)+'\n').encode()).hexdigest(), 'comparison launch binding')
    require([r['case'] for r in result['cases']] == [c['name'] for c in launch['cases']], 'ordered six-case inventory')
    scores = {}
    for case, receipt in zip(launch['cases'], result['cases']):
        require(files.parse(files.file_bytes(root/(case['name']+'.json'))) == receipt, 'independent case receipt')
        scores[case['name']] = bundle.verify_bundle(root/case['name'], receipt['manifest_sha256'],
            binding=case['binding'], checkpoint_identity=case['checkpoint']['identity'])
    expected = {'launch.json', 'runtime.json', 'child.log', 'comparison.json'}
    expected.update(c['name'] for c in launch['cases']); expected.update(c['name']+'.json' for c in launch['cases'])
    require({p.name for p in root.iterdir()} in (expected, expected|{'report.json'}), 'exact comparison directory inventory')
    summary = summarize(launch, scores)
    require(result['summary'] == summary, 'recomputed comparison decision')
    return summary


def supervise(source, launch_sha):
    check_service(source); launch = inputs_check(source, launch_sha)
    training.check_window(launch['deadline_unix'], launching=True); root = output_path(source)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'}, 'one fresh evaluation attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed',
        optimizer_steps=0, checkpoint_admitted=False, physical_motion_authorized=False)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                training.check_window(launch['deadline_unix']); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live evaluation inputs drift')
            report['child'] = files.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log', cwd=host.ROOT,
                env=files.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log'); report['comparison_sha256'] = host.digest(root/'comparison.json')
            report['summary'] = verify(root, launch, report['comparison_sha256'])
            report['idle_after'] = host.wait_idle(); report['decision'] = 'frozen-comparison-complete-not-capability'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True); parser.add_argument('--deadline-unix', type=int)
    parser.add_argument('--launch-sha256'); parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': print(canonical(prepare(args.source, args.deadline_unix)))
    elif args.mode == 'supervise': supervise(args.source, args.launch_sha256)
    else: child(args.source, args.launch_sha256, args.lock_fd)


if __name__ == '__main__': main()
