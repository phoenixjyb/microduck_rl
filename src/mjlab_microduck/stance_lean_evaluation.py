"""Bounded lean-lesson checkpoint comparison; no optimizer or pilot admission.

Declared by ``docs/experiments/2026-09-17-stance-lean-lesson-evaluation-probe.md``
(probe protocol ``football-b1n-lean-lesson-evaluation-probe-v1``) and judged by
the decision rule in ``docs/experiments/2026-09-16-stance-lean-lesson.md``.

Two modes, and they are deliberately different jobs.

``probe``
    One full-length case at the last common checkpoint, timed end to end through
    the same code path the evaluation uses. It records durations and policy tick
    counts **only** — no gate verdict, no tilt, no decision string. Its caps are
    the existing frozen 900 s child / 960 s service pair, which one case plus its
    prelude fits far inside; the probe widens no bound to fit.

``evaluate``
    The twelve declared cases: four common checkpoints (64/128/192/255) times
    three evaluation seeds (541/547/557), 128 environments each, one 5 s first
    attempt per environment from the nominal reset.

Caps are measured, not invented
-------------------------------
The predeclaration forbids sizing the twelve-case watchdog from an analogy or a
straight-line estimate, and requires a separately declared probe to measure it.
``CHILD_SECONDS``/``SERVICE_SECONDS`` are therefore ``None`` until that probe has
run, and the evaluate mode refuses to plan while they are. When they are filled
in they are a *transcription*, and ``measured_caps`` re-derives them from the
retained probe report on every launch, so a hand-edited constant — or a probe
re-run that produced different numbers — is refused rather than launched.

The service's ``RuntimeMaxUSec`` rendering is computed by ``systemd_runtime_max``
from the seconds rather than typed in. That string is derived from two renderings
read off the host (960 s shows as ``16min``, 1,753 s as ``29min 13s``); guessing
it is exactly how the throughput probe's first launch failed its own self-check.

One honest caveat, stated here rather than discovered mid-launch. The probe
measures a **single** case, so its per-case figure is one sample, not a maximum
over samples the way the throughput probe's was. The declared 1.25 safety factor
therefore carries more weight here, and ``check_window`` enforces the
predeclared hard window condition rather than merely warning.
"""

import argparse
from copy import deepcopy
import gc
from hashlib import sha256
import math
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_evaluation as evaluation
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_evaluation_worker as worker
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = smoke.host, smoke.supervisor
MODULE = 'mjlab_microduck.stance_lean_evaluation'
PROTOCOL = 'football-b1n-lean-lesson-evaluation-v1'
TRAINING_SOURCE = '7bbc75fb84e42c3fce2856f23114157762c019ec'
TRAINING_REPORT = 'c205ca0fb386b1c9653946f315291848dce5583706236e3c3dda40b0192ef9be'
WORLDS, REQUIRED_PASSES = 128, 122
ITERATIONS = checkpoint.LEAN_CHECKPOINTS
POLICY_TICKS = 250
CASES = len(ITERATIONS)*len(trace.SEEDS)
ATTEMPTS = CASES*WORLDS
MODES = ('probe', 'evaluate')
DIRECTORIES = {'probe': 'stance-lean-eval-probe-', 'evaluate': 'stance-lean-evaluation-'}
SERVICES = {'probe': 'microduck-lean-eval-probe-', 'evaluate': 'microduck-lean-evaluation-'}
# The unchanged scorer gate, re-exported so the decision cannot be recorded
# against a different number than the one attempts were actually scored with.
TILT_GATE_RAD = evaluation.TILT_GATE_RAD

# --- probe: one case, inside the existing frozen pair -------------------------
PROBE_ITERATION, PROBE_SEED = ITERATIONS[-1], trace.SEEDS[0]
PROBE_CHILD_SECONDS, PROBE_SERVICE_SECONDS = smoke.CHILD_SECONDS, smoke.SERVICE_SECONDS
PROBE_CLOSEOUT_SECONDS = smoke.CLOSEOUT_SECONDS
PROBE_SAFETY = 1.25
WATCHDOG_MARGIN_SECONDS = 60
MAX_WINDOW_SECONDS = 3600
SERVICE_PROPERTIES = ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState')
VALID_STOP_REASON = 'all-first-attempts-complete'
# The trace protocol the probe binds, so a probe bundle can never be read as an
# evaluation case and vice versa. Both are lean; only the directory differs.
PROBE_TRACE_PROTOCOL = trace.LEAN_PROTOCOL

# --- evaluate: transcribed from the probe, never typed in as an estimate ------
# Filled in only from the retained probe measurement. ``None`` means "not yet
# measured", and the evaluate mode fails closed rather than borrowing a bound.
PROBE_SOURCE = None
PROBE_REPORT = None
CHILD_SECONDS, SERVICE_SECONDS = None, None


def systemd_runtime_max(seconds):
    """systemd's rendering of ``RuntimeMaxSec=<seconds>``.

    Derived from two renderings read off the host rather than assumed: the proven
    960 s smoke service shows as ``16min``, and the 1,753 s lean-lesson service as
    ``29min 13s``. A wrong guess here is how the throughput probe's first launch
    failed its own self-check while every property was in fact correct.
    """
    require(type(seconds) is int and seconds > 0, 'positive integer service timeout')
    minutes, rest = divmod(seconds, 60)
    if not rest: return f'{minutes}min'
    if not minutes: return f'{rest}s'
    return f'{minutes}min {rest}s'


def derive_caps(probe_report):
    """The predeclared cap rule, applied to a retained probe report.

    ``unit`` is what repeats per case; ``prelude`` happens once. The 1.25 factor
    is applied to the whole prediction, never to a mean.
    """
    require(probe_report.get('protocol') == PROTOCOL and probe_report.get('mode') == 'probe',
            'a lean evaluation probe report')
    require(probe_report.get('decision') == 'probe-measured', 'a probe that measured a full-length case')
    measured = probe_report.get('measurements')
    require(type(measured) is dict, 'probe measurements')
    for key in ('prelude_seconds', 'env_seconds', 'case_seconds'):
        value = measured.get(key)
        require(type(value) is float and math.isfinite(value) and value > 0,
                'finite positive probe timing: '+key)
    unit = measured['env_seconds']+measured['case_seconds']
    predicted = measured['prelude_seconds']+CASES*unit
    service = math.ceil(PROBE_SAFETY*predicted)
    return dict(unit_seconds=unit, predicted_seconds=predicted, service_seconds=service,
        child_seconds=service-WATCHDOG_MARGIN_SECONDS, safety_factor=PROBE_SAFETY,
        closeout_seconds=PROBE_CLOSEOUT_SECONDS, cases=CASES)


def probe_output_path():
    require(type(PROBE_SOURCE) is str, 'a pinned probe source')
    return lean_output_path(PROBE_SOURCE, 'probe')


def measured_caps():
    """Re-derive the twelve-case caps from the retained probe and require agreement.

    The module constants are a transcription of the probe's measurement. This is
    what makes the transcription checkable instead of trusted.
    """
    require(type(CHILD_SECONDS) is int and type(SERVICE_SECONDS) is int,
            'twelve-case caps are not measured yet; run the declared probe and pin its result')
    require(type(PROBE_REPORT) is str, 'a pinned probe report hash')
    raw = files.file_bytes(probe_output_path()/'report.json')
    require(sha256(raw).hexdigest() == PROBE_REPORT, 'independent probe report hash')
    derived = derive_caps(files.parse(raw))
    require((derived['child_seconds'], derived['service_seconds']) == (CHILD_SECONDS, SERVICE_SECONDS),
            'declared caps equal the probe-derived caps')
    require(SERVICE_SECONDS == CHILD_SECONDS+WATCHDOG_MARGIN_SECONDS,
            'child watchdog is the service cap less the declared margin')
    require(derived['service_seconds']+PROBE_CLOSEOUT_SECONDS+WATCHDOG_MARGIN_SECONDS <= MAX_WINDOW_SECONDS,
            'twelve-case caps fit one bounded window')
    return dict(derived, runtime_max=systemd_runtime_max(SERVICE_SECONDS))


def service_caps(mode):
    """The one declared bound for a mode. No mode borrows another's."""
    require(mode in MODES, 'declared lean evaluation mode')
    if mode == 'probe':
        return dict(child_seconds=PROBE_CHILD_SECONDS, service_seconds=PROBE_SERVICE_SECONDS,
                    closeout_seconds=PROBE_CLOSEOUT_SECONDS,
                    runtime_max=systemd_runtime_max(PROBE_SERVICE_SECONDS))
    return measured_caps()


def supervisor_wrapper(mode):
    """The frozen child bound for a mode.

    The probe reuses the proven 900 s smoke wrapper. The twelve-case run needs a
    longer bound, which is declared only after the probe measures it; until then
    this refuses rather than borrowing a bound that is too small.
    """
    require(mode in MODES, 'declared lean evaluation mode')
    if mode == 'probe': return files.supervised_stance_smoke
    require(hasattr(files, 'supervised_lean_evaluation'),
            'the twelve-case child bound is declared only after the probe is measured')
    return files.supervised_lean_evaluation


def training_inputs():
    """Authenticate the immutable lean-lesson archive before any tensor loading."""
    root = lean.output_path(TRAINING_SOURCE)
    raw = files.file_bytes(root/'report.json')
    require(sha256(raw).hexdigest() == TRAINING_REPORT, 'independent lean-lesson training report hash')
    report = files.parse(raw)
    require(report['decision'] == 'lean-lesson-complete-not-capability'
            and report['child']['returncode'] == 0, 'successful completed lean lesson')
    require({p.name for p in root.iterdir()} == set(report['files'])|{'report.json'},
            'exact lean-lesson archive inventory')
    for name, digest in report['files'].items():
        require(name == os.path.basename(name) and name not in ('.', '..'), 'plain lean-lesson filename')
        require(host.digest(root/name) == digest, 'immutable lean-lesson archive hash: '+name)
    launch = files.parse(files.file_bytes(root/'launch.json'))
    completed = lean.verify_completed(root, TRAINING_SOURCE, report['launch_sha256'], launch)
    by_iteration = {c['identity']['iteration']: c for c in completed['checkpoints']}
    require(set(by_iteration) == set(range(-1, checkpoint.LEAN_UPDATES)),
            'every lean-lesson checkpoint present exactly once')
    selected = [by_iteration[iteration] for iteration in ITERATIONS]
    require([c['identity']['iteration'] for c in selected] == list(ITERATIONS),
            'fixed common-checkpoint selection')
    for saved in selected:
        # The evaluable-iteration admission is enforced by the loader, not here.
        checkpoint.load_lean_evaluation(files.file_bytes(root/saved['file']), saved['sha256'], saved['identity'])
    return dict(source=TRAINING_SOURCE, report_sha256=TRAINING_REPORT, checkpoints=selected)


def plan(source, inputs, runtime_sha, retained, deadline, mode):
    files.hex_id(source, 40); files.hex_id(runtime_sha, 64)
    require(mode in MODES and type(deadline) is int and deadline > 0,
            'declared lean evaluation mode and explicit deadline')
    # Resolve the declared caps first: an unmeasured twelve-case budget must fail
    # before any archive authentication, hash work or directory creation.
    caps = service_caps(mode)
    require(retained['source'] == TRAINING_SOURCE and retained['report_sha256'] == TRAINING_REPORT,
            'fixed completed lean-lesson parent')
    require([c['identity']['iteration'] for c in retained['checkpoints']] == list(ITERATIONS),
            'all four common lean-lesson checkpoints')
    selected = retained['checkpoints'] if mode == 'evaluate' else [
        c for c in retained['checkpoints'] if c['identity']['iteration'] == PROBE_ITERATION]
    seeds = trace.SEEDS if mode == 'evaluate' else (PROBE_SEED,)
    require(len(selected) == (len(ITERATIONS) if mode == 'evaluate' else 1), 'declared checkpoint coverage')
    cases = []
    for saved in selected:
        meta = saved['identity']
        checkpoint.validate_identity(meta, evaluation=checkpoint.LEAN_PURPOSE)
        require(meta['purpose'] == checkpoint.LEAN_PURPOSE and meta['source'] == TRAINING_SOURCE
                and meta['iteration'] in ITERATIONS, 'distinct lean-lesson checkpoint identity')
        files.hex_id(saved['sha256'], 64)
        require(saved['file'] == f'model_{meta["iteration"]}.pt', 'exact lean-lesson checkpoint filename')
        for seed in seeds:
            binding = dict(protocol=PROBE_TRACE_PROTOCOL, source=source, runtime_sha256=runtime_sha,
                checkpoint_sha256=saved['sha256'], checkpoint_iteration=meta['iteration'],
                evaluation_seed=seed, worlds=WORLDS, capture_device='cuda:0')
            launch = bundle.launch_bytes(binding, meta); binding['launch_sha256'] = sha256(launch).hexdigest()
            cases.append(dict(name=f'lean-{meta["iteration"]}-seed-{seed}',
                              binding=binding, checkpoint=deepcopy(saved)))
    return dict(protocol=PROTOCOL, mode=mode, source=source, inputs=inputs, runtime_sha256=runtime_sha,
        retained_training=deepcopy(retained), deadline_unix=deadline, cases=cases,
        worlds_per_case=WORLDS, cases_required=len(cases), attempts_required=len(cases)*WORLDS,
        policy_ticks=POLICY_TICKS, child_timeout_seconds=caps['child_seconds'],
        service_timeout_seconds=caps['service_seconds'], service_runtime_max=caps['runtime_max'],
        closeout_seconds=caps['closeout_seconds'], watchdog_margin_seconds=WATCHDOG_MARGIN_SECONDS,
        safety_factor=PROBE_SAFETY, probe_source=PROBE_SOURCE, probe_report_sha256=PROBE_REPORT,
        actor_device='cpu', physics_device='cuda:0', forward_graph=False, optimizer_steps=0,
        reset_policy='one-nominal-first-attempt-no-auto-reset',
        seed_interpretation='nominal-repeatability-not-randomized-generalization',
        tilt_gate_rad=TILT_GATE_RAD, tilt_gate_relaxed=False,
        checkpoint_admitted=False, learned_stance_accepted=False, physical_motion_authorized=False)


def lean_output_path(source, mode):
    files.hex_id(source, 40)
    require(mode in MODES, 'declared lean evaluation mode')
    return host.ROOT/'artifacts/evaluations'/(DIRECTORIES[mode]+source[:12])


def output_path(source, mode):
    return lean_output_path(source, mode)


def service_name(source, mode):
    files.hex_id(source, 40)
    require(mode in MODES, 'declared lean evaluation mode')
    return SERVICES[mode]+source[:12]+'.service'


def check_window(deadline, mode, *, launching=False):
    """A fresh absolute window. Expired authority is never a window."""
    require(type(deadline) is int, 'explicit integer deadline')
    now = time.time()
    require(math.isfinite(now), 'finite clock')
    remaining = deadline-now
    require(remaining > 0, 'a window in the future; expired authority is not a window')
    require(remaining <= MAX_WINDOW_SECONDS, 'one bounded job inside 60 minutes')
    if launching:
        caps = service_caps(mode)
        require(remaining > caps['service_seconds']+caps['closeout_seconds']+WATCHDOG_MARGIN_SECONDS,
                'lean evaluation needs a fresh window with its full closeout reserve')


def prepare(source, deadline, mode):
    check_window(deadline, mode, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean evaluation preparation')
    inputs = host.identity(source); retained = training_inputs()
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, inputs, sha256(runtime).hexdigest(), retained, deadline, mode)
    root = files.native._plain_path(output_path(source, mode)); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runtime.json', runtime); files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source, mode), mode=mode,
        cases=len(launch['cases']), launch_sha256=host.digest(root/'launch.json'))


def inputs_check(source, launch_sha, mode):
    root = output_path(source, mode)
    raw = files.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'independent lean evaluation launch hash')
    launch = files.parse(raw); runtime = files.file_bytes(root/'runtime.json')
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest(),
        training_inputs(), launch['deadline_unix'], mode), 'exact lean evaluation plan')
    plant.checked_runtime(files.parse(runtime), source)
    return launch


def check_service(source, mode):
    caps = service_caps(mode)
    values = {k: host.read('systemctl', '--user', 'show', service_name(source, mode), '-p', k, '--value')
              for k in SERVICE_PROPERTIES}
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec=caps['runtime_max'],
                           KillMode='control-group', ActiveState='active'),
            'independently timed lean evaluation service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean evaluation supervisor')


def run_cases(launch, root, runtime, deadline):
    """Run every declared case through the one shared measured path.

    Timings are recorded in both modes so the probed path is the path the
    twelve-case run takes: the probe reads them, the evaluation ignores them.
    """
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    started = time.monotonic(); scores = {}; receipts = []
    for index, case in enumerate(launch['cases']):
        require(time.monotonic() < deadline, 'whole lean comparison time budget')
        seed = case['binding']['evaluation_seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        saved = case['checkpoint']; binding = case['binding']
        cp_raw = files.file_bytes(lean.output_path(TRAINING_SOURCE)/saved['file'])
        launch_raw = bundle.launch_bytes({k: v for k, v in binding.items() if k != 'launch_sha256'},
                                         saved['identity'])
        before_env = time.monotonic()
        env = WarpStanceRuntime(WORLDS, device='cuda:0')
        after_env = time.monotonic()
        require(env.forward_graph is None and env.wp_device.is_cuda, 'lean-only actual CUDA evaluation')
        result = worker.evaluate_owned_case(root/case['name'], env, binding=binding,
            checkpoint_raw=cp_raw, checkpoint_identity=saved['identity'], runtime_raw=runtime,
            launch_raw=launch_raw, deadline_monotonic=deadline, policy_tick_limit=POLICY_TICKS)
        after_case = time.monotonic()
        require(env.forward_graph is None, 'graph remained disabled')
        receipts.append(dict(case=case['name'], manifest_sha256=result['manifest_sha256'],
            collection=result['collection'],
            timings=dict(prelude_seconds=(before_env-started) if index == 0 else None,
                env_seconds=after_env-before_env, case_seconds=after_case-after_env,
                elapsed_seconds=after_case-started)))
        files.write_json(root/(case['name']+'.json'), receipts[-1])
        scores[case['name']] = result['score']
        print(canonical(dict(case=case['name'], complete=result['score']['complete_attempts'],
            numerical_passes=result['score']['numerical_passes'],
            policy_ticks=result['collection']['policy_ticks'])), flush=True)
        del env; gc.collect()
    return scores, receipts


def probe_validity(receipt):
    """A short case cannot size a full-length run."""
    collection = receipt['collection']
    return (collection['policy_ticks'] == POLICY_TICKS and collection['stop_reason'] == VALID_STOP_REASON)


def probe_measurements(launch, receipts):
    require(launch['mode'] == 'probe' and len(receipts) == 1, 'one measured probe case')
    receipt = receipts[0]; timings = receipt['timings']
    require(probe_validity(receipt),
            'probe-invalid-short-case: a short case cannot size a full-length run')
    return dict(prelude_seconds=timings['prelude_seconds'], env_seconds=timings['env_seconds'],
        case_seconds=timings['case_seconds'], policy_ticks=receipt['collection']['policy_ticks'],
        stop_reason=receipt['collection']['stop_reason'],
        checkpoint_iteration=launch['cases'][0]['binding']['checkpoint_iteration'],
        evaluation_seed=launch['cases'][0]['binding']['evaluation_seed'],
        cases_projected=CASES)


def child(source, launch_sha, fd, mode):
    smoke.inherited_lease(fd); launch = inputs_check(source, launch_sha, mode)
    require(launch['mode'] == mode, 'child mode matches the launch document')
    check_window(launch['deadline_unix'], mode, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'explicit evaluation CUDA0')
    host.wait_idle(); root = output_path(source, mode); runtime = files.file_bytes(root/'runtime.json')
    end = time.monotonic()+launch['child_timeout_seconds']-30
    scores, receipts = run_cases(launch, root, runtime, end)
    require(inputs_check(source, launch_sha, mode) == launch, 'unchanged lean comparison inputs')
    if mode == 'probe':
        files.write_json(root/'measurements.json', dict(protocol=PROTOCOL, mode='probe',
            launch_sha256=launch_sha, measurements=probe_measurements(launch, receipts)))
        return
    files.write_json(root/'comparison.json', dict(launch_sha256=launch_sha, cases=receipts,
                                                summary=summarize(launch, scores)))


def summarize(launch, scores):
    """The predeclared decision rule, applied to twelve verified case scores."""
    require(launch['mode'] == 'evaluate', 'a probe is not a decision')
    require(launch == plan(launch['source'], launch['inputs'], launch['runtime_sha256'],
        launch['retained_training'], launch['deadline_unix'], 'evaluate'), 'exact fixed comparison plan')
    require(set(scores) == {c['name'] for c in launch['cases']}, 'all twelve cases required')
    rows = []
    for case in launch['cases']:
        score = scores[case['name']]; attempts = score['attempts']
        require(score['binding'] == case['binding'] and score['protocol'] == PROBE_TRACE_PROTOCOL,
                'separate lean trace identity')
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
        # Tilt is reported for every checkpoint and seed, not only survivors: the
        # trend across 64/128/192/255 is the primary budget-versus-objective
        # evidence. A non-full-duration attempt has no tilt p95 and is counted
        # separately rather than imputed.
        tilt = [a['metrics']['final_second_tilt_p95_rad'] for a in attempts
                if a['metrics']['final_second_tilt_p95_rad'] is not None]
        rows.append(dict(case=case['name'], iteration=case['binding']['checkpoint_iteration'],
            seed=case['binding']['evaluation_seed'], complete_attempts=WORLDS, numerical_passes=passes,
            meets_original_per_seed_threshold=passes >= REQUIRED_PASSES,
            full_duration_attempts=len(tilt),
            tilt_p95_rad_min=min(tilt) if tilt else None,
            tilt_p95_rad_max=max(tilt) if tilt else None,
            tilt_p95_rad_mean=sum(tilt)/len(tilt) if tilt else None,
            mean_first_attempt_duration_s=sum(a['last_physics_step']*.002 for a in attempts)/WORLDS,
            hard_failures=sum(a['hard_failure'] for a in attempts),
            failed_gates={key: sum(not a['gates'][key] for a in attempts) for key in attempts[0]['gates']}))
    per_checkpoint = {}
    for iteration in ITERATIONS:
        seeds = [r for r in rows if r['iteration'] == iteration]
        require(len(seeds) == len(trace.SEEDS), 'all three evaluation seeds per checkpoint')
        per_checkpoint[str(iteration)] = dict(
            passes_all_seeds=all(r['meets_original_per_seed_threshold'] for r in seeds),
            seeds={str(r['seed']): dict(numerical_passes=r['numerical_passes'],
                meets_original_per_seed_threshold=r['meets_original_per_seed_threshold'],
                full_duration_attempts=r['full_duration_attempts'],
                tilt_p95_rad_min=r['tilt_p95_rad_min'], tilt_p95_rad_max=r['tilt_p95_rad_max'],
                tilt_p95_rad_mean=r['tilt_p95_rad_mean'],
                hard_failures=r['hard_failures']) for r in seeds})
    survivors = [i for i in ITERATIONS if per_checkpoint[str(i)]['passes_all_seeds']]
    passed = bool(survivors)
    return dict(protocol=PROTOCOL, mode='evaluate', rows=rows, per_checkpoint=per_checkpoint,
        complete_attempts=ATTEMPTS, passing_checkpoints=survivors,
        lean_lesson_numerical_gate_passed=passed,
        decision='lean-lesson-passed-nominal' if passed else 'lean-lesson-rejected-objective-binds',
        tilt_gate_rad=TILT_GATE_RAD, tilt_gate_relaxed=False,
        checkpoint_admitted=False, learned_stance_accepted=False, football_balance_accepted=False,
        physical_motion_authorized=False)


def verify(root, launch, comparison_sha):
    raw = files.file_bytes(root/'comparison.json')
    require(sha256(raw).hexdigest() == comparison_sha, 'independent comparison hash')
    result = files.parse(raw)
    require(result['launch_sha256'] == sha256((canonical(launch)+'\n').encode()).hexdigest(),
            'comparison launch binding')
    require([r['case'] for r in result['cases']] == [c['name'] for c in launch['cases']],
            'ordered twelve-case inventory')
    scores = {}
    for case, receipt in zip(launch['cases'], result['cases']):
        require(files.parse(files.file_bytes(root/(case['name']+'.json'))) == receipt,
                'independent case receipt')
        scores[case['name']] = bundle.verify_bundle(root/case['name'], receipt['manifest_sha256'],
            binding=case['binding'], checkpoint_identity=case['checkpoint']['identity'])
    expected = {'launch.json', 'runtime.json', 'child.log', 'comparison.json'}
    expected.update(c['name'] for c in launch['cases']); expected.update(c['name']+'.json' for c in launch['cases'])
    require({p.name for p in root.iterdir()} in (expected, expected|{'report.json'}),
            'exact comparison directory inventory')
    summary = summarize(launch, scores)
    require(result['summary'] == summary, 'recomputed lean comparison decision')
    return summary


def verify_probe(root, launch, launch_sha, measurements_sha):
    raw = files.file_bytes(root/'measurements.json')
    require(sha256(raw).hexdigest() == measurements_sha, 'independent probe measurement hash')
    recorded = files.parse(raw)
    case = launch['cases'][0]
    receipt = files.parse(files.file_bytes(root/(case['name']+'.json')))
    require(receipt['collection']['policy_ticks'] == POLICY_TICKS, 'probe measured a full-length case')
    score = bundle.verify_bundle(root/case['name'], receipt['manifest_sha256'],
        binding=case['binding'], checkpoint_identity=case['checkpoint']['identity'])
    require(score['complete_attempts'] == WORLDS, 'probe case ran every first attempt')
    require(recorded == dict(protocol=PROTOCOL, mode='probe', launch_sha256=launch_sha,
            measurements=probe_measurements(launch, [receipt])), 'recomputed probe measurement')
    expected = {'launch.json', 'runtime.json', 'child.log', 'measurements.json',
                case['name'], case['name']+'.json'}
    require({p.name for p in root.iterdir()} in (expected, expected|{'report.json'}),
            'exact probe directory inventory')
    return dict(measurements=recorded['measurements'], derived_caps=derive_caps(
        dict(protocol=PROTOCOL, mode='probe', decision='probe-measured',
             measurements=recorded['measurements'])))


def supervise(source, launch_sha, mode):
    check_service(source, mode); launch = inputs_check(source, launch_sha, mode)
    require(launch['mode'] == mode, 'supervisor mode matches the launch document')
    check_window(launch['deadline_unix'], mode, launching=True); root = output_path(source, mode)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'},
            'one fresh lean evaluation attempt')
    report = dict(protocol=PROTOCOL, mode=mode, launch_sha256=launch_sha, decision='failed',
        tilt_gate_rad=TILT_GATE_RAD, tilt_gate_relaxed=False, optimizer_steps=0,
        checkpoint_admitted=False, learned_stance_accepted=False, physical_motion_authorized=False)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix'], mode); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live lean evaluation inputs drift')
            report['child'] = supervisor_wrapper(mode)(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--mode', mode, '--lock-fd', str(fd)],
                root/'child.log', cwd=host.ROOT, env=files.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            if mode == 'probe':
                report['measurements_sha256'] = host.digest(root/'measurements.json')
                report['probe'] = verify_probe(root, launch, launch_sha, report['measurements_sha256'])
                report['decision'] = 'probe-measured'
            else:
                report['comparison_sha256'] = host.digest(root/'comparison.json')
                report['summary'] = verify(root, launch, report['comparison_sha256'])
                report['decision'] = report['summary']['decision']
            report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--job', choices=MODES)
    parser.add_argument('--deadline-unix', type=int)
    parser.add_argument('--launch-sha256')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(canonical(prepare(args.source, args.deadline_unix, args.job)))
    elif args.mode == 'supervise':
        supervise(args.source, args.launch_sha256, args.job)
    else:
        child(args.source, args.launch_sha256, args.lock_fd, args.job)


if __name__ == '__main__': main()
