"""Twenty-five fresh CPU first attempts for a fixed cardinal dose/timing screen.

This protocol only screens an unchanged frozen parent. It does not train,
select a checkpoint, reset an attempt, admit recovery, or authorize motion.
Every raw case is retained before its independent score is used.
"""
import argparse
import gc
from hashlib import sha256
import io
import os
import time

import torch

from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_cuda_schedule_probe as cuda_prior
from mjlab_microduck import stance_recovery_dose_screen as prior
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_schedule_trace as evidence
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1d-cpu-cardinal-dose-timing-screen-v1'
MODULE = 'mjlab_microduck.stance_recovery_broad_screen'
ZERO_CELL = 'zero-wrench'
ONSETS = (250, 500, 750)
DIRECTIONS = ('+x', '-x', '+y', '-y')
DOSES = ((2, 20), (4, 10))
CELL_IDS = (ZERO_CELL,) + tuple(
    f'{direction}-{newtons}n-{steps}steps-t{onset}'
    for onset in ONSETS for direction in DIRECTIONS for newtons, steps in DOSES)
PREFIX_STEP = 250
POLICY_TICKS = 250
PHYSICS_STEPS = 2500
EVALUATION_SEED = 671
SERVICE_SECONDS = 1440
CLOSEOUT_SECONDS = 600
WINDOW_MARGIN_SECONDS = 60
LAUNCH_RESERVE = SERVICE_SECONDS + CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS
COLLECTION_SECONDS = 60
CASE_RESERVE_SECONDS = 80
PREVIOUS_THREE_CASE_SECONDS = 131.005
TIMING_FACTOR = 1.25
PROJECTED_SERVICE_SECONDS = PREVIOUS_THREE_CASE_SECONDS / 3 * len(CELL_IDS) * TIMING_FACTOR
PARENT_CHECKPOINT_SHA256 = '2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5'
PARENT_STATE_SHA256 = 'e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329'
CLOSED_CUDA_SOURCE = '84ce52af2fc501830426a4290166f860a7d1a2fe'
CLOSED_CUDA_SHA256 = 'cb2c5cb2a53c67c7f84bf01df1a884d5ab5365e816c95cde051fc050b121a241'
CLOSED_DOSE_SOURCE = 'd55e7ef5f86ffdee69e2de468be64dc023c1bafe'
CLOSED_DOSE_SHA256 = '30bf131f203b8c38f20ee3644ba16912aa380e83130838504ee4322962066d12'
CLOSED_DOSE_REPORT_SHA256 = '220a5e7456468c0e175519a8a2d01423c495582d99a1768e15ad95c56fbf33f9'

base = prior.base
CASE_FILES = {f'case-{i}{suffix}' for i in range(len(CELL_IDS))
              for suffix in ('.pt', '.json', '-replay.json')}
PREPARE_FILES = {'checkpoint.pt', 'launch.json', 'cpu-qualification.json'} | CASE_FILES
COMPLETE_FILES = PREPARE_FILES | {'report.json'}


def declarations(source):
    """The exact ordered zero-plus-cardinal dose/timing matrix."""
    base.files.hex_id(source, 40)
    return [base.schedule.declaration(source, 'dose', 'held-out', [cell])
            for cell in CELL_IDS]


def output_path(source):
    base.files.hex_id(source, 40)
    return base.host.ROOT / 'artifacts/evaluations' / (
        'stance-wsl-cpu-cardinal-dose-timing-' + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ('run', 'closeout'), 'declared broad CPU screen service mode')
    return f'microduck-cpu-broad-{mode}-{source[:12]}.service'


def service_properties(source, mode):
    seconds = dict(run=SERVICE_SECONDS, closeout=CLOSEOUT_SECONDS)[mode]
    props = {key: base.host.read('systemctl', '--user', 'show', service_name(source, mode),
            '-p', key, '--value') for key in
            ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax',
             'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    require(props == dict(MainPID=str(os.getpid()), ActiveState='active',
        RuntimeMaxUSec=f'{seconds // 60}min', MemoryMax=str(2 * 1024**3),
        CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'),
        'exact independently capped broad CPU service')
    return props


def _closed_cuda_prerequisite():
    require(type(CLOSED_CUDA_SHA256) is str and len(CLOSED_CUDA_SHA256) == 64
            and all(c in '0123456789abcdef' for c in CLOSED_CUDA_SHA256),
            'pinned independent CUDA closeout digest required')
    root = cuda_prior.output_path(CLOSED_CUDA_SOURCE)
    closeout_raw = base.files.file_bytes(root / 'independent-closeout.json')
    require(sha256(closeout_raw).hexdigest() == CLOSED_CUDA_SHA256,
            'exact independently closed scheduled CUDA prerequisite bytes')
    closeout = base.files.parse(closeout_raw)
    report_raw = base.files.file_bytes(root / 'report.json')
    report = base.files.parse(report_raw)
    expected_names = cuda_prior.COMPLETE_FILES | {'report.json'}
    require(closeout['source'] == report['source'] == CLOSED_CUDA_SOURCE
            and closeout['report_sha256'] == sha256(report_raw).hexdigest()
            and closeout['decision'] == report['decision'] == 'scheduled-cuda-integration-qualified'
            and closeout['whole_cpu_rescore_identical'] is True
            and closeout['cases_checked'] == 2
            and all(closeout[k] is False and report[k] is False
                    for k in base.baseline.FALSE_FLAGS)
            and set(closeout['files_rehashed']) == expected_names
            and {p.name for p in root.iterdir()} == expected_names | {'independent-closeout.json'},
            'complete qualified scheduled CUDA prerequisite and static inventory')
    for name, item in closeout['files_rehashed'].items():
        path = root / name
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and type(item) is dict and set(item) == {'sha256', 'bytes'}
                and type(item['bytes']) is int and path.stat().st_size == item['bytes']
                and base.host.digest(path) == item['sha256'],
                'unchanged scheduled CUDA prerequisite artifact ' + str(name))
    return dict(source=CLOSED_CUDA_SOURCE, closeout_sha256=CLOSED_CUDA_SHA256,
        report_sha256=closeout['report_sha256'], decision=closeout['decision'],
        whole_cpu_rescore_identical=True, cases_checked=2)


def _closed_dose_timing_prerequisite():
    """Authenticate the completed three-case run used for the timing projection."""
    root = prior.output_path(CLOSED_DOSE_SOURCE)
    closeout_raw = base.files.file_bytes(root / 'independent-closeout.json')
    require(sha256(closeout_raw).hexdigest() == CLOSED_DOSE_SHA256,
            'exact independently closed three-case CPU dose timing receipt bytes')
    closeout = base.files.parse(closeout_raw)
    report_raw = base.files.file_bytes(root / 'report.json')
    report_digest = sha256(report_raw).hexdigest()
    report = base.files.parse(report_raw)
    expected_names = {'checkpoint.pt', 'launch.json', 'report.json'} | {
        f'case-{index}{suffix}' for index in range(3)
        for suffix in ('.pt', '.json', '-replay.json')}
    _check_dose_timing_receipt(closeout, report, report_digest,
        expected_names, {path.name for path in root.iterdir()})
    for name, item in closeout['files_rehashed'].items():
        path = root / name
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and type(item) is dict and set(item) == {'sha256', 'bytes'}
                and type(item['bytes']) is int and path.stat().st_size == item['bytes']
                and base.host.digest(path) == item['sha256'],
                'unchanged completed CPU dose timing artifact ' + str(name))
    return dict(source=CLOSED_DOSE_SOURCE, closeout_sha256=CLOSED_DOSE_SHA256,
        report_sha256=report_digest, decision=closeout['decision'],
        whole_cpu_rescore_identical=True, cases_checked=3)


def _check_dose_timing_receipt(closeout, report, report_digest,
                               expected_names, actual_names):
    """Validate parsed receipt semantics; byte pins are checked by the caller."""
    require(report_digest == CLOSED_DOSE_REPORT_SHA256
            and closeout['source'] == report['source'] == CLOSED_DOSE_SOURCE
            and closeout['protocol'] == 'cpu-dose-screen-independent-closeout-v1'
            and report['protocol'] == prior.PROTOCOL
            and closeout['report_sha256'] == report_digest
            and closeout['decision'] == report['decision'] == 'frozen-dose-screen-no-deficit'
            and closeout['whole_cpu_rescore_identical'] is True
            and closeout['cases_checked'] == 3
            and closeout['complete_force_delivery'] is True
            and closeout['prefixes_identical'] is True
            and closeout['optimizer_steps'] == report['optimizer_steps'] == 0
            and closeout['cuda_initialized'] is report['cuda_initialized'] is False
            and closeout['independent_gpu_attestation'] is False
            and closeout['whole_trajectory_physics_resimulated'] is False
            and all(closeout[k] is False and report[k] is False
                    for k in base.baseline.FALSE_FLAGS)
            and set(closeout['files_rehashed']) == expected_names
            and actual_names == expected_names | {'independent-closeout.json'},
            'complete no-deficit three-case CPU timing receipt and static inventory')


def prerequisites():
    """Require CPU ancestry, timing evidence, and the exact qualified CUDA closeout."""
    cpu_receipt, checkpoint_raw = prior.prerequisite()
    require(sha256(checkpoint_raw).hexdigest() == PARENT_CHECKPOINT_SHA256
            == base.baseline.CHECKPOINT_SHA256
            and parent.PARENT_STATE_SHA256 == PARENT_STATE_SHA256,
            'unchanged frozen parent checkpoint and restored state pins')
    cuda_receipt = _closed_cuda_prerequisite()
    dose_timing_receipt = _closed_dose_timing_prerequisite()
    return dict(cpu_dose_ancestry=cpu_receipt, cpu_dose_timing=dose_timing_receipt,
        cuda_integration=cuda_receipt), checkpoint_raw


def _context(source):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CUDA-hidden CPU broad-screen process')
    identity = base.host.identity(source)
    cpu_profile = base.profile.checked_receipt()
    services = base.protected_state()
    require(all(value == 'inactive' for value in services.values()),
            'protected services remain inactive')
    return dict(source_identity=identity, cpu_math_profile=cpu_profile,
        preserved_filmbrain=base.retained.d0.filmbrain_state(),
        protected_services=services)


def _projected_timing():
    require(PROJECTED_SERVICE_SECONDS < SERVICE_SECONDS,
            'measured all-case CPU projection fits the fresh service cap')
    return dict(previous_three_case_seconds=PREVIOUS_THREE_CASE_SECONDS,
        previous_cases=3, planned_cases=len(CELL_IDS), factor=TIMING_FACTOR,
        projected_service_seconds=PROJECTED_SERVICE_SECONDS,
        service_seconds=SERVICE_SECONDS)


def _screen_inputs(scores, prefixes):
    require(type(scores) is list and len(scores) == len(CELL_IDS)
            and type(prefixes) is list and len(prefixes) == len(CELL_IDS),
            'exact ordered 25-case screen inputs')
    candidate_deficits = []
    valid_cases = []
    for index, (cell, score) in enumerate(zip(CELL_IDS, scores)):
        require(type(score) is dict and score.get('cell') == cell
                and score.get('protocol') == evidence.PROTOCOL
                and score.get('strict_actor_restore') is True
                and score.get('actor_replay_max_abs_error') == 0.
                and score.get('whole_trajectory_physics_resimulated') is False
                and score.get('thermal_model_applied') is False
                and all(score.get(key) is False for key in base.baseline.FALSE_FLAGS)
                and type(score.get('numerical_diagnostic')) is dict
                and type(score.get('collection')) is dict
                and type(score.get('pulse')) is dict,
                'ordered independently scored broad-screen row ' + str(index))
        numeric = score['numerical_diagnostic']; collection = score['collection']
        pulse = score['pulse']
        require(type(numeric.get('candidate_pass')) is bool
                and type(numeric.get('complete_first_attempt')) is bool
                and type(collection.get('policy_ticks')) is int
                and 1 <= collection['policy_ticks'] <= POLICY_TICKS
                and type(collection.get('elapsed_seconds')) is float
                and 0 < collection['elapsed_seconds'] < COLLECTION_SECONDS
                and collection.get('stop_reason') in
                    ('policy-tick-limit', 'all-first-attempts-complete', 'wall-budget-exhausted')
                and type(pulse.get('checked_physics_steps')) is int
                and 0 < pulse['checked_physics_steps'] <= PHYSICS_STEPS,
                'typed bounded full-attempt score row')
        pulse_ok = all(pulse.get(key) is True for key in (
            'complete_pulse_delivery', 'complete_phase_checks',
            'recorded_phase_checks_valid', 'exact_full_force_arrays_checked',
            'unforced_post_arrays_checked'))
        terminal_complete = (numeric['complete_first_attempt'] is True
            and collection['stop_reason'] == 'all-first-attempts-complete')
        duration_complete = (collection['policy_ticks'] == POLICY_TICKS
            and pulse['checked_physics_steps'] == PHYSICS_STEPS
            and collection['stop_reason'] in ('policy-tick-limit', 'all-first-attempts-complete'))
        valid_cases.append(bool(pulse_ok and (terminal_complete or duration_complete)))
        if index and numeric['candidate_pass'] is False:
            candidate_deficits.append(cell)

    prefix_valid = all(type(item) is str and len(item) == 64
        and all(char in '0123456789abcdef' for char in item) for item in prefixes)
    prefixes_match = bool(prefix_valid and len(set(prefixes)) == 1)
    zero_control_valid = scores[0]['numerical_diagnostic']['candidate_pass'] is True
    complete = all(valid_cases)
    if not zero_control_valid or not prefixes_match or not complete:
        decision = 'cpu-cardinal-dose-timing-inconclusive'
    elif candidate_deficits:
        decision = 'cpu-cardinal-dose-timing-candidate-deficit'
    else:
        decision = 'cpu-cardinal-dose-timing-no-deficit'
    return dict(protocol=PROTOCOL, decision=decision,
        candidate_deficit_cells=candidate_deficits,
        zero_control_valid=zero_control_valid,
        complete_declared_force_phases=complete,
        prefixes_identical=prefixes_match,
        complete_cases=sum(valid_cases), cases_expected=len(CELL_IDS),
        full_duration_gate_is_acceptance=False,
        promotion_authorized=False, optimizer_steps=0, simulator_resets=0,
        auto_reset=False, **base.baseline.FALSE_FLAGS)


def _write_capture(path, raw):
    base.retained.write_capture(path, raw)


def _capture_metadata(source, index, declaration, raw, value):
    return dict(protocol=PROTOCOL, source=source, index=index,
        cell=declaration['cell_ids'][0], declaration=declaration,
        capture_sha256=sha256(raw).hexdigest(), capture_bytes=len(raw),
        collection=value['collection'], optimizer_steps=0, simulator_resets=0,
        auto_reset=False, **base.baseline.FALSE_FLAGS)


def _case_prefix(raw, digest, checkpoint_raw, declaration, compiled_plant):
    score = evidence.verify(raw, digest, checkpoint_raw, declaration, compiled_plant)
    if score['pulse']['checked_physics_steps'] <= PREFIX_STEP:
        return score, None
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    try:
        prefix = evidence.prefix_hash(value, PREFIX_STEP)
    except ValueError:
        prefix = None
    return score, prefix


def run(source):
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE)
    require(CELL_IDS[0] == ZERO_CELL and len(CELL_IDS) == 25
            and base.evidence.EVALUATION_SEED == EVALUATION_SEED
            and PROJECTED_SERVICE_SECONDS < SERVICE_SECONDS,
            'fixed 25-case matrix and measured run-time projection')
    properties = service_properties(source, 'run')
    initial_context = _context(source)
    prerequisite, checkpoint_raw = prerequisites()
    require(len(checkpoint_raw) > 0 and sha256(checkpoint_raw).hexdigest()
            == PARENT_CHECKPOINT_SHA256, 'exact retained unchanged parent bytes')
    rows = declarations(source)
    root = base.files.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    _write_capture(root / 'checkpoint.pt', checkpoint_raw)
    launch = None
    launch_path = root / 'launch.json'
    report = dict(protocol=PROTOCOL, source=source,
        decision='cpu-cardinal-dose-timing-incomplete', optimizer_steps=0,
        simulator_resets=0, auto_reset=False, **base.baseline.FALSE_FLAGS)
    scores, prefixes, captures = [], [], []
    try:
        from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
        for index, declaration in enumerate(rows):
            window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
            require(time.monotonic() - started < SERVICE_SECONDS - CASE_RESERVE_SECONDS,
                    'reserve next 60-second CPU case and its score/retention')
            require(_context(source) == initial_context,
                    'unchanged source/profile/FilmBrain/protected services before each cell')
            base.seed_cpu()
            env = ScheduledRecoveryRuntime(declaration, device='cpu')
            require(type(env) is ScheduledRecoveryRuntime and env.n == 1
                    and env.live.all() and not env.steps.any()
                    and str(env.device) == 'cpu' and not env.wp_device.is_cuda
                    and env.forward_graph is None and env.solved_field_check == 'packed',
                    'fresh one-world eager-packed CPU scheduled attempt')
            if launch is None:
                launch = dict(protocol=PROTOCOL, source=source, **initial_context,
                    prerequisite=prerequisite, campaign_window=window.declaration(),
                    declarations=rows, cell_ids=list(CELL_IDS),
                    evaluation_seed=EVALUATION_SEED, worlds=1,
                    parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                    parent_state_sha256=PARENT_STATE_SHA256,
                    parent_identity=parent.expected_identity(),
                    policy_ticks=POLICY_TICKS, physics_steps=PHYSICS_STEPS,
                    prefix_step=PREFIX_STEP, collection_seconds=COLLECTION_SECONDS,
                    service_seconds=SERVICE_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
                    launch_reserve_seconds=LAUNCH_RESERVE,
                    per_case_reserve_seconds=CASE_RESERVE_SECONDS,
                    timing_projection=_projected_timing(), service_properties=properties,
                    compiled_plant=env.binding, optimizer_steps=0, simulator_resets=0,
                    auto_reset=False, **base.baseline.FALSE_FLAGS)
                base.files.write_json(launch_path, launch)
            require(env.binding == launch['compiled_plant'],
                    'same actual compiled plant across fresh cardinal cases')
            case_started = time.monotonic()
            try:
                value = evidence.collect(env, checkpoint_raw,
                    deadline_monotonic=min(case_started + COLLECTION_SECONDS,
                        started + SERVICE_SECONDS - CASE_RESERVE_SECONDS),
                    policy_tick_limit=POLICY_TICKS)
            except Exception as exc:
                base.files.write_json(root / f'case-{index}-failure.json', dict(
                    protocol=PROTOCOL, source=source, index=index,
                    cell=declaration['cell_ids'][0], stage='collection',
                    error_type=type(exc).__name__, error=str(exc),
                    current_case_trace_retained=False, **base.baseline.FALSE_FLAGS))
                raise
            raw = evidence.encode(value)
            capture = _capture_metadata(source, index, declaration, raw, value)
            _write_capture(root / f'case-{index}.pt', raw)
            base.files.write_json(root / f'case-{index}.json', capture)
            del env, value
            gc.collect()

            # Retention and its whole-byte digest precede actor/plant/control,
            # force-phase, numerical, or prefix judgment for this case.
            score, prefix = _case_prefix(raw, capture['capture_sha256'],
                checkpoint_raw, declaration, launch['compiled_plant'])
            base.files.write_json(root / f'case-{index}-replay.json', score)
            scores.append(score); prefixes.append(prefix); captures.append(capture)
            del raw
            gc.collect()
            window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
            require(_context(source) == initial_context,
                    'unchanged source/profile/FilmBrain/protected services after each cell')

        screen = _screen_inputs(scores, prefixes)
        require(_context(source) == initial_context and not torch.cuda.is_initialized()
                and time.monotonic() - started < SERVICE_SECONDS,
                'unchanged bounded CPU broad-screen closeout')
        window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
        qualification = dict(protocol=PROTOCOL, source=source,
            captures=captures, scores=scores, prefixes=prefixes,
            screening=screen, elapsed_seconds=float(time.monotonic() - started),
            cpu_initialized_only=True, optimizer_steps=0, simulator_resets=0,
            auto_reset=False, **base.baseline.FALSE_FLAGS)
        base.files.write_json(root / 'cpu-qualification.json', qualification)
        # Launch is now bound to the actual compiled plant and qualification.
        require(launch is not None, 'actual compiled plant launch binding retained')
        qualification_sha256 = base.host.digest(root / 'cpu-qualification.json')
        launch_sha256 = base.host.digest(launch_path)
        report.update(decision=screen['decision'], screening=screen,
            launch_sha256=launch_sha256,
            qualification_sha256=qualification_sha256,
            service_properties=properties, elapsed_seconds=float(time.monotonic() - started),
            genuine_cpu_first_attempts=len(CELL_IDS), cuda_initialized=False,
            source_unchanged=True, filmbrain_unchanged=True,
            protected_services_inactive=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc),
            error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['elapsed_seconds'] = float(time.monotonic() - started)
        report['files'] = {p.name: base.host.digest(p)
                           for p in sorted(root.iterdir()) if p.is_file()}
        base.files.write_json(root / 'report.json', report)
    return dict(output=str(root), launch_sha256=launch_sha256,
        decision=report['screening']['decision'])


def checked(source, launch_sha256):
    base.files.hex_id(launch_sha256, 64)
    root = output_path(source)
    launch_raw = base.files.file_bytes(root / 'launch.json')
    require(sha256(launch_raw).hexdigest() == launch_sha256,
            'whole broad-screen launch bytes before parse')
    launch = base.files.parse(launch_raw)
    current = _context(source)
    require(launch['protocol'] == PROTOCOL and launch['source'] == source
            and {key: launch[key] for key in current} == current
            and launch['campaign_window'] == window.declaration()
            and launch['declarations'] == declarations(source)
            and launch['cell_ids'] == list(CELL_IDS)
            and launch['evaluation_seed'] == EVALUATION_SEED and launch['worlds'] == 1
            and launch['parent_checkpoint_sha256'] == PARENT_CHECKPOINT_SHA256
            and launch['parent_state_sha256'] == PARENT_STATE_SHA256
            and launch['parent_identity'] == parent.expected_identity()
            and launch['policy_ticks'] == POLICY_TICKS
            and launch['physics_steps'] == PHYSICS_STEPS
            and launch['prefix_step'] == PREFIX_STEP
            and launch['collection_seconds'] == COLLECTION_SECONDS
            and launch['service_seconds'] == SERVICE_SECONDS
            and launch['closeout_seconds'] == CLOSEOUT_SECONDS
            and launch['launch_reserve_seconds'] == LAUNCH_RESERVE
            and launch['per_case_reserve_seconds'] == CASE_RESERVE_SECONDS
            and launch['timing_projection'] == _projected_timing()
            and launch['optimizer_steps'] == 0 and launch['simulator_resets'] == 0
            and launch['auto_reset'] is False
            and all(launch[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact source/profile/window/parent/cardinal launch binding')
    prerequisite, checkpoint_raw = prerequisites()
    require(launch['prerequisite'] == prerequisite
            and sha256(checkpoint_raw).hexdigest() == PARENT_CHECKPOINT_SHA256
            and base.files.file_bytes(root / 'checkpoint.pt',
                limit=base.retained.evaluation.checkpoint.LIMIT) == checkpoint_raw,
            'unchanged closed prerequisites and immutable parent bytes')
    qualification_raw = base.files.file_bytes(root / 'cpu-qualification.json')
    q = base.files.parse(qualification_raw)
    require(q['protocol'] == PROTOCOL and q['source'] == source
            and len(q['captures']) == len(q['scores']) == len(q['prefixes']) == len(CELL_IDS)
            and type(q['elapsed_seconds']) is float
            and 0 < q['elapsed_seconds'] < SERVICE_SECONDS
            and q['cpu_initialized_only'] is True
            and all(q[key] is False for key in base.baseline.FALSE_FLAGS)
            and q['optimizer_steps'] == 0 and q['simulator_resets'] == 0
            and q['auto_reset'] is False,
            'complete bounded CPU qualification receipt')
    return launch, checkpoint_raw, q, sha256(qualification_raw).hexdigest()


def replay(source, launch, checkpoint_raw, qualification):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'CUDA-hidden broad-screen replay')
    root = output_path(source); scores = []; prefixes = []
    for index, (cell, declaration) in enumerate(zip(CELL_IDS, launch['declarations'])):
        raw = base.files.file_bytes(root / f'case-{index}.pt', limit=evidence.LIMIT)
        capture = base.files.parse(base.files.file_bytes(root / f'case-{index}.json'))
        expected_capture = qualification['captures'][index]
        require(capture == expected_capture and capture['protocol'] == PROTOCOL
                and capture['source'] == source and capture['index'] == index
                and capture['cell'] == cell and capture['declaration'] == declaration
                and capture['capture_bytes'] == len(raw)
                and sha256(raw).hexdigest() == capture['capture_sha256']
                and capture['optimizer_steps'] == 0 and capture['simulator_resets'] == 0
                and capture['auto_reset'] is False
                and all(capture[key] is False for key in base.baseline.FALSE_FLAGS),
                'exact retained raw cardinal case metadata')
        score = evidence.verify(raw, capture['capture_sha256'], checkpoint_raw,
            declaration, launch['compiled_plant'])
        retained_score = base.files.parse(base.files.file_bytes(
            root / f'case-{index}-replay.json'))
        require(score == retained_score == qualification['scores'][index]
                and score['collection'] == capture['collection'],
                'fresh independent whole-trace score equals retained receipt')
        prefix = None
        if score['pulse']['checked_physics_steps'] > PREFIX_STEP:
            value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            try:
                prefix = evidence.prefix_hash(value, PREFIX_STEP)
            except ValueError:
                prefix = None
        scores.append(score); prefixes.append(prefix)
        del raw
        gc.collect()
    screen = _screen_inputs(scores, prefixes)
    require(prefixes == qualification['prefixes']
            and screen == qualification['screening'],
            'deterministic fresh broad-screen decision and prefix comparison')
    return dict(scores=scores, prefixes=prefixes, screening=screen)


def closeout(source, launch_sha256):
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS)
    properties = service_properties(source, 'closeout')
    launch, checkpoint_raw, qualification, qualification_sha256 = checked(source, launch_sha256)
    root = output_path(source)
    report_raw = base.files.file_bytes(root / 'report.json')
    report = base.files.parse(report_raw)
    require(report.get('protocol') == PROTOCOL and report.get('source') == source
            and report.get('launch_sha256') == launch_sha256
            and report.get('qualification_sha256') == qualification_sha256
            and report.get('optimizer_steps') == 0
            and report.get('simulator_resets') == 0
            and report.get('auto_reset') is False
            and report.get('cuda_initialized') is False
            and all(report.get(key) is False for key in base.baseline.FALSE_FLAGS)
            and report.get('decision') == qualification['screening']['decision']
            and report.get('screening') == qualification['screening']
            and report['screening']['decision'] in (
                'cpu-cardinal-dose-timing-no-deficit',
                'cpu-cardinal-dose-timing-candidate-deficit',
                'cpu-cardinal-dose-timing-inconclusive'),
            'complete qualified/rejected broad-screen report')
    require(set(report['files']) == PREPARE_FILES
            and {path.name for path in root.iterdir()} == COMPLETE_FILES,
            'exact static complete broad-screen inventory')
    inventory = {}
    for name, digest in report['files'].items():
        path = root / name
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and base.host.digest(path) == digest,
                'whole retained broad-screen file hash ' + str(name))
        inventory[name] = dict(sha256=digest, bytes=path.stat().st_size)
    rescored = replay(source, launch, checkpoint_raw, qualification)
    require(rescored['screening'] == report['screening'],
            'fresh whole CPU rescore and exact deterministic decision')
    current = _context(source)
    require(current == {key: launch[key] for key in current}
            and time.monotonic() - started < CLOSEOUT_SECONDS,
            'unchanged bounded independent CPU closeout')
    window.check()
    inventory['report.json'] = dict(sha256=sha256(report_raw).hexdigest(),
        bytes=len(report_raw))
    result = closeout_result(source, launch_sha256, report_raw, rescored, inventory,
                             properties, float(time.monotonic() - started))
    base.files.write_json(root / 'independent-closeout.json', result)
    return result


def closeout_result(source, launch_sha256, report_raw, rescored, inventory, properties, elapsed):
    """Receipt construction only; native validation is exclusively in closeout."""
    return dict(protocol=PROTOCOL, source=source,
        launch_sha256=launch_sha256, report_sha256=sha256(report_raw).hexdigest(),
        decision=rescored['screening']['decision'], screening=rescored['screening'],
        files_rehashed=inventory, cases_checked=len(CELL_IDS),
        whole_cpu_rescore_identical=True, cuda_initialized=False,
        whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, service_properties=properties,
        optimizer_steps=0, simulator_resets=0, auto_reset=False,
        elapsed_seconds=elapsed,
        **base.baseline.FALSE_FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'closeout'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--launch-sha256')
    args = parser.parse_args(argv)
    if args.mode == 'run':
        print(run(args.source))
    else:
        require(args.launch_sha256 is not None, 'closeout requires exact launch digest')
        print(closeout(args.source, args.launch_sha256))


if __name__ == '__main__':
    main()
