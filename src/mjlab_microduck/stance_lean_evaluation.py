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
# Measured on 100.100 by the declared probe at source cdb05a5667ca on 2026-09-17.
# The probe ran one full-length case at iteration 255 / seed 541 and recorded
# 0.0008308729156851768 s prelude, 0.719240453094244 s environment construction
# and 93.38529451098293 s of case work. That gives a 94.10453496407717 s
# repeating unit, a 1,129.2552504418418 s prediction for twelve cases, and a
# 1,412 s service cap at the declared 1.25 factor.
#
# These are a transcription, not the authority. ``measured_caps`` re-derives them
# from the retained probe report on every launch, so a hand-edited constant, or a
# probe re-run that produced different numbers, is refused rather than launched.
PROBE_SOURCE = 'cdb05a5667ca0738e06f87737c1fda51203f12b3'
PROBE_REPORT = 'a91b5a068be96def6ae7fa18d7109ea75cb10b64f870a7d3991515589b80db58'
CHILD_SECONDS, SERVICE_SECONDS = 1352, 1412

# --- the judged continuations -------------------------------------------------
# The two weight-initialized continuations this comparison path judges. They
# differ in purpose, trace protocol, evidence directory and decision strings, and
# in whether the training archive they read is pinned in advance or named by the
# caller. Everything else -- the twelve cases, the scorer, the 122/128 per-seed
# threshold and the 0.0873 rad gate -- is shared by reference, not re-typed.
#
# The probe stays lesson-only on purpose. It has already run, its measurement is
# the shared basis for both purposes' caps, and a replication must not be able to
# re-measure its way to a different bound.
LESSON = dict(label='lean-lesson', protocol=PROTOCOL, purpose=checkpoint.LEAN_PURPOSE,
              trace_protocol=trace.LEAN_PROTOCOL, iterations=checkpoint.LEAN_CHECKPOINTS,
              directories={'probe': 'stance-lean-eval-probe-',
                           'evaluate': 'stance-lean-evaluation-'},
              services={'probe': 'microduck-lean-eval-probe-',
                        'evaluate': 'microduck-lean-evaluation-'},
              path_seed=False, case_prefix='lean', training_source=TRAINING_SOURCE,
              training_report=TRAINING_REPORT, gate_key='lean_lesson_numerical_gate_passed',
              passed='lean-lesson-passed-nominal', rejected='lean-lesson-rejected-objective-binds')
REPLICATION = dict(label='lean-replication',
                   protocol='football-b1n-lean-replication-evaluation-v1',
                   purpose=checkpoint.LEAN_REPLICATION_PURPOSE,
                   trace_protocol=trace.LEAN_REPLICATION_PROTOCOL,
                   iterations=checkpoint.LEAN_REPLICATION_CHECKPOINTS,
                   directories={'probe': 'stance-lean-repl-eval-probe-',
                                'evaluate': 'stance-lean-replication-eval-'},
                   services={'probe': 'microduck-lean-repl-eval-probe-',
                             'evaluate': 'microduck-lean-replication-eval-'},
                   path_seed=True, case_prefix='repl', training_source=None, training_report=None,
                   gate_key='lean_replication_numerical_gate_passed',
                   passed='lean-replication-seed-passed',
                   rejected='lean-replication-seed-rejected')
EVALUATIONS = {d['label']: d for d in (LESSON, REPLICATION)}


def evaluation_of(label):
    require(label in EVALUATIONS, 'declared judged continuation')
    return EVALUATIONS[label]


def cases_of(declaration):
    return len(declaration['iterations'])*len(trace.SEEDS)


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


def derive_caps(measurements):
    """The predeclared cap rule, applied to measured probe timings.

    Takes the measurements block itself, not a report: the rule is about three
    numbers, and keeping it separate from which report they came from is what
    stops a caller from handing it a document that merely looks like one.
    ``unit`` is what repeats per case; ``prelude`` happens once. The 1.25 factor
    is applied to the whole prediction, never to a mean.
    """
    require(type(measurements) is dict, 'probe measurements')
    for key in ('prelude_seconds', 'env_seconds', 'case_seconds'):
        value = measurements.get(key)
        require(type(value) is float and math.isfinite(value) and value > 0,
                'finite positive probe timing: '+key)
    unit = measurements['env_seconds']+measurements['case_seconds']
    predicted = measurements['prelude_seconds']+CASES*unit
    service = math.ceil(PROBE_SAFETY*predicted)
    return dict(unit_seconds=unit, predicted_seconds=predicted, service_seconds=service,
        child_seconds=service-WATCHDOG_MARGIN_SECONDS, safety_factor=PROBE_SAFETY,
        closeout_seconds=PROBE_CLOSEOUT_SECONDS, cases=CASES)


def probe_measurements_of(report):
    """The measurements block of a retained probe report, with its decision checked."""
    require(report.get('protocol') == PROTOCOL and report.get('mode') == 'probe',
            'a lean evaluation probe report')
    require(report.get('decision') == 'probe-measured', 'a probe that measured a full-length case')
    require(type(report.get('probe')) is dict, 'probe result block')
    return report['probe']['measurements']


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
    derived = derive_caps(probe_measurements_of(files.parse(raw)))
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


def parser():
    """The one command line, shared by ``main`` and the supervisor's child argv.

    ``supervise`` builds a child command line that this same parser must accept.
    Keeping the parser in a function is what lets a test check that the two
    agree, instead of discovering a mismatch only when a launch fails.
    """
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    result.add_argument('--source', required=True)
    # ``mode`` is already the action (prepare/supervise/child); the job is which
    # evaluation this is, so it gets its own name.
    result.add_argument('--job', choices=MODES)
    # And which continuation is being judged. Named separately from ``--job``
    # because ``--job`` already means probe-versus-evaluate here.
    result.add_argument('--continuation', choices=tuple(EVALUATIONS), default=LESSON['label'])
    result.add_argument('--seed', type=int)
    # Which completed training run to judge. Required by a continuation that does
    # not pin one, refused by one that does.
    result.add_argument('--training-source')
    result.add_argument('--deadline-unix', type=int)
    result.add_argument('--launch-sha256')
    result.add_argument('--lock-fd', type=int)
    return result


def child_command(source, launch_sha, mode, fd, declaration=LESSON, seed=None):
    """The exact argv the supervisor hands to its bounded child."""
    files.hex_id(source, 40); files.hex_id(launch_sha, 64)
    require(mode in MODES and type(fd) is int, 'declared lean evaluation mode and lease fd')
    argv = [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
            '--launch-sha256', launch_sha, '--job', mode, '--lock-fd', str(fd),
            '--continuation', declaration['label']]
    # Omitted for a single-seed continuation, whose seed is fixed by its own
    # declaration; supplied for one whose runs share a commit and differ by seed.
    return argv if seed is None else argv+['--seed', str(seed)]


# Which loader may read the judged run's exports, keyed by the same label the
# declaration carries. The replication's exports are not readable through the
# lesson's loader and vice versa, so a comparison can never judge one purpose's
# archive through the other purpose's identity rules.
TRAINING_LOADERS = {LESSON['label']: checkpoint.load_lean_evaluation,
                    REPLICATION['label']: checkpoint.load_lean_replication_evaluation}


def training_inputs(declaration=LESSON, seed=None, source=None):
    """Authenticate the immutable training archive before any tensor loading.

    The lesson's archive is pinned in advance, by source and report hash. A
    replication's cannot be: it is produced by a run that does not exist yet. So
    its source is named by the caller and its report hash is *derived* from the
    archive, then frozen into the launch document that every later step re-checks
    against. Nothing downstream can re-point the comparison at a different run.
    """
    training = lean.declaration_of(declaration['label'])
    seed = training['seeds'][0] if seed is None else seed
    lean.seed_of(training, seed)
    # A pinned continuation names its own source and must not be re-pointed by a
    # caller; one that names its source at prepare time requires it.
    if declaration['training_source'] is not None:
        require(source is None or source == declaration['training_source'],
                'a pinned continuation names its own training source')
    source = declaration['training_source'] if source is None else source
    require(type(source) is str, 'a declared training source to judge')
    root = lean.output_path(source, training, seed)
    raw = files.file_bytes(root/'report.json')
    digest = sha256(raw).hexdigest()
    if declaration['training_report'] is not None:
        require(digest == declaration['training_report'], 'independent training report hash')
    report = files.parse(raw)
    require(report['decision'] == training['decision'] and report['child']['returncode'] == 0,
            'successful completed training run')
    require({p.name for p in root.iterdir()} == set(report['files'])|{'report.json'},
            'exact training archive inventory')
    for name, file_digest in report['files'].items():
        require(name == os.path.basename(name) and name not in ('.', '..'),
                'plain training filename')
        require(host.digest(root/name) == file_digest, 'immutable training archive hash: '+name)
    launch = files.parse(files.file_bytes(root/'launch.json'))
    completed = lean.verify_completed(root, source, report['launch_sha256'], launch, training, seed)
    by_iteration = {c['identity']['iteration']: c for c in completed['checkpoints']}
    require(set(by_iteration) == set(range(-1, lean.UPDATES)),
            'every training checkpoint present exactly once')
    selected = [by_iteration[iteration] for iteration in declaration['iterations']]
    require([c['identity']['iteration'] for c in selected] == list(declaration['iterations']),
            'fixed common-checkpoint selection')
    for saved in selected:
        # The evaluable-iteration admission is enforced by the loader, not here.
        TRAINING_LOADERS[declaration['label']](
            files.file_bytes(root/saved['file']), saved['sha256'], saved['identity'])
    # Deliberately the same three keys as before the replication existed. The
    # launch document embeds this block verbatim and ``summarize`` re-derives the
    # whole plan and requires equality, so adding a key here would make every
    # already-retained lesson launch document fail its own re-verification.
    return dict(source=source, report_sha256=digest, checkpoints=selected)


def plan(source, inputs, runtime_sha, retained, deadline, mode, declaration=LESSON):
    files.hex_id(source, 40); files.hex_id(runtime_sha, 64)
    require(mode in MODES and type(deadline) is int and deadline > 0,
            'declared lean evaluation mode and explicit deadline')
    # Resolve the declared caps first: an unmeasured twelve-case budget must fail
    # before any archive authentication, hash work or directory creation.
    caps = service_caps(mode)
    iterations = declaration['iterations']
    # A pinned declaration is checked against its constant here. One that names
    # its source at prepare time has nothing to compare against at plan time: its
    # source arrives inside ``retained`` and is bound by the launch hash, which
    # ``inputs_check`` re-derives on every run.
    if declaration['training_source'] is not None:
        require(retained['source'] == declaration['training_source'],
                'the declared completed training run')
    if declaration['training_report'] is not None:
        require(retained['report_sha256'] == declaration['training_report'],
                'fixed completed training report')
    require([c['identity']['iteration'] for c in retained['checkpoints']] == list(iterations),
            'all common checkpoints of the declared continuation')
    selected = retained['checkpoints'] if mode == 'evaluate' else [
        c for c in retained['checkpoints'] if c['identity']['iteration'] == PROBE_ITERATION]
    seeds = trace.SEEDS if mode == 'evaluate' else (PROBE_SEED,)
    require(len(selected) == (len(iterations) if mode == 'evaluate' else 1),
            'declared checkpoint coverage')
    cases = []
    for saved in selected:
        meta = saved['identity']
        checkpoint.validate_identity(meta, evaluation=declaration['purpose'])
        require(meta['purpose'] == declaration['purpose'] and meta['source'] == retained['source']
                and meta['iteration'] in iterations, 'distinct checkpoint identity')
        files.hex_id(saved['sha256'], 64)
        require(saved['file'] == f'model_{meta["iteration"]}.pt', 'exact checkpoint filename')
        for evaluation_seed in seeds:
            binding = dict(protocol=declaration['trace_protocol'], source=source,
                runtime_sha256=runtime_sha, checkpoint_sha256=saved['sha256'],
                checkpoint_iteration=meta['iteration'], evaluation_seed=evaluation_seed,
                worlds=WORLDS, capture_device='cuda:0')
            launch = bundle.launch_bytes(binding, meta); binding['launch_sha256'] = sha256(launch).hexdigest()
            cases.append(dict(
                name=f'{declaration["case_prefix"]}-{meta["iteration"]}-seed-{evaluation_seed}',
                binding=binding, checkpoint=deepcopy(saved)))
    return dict(protocol=declaration['protocol'], mode=mode, source=source, inputs=inputs,
        runtime_sha256=runtime_sha,
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


def seed_suffix(declaration, seed):
    """The training-seed path component, present only where a run needs one.

    A replication's three runs share one commit, so a source-keyed directory
    would make them overwrite each other. The lesson's evidence path is already
    retained and must not move, so it carries no seed component.
    """
    if not declaration['path_seed']: return ''
    training = lean.declaration_of(declaration['label'])
    return f'-seed-{lean.seed_of(training, seed)}'


def lean_output_path(source, mode, declaration=LESSON, seed=None):
    files.hex_id(source, 40)
    require(mode in MODES, 'declared lean evaluation mode')
    return host.ROOT/'artifacts/evaluations'/(
        declaration['directories'][mode]+source[:12]+seed_suffix(declaration, seed))


def output_path(source, mode, declaration=LESSON, seed=None):
    return lean_output_path(source, mode, declaration, seed)


def service_name(source, mode, declaration=LESSON, seed=None):
    files.hex_id(source, 40)
    require(mode in MODES, 'declared lean evaluation mode')
    return declaration['services'][mode]+source[:12]+seed_suffix(declaration, seed)+'.service'


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


def prepare(source, deadline, mode, declaration=LESSON, seed=None, training_source=None):
    require(host.execution.PROFILE['name'] != host.execution.WSL,
            'WSL full evaluation needs its separately measured timing gate')
    check_window(deadline, mode, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean evaluation preparation')
    inputs = host.identity(source)
    retained = training_inputs(declaration, seed, training_source)
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, inputs, sha256(runtime).hexdigest(), retained, deadline, mode, declaration)
    root = files.native._plain_path(output_path(source, mode, declaration, seed))
    root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runtime.json', runtime); files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source, mode, declaration, seed), mode=mode,
        cases=len(launch['cases']), launch_sha256=host.digest(root/'launch.json'))


def inputs_check(source, launch_sha, mode, declaration=LESSON, seed=None):
    root = output_path(source, mode, declaration, seed)
    raw = files.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'independent lean evaluation launch hash')
    launch = files.parse(raw); runtime = files.file_bytes(root/'runtime.json')
    # The judged run's source is read back from the hash-bound launch document
    # rather than taken from the caller, so it cannot be re-pointed after prepare.
    retained = training_inputs(declaration, seed, launch['retained_training']['source'])
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest(),
        retained, launch['deadline_unix'], mode, declaration), 'exact lean evaluation plan')
    plant.checked_runtime(files.parse(runtime), source)
    return launch


def check_service(source, mode, declaration=LESSON, seed=None):
    caps = service_caps(mode)
    values = {k: host.read('systemctl', '--user', 'show',
                           service_name(source, mode, declaration, seed), '-p', k, '--value')
              for k in SERVICE_PROPERTIES}
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec=caps['runtime_max'],
                           KillMode='control-group', ActiveState='active'),
            'independently timed lean evaluation service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean evaluation supervisor')


def run_cases(launch, root, runtime, deadline, declaration=LESSON):
    """Run every declared case through the one shared measured path.

    Timings are recorded in both modes so the probed path is the path the
    twelve-case run takes: the probe reads them, the evaluation ignores them.
    """
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    training = lean.declaration_of(declaration['label'])
    started = time.monotonic(); scores = {}; receipts = []
    for index, case in enumerate(launch['cases']):
        require(time.monotonic() < deadline, 'whole lean comparison time budget')
        seed = case['binding']['evaluation_seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        saved = case['checkpoint']; binding = case['binding']
        # Read the training seed off the checkpoint identity rather than taking it
        # as another argument: the identity is what the archive itself attests, so
        # it cannot disagree with the run the weights actually came from.
        cp_raw = files.file_bytes(lean.output_path(
            launch['retained_training']['source'], training,
            saved['identity']['training_seed'])/saved['file'])
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


def child(source, launch_sha, fd, mode, declaration=LESSON, seed=None):
    smoke.inherited_lease(fd)
    launch = inputs_check(source, launch_sha, mode, declaration, seed)
    require(launch['mode'] == mode, 'child mode matches the launch document')
    check_window(launch['deadline_unix'], mode, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'explicit evaluation CUDA0')
    host.wait_idle()
    root = output_path(source, mode, declaration, seed)
    runtime = files.file_bytes(root/'runtime.json')
    end = time.monotonic()+launch['child_timeout_seconds']-30
    scores, receipts = run_cases(launch, root, runtime, end, declaration)
    require(inputs_check(source, launch_sha, mode, declaration, seed) == launch,
            'unchanged lean comparison inputs')
    if mode == 'probe':
        files.write_json(root/'measurements.json', dict(protocol=PROTOCOL, mode='probe',
            launch_sha256=launch_sha, measurements=probe_measurements(launch, receipts)))
        return
    files.write_json(root/'comparison.json', dict(launch_sha256=launch_sha, cases=receipts,
                                                summary=summarize(launch, scores, declaration)))


def summarize(launch, scores, declaration=LESSON):
    """The predeclared decision rule, applied to twelve verified case scores."""
    require(launch['mode'] == 'evaluate', 'a probe is not a decision')
    require(launch == plan(launch['source'], launch['inputs'], launch['runtime_sha256'],
        launch['retained_training'], launch['deadline_unix'], 'evaluate', declaration),
        'exact fixed comparison plan')
    require(set(scores) == {c['name'] for c in launch['cases']}, 'all twelve cases required')
    rows = []
    for case in launch['cases']:
        score = scores[case['name']]; attempts = score['attempts']
        require(score['binding'] == case['binding'] and score['protocol'] == declaration['trace_protocol'],
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
    for iteration in declaration['iterations']:
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
    survivors = [i for i in declaration['iterations'] if per_checkpoint[str(i)]['passes_all_seeds']]
    passed = bool(survivors)
    # The gate key is named by the declaration rather than shared, so a
    # replication summary cannot be read as a lesson summary. The lesson's key is
    # retained verbatim: it is already inside a hashed, retained comparison
    # document, and renaming it would break that document's re-verification.
    return dict(protocol=declaration['protocol'], mode='evaluate', rows=rows,
        per_checkpoint=per_checkpoint,
        complete_attempts=ATTEMPTS, passing_checkpoints=survivors,
        **{declaration['gate_key']: passed},
        decision=declaration['passed'] if passed else declaration['rejected'],
        tilt_gate_rad=TILT_GATE_RAD, tilt_gate_relaxed=False,
        checkpoint_admitted=False, learned_stance_accepted=False, football_balance_accepted=False,
        physical_motion_authorized=False)


def verify(root, launch, comparison_sha, declaration=LESSON):
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
    summary = summarize(launch, scores, declaration)
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
    return dict(measurements=recorded['measurements'],
                derived_caps=derive_caps(recorded['measurements']))


def supervise(source, launch_sha, mode, declaration=LESSON, seed=None):
    check_service(source, mode, declaration, seed)
    launch = inputs_check(source, launch_sha, mode, declaration, seed)
    require(launch['mode'] == mode, 'supervisor mode matches the launch document')
    check_window(launch['deadline_unix'], mode, launching=True)
    root = output_path(source, mode, declaration, seed)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'},
            'one fresh lean evaluation attempt')
    report = dict(protocol=declaration['protocol'], mode=mode, launch_sha256=launch_sha,
        decision='failed', tilt_gate_rad=TILT_GATE_RAD, tilt_gate_relaxed=False, optimizer_steps=0,
        checkpoint_admitted=False, learned_stance_accepted=False, physical_motion_authorized=False)
    try:
        with files.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix'], mode); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live lean evaluation inputs drift')
            report['child'] = supervisor_wrapper(mode)(
                child_command(source, launch_sha, mode, fd, declaration, seed),
                root/'child.log', cwd=host.ROOT, env=files.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            if mode == 'probe':
                report['measurements_sha256'] = host.digest(root/'measurements.json')
                report['probe'] = verify_probe(root, launch, launch_sha, report['measurements_sha256'])
                report['decision'] = 'probe-measured'
            else:
                report['comparison_sha256'] = host.digest(root/'comparison.json')
                report['summary'] = verify(root, launch, report['comparison_sha256'], declaration)
                report['decision'] = report['summary']['decision']
            report['idle_after'] = host.wait_idle()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json', report)


def main():
    args = parser().parse_args()
    declaration = evaluation_of(args.continuation)
    if args.mode == 'prepare':
        print(canonical(prepare(args.source, args.deadline_unix, args.job, declaration,
                                args.seed, args.training_source)))
    elif args.mode == 'supervise':
        supervise(args.source, args.launch_sha256, args.job, declaration, args.seed)
    else:
        child(args.source, args.launch_sha256, args.lock_fd, args.job, declaration, args.seed)


if __name__ == '__main__': main()
