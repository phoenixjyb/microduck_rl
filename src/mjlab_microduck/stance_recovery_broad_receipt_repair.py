"""Bounded CPU re-audit of immutable captures after a closeout-constructor failure.

This protocol never collects a simulator attempt and never repairs or rewrites
the original broad-screen directory. It independently replays its 25 captures
and binds the result to the failed closeout invocation and the current clean
evaluator source.
"""
import argparse
import ast
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import time

import torch

from mjlab_microduck import stance_recovery_broad_screen as broad
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1d-cpu-cardinal-dose-timing-receipt-repair-v1'
ARTIFACT_SOURCE = '1079a104a9e3bf83d56c0586cd2dd0374990e259'
ARTIFACT_LAUNCH_SHA256 = 'cd016be8c088842f2697611d54047f937ff85f07298f347e6fcb7624adcfa7f8'
ARTIFACT_REPORT_SHA256 = '2cfce295e32317d0ef4dafe201859b932235406679b1d4e66e850404dd39c783'
ARTIFACT_BROAD_SHA256 = '1c62fc70a4a0e6669f99dd8a115a6867b3dab1dca2a33f50810a2fef2340aedc'
ARTIFACT_BROAD_GIT_BLOB = 'f8b87e612f77f1ad1e3ab4811fa1b5d07574e043'
ARTIFACT_BROAD_PREFIX_SHA256 = '1839b226e72c62df644edd67214d4f4c9ecc4b25b4763f64d86163d80a968546'
FAILED_INVOCATION = '5cf35c7873924d9d9218c3d8764c0b56'
# Immutable original closeout-failure evidence, retained before source repair.
FAILURE_RECEIPT_SHA256 = '5d64c57fe65e3e38272f8f4aeafeef81017d338a499ffb4744073477f03a5cb2'
FAILURE_JOURNAL_SHA256 = '8d4f162edab25b285f15bc294b30b7ce8ef5d728f3533545cffefd55cb763bc2'
FAILURE_RECEIPT_RELATIVE_PATH = 'artifacts/tools/cardinal-closeout-failure-1079a104a9e3/receipt.json'
FAILURE_JOURNAL_RELATIVE_PATH = 'artifacts/tools/cardinal-closeout-failure-1079a104a9e3/journal.log'

SERVICE_SECONDS = 600
WINDOW_MARGIN_SECONDS = 60
LAUNCH_RESERVE_SECONDS = SERVICE_SECONDS + WINDOW_MARGIN_SECONDS
MEMORY_BYTES = 2 * 1024**3
CPU_QUOTA = '2s'
NICE = '10'
KILL_MODE = 'control-group'
OUTPUT_PREFIX = 'stance-wsl-cpu-broad-receipt-repair-'
ARTIFACT_RELATIVE_PATH = Path('artifacts/evaluations') / (
    'stance-wsl-cpu-cardinal-dose-timing-' + ARTIFACT_SOURCE[:12])
BROAD_SOURCE_PATH = Path('src/mjlab_microduck/stance_recovery_broad_screen.py')

# Whole-file source pins cover every primitive in the fresh CPU scorer's
# transitive replay path at the failed evaluator revision.
SOURCE_DEPENDENCIES = {
    'src/mjlab_microduck/stance_recovery_schedule_trace.py': '1b0ef1e4d4a562a694e40fd0e12b9f87ba32ca5a6446a1df0d88f6766a7b6213',
    'src/mjlab_microduck/stance_recovery_schedule.py': '06e6395f0e59bc54817d79002e34e4ed677e40bba5fd9261b5e94e67129ab670',
    'src/mjlab_microduck/stance_attempt_trace.py': 'fe63ff56192e76b548f6b98797b0f628ddf3eb94cedafac9e996ffe063ec32dc',
    'src/mjlab_microduck/stance_checkpoint.py': '5fe6d52a0114d1341353e55f4b33b8ce25b80d3415b93b4be1e86790087f4fc7',
    'src/mjlab_microduck/stance_control_evidence.py': 'fd2e3b6679fec725cc9173c676d596fbb3a36639941fb01b1b44cae47a6dec19',
    'src/mjlab_microduck/stance_cpu_replay_profile.py': '8d16d8955ffcdcac46d858eb9ec995b4cffda1065d12bde27e9c6aefe4e91956',
    'src/mjlab_microduck/stance_disturbance_fixture.py': '8cf75fdd0d5428415a4a3abef08f6e2b0ffad29cfcbd87133ef2d1729fc975cd',
    'src/mjlab_microduck/stance_forward_probe.py': 'c5502f027121f06643a58299408a8d7a0fcfc77f2181cf9bea3cf1184bc0d63c',
    'src/mjlab_microduck/stance_plant_evidence.py': '280e643d290466c72b761e551ac3b69127497c1bcc904ac64985e1d1e7686deb',
    'src/mjlab_microduck/stance_recovery_contract.py': '640e0535d82957985f3919afd123e89f4134b9e6ed6cb3bb8208c429b515b849',
    'src/mjlab_microduck/stance_recovery_parent.py': '7975224d9da05e47f75618c95c7bad56ea91f29729b895cf2ac09c1c01211c94',
    'src/mjlab_microduck/stance_recovery_matrix.py': 'ce3238b6b1b238fbfb3f6196169a079af505219111fea3b5cbf2ea012721f0e8',
    'src/mjlab_microduck/stance_transition.py': 'fa9d809b8c67568bc21137add70706a70b892e838fec1088185301a336edfa8a',
    'src/mjlab_microduck/stance_evaluation.py': '8a4da46f3133c1eb00255dbda567d7b57a703fc2696cd1fca9a45c18f925a63d',
    'src/mjlab_microduck/stance_recovery_lesson_plan.py': '0671f34ae9a85492ca99ab0d0e2a3743e63669306d8a9d9a038a152f08754aad',
    'src/mjlab_microduck/first_attempt_smoke.py': '01bb1a0e031974b03b79280f417f742d02771eea5aab048d2d3bcf1b495a1d3b',
}
BROAD_FUNCTION_SHA256 = {
    'declarations': 'f748f61f387461ba03061a46479eb4d0fccd1e4f94d9c63e4e9223618459ad94',
    'prerequisites': '1d821b71f1e77969b5708d0ab8b4ca60c564df02b04ef94c47c71f0d194ca1e6',
    '_screen_inputs': '2bdf55503b3c5d7a9b916372e7d7b96b0547fbc69c4bda8f6013a101a31363d8',
    '_capture_metadata': 'fa07a13b462ea07328f743820e53b5aea59776e0aa566e523f7436f6de57f4c4',
    '_case_prefix': '75d37588ebcc5b5d6aa45a8fd0adc20ac55ce1c26da6ca2590ca30a3c97a285b',
    'replay': '3fd4c995ca367ec214decc899f55329883c4028eea46972ab051b5d6bb672b06',
}

base = broad.base
REPAIR_FILES = {'launch.json', 'source-inventory.json', 'receipt.json'}


def output_path(evaluator_source):
    base.files.hex_id(evaluator_source, 40)
    require(evaluator_source != ARTIFACT_SOURCE,
            're-audit evaluator source must be distinct from failed artifact source')
    return base.host.ROOT / 'artifacts/evaluations' / (OUTPUT_PREFIX + evaluator_source[:12])


def service_name(evaluator_source):
    output_path(evaluator_source)
    return f'microduck-cpu-broad-repair-{evaluator_source[:12]}.service'


def service_properties(evaluator_source):
    props = {key: base.host.read('systemctl', '--user', 'show',
        service_name(evaluator_source), '-p', key, '--value') for key in
        ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax',
         'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    expected = dict(MainPID=str(os.getpid()), ActiveState='active',
        RuntimeMaxUSec='10min', MemoryMax=str(MEMORY_BYTES),
        CPUQuotaPerSecUSec=CPU_QUOTA, Nice=NICE, KillMode=KILL_MODE)
    require(props == expected, 'exact bounded independent CPU re-audit service')
    return props


def _function_digests(source_bytes, names):
    text = source_bytes.decode('utf-8')
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            segment = ast.get_source_segment(text, node)
            require(type(segment) is str, 'extract exact pinned Python source segment')
            found[node.name] = sha256(segment.encode('utf-8')).hexdigest()
    require(set(found) == set(names), 'all required broad scoring functions present')
    return found


def _git_show(repo, revision, relative_path):
    result = subprocess.run(['git', 'show', f'{revision}:{relative_path.as_posix()}'],
        cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10)
    return result.stdout


def _source_dependency_closure(repo, revision, initial_paths):
    """Find all in-package Python imports reachable from the bound modules."""
    queue = list(initial_paths)
    seen = set()
    originals = {}
    while queue:
        relative = Path(queue.pop()).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        try:
            raw = _git_show(repo, revision, Path(relative))
        except subprocess.CalledProcessError:
            continue  # package attributes and unavailable optional modules
        originals[relative] = raw
        tree = ast.parse(raw.decode('utf-8'))
        for node in ast.walk(tree):
            module_names = []
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('mjlab_microduck.'):
                    module_names.append(node.module)
                elif node.module == 'mjlab_microduck':
                    module_names.extend('mjlab_microduck.' + alias.name
                        for alias in node.names if alias.name != '*')
                elif node.level and node.module is None:
                    # Package-local `from . import name` forms.
                    module_names.extend('mjlab_microduck.' + alias.name
                        for alias in node.names if alias.name != '*')
            elif isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names
                    if alias.name.startswith('mjlab_microduck.'))
            for module_name in module_names:
                suffix = module_name.removeprefix('mjlab_microduck.')
                queue.append('src/mjlab_microduck/' + suffix.replace('.', '/') + '.py')
    return originals


def source_inventory():
    """Pin original source bytes and verify imported replay code is unchanged."""
    repo = Path(broad.__file__).resolve().parents[2]
    old_broad = _git_show(repo, ARTIFACT_SOURCE, BROAD_SOURCE_PATH)
    require(sha256(old_broad).hexdigest() == ARTIFACT_BROAD_SHA256,
            'exact original broad-screen source bytes at failed revision')
    rev = subprocess.run(['git', 'rev-parse',
        f'{ARTIFACT_SOURCE}:{BROAD_SOURCE_PATH.as_posix()}'], cwd=repo,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=10).stdout.decode().strip()
    require(rev == ARTIFACT_BROAD_GIT_BLOB,
            'original broad-screen Git blob identity')
    names = tuple(BROAD_FUNCTION_SHA256)
    current_broad = Path(broad.__file__).read_bytes()
    old_prefix = old_broad.split(b'\ndef closeout(', 1)[0]
    current_prefix = current_broad.split(b'\ndef closeout(', 1)[0]
    require(sha256(old_prefix).hexdigest() == ARTIFACT_BROAD_PREFIX_SHA256
            and current_prefix == old_prefix,
            'complete broad-screen source prefix unchanged before closeout')
    old_functions = _function_digests(old_broad, names)
    current_functions = _function_digests(current_broad, names)
    require(old_functions == BROAD_FUNCTION_SHA256
            and current_functions == BROAD_FUNCTION_SHA256,
            'unchanged original/current broad scoring and declaration functions')
    originals = _source_dependency_closure(repo, ARTIFACT_SOURCE,
        [BROAD_SOURCE_PATH.as_posix()])
    entries = {}
    for relative, old_raw in originals.items():
        current_raw = (repo / relative).read_bytes()
        artifact_digest = sha256(old_raw).hexdigest()
        expected_digest = SOURCE_DEPENDENCIES.get(relative)
        require(expected_digest is None or artifact_digest == expected_digest,
                'static replay/scoring source pin ' + relative)
        if relative == BROAD_SOURCE_PATH.as_posix():
            require(current_raw.split(b'\ndef closeout(', 1)[0]
                    == old_raw.split(b'\ndef closeout(', 1)[0],
                    'unchanged broad source prefix in dependency closure')
            current_digest = sha256(current_raw).hexdigest()
        else:
            current_digest = sha256(current_raw).hexdigest()
            require(current_digest == artifact_digest,
                    'unchanged transitive replay/scoring dependency ' + relative)
        entries[relative] = dict(artifact_source_sha256=artifact_digest,
            evaluator_source_sha256=current_digest)
    return dict(artifact_source=ARTIFACT_SOURCE,
        broad_source_sha256=ARTIFACT_BROAD_SHA256,
        broad_git_blob=ARTIFACT_BROAD_GIT_BLOB,
        broad_prefix_sha256=ARTIFACT_BROAD_PREFIX_SHA256,
        broad_functions=old_functions, dependencies=entries)


def _check_source_context(original_context, evaluator_context,
                          artifact_source=ARTIFACT_SOURCE, evaluator_source=None):
    """Compare every launch context field, allowing only identity.source to differ."""
    require(type(original_context) is dict and type(evaluator_context) is dict
            and set(original_context) == set(evaluator_context)
            and 'source_identity' in original_context,
            'same captured and current source/profile/service context schema')
    old_identity = original_context['source_identity']
    new_identity = evaluator_context['source_identity']
    require(type(old_identity) is dict and type(new_identity) is dict
            and old_identity.get('source') == artifact_source
            and new_identity.get('source') == evaluator_source,
            'separate exact artifact and evaluator source identities')
    old_identity = dict(old_identity)
    new_identity = dict(new_identity)
    old_identity.pop('source')
    new_identity.pop('source')
    require(old_identity == new_identity
            and all(original_context[key] == evaluator_context[key]
                    for key in set(original_context) - {'source_identity'}),
            'captured profile/FilmBrain/protected-service identity unchanged')
    return True


def _check_static_matrix(launch, qualification):
    require(launch['source'] == ARTIFACT_SOURCE
            and launch['protocol'] == broad.PROTOCOL
            and launch['declarations'] == broad.declarations(ARTIFACT_SOURCE)
            and launch['campaign_window'] == window.declaration()
            and launch['cell_ids'] == list(broad.CELL_IDS)
            and len(launch['declarations']) == 25
            and launch['evaluation_seed'] == broad.EVALUATION_SEED == 671
            and launch['worlds'] == 1
            and launch['parent_checkpoint_sha256'] == broad.PARENT_CHECKPOINT_SHA256
            and launch['parent_state_sha256'] == broad.PARENT_STATE_SHA256
            and launch['parent_identity'] == broad.parent.expected_identity()
            and launch['policy_ticks'] == 250
            and launch['physics_steps'] == 2500
            and launch['prefix_step'] == 250
            and launch['collection_seconds'] == broad.COLLECTION_SECONDS
            and launch['service_seconds'] == broad.SERVICE_SECONDS
            and launch['closeout_seconds'] == broad.CLOSEOUT_SECONDS
            and launch['launch_reserve_seconds'] == broad.LAUNCH_RESERVE
            and launch['per_case_reserve_seconds'] == broad.CASE_RESERVE_SECONDS
            and launch['timing_projection'] == broad._projected_timing()
            and launch['optimizer_steps'] == launch['simulator_resets'] == 0
            and launch['auto_reset'] is False
            and all(launch[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact immutable original 25-case launch matrix and no-training flags')
    props = launch['service_properties']
    require(type(props) is dict
            and set(props) == {'MainPID', 'ActiveState', 'RuntimeMaxUSec',
                'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice', 'KillMode'}
            and type(props['MainPID']) is str and props['MainPID'].isdigit()
            and props['MainPID'] != '0' and props['ActiveState'] == 'active'
            and props['RuntimeMaxUSec'] == '24min'
            and props['MemoryMax'] == str(2 * 1024**3)
            and props['CPUQuotaPerSecUSec'] == '2s'
            and props['Nice'] == '10' and props['KillMode'] == 'control-group',
            'exact original broad capture service cap and active process receipt')
    require(qualification['protocol'] == broad.PROTOCOL
            and qualification['source'] == ARTIFACT_SOURCE
            and len(qualification['captures']) == len(qualification['scores'])
                == len(qualification['prefixes']) == 25
            and type(qualification['elapsed_seconds']) is float
            and 0 < qualification['elapsed_seconds'] < broad.SERVICE_SECONDS
            and qualification['cpu_initialized_only'] is True
            and qualification['optimizer_steps'] == qualification['simulator_resets'] == 0
            and qualification['auto_reset'] is False
            and all(qualification[key] is False for key in base.baseline.FALSE_FLAGS),
            'complete immutable original CPU qualification record')
    return True


def _check_inventory(root, report):
    require(set(report['files']) == broad.PREPARE_FILES
            and {path.name for path in root.iterdir()} == broad.COMPLETE_FILES,
            'exact 79-file original broad-screen inventory')
    inventory = {}
    for name, digest in report['files'].items():
        path = root / name
        require(type(name) is str and '/' not in name and name not in ('.', '..')
                and type(digest) is str and len(digest) == 64
                and base.host.digest(path) == digest,
                'unchanged whole original broad-screen artifact ' + str(name))
        inventory[name] = dict(sha256=digest, bytes=path.stat().st_size)
    report_path = root / 'report.json'
    report_raw = base.files.file_bytes(report_path)
    require(sha256(report_raw).hexdigest() == ARTIFACT_REPORT_SHA256,
            'whole original report bytes remain pinned in static inventory')
    inventory['report.json'] = dict(sha256=ARTIFACT_REPORT_SHA256,
        bytes=len(report_raw))
    require(set(inventory) == broad.COMPLETE_FILES,
            'complete original artifact inventory includes report bytes')
    return inventory


def _failure_link():
    require(type(FAILURE_RECEIPT_SHA256) is str and len(FAILURE_RECEIPT_SHA256) == 64
            and type(FAILURE_JOURNAL_SHA256) is str and len(FAILURE_JOURNAL_SHA256) == 64
            and type(FAILURE_RECEIPT_RELATIVE_PATH) is str
            and type(FAILURE_JOURNAL_RELATIVE_PATH) is str,
            'pinned original closeout failure receipt and supervisor journal required')
    for digest in (FAILURE_RECEIPT_SHA256, FAILURE_JOURNAL_SHA256):
        base.files.hex_id(digest, 64)
    for path in (FAILURE_RECEIPT_RELATIVE_PATH, FAILURE_JOURNAL_RELATIVE_PATH):
        require(path and not Path(path).is_absolute()
                and '..' not in Path(path).parts,
                'failure linkage path remains inside the retained evidence root')
    root = base.host.ROOT
    receipt_path = root / FAILURE_RECEIPT_RELATIVE_PATH
    journal_path = root / FAILURE_JOURNAL_RELATIVE_PATH
    receipt_raw = base.files.file_bytes(receipt_path, limit=2 * 1024**20)
    journal_raw = base.files.file_bytes(journal_path, limit=16 * 1024**20)
    require(sha256(receipt_raw).hexdigest() == FAILURE_RECEIPT_SHA256
            and sha256(journal_raw).hexdigest() == FAILURE_JOURNAL_SHA256,
            'exact immutable original closeout failure receipt and journal bytes')
    receipt = base.files.parse(receipt_raw)
    _check_failure_evidence(receipt, journal_raw)
    return dict(invocation_id=FAILED_INVOCATION,
        receipt_path=FAILURE_RECEIPT_RELATIVE_PATH,
        receipt_sha256=FAILURE_RECEIPT_SHA256,
        journal_path=FAILURE_JOURNAL_RELATIVE_PATH,
        journal_sha256=FAILURE_JOURNAL_SHA256,
        service=receipt['service'], failure_stage=receipt['failure_stage'],
        error=receipt['error'], exec_main_status=receipt['exec_main_status'])


def _check_failure_evidence(receipt, journal_raw):
    require(receipt['protocol'] == 'cpu-cardinal-closeout-failure-evidence-v1'
            and receipt['invocation_id'] == FAILED_INVOCATION
            and receipt['artifact_source'] == ARTIFACT_SOURCE
            and receipt['report_sha256'] == ARTIFACT_REPORT_SHA256
            and receipt['source_file_sha256'] == ARTIFACT_BROAD_SHA256
            and receipt['service'] == 'microduck-cpu-broad-closeout-1079a104a9e3.service'
            and receipt['failure_stage'] == 'receipt-construction-after-replay-guards'
            and receipt['error'] == "TypeError: dict() got multiple values for keyword argument 'independent_gpu_attestation'"
            and receipt['journal_sha256'] == FAILURE_JOURNAL_SHA256
            and receipt['journal_bytes'] == len(journal_raw)
            and receipt['exec_main_status'] == 1 and receipt['status'] == 'failed'
            and receipt['acceptance_changed'] is False
            and receipt['original_artifacts_modified'] is False
            and receipt['original_service_restarted'] is False
            and b"TypeError: dict() got multiple values for keyword argument 'independent_gpu_attestation'" in journal_raw,
            'original failed closeout invocation and exact constructor-only error')
    return True


def _old_failed_service_state():
    service = 'microduck-cpu-broad-closeout-1079a104a9e3.service'
    keys = ('ActiveState', 'MainPID', 'NRestarts', 'ExecMainStatus')
    state = {key: base.host.read('systemctl', '--user', 'show', service,
        '-p', key, '--value') for key in keys}
    require(state == dict(ActiveState='failed', MainPID='0', NRestarts='0',
        ExecMainStatus='1'), 'original failed closeout service remains untouched')
    return state


def receipt_result(evaluator_source, launch_sha256, inventory,
                   original_failure, source_hashes, context_binding,
                   replay, service, elapsed):
    """Pure receipt builder kept separate from filesystem/replay validation."""
    base.files.hex_id(evaluator_source, 40)
    base.files.hex_id(launch_sha256, 64)
    require(evaluator_source != ARTIFACT_SOURCE
            and set(inventory) == broad.COMPLETE_FILES
            and type(replay) is dict
            and replay['screening']['decision'] in (
                'cpu-cardinal-dose-timing-no-deficit',
                'cpu-cardinal-dose-timing-candidate-deficit',
                'cpu-cardinal-dose-timing-inconclusive')
            and replay['screening']['cases_expected'] == 25
            and replay['screening']['complete_cases'] == 25
            and replay['screening']['full_duration_gate_is_acceptance'] is False
            and replay['screening']['promotion_authorized'] is False
            and type(context_binding) is dict
            and set(context_binding) == {'original_source_identity', 'evaluator_context'}
            and type(elapsed) is float and 0 < elapsed < SERVICE_SECONDS,
            'complete bounded whole-capture re-audit decision')
    return dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, original_launch_sha256=ARTIFACT_LAUNCH_SHA256,
        original_report_sha256=ARTIFACT_REPORT_SHA256,
        failure=original_failure, original_inventory=inventory,
        source_hashes=source_hashes, context_binding=context_binding,
        screening=replay['screening'],
        scores=replay['scores'], prefixes=replay['prefixes'],
        cases_checked=25, whole_cpu_rescore_identical=True,
        cuda_initialized=False,
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        optimizer_steps=0, simulator_resets=0, auto_reset=False,
        service_properties=service, elapsed_seconds=elapsed,
        **base.baseline.FALSE_FLAGS)


def run(evaluator_source):
    started = time.monotonic()
    base.files.hex_id(evaluator_source, 40)
    require(evaluator_source != ARTIFACT_SOURCE,
            'fresh evaluator source is distinct from immutable artifact source')
    window.check(reserve_seconds=LAUNCH_RESERVE_SECONDS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'CUDA-hidden CPU capture re-audit')
    properties = service_properties(evaluator_source)
    current_context = broad._context(evaluator_source)
    old_root = base.host.ROOT / ARTIFACT_RELATIVE_PATH
    launch_raw = base.files.file_bytes(old_root / 'launch.json')
    report_raw = base.files.file_bytes(old_root / 'report.json')
    require(sha256(launch_raw).hexdigest() == ARTIFACT_LAUNCH_SHA256
            and sha256(report_raw).hexdigest() == ARTIFACT_REPORT_SHA256,
            'whole original launch and report bytes before parse')
    launch = base.files.parse(launch_raw)
    report = base.files.parse(report_raw)
    require(report['protocol'] == broad.PROTOCOL
            and report['source'] == ARTIFACT_SOURCE
            and report['launch_sha256'] == ARTIFACT_LAUNCH_SHA256
            and report['qualification_sha256'] is not None
            and report['decision'] == 'cpu-cardinal-dose-timing-no-deficit'
            and report['screening']['decision'] == report['decision']
            and report['screening']['cases_expected'] == 25
            and report['screening']['complete_cases'] == 25
            and report['screening']['complete_declared_force_phases'] is True
            and report['screening']['prefixes_identical'] is True
            and report['screening']['full_duration_gate_is_acceptance'] is False
            and report['screening']['promotion_authorized'] is False
            and report['optimizer_steps'] == 0 and report['simulator_resets'] == 0
            and type(report['elapsed_seconds']) is float
            and 0 < report['elapsed_seconds'] < broad.SERVICE_SECONDS
            and report['genuine_cpu_first_attempts'] == 25
            and report['auto_reset'] is False
            and report['cuda_initialized'] is False
            and report['source_unchanged'] is True
            and report['filmbrain_unchanged'] is True
            and report['protected_services_inactive'] is True
            and all(report[key] is False for key in base.baseline.FALSE_FLAGS),
            'original successful capture report remains non-admitting')
    original_context = {key: launch[key] for key in current_context}
    _check_source_context(original_context, current_context,
        artifact_source=ARTIFACT_SOURCE, evaluator_source=evaluator_source)
    qualification_raw = base.files.file_bytes(old_root / 'cpu-qualification.json')
    require(sha256(qualification_raw).hexdigest() == report['qualification_sha256'],
            'exact original qualification bytes named by report')
    qualification = base.files.parse(qualification_raw)
    _check_static_matrix(launch, qualification)
    require(report['files'].get('cpu-qualification.json') == sha256(qualification_raw).hexdigest(),
            'qualification file remains in original report inventory')
    inventory = _check_inventory(old_root, report)
    failure = _failure_link()
    old_service_before = _old_failed_service_state()
    failure['old_service_state'] = old_service_before
    source_hashes = source_inventory()
    prerequisite, checkpoint_raw = broad.prerequisites()
    require(launch['prerequisite'] == prerequisite
            and sha256(checkpoint_raw).hexdigest() == broad.PARENT_CHECKPOINT_SHA256
            and base.files.file_bytes(old_root / 'checkpoint.pt',
                limit=base.retained.evaluation.checkpoint.LIMIT) == checkpoint_raw,
            'original launch still binds closed prerequisites and exact parent bytes')
    require(time.monotonic() - started < SERVICE_SECONDS
            and broad._context(evaluator_source) == current_context,
            'source/profile/services unchanged before replay and cap retained')

    result_root = output_path(evaluator_source)
    native_root = base.files.native._plain_path(result_root)
    native_root.mkdir(exist_ok=False)
    launch_record = dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, artifact_launch_sha256=ARTIFACT_LAUNCH_SHA256,
        artifact_report_sha256=ARTIFACT_REPORT_SHA256, failure=failure,
        evaluator_context=current_context,
        original_source_identity=launch['source_identity'],
        source_hashes=source_hashes, service_properties=properties,
        service_seconds=SERVICE_SECONDS, launch_reserve_seconds=LAUNCH_RESERVE_SECONDS,
        optimizer_steps=0, simulator_resets=0, auto_reset=False,
        **base.baseline.FALSE_FLAGS)
    base.files.write_json(result_root / 'launch.json', launch_record)
    base.files.write_json(result_root / 'source-inventory.json', dict(
        original_capture_files=inventory, source_hashes=source_hashes,
        artifact_source=ARTIFACT_SOURCE, evaluator_source=evaluator_source))
    result = dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, decision='cpu-broad-receipt-repair-incomplete',
        failure=failure, optimizer_steps=0, **base.baseline.FALSE_FLAGS)
    try:
        replayed = broad.replay(ARTIFACT_SOURCE, launch, checkpoint_raw, qualification)
        require(replayed['screening'] == report['screening']
                and replayed['screening']['decision'] == report['decision'],
                'fresh 25-case whole CPU replay equals immutable original report')
        require(broad._context(evaluator_source) == current_context
                and service_properties(evaluator_source) == properties
                and _old_failed_service_state() == old_service_before
                and _check_inventory(old_root, report) == inventory
                and _failure_link() == {key: value for key, value in failure.items()
                    if key != 'old_service_state'}
                and time.monotonic() - started < SERVICE_SECONDS,
                'source/profile/services unchanged after bounded replay')
        window.check()
        result = receipt_result(evaluator_source, ARTIFACT_LAUNCH_SHA256,
            inventory, failure, source_hashes,
            dict(original_source_identity=launch['source_identity'],
                evaluator_context=current_context), replayed, properties,
            float(time.monotonic() - started))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['files'] = {path.name: base.host.digest(path)
            for path in sorted(native_root.iterdir()) if path.is_file()}
        base.files.write_json(result_root / 'receipt.json', result)
    return dict(output=str(result_root), decision=result['screening']['decision'],
        artifact_source=ARTIFACT_SOURCE, evaluator_source=evaluator_source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True,
        help='clean current evaluator source; never the old artifact source')
    args = parser.parse_args(argv)
    print(run(args.source))


if __name__ == '__main__':
    main()
