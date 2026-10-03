"""Independent CPU replay of the immutable stochastic PPO capture after a scorer fix.

This sibling audit only re-scores an already retained two-world capture. It
does not recollect, mutate the old evidence, update a policy, or qualify GPU or
physical recovery.
"""
from hashlib import sha256
import argparse
import math
import os
from pathlib import Path
import time

import torch

from mjlab_microduck import stance_recovery_broad_receipt_repair as prior_repair
from mjlab_microduck import stance_recovery_broad_screen as prerequisites
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_ppo_probe as probe
from mjlab_microduck import stance_recovery_ppo_trace as evidence
from mjlab_microduck.first_attempt_smoke import require

base = prerequisites.base
PROTOCOL = 'football-b1d-cpu-stochastic-ppo-receipt-repair-v1'
ARTIFACT_SOURCE = 'b1878b8715efbb7ea6e166bc7d345efecf5285de'
ARTIFACT_LAUNCH_SHA256 = '0dbe88f4b580507b26eac1bf17ca5d20647cb016103c6a8a43c170f54f79a528'
ARTIFACT_REPORT_SHA256 = '20bfa6222ab737c5f19a299d5a15f864078e6e3c49eb8ef1c4d55702aefedebf'
ARTIFACT_CAPTURE_SHA256 = '180675277204d08b54fdd4daa605d8c7fc2f3a074b2f65c0c15c3ce3c76bfac3'
ARTIFACT_CAPTURE_BYTES = 6_889_991
ARTIFACT_PARENT_SHA256 = '2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5'
OLD_TRACE_SHA256 = '1d7d9b5e066b14092c1c73c0f65a82d5a67c7931248cdc47ff69b7bf73483384'
FIXED_TRACE_SHA256 = '5c5aee58d4e384cf28b61a5377c50065c27619b4003631a8f5dc663513f8580a'
FAILED_INVOCATION = 'ccd6921149844ace9fa9e12cca4bd816'
FAILURE_DIRECTORY = 'artifacts/tools/stochastic-ppo-closeout-failure-b1878b8715ef'
FAILURE_RECEIPT_SHA256 = '7c96ad502b9f38778965fcade4d9c5b6d15e98dfcbc893a34e7421466a1ce9b6'
FAILURE_JOURNAL_SHA256 = '5e155e8284847062e7af0f0a78d6685f3f24767d1f5e79f21528d79b0168793f'
FAILURE_RECEIPT_LIMIT = 2 * 1024**2
FAILURE_JOURNAL_LIMIT = 16 * 1024**2
CAPTURE_METADATA_SHA256 = '481768361b245c2f92483c878eb081d0268f2d05e093a094867d019585f7d09d'
SERVICE_SECONDS, LAUNCH_RESERVE_SECONDS = 120, 180
MEMORY_BYTES, CPU_QUOTA, NICE, KILL_MODE = 2 * 1024**3, '2s', '10', 'control-group'
OUTPUT_PREFIX = 'stance-wsl-cpu-ppo-receipt-repair-'
COMPLETE_FILES = {'launch.json', 'source-inventory.json', 'receipt.json'}
ARTIFACT_ROOT = Path('artifacts/evaluations') / (
    'stance-wsl-cpu-stochastic-ppo-transition-' + ARTIFACT_SOURCE[:12])


def output_path(evaluator_source):
    base.files.hex_id(evaluator_source, 40)
    require(evaluator_source != ARTIFACT_SOURCE,
            'receipt-repair evaluator must differ from immutable artifact source')
    return base.host.ROOT / 'artifacts/evaluations' / (OUTPUT_PREFIX + evaluator_source[:12])


def service_name(source):
    output_path(source)
    return f'microduck-cpu-ppo-receipt-repair-{source[:12]}.service'


def _recorded_properties(properties):
    require(type(properties) is dict and set(properties) == {
        'MainPID', 'ActiveState', 'RuntimeMaxUSec', 'MemoryMax',
        'CPUQuotaPerSecUSec', 'Nice', 'KillMode'}
        and type(properties['MainPID']) is str and properties['MainPID'].isdecimal()
        and int(properties['MainPID']) > 0 and properties['ActiveState'] == 'active'
        and properties['RuntimeMaxUSec'] == '2min'
        and properties['MemoryMax'] == str(MEMORY_BYTES)
        and properties['CPUQuotaPerSecUSec'] == CPU_QUOTA
        and properties['Nice'] == NICE and properties['KillMode'] == KILL_MODE,
        'exact recorded 120-second/2-GiB CPU receipt-repair cap')
    return True


def service_properties(source):
    name = service_name(source)
    props = {key: base.host.read('systemctl', '--user', 'show', name,
        '-p', key, '--value') for key in ('MainPID', 'ActiveState', 'RuntimeMaxUSec',
        'MemoryMax', 'CPUQuotaPerSecUSec', 'Nice', 'KillMode')}
    expected = dict(MainPID=str(os.getpid()), ActiveState='active',
        RuntimeMaxUSec='2min', MemoryMax=str(MEMORY_BYTES),
        CPUQuotaPerSecUSec=CPU_QUOTA, Nice=NICE, KillMode=KILL_MODE)
    require(props == expected, 'exact owned bounded CPU receipt-repair service')
    running = base.host.read('systemctl', '--user', 'list-units', '--state=running',
                             '--no-legend', 'microduck*')
    require({line.split()[0] for line in running.splitlines()} == {name},
            'only this owned Duck service runs during receipt repair')
    return props


def _check_source_context(original, current, evaluator_source):
    return prior_repair._check_source_context(original, current,
        artifact_source=ARTIFACT_SOURCE, evaluator_source=evaluator_source)


def _check_failure_evidence(receipt, journal_raw):
    """Check parsed failure meaning in addition to pinned receipt/journal hashes."""
    require(type(receipt) is dict
            and receipt.get('protocol') == 'cpu-stochastic-ppo-closeout-failure-evidence-v1'
            and receipt.get('artifact_source') == ARTIFACT_SOURCE
            and receipt.get('invocation_id') == FAILED_INVOCATION
            and receipt.get('service') == f'microduck-cpu-ppo-closeout-{ARTIFACT_SOURCE[:12]}.service'
            and receipt.get('status') == 'failed' and receipt.get('exec_main_status') == 1
            and receipt.get('source_file_sha256') == OLD_TRACE_SHA256
            and receipt.get('launch_sha256') == ARTIFACT_LAUNCH_SHA256
            and receipt.get('report_sha256') == ARTIFACT_REPORT_SHA256
            and receipt.get('capture_sha256') == ARTIFACT_CAPTURE_SHA256
            and receipt.get('error') == 'ValueError: action, bridge receipt, and terminal observation binding'
            and receipt.get('failure_stage') == 'bridge-reward-layout-before-physical-and-stochastic-replay'
            and receipt.get('journal_sha256') == FAILURE_JOURNAL_SHA256
            and receipt.get('journal_bytes') == len(journal_raw)
            and receipt.get('exec_main_status') == 1 and receipt.get('status') == 'failed'
            and type(receipt.get('diagnosis')) is dict,
            'pinned failed closeout invocation and reward-layout diagnosis')
    diagnosis = receipt['diagnosis']
    failed_ticks = diagnosis.get('failed_ticks')
    require(type(failed_ticks) is list and len(failed_ticks) == evidence.HORIZON == 28
            and all(type(tick) is dict and tick == dict(index=index, failed=['reward'],
                bridge_reward_shape=[2], stored_reward_shape=[2, 1],
                bridge_reward_dtype='torch.float32', stored_reward_dtype='torch.float32',
                value_equal_after_squeeze=True)
                for index, tick in enumerate(failed_ticks))
            and diagnosis.get('capture_sha256') == ARTIFACT_CAPTURE_SHA256
            and diagnosis.get('capture_collection') == dict(accepted_complete=True,
                elapsed_seconds=3.5569308400154114, failure=None,
                policy_ticks=28, stop_reason='transition-limit')
            and diagnosis.get('ticks') == 28 and diagnosis.get('cuda_initialized') is False
            and diagnosis.get('source_unchanged') is True
            and 'ValueError: action, bridge receipt, and terminal observation binding'
                in journal_raw.decode('utf-8', errors='replace'),
            'actual 28-tick diagnosis identifies reward-only layout failure')
    inventory = receipt.get('original_inventory')
    require(type(inventory) is dict and set(inventory) == probe.COMPLETE_FILES
            and all(type(item) is dict and set(item) == {'sha256', 'bytes'}
                    and type(item['sha256']) is str and len(item['sha256']) == 64
                    and type(item['bytes']) is int and item['bytes'] > 0
                    for item in inventory.values())
            and inventory['launch.json']['sha256'] == ARTIFACT_LAUNCH_SHA256
            and inventory['report.json']['sha256'] == ARTIFACT_REPORT_SHA256
            and inventory['capture.json']['sha256'] == CAPTURE_METADATA_SHA256
            and inventory['capture.json']['bytes'] == 3120
            and inventory['capture.pt']['sha256'] == ARTIFACT_CAPTURE_SHA256
            and inventory['capture.pt']['bytes'] == ARTIFACT_CAPTURE_BYTES
            and inventory['checkpoint.pt']['sha256'] == ARTIFACT_PARENT_SHA256
            and inventory['checkpoint.pt']['bytes'] == 256368
            and inventory['launch.json']['bytes'] == 17884
            and inventory['report.json']['bytes'] == 4414,
            'exact five-file immutable original capture inventory')
    require(receipt.get('optimizer_steps') == 0
            and receipt.get('original_artifacts_modified') is False
            and receipt.get('original_service_restarted') is False
            and receipt.get('acceptance_changed') is False
            and all(receipt.get(key) is False for key in base.baseline.FALSE_FLAGS),
            'failure receipt proves no mutation, restart, recollection or admission')
    return True


def _failure_link():
    """Read the real pinned evidence files with explicit byte bounds."""
    root = base.host.ROOT / FAILURE_DIRECTORY
    receipt_path, journal_path = root / 'receipt.json', root / 'journal.log'
    receipt_raw = base.files.file_bytes(receipt_path, limit=FAILURE_RECEIPT_LIMIT)
    journal_raw = base.files.file_bytes(journal_path, limit=FAILURE_JOURNAL_LIMIT)
    require(sha256(receipt_raw).hexdigest() == FAILURE_RECEIPT_SHA256
            and sha256(journal_raw).hexdigest() == FAILURE_JOURNAL_SHA256,
            'whole immutable failed-closeout evidence hashes')
    receipt = base.files.parse(receipt_raw)
    _check_failure_evidence(receipt, journal_raw)
    return dict(directory=FAILURE_DIRECTORY, invocation_id=FAILED_INVOCATION,
        receipt_sha256=FAILURE_RECEIPT_SHA256, journal_sha256=FAILURE_JOURNAL_SHA256,
        service=receipt['service'], original_inventory=receipt['original_inventory'])


def _old_service_state():
    service = f'microduck-cpu-ppo-closeout-{ARTIFACT_SOURCE[:12]}.service'
    state = {key: base.host.read('systemctl', '--user', 'show', service,
        '-p', key, '--value') for key in ('ActiveState', 'MainPID', 'NRestarts', 'ExecMainStatus')}
    require(state == dict(ActiveState='failed', MainPID='0', NRestarts='0', ExecMainStatus='1'),
            'original failed closeout service remains untouched')
    return state


def _source_inventory(repo=None):
    repo = Path(probe.__file__).resolve().parents[2] if repo is None else Path(repo)
    originals = prior_repair._source_dependency_closure(repo, ARTIFACT_SOURCE,
        ['src/mjlab_microduck/stance_recovery_ppo_probe.py',
         'src/mjlab_microduck/stance_recovery_ppo_trace.py'])
    trace_path = 'src/mjlab_microduck/stance_recovery_ppo_trace.py'
    require(trace_path in originals
            and sha256(originals[trace_path]).hexdigest() == OLD_TRACE_SHA256,
            'exact old trace source at immutable artifact revision')
    entries = {}
    for relative, original in originals.items():
        current = (repo / relative).read_bytes()
        old_hash = sha256(original).hexdigest()
        current_hash = sha256(current).hexdigest()
        expected = FIXED_TRACE_SHA256 if relative == trace_path else old_hash
        require(current_hash == expected,
                'only the pinned strict-reward trace leaf may differ: ' + relative)
        entries[relative] = dict(artifact_source_sha256=old_hash,
                                 evaluator_source_sha256=current_hash)
    return dict(artifact_source=ARTIFACT_SOURCE, artifact_trace_sha256=OLD_TRACE_SHA256,
                fixed_trace_sha256=FIXED_TRACE_SHA256, dependencies=entries)


def _check_replay(score):
    expected_keys = {'protocol', 'collection', 'storage_step', 'policy_replay_max_abs_error',
        'private_rng_replayed', 'plant', 'control', 'pulse', 'validated_policy_ticks',
        'uncorroborated_storage_transitions', 'private_rng_matches_final',
        'caller_rng_unchanged', 'within_wall_cap',
        'complete_two_world_transition_qualification', 'whole_trajectory_physics_resimulated',
        'thermal_model_applied'} | set(base.baseline.FALSE_FLAGS)
    collection = score.get('collection') if type(score) is dict else None
    require(type(score) is dict and set(score) == expected_keys
            and score['protocol'] == evidence.PROTOCOL
            and type(collection) is dict
            and collection.get('policy_ticks') == evidence.HORIZON
            and collection.get('stop_reason') == 'transition-limit'
            and collection.get('accepted_complete') is True
            and collection.get('failure') is None
            and type(collection.get('elapsed_seconds')) is float
            and math.isfinite(collection['elapsed_seconds'])
            and 0 < collection['elapsed_seconds'] < evidence.WALL_LIMIT
            and score['storage_step'] == score['validated_policy_ticks'] == evidence.HORIZON
            and score['uncorroborated_storage_transitions'] == 0
            and score['policy_replay_max_abs_error'] == 0.0
            and score['private_rng_replayed'] is True
            and score['private_rng_matches_final'] is True
            and score['caller_rng_unchanged'] is True
            and score['within_wall_cap'] is True
            and score['complete_two_world_transition_qualification'] is True
            and type(score['pulse']) is dict
            and score['pulse'].get('checked_physics_steps') == 280
            and score['pulse'].get('window_steps_per_row') == [0, 20]
            and score['pulse'].get('delivered_nonzero_steps_per_row') == [0, 20]
            and score['pulse'].get('complete_pulse_delivery') is True
            and score['pulse'].get('complete_full_batch_phase_checks') is True
            and score['pulse'].get('unforced_post_arrays_zero') is True
            and score['whole_trajectory_physics_resimulated'] is False
            and score['thermal_model_applied'] is False
            and all(score[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact actual independent scorer schema and complete CPU2 result')
    return True


def _rehash_original(root, expected):
    require({path.name for path in root.iterdir()} == probe.COMPLETE_FILES,
            'original five-file artifact set remains unchanged')
    for name, entry in expected.items():
        limit = (evidence.LIMIT if name == 'capture.pt' else
            base.retained.evaluation.checkpoint.LIMIT if name == 'checkpoint.pt' else
            FAILURE_RECEIPT_LIMIT)
        raw = base.files.file_bytes(root / name, limit=limit)
        require(len(raw) == entry['bytes'] and sha256(raw).hexdigest() == entry['sha256'],
                'original capture remains unchanged after replay: ' + name)
    return True


def _check_original(launch_raw, report_raw, root, evaluator_source, current_context,
                    prerequisites_record, checkpoint_raw, failure, inventory):
    require(sha256(launch_raw).hexdigest() == ARTIFACT_LAUNCH_SHA256
            and sha256(report_raw).hexdigest() == ARTIFACT_REPORT_SHA256,
            'whole original launch/report hashes before parse')
    launch, report = base.files.parse(launch_raw), base.files.parse(report_raw)
    _check_source_context({key: launch[key] for key in current_context}, current_context,
                          evaluator_source)
    require(launch['protocol'] == probe.PROTOCOL and launch['source'] == ARTIFACT_SOURCE
            and launch['prerequisites'] == prerequisites_record
            and launch['declaration'] == probe.declaration(ARTIFACT_SOURCE)
            and launch['trace_binding'] == evidence.binding(launch['declaration'],
                launch['compiled_plant'], launch['cpu_math_profile'])
            and launch['parent_checkpoint_sha256'] == ARTIFACT_PARENT_SHA256
            and sha256(checkpoint_raw).hexdigest() == ARTIFACT_PARENT_SHA256
            and launch['learner_seed'] == evidence.SEED == 653
            and launch['worlds'] == 2 and launch['horizon'] == evidence.HORIZON == 28
            and launch['collection_seconds'] == evidence.WALL_LIMIT == 120.0
            and launch['capture_bytes_limit'] == evidence.LIMIT
            and launch['service_seconds'] == probe.SERVICE_SECONDS == 180
            and launch['closeout_seconds'] == probe.CLOSEOUT_SECONDS == 120
            and launch['launch_reserve_seconds'] == probe.LAUNCH_RESERVE == 360
            and launch['optimizer_steps'] == 0 and launch['training_update_performed'] is False
            and launch['service_properties']['RuntimeMaxUSec'] == '3min'
            and launch['service_properties']['MemoryMax'] == str(MEMORY_BYTES)
            and launch['service_properties']['CPUQuotaPerSecUSec'] == CPU_QUOTA
            and launch['service_properties']['Nice'] == NICE
            and launch['service_properties']['KillMode'] == KILL_MODE
            and all(launch[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact old seed-653 CPU2 no-update launch and immutable parent')
    capture_raw = base.files.file_bytes(root / 'capture.json', limit=FAILURE_RECEIPT_LIMIT)
    capture = base.files.parse(capture_raw)
    run_properties = launch['service_properties']
    probe._recorded_run_properties(run_properties)
    require(capture['protocol'] == probe.PROTOCOL and capture['source'] == ARTIFACT_SOURCE
            and capture['binding'] == launch['trace_binding']
            and capture['capture_bytes'] == ARTIFACT_CAPTURE_BYTES
            and capture['capture_sha256'] == ARTIFACT_CAPTURE_SHA256
            and capture['optimizer_steps'] == 0 and capture['training_update_performed'] is False
            and all(capture[key] is False for key in base.baseline.FALSE_FLAGS),
            'exact original retained CPU2 capture metadata and no-update contract')
    admissible = probe._check_report(report, capture, ARTIFACT_SOURCE,
                                     ARTIFACT_LAUNCH_SHA256, run_properties)
    require(admissible is True and report['postchecks_passed'] is True
            and report['elapsed_seconds'] < probe.SERVICE_SECONDS,
            'original run completed all postchecks inside its declared cap')
    original_inventory = {}
    require({path.name for path in root.iterdir()} == probe.COMPLETE_FILES,
            'exact original five-file capture inventory')
    for name, digest in report['files'].items():
        limit = (evidence.LIMIT if name == 'capture.pt' else
            base.retained.evaluation.checkpoint.LIMIT if name == 'checkpoint.pt' else
            FAILURE_RECEIPT_LIMIT)
        raw = base.files.file_bytes(root / name, limit=limit)
        actual = sha256(raw).hexdigest()
        require(actual == digest, 'original capture file digest ' + name)
        original_inventory[name] = dict(sha256=actual, bytes=len(raw))
    original_inventory['report.json'] = dict(sha256=ARTIFACT_REPORT_SHA256,
                                               bytes=len(report_raw))
    require(set(original_inventory) == probe.COMPLETE_FILES
            and original_inventory == inventory
            and original_inventory == failure['original_inventory'],
            'all five capture files plus report match failure evidence inventory')
    require(original_inventory['capture.pt']['sha256'] == ARTIFACT_CAPTURE_SHA256
            and original_inventory['capture.pt']['bytes'] == ARTIFACT_CAPTURE_BYTES
            and original_inventory['checkpoint.pt']['sha256'] == ARTIFACT_PARENT_SHA256,
            'exact original capture and parent bytes')
    return launch, report, capture, checkpoint_raw, original_inventory


def receipt_result(evaluator_source, launch_sha256, failure, inventory, source_hashes,
                   context_binding, replay, service, elapsed):
    """Pure result constructor: success requires full independently replayed trace."""
    base.files.hex_id(evaluator_source, 40)
    base.files.hex_id(launch_sha256, 64)
    _check_replay(replay)
    require(evaluator_source != ARTIFACT_SOURCE
            and launch_sha256 == ARTIFACT_LAUNCH_SHA256
            and set(inventory) == probe.COMPLETE_FILES
            and type(context_binding) is dict
            and set(context_binding) == {'original_source_identity', 'evaluator_context'}
            and type(replay) is dict
            and replay.get('complete_two_world_transition_qualification') is True
            and all(replay.get(key) is False for key in base.baseline.FALSE_FLAGS)
            and type(elapsed) is float and math.isfinite(elapsed)
            and 0 < elapsed < SERVICE_SECONDS,
            'complete bounded independent replay of immutable CPU2 capture')
    _recorded_properties(service)
    require(all(type(item) is dict and set(item) == {'sha256', 'bytes'}
                for item in inventory.values())
            and inventory['launch.json'] == dict(sha256=ARTIFACT_LAUNCH_SHA256, bytes=17884)
            and inventory['report.json'] == dict(sha256=ARTIFACT_REPORT_SHA256, bytes=4414)
            and inventory['capture.json'] == dict(sha256=CAPTURE_METADATA_SHA256, bytes=3120)
            and inventory['capture.pt'] == dict(sha256=ARTIFACT_CAPTURE_SHA256,
                                                 bytes=ARTIFACT_CAPTURE_BYTES)
            and inventory['checkpoint.pt'] == dict(sha256=ARTIFACT_PARENT_SHA256, bytes=256368)
            and type(failure) is dict
            and failure.get('directory') == FAILURE_DIRECTORY
            and failure.get('invocation_id') == FAILED_INVOCATION
            and failure.get('receipt_sha256') == FAILURE_RECEIPT_SHA256
            and failure.get('journal_sha256') == FAILURE_JOURNAL_SHA256
            and failure.get('original_inventory') == inventory
            and type(context_binding['original_source_identity']) is dict
            and type(context_binding['evaluator_context']) is dict
            and context_binding['original_source_identity'].get('source') == ARTIFACT_SOURCE
            and context_binding['evaluator_context'].get('source_identity', {}).get('source') == evaluator_source
            and source_hashes['artifact_source'] == ARTIFACT_SOURCE
            and source_hashes['artifact_trace_sha256'] == OLD_TRACE_SHA256
            and source_hashes['fixed_trace_sha256'] == FIXED_TRACE_SHA256,
            'exact old/evaluator source and retained five-file inventory')
    original_id = dict(context_binding['original_source_identity'])
    evaluator_id = dict(context_binding['evaluator_context']['source_identity'])
    original_id.pop('source')
    evaluator_id.pop('source')
    require(original_id == evaluator_id,
            'artifact and evaluator source identities match apart from revision')
    deps = source_hashes.get('dependencies')
    require(type(deps) is dict and deps.get(
        'src/mjlab_microduck/stance_recovery_ppo_trace.py') == dict(
            artifact_source_sha256=OLD_TRACE_SHA256,
            evaluator_source_sha256=FIXED_TRACE_SHA256)
            and all(item['artifact_source_sha256'] == item['evaluator_source_sha256']
                    for path, item in deps.items()
                    if path != 'src/mjlab_microduck/stance_recovery_ppo_trace.py'),
            'only the strict reward trace leaf differs in transitive source closure')
    return dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, original_launch_sha256=ARTIFACT_LAUNCH_SHA256,
        original_report_sha256=ARTIFACT_REPORT_SHA256,
        original_capture_sha256=ARTIFACT_CAPTURE_SHA256,
        failed_invocation=FAILED_INVOCATION, failure=failure,
        original_inventory=inventory, source_hashes=source_hashes,
        context_binding=context_binding, replay=replay,
        independent_cpu_replay=True, run_admissible=True,
        optimizer_steps=0, training_update_performed=False, simulator_resets=0,
        cuda_initialized=False, full_episode_timeout_reset_qualified=False,
        finite_optimizer_step_qualified=False, whole_trajectory_physics_resimulated=False,
        thermal_model_applied=False, service_properties=service,
        elapsed_seconds=elapsed, **base.baseline.FALSE_FLAGS)


def audit(evaluator_source):
    """Run the bounded replay audit; never starts collection or mutates old files."""
    started = time.monotonic()
    base.files.hex_id(evaluator_source, 40)
    require(evaluator_source != ARTIFACT_SOURCE, 'distinct clean evaluator source')
    window.check(reserve_seconds=LAUNCH_RESERVE_SECONDS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CUDA-hidden CPU-only receipt repair')
    service = service_properties(evaluator_source)
    context = prerequisites._context(evaluator_source)
    previous, checkpoint_raw = prerequisites.prerequisites()
    old_root = base.host.ROOT / ARTIFACT_ROOT
    launch_raw = base.files.file_bytes(old_root / 'launch.json', limit=FAILURE_RECEIPT_LIMIT)
    report_raw = base.files.file_bytes(old_root / 'report.json', limit=FAILURE_RECEIPT_LIMIT)
    failure = _failure_link()
    old_service = _old_service_state()
    source_hashes = _source_inventory()
    inventory = failure['original_inventory']
    launch, report, capture, checkpoint_raw, inventory = _check_original(
        launch_raw, report_raw, old_root, evaluator_source, context, previous,
        checkpoint_raw, failure, inventory)
    require(time.monotonic() - started < SERVICE_SECONDS
            and prerequisites._context(evaluator_source) == context,
            'context/source unchanged before independent replay')

    out = output_path(evaluator_source)
    native_out = base.files.native._plain_path(out)
    native_out.mkdir(exist_ok=False)
    launch_record = dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, original_launch_sha256=ARTIFACT_LAUNCH_SHA256,
        original_report_sha256=ARTIFACT_REPORT_SHA256, failure=failure,
        original_source_identity=launch['source_identity'], evaluator_context=context,
        source_hashes=source_hashes, service_properties=service,
        service_seconds=SERVICE_SECONDS, launch_reserve_seconds=LAUNCH_RESERVE_SECONDS,
        optimizer_steps=0, simulator_resets=0, **base.baseline.FALSE_FLAGS)
    base.files.write_json(out / 'launch.json', launch_record)
    base.files.write_json(out / 'source-inventory.json', dict(
        original_capture_files=inventory, source_hashes=source_hashes,
        artifact_source=ARTIFACT_SOURCE, evaluator_source=evaluator_source))
    result = dict(protocol=PROTOCOL, artifact_source=ARTIFACT_SOURCE,
        evaluator_source=evaluator_source, decision='cpu-ppo-receipt-repair-incomplete',
        optimizer_steps=0, **base.baseline.FALSE_FLAGS)
    try:
        score = evidence.verify(base.files.file_bytes(old_root / 'capture.pt', limit=evidence.LIMIT),
            ARTIFACT_CAPTURE_SHA256, checkpoint_raw, launch['declaration'],
            launch['compiled_plant'])
        _check_replay(score)
        require(score['collection'] == capture['collection']
                and score['storage_step'] == capture['storage_step'],
                'independent complete seed-653 28x2 CPU trace replay')
        require(prerequisites._context(evaluator_source) == context
                and service_properties(evaluator_source) == service
                and _old_service_state() == old_service
                and _rehash_original(old_root, inventory)
                and _failure_link() == failure
                and _source_inventory() == source_hashes
                and time.monotonic() - started < SERVICE_SECONDS,
                'all original evidence, failure, service and evaluator context unchanged')
        window.check()
        result = receipt_result(evaluator_source, ARTIFACT_LAUNCH_SHA256, failure,
            inventory, source_hashes,
            dict(original_source_identity=launch['source_identity'], evaluator_context=context),
            score, service, float(time.monotonic() - started))
        result['decision'] = 'cpu-ppo-receipt-repair-replayed-complete-cpu2-trace'
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:500])
        raise
    finally:
        result['files'] = {path.name: base.host.digest(path)
                           for path in sorted(native_out.iterdir()) if path.is_file()}
        base.files.write_json(out / 'receipt.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True,
        help='clean distinct evaluator source; never the immutable capture source')
    args = parser.parse_args(argv)
    result = audit(args.source)
    print(dict(output=str(output_path(args.source)), decision=result['decision'],
        artifact_source=ARTIFACT_SOURCE, evaluator_source=args.source,
        receipt_sha256=base.host.digest(output_path(args.source) / 'receipt.json')))


if __name__ == '__main__':
    main()
