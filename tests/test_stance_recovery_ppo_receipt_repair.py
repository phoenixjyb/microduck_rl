"""Pure saved-capture receipt checks; no host, simulator, or replay claim."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_ppo_probe as probe
from mjlab_microduck import stance_recovery_ppo_receipt_repair as repair


EVALUATOR = 'a' * 40


def actual_failure_root():
    """Use native retained evidence or its exact Mac mirror, never a runtime alias."""
    repo = Path(__file__).resolve().parents[1]
    native = repo / repair.FAILURE_DIRECTORY
    mirror = repo / (
        'artifacts/retained/stochastic-ppo-failed-b1878b8715ef.7ZfWmn/'
        'stochastic-ppo-closeout-failure-b1878b8715ef')
    if (native / 'receipt.json').is_file():
        return native
    if (mirror / 'receipt.json').is_file():
        return mirror
    pytest.skip('actual failure evidence is retained only on campaign hosts')


def context(source, profile='pinned'):
    return dict(source_identity=dict(source=source, branch='clean', revision='current'),
        cpu_math_profile={'profile': profile}, preserved_filmbrain={'generation': 4},
        protected_services={'worker': 'inactive', 'vllm': 'inactive'})


def inventory():
    return {name: dict(sha256=(repair.ARTIFACT_CAPTURE_SHA256 if name == 'capture.pt'
            else repair.ARTIFACT_REPORT_SHA256 if name == 'report.json'
            else repair.ARTIFACT_PARENT_SHA256 if name == 'checkpoint.pt'
            else repair.ARTIFACT_LAUNCH_SHA256 if name == 'launch.json'
            else repair.CAPTURE_METADATA_SHA256 if name == 'capture.json' else '1' * 64),
            bytes=(repair.ARTIFACT_CAPTURE_BYTES if name == 'capture.pt' else 3120
                   if name == 'capture.json' else 17884 if name == 'launch.json' else
                   4414 if name == 'report.json' else 256368 if name == 'checkpoint.pt' else 128))
        for name in probe.COMPLETE_FILES}


def properties():
    return dict(MainPID='1234', ActiveState='active', RuntimeMaxUSec='2min',
        MemoryMax=str(repair.MEMORY_BYTES), CPUQuotaPerSecUSec='2s',
        Nice='10', KillMode='control-group')


def score():
    return dict(protocol=repair.evidence.PROTOCOL,
        collection={'policy_ticks': 28, 'elapsed_seconds': 3.5569308400154114,
            'stop_reason': 'transition-limit', 'accepted_complete': True, 'failure': None},
        storage_step=28, policy_replay_max_abs_error=0.0,
        private_rng_replayed=True, plant={}, control={}, pulse=dict(
            checked_physics_steps=280, window_steps_per_row=[0, 20],
            delivered_nonzero_steps_per_row=[0, 20], complete_pulse_delivery=True,
            complete_full_batch_phase_checks=True, unforced_post_arrays_zero=True),
        validated_policy_ticks=28, uncorroborated_storage_transitions=0,
        private_rng_matches_final=True, caller_rng_unchanged=True, within_wall_cap=True,
        complete_two_world_transition_qualification=True,
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **baseline.FALSE_FLAGS)


def source_hashes():
    return dict(artifact_source=repair.ARTIFACT_SOURCE,
        artifact_trace_sha256=repair.OLD_TRACE_SHA256,
        fixed_trace_sha256=repair.FIXED_TRACE_SHA256,
        dependencies={'src/mjlab_microduck/stance_recovery_ppo_trace.py': dict(
            artifact_source_sha256=repair.OLD_TRACE_SHA256,
            evaluator_source_sha256=repair.FIXED_TRACE_SHA256)})


def result(**overrides):
    args = dict(evaluator_source=EVALUATOR,
        launch_sha256=repair.ARTIFACT_LAUNCH_SHA256,
        failure=dict(directory=repair.FAILURE_DIRECTORY, invocation_id=repair.FAILED_INVOCATION,
            receipt_sha256=repair.FAILURE_RECEIPT_SHA256,
            journal_sha256=repair.FAILURE_JOURNAL_SHA256, original_inventory=inventory()),
        inventory=inventory(),
        source_hashes=source_hashes(), context_binding=dict(
            original_source_identity=context(repair.ARTIFACT_SOURCE)['source_identity'],
            evaluator_context=context(EVALUATOR)), replay=score(),
        service=properties(), elapsed=30.0)
    args.update(overrides)
    return repair.receipt_result(**args)


def test_source_separation_protocol_and_static_caps():
    assert repair.ARTIFACT_SOURCE == 'b1878b8715efbb7ea6e166bc7d345efecf5285de'
    assert repair.output_path(EVALUATOR).name.endswith(EVALUATOR[:12])
    assert repair.service_name(EVALUATOR) == f'microduck-cpu-ppo-receipt-repair-{EVALUATOR[:12]}.service'
    assert repair.SERVICE_SECONDS == 120 and repair.LAUNCH_RESERVE_SECONDS == 180
    assert repair.MEMORY_BYTES == 2 * 1024**3
    assert repair.FAILURE_RECEIPT_LIMIT == 2 * 1024**2
    assert repair.FAILURE_JOURNAL_LIMIT == 16 * 1024**2
    assert repair.FAILURE_DIRECTORY == (
        'artifacts/tools/stochastic-ppo-closeout-failure-b1878b8715ef')
    with pytest.raises(ValueError, match='differ'):
        repair.output_path(repair.ARTIFACT_SOURCE)
    with pytest.raises(ValueError, match='exact recorded'):
        repair._recorded_properties(properties() | {'MemoryMax': str(3 * 1024**3)})


def test_owned_service_requires_exact_properties_and_exclusivity(monkeypatch):
    expected = properties() | {'MainPID': str(repair.os.getpid())}

    def read(*args):
        if 'list-units' in args:
            return repair.service_name(EVALUATOR) + ' loaded active running unit\n'
        return expected[args[-2]]

    monkeypatch.setattr(repair.base.host, 'read', read)
    assert repair.service_properties(EVALUATOR) == expected
    monkeypatch.setattr(repair.base.host, 'read', lambda *args:
        'microduck-cpu-ppo-closeout-b1878b8715ef.service loaded active running unit\n'
        if 'list-units' in args else expected[args[-2]])
    with pytest.raises(ValueError, match='only this owned Duck service'):
        repair.service_properties(EVALUATOR)


def test_context_allows_only_source_identity_to_change():
    assert repair._check_source_context(context(repair.ARTIFACT_SOURCE),
                                        context(EVALUATOR), EVALUATOR)
    with pytest.raises(ValueError, match='captured profile/FilmBrain/protected-service'):
        repair._check_source_context(context(repair.ARTIFACT_SOURCE),
                                     context(EVALUATOR, 'changed'), EVALUATOR)
    with pytest.raises(ValueError, match='separate exact artifact and evaluator'):
        repair._check_source_context(context('b' * 40), context(EVALUATOR), EVALUATOR)


def test_transitive_source_inventory_pins_only_the_fixed_reward_leaf():
    result = repair._source_inventory()
    trace = result['dependencies']['src/mjlab_microduck/stance_recovery_ppo_trace.py']
    assert result['artifact_trace_sha256'] == repair.OLD_TRACE_SHA256
    assert result['fixed_trace_sha256'] == repair.FIXED_TRACE_SHA256
    assert trace == dict(artifact_source_sha256=repair.OLD_TRACE_SHA256,
                        evaluator_source_sha256=repair.FIXED_TRACE_SHA256)
    assert all(item['artifact_source_sha256'] == item['evaluator_source_sha256']
               for path, item in result['dependencies'].items()
               if path != 'src/mjlab_microduck/stance_recovery_ppo_trace.py')


def test_success_receipt_requires_complete_replay_and_remains_nonadmitting():
    receipt = result()
    assert receipt['protocol'] == repair.PROTOCOL
    assert receipt['artifact_source'] == repair.ARTIFACT_SOURCE
    assert receipt['evaluator_source'] == EVALUATOR
    assert receipt['original_inventory'] == inventory()
    assert receipt['independent_cpu_replay'] is True
    assert receipt['optimizer_steps'] == receipt['simulator_resets'] == 0
    assert receipt['cuda_initialized'] is False
    assert receipt['full_episode_timeout_reset_qualified'] is False
    assert receipt['finite_optimizer_step_qualified'] is False
    assert all(receipt[key] is False for key in baseline.FALSE_FLAGS)


def test_replay_constructor_accepts_actual_trace_score_shape_and_rejects_gaps():
    assert repair._check_replay(score())
    for damage in ('wrong-protocol', 'incomplete', 'ticks', 'pulse', 'rng', 'flag'):
        changed = score()
        if damage == 'wrong-protocol':
            changed['protocol'] = 'wrong'
        elif damage == 'incomplete':
            changed['collection']['accepted_complete'] = False
        elif damage == 'ticks':
            changed['uncorroborated_storage_transitions'] = 1
        elif damage == 'pulse':
            changed['pulse']['window_steps_per_row'] = [0, 19]
        elif damage == 'rng':
            changed['private_rng_matches_final'] = False
        else:
            changed['training_admitted'] = True
        with pytest.raises(ValueError):
            repair._check_replay(changed)


@pytest.mark.parametrize('damage', [
    'partial', 'optimizer', 'training', 'flags', 'source', 'inventory', 'service',
    'elapsed', 'failure', 'context',
])
def test_receipt_rejects_incomplete_or_misbound_audit(damage):
    args = dict(evaluator_source=EVALUATOR,
        launch_sha256=repair.ARTIFACT_LAUNCH_SHA256,
        failure=dict(invocation_id=repair.FAILED_INVOCATION), inventory=inventory(),
        source_hashes=source_hashes(), context_binding=dict(
            original_source_identity=context(repair.ARTIFACT_SOURCE)['source_identity'],
            evaluator_context=context(EVALUATOR)), replay=score(),
        service=properties(), elapsed=30.0)
    if damage == 'partial':
        args['replay'] = score() | {'complete_two_world_transition_qualification': False}
    elif damage == 'optimizer':
        args['replay'] = score() | {'optimizer_steps': 1}
    elif damage == 'training':
        args['replay'] = score() | {'training_update_performed': True}
    elif damage == 'flags':
        args['replay'] = score() | {'training_admitted': True}
    elif damage == 'source':
        args['evaluator_source'] = repair.ARTIFACT_SOURCE
    elif damage == 'inventory':
        args['inventory'].pop('capture.pt')
    elif damage == 'service':
        args['service'] = properties() | {'CPUQuotaPerSecUSec': '4s'}
    elif damage == 'elapsed':
        args['elapsed'] = 120.0
    elif damage == 'failure':
        args['launch_sha256'] = 'f' * 64
    else:
        args['context_binding'] = {'wrong': True}
    with pytest.raises(ValueError):
        repair.receipt_result(**args)


def test_actual_retained_failure_receipt_and_journal_match_semantics_and_pins():
    retained = actual_failure_root()
    receipt_raw = (retained / 'receipt.json').read_bytes()
    journal = (retained / 'journal.log').read_bytes()
    assert sha256(receipt_raw).hexdigest() == repair.FAILURE_RECEIPT_SHA256
    assert sha256(journal).hexdigest() == repair.FAILURE_JOURNAL_SHA256
    receipt = json.loads(receipt_raw)
    assert repair._check_failure_evidence(receipt, journal)


def test_failure_link_reads_real_bounded_files_and_checks_parsed_semantics(tmp_path, monkeypatch):
    original = actual_failure_root()
    receipt_raw = (original / 'receipt.json').read_bytes()
    journal = (original / 'journal.log').read_bytes()
    receipt = json.loads(receipt_raw)
    root = tmp_path / repair.FAILURE_DIRECTORY
    root.mkdir(parents=True)
    (root / 'receipt.json').write_bytes(receipt_raw)
    (root / 'journal.log').write_bytes(journal)
    monkeypatch.setattr(repair.base.host, 'ROOT', tmp_path)
    result = repair._failure_link()
    assert result['invocation_id'] == repair.FAILED_INVOCATION
    assert result['original_inventory'] == receipt['original_inventory']


def test_failure_reader_rejects_over_limit_and_hash_mismatch(tmp_path, monkeypatch):
    root = tmp_path / repair.FAILURE_DIRECTORY
    root.mkdir(parents=True)
    (root / 'receipt.json').write_bytes(b'{}')
    (root / 'journal.log').write_bytes(b'journal')
    monkeypatch.setattr(repair.base.host, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        repair._failure_link()
    monkeypatch.setattr(repair, 'FAILURE_RECEIPT_SHA256', sha256(b'{}').hexdigest())
    monkeypatch.setattr(repair, 'FAILURE_JOURNAL_SHA256', sha256(b'journal').hexdigest())
    monkeypatch.setattr(repair, 'FAILURE_RECEIPT_LIMIT', 1)
    with pytest.raises(ValueError):
        repair._failure_link()
