"""Synthetic policy/receipt/file checks; no host, simulator or replay evidence."""
from hashlib import sha256

import pytest

from mjlab_microduck import stance_recovery_broad_receipt_repair as repair
from mjlab_microduck import stance_recovery_broad_screen as broad


ARTIFACT = repair.ARTIFACT_SOURCE
EVALUATOR = 'a' * 40


def test_distinct_sources_and_exact_bounded_service_window():
    assert ARTIFACT == '1079a104a9e3bf83d56c0586cd2dd0374990e259'
    assert EVALUATOR != ARTIFACT
    assert repair.output_path(EVALUATOR).name.endswith(EVALUATOR[:12])
    assert repair.service_name(EVALUATOR) == f'microduck-cpu-broad-repair-{EVALUATOR[:12]}.service'
    assert repair.SERVICE_SECONDS == 600
    assert repair.LAUNCH_RESERVE_SECONDS == 660
    assert repair.MEMORY_BYTES == 2 * 1024**3
    assert repair.CPU_QUOTA == '2s'
    assert repair.NICE == '10'
    assert repair.KILL_MODE == 'control-group'
    repair.window.check(now=repair.window.START,
        reserve_seconds=repair.LAUNCH_RESERVE_SECONDS)
    for now in (repair.window.CUTOFF - repair.LAUNCH_RESERVE_SECONDS,
                repair.window.CUTOFF, repair.window.CUTOFF + 1):
        with pytest.raises(ValueError):
            repair.window.check(now=now, reserve_seconds=repair.LAUNCH_RESERVE_SECONDS)
    with pytest.raises(ValueError, match='distinct'):
        repair.output_path(ARTIFACT)


def test_service_guard_rejects_any_cap_change(monkeypatch):
    expected = dict(MainPID=str(repair.os.getpid()), ActiveState='active',
        RuntimeMaxUSec='10min', MemoryMax=str(repair.MEMORY_BYTES),
        CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group')
    monkeypatch.setattr(repair.base.host, 'read', lambda *args: expected[args[-2]])
    assert repair.service_properties(EVALUATOR) == expected
    expected['MemoryMax'] = str(3 * 1024**3)
    with pytest.raises(ValueError, match='exact bounded independent CPU re-audit service'):
        repair.service_properties(EVALUATOR)


def context(source, profile_tag='pinned'):
    return dict(source_identity=dict(source=source, branch='clean', revision='same'),
        cpu_math_profile={'profile': profile_tag},
        preserved_filmbrain={'generation': 4},
        protected_services={'worker': 'inactive', 'vllm': 'inactive'})


def test_context_binding_allows_only_source_identity_source_to_differ():
    assert repair._check_source_context(context(ARTIFACT), context(EVALUATOR),
        artifact_source=ARTIFACT, evaluator_source=EVALUATOR)
    altered = context(EVALUATOR, profile_tag='changed')
    with pytest.raises(ValueError, match='captured profile/FilmBrain/protected-service identity'):
        repair._check_source_context(context(ARTIFACT), altered,
            artifact_source=ARTIFACT, evaluator_source=EVALUATOR)
    wrong_old = context('b' * 40)
    with pytest.raises(ValueError, match='separate exact artifact and evaluator source identities'):
        repair._check_source_context(wrong_old, context(EVALUATOR),
            artifact_source=ARTIFACT, evaluator_source=EVALUATOR)


def original_static_records():
    launch = dict(source=ARTIFACT, protocol=broad.PROTOCOL,
        declarations=broad.declarations(ARTIFACT), cell_ids=list(broad.CELL_IDS),
        campaign_window=repair.window.declaration(),
        evaluation_seed=671, worlds=1, policy_ticks=250, physics_steps=2500,
        parent_checkpoint_sha256=broad.PARENT_CHECKPOINT_SHA256,
        parent_state_sha256=broad.PARENT_STATE_SHA256,
        parent_identity=broad.parent.expected_identity(),
        prefix_step=250, collection_seconds=broad.COLLECTION_SECONDS,
        service_seconds=broad.SERVICE_SECONDS, closeout_seconds=broad.CLOSEOUT_SECONDS,
        launch_reserve_seconds=broad.LAUNCH_RESERVE,
        per_case_reserve_seconds=broad.CASE_RESERVE_SECONDS,
        timing_projection=broad._projected_timing(), optimizer_steps=0, simulator_resets=0,
        service_properties=dict(MainPID='123', ActiveState='active', RuntimeMaxUSec='24min',
            MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group'),
        auto_reset=False, **repair.base.baseline.FALSE_FLAGS)
    qualification = dict(protocol=broad.PROTOCOL, source=ARTIFACT,
        captures=[{} for _ in broad.CELL_IDS], scores=[{} for _ in broad.CELL_IDS],
        prefixes=['f' * 64 for _ in broad.CELL_IDS], cpu_initialized_only=True,
        elapsed_seconds=10.0, optimizer_steps=0, simulator_resets=0, auto_reset=False,
        **repair.base.baseline.FALSE_FLAGS)
    return launch, qualification


def test_static_matrix_guard_requires_exact_original_25_fresh_cells():
    launch, qualification = original_static_records()
    assert repair._check_static_matrix(launch, qualification)
    launch['declarations'][8] = launch['declarations'][7]
    with pytest.raises(ValueError, match='exact immutable original 25-case launch matrix'):
        repair._check_static_matrix(launch, qualification)


def test_receipt_is_nonadmitting_and_distinguishes_artifact_from_evaluator():
    screening = dict(decision='cpu-cardinal-dose-timing-no-deficit',
        cases_expected=25, complete_cases=25,
        full_duration_gate_is_acceptance=False, promotion_authorized=False)
    replay = dict(screening=screening, scores=[{} for _ in broad.CELL_IDS],
        prefixes=['f' * 64 for _ in broad.CELL_IDS])
    failure = dict(invocation_id=repair.FAILED_INVOCATION,
        receipt_sha256='1' * 64, journal_sha256='2' * 64)
    inventory = {name: dict(sha256='0' * 64, bytes=1)
        for name in broad.COMPLETE_FILES}
    binding = dict(original_source_identity=context(ARTIFACT)['source_identity'],
        evaluator_context=context(EVALUATOR))
    result = repair.receipt_result(EVALUATOR, repair.ARTIFACT_LAUNCH_SHA256,
        inventory, failure, {'source': '3' * 64}, binding,
        replay, {'MainPID': '1'}, 12.5)
    assert result['artifact_source'] == ARTIFACT
    assert result['evaluator_source'] == EVALUATOR
    assert result['failure'] == failure
    assert result['whole_cpu_rescore_identical'] is True
    assert result['cases_checked'] == 25
    assert result['cuda_initialized'] is False
    assert result['independent_gpu_attestation'] is False
    assert result['whole_trajectory_physics_resimulated'] is False
    assert result['thermal_model_applied'] is False
    assert result['optimizer_steps'] == result['simulator_resets'] == 0
    assert all(result[key] is False for key in repair.base.baseline.FALSE_FLAGS)
    assert len(result['original_inventory']) == 79


@pytest.mark.parametrize('damage', ['partial', 'acceptance', 'wrong-source', 'elapsed', 'inventory'])
def test_receipt_builder_rejects_partial_or_misbound_audit(damage):
    screening = dict(decision='cpu-cardinal-dose-timing-no-deficit',
        cases_expected=25, complete_cases=25,
        full_duration_gate_is_acceptance=False, promotion_authorized=False)
    replay = dict(screening=screening, scores=[{} for _ in broad.CELL_IDS],
        prefixes=['f' * 64 for _ in broad.CELL_IDS])
    inventory = {name: dict(sha256='0' * 64, bytes=1)
        for name in broad.COMPLETE_FILES}
    binding = dict(original_source_identity=context(ARTIFACT)['source_identity'],
        evaluator_context=context(EVALUATOR))
    elapsed = 10.0
    evaluator = EVALUATOR
    if damage == 'partial':
        screening['complete_cases'] = 24
    elif damage == 'acceptance':
        screening['promotion_authorized'] = True
    elif damage == 'wrong-source':
        evaluator = ARTIFACT
    elif damage == 'elapsed':
        elapsed = 600.0
    else:
        inventory.pop('case-24.pt')
    with pytest.raises(ValueError, match='complete bounded whole-capture re-audit decision|distinct'):
        repair.receipt_result(evaluator, repair.ARTIFACT_LAUNCH_SHA256,
            inventory, {}, {}, binding, replay, {}, elapsed)


def test_replay_dependencies_and_original_failure_are_explicitly_pinned():
    assert repair.ARTIFACT_BROAD_SHA256 == '1c62fc70a4a0e6669f99dd8a115a6867b3dab1dca2a33f50810a2fef2340aedc'
    assert repair.ARTIFACT_BROAD_GIT_BLOB == 'f8b87e612f77f1ad1e3ab4811fa1b5d07574e043'
    assert repair.BROAD_FUNCTION_SHA256['replay']
    assert repair.SOURCE_DEPENDENCIES['src/mjlab_microduck/stance_recovery_schedule_trace.py'] == '1b0ef1e4d4a562a694e40fd0e12b9f87ba32ca5a6446a1df0d88f6766a7b6213'
    assert repair.FAILED_INVOCATION == '5cf35c7873924d9d9218c3d8764c0b56'
    assert repair.FAILURE_RECEIPT_SHA256 == '5d64c57fe65e3e38272f8f4aeafeef81017d338a499ffb4744073477f03a5cb2'
    assert repair.FAILURE_JOURNAL_SHA256 == '8d4f162edab25b285f15bc294b30b7ce8ef5d728f3533545cffefd55cb763bc2'


def test_failure_linkage_semantics_are_pure_checks_not_host_attestation():
    message = b"TypeError: dict() got multiple values for keyword argument 'independent_gpu_attestation'"
    journal = b"captured exact failed invocation: " + message
    receipt = dict(protocol='cpu-cardinal-closeout-failure-evidence-v1',
        invocation_id=repair.FAILED_INVOCATION, artifact_source=ARTIFACT,
        report_sha256=repair.ARTIFACT_REPORT_SHA256,
        source_file_sha256=repair.ARTIFACT_BROAD_SHA256,
        service='microduck-cpu-broad-closeout-1079a104a9e3.service',
        failure_stage='receipt-construction-after-replay-guards',
        error=message.decode(), journal_sha256=repair.FAILURE_JOURNAL_SHA256,
        journal_bytes=len(journal), exec_main_status=1, status='failed',
        acceptance_changed=False, original_artifacts_modified=False,
        original_service_restarted=False)
    assert repair._check_failure_evidence(receipt, journal)
    receipt['original_service_restarted'] = True
    with pytest.raises(ValueError, match='original failed closeout invocation'):
        repair._check_failure_evidence(receipt, journal)


def test_failure_link_reads_real_small_files_with_exact_mebibyte_limits(tmp_path, monkeypatch):
    # Exercise the actual bounded reader, not only the parsed schema or a spy.
    assert repair.FAILURE_RECEIPT_LIMIT == 2 * 1024**2
    assert repair.FAILURE_JOURNAL_LIMIT == 16 * 1024**2
    journal = b"TypeError: dict() got multiple values for keyword argument 'independent_gpu_attestation'"
    receipt = dict(protocol='cpu-cardinal-closeout-failure-evidence-v1',
        invocation_id=repair.FAILED_INVOCATION, artifact_source=ARTIFACT,
        report_sha256=repair.ARTIFACT_REPORT_SHA256,
        source_file_sha256=repair.ARTIFACT_BROAD_SHA256,
        service='microduck-cpu-broad-closeout-1079a104a9e3.service',
        failure_stage='receipt-construction-after-replay-guards',
        error=journal.decode(), journal_sha256=sha256(journal).hexdigest(),
        journal_bytes=len(journal), exec_main_status=1, status='failed',
        acceptance_changed=False, original_artifacts_modified=False,
        original_service_restarted=False)
    receipt_path = tmp_path / repair.FAILURE_RECEIPT_RELATIVE_PATH
    receipt_path.parent.mkdir(parents=True)
    repair.base.files.write_json(receipt_path, receipt)
    (tmp_path / repair.FAILURE_JOURNAL_RELATIVE_PATH).write_bytes(journal)
    monkeypatch.setattr(repair.base.host, 'ROOT', tmp_path)
    monkeypatch.setattr(repair, 'FAILURE_RECEIPT_SHA256', sha256(receipt_path.read_bytes()).hexdigest())
    monkeypatch.setattr(repair, 'FAILURE_JOURNAL_SHA256', sha256(journal).hexdigest())
    result = repair._failure_link()
    assert result['receipt_sha256'] == repair.FAILURE_RECEIPT_SHA256
    assert result['journal_sha256'] == repair.FAILURE_JOURNAL_SHA256


def test_preflight_link_reads_real_small_files_and_binds_parsed_identity(tmp_path, monkeypatch):
    journal = b"OverflowError: cannot fit 'int' into an index-sized integer"
    receipt = dict(protocol='cpu-cardinal-repair-preflight-failure-evidence-v1',
        evaluator_source=repair.PREFLIGHT_FAILURE_SOURCE, artifact_source=ARTIFACT,
        invocation_id=repair.PREFLIGHT_FAILURE_INVOCATION,
        service='microduck-cpu-broad-repair-1dc0b3415942.service', status='failed', exec_main_status=1,
        source_file_sha256='5d1e396f619f86b3796dc11c319571feb873f9d808f11e928f0d9d327a78e508',
        report_sha256=repair.ARTIFACT_REPORT_SHA256,
        failure_stage='failure-evidence-read-before-replay',
        error='OverflowError: cannot fit an integer read limit into a platform index',
        journal_sha256=sha256(journal).hexdigest(), journal_bytes=len(journal),
        original_artifacts_modified=False, original_service_restarted=False,
        repair_service_restarted=False, capture_collection_performed=False,
        replay_started=False, repair_output_created=False, acceptance_changed=False)
    root = tmp_path / repair.PREFLIGHT_FAILURE_DIRECTORY
    root.mkdir(parents=True)
    repair.base.files.write_json(root / 'receipt.json', receipt)
    (root / 'journal.log').write_bytes(journal)
    monkeypatch.setattr(repair.base.host, 'ROOT', tmp_path)
    monkeypatch.setattr(repair, 'PREFLIGHT_FAILURE_RECEIPT_SHA256', sha256((root / 'receipt.json').read_bytes()).hexdigest())
    monkeypatch.setattr(repair, 'PREFLIGHT_FAILURE_JOURNAL_SHA256', sha256(journal).hexdigest())
    state = dict(ActiveState='failed', MainPID='0', NRestarts='0', ExecMainStatus='1')
    monkeypatch.setattr(repair.base.host, 'read', lambda *args: state[args[-2]])
    result = repair._preflight_failure_link()
    assert result['invocation_id'] == repair.PREFLIGHT_FAILURE_INVOCATION
    assert result['service_state'] == state
    for field, value in (('evaluator_source', 'b'*40), ('invocation_id', 'c'*32),
                         ('repair_service_restarted', True), ('replay_started', True)):
        with pytest.raises(ValueError, match='receipt semantics'):
            repair._check_preflight_failure_evidence(receipt | {field: value}, journal)
