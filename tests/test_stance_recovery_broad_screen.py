"""Pure protocol/decision tests; no native capture or capability attestation."""
import pytest

from mjlab_microduck import stance_recovery_broad_screen as screen
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_trace as evidence


SOURCE = 'a' * 40


def synthetic_score(cell, *, candidate_pass=True, full=True, force_complete=True):
    """Decision-only fixture, not evidence for a real first attempt."""
    ticks = 250 if full else 30
    steps = 2500 if full else 300
    return dict(cell=cell, protocol=evidence.PROTOCOL,
        strict_actor_restore=True, actor_replay_max_abs_error=0.,
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **screen.base.baseline.FALSE_FLAGS,
        numerical_diagnostic=dict(candidate_pass=candidate_pass,
            complete_first_attempt=True),
        collection=dict(policy_ticks=ticks,
            elapsed_seconds=1.0,
            stop_reason='policy-tick-limit' if full else 'all-first-attempts-complete'),
        pulse=dict(checked_physics_steps=steps,
            complete_pulse_delivery=force_complete,
            complete_phase_checks=force_complete,
            recorded_phase_checks_valid=True,
            exact_full_force_arrays_checked=True,
            unforced_post_arrays_checked=True))


def synthetic_rows():
    return [synthetic_score(cell) for cell in screen.CELL_IDS]


def test_fixed_order_is_one_zero_case_then_24_literal_heldout_cardinal_cells():
    expected = ['zero-wrench'] + [
        f'{direction}-{newtons}n-{steps}steps-t{onset}'
        for onset in (250, 500, 750)
        for direction in ('+x', '-x', '+y', '-y')
        for newtons, steps in ((2, 20), (4, 10))]
    assert list(screen.CELL_IDS) == expected
    rows = screen.declarations(SOURCE)
    assert len(rows) == 25
    assert [row['cell_ids'][0] for row in rows] == expected
    assert all(row['stage'] == 'dose' and row['split'] == 'held-out'
               and row['worlds'] == 1 for row in rows)
    assert all(schedule.checked(row) == row for row in rows)


def test_new_window_service_caps_and_projection_are_fixed():
    assert screen.SERVICE_SECONDS == 1440
    assert screen.CLOSEOUT_SECONDS == 600
    assert screen.LAUNCH_RESERVE == 2100
    assert screen.COLLECTION_SECONDS == 60
    assert screen.CASE_RESERVE_SECONDS == 80
    assert screen.PROJECTED_SERVICE_SECONDS == pytest.approx(1364.6354166666667)
    assert screen.PROJECTED_SERVICE_SECONDS < screen.SERVICE_SECONDS
    assert screen.service_name(SOURCE, 'run') == 'microduck-cpu-broad-run-aaaaaaaaaaaa.service'
    assert screen.service_name(SOURCE, 'closeout') == 'microduck-cpu-broad-closeout-aaaaaaaaaaaa.service'
    with pytest.raises(ValueError):
        screen.service_name(SOURCE, 'supervise')
    window.check(now=window.START, reserve_seconds=screen.LAUNCH_RESERVE)
    for now in (window.START - 1, window.CUTOFF - screen.LAUNCH_RESERVE,
                window.CUTOFF, window.CUTOFF + 1):
        with pytest.raises(ValueError):
            window.check(now=now, reserve_seconds=screen.LAUNCH_RESERVE)


def test_service_properties_reject_any_widened_cap(monkeypatch):
    expected = dict(MainPID=str(screen.os.getpid()), ActiveState='active',
        RuntimeMaxUSec='24min', MemoryMax=str(2 * 1024**3),
        CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group')
    monkeypatch.setattr(screen.base.host, 'read', lambda *args: expected[args[-2]])
    assert screen.service_properties(SOURCE, 'run') == expected
    expected['RuntimeMaxUSec'] = '25min'
    with pytest.raises(ValueError, match='exact independently capped broad CPU service'):
        screen.service_properties(SOURCE, 'run')


def test_zero_and_matched_complete_matrix_produce_no_deficit():
    result = screen._screen_inputs(synthetic_rows(), ['f' * 64] * 25)
    assert result['decision'] == 'cpu-cardinal-dose-timing-no-deficit'
    assert result['zero_control_valid'] is True
    assert result['complete_declared_force_phases'] is True
    assert result['prefixes_identical'] is True
    assert result['complete_cases'] == 25
    assert result['candidate_deficit_cells'] == []
    assert result['promotion_authorized'] is False
    assert all(result[key] is False for key in screen.base.baseline.FALSE_FLAGS)


def test_valid_unexpected_terminal_can_be_a_candidate_deficit():
    rows = synthetic_rows()
    rows[1] = synthetic_score(screen.CELL_IDS[1], candidate_pass=False, full=False)
    result = screen._screen_inputs(rows, ['1' * 64] * 25)
    assert result['decision'] == 'cpu-cardinal-dose-timing-candidate-deficit'
    assert result['candidate_deficit_cells'] == [screen.CELL_IDS[1]]
    assert result['complete_cases'] == 25


@pytest.mark.parametrize('damage', ['zero', 'prefix', 'force', 'partial', 'order'])
def test_invalid_zero_prefix_force_or_partial_capture_is_inconclusive(damage):
    rows = synthetic_rows(); prefixes = ['2' * 64] * 25
    if damage == 'zero':
        rows[0]['numerical_diagnostic']['candidate_pass'] = False
    elif damage == 'prefix':
        prefixes[-1] = '3' * 64
    elif damage == 'force':
        rows[4]['pulse']['complete_phase_checks'] = False
    elif damage == 'partial':
        rows[4] = synthetic_score(screen.CELL_IDS[4], full=False)
        rows[4]['collection']['stop_reason'] = 'wall-budget-exhausted'
        rows[4]['numerical_diagnostic']['complete_first_attempt'] = False
    else:
        rows[1]['cell'] = 'unexpected-cell'
    if damage == 'order':
        with pytest.raises(ValueError, match='ordered independently scored broad-screen row'):
            screen._screen_inputs(rows, prefixes)
    else:
        result = screen._screen_inputs(rows, prefixes)
        assert result['decision'] == 'cpu-cardinal-dose-timing-inconclusive'
        assert result['promotion_authorized'] is False


def test_closed_cuda_parent_receipt_is_pinned_to_verified_result():
    assert screen.CLOSED_CUDA_SOURCE == '84ce52af2fc501830426a4290166f860a7d1a2fe'
    assert screen.CLOSED_CUDA_SHA256 == 'cb2c5cb2a53c67c7f84bf01df1a884d5ab5365e816c95cde051fc050b121a241'
    assert screen.PARENT_CHECKPOINT_SHA256 == '2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5'
    assert screen.PARENT_STATE_SHA256 == 'e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329'
    assert screen.PREPARE_FILES.isdisjoint({'report.json'})
    assert len(screen.CASE_FILES) == 75
    assert len(screen.PREPARE_FILES) == 78
    assert screen.COMPLETE_FILES == screen.PREPARE_FILES | {'report.json'}


def dose_timing_receipts():
    expected = {'checkpoint.pt', 'launch.json', 'report.json'} | {
        f'case-{index}{suffix}' for index in range(3)
        for suffix in ('.pt', '.json', '-replay.json')}
    report = dict(protocol=screen.prior.PROTOCOL,
        source=screen.CLOSED_DOSE_SOURCE,
        decision='frozen-dose-screen-no-deficit', optimizer_steps=0,
        cuda_initialized=False, **screen.base.baseline.FALSE_FLAGS)
    closeout = dict(protocol='cpu-dose-screen-independent-closeout-v1',
        source=screen.CLOSED_DOSE_SOURCE, report_sha256=screen.CLOSED_DOSE_REPORT_SHA256,
        decision='frozen-dose-screen-no-deficit', whole_cpu_rescore_identical=True,
        cases_checked=3, complete_force_delivery=True, prefixes_identical=True,
        optimizer_steps=0, cuda_initialized=False,
        files_rehashed={name: {'sha256': '0' * 64, 'bytes': 1} for name in expected},
        **screen.base.baseline.FALSE_FLAGS)
    closeout['independent_gpu_attestation'] = False
    closeout['whole_trajectory_physics_resimulated'] = False
    return closeout, report, expected


def test_completed_three_case_timing_receipt_is_separately_pinned():
    assert screen.CLOSED_DOSE_SOURCE == 'd55e7ef5f86ffdee69e2de468be64dc023c1bafe'
    assert screen.CLOSED_DOSE_SHA256 == '30bf131f203b8c38f20ee3644ba16912aa380e83130838504ee4322962066d12'
    assert screen.CLOSED_DOSE_REPORT_SHA256 == '220a5e7456468c0e175519a8a2d01423c495582d99a1768e15ad95c56fbf33f9'
    closeout, report, expected = dose_timing_receipts()
    assert len(expected) == 12
    screen._check_dose_timing_receipt(closeout, report,
        screen.CLOSED_DOSE_REPORT_SHA256, expected,
        expected | {'independent-closeout.json'})


def test_receipt_construction_has_one_owner_for_all_false_flags():
    # Synthetic metadata exercises the actual final constructor, not native proof.
    screen_result = screen._screen_inputs(synthetic_rows(), ['f' * 64] * 25)
    result = screen.closeout_result(SOURCE, 'b'*64, b'synthetic report',
        dict(screening=screen_result), {}, {}, 1.0)
    assert result['cases_checked'] == 25
    assert result['decision'] == 'cpu-cardinal-dose-timing-no-deficit'
    assert all(result[key] is False for key in screen.base.baseline.FALSE_FLAGS)


@pytest.mark.parametrize('damage', ['report-hash', 'source', 'cases', 'decision', 'inventory', 'cuda'])
def test_timing_receipt_rejects_mismatched_closeout_or_report(damage):
    closeout, report, expected = dose_timing_receipts()
    digest = screen.CLOSED_DOSE_REPORT_SHA256
    actual = expected | {'independent-closeout.json'}
    if damage == 'report-hash':
        digest = 'f' * 64
    elif damage == 'source':
        closeout['source'] = 'b' * 40
    elif damage == 'cases':
        closeout['cases_checked'] = 2
    elif damage == 'decision':
        report['decision'] = 'frozen-dose-screen-measured-deficit'
    elif damage == 'inventory':
        actual = expected
    else:
        closeout['cuda_initialized'] = True
    with pytest.raises(ValueError, match='complete no-deficit three-case CPU timing receipt'):
        screen._check_dose_timing_receipt(closeout, report, digest, expected, actual)
