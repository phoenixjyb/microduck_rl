"""Declared lean-lesson evaluation scope, caps and decision rule; no GPU here."""
from copy import deepcopy
from hashlib import sha256
import pytest
import time
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_lean_evaluation as ev
from mjlab_microduck.first_attempt_smoke import canonical


def lean_identity(iteration):
    """A well-formed lean-lesson identity; no weights are needed to validate it."""
    return dict(protocol=cp.PROTOCOL, source=ev.TRAINING_SOURCE, runtime_sha256='b'*64,
        training_launch_sha256='c'*64, purpose=cp.LEAN_PURPOSE, training_seed=cp.LEAN_SEED,
        worlds=cp.LEAN_WORLDS, iteration=iteration, initial_state_sha256='d'*64,
        parent_checkpoint_sha256=cp.LEAN_PARENT_SHA256, architecture=deepcopy(cp.ARCHITECTURE))


def synthetic_retained():
    return dict(source=ev.TRAINING_SOURCE, report_sha256=ev.TRAINING_REPORT,
        checkpoints=[dict(file=f'model_{i}.pt', sha256='f'*64, identity=lean_identity(i))
                     for i in ev.ITERATIONS])


def probe_plan():
    return ev.plan('a'*40, dict(), 'e'*64, synthetic_retained(), 1789606469, 'probe')


def evaluate_plan():
    return ev.plan('a'*40, dict(), 'e'*64, synthetic_retained(), 1789606469, 'evaluate')


def pinned_probe(monkeypatch, tmp_path, *, prelude_seconds=40.0, env_seconds=20.0, case_seconds=60.0):
    """Pin a synthetic but self-consistent probe so the evaluate mode can plan."""
    measurements = dict(prelude_seconds=prelude_seconds, env_seconds=env_seconds,
        case_seconds=case_seconds, policy_ticks=ev.POLICY_TICKS, stop_reason=ev.VALID_STOP_REASON,
        checkpoint_iteration=ev.PROBE_ITERATION, evaluation_seed=ev.PROBE_SEED,
        cases_projected=ev.CASES)
    report = dict(protocol=ev.PROTOCOL, mode='probe', decision='probe-measured',
                  launch_sha256='a'*64, measurements=measurements)
    root = tmp_path/'probe'; root.mkdir()
    raw = (canonical(report)+'\n').encode(); (root/'report.json').write_bytes(raw)
    derived = ev.derive_caps(report)
    monkeypatch.setattr(ev, 'PROBE_SOURCE', 'a'*40)
    monkeypatch.setattr(ev, 'PROBE_REPORT', sha256(raw).hexdigest())
    monkeypatch.setattr(ev, 'CHILD_SECONDS', derived['child_seconds'])
    monkeypatch.setattr(ev, 'SERVICE_SECONDS', derived['service_seconds'])
    monkeypatch.setattr(ev, 'probe_output_path', lambda: root)
    return derived


def synthetic_scores(launch, *, passes=None, tilt=.05):
    """Synthetic case scores; a fixture for the summarizer, never a result."""
    scores = {}
    for case in launch['cases']:
        count = ev.REQUIRED_PASSES if passes is None else passes
        attempts = [dict(world_id=i, complete_first_attempt=True, candidate_pass=i < count,
            last_physics_step=2500, hard_failure=False,
            metrics=dict(final_second_tilt_p95_rad=tilt), gates=dict(full_duration=True))
            for i in range(ev.WORLDS)]
        score = dict(protocol=trace.LEAN_PROTOCOL, binding=case['binding'], attempts=attempts,
            complete_attempts=ev.WORLDS, numerical_passes=count,
            terminal_contact_records_checked=ev.WORLDS)
        score.update({k: True for k in ('trajectory_continuity_validated', 'strict_checkpoint_checked',
            'deterministic_actor_replay_checked', 'compiled_plant_checked', 'nominal_reset_checked',
            'terminal_contact_summary_checked', 'kinematic_observations_checked', 'action_slew_checked',
            'delayed_motor_targets_checked', 'motor_commit_masks_checked', 'voltage_history_checked')})
        score.update({k: False for k in ('checkpoint_admitted', 'learned_stance_accepted',
            'physical_motion_authorized', 'provenance_validated')})
        scores[case['name']] = score
    return scores


def set_passes(score, count):
    """Change a case's pass count and its attempts consistently."""
    for i, attempt in enumerate(score['attempts']): attempt['candidate_pass'] = i < count
    score['numerical_passes'] = count


# --- caps: measured, not invented ---------------------------------------------

@pytest.mark.parametrize('seconds,rendered', [(960, '16min'), (1753, '29min 13s'), (900, '15min'),
    (1693, '28min 13s'), (45, '45s')])
def test_systemd_rendering_matches_the_host_read_strings(seconds, rendered):
    """960 s and 1,753 s are the two renderings actually read off the host."""
    assert ev.systemd_runtime_max(seconds) == rendered
    with pytest.raises(ValueError): ev.systemd_runtime_max(0)


def test_probe_reuses_the_frozen_pair_and_declares_no_new_bound():
    caps = ev.service_caps('probe')
    assert caps == dict(child_seconds=900, service_seconds=960, closeout_seconds=600, runtime_max='16min')
    assert caps['child_seconds'] == ev.smoke.CHILD_SECONDS
    assert caps['service_seconds'] == ev.smoke.SERVICE_SECONDS
    assert ev.supervisor_wrapper('probe') is ev.files.supervised_stance_smoke


def test_evaluate_caps_fail_closed_until_the_probe_is_pinned():
    assert ev.CHILD_SECONDS is None and ev.SERVICE_SECONDS is None and ev.PROBE_REPORT is None
    with pytest.raises(ValueError, match='not measured yet'):
        ev.service_caps('evaluate')
    with pytest.raises(ValueError, match='not measured yet'):
        evaluate_plan()
    with pytest.raises(ValueError, match='declared only after the probe is measured'):
        ev.supervisor_wrapper('evaluate')


def test_declared_cap_rule_applied_to_a_probe_report():
    report = dict(protocol=ev.PROTOCOL, mode='probe', decision='probe-measured',
        measurements=dict(prelude_seconds=40.0, env_seconds=20.0, case_seconds=60.0))
    derived = ev.derive_caps(report)
    assert derived['unit_seconds'] == 80.0
    assert derived['predicted_seconds'] == 40.0+ev.CASES*80.0 == 1000.0
    assert derived['service_seconds'] == 1250 and derived['child_seconds'] == 1190
    assert derived['safety_factor'] == 1.25 and derived['cases'] == 12


@pytest.mark.parametrize('damage', ['not_measured', 'wrong_protocol', 'wrong_mode', 'missing_timing',
    'negative', 'nan'])
def test_cap_rule_refuses_an_unusable_probe_report(damage):
    report = dict(protocol=ev.PROTOCOL, mode='probe', decision='probe-measured',
        measurements=dict(prelude_seconds=40.0, env_seconds=20.0, case_seconds=60.0))
    if damage == 'not_measured': report['decision'] = 'probe-invalid-short-case'
    elif damage == 'wrong_protocol': report['protocol'] = 'football-b1n-something-else'
    elif damage == 'wrong_mode': report['mode'] = 'evaluate'
    elif damage == 'missing_timing': del report['measurements']['case_seconds']
    elif damage == 'negative': report['measurements']['env_seconds'] = -1.0
    else: report['measurements']['prelude_seconds'] = float('nan')
    with pytest.raises(ValueError): ev.derive_caps(report)


def test_measured_caps_reject_a_hand_edited_constant(monkeypatch, tmp_path):
    derived = pinned_probe(monkeypatch, tmp_path)
    assert ev.service_caps('evaluate')['service_seconds'] == derived['service_seconds']
    monkeypatch.setattr(ev, 'SERVICE_SECONDS', derived['service_seconds']+1)
    monkeypatch.setattr(ev, 'CHILD_SECONDS', derived['child_seconds']+1)
    with pytest.raises(ValueError, match='probe-derived caps'):
        ev.service_caps('evaluate')


def test_measured_caps_refuse_a_probe_that_cannot_fit_one_window(monkeypatch, tmp_path):
    """The predeclared hard window condition, not a warning."""
    derived = pinned_probe(monkeypatch, tmp_path, case_seconds=250.0)
    assert derived['service_seconds'] > ev.MAX_WINDOW_SECONDS
    with pytest.raises(ValueError, match='fit one bounded window'):
        ev.service_caps('evaluate')


def test_measured_caps_require_the_watchdog_margin(monkeypatch, tmp_path):
    derived = pinned_probe(monkeypatch, tmp_path)
    monkeypatch.setattr(ev, 'SERVICE_SECONDS', derived['service_seconds']+5)
    monkeypatch.setattr(ev, 'CHILD_SECONDS', derived['child_seconds'])
    with pytest.raises(ValueError, match='probe-derived caps'):
        ev.service_caps('evaluate')


# --- windows ------------------------------------------------------------------

def test_window_requires_fresh_authority_and_refuses_expired():
    now = int(time.time())
    with pytest.raises(ValueError, match='expired authority'): ev.check_window(now-1, 'probe')
    with pytest.raises(ValueError, match='60 minutes'): ev.check_window(now+ev.MAX_WINDOW_SECONDS+60, 'probe')
    # 960 service + 600 closeout + 60 margin = 1,620 s, so a 1,000 s window is too short.
    with pytest.raises(ValueError, match='closeout reserve'):
        ev.check_window(now+1000, 'probe', launching=True)
    ev.check_window(now+2000, 'probe', launching=True)
    with pytest.raises(ValueError, match='explicit integer deadline'): ev.check_window(2000.0, 'probe')


# --- probe validity -----------------------------------------------------------

@pytest.mark.parametrize('ticks,reason,valid', [(250, ev.VALID_STOP_REASON, True),
    (59, ev.VALID_STOP_REASON, False), (250, 'wall-budget-exhausted', False),
    (250, 'policy-tick-limit', False), (249, ev.VALID_STOP_REASON, False)])
def test_probe_validity_requires_a_full_length_case(ticks, reason, valid):
    receipt = dict(collection=dict(policy_ticks=ticks, stop_reason=reason))
    assert ev.probe_validity(receipt) is valid


# --- plan scope ---------------------------------------------------------------

def test_probe_plan_is_one_case_inside_the_frozen_pair():
    launch = probe_plan()
    assert launch['mode'] == 'probe' and len(launch['cases']) == 1
    assert launch['cases_required'] == 1 and launch['attempts_required'] == ev.WORLDS
    assert launch['child_timeout_seconds'] == 900 and launch['service_timeout_seconds'] == 960
    assert launch['service_runtime_max'] == '16min' and launch['closeout_seconds'] == 600
    assert launch['policy_ticks'] == 250 and launch['worlds_per_case'] == ev.WORLDS
    case = launch['cases'][0]
    assert case['name'] == f'lean-{ev.PROBE_ITERATION}-seed-{ev.PROBE_SEED}'
    assert case['binding']['protocol'] == trace.LEAN_PROTOCOL
    assert case['binding']['checkpoint_iteration'] == ev.PROBE_ITERATION == 255
    assert case['binding']['evaluation_seed'] == ev.PROBE_SEED == 541


def test_evaluate_plan_declares_twelve_cases_and_no_admission(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan()
    assert launch['mode'] == 'evaluate' and len(launch['cases']) == 12
    assert launch['cases_required'] == ev.CASES == 12
    assert launch['attempts_required'] == ev.ATTEMPTS == 1536
    assert launch['service_runtime_max'] == ev.systemd_runtime_max(launch['service_timeout_seconds'])
    assert [c['binding']['checkpoint_iteration'] for c in launch['cases']] == [
        i for i in ev.ITERATIONS for _ in trace.SEEDS]
    assert [c['binding']['evaluation_seed'] for c in launch['cases']] == list(trace.SEEDS)*len(ev.ITERATIONS)
    assert launch['tilt_gate_rad'] == .0873 and launch['tilt_gate_relaxed'] is False
    assert all(launch[k] is False for k in ('checkpoint_admitted', 'learned_stance_accepted',
        'physical_motion_authorized', 'forward_graph'))


def test_probe_and_evaluation_never_share_a_directory_or_service():
    """A probe bundle must never be mistakable for an evaluation case."""
    for mode in ev.MODES:
        assert ev.DIRECTORIES[mode] in ev.output_path('a'*40, mode).name
        assert ev.service_name('a'*40, mode).startswith(ev.SERVICES[mode])
    assert ev.output_path('a'*40, 'probe') != ev.output_path('a'*40, 'evaluate')
    assert ev.service_name('a'*40, 'probe') != ev.service_name('a'*40, 'evaluate')
    with pytest.raises(ValueError, match='declared lean evaluation mode'):
        ev.output_path('a'*40, 'training')


def test_probe_protocol_is_the_lean_trace_protocol():
    """The probe and the evaluation share one trace protocol; only the job differs."""
    assert ev.PROBE_TRACE_PROTOCOL == trace.LEAN_PROTOCOL
    assert ev.PROBE_TRACE_PROTOCOL in trace.ITERATIONS


# --- the decision rule --------------------------------------------------------

def test_a_probe_is_not_a_decision():
    with pytest.raises(ValueError, match='a probe is not a decision'): ev.summarize(probe_plan(), {})


def test_decision_rule_passes_only_on_all_three_seeds(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan()
    summary = ev.summarize(launch, synthetic_scores(launch))
    assert summary['decision'] == 'lean-lesson-passed-nominal'
    assert summary['passing_checkpoints'] == list(ev.ITERATIONS)
    assert summary['complete_attempts'] == 1536 and summary['lean_lesson_numerical_gate_passed'] is True
    assert summary['tilt_gate_rad'] == .0873 and summary['tilt_gate_relaxed'] is False
    assert not summary['checkpoint_admitted'] and not summary['learned_stance_accepted']
    assert not summary['football_balance_accepted'] and not summary['physical_motion_authorized']


def test_a_near_miss_on_one_seed_rejects_that_checkpoint(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan(); scores = synthetic_scores(launch)
    set_passes(scores['lean-128-seed-547'], ev.REQUIRED_PASSES-1)
    summary = ev.summarize(launch, scores)
    assert summary['per_checkpoint']['128']['passes_all_seeds'] is False
    assert summary['per_checkpoint']['128']['seeds']['547']['meets_original_per_seed_threshold'] is False
    assert summary['passing_checkpoints'] == [64, 192, 255]
    assert summary['decision'] == 'lean-lesson-passed-nominal'


def test_no_checkpoint_passing_records_the_declared_rejection(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan(); scores = synthetic_scores(launch)
    for case in scores: set_passes(scores[case], 0)
    summary = ev.summarize(launch, scores)
    assert summary['passing_checkpoints'] == [] and summary['decision'] == 'lean-lesson-rejected-objective-binds'
    assert summary['lean_lesson_numerical_gate_passed'] is False


def test_tilt_is_reported_per_seed_for_every_checkpoint(monkeypatch, tmp_path):
    """The trend across 64/128/192/255 is the budget-versus-objective evidence."""
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan(); summary = ev.summarize(launch, synthetic_scores(launch, tilt=.14))
    assert set(summary['per_checkpoint']) == {str(i) for i in ev.ITERATIONS}
    for iteration in ev.ITERATIONS:
        seeds = summary['per_checkpoint'][str(iteration)]['seeds']
        assert set(seeds) == {str(s) for s in trace.SEEDS}
        for seed in trace.SEEDS:
            assert seeds[str(seed)]['tilt_p95_rad_max'] == .14
            assert seeds[str(seed)]['full_duration_attempts'] == ev.WORLDS


def test_a_dead_case_reports_no_tilt_rather_than_imputing_one(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan(); scores = synthetic_scores(launch)
    case = 'lean-64-seed-541'
    for attempt in scores[case]['attempts']:
        attempt['metrics']['final_second_tilt_p95_rad'] = None
    set_passes(scores[case], 0)
    seed = ev.summarize(launch, scores)['per_checkpoint']['64']['seeds']['541']
    assert seed['full_duration_attempts'] == 0
    assert seed['tilt_p95_rad_max'] is None and seed['tilt_p95_rad_min'] is None
    assert seed['tilt_p95_rad_mean'] is None


@pytest.mark.parametrize('damage', ['missing_case', 'partial', 'false_count', 'admitted',
    'relaxed_gate', 'changed_plan', 'wrong_protocol'])
def test_incomplete_or_inconsistent_evidence_cannot_become_a_decision(monkeypatch, tmp_path, damage):
    pinned_probe(monkeypatch, tmp_path)
    launch = evaluate_plan(); scores = synthetic_scores(launch)
    one = scores['lean-192-seed-557']
    if damage == 'missing_case': scores.pop('lean-64-seed-541')
    elif damage == 'partial': one['attempts'][0]['complete_first_attempt'] = False
    elif damage == 'false_count': one['numerical_passes'] = ev.REQUIRED_PASSES+1
    elif damage == 'admitted': one['checkpoint_admitted'] = True
    elif damage == 'relaxed_gate': launch['tilt_gate_relaxed'] = True
    elif damage == 'changed_plan': launch['worlds_per_case'] = 64
    else: one['protocol'] = trace.PROTOCOL
    with pytest.raises(ValueError): ev.summarize(launch, scores)


def test_gate_constant_is_the_scorers_own_and_is_not_relaxable():
    """The 0.0873 rad limit is a named predeclared constant, not a tunable."""
    assert ev.TILT_GATE_RAD == ev.evaluation.TILT_GATE_RAD == .0873


# --- the command line ---------------------------------------------------------

def test_child_command_line_is_accepted_by_the_parsers_own_rules():
    """The supervisor's child argv and ``main`` must agree.

    This is the check whose absence let a real launch die with
    ``unrecognized arguments: --mode probe``: ``supervise`` built the child
    command with one flag name while the parser declared another. A unit test of
    either side alone would not have caught it.
    """
    for mode in ev.MODES:
        argv = ev.child_command('a'*40, 'b'*64, mode, 7)
        assert argv[:3] == [str(ev.host.ROOT/'.venv/bin/python'), '-m', ev.MODULE]
        parsed = ev.parser().parse_args(argv[3:])
        assert parsed.mode == 'child' and parsed.job == mode
        assert parsed.source == 'a'*40 and parsed.launch_sha256 == 'b'*64 and parsed.lock_fd == 7
    with pytest.raises(ValueError, match='declared lean evaluation mode'):
        ev.child_command('a'*40, 'b'*64, 'training', 7)


def test_every_declared_command_line_parses():
    for mode in ev.MODES:
        prepare = ev.parser().parse_args(['prepare', '--source', 'a'*40,
                                          '--deadline-unix', '123', '--job', mode])
        assert prepare.mode == 'prepare' and prepare.job == mode and prepare.deadline_unix == 123
        supervise = ev.parser().parse_args(['supervise', '--source', 'a'*40,
                                            '--launch-sha256', 'b'*64, '--job', mode])
        assert supervise.mode == 'supervise' and supervise.job == mode
    # An undeclared job, or a missing required source, is refused by the parser.
    with pytest.raises(SystemExit): ev.parser().parse_args(['child', '--source', 'a'*40, '--job', 'nonsense'])
    with pytest.raises(SystemExit): ev.parser().parse_args(['prepare', '--job', 'probe'])

