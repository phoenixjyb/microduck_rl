"""Fresh-seed replication: declared seeds, distinct purpose, no moved bound.

These tests pin the guarantees from
``docs/experiments/2026-09-17-stance-lean-replication.md``. The replication
changes exactly one axis -- the learner seed -- so the two things that must hold
are that the seed set is a bounded declared list, and that neither purpose can
read, name or overwrite the other's evidence.
"""
from copy import deepcopy
from hashlib import sha256

import pytest

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_eager_learning as eager
from mjlab_microduck import stance_lean_evaluation as ev
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck import stance_lean_replication_campaign as camp

from test_stance_lean_evaluation import pinned_probe, synthetic_retained, synthetic_scores
from test_stance_lean_lesson import synthetic_parent

SOURCE, LAUNCH, RUNTIME = 'a'*40, 'b'*64, 'c'*64
OTHER_SOURCE = 'd'*40


def replication_learner(monkeypatch, seed=577):
    """A weight-initialized learner built at a declared replication seed."""
    raw, meta, digest = synthetic_parent(monkeypatch)
    parent = cp.load_lean_parent(raw, digest, meta)
    return lean.LeanStanceLearner(parent, seed=seed), parent, digest


# --- the declared seed set ----------------------------------------------------

def test_replication_seeds_are_the_declared_bounded_literals():
    assert cp.LEAN_REPLICATION_SEEDS == (577, 587, 593)
    assert all(type(s) is int for s in cp.LEAN_REPLICATION_SEEDS)
    assert cp.LEAN_REPLICATION_PURPOSE == 'lean-replication'
    assert cp.LEAN_REPLICATION_PURPOSE in cp.PURPOSES
    assert cp.LEAN_REPLICATION_PURPOSE not in cp.FRESH_PURPOSES


def test_replication_seeds_are_disjoint_from_every_other_declared_seed():
    """A seed is spent once: no replication seed may repeat another purpose's."""
    others = {s for purpose, (seeds, _, _) in cp.SCOPE.items()
              if purpose != cp.LEAN_REPLICATION_PURPOSE for s in seeds}
    others |= set(trace.SEEDS)
    assert not (set(cp.LEAN_REPLICATION_SEEDS) & others)
    assert len(set(cp.LEAN_REPLICATION_SEEDS)) == len(cp.LEAN_REPLICATION_SEEDS)


def test_every_replication_seed_is_admitted_and_its_neighbours_are_not():
    """The allowlist gained three literals, not a range."""
    for seed in cp.LEAN_REPLICATION_SEEDS:
        actor, critic = cp.fresh_models(seed)
        assert cp.state_hash(cp.states_of(actor, critic)) == cp.state_hash(cp.states_of(*cp.fresh_models(seed)))
    for seed in (576, 578, 586, 588, 592, 594, 579, 0, True):
        with pytest.raises(ValueError, match='predeclared fresh initialization seed'):
            cp.fresh_models(seed)


def test_the_replication_scope_matches_the_lesson_on_every_other_axis():
    """The seed is the single changed axis; worlds and budget are untouched."""
    lesson_seeds, lesson_worlds, lesson_updates = cp.SCOPE[cp.LEAN_PURPOSE]
    repl_seeds, repl_worlds, repl_updates = cp.SCOPE[cp.LEAN_REPLICATION_PURPOSE]
    assert lesson_seeds == (571,) and repl_seeds == cp.LEAN_REPLICATION_SEEDS
    assert (repl_worlds, repl_updates) == (lesson_worlds, lesson_updates) == (64, 256)
    assert cp.EVALUABLE[cp.LEAN_REPLICATION_PURPOSE] == cp.EVALUABLE[cp.LEAN_PURPOSE]


# --- neither purpose may borrow the other's identity --------------------------

def test_a_replication_identity_records_its_own_purpose_and_seed(monkeypatch):
    learner, _, _ = replication_learner(monkeypatch, 587)
    meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 64, lean.REPLICATION)
    assert meta['purpose'] == 'lean-replication' and meta['training_seed'] == 587
    assert meta['iteration'] == 64 and meta['worlds'] == 64
    # And it is a real identity: the loader that owns the purpose accepts it.
    cp.validate_identity(meta, evaluation=cp.LEAN_REPLICATION_PURPOSE)
    # The lesson's evaluation path must not admit it.
    with pytest.raises(ValueError, match='only declared lean-lesson checkpoints may be evaluated'):
        cp.validate_identity(meta, evaluation=cp.LEAN_PURPOSE)


def test_a_replication_seed_cannot_form_a_lesson_identity(monkeypatch):
    """The lesson declaration refuses a replication seed outright."""
    learner, _, _ = replication_learner(monkeypatch, 593)
    with pytest.raises(ValueError, match='declared learner seed for lean-lesson'):
        lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 64, lean.LESSON)


def test_the_two_purposes_cannot_load_each_others_exports(monkeypatch):
    """Both directions, on real encoded bytes rather than a hand-built dict."""
    learner, _, _ = replication_learner(monkeypatch, 577)
    repl_meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 64, lean.REPLICATION)
    raw = cp.encode(learner.actor, learner.critic, repl_meta)
    digest = sha256(raw).hexdigest()
    # Its own path accepts it.
    cp.load_lean_replication_evaluation(raw, digest, repl_meta)
    # The lesson's evaluation path must refuse a replication export.
    with pytest.raises(ValueError, match='lean-lesson evaluation export'):
        cp.load_lean_evaluation(raw, digest, repl_meta)
    # And the lesson's training loader must refuse it too.
    with pytest.raises(ValueError, match='only declared lean-lesson checkpoints'):
        cp.load_lean_lesson(raw, digest, repl_meta)


def test_a_lesson_export_is_not_readable_through_the_replication_loader(monkeypatch):
    raw, meta, digest = synthetic_parent(monkeypatch)
    parent = cp.load_lean_parent(raw, digest, meta)
    lesson_learner = lean.LeanStanceLearner(parent, seed=571)
    lesson_meta = lean.identity(SOURCE, LAUNCH, RUNTIME, lesson_learner, 64, lean.LESSON)
    blob = cp.encode(lesson_learner.actor, lesson_learner.critic, lesson_meta)
    blob_digest = sha256(blob).hexdigest()
    with pytest.raises(ValueError, match='lean replication evaluation export'):
        cp.load_lean_replication_evaluation(blob, blob_digest, lesson_meta)
    with pytest.raises(ValueError, match='only declared lean replication checkpoints'):
        cp.load_lean_replication(blob, blob_digest, lesson_meta)


def test_the_replication_loader_refuses_an_undeclared_iteration(monkeypatch):
    """Purpose alone is not enough; the iteration must be admitted as well."""
    learner, _, _ = replication_learner(monkeypatch, 577)
    meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 200, lean.REPLICATION)
    raw = cp.encode(learner.actor, learner.critic, meta)
    with pytest.raises(ValueError, match='only declared lean replication checkpoints'):
        cp.load_lean_replication(raw, sha256(raw).hexdigest(), meta)


# --- evidence paths -----------------------------------------------------------

def test_the_lesson_path_is_unchanged():
    """A regression guard: the frozen evidence directory must not move."""
    assert lean.output_path(SOURCE).name == 'stance-lean-lesson-' + SOURCE[:12]
    assert lean.service_name(SOURCE) == 'microduck-lean-lesson-' + SOURCE[:12] + '.service'


def test_replication_paths_carry_the_seed_so_three_runs_cannot_collide():
    paths = [lean.output_path(OTHER_SOURCE, lean.REPLICATION, seed)
             for seed in cp.LEAN_REPLICATION_SEEDS]
    services = [lean.service_name(OTHER_SOURCE, lean.REPLICATION, seed)
                for seed in cp.LEAN_REPLICATION_SEEDS]
    assert len(set(paths)) == len(set(services)) == len(cp.LEAN_REPLICATION_SEEDS)
    for seed, path, service in zip(cp.LEAN_REPLICATION_SEEDS, paths, services):
        assert path.name == f'stance-lean-replication-{OTHER_SOURCE[:12]}-seed-{seed}'
        assert service == f'microduck-lean-replication-{OTHER_SOURCE[:12]}-seed-{seed}.service'
    # And none of them is the lesson's directory.
    assert lean.output_path(OTHER_SOURCE).name not in {p.name for p in paths}


def test_an_undeclared_seed_is_refused_by_both_purposes():
    for declaration, seed in ((lean.LESSON, 577), (lean.LESSON, 593),
                              (lean.REPLICATION, 571), (lean.REPLICATION, 579)):
        with pytest.raises(ValueError, match='declared learner seed for ' + declaration['label']):
            lean.output_path(OTHER_SOURCE, declaration, seed)
        with pytest.raises(ValueError, match='declared learner seed for ' + declaration['label']):
            lean.service_name(OTHER_SOURCE, declaration, seed)


# --- the plan and the child command line --------------------------------------

def test_the_lesson_plan_is_unchanged():
    launch = lean.plan(SOURCE, {}, RUNTIME, 1)
    assert launch['protocol'] == 'football-b1n-lean-lesson-v1'
    assert launch['purpose'] == 'lean-lesson' and launch['seed'] == 571
    assert (launch['worlds'], launch['updates']) == (64, 256)


def test_the_replication_plan_changes_only_purpose_and_seed():
    lesson = lean.plan(SOURCE, {}, RUNTIME, 1)
    repl = lean.plan(SOURCE, {}, RUNTIME, 1, lean.REPLICATION, 587)
    assert repl['protocol'] == 'football-b1n-lean-replication-v1'
    assert repl['purpose'] == 'lean-replication' and repl['seed'] == 587
    differing = {k for k in lesson if lesson[k] != repl[k]}
    assert differing == {'protocol', 'purpose', 'seed'}


def test_the_plan_refuses_a_seed_the_declaration_does_not_name():
    with pytest.raises(ValueError, match='declared learner seed for lean-replication'):
        lean.plan(SOURCE, {}, RUNTIME, 1, lean.REPLICATION, 571)


def test_child_command_line_is_accepted_by_the_parsers_own_rules():
    """The supervisor's argv and the parser must agree, or a launch dies at once.

    The lean evaluation probe's first attempt failed exactly here: the supervisor
    built ``--mode`` while the parser declared ``--job``. Neither side was wrong
    alone, so only feeding one into the other catches it.
    """
    argv = lean.child_command(SOURCE, LAUNCH, 7, lean.REPLICATION, 593)
    args = lean.parser().parse_args(argv[3:])
    assert args.mode == 'child' and args.job == 'lean-replication' and args.seed == 593
    assert args.source == SOURCE and args.launch_sha256 == LAUNCH and args.lock_fd == 7
    # The lesson's own command line keeps its defaults, so its existing launches
    # are unaffected by the new options.
    lesson_args = lean.parser().parse_args(lean.child_command(SOURCE, LAUNCH, 7)[3:])
    assert lesson_args.job == 'lean-lesson' and lesson_args.seed == 571


def test_child_command_refuses_an_undeclared_seed():
    with pytest.raises(ValueError, match='declared learner seed for lean-replication'):
        lean.child_command(SOURCE, LAUNCH, 7, lean.REPLICATION, 571)


def test_an_unknown_job_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        lean.parser().parse_args(['child', '--source', SOURCE, '--job', 'lean-something'])


# --- the learner actually runs at the declared seed ---------------------------

def test_the_learner_is_built_at_the_declared_seed_and_keeps_parent_weights(monkeypatch):
    learner, parent, _ = replication_learner(monkeypatch, 577)
    assert (learner.n, learner.seed, learner.UPDATE_LIMIT) == (64, 577, 256)
    assert learner.weight_initialized is True and learner.restored_fixture_only is False
    # Weights are the parent's, not a seed-577 fresh initializer.
    assert learner.initial_hash == parent['parent_state_sha256']
    assert learner.initial_hash != cp.state_hash(cp.states_of(*cp.fresh_models(577)))
    # Initialization, not a resume.
    assert not learner.algorithm.optimizer.state


class Bridge:
    """Minimal stand-in: ``run_updates`` only reads ``n`` before its seed guard."""
    n = lean.WORLDS


def test_run_updates_refuses_a_lesson_seed_under_the_replication_declaration(monkeypatch):
    raw, meta, digest = synthetic_parent(monkeypatch)
    lesson_learner = lean.LeanStanceLearner(cp.load_lean_parent(raw, digest, meta), seed=571)
    with pytest.raises(ValueError, match='declared learner seed for lean-replication'):
        lean.run_updates(lesson_learner, Bridge(), None, SOURCE, LAUNCH, RUNTIME,
                         deadline=1e9, declaration=lean.REPLICATION)


# --- no frozen bound or gate moved --------------------------------------------

def test_no_existing_frozen_bound_moved():
    assert (lean.CHILD_SECONDS, lean.SERVICE_SECONDS) == (1693, 1753)
    assert lean.CLOSEOUT_SECONDS == 600 and lean.WATCHDOG_MARGIN_SECONDS == 60
    assert lean.SERVICE_RUNTIME_MAX == '29min 13s'
    assert (lean.WORLDS, lean.UPDATES) == (64, 256)
    assert lean.CHECKPOINTS == (64, 128, 192, 255)
    assert lean.SEED == cp.LEAN_SEED == 571


def test_the_two_declarations_are_registered_and_distinct():
    assert set(lean.DECLARATIONS) == {'lean-lesson', 'lean-replication'}
    assert lean.declaration_of('lean-replication') is lean.REPLICATION
    assert lean.declaration_of('lean-lesson') is lean.LESSON
    assert lean.LESSON['purpose'] != lean.REPLICATION['purpose']
    assert lean.LESSON['protocol'] != lean.REPLICATION['protocol']
    assert lean.LESSON['path_seed'] is False and lean.REPLICATION['path_seed'] is True
    with pytest.raises(ValueError, match='declared weight-initialized continuation'):
        lean.declaration_of('lean-anything')


# --- the judged continuation: its own protocol, names and decision strings -----

def replication_identity(iteration, seed=577):
    """A well-formed replication identity; no weights are needed to validate it."""
    return dict(protocol=cp.PROTOCOL, source=SOURCE, runtime_sha256='b'*64,
        training_launch_sha256='c'*64, purpose=cp.LEAN_REPLICATION_PURPOSE, training_seed=seed,
        worlds=cp.LEAN_WORLDS, iteration=iteration, initial_state_sha256='d'*64,
        parent_checkpoint_sha256=cp.LEAN_PARENT_SHA256, architecture=deepcopy(cp.ARCHITECTURE))


def replication_retained(seed=577):
    return dict(source=SOURCE, report_sha256='9'*64,
        checkpoints=[dict(file=f'model_{i}.pt', sha256='f'*64,
                          identity=replication_identity(i, seed))
                     for i in cp.LEAN_REPLICATION_CHECKPOINTS])


def replication_plan(seed=577):
    return ev.plan(SOURCE, dict(), 'e'*64, replication_retained(seed), 1789606469,
                   'evaluate', ev.REPLICATION)


def replication_scores(launch, *, passes=None):
    """Synthetic replication scores for the summarizer; never a result."""
    scores = {}
    for case in launch['cases']:
        count = ev.REQUIRED_PASSES if passes is None else passes
        attempts = [dict(world_id=i, complete_first_attempt=True, candidate_pass=i < count,
            last_physics_step=2500, hard_failure=False,
            metrics=dict(final_second_tilt_p95_rad=.05), gates=dict(full_duration=True))
            for i in range(ev.WORLDS)]
        score = dict(protocol=trace.LEAN_REPLICATION_PROTOCOL, binding=case['binding'],
            attempts=attempts, complete_attempts=ev.WORLDS, numerical_passes=count,
            terminal_contact_records_checked=ev.WORLDS)
        score.update({k: True for k in ('trajectory_continuity_validated', 'strict_checkpoint_checked',
            'deterministic_actor_replay_checked', 'compiled_plant_checked', 'nominal_reset_checked',
            'terminal_contact_summary_checked', 'kinematic_observations_checked', 'action_slew_checked',
            'delayed_motor_targets_checked', 'motor_commit_masks_checked', 'voltage_history_checked')})
        score.update({k: False for k in ('checkpoint_admitted', 'learned_stance_accepted',
            'physical_motion_authorized', 'provenance_validated')})
        scores[case['name']] = score
    return scores


def test_the_lesson_evaluation_plan_and_paths_are_unchanged(monkeypatch, tmp_path):
    """The frozen comparison path must not move under the generalization."""
    pinned_probe(monkeypatch, tmp_path)
    launch = ev.plan(SOURCE, dict(), 'e'*64, synthetic_retained(), 1789606469, 'evaluate', ev.LESSON)
    assert launch['protocol'] == 'football-b1n-lean-lesson-evaluation-v1'
    assert {c['binding']['protocol'] for c in launch['cases']} == {trace.LEAN_PROTOCOL}
    assert all(c['name'].startswith('lean-') for c in launch['cases'])
    assert ev.output_path(SOURCE, 'evaluate').name == 'stance-lean-evaluation-' + SOURCE[:12]
    assert ev.output_path(SOURCE, 'probe').name == 'stance-lean-eval-probe-' + SOURCE[:12]
    assert ev.service_name(SOURCE, 'evaluate') == \
        'microduck-lean-evaluation-' + SOURCE[:12] + '.service'
    # And the lesson summary keeps the key and strings already retained and hashed.
    summary = ev.summarize(launch, synthetic_scores(launch), ev.LESSON)
    assert summary['protocol'] == 'football-b1n-lean-lesson-evaluation-v1'
    assert summary['decision'] == 'lean-lesson-passed-nominal'
    assert summary['lean_lesson_numerical_gate_passed'] is True
    assert 'lean_replication_numerical_gate_passed' not in summary


def test_the_replication_plan_differs_only_in_identity_and_names(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    lesson = ev.plan(SOURCE, dict(), 'e'*64, synthetic_retained(), 1789606469, 'evaluate', ev.LESSON)
    repl = replication_plan(577)
    assert repl['protocol'] == 'football-b1n-lean-replication-evaluation-v1'
    assert {c['binding']['protocol'] for c in repl['cases']} == {trace.LEAN_REPLICATION_PROTOCOL}
    assert all(c['name'].startswith('repl-') for c in repl['cases'])
    assert {c['checkpoint']['identity']['purpose'] for c in repl['cases']} == \
        {cp.LEAN_REPLICATION_PURPOSE}
    assert {c['checkpoint']['identity']['training_seed'] for c in repl['cases']} == {577}
    # Same twelve cases, same worlds and the same unchanged gate: the comparison
    # is apples to apples with the lesson, which is the whole point.
    assert len(repl['cases']) == len(lesson['cases']) == 12
    assert repl['worlds_per_case'] == lesson['worlds_per_case'] == 128
    assert repl['attempts_required'] == lesson['attempts_required'] == 1536
    assert repl['tilt_gate_rad'] == lesson['tilt_gate_rad'] == ev.TILT_GATE_RAD


def test_the_replication_summary_uses_its_own_decision_strings(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = replication_plan(577)
    summary = ev.summarize(launch, replication_scores(launch), ev.REPLICATION)
    assert summary['protocol'] == 'football-b1n-lean-replication-evaluation-v1'
    assert summary['decision'] == 'lean-replication-seed-passed'
    assert summary['lean_replication_numerical_gate_passed'] is True
    assert 'lean_lesson_numerical_gate_passed' not in summary
    assert summary['passing_checkpoints'] == list(cp.LEAN_REPLICATION_CHECKPOINTS)
    assert summary['tilt_gate_rad'] == ev.TILT_GATE_RAD
    assert summary['tilt_gate_relaxed'] is False
    for flag in ('checkpoint_admitted', 'learned_stance_accepted',
                 'football_balance_accepted', 'physical_motion_authorized'):
        assert summary[flag] is False


def test_a_replication_seed_that_misses_the_gate_records_its_own_rejection(monkeypatch, tmp_path):
    pinned_probe(monkeypatch, tmp_path)
    launch = replication_plan(577)
    summary = ev.summarize(launch, replication_scores(launch, passes=100), ev.REPLICATION)
    assert summary['decision'] == 'lean-replication-seed-rejected'
    assert summary['passing_checkpoints'] == []
    assert summary['lean_replication_numerical_gate_passed'] is False


def test_a_replication_summary_cannot_be_scored_as_a_lesson(monkeypatch, tmp_path):
    """A replication launch cannot be judged under the lesson's declaration.

    The refusal comes early, at the pinned training source, because the lesson
    declaration names its own archive and a replication launch carries a
    different one. Either way it is refused before any score is read.
    """
    pinned_probe(monkeypatch, tmp_path)
    launch = replication_plan(577)
    with pytest.raises(ValueError, match='the declared completed training run'):
        ev.summarize(launch, replication_scores(launch), ev.LESSON)


def test_replication_evaluation_paths_carry_the_seed():
    names = {ev.output_path(SOURCE, 'evaluate', ev.REPLICATION, s).name
             for s in cp.LEAN_REPLICATION_SEEDS}
    assert len(names) == len(cp.LEAN_REPLICATION_SEEDS)
    for seed in cp.LEAN_REPLICATION_SEEDS:
        assert ev.output_path(SOURCE, 'evaluate', ev.REPLICATION, seed).name == \
            f'stance-lean-replication-eval-{SOURCE[:12]}-seed-{seed}'
        assert ev.service_name(SOURCE, 'evaluate', ev.REPLICATION, seed) == \
            f'microduck-lean-replication-eval-{SOURCE[:12]}-seed-{seed}.service'
    # A replication path is never the lesson's, and an undeclared seed is refused.
    assert ev.output_path(SOURCE, 'evaluate').name not in names
    with pytest.raises(ValueError, match='declared learner seed for lean-replication'):
        ev.output_path(SOURCE, 'evaluate', ev.REPLICATION, 571)


def test_the_evaluation_child_command_parses_for_both_continuations():
    argv = ev.child_command(SOURCE, LAUNCH, 'evaluate', 7, ev.REPLICATION, 587)
    args = ev.parser().parse_args(argv[3:])
    assert args.mode == 'child' and args.job == 'evaluate'
    assert args.continuation == 'lean-replication' and args.seed == 587
    assert args.source == SOURCE and args.launch_sha256 == LAUNCH and args.lock_fd == 7
    # The lesson's own line keeps its defaults, so existing launches are unaffected.
    lesson_args = ev.parser().parse_args(ev.child_command(SOURCE, LAUNCH, 'evaluate', 7)[3:])
    assert lesson_args.continuation == 'lean-lesson' and lesson_args.seed is None


def test_an_unknown_continuation_is_refused():
    with pytest.raises(SystemExit):
        ev.parser().parse_args(['child', '--source', SOURCE, '--continuation', 'lean-other'])
    with pytest.raises(ValueError, match='declared judged continuation'):
        ev.evaluation_of('lean-other')


def test_the_judged_continuations_are_registered_and_distinct():
    assert set(ev.EVALUATIONS) == {'lean-lesson', 'lean-replication'}
    assert ev.evaluation_of('lean-replication') is ev.REPLICATION
    assert ev.evaluation_of('lean-lesson') is ev.LESSON
    assert ev.LESSON['protocol'] != ev.REPLICATION['protocol']
    assert ev.LESSON['trace_protocol'] != ev.REPLICATION['trace_protocol']
    assert ev.LESSON['gate_key'] != ev.REPLICATION['gate_key']
    assert ev.LESSON['passed'] != ev.REPLICATION['passed']
    # The replication names its training source at prepare time; the lesson pins it.
    assert ev.LESSON['training_source'] is not None
    assert ev.REPLICATION['training_source'] is None
    # Both judge the same twelve cases under the same gate.
    assert ev.cases_of(ev.LESSON) == ev.cases_of(ev.REPLICATION) == ev.CASES == 12


def test_the_replication_loader_pairs_with_the_replication_training_declaration():
    assert ev.TRAINING_LOADERS['lean-lesson'] is cp.load_lean_evaluation
    assert ev.TRAINING_LOADERS['lean-replication'] is cp.load_lean_replication_evaluation
    assert set(ev.TRAINING_LOADERS) == set(ev.EVALUATIONS) == set(lean.DECLARATIONS)


# --- the cross-seed campaign: the declared rule over all three seeds ----------
#
# A per-seed verdict is not the experiment's verdict. One seed passing says
# nothing about whether the recipe replicates, so the campaign is what turns
# three per-seed summaries into the one string the predeclaration names.

def per_seed_summary(decision, *, checkpoints=None, complete=None, **overrides):
    """A per-seed verdict shaped like the one ``evaluation.verify`` returns.

    Only the keys ``campaign`` reads are meaningful here; the rest are the flags
    a real summary carries, so a fabricated one cannot pass by omission.
    """
    result = dict(protocol=ev.REPLICATION['protocol'], mode='evaluate', decision=decision,
        complete_attempts=ev.ATTEMPTS if complete is None else complete,
        passing_checkpoints=(list(cp.LEAN_REPLICATION_CHECKPOINTS)
                             if decision == camp.PASSED else []),
        training_source=OTHER_SOURCE, training_report_sha256='b'*64,
        report_sha256='c'*64, launch_sha256='d'*64,
        tilt_gate_rad=ev.TILT_GATE_RAD, tilt_gate_relaxed=False,
        checkpoint_admitted=False, learned_stance_accepted=False,
        football_balance_accepted=False, physical_motion_authorized=False)
    result.update(overrides)
    return result


def test_the_campaign_declares_the_four_verdicts_and_the_three_seeds():
    assert camp.PROTOCOL == 'football-b1n-lean-replication-campaign-v1'
    assert camp.SEEDS == cp.LEAN_REPLICATION_SEEDS == (577, 587, 593)
    assert camp.DECISIONS == ('lean-replication-passed', 'lean-replication-seed-dependent',
                              'lean-replication-rejected', 'lean-replication-incomplete')
    # The campaign's strings are not a per-seed verdict's, in either direction:
    # 'lean-replication-passed' is three seeds, 'lean-replication-seed-passed' is one.
    assert camp.PASSED == 'lean-replication-seed-passed'
    assert camp.REJECTED == 'lean-replication-seed-rejected'
    assert camp.PASSED not in camp.DECISIONS and camp.REJECTED not in camp.DECISIONS


def test_all_three_seeds_passing_is_the_declared_pass():
    verdicts = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS}
    assert camp.decide(verdicts) == 'lean-replication-passed'


@pytest.mark.parametrize('passing', [(577,), (587,), (593,), (577, 587), (577, 593), (587, 593)])
def test_some_but_not_all_seeds_passing_is_seed_dependent(passing):
    """One or two passing seeds is the experiment's verdict only as 'dependent'."""
    verdicts = {seed: per_seed_summary(camp.PASSED if seed in passing else camp.REJECTED)
                for seed in camp.SEEDS}
    assert camp.decide(verdicts) == 'lean-replication-seed-dependent'


def test_no_seed_passing_is_the_declared_rejection():
    verdicts = {seed: per_seed_summary(camp.REJECTED) for seed in camp.SEEDS}
    assert camp.decide(verdicts) == 'lean-replication-rejected'


@pytest.mark.parametrize('broken', [(577,), (587,), (593,)])
def test_one_unrunnable_seed_makes_the_whole_campaign_incomplete(broken):
    """The short-circuit outranks a pass, and the seed is not dropped.

    Two passing seeds and one that never ran must be reported as neither
    ``seed-dependent`` (two of three judged) nor ``passed`` (the two that ran).
    """
    verdicts = {seed: (dict(error='ValueError', message='no retained comparison')
                       if seed in broken else per_seed_summary(camp.PASSED))
                for seed in camp.SEEDS}
    assert camp.decide(verdicts) == 'lean-replication-incomplete'


def test_a_missing_or_invented_seed_cannot_change_the_denominator():
    short = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS[:2]}
    with pytest.raises(ValueError, match='all three declared replication seeds'):
        camp.decide(short)
    long = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS + (599,)}
    with pytest.raises(ValueError, match='all three declared replication seeds'):
        camp.decide(long)


@pytest.mark.parametrize('decision', ['lean-replication-passed', 'lean-replication-incomplete',
                                      'lean-replication-seed-dependent',
                                      'lean-lesson-passed-nominal', 'failed', None])
def test_a_verdict_the_campaign_may_not_combine_is_refused(decision):
    """A per-seed verdict is one of exactly two strings; nothing else combines."""
    verdicts = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS}
    verdicts[587] = per_seed_summary(decision)
    with pytest.raises(ValueError, match='per-seed verdicts this campaign may combine'):
        camp.decide(verdicts)


# --- the campaign document -----------------------------------------------------

def campaign_at(monkeypatch, tmp_path, verdicts, broken=(), wrong_training=()):
    """Run ``campaign`` against a temporary directory and injected verdicts.

    The per-seed re-derivation is ``seed_verdict``'s job and is exercised below
    and in the evaluation tests; what is under test here is the orchestration --
    that a broken seed is recorded rather than raised or dropped, and that the
    declared rule is what gets written down.
    """
    monkeypatch.setattr(camp, 'campaign_root', lambda source: tmp_path/'campaign')
    def fake(source, seed, training_source):
        assert training_source == OTHER_SOURCE
        if seed in broken:
            raise ValueError(f'no retained comparison for seed {seed}')
        return dict(verdicts[seed], training_source=('f'*40 if seed in wrong_training
                                                     else training_source))
    monkeypatch.setattr(camp, 'seed_verdict', fake)
    return camp.campaign(SOURCE, OTHER_SOURCE)


def test_the_campaign_records_the_declared_verdict_and_admits_nothing(monkeypatch, tmp_path):
    verdicts = {577: per_seed_summary(camp.PASSED), 587: per_seed_summary(camp.REJECTED),
                593: per_seed_summary(camp.PASSED)}
    result = campaign_at(monkeypatch, tmp_path, verdicts)
    assert result['protocol'] == camp.PROTOCOL and result['mode'] == 'evaluate'
    assert result['source'] == SOURCE and result['training_source'] == OTHER_SOURCE
    assert result['seeds'] == list(camp.SEEDS)
    assert result['decision'] == 'lean-replication-seed-dependent'
    assert result['passing_seeds'] == [577, 593]
    assert set(result['per_seed']) == {'577', '587', '593'}
    assert result['per_seed']['577']['decision'] == camp.PASSED
    assert result['per_seed']['577']['training_source'] == OTHER_SOURCE
    assert result['per_seed']['577']['training_report_sha256'] == 'b'*64
    assert result['per_seed']['577']['report_sha256'] == 'c'*64
    assert result['per_seed']['577']['launch_sha256'] == 'd'*64
    assert result['per_seed']['587']['passing_checkpoints'] == []
    assert result['tilt_gate_rad'] == ev.TILT_GATE_RAD and result['tilt_gate_relaxed'] is False
    for flag in ('checkpoint_admitted', 'learned_stance_accepted',
                 'football_balance_accepted', 'physical_motion_authorized'):
        assert result[flag] is False
    # Retained, not merely returned: the written bytes re-parse to this document.
    assert camp.files.parse(camp.files.file_bytes(tmp_path/'campaign'/'campaign.json')) == result


def test_a_seed_that_cannot_be_verified_is_recorded_rather_than_swallowed(monkeypatch, tmp_path):
    verdicts = {577: per_seed_summary(camp.PASSED), 587: per_seed_summary(camp.PASSED)}
    result = campaign_at(monkeypatch, tmp_path, verdicts, broken=(593,))
    assert result['decision'] == 'lean-replication-incomplete'
    # The two seeds that verified are still reported, and the third is an error
    # record -- not a rejection, and not an absence.
    assert result['passing_seeds'] == [577, 587]
    assert set(result['per_seed']) == {'577', '587', '593'}
    assert result['per_seed']['593']['error'] == 'ValueError'
    assert 'no retained comparison' in result['per_seed']['593']['message']
    assert 'decision' not in result['per_seed']['593']


def test_a_training_source_mismatch_makes_the_campaign_incomplete(monkeypatch, tmp_path):
    verdicts = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS}
    result = campaign_at(monkeypatch, tmp_path, verdicts, wrong_training=(593,))
    assert result['decision'] == 'lean-replication-incomplete'
    assert result['per_seed']['593']['error'] == 'ValueError'
    assert 'declared training source' in result['per_seed']['593']['message']
    assert 'decision' not in result['per_seed']['593']


def test_the_campaign_refuses_a_source_that_is_not_a_commit():
    with pytest.raises(ValueError, match='exact hexadecimal identity'):
        camp.campaign('not-a-commit', OTHER_SOURCE)
    with pytest.raises(ValueError, match='exact hexadecimal identity'):
        camp.campaign(SOURCE, 'short')
    with pytest.raises(ValueError, match='the replication training source to judge'):
        camp.campaign(SOURCE, None)


def test_the_campaign_directory_is_keyed_by_source_and_never_reused(monkeypatch, tmp_path):
    assert camp.campaign_root(SOURCE).name == 'stance-lean-replication-campaign-' + SOURCE[:12]
    assert camp.campaign_root(SOURCE).parent.name == 'evaluations'
    verdicts = {seed: per_seed_summary(camp.PASSED) for seed in camp.SEEDS}
    first = campaign_at(monkeypatch, tmp_path, verdicts)
    assert first['decision'] == 'lean-replication-passed'
    with pytest.raises(FileExistsError):
        campaign_at(monkeypatch, tmp_path, verdicts)


# --- a per-seed report this campaign will not even read -----------------------

def report_at(monkeypatch, tmp_path, **overrides):
    """A retained replication report at a temporary seed directory."""
    root = tmp_path/'seed-root'
    root.mkdir()
    report = dict(protocol=ev.REPLICATION['protocol'], mode='evaluate', decision=camp.PASSED,
                  comparison_sha256='a'*64, tilt_gate_rad=ev.TILT_GATE_RAD,
                  tilt_gate_relaxed=False)
    report.update(overrides)
    camp.files.write_json(root/'report.json', report)
    monkeypatch.setattr(camp, 'evaluation_root', lambda source, seed: root)
    return root


def test_a_seed_verdict_refuses_a_different_training_source(monkeypatch, tmp_path):
    root = report_at(monkeypatch, tmp_path)
    camp.files.write_json(root/'launch.json', dict(
        retained_training=dict(source=OTHER_SOURCE, report_sha256='e'*64)))
    with pytest.raises(ValueError, match='each replication seed judged the declared training source'):
        camp.seed_verdict(SOURCE, 577, training_source='f'*40)


def test_a_seed_verdict_refuses_a_report_that_is_not_a_replication_comparison(monkeypatch, tmp_path):
    report_at(monkeypatch, tmp_path, protocol=ev.LESSON['protocol'])
    with pytest.raises(ValueError, match='a replication comparison report'):
        camp.seed_verdict(SOURCE, 577)


@pytest.mark.parametrize('decision', ['failed', None, 'lean-replication-passed',
                                      'lean-replication-incomplete',
                                      'lean-lesson-passed-nominal'])
def test_a_seed_verdict_refuses_an_unfinished_or_foreign_report(monkeypatch, tmp_path, decision):
    """Only a completed per-seed verdict may enter the campaign's denominator."""
    report_at(monkeypatch, tmp_path, decision=decision)
    with pytest.raises(ValueError, match='a completed per-seed replication verdict'):
        camp.seed_verdict(SOURCE, 577)


def test_a_seed_verdict_refuses_a_report_whose_gate_was_relaxed(monkeypatch, tmp_path):
    report_at(monkeypatch, tmp_path, tilt_gate_relaxed=True)
    with pytest.raises(ValueError, match='a per-seed verdict with the gate unrelaxed'):
        camp.seed_verdict(SOURCE, 577)
