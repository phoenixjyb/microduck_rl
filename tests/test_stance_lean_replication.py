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
from mjlab_microduck import stance_lean_lesson as lean

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
