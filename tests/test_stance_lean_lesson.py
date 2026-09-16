"""Lean-lesson weight initialization: CPU fixtures only, never a GPU claim.

These tests pin the honesty-critical distinction from
``docs/experiments/2026-09-16-stance-lean-lesson.md``: a lean-lesson run starts
from reviewed frozen *weights* while its optimizer, rollout storage, simulator
state and RNG all start fresh. It is initialization, not a resume.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_eager_learning as eager
from mjlab_microduck import stance_lean_lesson as lean
from mjlab_microduck.stance_attempt_trace import CHECKPOINTS as PILOT_CHECKPOINTS
from mjlab_microduck.stance_ppo import CpuStanceLearner

PARENT_ARCHIVE = (Path(__file__).resolve().parents[1]/'artifacts/evaluations'
                  /'stance-eager-learning-c8f6b994a299/model_127.pt')
SOURCE, LAUNCH, RUNTIME = 'a'*40, 'b'*64, 'c'*64


class Synthetic:
    """Minimal real-plant stand-in that is explicitly not a synthetic fixture."""
    n = lean.WORLDS
    device = torch.device('cpu')

    def __init__(self):
        self.live = torch.ones(self.n, dtype=torch.bool)
        self.tick = 0
        self.obs = torch.zeros(self.n, 50)

    def observations(self):
        return dict(actor=self.obs[:, :44].clone(), critic=self.obs.clone())

    def step(self, action):
        self.tick += 1
        self.obs[:, 0] = self.tick/10000
        terminated = torch.zeros(self.n, dtype=torch.bool)
        timed_out = torch.zeros(self.n, dtype=torch.bool)
        self.live = ~(terminated | timed_out)
        self.records = [None]*self.n
        return dict(observation=self.observations(), reward=torch.full((self.n,), .03),
            terminated=terminated, timed_out=timed_out, live=self.live.clone(),
            executed_steps=torch.full((self.n,), 10, dtype=torch.long),
            terminal_records=deepcopy(self.records), boundaries=[{'synthetic': True}])

    def reset(self, rows):
        self.live[rows] = True
        return deepcopy(self.records)


def synthetic_parent(monkeypatch, *, seed=563, iteration=127, source=cp.LEAN_PARENT_SOURCE):
    """An eager final export standing in for the pinned artifact.

    The pinned parent hash is a deliberate literal, so exercising the mechanism
    requires re-pinning to a fixture. The real-artifact test below keeps the
    literal honest.
    """
    learner = eager.EagerLearner()
    assert learner.seed == seed
    meta = eager.identity(source, LAUNCH, RUNTIME, learner, iteration)
    raw = cp.encode(learner.actor, learner.critic, meta)
    digest = sha256(raw).hexdigest()
    monkeypatch.setattr(cp, 'LEAN_PARENT_SHA256', digest)
    monkeypatch.setattr(cp, 'LEAN_PARENT_SOURCE', source)
    monkeypatch.setattr(cp, 'LEAN_PARENT_ITERATION', iteration)
    return raw, meta, digest


def lean_learner(monkeypatch, **kwargs):
    raw, meta, digest = synthetic_parent(monkeypatch, **kwargs)
    parent = cp.load_lean_parent(raw, digest, meta)
    return lean.LeanStanceLearner(parent), parent, meta, digest


def test_declared_scope_matches_the_predeclaration():
    assert (lean.SEED, lean.WORLDS, lean.UPDATES) == (571, 64, 256)
    assert lean.CHECKPOINTS == (64, 128, 192, 255)
    assert lean.PROTOCOL == 'football-b1n-lean-lesson-v1'
    assert cp.LEAN_PURPOSE == 'lean-lesson'
    assert cp.LEAN_PARENT_FILE == 'model_127.pt' and cp.LEAN_PARENT_ITERATION == 127
    assert cp.LEAN_PARENT_SHA256 == (
        '46cd52b53f7b8b9fb220aed96d78cd961423c606e906a4df7330422ae4786e93')
    assert 'lean-lesson' in cp.PURPOSES and 'eager-learning' in cp.FRESH_PURPOSES
    assert 'lean-lesson' not in cp.FRESH_PURPOSES
    assert cp.EVALUABLE['pilot'] == PILOT_CHECKPOINTS
    assert cp.EVALUABLE['lean-lesson'] == lean.CHECKPOINTS


def test_fresh_seed_allowlist_admits_571_and_stays_bounded():
    actor, critic = cp.fresh_models(571)
    assert cp.state_hash(cp.states_of(actor, critic)) == (
        'a4ff457888cba6b452f7def475ac4e637b3b47fc1832ff8338d2341efebaf5f6')
    for seed in (570, 572, 0, 521*2, True):
        with pytest.raises(ValueError, match='predeclared fresh initialization seed'):
            cp.fresh_models(seed)


def test_weight_initialized_continuation_is_not_a_fresh_initializer(monkeypatch):
    learner, parent, _, _ = lean_learner(monkeypatch)
    assert (learner.n, learner.seed, learner.UPDATE_LIMIT) == (64, 571, 256)
    assert learner.updates == 0 and learner.phase == 'empty' and not learner.faulted
    # The guard that confines a fixture-only restore to a synthetic plant is
    # untouched, and this path must not trip it.
    assert learner.restored_fixture_only is False
    assert learner.weight_initialized is True
    assert learner.parent_checkpoint_sha256 == cp.LEAN_PARENT_SHA256
    assert learner.initial_hash == parent['parent_state_sha256']
    fresh = cp.state_hash(cp.states_of(*cp.fresh_models(lean.SEED)))
    assert learner.initial_hash != fresh
    # Initialization, not a resume: no optimizer, simulator or replay state.
    assert not learner.algorithm.optimizer.state
    assert learner.algorithm.actor is learner.actor and learner.algorithm.critic is learner.critic
    assert learner.actor.parameters().__next__().requires_grad is True
    assert torch.cuda.is_initialized() is False


def test_weight_initialized_learner_may_drive_a_real_plant(monkeypatch):
    """The whole point of not setting restored_fixture_only."""
    learner, _, _, _ = lean_learner(monkeypatch)
    env = Synthetic()
    assert not hasattr(env, 'synthetic_ppo_fixture')
    result = learner.collect_one(env)
    assert result['live'].all() and learner.phase == 'collecting'
    # The unchanged restore guard still refuses a fixture-only learner here.
    learner.restored_fixture_only = True
    with pytest.raises(ValueError, match='restored learner cannot resume simulator'):
        learner.collect_one(Synthetic())


def test_install_parent_is_once_only_and_requires_a_weight_only_record(monkeypatch):
    learner, parent, _, _ = lean_learner(monkeypatch)
    with pytest.raises(ValueError, match='parent installed exactly once'):
        learner.install_parent(parent)
    raw, meta, digest = synthetic_parent(monkeypatch, source='d'*40)
    fresh = cp.load_lean_parent(raw, digest, meta)
    for key in ('optimizer_restored', 'simulator_restored', 'replay_restored',
                'normalization_restored'):
        with pytest.raises(ValueError, match='no restored state: '+key):
            lean.LeanStanceLearner(dict(fresh, **{key: True}))
    with pytest.raises(ValueError, match='weight initialization, not a resume'):
        lean.LeanStanceLearner(dict(fresh, weight_initialized=False))
    with pytest.raises(ValueError, match='pinned lean-lesson parent export'):
        lean.LeanStanceLearner(dict(fresh, parent_checkpoint_sha256='0'*64))
    with pytest.raises(ValueError, match='identity belongs to a weight-initialized'):
        lean.identity(SOURCE, LAUNCH, RUNTIME, CpuStanceLearner(2), 64)
    with pytest.raises(ValueError, match='parent provides both model groups'):
        lean.LeanStanceLearner({k: v for k, v in fresh.items() if k != 'actor'})


def test_parent_is_read_only_and_corruption_is_refused_before_load(monkeypatch):
    raw, meta, digest = synthetic_parent(monkeypatch)
    before = sha256(raw).hexdigest()
    parent = cp.load_lean_parent(raw, digest, meta)
    assert sha256(raw).hexdigest() == before == digest
    assert parent['checkpoint_admitted'] is False
    with pytest.raises(ValueError, match='pinned lean-lesson parent export'):
        cp.load_lean_parent(raw, 'e'*64, meta)
    with pytest.raises(ValueError, match='parent checkpoint byte hash mismatch'):
        cp.load_lean_parent(raw[:-1]+bytes([raw[-1] ^ 1]), digest, meta)
    for changed in (dict(meta, iteration=126), dict(meta, source='d'*40),
                    dict(meta, purpose='pilot'), dict(meta, training_seed=523)):
        with pytest.raises(ValueError):
            cp.load_lean_parent(raw, digest, changed)
    with pytest.raises(ValueError):
        cp.load_lean_parent(meta, digest, meta)


def test_lean_identity_is_evaluable_only_on_its_own_path(monkeypatch):
    learner, _, _, _ = lean_learner(monkeypatch)
    meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 64)
    raw = cp.encode(learner.actor, learner.critic, meta)
    digest = sha256(raw).hexdigest()
    assert cp.load_lean_evaluation(raw, digest, meta)[0] is not None
    assert cp.load_lean_lesson(raw, digest, meta)[0] is not None
    # The retained pilot evaluation path must keep refusing these exports.
    with pytest.raises(ValueError, match='only declared pilot'):
        cp.load_evaluation(raw, digest, meta)
    with pytest.raises(ValueError, match='only eager initializer/final diagnostic'):
        cp.load_eager_diagnostic(raw, digest, meta)
    # And the lean paths must refuse a pilot export.
    pilot = CpuStanceLearner(2)
    pilot._initialize(2, seed=521)
    pilot_meta = dict(protocol=cp.PROTOCOL, source=SOURCE, runtime_sha256=RUNTIME,
        training_launch_sha256=LAUNCH, purpose='pilot', training_seed=521, worlds=512,
        iteration=PILOT_CHECKPOINTS[0], initial_state_sha256=pilot.initial_hash,
        architecture=dict(cp.ARCHITECTURE))
    pilot_raw = cp.encode(pilot.actor, pilot.critic, pilot_meta)
    pilot_digest = sha256(pilot_raw).hexdigest()
    assert cp.load_evaluation(pilot_raw, pilot_digest, pilot_meta)[0] is not None
    with pytest.raises(ValueError, match='lean-lesson evaluation export'):
        cp.load_lean_evaluation(pilot_raw, pilot_digest, pilot_meta)
    with pytest.raises(ValueError, match='only declared lean-lesson checkpoints'):
        cp.load_lean_lesson(pilot_raw, pilot_digest, pilot_meta)


def test_only_the_four_common_checkpoints_are_evaluable(monkeypatch):
    learner, _, _, _ = lean_learner(monkeypatch)
    initializer = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, -1)
    cp.validate_identity(initializer, evaluation=False)
    # Retained for the starting-state proof, but never scored.
    with pytest.raises(ValueError, match='only declared lean-lesson checkpoints'):
        cp.validate_identity(initializer, evaluation='lean-lesson')
    for iteration in lean.CHECKPOINTS:
        cp.validate_identity(lean.identity(SOURCE, LAUNCH, RUNTIME, learner, iteration),
                             evaluation='lean-lesson')
    # Structurally well-formed but undeclared: valid to save, never to score.
    for iteration in (0, 63, 100, 200):
        meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, iteration)
        cp.validate_identity(meta, evaluation=False)
        with pytest.raises(ValueError, match='only declared lean-lesson checkpoints'):
            cp.validate_identity(meta, evaluation='lean-lesson')
    for iteration in (256, 511, -2):
        with pytest.raises(ValueError, match='bounded saved iteration'):
            cp.validate_identity(lean.identity(SOURCE, LAUNCH, RUNTIME, learner, iteration),
                                 evaluation=False)


def test_lean_identity_cannot_borrow_a_fresh_or_unpinned_start(monkeypatch):
    learner, _, _, _ = lean_learner(monkeypatch)
    meta = lean.identity(SOURCE, LAUNCH, RUNTIME, learner, 64)
    fresh = cp.state_hash(cp.states_of(*cp.fresh_models(lean.SEED)))
    with pytest.raises(ValueError, match='a lean-lesson start must not equal a fresh initializer'):
        cp.validate_identity(dict(meta, initial_state_sha256=fresh), evaluation=False)
    with pytest.raises(ValueError, match='pinned lean-lesson parent export'):
        cp.validate_identity(dict(meta, parent_checkpoint_sha256='0'*64), evaluation=False)
    with pytest.raises(ValueError, match='exact checkpoint identity'):
        cp.validate_identity({k: v for k, v in meta.items() if k != 'parent_checkpoint_sha256'},
                             evaluation=False)
    # A fresh-purpose identity must not carry the lean-only parent key either.
    eager_meta = eager.identity(SOURCE, LAUNCH, RUNTIME, eager.EagerLearner(), 127)
    with pytest.raises(ValueError, match='exact checkpoint identity'):
        cp.validate_identity(dict(eager_meta, parent_checkpoint_sha256=cp.LEAN_PARENT_SHA256),
                             evaluation=False)


@pytest.mark.skipif(not PARENT_ARCHIVE.exists(), reason='pinned parent artifact not present')
def test_pinned_parent_artifact_matches_its_declared_hash():
    """Keeps the literal pin honest; skipped where the gitignored artifact is absent."""
    raw = PARENT_ARCHIVE.read_bytes()
    assert sha256(raw).hexdigest() == cp.LEAN_PARENT_SHA256
    import io
    identity = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['identity']
    assert identity['purpose'] == 'eager-learning'
    assert identity['iteration'] == cp.LEAN_PARENT_ITERATION
    assert identity['source'] == cp.LEAN_PARENT_SOURCE
    parent = cp.load_lean_parent(raw, cp.LEAN_PARENT_SHA256, identity)
    assert sha256(PARENT_ARCHIVE.read_bytes()).hexdigest() == cp.LEAN_PARENT_SHA256
    learner = lean.LeanStanceLearner(parent)
    assert learner.initial_hash == parent['parent_state_sha256']
    assert learner.restored_fixture_only is False
    assert learner.initial_hash != cp.state_hash(cp.states_of(*cp.fresh_models(lean.SEED)))
