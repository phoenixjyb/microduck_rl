"""Lean-lesson weight-initialized continuation: the CPU initialization path.

Declared by ``docs/experiments/2026-09-16-stance-lean-lesson.md`` under protocol
``football-b1n-lean-lesson-v1``. That document is a predeclaration: it fixes one
bounded lesson and its gates so the budget cannot be renegotiated afterwards.

What this module is
-------------------
The honest alternative to a resume. ``model_127.pt`` is a *weight export*: it
contains only ``identity`` and ``states``, with no Adam moments, no rollout
storage, no simulator state and no RNG state. So a run seeded from it shares a
point in parameter space with its parent but shares no optimization history.
This module exists to make that distinction structural rather than a matter of
good intentions:

- weights come from the pinned parent export, hash-checked before deserialization;
- the optimizer starts empty and is asserted empty after installation;
- the learner RNG, rollout storage and simulator state all start fresh;
- ``restored_fixture_only`` stays ``False``, so the unchanged guard in
  ``CpuStanceLearner.collect_one`` still permits a real plant. That guard is
  correct and is not touched;
- the recorded ``initial_state_sha256`` is the *loaded parent* state hash, and is
  asserted to differ from a fresh initializer.

What this module is not
-----------------------
There is deliberately no supervisor, no child process, no watchdog and no
service cap here. The predeclaration requires a separately declared *measured*
throughput probe to size those, and states plainly that the straight-line
estimate "must not be used to size the watchdog". Until that probe runs, any cap
written here would be an invented number, so none is written.
"""

from copy import deepcopy

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_ppo import CpuStanceLearner

MODULE = 'mjlab_microduck.stance_lean_lesson'
PROTOCOL = 'football-b1n-lean-lesson-v1'
SEED, WORLDS, UPDATES = checkpoint.LEAN_SEED, checkpoint.LEAN_WORLDS, checkpoint.LEAN_UPDATES
CHECKPOINTS = checkpoint.LEAN_CHECKPOINTS


def parent_record(raw, parent_identity):
    """Read-only load of the frozen parent export.

    The pinned byte hash is checked before any deserialization, so a corrupted
    or substituted parent export is refused rather than loaded.
    """
    return checkpoint.load_lean_parent(raw, checkpoint.LEAN_PARENT_SHA256, parent_identity)


class LeanStanceLearner(CpuStanceLearner):
    """Weight-initialized continuation over a real plant; fresh optimizer.

    ``_initialize`` builds the stock CPU actor/critic, a fresh Adam optimizer and
    an empty rollout store at the declared seed 571. ``install_parent`` then
    overwrites the actor and critic *in place*, so the optimizer and the policy
    keep referring to the very same modules and the loaded weights really are the
    weights the policy samples from.
    """

    UPDATE_LIMIT = UPDATES

    def __init__(self, parent):
        self._initialize(WORLDS, seed=SEED)
        self.parent_checkpoint_sha256 = None
        self.install_parent(parent)

    def install_parent(self, parent):
        require(type(parent) is dict, 'weight-initialized parent record')
        require(parent.get('weight_initialized') is True, 'weight initialization, not a resume')
        for key in ('optimizer_restored', 'simulator_restored', 'replay_restored',
                    'normalization_restored'):
            require(parent.get(key) is False, 'no restored state: '+key)
        require(parent.get('parent_checkpoint_sha256') == checkpoint.LEAN_PARENT_SHA256,
                'pinned lean-lesson parent export')
        require(self.parent_checkpoint_sha256 is None, 'parent installed exactly once')
        require(self.updates == 0 and self.phase == 'empty' and not self.faulted,
                'fresh learner before weight installation')
        # Weight initialization is not a fixture restore. If this were True the
        # unchanged collect_one guard would confine the run to a synthetic plant.
        require(self.restored_fixture_only is False, 'weight initialization is not a fixture restore')
        # The policy that samples and the optimizer that steps must be the very
        # modules being written, or the installed weights would be inert.
        require(self.algorithm.actor is self.actor and self.algorithm.critic is self.critic,
                'optimizer and policy share the initialized modules')
        require(not self.algorithm.optimizer.state, 'Adam moments start empty')
        actor, critic = parent.get('actor'), parent.get('critic')
        require(actor is not None and critic is not None, 'parent provides both model groups')
        self.actor.load_state_dict(actor.state_dict(), strict=True)
        self.critic.load_state_dict(critic.state_dict(), strict=True)
        self.initial_hash = checkpoint.state_hash(checkpoint.states_of(self.actor, self.critic))
        require(self.initial_hash == parent['parent_state_sha256'],
                'declared start equals the reviewed parent weights')
        fresh = checkpoint.state_hash(checkpoint.states_of(*checkpoint.fresh_models(SEED)))
        require(self.initial_hash != fresh, 'a lean-lesson start is not a fresh initializer')
        # Installation must not have created any optimizer state as a side effect.
        require(not self.algorithm.optimizer.state
                and all(not self.algorithm.optimizer.state.get(p)
                        for group in self.algorithm.optimizer.param_groups
                        for p in group['params']), 'no Adam moments after weight installation')
        self.parent_checkpoint_sha256 = parent['parent_checkpoint_sha256']

    @property
    def weight_initialized(self):
        return self.parent_checkpoint_sha256 is not None


def started_learner(raw, parent_identity):
    """Build a real, weight-initialized learner from the frozen parent export."""
    return LeanStanceLearner(parent_record(raw, parent_identity))


def identity(source, launch_sha, runtime_sha, learner, iteration):
    """Run identity for one lean-lesson export.

    Names the parent export hash explicitly and records the loaded parent state
    hash as the starting ``initial_state_sha256``. It never claims to continue
    the parent's optimization trajectory.
    """
    require(type(learner) is LeanStanceLearner and learner.weight_initialized,
            'identity belongs to a weight-initialized lean-lesson learner')
    return dict(protocol=checkpoint.PROTOCOL, source=source, runtime_sha256=runtime_sha,
        training_launch_sha256=launch_sha, purpose=checkpoint.LEAN_PURPOSE, training_seed=SEED,
        worlds=WORLDS, iteration=iteration, initial_state_sha256=learner.initial_hash,
        parent_checkpoint_sha256=learner.parent_checkpoint_sha256,
        architecture=deepcopy(checkpoint.ARCHITECTURE))
