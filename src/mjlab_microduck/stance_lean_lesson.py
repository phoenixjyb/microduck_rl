"""Lean-lesson weight-initialized continuation: initialization, run and supervisor.

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

Caps are measured, not invented
-------------------------------
The predeclaration forbids sizing the watchdog from the straight-line estimate and
requires a separately declared *measured* probe to do it. That probe has now run
(``docs/experiments/2026-09-16-stance-lean-lesson-throughput-probe.md``): it
measured a 5.395445651840419 s worst collection update and a 0.07417336199432611 s
worst CPU optimizer update, which give a 1,402 s prediction, a 1,753 s service cap
at the declared 1.25 factor and a 1,693 s child watchdog. Those are the constants
below. They are not rounded, and no bound is relaxed to make a run fit.

One honest caveat, stated here rather than discovered mid-launch. The probe's
collection timer covers the observation build, the policy forward pass and the
physics step, but not ``collect_one``'s own bookkeeping or the per-tick evidence
write. The parent's realized 5.7497 s/update shows that gap is about 0.35 s per
update, so a 256-update run is expected near 1,490 s against the 1,663 s internal
child deadline and the 1,693 s watchdog. The declared 1.25 factor is what covers
that difference; the remaining headroom is roughly 11%.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import io
import math
import os
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_ppo import CONFIG, CpuStanceLearner, STEPS

host, supervisor = smoke.host, smoke.supervisor
MODULE = 'mjlab_microduck.stance_lean_lesson'
PROTOCOL = 'football-b1n-lean-lesson-v1'
SEED, WORLDS, UPDATES = checkpoint.LEAN_SEED, checkpoint.LEAN_WORLDS, checkpoint.LEAN_UPDATES
CHECKPOINTS = checkpoint.LEAN_CHECKPOINTS
# Measured by the throughput probe; see the module docstring. Not estimates.
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT_SECONDS = 1693, 1753, 600
WATCHDOG_MARGIN_SECONDS = 60
# systemd's own rendering of `RuntimeMaxSec=1753`, read back from the host rather
# than assumed. Guessing this string is exactly how the probe's first launch
# failed its own self-check.
SERVICE_RUNTIME_MAX = '29min 13s'
MAX_WINDOW_SECONDS = 3600
SERVICE_PROPERTIES = ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState')

# The weight-initialized continuations this runner serves. They differ in exactly
# three things -- purpose, protocol and learner seed -- so they share one code
# path and one declaration shape rather than two copies that could drift apart.
# ``LESSON`` is the default on every function below, so the frozen 571 path is
# bit-for-bit unchanged and every existing call site keeps working.
#
# ``path_seed`` is explicit rather than inferred: the lesson's retained evidence
# lives at a source-keyed path that must not move, while the replication's three
# runs share one commit and would otherwise collide on a single directory.
LESSON = dict(label='lean-lesson', protocol=PROTOCOL, purpose=checkpoint.LEAN_PURPOSE,
              seeds=(SEED,), path_seed=False, directory='stance-lean-lesson-',
              service='microduck-lean-lesson-',
              decision='lean-lesson-complete-not-capability')
REPLICATION = dict(label='lean-replication',
                   protocol='football-b1n-lean-replication-v1',
                   purpose=checkpoint.LEAN_REPLICATION_PURPOSE,
                   seeds=checkpoint.LEAN_REPLICATION_SEEDS, path_seed=True,
                   directory='stance-lean-replication-',
                   service='microduck-lean-replication-',
                   decision='lean-replication-complete-not-capability')
DECLARATIONS = {d['label']: d for d in (LESSON, REPLICATION)}


def declaration_of(label):
    require(label in DECLARATIONS, 'declared weight-initialized continuation')
    return DECLARATIONS[label]


def seed_of(declaration, seed):
    """Refuse a seed the declaration did not name, and refuse a missing one."""
    require(type(seed) is int and seed in declaration['seeds'],
            'declared learner seed for '+declaration['label'])
    return seed


def parent_record(raw, parent_identity):
    """Read-only load of the frozen parent export.

    The pinned byte hash is checked before any deserialization, so a corrupted
    or substituted parent export is refused rather than loaded.
    """
    return checkpoint.load_lean_parent(raw, checkpoint.LEAN_PARENT_SHA256, parent_identity)


class LeanStanceLearner(CpuStanceLearner):
    """Weight-initialized continuation over a real plant; fresh optimizer.

    ``_initialize`` builds the stock CPU actor/critic, a fresh Adam optimizer and
    an empty rollout store at the declared learner seed, which defaults to the
    lesson's 571 and may be any seed a weight-initialized purpose declares.
    ``install_parent`` then overwrites the actor and critic *in place*, so the
    optimizer and the policy keep referring to the very same modules and the
    loaded weights really are the weights the policy samples from.
    """

    UPDATE_LIMIT = UPDATES

    def __init__(self, parent, seed=SEED):
        require(type(seed) is int and seed in checkpoint.FRESH_SEEDS,
                'declared weight-initialized learner seed')
        self._initialize(WORLDS, seed=seed)
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
        # Compared against this learner's own seed, not the lesson's. Reading the
        # module constant here would silently check a replication against the
        # seed-571 initializer instead of the one it was actually built at.
        fresh = checkpoint.state_hash(checkpoint.states_of(*checkpoint.fresh_models(self.seed)))
        require(self.initial_hash != fresh, 'a weight-initialized start is not a fresh initializer')
        # Installation must not have created any optimizer state as a side effect.
        require(not self.algorithm.optimizer.state
                and all(not self.algorithm.optimizer.state.get(p)
                        for group in self.algorithm.optimizer.param_groups
                        for p in group['params']), 'no Adam moments after weight installation')
        self.parent_checkpoint_sha256 = parent['parent_checkpoint_sha256']

    @property
    def weight_initialized(self):
        return self.parent_checkpoint_sha256 is not None


def started_learner(raw, parent_identity, seed=SEED):
    """Build a real, weight-initialized learner from the frozen parent export."""
    return LeanStanceLearner(parent_record(raw, parent_identity), seed=seed)


def identity(source, launch_sha, runtime_sha, learner, iteration, declaration=LESSON):
    """Run identity for one weight-initialized export.

    Names the parent export hash explicitly and records the loaded parent state
    hash as the starting ``initial_state_sha256``. It never claims to continue
    the parent's optimization trajectory.
    """
    require(type(learner) is LeanStanceLearner and learner.weight_initialized,
            'identity belongs to a weight-initialized learner')
    # Read from the learner rather than taken as a separate argument, so the
    # recorded seed cannot disagree with the seed the policy was built at.
    seed = seed_of(declaration, learner.seed)
    return dict(protocol=checkpoint.PROTOCOL, source=source, runtime_sha256=runtime_sha,
        training_launch_sha256=launch_sha, purpose=declaration['purpose'], training_seed=seed,
        worlds=WORLDS, iteration=iteration, initial_state_sha256=learner.initial_hash,
        parent_checkpoint_sha256=learner.parent_checkpoint_sha256,
        architecture=deepcopy(checkpoint.ARCHITECTURE))


def parent_path():
    """The pinned frozen parent export on this host; read-only input, never written."""
    return (host.ROOT/'artifacts/evaluations'
            /('stance-eager-learning-'+checkpoint.LEAN_PARENT_SOURCE[:12])
            /checkpoint.LEAN_PARENT_FILE)


def parent_bytes(root):
    """The copied parent archive, with its pinned hash checked before any load."""
    raw = supervisor.file_bytes(root/'parent.pt', limit=checkpoint.LIMIT)
    require(sha256(raw).hexdigest() == checkpoint.LEAN_PARENT_SHA256,
            'pinned lean-lesson parent export')
    return raw


def started_learner_from(raw, seed=SEED):
    """Deserialize only after the pinned hash has been verified."""
    digest = sha256(raw).hexdigest()
    require(digest == checkpoint.LEAN_PARENT_SHA256, 'pinned lean-lesson parent export')
    parent_identity = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['identity']
    return started_learner(raw, parent_identity, seed=seed)


def output_path(source, declaration=LESSON, seed=SEED):
    """Evidence directory. The lesson keeps its source-keyed path; a purpose whose
    runs share one commit carries the seed so they cannot overwrite each other."""
    supervisor.hex_id(source, 40); seed_of(declaration, seed)
    suffix = f'-seed-{seed}' if declaration['path_seed'] else ''
    return host.ROOT/'artifacts/evaluations'/(declaration['directory']+source[:12]+suffix)


def service_name(source, declaration=LESSON, seed=SEED):
    supervisor.hex_id(source, 40); seed_of(declaration, seed)
    suffix = f'-seed-{seed}' if declaration['path_seed'] else ''
    return declaration['service']+source[:12]+suffix+'.service'


def check_window(deadline, *, launching=False):
    """A fresh absolute window. Expired authority is never a window."""
    require(type(deadline) is int, 'explicit integer deadline')
    now = time.time()
    require(math.isfinite(now), 'finite clock')
    remaining = deadline-now
    require(remaining > 0, 'a window in the future; expired authority is not a window')
    require(remaining <= MAX_WINDOW_SECONDS, 'one bounded job inside 60 minutes')
    if launching:
        require(remaining > SERVICE_SECONDS+CLOSEOUT_SECONDS+WATCHDOG_MARGIN_SECONDS,
                'lean-lesson run needs a fresh 41-to-60-minute window')


def plan(source, inputs, runtime_sha, deadline, declaration=LESSON, seed=SEED):
    supervisor.hex_id(source, 40); supervisor.hex_id(runtime_sha, 64); seed_of(declaration, seed)
    require(type(deadline) is int and deadline > 0, 'explicit integer deadline')
    return dict(protocol=declaration['protocol'], source=source, inputs=inputs,
        runtime_sha256=runtime_sha,
        purpose=declaration['purpose'], worlds=WORLDS, seed=seed, updates=UPDATES,
        steps_per_update=STEPS, optimizer=deepcopy(CONFIG), learner_device='cpu',
        physics_device='cuda:0', forward_graph=False, deadline_unix=deadline,
        child_timeout_seconds=CHILD_SECONDS, service_timeout_seconds=SERVICE_SECONDS,
        closeout_seconds=CLOSEOUT_SECONDS, watchdog_margin_seconds=WATCHDOG_MARGIN_SECONDS,
        checkpoints=list(range(-1, UPDATES)), common_checkpoints=list(CHECKPOINTS),
        parent_checkpoint_sha256=checkpoint.LEAN_PARENT_SHA256,
        parent_source=checkpoint.LEAN_PARENT_SOURCE,
        parent_iteration=checkpoint.LEAN_PARENT_ITERATION, parent_file=checkpoint.LEAN_PARENT_FILE,
        weight_initialized=True, optimizer_state_restored=False,
        simulation_resume_authorized=False, pilot_parent_authorized=False,
        learned_stance=False, physical_motion_authorized=False)


def prepare(source, deadline, declaration=LESSON, seed=SEED):
    """CPU-only preparation. Copies the pinned parent in as a read-only input."""
    seed_of(declaration, seed)
    check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean-lesson preparation')
    inputs = host.identity(source)
    raw = supervisor.file_bytes(parent_path(), limit=checkpoint.LIMIT)
    require(sha256(raw).hexdigest() == checkpoint.LEAN_PARENT_SHA256,
            'pinned lean-lesson parent export')
    # Preflight the weight installation on CPU, before any GPU allocation exists.
    learner = started_learner_from(raw, seed=seed)
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    root = supervisor.native._plain_path(output_path(source, declaration, seed))
    root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'parent.pt', raw)
    smoke.write_bytes(root/'runtime.json', runtime)
    launch = plan(source, inputs, sha256(runtime).hexdigest(), deadline, declaration, seed)
    supervisor.write_json(root/'launch.json', launch)
    require(learner.updates == 0 and not torch.cuda.is_initialized(), 'preflight cannot train')
    return dict(output=str(root), service=service_name(source, declaration, seed),
        launch_sha256=host.digest(root/'launch.json'),
        parent_checkpoint_sha256=checkpoint.LEAN_PARENT_SHA256,
        initial_state_sha256=learner.initial_hash)


def inputs_check(source, launch_sha, declaration=LESSON, seed=SEED):
    root = output_path(source, declaration, seed); raw = supervisor.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'independent lean-lesson launch hash')
    launch = supervisor.parse(raw); runtime = supervisor.file_bytes(root/'runtime.json')
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest(),
                          launch['deadline_unix'], declaration, seed),
            'exact lean-lesson source/runtime/plan')
    plant.checked_runtime(supervisor.parse(runtime), source)
    # The parent archive is read-only input and must be intact at every check.
    parent_bytes(root)
    return launch


def check_service(source, declaration=LESSON, seed=SEED):
    """Refuse a bare shell launch: verify the independently timed owner service."""
    values = {k: host.read('systemctl', '--user', 'show', service_name(source, declaration, seed),
                           '-p', k, '--value') for k in SERVICE_PROPERTIES}
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec=SERVICE_RUNTIME_MAX,
                           KillMode='control-group', ActiveState='active'),
            'independently timed lean-lesson service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only lean-lesson supervisor')


def run_updates(learner, bridge, root, source, launch_sha, runtime_sha, *, deadline,
                declaration=LESSON):
    require(type(learner) is LeanStanceLearner and learner.n == bridge.n == WORLDS
            and learner.updates == 0 and learner.weight_initialized
            and not learner.restored_fixture_only, 'fresh exact weight-initialized learner')
    # Checked after the type guard: a learner that is not a weight-initialized
    # continuation at all should be refused as such, not as a wrong seed.
    seed_of(declaration, learner.seed)
    def save(iteration):
        return smoke.save_weights(root,
            identity(source, launch_sha, runtime_sha, learner, iteration, declaration), learner)
    exports = [save(-1)]; started = time.monotonic()
    for update in range(UPDATES):
        for tick in range(STEPS):
            require(time.monotonic() < deadline, 'lean-lesson collection deadline')
            result = learner.collect_one(bridge)
            supervisor.write_json(root/f'tick-{update:03d}-{tick:02d}.json',
                                  smoke.tick_evidence(result, update*STEPS+tick))
        require(time.monotonic() < deadline, 'lean-lesson update deadline')
        metrics = learner.update(); saved = save(update); exports.append(saved)
        supervisor.write_json(root/f'update-{update:03d}.json', dict(iteration=update, **metrics,
            elapsed_s=time.monotonic()-started, checkpoint=saved))
        print(f'Lean-lesson completed update {update+1}/{UPDATES}', flush=True)
    return exports


def child(source, launch_sha, fd, declaration=LESSON, seed=SEED):
    seed_of(declaration, seed)
    smoke.inherited_lease(fd); launch = inputs_check(source, launch_sha, declaration, seed)
    check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(),
            'explicit CUDA0 lean-lesson child')
    host.wait_idle(); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(WORLDS, device='cuda:0')
    require(str(env.device) == str(env.wp_device) == 'cuda:0' and env.wp_device.is_cuda
            and env.forward_graph is None, 'actual lean-lesson CUDA physics; no graph')
    root = output_path(source, declaration, seed)
    runtime = supervisor.parse(supervisor.file_bytes(root/'runtime.json'))
    require(plant.describe(env.native) == runtime['plant'], 'actual lean-lesson plant')
    raw = parent_bytes(root)
    before = sha256(raw).hexdigest()
    learner = started_learner_from(raw, seed=seed)
    exports = run_updates(learner, smoke.PhysicsBridge(env), root, source, launch_sha,
        launch['runtime_sha256'], deadline=time.monotonic()+CHILD_SECONDS-30,
        declaration=declaration)
    after = sha256(parent_bytes(root)).hexdigest()
    require(before == after == checkpoint.LEAN_PARENT_SHA256,
            'parent archive byte-identical after the lean lesson')
    require(inputs_check(source, launch_sha, declaration, seed) == launch
            and env.forward_graph is None, 'unchanged completed lean-lesson inputs')
    supervisor.write_json(root/'completed.json', dict(protocol=declaration['protocol'],
        launch_sha256=launch_sha,
        completed_updates=UPDATES, checkpoints=exports, physics_device=str(env.device),
        learner_device='cpu', forward_graph=False, seed=seed, worlds=WORLDS,
        purpose=declaration['purpose'], common_checkpoints=list(CHECKPOINTS),
        parent_checkpoint_sha256=checkpoint.LEAN_PARENT_SHA256,
        parent_source=checkpoint.LEAN_PARENT_SOURCE, weight_initialized=True,
        optimizer_state_restored=False, simulation_resume_authorized=False,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False))


def verify_completed(root, source, launch_sha, launch, declaration=LESSON, seed=SEED):
    seed_of(declaration, seed)
    result = supervisor.parse(supervisor.file_bytes(root/'completed.json'))
    require(result['protocol'] == declaration['protocol'] and result['launch_sha256'] == launch_sha
        and result['completed_updates'] == UPDATES and result['seed'] == seed
        and result['worlds'] == WORLDS and result['physics_device'] == 'cuda:0'
        and result['learner_device'] == 'cpu' and result['purpose'] == declaration['purpose']
        and result['parent_checkpoint_sha256'] == checkpoint.LEAN_PARENT_SHA256
        and result['common_checkpoints'] == list(CHECKPOINTS)
        and all(result[k] is False for k in ('forward_graph', 'optimizer_state_restored',
                'simulation_resume_authorized', 'pilot_parent_authorized', 'learned_stance',
                'physical_motion_authorized')),
        'completed lean-lesson scope and counters')
    require(result['weight_initialized'] is True, 'completed run was weight-initialized')
    raw = parent_bytes(root)
    learner = started_learner_from(raw, seed=seed)
    models = checkpoint.fresh_models(seed)
    fresh = checkpoint.state_hash(checkpoint.states_of(*models))
    require(len(result['checkpoints']) == UPDATES+1, 'all lean-lesson checkpoints')
    names = {'launch.json', 'runtime.json', 'parent.pt', 'child.log', 'completed.json'}
    for iteration, saved in zip(range(-1, UPDATES), result['checkpoints']):
        meta = identity(source, launch_sha, launch['runtime_sha256'], learner, iteration, declaration)
        name = 'initial.pt' if iteration == -1 else f'model_{iteration}.pt'; names.add(name)
        blob = supervisor.file_bytes(root/name, limit=checkpoint.LIMIT)
        value = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        require(saved == dict(file=name, sha256=sha256(blob).hexdigest(), identity=meta),
                'retained lean-lesson checkpoint hash/identity')
        require(set(value) == {'identity', 'states'} and value['identity'] == meta,
                'saved lean-lesson metadata')
        checkpoint.validate_states(value['states'], *models)
        if iteration < 0:
            # The retained initializer must be the parent's weights, and must not
            # be a fresh initializer. This is the weight-init claim, on the file.
            actual = checkpoint.state_hash(value['states'])
            require(actual == learner.initial_hash, 'retained initializer is the parent weights')
            require(actual != fresh, 'a lean-lesson start is not a fresh initializer')
        else:
            names.add(f'update-{iteration:03d}.json')
            update = supervisor.parse(supervisor.file_bytes(root/f'update-{iteration:03d}.json'))
            require(update['iteration'] == iteration and update['completed_updates'] == iteration+1
                and update['checkpoint'] == saved and update['checkpoint_admitted'] is False
                and update['physical_motion_authorized'] is False
                and update['cpu_fixture_only'] is False, 'completed lean-lesson update receipt')
            require(update['metrics'] and all(type(v) in (int, float) and math.isfinite(v)
                    for v in update['metrics'].values()), 'finite retained optimizer metrics')
            for tick in range(STEPS):
                names.add(f'tick-{iteration:03d}-{tick:02d}.json')
                record = supervisor.parse(supervisor.file_bytes(root/f'tick-{iteration:03d}-{tick:02d}.json'))
                require(record['tick'] == iteration*STEPS+tick and len(record['reward']) == WORLDS
                        and len(record['terminal_records']) == WORLDS,
                        'complete ordered lean-lesson tick records')
    require({p.name for p in root.iterdir()} in (names, names|{'report.json'}),
            'exact lean-lesson evidence inventory')
    return result


def child_command(source, launch_sha, fd, declaration=LESSON, seed=SEED):
    """The exact argv the supervisor hands to its bounded child.

    Kept in a function so a test can feed it back through ``parser()``. A
    supervisor that builds an option the parser does not declare fails only at
    launch time, which is precisely how the lean evaluation probe's first attempt
    died; the two sides must be checked against each other, not separately.
    """
    seed_of(declaration, seed)
    return [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
            '--launch-sha256', launch_sha, '--lock-fd', str(fd),
            '--job', declaration['label'], '--seed', str(seed)]


def supervise(source, launch_sha, declaration=LESSON, seed=SEED):
    seed_of(declaration, seed)
    check_service(source, declaration, seed)
    launch = inputs_check(source, launch_sha, declaration, seed)
    check_window(launch['deadline_unix'], launching=True)
    root = output_path(source, declaration, seed)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json', 'parent.pt'},
            'one fresh lean-lesson attempt')
    report = dict(protocol=declaration['protocol'], launch_sha256=launch_sha, decision='failed',
        purpose=declaration['purpose'], seed=seed, forward_graph=False, learned_stance=False,
        pilot_parent_authorized=False, physical_motion_authorized=False)
    try:
        with supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix']); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live lean-lesson source drift')
            report['child'] = supervisor.supervised_lean_lesson(
                child_command(source, launch_sha, fd, declaration, seed), root/'child.log',
                cwd=host.ROOT, env=supervisor.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            verify_completed(root, source, launch_sha, launch, declaration, seed)
            report['idle_after'] = host.wait_idle()
            report['decision'] = declaration['decision']
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc),
                      error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        supervisor.write_json(root/'report.json', report)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    result.add_argument('--source', required=True)
    result.add_argument('--deadline-unix', type=int)
    result.add_argument('--launch-sha256')
    result.add_argument('--lock-fd', type=int)
    # Which weight-initialized continuation this process is running, and at which
    # declared learner seed. Both are required to resolve the evidence directory,
    # so a supervisor and its child must agree on them or the child reads a
    # different launch document than the one the supervisor wrote.
    result.add_argument('--job', choices=tuple(DECLARATIONS), default=LESSON['label'])
    result.add_argument('--seed', type=int, default=SEED)
    return result


def main():
    args = parser().parse_args()
    declaration = declaration_of(args.job)
    if args.mode == 'prepare':
        print(canonical(prepare(args.source, args.deadline_unix, declaration, args.seed)))
    elif args.mode == 'supervise':
        supervise(args.source, args.launch_sha256, declaration, args.seed)
    else:
        child(args.source, args.launch_sha256, args.lock_fd, declaration, args.seed)


if __name__ == '__main__':
    main()
