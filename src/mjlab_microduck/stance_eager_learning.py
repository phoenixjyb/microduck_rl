"""One fresh 64-world/128-update eager learning diagnostic, not the full pilot."""
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

from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.stance_ppo import CpuStanceLearner, CONFIG, STEPS
from mjlab_microduck.first_attempt_smoke import canonical, require

host, supervisor = smoke.host, smoke.supervisor
MODULE = 'mjlab_microduck.stance_eager_learning'
PROTOCOL = 'football-b1n-eager-learning-diagnostic-v1'
WORLDS, UPDATES, SEED = 64, 128, 563
# Reuse the unchanged, independently watched 900-second process-group wrapper.
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT_SECONDS = 900, 960, 600


class EagerLearner(CpuStanceLearner):
    UPDATE_LIMIT = UPDATES

    def __init__(self):
        self._initialize(WORLDS, seed=SEED)


def output_path(source):
    supervisor.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-eager-learning-'+source[:12])


def service_name(source):
    supervisor.hex_id(source, 40)
    return 'microduck-stance-eager-'+source[:12]+'.service'


def check_window(deadline, *, launching=False):
    require(type(deadline) is int, 'explicit integer deadline')
    now = time.time()
    require(math.isfinite(now) and now+CLOSEOUT_SECONDS < deadline, 'eager closeout boundary')
    if launching:
        require(now+SERVICE_SECONDS+CLOSEOUT_SECONDS < deadline <= now+3600,
                'one bounded job needs a fresh 26-to-60-minute window')


def plan(source, inputs, runtime_sha, deadline):
    supervisor.hex_id(source, 40); supervisor.hex_id(runtime_sha, 64)
    require(type(deadline) is int and deadline > 0, 'explicit integer deadline')
    return dict(protocol=PROTOCOL, source=source, inputs=inputs, runtime_sha256=runtime_sha,
        purpose='eager-learning', worlds=WORLDS, seed=SEED, updates=UPDATES,
        steps_per_update=STEPS, optimizer=deepcopy(CONFIG), learner_device='cpu',
        physics_device='cuda:0', forward_graph=False, deadline_unix=deadline,
        child_timeout_seconds=CHILD_SECONDS, service_timeout_seconds=SERVICE_SECONDS,
        closeout_seconds=CLOSEOUT_SECONDS, checkpoints=list(range(-1, UPDATES)),
        simulation_resume_authorized=False, pilot_parent_authorized=False,
        learned_stance=False, physical_motion_authorized=False)


def prepare(source, deadline):
    check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only eager preparation')
    inputs = host.identity(source); learner = EagerLearner()
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    root = supervisor.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runtime.json', runtime)
    launch = plan(source, inputs, sha256(runtime).hexdigest(), deadline)
    supervisor.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source),
        launch_sha256=host.digest(root/'launch.json'), initial_state_sha256=learner.initial_hash)


def inputs_check(source, launch_sha):
    root = output_path(source); raw = supervisor.file_bytes(root/'launch.json')
    require(sha256(raw).hexdigest() == launch_sha, 'independent eager launch hash')
    launch = supervisor.parse(raw); runtime = supervisor.file_bytes(root/'runtime.json')
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest(),
                          launch['deadline_unix']), 'exact eager source/runtime/plan')
    plant.checked_runtime(supervisor.parse(runtime), source)
    return launch


def check_service(source):
    fields = ('MainPID', 'ActiveState', 'RuntimeMaxUSec', 'KillMode')
    raw = host.read('systemctl', '--user', 'show', service_name(source),
                    *('--property='+k for k in fields))
    state = dict(line.split('=', 1) for line in raw.splitlines())
    require(state == dict(MainPID=str(os.getpid()), ActiveState='active',
                          RuntimeMaxUSec='16min', KillMode='control-group'),
            'independent eager service boundary')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only eager supervisor')


def identity(source, launch_sha, runtime_sha, learner, iteration):
    return dict(protocol=checkpoint.PROTOCOL, source=source, runtime_sha256=runtime_sha,
        training_launch_sha256=launch_sha, purpose='eager-learning', training_seed=SEED,
        worlds=WORLDS, iteration=iteration, initial_state_sha256=learner.initial_hash,
        architecture=deepcopy(checkpoint.ARCHITECTURE))


def run_updates(learner, bridge, root, source, launch_sha, runtime_sha, *, deadline):
    require(type(learner) is EagerLearner and learner.n == bridge.n == WORLDS
            and learner.seed == SEED and learner.updates == 0
            and not learner.restored_fixture_only, 'fresh exact eager learner')
    def save(iteration):
        return smoke.save_weights(root, identity(source, launch_sha, runtime_sha, learner, iteration), learner)
    exports = [save(-1)]; started = time.monotonic()
    for update in range(UPDATES):
        for tick in range(STEPS):
            require(time.monotonic() < deadline, 'eager collection deadline')
            result = learner.collect_one(bridge)
            supervisor.write_json(root/f'tick-{update:03d}-{tick:02d}.json',
                                  smoke.tick_evidence(result, update*STEPS+tick))
        require(time.monotonic() < deadline, 'eager update deadline')
        metrics = learner.update(); saved = save(update); exports.append(saved)
        supervisor.write_json(root/f'update-{update:03d}.json', dict(iteration=update, **metrics,
            elapsed_s=time.monotonic()-started, checkpoint=saved))
        print(f'Eager completed update {update+1}/{UPDATES}', flush=True)
    return exports


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd); launch = inputs_check(source, launch_sha)
    check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit CUDA0 eager')
    host.wait_idle(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(WORLDS, device='cuda:0')
    require(str(env.device) == str(env.wp_device) == 'cuda:0' and env.wp_device.is_cuda
            and env.forward_graph is None, 'actual eager CUDA physics; no graph')
    root = output_path(source)
    runtime = supervisor.parse(supervisor.file_bytes(root/'runtime.json'))
    require(plant.describe(env.native) == runtime['plant'], 'actual eager plant')
    exports = run_updates(EagerLearner(), smoke.PhysicsBridge(env), root, source, launch_sha,
        launch['runtime_sha256'], deadline=time.monotonic()+CHILD_SECONDS-30)
    require(inputs_check(source, launch_sha) == launch and env.forward_graph is None, 'unchanged completed eager inputs')
    supervisor.write_json(root/'completed.json', dict(protocol=PROTOCOL, launch_sha256=launch_sha,
        completed_updates=UPDATES, checkpoints=exports, physics_device=str(env.device),
        learner_device='cpu', forward_graph=False, seed=SEED, worlds=WORLDS,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False))


def verify_completed(root, source, launch_sha, launch):
    result = supervisor.parse(supervisor.file_bytes(root/'completed.json'))
    require(result['protocol'] == PROTOCOL and result['launch_sha256'] == launch_sha
        and result['completed_updates'] == UPDATES and result['seed'] == SEED and result['worlds'] == WORLDS
        and result['physics_device'] == 'cuda:0' and result['learner_device'] == 'cpu'
        and all(result[k] is False for k in ('forward_graph', 'pilot_parent_authorized',
                                            'learned_stance', 'physical_motion_authorized')),
        'completed eager scope and counters')
    learner = EagerLearner(); models = checkpoint.fresh_models(SEED)
    require(len(result['checkpoints']) == UPDATES+1, 'all eager checkpoints')
    names = {'launch.json', 'runtime.json', 'child.log', 'completed.json'}
    for iteration, saved in zip(range(-1, UPDATES), result['checkpoints']):
        meta = identity(source, launch_sha, launch['runtime_sha256'], learner, iteration)
        name = 'initial.pt' if iteration == -1 else f'model_{iteration}.pt'; names.add(name)
        raw = supervisor.file_bytes(root/name, limit=checkpoint.LIMIT)
        require(saved == dict(file=name, sha256=sha256(raw).hexdigest(), identity=meta), 'retained eager checkpoint hash/identity')
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        require(set(value) == {'identity', 'states'} and value['identity'] == meta, 'saved eager metadata')
        checkpoint.validate_states(value['states'], *models)
        if iteration < 0:
            require(checkpoint.state_hash(value['states']) == learner.initial_hash, 'actual fresh eager initializer')
        else:
            name = f'update-{iteration:03d}.json'; names.add(name)
            update = supervisor.parse(supervisor.file_bytes(root/name))
            require(update['iteration'] == iteration and update['completed_updates'] == iteration+1
                and update['checkpoint'] == saved and update['checkpoint_admitted'] is False
                and update['physical_motion_authorized'] is False and update['cpu_fixture_only'] is False,
                'completed eager update receipt')
            require(update['metrics'] and all(type(v) in (int, float) and math.isfinite(v)
                    for v in update['metrics'].values()), 'finite retained optimizer metrics')
            for tick in range(STEPS):
                name = f'tick-{iteration:03d}-{tick:02d}.json'; names.add(name)
                record = supervisor.parse(supervisor.file_bytes(root/name))
                require(record['tick'] == iteration*STEPS+tick and len(record['reward']) == WORLDS
                        and len(record['terminal_records']) == WORLDS, 'complete ordered eager tick records')
    require({p.name for p in root.iterdir()} in (names, names|{'report.json'}), 'exact eager evidence inventory')
    return result


def supervise(source, launch_sha):
    check_service(source); launch = inputs_check(source, launch_sha)
    check_window(launch['deadline_unix'], launching=True); root = output_path(source)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'}, 'one fresh eager attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed',
        forward_graph=False, learned_stance=False, pilot_parent_authorized=False, physical_motion_authorized=False)
    try:
        with supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                check_window(launch['deadline_unix']); host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live eager source/runtime drift')
            report['child'] = supervisor.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log', cwd=host.ROOT,
                env=supervisor.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log'); verify_completed(root, source, launch_sha, launch)
            report['idle_after'] = host.wait_idle()
            report['decision'] = 'eager-learning-complete-not-capability'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        supervisor.write_json(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--deadline-unix', type=int)
    parser.add_argument('--launch-sha256')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': print(canonical(prepare(args.source, args.deadline_unix)))
    elif args.mode == 'supervise': supervise(args.source, args.launch_sha256)
    else: child(args.source, args.launch_sha256, args.lock_fd)


if __name__ == '__main__': main()
