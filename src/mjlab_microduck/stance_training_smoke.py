"""One disposable 64-world/16-update smoke. No pilot, resume or promotion mode."""
import argparse
from copy import deepcopy
from hashlib import sha256
import io
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_ppo import CpuStanceLearner, CONFIG, STEPS, finite
from mjlab_microduck.first_attempt_smoke import canonical, require

supervisor = host.supervisor
PROTOCOL = 'football-b1n-disposable-training-smoke-v1'
WORLDS, UPDATES, SEED = 64, 16, 523
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT_SECONDS = 900, 960, 600


class SmokeStanceLearner(CpuStanceLearner):
    """Same CPU PPO math with fixed smoke bounds; no simulator resume codec."""
    UPDATE_LIMIT = UPDATES

    def __init__(self):
        self._initialize(WORLDS)


class PhysicsBridge:
    """CPU learner interface over an explicitly owned CPU-test or CUDA plant.

    Only transfer data; do not change reward, action, terminal or reset semantics.
    Raw sampled actions go to physics; applied corrections remain in observation.
    """
    def __init__(self, env):
        self.env = env; self.n = env.n; self.device = torch.device('cpu')
        self.physics_device = str(env.device)

    @property
    def live(self): return self.env.live.detach().cpu().clone()

    def observations(self): return trace.owned(self.env.observations())

    def step(self, action):
        finite(action, (self.n, 10), 'bridge action')
        return trace.owned(self.env.step(action.to(self.env.device)))

    def reset(self, rows):
        finite(rows, (self.n,), 'bridge reset mask', dtype=torch.bool)
        return self.env.reset(rows.to(self.env.device))


def output_path(source):
    supervisor.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-training-smoke-'+source[:12])


def service_name(source):
    supervisor.hex_id(source, 40)
    return 'microduck-stance-smoke-'+source[:12]+'.service'


def check_window():
    require(time.time()+SERVICE_SECONDS+CLOSEOUT_SECONDS < host.CUTOFF, 'full smoke and closeout must fit window')


def plan(source, inputs, runtime_sha):
    supervisor.hex_id(source, 40); supervisor.hex_id(runtime_sha, 64)
    return dict(protocol=PROTOCOL, source=source, inputs=inputs, runtime_sha256=runtime_sha,
        purpose='smoke', worlds=WORLDS, seed=SEED, updates=UPDATES, steps_per_update=STEPS,
        optimizer=deepcopy(CONFIG), learner_device='cpu', physics_device='cuda:0',
        child_timeout_seconds=CHILD_SECONDS, service_timeout_seconds=SERVICE_SECONDS,
        closeout_seconds=CLOSEOUT_SECONDS, cutoff_unix=host.CUTOFF,
        checkpoints=list(range(-1, UPDATES)), simulation_resume_authorized=False,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False)


def write_bytes(path, raw):
    require(type(raw) is bytes and 0 < len(raw) <= 16*1024*1024, 'bounded smoke bytes')
    path = supervisor.native._plain_path(path)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    supervisor.native._fsync_dir(path.parent)


def prepare(source):
    check_window()
    require(not torch.cuda.is_initialized(), 'CPU-only smoke preparation')
    inputs = host.identity(source)
    # Check both model and optimizer sources before any GPU allocation.
    learner = SmokeStanceLearner()
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    root = supervisor.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    write_bytes(root/'runtime.json', runtime)
    supervisor.write_json(root/'launch.json', plan(source, inputs, sha256(runtime).hexdigest()))
    require(learner.updates == 0 and not torch.cuda.is_initialized(), 'preflight cannot train')
    return dict(output=str(root), launch_sha256=host.digest(root/'launch.json'))


def inputs_check(source, launch_sha):
    supervisor.hex_id(launch_sha, 64)
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent smoke launch hash')
    launch = supervisor.parse(supervisor.file_bytes(root/'launch.json'))
    runtime = supervisor.file_bytes(root/'runtime.json')
    require(launch == plan(source, host.identity(source), sha256(runtime).hexdigest()), 'unchanged smoke launch inputs')
    plant.checked_runtime(supervisor.parse(runtime), source)
    return launch


def check_service(source):
    """Refuse a bare shell launch: verify the independently timed owner service."""
    values = {}
    for key in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState'):
        values[key] = host.read('systemctl', '--user', 'show', service_name(source), '-p', key, '--value')
    require(values == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='16min',
                           KillMode='control-group', ActiveState='active'), 'independently timed smoke service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU supervisor only')


def checkpoint_identity(source, launch_sha, runtime_sha, learner, iteration):
    return dict(protocol=checkpoint.PROTOCOL, source=source, runtime_sha256=runtime_sha,
        training_launch_sha256=launch_sha, purpose='smoke', training_seed=SEED,
        worlds=WORLDS, iteration=iteration, initial_state_sha256=learner.initial_hash,
        architecture=deepcopy(checkpoint.ARCHITECTURE))


def save_weights(root, identity, learner):
    require(not learner.faulted and learner.phase == 'empty', 'healthy complete update before export')
    raw = checkpoint.encode(learner.actor, learner.critic, identity)
    name = 'initial.pt' if identity['iteration'] == -1 else 'model_'+str(identity['iteration'])+'.pt'
    write_bytes(root/name, raw)
    return dict(file=name, sha256=sha256(raw).hexdigest(), identity=identity)


def tick_evidence(result, index):
    """Training diagnostics, not held-out no-reset performance scores."""
    frames = result['boundaries']
    require(len(frames) >= 1, 'training tick needs physical boundaries')
    # Count each executed row once; never count frozen failed siblings again.
    torque = speed = tilt = soft = samples = 0; height = None
    for before, after in zip(frames, frames[1:]):
        live = after['physics_steps'] > before['physics_steps']
        if not live.any(): continue
        state = after['state']
        torque = max(torque, float(state['torque'][live].abs().max()))
        speed = max(speed, float(state['joint_velocity'][live].abs().max()))
        tilt = max(tilt, float(state['tilt'][live].max()))
        value = float(state['height'][live].min()); height = value if height is None else min(height, value)
        soft += int(after['soft_limit_mask'][live].sum()); samples += int(live.sum())*14
    return dict(tick=index, reward=result['reward'].tolist(), executed_steps=result['executed_steps'].tolist(),
        reward_terms={k: v.tolist() for k, v in result.get('term_sums', {}).items()},
        failures=int(result['terminated'].sum()), timeouts=int(result['timed_out'].sum()),
        terminal_records=result['terminal_records'], max_applied_torque_nm=torque,
        max_joint_speed_rad_s=speed, max_tilt_rad=tilt, minimum_height_m=height,
        soft_joint_samples=soft, executed_joint_samples=samples,
        thermal_model_available=False, spring_bottoming_applicable=False)


def run_updates(learner, bridge, root, source, launch_sha, runtime_sha, *, deadline):
    """No environment constructor here. Test seam cannot attest GPU execution."""
    require(learner.n == bridge.n == WORLDS and learner.updates == 0, 'fresh exact smoke learner')
    exports = [save_weights(root, checkpoint_identity(source, launch_sha, runtime_sha, learner, -1), learner)]
    started = time.monotonic()
    for update in range(UPDATES):
        for tick in range(STEPS):
            require(time.monotonic() < deadline, 'smoke collection deadline')
            result = learner.collect_one(bridge)
            supervisor.write_json(root/f'tick-{update:02d}-{tick:02d}.json', tick_evidence(result, update*STEPS+tick))
        require(time.monotonic() < deadline, 'smoke update deadline')
        metrics = learner.update()
        record = save_weights(root, checkpoint_identity(source, launch_sha, runtime_sha, learner, update), learner)
        exports.append(record)
        supervisor.write_json(root/f'update-{update:02d}.json', dict(iteration=update, **metrics,
            elapsed_s=time.monotonic()-started, checkpoint=record))
        print(f'Smoke completed update {update+1}/{UPDATES}', flush=True)
    return exports


def inherited_lease(fd):
    inherited = os.fstat(fd); actual = supervisor.native._plain_path(supervisor.LOCK).stat()
    require((inherited.st_dev, inherited.st_ino) == (actual.st_dev, actual.st_ino), 'inherited GPU lease inode')
    try:
        with supervisor.gpu_lease(): raise ValueError('parent GPU lease is not held')
    except BlockingIOError: pass


def child(source, launch_sha, fd):
    inherited_lease(fd); check_window()
    launch = inputs_check(source, launch_sha)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit CUDA0 smoke')
    host.wait_idle()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(WORLDS, device='cuda:0')
    require(str(env.device) == str(env.wp_device) == 'cuda:0' and env.wp_device.is_cuda, 'actual CUDA physics')
    runtime = supervisor.parse(supervisor.file_bytes(output_path(source)/'runtime.json'))
    require(plant.describe(env.native) == runtime['plant'], 'actual smoke plant')
    learner = SmokeStanceLearner(); root = output_path(source)
    exports = run_updates(learner, PhysicsBridge(env), root, source, launch_sha, launch['runtime_sha256'],
                          deadline=time.monotonic()+CHILD_SECONDS-30)
    require(inputs_check(source, launch_sha) == launch, 'unchanged completed smoke inputs')
    supervisor.write_json(root/'completed.json', dict(protocol=PROTOCOL, launch_sha256=launch_sha,
        completed_updates=UPDATES, checkpoints=exports, physics_device=str(env.device),
        learner_device='cpu', seed=SEED, worlds=WORLDS, pilot_parent_authorized=False,
        learned_stance=False, physical_motion_authorized=False))


def verify_completed(root, source, launch_sha, launch):
    result = supervisor.parse(supervisor.file_bytes(root/'completed.json'))
    require(result['protocol'] == PROTOCOL and result['launch_sha256'] == launch_sha
        and result['completed_updates'] == UPDATES and result['seed'] == SEED and result['worlds'] == WORLDS
        and result['physics_device'] == 'cuda:0' and result['learner_device'] == 'cpu'
        and all(result[k] is False for k in ('pilot_parent_authorized', 'learned_stance', 'physical_motion_authorized')),
        'completed smoke scope and counters')
    learner = SmokeStanceLearner()
    require(len(result['checkpoints']) == UPDATES+1, 'all smoke checkpoints')
    for iteration, saved in zip(range(-1, UPDATES), result['checkpoints']):
        identity = checkpoint_identity(source, launch_sha, launch['runtime_sha256'], learner, iteration)
        name = 'initial.pt' if iteration == -1 else f'model_{iteration}.pt'
        raw = supervisor.file_bytes(root/name, limit=checkpoint.LIMIT)
        require(saved == dict(file=name, sha256=sha256(raw).hexdigest(), identity=identity), 'retained checkpoint identity')
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        require(value['identity'] == identity, 'saved smoke metadata')
        checkpoint.validate_states(value['states'], *checkpoint.fresh_models(SEED))
        if iteration < 0:
            require(checkpoint.state_hash(value['states']) == learner.initial_hash, 'actual fresh smoke initialization')
        else:
            update = supervisor.parse(supervisor.file_bytes(root/f'update-{iteration:02d}.json'))
            require(update['iteration'] == iteration and update['completed_updates'] == iteration+1
                    and update['checkpoint'] == saved and update['checkpoint_admitted'] is False
                    and update['physical_motion_authorized'] is False and update['cpu_fixture_only'] is False,
                    'completed update receipt')
            for tick in range(STEPS):
                record = supervisor.parse(supervisor.file_bytes(root/f'tick-{iteration:02d}-{tick:02d}.json'))
                require(record['tick'] == iteration*STEPS+tick and len(record['reward']) == WORLDS
                        and len(record['terminal_records']) == WORLDS, 'complete ordered training tick records')
    return result


def supervise(source, launch_sha):
    check_window(); check_service(source)
    root = output_path(source); launch = inputs_check(source, launch_sha)
    require({p.name for p in root.iterdir()} == {'launch.json', 'runtime.json'}, 'one fresh smoke attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed',
        learned_stance=False, pilot_parent_authorized=False, physical_motion_authorized=False)
    try:
        with supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+CLOSEOUT_SECONDS < host.CUTOFF, 'smoke closeout boundary')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live smoke source/runtime drift')
            report['child'] = supervisor.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'), '-m', __name__, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log', cwd=host.ROOT,
                env=supervisor.child_environment(), lock_fd=fd, guard=guard)
            host.check_log(root/'child.log')
            verify_completed(root, source, launch_sha, launch)
            report['idle_after'] = host.wait_idle()
            report['decision'] = 'disposable-training-smoke-complete-not-capability'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), error_notes=getattr(exc, '__notes__', []))
        raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        supervisor.write_json(root/'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    p.add_argument('--source', required=True)
    p.add_argument('--launch-sha256')
    p.add_argument('--lock-fd', type=int)
    args = p.parse_args()
    if args.mode == 'prepare': print(canonical(prepare(args.source)))
    elif args.mode == 'supervise': supervise(args.source, args.launch_sha256)
    else: child(args.source, args.launch_sha256, args.lock_fd)


if __name__ == '__main__': main()
