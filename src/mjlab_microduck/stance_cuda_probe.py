"""One source-bound, leased, timed CUDA integration probe; never an optimizer."""

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Canonical package registration before importing the BAM-dependent adapters.
import mjlab
import torch

from mjlab_microduck import foundation_command_campaign as supervisor
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.football_stance_probe import asset_hashes
from mjlab_microduck.gpu_idle_gate import wait_idle

PROTOCOL = 'football-b1n-cuda-integration-v1'
ROOT = Path('/home/converge/work/microduck_rl-athletics-obstacle-curriculum')
BRANCH = 'feat/athletics-obstacle-curriculum'
MACHINE = '0c79e415429b4933a400159bfa79a34d'
GPU = 'GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd'
DRIVER = '595.84'
CUTOFF = datetime(2026, 9, 9, 23, 30, tzinfo=timezone.utc).timestamp()
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT_SECONDS = 120, 180, 600
PIN_FILE = Path(__file__).resolve().parents[2]/'docs/experiments/2026-09-09-stance-cuda-probe-runtime.json'
PACKAGES = {'mjlab': 'mjlab', 'mujoco-warp': 'mujoco_warp', 'better-actuator-models': 'bam'}
VERSIONS = {'torch': '2.9.1', 'warp-lang': '1.12.0', 'mujoco': '3.10.0',
            'mjlab': '1.3.0', 'mujoco-warp': '3.8.1', 'better-actuator-models': '1.0.1'}
BAD_LOG = re.compile(r'\b(?:warning|warn|overflow|nan|inf(?:inity)?|nonfinite)\b', re.I)
STARTUP_INFO = '[mdp] Patches 1-2 active: NaN-safe reward/advantage'


def digest(path):
    return sha256(supervisor.file_bytes(path, allow_empty=True)).hexdigest()


def python_trees():
    result = {}
    for dist, package in PACKAGES.items():
        root = distribution(dist).locate_file(package)
        files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
        require(len(files) > 5, 'nonempty installed Python source tree')
        raw = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
        result[dist] = dict(version=version(dist), python_file_count=len(files),
                            python_tree_sha256=sha256(raw).hexdigest())
    return result


def read(*command):
    return subprocess.check_output(command, text=True, timeout=5).strip()


def identity(source):
    """Read-only host/source/runtime/asset check, with no CUDA allocation."""
    supervisor.hex_id(source, 40)
    require(sys.platform == 'linux' and Path.cwd().resolve() == ROOT, 'exact Linux worktree')
    require(Path(sys.prefix).resolve() == (ROOT/'.venv').resolve(), 'exact frozen environment')
    require(Path('/etc/machine-id').read_text().strip() == MACHINE, 'exact authorized host')
    require(read('git', 'branch', '--show-current') == BRANCH
            and read('git', 'rev-parse', 'HEAD') == source
            and not read('git', 'status', '--porcelain'), 'clean exact source')
    gpu = read('nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits')
    require([v.strip() for v in gpu.split(',')] == [GPU, DRIVER], 'exact single GPU and driver')
    versions = {name: version(name) for name in VERSIONS}
    require(versions == VERSIONS, 'frozen package versions')
    trees = python_trees()
    require(trees == supervisor.parse(PIN_FILE.read_bytes()), 'reviewed dependency Python trees')
    from mjlab_microduck.robot.microduck_constants import actuators
    return dict(source=source, branch=BRANCH, machine_id=MACHINE, gpu_uuid=GPU,
        driver=DRIVER, python_version=sys.version.split()[0], versions=versions,
        dependency_python_trees=trees, robot_assets=asset_hashes(),
        motor_parameters_sha256=digest(Path(actuators._resolved_json_path)),
        lock_sha256=digest(ROOT/'uv.lock'), pyproject_sha256=digest(ROOT/'pyproject.toml'),
        complete_binary_runtime_equivalence_verified=False)


def plan(source, inputs):
    return dict(protocol=PROTOCOL, source=source, inputs=inputs, device='cuda:0', seed=527,
        worlds=2, normal_policy_ticks=2, synthetic_isolation_policy_ticks=2,
        child_timeout_seconds=CHILD_SECONDS, service_timeout_seconds=SERVICE_SECONDS,
        closeout_seconds=CLOSEOUT_SECONDS, cutoff_unix=CUTOFF,
        optimizer_steps=0, checkpoint_files=[], physical_motion_authorized=False)


def output_path(source):
    supervisor.hex_id(source, 40)
    return ROOT/'artifacts/evaluations'/('stance-cuda-probe-'+source[:12])


def check_window(now=None):
    require((time.time() if now is None else now)+SERVICE_SECONDS+CLOSEOUT_SECONDS < CUTOFF,
            'insufficient independent service and closeout window')


def check_log(path):
    if not path.exists(): return
    raw = supervisor.file_bytes(path, limit=16*1024*1024, allow_empty=True)
    # This exact source-bound startup declaration is not a numeric observation.
    # Do not exempt prefixes, suffixes, other NaN-safe text, or actual warnings.
    lines = raw.decode('utf-8', errors='replace').splitlines()
    match = BAD_LOG.search('\n'.join(line for line in lines if line != STARTUP_INFO))
    require(match is None, 'numerical/backend warning in retained child log')


def serial(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if is_dataclass(value): return serial(asdict(value))
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


def frozen_row(env, row):
    from mjlab_microduck.stance_warp_integrator import STATE_FIELDS
    result = {n: env._view(n)[row].clone() for n in (*STATE_FIELDS, 'ctrl')}
    result.update(queue=env.delay.queue[row].clone(), target=env.delay.target[row].clone(),
        correction=env.delay.correction[row].clone(), history=env.motor.previous[row].clone(),
        voltage=env.motor.voltage[row].clone(), kp=env.motor.kp[row].clone())
    result.update({n: v[row].clone() for n, v in env.motor.fields.items()})
    result.update({n: v[row].clone() for n, v in env.observations().items()})
    return result


def cases(device):
    """Same short integration cases on CPU for tests; only child requests CUDA."""
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    torch.manual_seed(527)
    normal = WarpStanceRuntime(2, device=device)
    traces = []; rewards = []; started = time.monotonic()
    for i in range(2):
        result = normal.step(torch.full((2, 10), float(i), device=device))
        require(result['executed_steps'].tolist() == [10, 10] and result['live'].all().item(),
                'normal short integration must complete without failure')
        traces.extend(result['boundaries'][int(i > 0):])
        rewards.append(result['reward'])
    normal_result = dict(steps=normal.steps.tolist(), boundaries=serial(traces),
                        rewards=serial(rewards), steady_elapsed_s=time.monotonic()-started)
    backend = dict(torch_device=str(normal.device), warp_device=str(normal.wp_device),
                   warp_is_cuda=normal.wp_device.is_cuda,
                   torch_cuda_initialized=torch.cuda.is_initialized())
    del normal

    env = WarpStanceRuntime(2, device=device)
    integrate = env.integrator.integrate; count = 0
    def injected(live):
        nonlocal count
        integrate(live); count += 1
        if count == 1:
            # Explicit synthetic test, not a policy action, push lesson or score.
            import math
            env._view('qpos')[0, 3:7] = torch.tensor(
                [math.cos(.2), 0, math.sin(.2), 0], dtype=torch.float32, device=device)
    env.integrator.integrate = injected  # Only this owned object, not a shared library.
    first = env.step(torch.zeros(2, 10, device=device))
    require(first['executed_steps'].tolist() == [1, 10], 'first substep terminal isolation')
    terminal = first['terminal_records'][0]
    require(terminal is not None and terminal['physics_step'] == 1, 'first terminal retained')
    before = frozen_row(env, 0)
    second = env.step(torch.ones(2, 10, device=device))
    require(second['executed_steps'].tolist() == [0, 10] and second['reward'][0].item() == 0,
            'closed world cannot advance or earn reward')
    require(second['terminal_records'][0] == terminal, 'terminal record immutable')
    require(all(torch.equal(v, frozen_row(env, 0)[k]) for k, v in before.items()), 'closed fields exact')
    sibling = frozen_row(env, 1)
    retained = env.reset(torch.tensor([True, False], device=device))
    require(retained[0] == terminal and env.steps.tolist() == [0, 20], 'selective reset retention')
    require(all(torch.equal(v, frozen_row(env, 1)[k]) for k, v in sibling.items()), 'reset sibling fields exact')
    isolation = dict(synthetic_failure=True, first_steps=[1, 10], next_steps=[0, 10],
        before_reset_steps=second['episode_steps'].tolist(), after_reset_steps=env.steps.tolist(),
        first_terminal=terminal, first_boundaries=serial(first['boundaries']),
        next_boundaries=serial(second['boundaries']), frozen_fields_equal=True,
        reset_sibling_fields_equal=True)
    return dict(protocol=PROTOCOL, backend=backend, normal=normal_result, isolation=isolation,
        checks_passed=True, optimizer_steps=0, learned_stance=False, football_balance=False,
        complete_trajectory_evaluation=False, physical_motion_authorized=False)


def validate_payload(value, launch_sha):
    canonical(value)
    require(value['protocol'] == PROTOCOL and value['launch_sha256'] == launch_sha, 'source-bound probe payload')
    require(value['backend'] == dict(torch_device='cuda:0', warp_device='cuda:0',
                                    warp_is_cuda=True, torch_cuda_initialized=True), 'actual CUDA execution')
    require(value['checks_passed'] is True and value['optimizer_steps'] == 0
            and value['learned_stance'] is False and value['football_balance'] is False
            and value['complete_trajectory_evaluation'] is False
            and value['physical_motion_authorized'] is False, 'integration only, never capability admission')
    require(value['normal']['steps'] == [20, 20] and len(value['normal']['boundaries']) == 21,
            'normal continuous short prefix')
    require([b['physics_steps'] for b in value['normal']['boundaries']] == [[i, i] for i in range(21)],
            'continuous normal boundary counters')
    require(value['isolation']['before_reset_steps'] == [1, 20]
            and value['isolation']['after_reset_steps'] == [0, 20]
            and value['isolation']['frozen_fields_equal'] is True
            and value['isolation']['reset_sibling_fields_equal'] is True, 'isolation and selective reset checks')
    require(value['isolation']['synthetic_failure'] is True
            and value['isolation']['first_terminal']['physics_step'] == 1
            and value['isolation']['first_terminal']['terminated'] is True,
            'explicit first-step synthetic terminal')
    require([b['physics_steps'] for b in value['isolation']['first_boundaries']]
            == [[0, 0]]+[[1, i] for i in range(1, 11)], 'first terminal prefix counters')
    require([b['physics_steps'] for b in value['isolation']['next_boundaries']]
            == [[1, i] for i in range(10, 21)], 'frozen sibling prefix counters')


def prepare(source):
    check_window()
    require(not torch.cuda.is_initialized(), 'CPU-only preflight')
    inputs = identity(source)
    output = supervisor.native._plain_path(output_path(source))
    output.mkdir(exist_ok=False)
    supervisor.write_json(output/'launch.json', plan(source, inputs))
    return output


def child(source):
    output = output_path(source)
    launch_file = output/'launch.json'
    launch = supervisor.parse(supervisor.file_bytes(launch_file))
    require(launch == plan(source, identity(source)), 'unchanged child inputs')
    check_window()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available(), 'explicit available CUDA0')
    result = cases('cuda:0')
    require(identity(source) == launch['inputs'], 'child inputs unchanged after simulation')
    result['launch_sha256'] = digest(launch_file)
    validate_payload(result, result['launch_sha256'])
    supervisor.write_json(output/'probe.json', result)
    print('CUDA integration cases complete; no optimizer or capability admission', flush=True)


def supervise(source):
    output = supervisor.native._plain_path(output_path(source))
    launch_file = output/'launch.json'; log = output/'child.log'
    require(not any((output/n).exists() for n in ('child.log', 'probe.json', 'report.json')), 'one retained attempt, no overwrite')
    launch = supervisor.parse(supervisor.file_bytes(launch_file)); launch_sha = digest(launch_file)
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed', optimizer_steps=0)
    try:
        check_window()
        require(not torch.cuda.is_initialized(), 'supervisor must not own CUDA')
        require(launch == plan(source, identity(source)), 'unchanged launch inputs')
        with supervisor.gpu_lease() as fd:
            report['idle_before'] = wait_idle()
            def guard():
                require(time.time()+CLOSEOUT_SECONDS < CUTOFF, 'campaign closeout boundary')
                check_log(log)
                require(identity(source) == launch['inputs'], 'live source/runtime/asset identity changed')
            report['child'] = supervisor.supervised_process(
                [str(ROOT/'.venv/bin/python'), '-m', 'mjlab_microduck.stance_cuda_probe', 'child', '--source', source],
                log, cwd=ROOT, env=supervisor.child_environment(), lock_fd=fd,
                timeout=CHILD_SECONDS, guard=guard)
            check_log(log)
            result = supervisor.parse(supervisor.file_bytes(output/'probe.json'))
            validate_payload(result, launch_sha)
            require(identity(source) == launch['inputs'], 'supervisor inputs unchanged')
            report['idle_after'] = wait_idle()
            report['decision'] = 'cuda-integration-only-passed'
    except Exception as exc:
        report['error_type'] = type(exc).__name__; report['error'] = str(exc)
        raise
    finally:
        report['retained_files'] = {n: digest(output/n) for n in ('launch.json', 'child.log', 'probe.json') if (output/n).exists()}
        supervisor.write_json(output/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': print(prepare(args.source))
    elif args.mode == 'supervise': supervise(args.source)
    else: child(args.source)


if __name__ == '__main__': main()
