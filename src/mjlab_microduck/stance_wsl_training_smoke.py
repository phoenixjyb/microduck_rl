"""Disposable 16-update WSL training bring-up; never a replication/pilot parent.

Uses the existing smoke learner, purpose, seed, budget and receipt verifier.
The long-run timing rejection is preserved, not converted into admission.
"""
import argparse
from copy import deepcopy
import os
import random
import time
import numpy as np
import torch

from mjlab_microduck import stance_wsl_qualification as q
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.stance_ppo import CONFIG, STEPS
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files, execution = q.host, q.files, q.execution
MODULE = 'mjlab_microduck.stance_wsl_training_smoke'
PROTOCOL = 'football-b1n-wsl-disposable-training-v1'


def output_path(source):
    files.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-wsl-training-smoke-'+source[:12])


def service_name(source):
    return 'microduck-wsl-training-smoke-'+source[:12]+'.service'


def plan(source, deadline, runtime_sha):
    require(execution.PROFILE['name'] == execution.WSL, 'fixed authorized WSL profile')
    files.hex_id(runtime_sha, 64)
    return dict(protocol=PROTOCOL, source=source, inputs=host.identity(source),
        runtime_sha256=runtime_sha, deadline_unix=deadline,
        purpose='smoke', worlds=smoke.WORLDS, seed=smoke.SEED, updates=smoke.UPDATES,
        steps_per_update=STEPS, optimizer=deepcopy(CONFIG), child_seconds=900, service_seconds=960,
        timing_basis=q.validate_timing_basis(source), initialization='fresh-not-parent-weights',
        learner_device='cpu', physics_device='cuda:0', forward_graph=False,
        long_replication_authorized=False, pilot_parent_authorized=False,
        learned_stance=False, physical_motion_authorized=False)


def prepare(source, deadline):
    q.throughput.check_window(deadline, launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only preparation')
    runtime = plant.runtime_bytes(source, plant.build_entity().compile())
    launch = plan(source, deadline, files.digest(runtime))
    # Frozen-parent collection is a conservative sizing reference, not a claim
    # that random initial policy trajectories have identical runtime or behavior.
    basis = launch['timing_basis']['measured_caps']
    prediction = smoke.UPDATES*basis['per_update_worst_seconds']+basis['setup_seconds']
    require(prediction*1.25+60 < 960, 'short smoke fits existing fixed budget')
    root = output_path(source); root.mkdir(exist_ok=False)
    smoke.write_bytes(root/'runtime.json', runtime)
    files.write_json(root/'launch.json', launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent smoke launch hash')
    launch = files.parse(files.file_bytes(root/'launch.json'))
    require(launch == plan(source, launch['deadline_unix'],host.digest(root/'runtime.json')),
            'unchanged WSL smoke plan/source/runtime')
    plant.checked_runtime(files.parse(files.file_bytes(root/'runtime.json')), source)
    return launch


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd)
    launch = checked(source, launch_sha)
    q.throughput.check_window(launch['deadline_unix'], launching=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'explicit CUDA child')
    host.wait_idle()
    random.seed(smoke.SEED); np.random.seed(smoke.SEED); torch.manual_seed(smoke.SEED)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(smoke.WORLDS, device='cuda:0')
    root = output_path(source)
    runtime = files.parse(files.file_bytes(root/'runtime.json'))
    require(env.forward_graph is None and env.wp_device.is_cuda
            and plant.describe(env.native) == runtime['plant'], 'actual unchanged eager CUDA plant')
    exports = smoke.run_updates(smoke.SmokeStanceLearner(), smoke.PhysicsBridge(env), root,
        source, launch_sha, launch['runtime_sha256'], deadline=time.monotonic()+870)
    require(checked(source,launch_sha) == launch, 'unchanged completed smoke inputs')
    files.write_json(root/'completed.json', dict(protocol=smoke.PROTOCOL, launch_sha256=launch_sha,
        completed_updates=smoke.UPDATES, checkpoints=exports, physics_device='cuda:0',
        learner_device='cpu', seed=smoke.SEED, worlds=smoke.WORLDS,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False))


def verify(source):
    root = output_path(source)
    report = files.parse(files.file_bytes(root/'report.json'))
    launch = checked(source, report['launch_sha256'])
    require(report['protocol'] == PROTOCOL and report['decision'] == 'training-smoke-complete-not-capability',
            'completed disposable training, not capability')
    require(set(report['files']) == {p.name for p in root.iterdir()}-{'report.json'}, 'exact smoke inventory')
    require(all(host.digest(root/name) == digest for name,digest in report['files'].items()), 'smoke file hashes')
    result = smoke.verify_completed(root,source,report['launch_sha256'],launch)
    return dict(completed_updates=result['completed_updates'], checkpoint_count=len(result['checkpoints']),
        report_sha256=host.digest(root/'report.json'), long_replication_authorized=False,
        pilot_parent_authorized=False, learned_stance=False, physical_motion_authorized=False)


def supervise(source, launch_sha):
    launch=checked(source,launch_sha); root=output_path(source)
    q.throughput.check_window(launch['deadline_unix'],launching=True)
    require(q.throughput.service_state(service_name(source)) ==
            dict(MainPID=str(os.getpid()),RuntimeMaxUSec='16min',KillMode='control-group',ActiveState='active'),
            'independently timed smoke service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only supervisor')
    require({p.name for p in root.iterdir()} == {'launch.json','runtime.json'}, 'one fresh smoke')
    report=dict(protocol=PROTOCOL,launch_sha256=launch_sha,decision='failed')
    try:
        with files.gpu_lease() as fd:
            report['idle_before']=host.wait_idle()
            def guard():
                require(time.time()+600 < launch['deadline_unix'],'smoke closeout reserve')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'],'live source/host drift')
            report['child']=files.supervised_stance_smoke(
                [str(host.ROOT/'.venv/bin/python'),'-m',MODULE,'child','--source',source,
                 '--launch-sha256',launch_sha,'--lock-fd',str(fd)],root/'child.log',cwd=host.ROOT,
                env=files.child_environment(),lock_fd=fd,guard=guard)
            host.check_log(root/'child.log')
            smoke.verify_completed(root,source,launch_sha,launch)
            report['idle_after']=host.wait_idle()
            report['decision']='training-smoke-complete-not-capability'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc),error_notes=getattr(exc,'__notes__',[]));raise
    finally:
        report['files']={p.name:host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        files.write_json(root/'report.json',report)
    verify(source)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','supervise','child','verify'));p.add_argument('--source',required=True)
    p.add_argument('--deadline-unix',type=int);p.add_argument('--launch-sha256');p.add_argument('--lock-fd',type=int)
    a=p.parse_args()
    if a.mode=='prepare':print(canonical(prepare(a.source,a.deadline_unix)))
    elif a.mode=='supervise':supervise(a.source,a.launch_sha256)
    elif a.mode=='child':child(a.source,a.launch_sha256,a.lock_fd)
    else:print(canonical(verify(a.source)))


if __name__=='__main__':main()
