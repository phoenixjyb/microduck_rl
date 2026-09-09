"""Source-bound forward-graph differential/timing probe; never an optimizer."""
import argparse
import cProfile
import io
from hashlib import sha256
import os
import pstats
import random
import time

import numpy as np
import torch

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

MODULE = 'mjlab_microduck.stance_throughput_probe'
PROTOCOL = 'football-b1n-forward-graph-differential-v1'
CASES = ((64, False), (64, False), (64, True), (512, False), (512, False), (512, True))
TICKS = 24


def tree_hash(value):
    """Value identity, independent of object addresses and serialization layout."""
    h = sha256()
    def visit(v, path):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(canonical([path, str(t.dtype), list(t.shape)]).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, dict):
            h.update(canonical([path, 'dict', sorted(v)]).encode())
            for k in sorted(v): visit(v[k], path+[k])
        elif isinstance(v, (list, tuple)):
            h.update(canonical([path, type(v).__name__, len(v)]).encode())
            for i, item in enumerate(v): visit(item, path+[i])
        else: h.update(canonical([path, v]).encode())
    visit(trace.owned(value), [])
    return h.hexdigest()


def case(worlds, graph, device, *, ticks=TICKS, retain=None):
    """Real CPU reference test or CUDA candidate; deterministic, no learner."""
    random.seed(523); np.random.seed(523); torch.manual_seed(523)
    env = WarpStanceRuntime(worlds, device=device)
    if graph: env.enable_forward_graph()
    bridge = smoke.PhysicsBridge(env)
    rng = torch.Generator(device='cpu').manual_seed(523)
    # Two warmup ticks and a full nominal reset, identical for both arms.
    for _ in range(2): bridge.step(torch.zeros(worlds, 10))
    bridge.reset(torch.ones(worlds, dtype=torch.bool))
    integrate = env.integrator.integrate; count = 0
    def injected(rows):
        nonlocal count
        integrate(rows); count += 1
        if count == 1:
            # Explicit synthetic integration check, not a natural policy fall.
            env._view('qpos')[0, 3:7] = torch.tensor(
                [np.cos(.2), 0, np.sin(.2), 0], dtype=torch.float32, device=device)
    env.integrator.integrate = injected
    evidence = []; timing = []; prefix = []
    for i in range(ticks):
        action = .3*torch.randn(worlds, 10, generator=rng)
        started = time.monotonic(); result = bridge.step(action)
        elapsed = time.monotonic()-started
        # Keep the closed row frozen for the next tick, then selectively reset.
        require(i != 0 or result['executed_steps'][0] == 1, 'injected first-step terminal')
        require(i != 1 or result['executed_steps'][0] == 0, 'closed row must not advance')
        data = trace.owned(dict(result=result, control=env._control_snapshot(),
            integration={k: env._view(k) for k in ('qpos', 'qvel', 'time', 'qacc_warmstart')}))
        identity = tree_hash(data)
        if retain is not None and i < 4: prefix.append(retain(i, data))
        reset_started = time.monotonic()
        retained = bridge.reset(~bridge.live) if i >= 1 and not bridge.live.all() else None
        reset_s = time.monotonic()-reset_started
        evidence.append(dict(tick=i, state_sha256=identity, reset_sha256=tree_hash(retained),
                             executed_steps=result['executed_steps'].tolist()))
        timing.append(dict(step_s=elapsed, reset_s=reset_s))
    # Profile one extra tick separately: profiler overhead is excluded above.
    profile = cProfile.Profile(); profile.enable()
    bridge.step(torch.zeros(worlds, 10))
    profile.disable()
    stats = pstats.Stats(profile)
    rows = sorted(((key, v) for key, v in stats.stats.items()), key=lambda item: item[1][3], reverse=True)[:25]
    top = [dict(file=key[0], line=key[1], function=key[2], calls=v[1], self_s=v[2], cumulative_s=v[3]) for key, v in rows]
    return dict(worlds=worlds, graph=graph, device=str(env.device), warp_device=str(env.wp_device),
        graph_bound=env.forward_graph is not None, ticks=evidence, timing=timing, profile=top, diagnostic_prefix=prefix,
        synthetic_terminal=True, optimizer_steps=0, training_admitted=False, physical_motion_authorized=False)


def decide(cases):
    require(len(cases) == len(CASES), 'six declared throughput cases')
    for result, (worlds, graph) in zip(cases, CASES):
        require(result['worlds'] == worlds and result['graph'] is graph and result['graph_bound'] is graph
            and result['device'] == result['warp_device'] == 'cuda:0'
            and result['synthetic_terminal'] is True and result['optimizer_steps'] == 0
            and result['training_admitted'] is result['physical_motion_authorized'] is False, 'actual probe scope')
        require(len(result['ticks']) == len(result['timing']) == TICKS, 'complete timing prefix')
        for i, (tick, timing) in enumerate(zip(result['ticks'], result['timing'])):
            require(tick['tick'] == i and len(tick['executed_steps']) == worlds, 'ordered probe ticks')
            for key in ('state_sha256', 'reset_sha256'): host.supervisor.hex_id(tick[key], 64)
            require(timing['step_s'] > 0 and timing['reset_s'] >= 0, 'positive finite timing')
        canonical(result)
    comparisons = []
    for baseline, repeat, candidate in ((cases[0], cases[1], cases[2]), (cases[3], cases[4], cases[5])):
        repeated = baseline['ticks'] == repeat['ticks']
        equal = repeated and baseline['ticks'] == candidate['ticks']
        b = sum(t['step_s']+t['reset_s'] for r in (baseline, repeat) for t in r['timing'])/2
        c = sum(t['step_s']+t['reset_s'] for t in candidate['timing'])
        comparisons.append(dict(worlds=baseline['worlds'], baseline_repeat_match=repeated, exact_trajectory_match=equal,
            baseline_seconds=b, candidate_seconds=c, speedup=b/c,
            candidate_512_update_collection_seconds=c*512))
    return dict(protocol=PROTOCOL, comparisons=comparisons,
        decision='exact-short-probe-only' if all(c['exact_trajectory_match'] for c in comparisons) else 'differential-rejected',
        training_admitted=False, full_pilot_timing_established=False, physical_motion_authorized=False)


def output_path(source):
    host.supervisor.hex_id(source, 40)
    return host.ROOT/'artifacts/evaluations'/('stance-throughput-'+source[:12])


def plan(source):
    return dict(protocol=PROTOCOL, source=source, inputs=host.identity(source), plant=plant.reference(),
        cases=[list(c) for c in CASES], ticks=TICKS, warmup_ticks=2, profile_ticks=1, seed=523,
        child_seconds=120, service_seconds=180, cutoff_unix=host.CUTOFF,
        synthetic_terminal=True, optimizer_steps=0, training_admitted=False, physical_motion_authorized=False)


def prepare(source):
    host.check_window(); require(not torch.cuda.is_initialized(), 'CPU preparation')
    value = plan(source); root = host.supervisor.native._plain_path(output_path(source)); root.mkdir(exist_ok=False)
    host.supervisor.write_json(root/'launch.json', value)
    return dict(output=str(root), launch_sha256=host.digest(root/'launch.json'))


def checked(source, launch_sha):
    root = output_path(source)
    require(host.digest(root/'launch.json') == launch_sha, 'independent probe launch hash')
    value = host.supervisor.parse(host.supervisor.file_bytes(root/'launch.json'))
    require(value == plan(source), 'exact probe source/runtime/plant plan')
    return value


def child(source, launch_sha, fd):
    smoke.inherited_lease(fd); host.check_window(); checked(source, launch_sha)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'CUDA child only')
    host.wait_idle(); root = output_path(source); results = []
    for i, (worlds, graph) in enumerate(CASES):
        def retain(tick, data):
            out = io.BytesIO(); torch.save(data, out); raw = out.getvalue()
            name = f'case-{i}-tick-{tick}.pt'; smoke.write_bytes(root/name, raw)
            return dict(file=name, sha256=sha256(raw).hexdigest())
        result = case(worlds, graph, 'cuda:0', retain=retain); results.append(result)
        host.supervisor.write_json(root/f'case-{i}.json', result)
        print(f'Completed throughput case {i}: worlds={worlds}, graph={graph}', flush=True)
    checked(source, launch_sha)
    host.supervisor.write_json(root/'decision.json', decide(results))


def supervise(source, launch_sha):
    host.check_window(); launch = checked(source, launch_sha); root = output_path(source)
    unit = 'microduck-stance-throughput-'+source[:12]+'.service'
    actual = {k: host.read('systemctl', '--user', 'show', unit, '-p', k, '--value')
              for k in ('MainPID', 'RuntimeMaxUSec', 'KillMode', 'ActiveState')}
    require(actual == dict(MainPID=str(os.getpid()), RuntimeMaxUSec='3min', KillMode='control-group', ActiveState='active'),
            'independently timed probe service')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only supervisor')
    require({p.name for p in root.iterdir()} == {'launch.json'}, 'one fresh throughput attempt')
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision='failed', training_admitted=False)
    try:
        with host.supervisor.gpu_lease() as fd:
            report['idle_before'] = host.wait_idle()
            def guard():
                require(time.time()+600 < host.CUTOFF, 'closeout boundary')
                host.check_log(root/'child.log')
                require(host.identity(source) == launch['inputs'], 'live probe source drift')
            report['child'] = host.supervisor.supervised_process(
                [str(host.ROOT/'.venv/bin/python'), '-m', MODULE, 'child', '--source', source,
                 '--launch-sha256', launch_sha, '--lock-fd', str(fd)], root/'child.log', cwd=host.ROOT,
                env=host.supervisor.child_environment(), lock_fd=fd, timeout=120, guard=guard)
            host.check_log(root/'child.log')
            cases = [host.supervisor.parse(host.supervisor.file_bytes(root/f'case-{i}.json')) for i in range(len(CASES))]
            for i, result in enumerate(cases):
                require(len(result['diagnostic_prefix']) == 4, 'four retained diagnostic ticks')
                for tick, saved in enumerate(result['diagnostic_prefix']):
                    name = f'case-{i}-tick-{tick}.pt'
                    raw = host.supervisor.file_bytes(root/name, limit=16*1024*1024)
                    require(saved == dict(file=name, sha256=sha256(raw).hexdigest()), 'diagnostic prefix hash')
                    data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
                    require(tree_hash(data) == result['ticks'][tick]['state_sha256'], 'diagnostic prefix values')
            decision = decide(cases)
            require(decision == host.supervisor.parse(host.supervisor.file_bytes(root/'decision.json')), 'recomputed probe decision')
            report['result'] = decision; report['decision'] = decision['decision']
            report['idle_after'] = host.wait_idle()
    except Exception as e:
        report.update(error=str(e), error_notes=getattr(e, '__notes__', [])); raise
    finally:
        report['files'] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        host.supervisor.write_json(root/'report.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'supervise', 'child'))
    p.add_argument('--source', required=True); p.add_argument('--launch-sha256'); p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    if a.mode == 'prepare': print(canonical(prepare(a.source)))
    elif a.mode == 'supervise': supervise(a.source, a.launch_sha256)
    else: child(a.source, a.launch_sha256, a.lock_fd)


if __name__ == '__main__': main()
