from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from contextlib import contextmanager
import os
import io
from hashlib import sha256

import pytest
import torch
import warp as wp

from mjlab_microduck import stance_throughput_probe as probe
from mjlab_microduck import stance_forward_graph as graph
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


def test_real_cpu_bindings_allow_contents_but_reject_pointer_and_option_drift():
    env = WarpStanceRuntime(2, device='cpu')
    before = graph.binding((env.model, env.data))
    env.step(torch.zeros(2, 10))
    assert graph.binding((env.model, env.data)) == before
    env._view('ctrl')[0, 0] = .01
    assert graph.binding((env.model, env.data)) == before
    env.data.ctrl = wp.clone(env.data.ctrl)
    assert graph.binding((env.model, env.data)) != before
    before = graph.binding((env.model, env.data))
    env.model.opt.iterations += 1
    assert graph.binding((env.model, env.data)) != before


def test_cpu_graph_opt_in_refuses_without_false_cuda_claim():
    env = WarpStanceRuntime(2, device='cpu')
    assert env.forward_graph is None
    with pytest.raises(ValueError, match='requires CUDA'): env.enable_forward_graph()
    assert env.faulted and env.forward_graph is None


def test_graph_replay_checks_bound_array_before_launch(monkeypatch):
    @dataclass
    class Data:
        a: object
        iterations: int
    g = graph.ForwardGraph.__new__(graph.ForwardGraph)
    g.model = Data(wp.zeros(2, device='cpu'), 100); g.data = Data(wp.zeros(2, device='cpu'), 2)
    g.device = wp.get_device('cpu'); g.graph = object(); g.signature = graph.binding((g.model,g.data))
    calls = []; monkeypatch.setattr(graph.wp,'capture_launch',lambda obj: calls.append(obj))
    g.run(g.model,g.data); assert calls == [g.graph]
    with pytest.raises(ValueError,match='replaced'):g.run(g.model,Data(g.data.a,2))
    g.data.a = wp.clone(g.data.a)
    with pytest.raises(ValueError, match='replaced'): g.run(g.model,g.data)
    assert len(calls) == 1


def test_cpu_dispatch_seam_exact_values_and_synthetic_terminal(monkeypatch):
    first = probe.case(2,False,'cpu',ticks=2)
    # Dispatch seam only: this stand-in uses eager CPU forward, NOT a CUDA graph.
    def fake_enable(env):
        env.forward_graph = SimpleNamespace(run=lambda m,d: graph.mjwarp.forward(m,d))
    monkeypatch.setattr(WarpStanceRuntime,'enable_forward_graph',fake_enable)
    second = probe.case(2,True,'cpu',ticks=2)
    assert first['ticks'] == second['ticks']
    assert first['ticks'][0]['executed_steps'] == [1,10]
    assert first['ticks'][1]['executed_steps'] == [0,10]
    assert not torch.cuda.is_initialized()


def test_nonfinite_solve_still_faults_through_graph_dispatch(monkeypatch):
    env = WarpStanceRuntime(2,device='cpu')
    def bad_forward(model,data):
        graph.mjwarp.forward(model,data)
        env._view('qacc')[0,0] = float('nan')
    env.forward_graph = SimpleNamespace(run=bad_forward)
    monkeypatch.setattr(env.integrator,'integrate',lambda _: pytest.fail('invalid solve integrated'))
    with pytest.raises(ValueError,match='nonfinite solved'): env.step(torch.zeros(2,10))
    assert env.faulted


def fixture_cases():
    return [dict(worlds=n,graph=g,graph_bound=g,device='cuda:0',warp_device='cuda:0',
        synthetic_terminal=True,optimizer_steps=0,training_admitted=False,physical_motion_authorized=False,
        ticks=[dict(tick=i,state_sha256='a'*64,reset_sha256='b'*64,executed_steps=[10]*n) for i in range(24)],
        timing=[dict(step_s=.1 if g else .2,reset_s=.01) for _ in range(24)],profile=[])
        for n,g in probe.CASES]


def test_timing_does_not_admit_training_even_if_exact_and_faster():
    result=probe.decide(fixture_cases())
    assert result['decision']=='exact-short-probe-only'
    assert result['comparisons'][0]['speedup']==pytest.approx(.21/.11)
    assert not result['training_admitted'] and not result['full_pilot_timing_established']


@pytest.mark.parametrize('damage',['hash','missing','device','order','negative','nan','promotion'])
def test_matrix_refuses_incomplete_or_changed_claims(damage):
    cases=fixture_cases()
    if damage=='hash':
        cases[2]['ticks'][0]['state_sha256']='c'*64
        assert probe.decide(cases)['decision']=='differential-rejected'; return
    if damage=='missing':cases.pop()
    if damage=='device':cases[1]['device']='cpu'
    if damage=='order':cases.reverse()
    if damage=='negative':cases[0]['timing'][0]['step_s']=-1
    if damage=='nan':cases[0]['timing'][0]['reset_s']=float('nan')
    if damage=='promotion':cases[0]['training_admitted']=True
    with pytest.raises(ValueError):probe.decide(cases)


def test_eager_repeat_mismatch_is_not_attributed_to_graph():
    cases=fixture_cases();cases[1]['ticks'][0]['state_sha256']='c'*64
    result=probe.decide(cases)
    assert result['decision']=='differential-rejected'
    assert not result['comparisons'][0]['baseline_repeat_match']


@pytest.mark.parametrize('bad',[False,True])
def test_probe_supervisor_canonical_command_and_failed_evidence(tmp_path,monkeypatch,bad):
    monkeypatch.setattr(probe.host,'ROOT',tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(probe.host,'check_window',lambda:None)
    monkeypatch.setattr(probe.time,'time',lambda:probe.host.CUTOFF-10000)
    monkeypatch.setattr(probe.host,'identity',lambda _:dict(synthetic=True))
    monkeypatch.setattr(probe,'checked',lambda *a:dict(inputs=dict(synthetic=True)))
    monkeypatch.setattr(probe.host,'wait_idle',lambda:dict(synthetic=True))
    states=dict(MainPID=str(os.getpid()),RuntimeMaxUSec='3min',KillMode='control-group',ActiveState='active')
    monkeypatch.setattr(probe.host,'read',lambda *a:states[a[-2]])
    root=probe.output_path('a'*40);root.mkdir(parents=True)
    probe.host.supervisor.write_json(root/'launch.json',dict(synthetic=True))
    @contextmanager
    def lease():yield 4
    monkeypatch.setattr(probe.host.supervisor,'gpu_lease',lease)
    def process(command,log,**kw):
        assert command[1:4]==['-m','mjlab_microduck.stance_throughput_probe','child']
        assert kw['timeout']==120 and kw['lock_fd']==4
        log.write_text('WARNING: synthetic fault' if bad else 'synthetic normal')
        kw['guard']()
        cases=fixture_cases()
        for i,c in enumerate(cases):
            c['diagnostic_prefix']=[]
            for tick in range(4):
                data={'synthetic':torch.zeros(2)};out=io.BytesIO();torch.save(data,out);raw=out.getvalue()
                name=f'case-{i}-tick-{tick}.pt';(root/name).write_bytes(raw)
                c['diagnostic_prefix'].append(dict(file=name,sha256=sha256(raw).hexdigest()))
                c['ticks'][tick]['state_sha256']=probe.tree_hash(data)
            probe.host.supervisor.write_json(root/f'case-{i}.json',c)
        probe.host.supervisor.write_json(root/'decision.json',probe.decide(cases))
        return dict(synthetic=True)
    monkeypatch.setattr(probe.host.supervisor,'supervised_process',process)
    if bad:
        with pytest.raises(ValueError):probe.supervise('a'*40,'b'*64)
    else:probe.supervise('a'*40,'b'*64)
    report=probe.host.supervisor.parse((root/'report.json').read_bytes())
    assert report['decision']==('failed' if bad else 'exact-short-probe-only')
    with pytest.raises(ValueError,match='one fresh'):probe.supervise('a'*40,'b'*64)
