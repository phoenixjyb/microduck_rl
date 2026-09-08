"""Actual observation/motor managers, synthetic CPU robot and policy callbacks."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict
from mjlab.managers.observation_manager import ObservationManager

from mjlab_microduck import foundation_command_capture as capture
from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck.command_delivery import TERMS, DIMENSIONS
from mjlab_microduck.motor_step_stream import MotorStepStream, MotorStepCostCfg
from mjlab_microduck.speed_response_control import prepare_config as old_config


def sensor(env, name):
    return env.sensors[name]


class Actor(torch.nn.Module):
    def __init__(self, env):
        super().__init__(); self.register_buffer('normalizer_probe',torch.ones(1))
        self.env = env; self.calls = 0; self.mutate = False; self.nan = False
        self.eval()

    def forward(self, obs):
        self.calls += 1
        assert torch.equal(obs['actor'][:,48:51],self.env.command_manager.get_command('twist'))
        if self.mutate: self.normalizer_probe.add_(1)
        return torch.full((8,14),float('nan') if self.nan else 0.)


class Wrapped:
    def __init__(self, env, stream):
        self.unwrapped = env; self.stream = stream; self.steps = 0
        self.terminal_at = None; self.missing_capture = False; self.bad_done = False
        self.raise_step = False
        self.yaw_after_step = None

    def get_observations(self):
        return TensorDict(self.unwrapped.observation_manager.compute(),batch_size=[8])

    def step(self, actions):
        if self.raise_step: raise RuntimeError('synthetic physics failure')
        env = self.unwrapped; data = env.scene['robot'].data
        done = torch.zeros(8,dtype=torch.bool)
        if self.steps == self.terminal_at: done[2] = True
        if not self.missing_capture: self.stream.capture(data,done)
        self.steps += 1
        if self.yaw_after_step is not None:
            angle = torch.tensor(self.yaw_after_step/2)
            data.root_link_quat_w[:,0] = angle.cos()
            data.root_link_quat_w[:,3] = angle.sin()
        data.root_link_pos_w[:,0] += env.command_manager.get_command('twist')[:,0]*.02
        if done.any():
            # Simulate reset after the real MotorStepStream captures the sample.
            data.actuator_force.zero_()
            data.joint_vel.zero_()
        obs = env.observation_manager.compute(update_history=True)
        result = done.long()
        if self.bad_done: result[0] = 2
        return TensorDict(obs,batch_size=[8]),torch.zeros(8),result,{}


@pytest.fixture
def session():
    cfg,_ = old_config()
    group = deepcopy(cfg.observations['actor'])
    env = NS(num_envs=8,device='cpu',step_dt=.02)
    env.sensors = {name:torch.zeros((8,shape[0])) for name,shape in zip(TERMS,DIMENSIONS)}
    commands = dict(twist=torch.tensor([.3,0.,0.]).expand(8,3).clone(),
                    head_pose=torch.zeros(8,4),body_pose=torch.zeros(8,6))
    env.command_manager = NS(get_command=commands.__getitem__)
    for name in TERMS[:5]:
        group.terms[name].func = sensor; group.terms[name].params = {'name':name}
    env.observation_manager = ObservationManager({'actor':group},env)
    env.observation_manager.compute(update_history=True)  # Existing reset cache.
    env.event_manager = NS(get_term_cfg=cfg.events.__getitem__)
    velocity = torch.tensor([.1,0.,0.]).expand(8,3).clone()
    data = NS(root_link_lin_vel_w=velocity,root_link_lin_vel_b=velocity.clone(),
              root_link_quat_w=torch.tensor([1.,0.,0.,0.]).expand(8,4).clone(),
              root_link_pos_w=torch.zeros(8,3),actuator_force=torch.full((8,14),.12),
              joint_vel=torch.ones(8,14))
    env.scene = {'robot':NS(data=data)}
    stream = MotorStepStream(8,mapping.JOINTS,list(range(14)),device='cpu',cost_cfg=MotorStepCostCfg())
    env._microduck_motor_step_stream = stream
    return Wrapped(env,stream),Actor(env),stream


def run(session, **kwargs):
    return capture.capture(mapping.Cell(503,.1,'original'),*session,
                           budget_seconds=120,clock=kwargs.pop('clock',lambda:1.),**kwargs)


@pytest.mark.parametrize('speed',mapping.SPEEDS)
def test_config_changes_only_declared_speed_and_not_historical_defaults(speed):
    cell = mapping.Cell(503,speed,'original')
    cfg,agent = capture.prepare_config(cell); old,oa = old_config(seed=503)
    assert cfg.commands['twist'].ranges.lin_vel_x == (speed,speed)
    cfg.commands['twist'].ranges.lin_vel_x = (.3,.3)
    assert asdict(cfg) == asdict(old) and asdict(agent) == asdict(oa)


def test_full_capture_executes_fresh_input_and_preserves_actor_state(session):
    wrapped,actor,stream = session
    result = run(session)
    assert wrapped.steps == actor.calls == stream.next_step == 400
    assert result.trace.velocity.shape == (400,8,4)
    assert result.cached_commands[0,0,0].item() == pytest.approx(.3)
    assert result.trace.consumed[0,0,0].item() == pytest.approx(.1)
    assert result.evidence['actor_state_unchanged'] is True
    assert all(result.evidence[k] is False for k in (
        'checkpoint_loaded_verified','runtime_verified','physics_execution_verified',
        'policy_acceptance','training_admitted','physical_motion_authorized'))
    assert mapping.score(mapping.Cell(503,.1,'original'),result.trace)['classification'] == 'descriptive-cell-within-checks'
    assert actor.normalizer_probe.item() == 1.


def test_first_terminal_stops_without_reset_continuation_and_retains_pre_reset_motor(session):
    wrapped,actor,stream = session; wrapped.terminal_at = 2
    result = run(session)
    assert wrapped.steps == actor.calls == 3
    assert result.trace.dones[-1,2] and result.trace.legacy_force[-1].count_nonzero() == 0
    assert torch.all(result.trace.pre_force[-1] == .12)
    assert result.trace.position.shape[0] == 3


def test_changing_heading_uses_fresh_command_without_extra_sensor_computation(session,monkeypatch):
    wrapped,_,_ = session; wrapped.terminal_at = 2; wrapped.yaw_after_step = .1
    manager = wrapped.unwrapped.observation_manager
    original = manager.compute; draws = []
    def counted(*args,**kwargs):
        draws.append(kwargs.get('update_history',False))
        return original(*args,**kwargs)
    monkeypatch.setattr(manager,'compute',counted)
    result = run(session)
    assert draws == [False,True,True,True]
    torch.testing.assert_close(result.trace.consumed[:,0,2],torch.tensor([0.,-.02,-.04]))
    torch.testing.assert_close(result.cached_commands[:,0,2],torch.tensor([0.,0.,-.02]))


def test_actor_digest_includes_normalizer_buffers_and_rejects_submodule_train_mode(session):
    actor = session[1]; initial = capture.actor_digest(actor)
    actor.normalizer_probe.add_(.1)
    assert capture.actor_digest(actor) != initial
    actor.add_module('child',torch.nn.Linear(1,1))
    with pytest.raises(ValueError,match='evaluation mode'): capture.actor_digest(actor)


@pytest.mark.parametrize('mode',['nan','mutate','missing_capture','bad_done','raise_step','train'])
def test_failure_keeps_partial_evidence_and_never_returns_success(session,mode):
    wrapped,actor,_ = session
    if mode in ('nan','mutate'): setattr(actor,mode,True)
    elif mode == 'train': actor.train()
    else: setattr(wrapped,mode,True)
    with pytest.raises(capture.CaptureFailure) as error: run(session)
    e = error.value
    if mode == 'train': assert e.frames == [] and e.inference_steps == e.simulation_steps == 0
    elif mode == 'nan': assert len(e.frames) == e.inference_steps == 1 and e.simulation_steps == 0
    elif mode == 'mutate': assert len(e.frames) == e.simulation_steps == 400
    elif mode == 'raise_step': assert e.inference_steps == 1 and e.simulation_steps == 0
    else: assert e.simulation_steps == 1


def test_budget_and_clock_regression_stop_before_any_actor_call(session):
    for times in ([0.,120.],[2.,1.]):
        clock = iter(times).__next__
        with pytest.raises(capture.CaptureFailure) as error: run(session,clock=clock)
        assert error.value.frames == [] and session[1].calls == 0


def test_cooperative_timeout_after_step_preserves_completed_frame(session):
    times = iter([0.,0.,0.,0.,0.,121.])
    with pytest.raises(capture.CaptureFailure) as error: run(session,clock=times.__next__)
    e = error.value
    assert e.inference_steps == e.simulation_steps == 1
    assert 'pre_force' in e.frames[0]


def test_missing_stream_or_wrong_live_domain_refuses_capture(session):
    wrapped,_,_ = session
    wrapped.unwrapped.event_manager.get_term_cfg('randomize_com').params['ranges'] = (-.015,.015)
    with pytest.raises(capture.CaptureFailure,match='CoM'): run(session)
    assert wrapped.steps == 0


def test_byte_binding_requires_exact_checkpoint_and_runtime_pins(tmp_path,monkeypatch):
    model = tmp_path/'model.pt'; model.write_bytes(b'fixture-model-not-real-checkpoint')
    runtime = tmp_path/'runtime.py'; runtime.write_bytes(b'fixture-runtime')
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setitem(mapping.CHECKPOINTS,'original',sha(model))
    pins = {'runtime':(runtime,sha(runtime))}
    cell = mapping.Cell(503,.1,'original')
    result = capture.bind_files(cell,model,pins)
    assert result['declared_file_bytes_verified'] is True
    assert result['checkpoint_loaded'] is result['runtime_pin_coverage_verified'] is False
    runtime.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'): capture.bind_files(cell,model,pins)
    with pytest.raises(ValueError): capture.bind_files(cell,model,{})
    link = tmp_path/'link.pt'; link.symlink_to(model)
    with pytest.raises(ValueError,match='symlink'): capture.bind_files(cell,link,pins)
