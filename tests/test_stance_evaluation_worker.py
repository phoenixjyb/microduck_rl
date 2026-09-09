"""Short real CPU collection/retention; no GPU matrix or trained weights."""
from copy import deepcopy
from hashlib import sha256
import math
import time
import pytest
import torch

from mjlab_microduck import stance_evaluation_worker as worker
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime


def inputs(env):
    actor, critic = cp.fresh_models(521)
    meta = dict(protocol=cp.PROTOCOL, source='a'*40, runtime_sha256='b'*64,
        training_launch_sha256='c'*64, purpose='pilot', training_seed=521, worlds=512,
        iteration=128, initial_state_sha256=cp.state_hash(cp.states_of(actor, critic)), architecture=deepcopy(cp.ARCHITECTURE))
    raw = cp.encode(actor, critic, meta)  # Explicit untrained synthetic identity.
    runtime = plant.runtime_bytes('d'*40, env.native)
    binding = dict(protocol=trace.PROTOCOL, source='d'*40, runtime_sha256=sha256(runtime).hexdigest(),
        checkpoint_sha256=sha256(raw).hexdigest(), checkpoint_iteration=128,
        evaluation_seed=541, worlds=2, capture_device='cpu')
    launch = bundle.launch_bytes(binding, meta); binding['launch_sha256'] = sha256(launch).hexdigest()
    return dict(binding=binding, checkpoint_raw=raw, checkpoint_identity=meta, runtime_raw=runtime, launch_raw=launch)


def test_short_case_retains_control_physics_and_independent_replay(tmp_path):
    env = WarpStanceRuntime(2, device='cpu'); args = inputs(env)
    result = worker.evaluate_owned_case(tmp_path/'case', env, **args, deadline_monotonic=time.monotonic()+60, policy_tick_limit=1)
    assert result['collection']['policy_ticks'] == 1 and result['score']['complete_attempts'] == 0
    assert not result['score']['checkpoint_admitted'] and not torch.cuda.is_initialized()
    assert bundle.verify_bundle(tmp_path/'case', result['manifest_sha256'], binding=args['binding'],
        checkpoint_identity=args['checkpoint_identity']) == result['score']
    before = env.steps.clone()
    with pytest.raises(FileExistsError):
        worker.evaluate_owned_case(tmp_path/'case', env, **args, deadline_monotonic=time.monotonic()+60)
    assert torch.equal(before, env.steps)


def test_first_failures_complete_without_reset(monkeypatch):
    env = WarpStanceRuntime(2, device='cpu'); args = inputs(env)
    actor, _ = cp.load_evaluation(args['checkpoint_raw'], args['binding']['checkpoint_sha256'], args['checkpoint_identity'])
    integrate = env.integrator.integrate
    def tipped(rows):
        integrate(rows); env._view('qpos')[:, 3:7] = torch.tensor([math.cos(.2), 0, math.sin(.2), 0])
    monkeypatch.setattr(env.integrator, 'integrate', tipped)
    monkeypatch.setattr(env, 'reset', lambda *_: pytest.fail('evaluation reset'))
    result = worker.collect(env, actor, args['binding'], deadline_monotonic=time.monotonic()+60)
    score = trace.replay(result['payload'], args['binding'])
    assert result['collection']['stop_reason'] == 'all-first-attempts-complete'
    assert env.steps.tolist() == [1, 1] and score['complete_attempts'] == 2 and score['numerical_passes'] == 0


def test_wall_budget_keeps_partial_prefix_without_another_step():
    env = WarpStanceRuntime(2, device='cpu'); args = inputs(env)
    actor, _ = cp.load_evaluation(args['checkpoint_raw'], args['binding']['checkpoint_sha256'], args['checkpoint_identity'])
    times = iter([0., 0., 2.])
    result = worker.collect(env, actor, args['binding'], deadline_monotonic=1., clock=lambda: next(times))
    assert result['collection']['stop_reason'] == 'wall-budget-exhausted' and env.steps.tolist() == [10, 10]
    assert trace.replay(result['payload'], args['binding'])['complete_attempts'] == 0


def test_bad_reset_refused_before_stepping(tmp_path, monkeypatch):
    env = WarpStanceRuntime(2, device='cpu'); args = inputs(env)
    env._view('qpos')[:, 0] += .01
    monkeypatch.setattr(env, 'step', lambda *_a, **_kw: pytest.fail('bad reset stepped'))
    with pytest.raises(ValueError, match='nominal first reset'):
        worker.evaluate_owned_case(tmp_path/'case', env, **args, deadline_monotonic=time.monotonic()+60)
    assert not (tmp_path/'case').exists()
