"""Short CPU fixtures and deliberate corruption; no learned-skill admission."""
from copy import deepcopy
import math
import pytest
import torch

from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
from test_stance_attempt_trace import binding


@pytest.fixture(scope='module')
def sample():
    env = WarpStanceRuntime(2, device='cpu')
    spec = plant.describe(env.native)
    recorder = trace.FirstAttemptTrace(binding(), env.snapshot())
    obs = env.observations()['actor']; actions = torch.zeros(2, 10)
    recorder.append(env.step(actions), obs, actions)
    return recorder.payload(), spec


def test_fresh_compilation_nominal_reset_and_short_trace(sample):
    payload, spec = sample
    assert plant.checked_runtime(dict(source='a'*40, plant=spec), 'a'*40) == spec
    receipt = plant.check_trace(payload, spec)
    assert receipt['nominal_reset_checked'] and receipt['terminal_contact_records_checked'] == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field', ['qids', 'dofs', 'ranges', 'initial_qpos', 'floor',
    'assets', 'selected_fields_sha256', 'options', 'topology'])
def test_rehashed_runtime_cannot_invent_compiled_plant(sample, field):
    _, spec = sample
    changed = deepcopy(spec); changed[field] = None
    with pytest.raises(ValueError, match='compiled plant mismatch'):
        plant.checked_runtime(dict(source='a'*40, plant=changed), 'a'*40)


@pytest.mark.parametrize('field', ['qpos', 'qvel', 'physics_steps', 'correction'])
def test_reset_must_be_exact_not_merely_finite(sample, field):
    payload, spec = deepcopy(sample)
    f = payload['initial']
    if field == 'correction': f['observation']['actor'][0, 34] = .01
    else: f[field].flatten()[0] += 1
    with pytest.raises(ValueError, match='nominal first reset'): plant.check_trace(payload, spec)


@pytest.mark.parametrize('field', ['hard_limit', 'soft_limit_mask', 'joint_velocity',
    'gravity', 'angular', 'offset', 'speed', 'critic_velocity', 'height', 'support', 'tilt', 'quat'])
def test_misrecorded_frame_refused(sample, field):
    payload, spec = deepcopy(sample); f = payload['ticks'][0]['boundaries'][-1]
    if field == 'hard_limit': f['state'][field][0] = ~f['state'][field][0]
    elif field == 'soft_limit_mask': f[field][0, 0] = ~f[field][0, 0]
    elif field in ('joint_velocity', 'tilt'): f['state'][field].flatten()[0] += .1
    elif field == 'quat': f['qpos'][0, 3] += .1
    else:
        key, index = dict(gravity=('actor', 0), angular=('actor', 3), offset=('actor', 6),
            speed=('actor', 20), critic_velocity=('critic', 44), height=('critic', 47),
            support=('critic', 48))[field]
        f['observation'][key][0, index] += .1
    with pytest.raises(ValueError): plant.check_frame(f, spec)


def test_arbitrary_root_rotation_and_native_body_velocity():
    env = WarpStanceRuntime(2, device='cpu'); spec = plant.describe(env.native)
    quat = torch.tensor([.8, .2, -.3, .4]); quat /= quat.norm()
    env._view('qpos')[:, 3:7] = quat
    env._view('qvel')[:, :6] = torch.tensor([.1, -.2, .3, .2, -.3, .4])
    env._forward(); env._refresh(env.live)
    plant.check_frame(trace.owned(env.snapshot()), spec)


def test_real_terminal_contact_reduction_and_corruption(monkeypatch):
    env = WarpStanceRuntime(2, device='cpu'); spec = plant.describe(env.native)
    recorder = trace.FirstAttemptTrace(binding(), env.snapshot())
    integrate = env.integrator.integrate
    def tipped(rows):
        integrate(rows)
        env._view('qpos')[0, 3:7] = torch.tensor([math.cos(.2), 0, math.sin(.2), 0])
    monkeypatch.setattr(env.integrator, 'integrate', tipped)
    obs = env.observations()['actor']; action = torch.zeros(2, 10)
    recorder.append(env.step(action), obs, action)
    payload = recorder.payload(); trace.replay(payload, binding())
    assert plant.check_trace(payload, spec)['terminal_contact_records_checked'] == 1
    record = payload['ticks'][-1]['terminal_records'][0]
    bad = deepcopy(record); bad['state']['support'][0] += 1
    with pytest.raises(ValueError, match='terminal contact support'): plant.check_contacts(bad, spec, 2)
    bad = deepcopy(record); bad['state']['forbidden_contact'] = not bad['state']['forbidden_contact']
    with pytest.raises(ValueError, match='forbidden contact'): plant.check_contacts(bad, spec, 2)


@pytest.mark.parametrize('damage', ['world', 'geom', 'float_id', 'dim', 'shape', 'address', 'nonfinite'])
def test_contact_layout_cannot_be_reinterpreted(sample, damage):
    _, spec = sample
    # Explicit synthetic table tests schema only, not solver provenance.
    raw = dict(worldid=[0], geom=[[spec['floor'], spec['feet'][0]]], dim=[3],
        dist=[0.], pos=[[0., 0., 0.]], frame=[torch.eye(3).tolist()],
        friction=[[.6, .6, 0., 0., 0.]], force=[[1., 0., 0., 0., 0., 0.]], efc_address=[[0]])
    record = dict(world_id=0, contacts=raw, state=dict(support=[1., 0.], forbidden_contact=False))
    plant.check_contacts(record, spec, 2)
    if damage == 'world': raw['worldid'] = [1]
    elif damage == 'geom': raw['geom'][0][1] = spec['topology'][3]
    elif damage == 'float_id': raw['geom'][0][1] += .1
    elif damage == 'dim': raw['dim'] = [2]
    elif damage == 'shape': raw['geom'] = raw['geom'][0]
    elif damage == 'address': raw['efc_address'] = [0]
    else: raw['force'][0][0] = float('nan')
    with pytest.raises(ValueError): plant.check_contacts(record, spec, 2)
