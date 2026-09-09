"""CPU fixture learner codec, explicitly NOT a resumable simulation checkpoint."""
from copy import deepcopy
from hashlib import sha256
import io
import re

import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.stance_ppo import CpuStanceLearner, CONFIG
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'football-b1n-cpu-learner-fixture-v1'
LIMIT = 16*1024*1024


def binding_check(binding):
    require(set(binding) == {'source', 'runtime_sha256', 'fixture_launch_sha256'}, 'exact learner fixture binding')
    for key, value in binding.items():
        require(type(value) is str and re.fullmatch('[0-9a-f]{'+('40' if key == 'source' else '64')+'}', value), 'learner binding hash')


def optimizer_check(saved, learner, updates):
    require(type(saved) is dict and set(saved) == {'state', 'param_groups'}, 'exact Adam state groups')
    expected = learner.algorithm.optimizer.state_dict()['param_groups']
    require(saved['param_groups'] == expected, 'exact Adam configuration and parameter order')
    ids = expected[0]['params']
    parameters = learner.algorithm.optimizer.param_groups[0]['params']
    require(set(saved['state']) == (set(ids) if updates else set()), 'complete Adam moment inventory')
    for index, parameter in zip(ids, parameters):
        if not updates: continue
        value = saved['state'][index]
        require(set(value) == {'step', 'exp_avg', 'exp_avg_sq'}, 'exact Adam moments')
        for key, shape in (('step', ()), ('exp_avg', parameter.shape), ('exp_avg_sq', parameter.shape)):
            v = value[key]
            require(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.dtype == torch.float32
                    and v.shape == shape and torch.isfinite(v).all(), 'finite shaped Adam '+key)
        require(value['step'].item() == updates*20, 'exact Adam update count')
        require((value['exp_avg_sq'] >= 0).all(), 'nonnegative Adam second moment')


def encode(learner, binding):
    binding_check(binding); learner._healthy()
    require(type(learner) is CpuStanceLearner and learner.phase == 'empty' and learner.storage.step == 0
            and learner.algorithm.transition.actions is None, 'learner checkpoint only at empty iteration boundary')
    require(type(learner.updates) is int and 0 <= learner.updates <= 2, 'bounded fixture update count')
    optimizer = trace.owned(learner.algorithm.optimizer.state_dict())
    optimizer_check(optimizer, learner, learner.updates)
    states = checkpoint.states_of(learner.actor, learner.critic)
    checkpoint.validate_states(states, *checkpoint.fresh_models(523))
    value = dict(protocol=PROTOCOL, binding=deepcopy(binding), config=deepcopy(CONFIG), worlds=learner.n,
        seed=523, updates=learner.updates, initial_state_sha256=learner.initial_hash,
        states=states, optimizer=optimizer, torch_cpu_rng=learner.rng.clone(),
        environment_state_included=False, simulation_resume_authorized=False,
        checkpoint_admitted=False, physical_motion_authorized=False)
    buffer = io.BytesIO(); torch.save(value, buffer); raw = buffer.getvalue()
    require(len(raw) <= LIMIT, 'bounded learner checkpoint')
    return raw


def load_fixture(raw, expected_sha256, binding):
    """Restore into a new CPU learner, without resetting or resuming an environment.

    Only synthetic continuation tests may attach fresh stand-in inputs. A real
    simulator continuation needs separate physical/motor/FIFO/RNG state gates.
    """
    binding_check(binding)
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == expected_sha256,
            'learner checkpoint byte identity')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(set(value) == {'protocol', 'binding', 'config', 'worlds', 'seed', 'updates', 'initial_state_sha256',
        'states', 'optimizer', 'torch_cpu_rng', 'environment_state_included', 'simulation_resume_authorized',
        'checkpoint_admitted', 'physical_motion_authorized'}, 'exact learner checkpoint fields')
    require(value['protocol'] == PROTOCOL and value['binding'] == binding and value['config'] == CONFIG
            and type(value['seed']) is int and value['seed'] == 523, 'fixture learner identity')
    require(all(value[k] is False for k in ('environment_state_included', 'simulation_resume_authorized',
                'checkpoint_admitted', 'physical_motion_authorized')), 'learner fixture cannot authorize simulation')
    require(type(value['updates']) is int and 0 <= value['updates'] <= 2, 'bounded fixture update count')
    learner = CpuStanceLearner(value['worlds'])
    require(value['initial_state_sha256'] == learner.initial_hash, 'fresh learner initializer identity')
    checkpoint.validate_states(value['states'], learner.actor, learner.critic)
    optimizer_check(value['optimizer'], learner, value['updates'])
    rng = value['torch_cpu_rng']
    require(isinstance(rng, torch.Tensor) and rng.device.type == 'cpu' and rng.dtype == torch.uint8
            and rng.shape == learner.rng.shape, 'exact CPU generator state layout')
    torch.Generator(device='cpu').set_state(rng)  # Validate without touching caller RNG.
    learner.algorithm.optimizer.load_state_dict(value['optimizer'])
    learner.rng = rng.clone(); learner.updates = value['updates']; learner.restored_fixture_only = True
    return learner, dict(protocol=PROTOCOL, learner_restored=True, optimizer_restored=True,
        torch_cpu_rng_restored=True, simulation_resume_authorized=False,
        checkpoint_admitted=False, physical_motion_authorized=False)
