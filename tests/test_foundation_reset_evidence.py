"""Synthetic reset records test evidence plumbing, not simulator determinism."""

from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import foundation_reset_evidence as reset
from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck.first_attempt_smoke import canonical
from test_foundation_command_capture import session

CELL = mapping.Cell(503,.1,'original')


def augment(env,seed=503):
    env.cfg = NS(seed=seed)
    env.common_step_counter = 192000
    env.event_manager.domain_randomization_fields = {'tendon_length0'}
    env.sim = NS(model=NS(**{name:torch.ones(8,2) for name in reset.MODEL_FIELDS},
                         tendon_length0=torch.empty(8,0)),
                 data=NS(qpos=torch.zeros(8,21),qvel=torch.zeros(8,20)))
    env.scene['robot'].data.encoder_bias = torch.zeros(8,14)
    return env


def synthetic_report(env,cell=CELL,device='cpu'):
    report = reset.snapshot(augment(env,cell.seed),cell,reset_common_step=0)
    if device == 'cuda:0':
        # Reader test only: do not initialize CUDA to manufacture a fixture.
        report['device'] = device
        report['rng']['torch_cuda0'] = deepcopy(report['rng']['torch_cpu'])
    return report


def test_snapshot_preserves_rng_and_copies_state_without_observation_or_step(session):
    env = augment(session[0].unwrapped)
    before = canonical(reset.rng_record('cpu'))
    report = reset.snapshot(env,CELL,reset_common_step=0)
    reset.validate(json.loads(canonical(report)),CELL,device='cpu',common_step=192000)
    assert before == canonical(reset.rng_record('cpu'))
    assert session[0].steps == session[1].calls == 0
    env.sim.data.qpos.fill_(2)
    assert report['qpos']['values'][0][0] == 0
    assert report['model']['tendon_length0']['shape'] == [8,0]
    assert report['complete_internal_state'] is False


def test_actual_installed_warp_bridge_is_copied_without_mutation():
    import warp as wp
    from mjlab.sim.sim_data import TorchArray
    wp.init()
    values = torch.arange(12,dtype=torch.float32).reshape(2,6)
    bridge = TorchArray(wp.from_torch(values))
    record = reset.tensor_record(bridge)
    assert record == dict(shape=[2,6],dtype='torch.float32',values=values.tolist())
    values.zero_()
    assert record['values'][1][5] == 11


def test_arbitrary_duck_typed_tensor_adapter_is_not_trusted():
    with pytest.raises(ValueError): reset.tensor_record(NS(detach=lambda:torch.zeros(8,1)))


@pytest.mark.parametrize('change',['seed','time','counter','shape','nan','rng','cuda','model','admission'])
def test_corrupt_reset_receipt_fails_closed(session,change):
    report = synthetic_report(session[0].unwrapped)
    if change == 'seed': report['cell']['seed'] = 509
    elif change == 'time': report['timing'] = 'after first action'
    elif change == 'counter': report['reset_common_step'] = 192000
    elif change == 'shape': report['qpos']['shape'][0] = 7
    elif change == 'nan': report['qpos']['values'][0][0] = float('nan')
    elif change == 'rng': report['rng']['torch_cpu']['sha256'] = 'a'*64
    elif change == 'cuda': report['rng']['torch_cuda0'] = report['rng']['torch_cpu']
    elif change == 'model': report['model'].pop('body_ipos')
    elif change == 'admission': report['complete_internal_state'] = True
    with pytest.raises(ValueError): reset.validate(report,CELL,device='cpu',common_step=192000)
