"""Actual CPU Warp contacts plus synthetic ledger states, not a robot rollout."""

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
import warp as wp

from mjlab_microduck import stance_contact_evidence as evidence
from mjlab_microduck.stance_transition import PhysicsState


@pytest.fixture
def scene():
    native = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".002" solver="Newton" iterations="100" tolerance="1e-8"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body pos="-.1 0 .049"><freejoint/><geom name="left" type="sphere" size=".05" mass=".5"/></body>
        <body pos=".1 0 .049"><freejoint/><geom name="right" type="sphere" size=".05" mass=".5"/></body>
      </worldbody></mujoco>''')
    nd = mujoco.MjData(native); mujoco.mj_forward(native, nd)
    with wp.ScopedDevice('cpu'):
        model = mjwarp.put_model(native)
        data = mjwarp.put_data(native, nd, nworld=2, nconmax=8, njmax=32)
        mjwarp.forward(model, data)
    return native, nd, model, data


def test_actual_warp_contact_forces_match_native_simple_fixture_and_are_owned(scene):
    native, nd, model, data = scene
    table = evidence.read_contacts(model, data)
    assert len(table['worldid']) == 4
    support, forbidden = evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))
    expected = np.zeros(2)
    for i, c in enumerate(nd.contact):
        force = np.zeros(6); mujoco.mj_contactForce(native, nd, i, force)
        expected[int(c.geom[1])-1] += force[0]
    assert np.allclose(support.numpy(), expected[None], rtol=1e-4, atol=1e-5)
    assert not forbidden.any() and (support > 0).all()
    owned = table['pos'].clone()
    wp.to_torch(data.contact.pos).fill_(999)
    assert torch.equal(table['pos'], owned)


def test_stale_contact_and_constraint_padding_is_not_evidence(scene):
    _, _, model, data = scene
    count = int(wp.to_torch(data.nacon)[0])
    wp.to_torch(data.contact.worldid)[count:] = -99
    wp.to_torch(data.contact.pos)[count:] = float('nan')
    force = wp.to_torch(data.efc.force)
    for i, n in enumerate(wp.to_torch(data.nefc).tolist()): force[i, n:] = float('nan')
    table = evidence.read_contacts(model, data)
    assert len(table['worldid']) == count and torch.isfinite(table['force']).all()


@pytest.mark.parametrize('damage', ['capacity', 'world', 'address'])
def test_invalid_contacts_refused_before_force_kernel(scene, monkeypatch, damage):
    _, _, model, data = scene
    if damage == 'capacity': wp.to_torch(data.nacon)[0] = data.naconmax+1
    elif damage == 'world': wp.to_torch(data.contact.worldid)[0] = data.nworld
    else: wp.to_torch(data.contact.efc_address)[0, 0] = data.njmax+1
    monkeypatch.setattr(evidence.support, 'contact_force', lambda *a: pytest.fail('unsafe decoding'))
    with pytest.raises(ValueError): evidence.read_contacts(model, data)


def test_empty_contacts_have_exact_empty_force_shape(scene):
    _, _, model, data = scene
    wp.to_torch(data.nacon)[0] = 0
    table = evidence.read_contacts(model, data)
    assert table['force'].shape == (0, 6)
    support, forbidden = evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))
    assert not support.any() and not forbidden.any()


def test_pair_order_does_not_change_support_and_other_penetration_is_forbidden(scene):
    _, _, model, data = scene
    table = evidence.read_contacts(model, data)
    original = evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))[0]
    table['geom'] = table['geom'].flip(1)
    assert torch.equal(evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))[0], original)
    table['geom'][0] = torch.tensor([1, 2]); table['dist'][0] = -.001
    _, forbidden = evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))
    assert forbidden[int(table['worldid'][0])]


def ledger_inputs(scene):
    _, _, model, data = scene
    contacts = evidence.read_contacts(model, data)
    support, forbidden = evidence.contact_summary(contacts, 2, floor_id=0, foot_ids=(1, 2))
    # Deliberately synthetic rigid-Duck-shaped state for ledger/control-flow tests.
    state = PhysicsState(torch.tensor([.4, 0.]), torch.zeros(2, 3), torch.full((2,), .12),
        support, torch.zeros(2, 14), torch.zeros(2, 14), torch.zeros(2, dtype=torch.bool),
        forbidden, torch.zeros(2, dtype=torch.bool))
    qpos = torch.zeros(2, 21); qpos[:, 2] = .12; qpos[:, 3] = 1
    qpos[0, 3] = np.cos(.2); qpos[0, 5] = np.sin(.2)
    return dict(state=state, steps=torch.tensor([17, 17]), qpos=qpos, qvel=torch.zeros(2, 20),
                contacts=contacts, terminated=torch.tensor([True, False]), timed_out=torch.tensor([False, False]))


def test_first_terminal_owns_state_contacts_and_cannot_be_overwritten(scene):
    args = ledger_inputs(scene); ledger = evidence.FirstTerminalContacts(2)
    ledger.capture(**args); before = ledger.records()
    assert before[0]['physics_step'] == 17 and before[1] is None
    assert all(w == 0 for w in before[0]['contacts']['worldid'])
    args['contacts']['force'].fill_(999); args['qpos'].fill_(999)
    assert ledger.records() == before
    returned = ledger.records(); returned[0]['qpos'][0] = -123
    assert ledger.records() == before
    with pytest.raises(ValueError, match='overwritten'): ledger.capture(**args)


def test_unjustified_termination_does_not_partially_store_another_world(scene):
    args = ledger_inputs(scene); ledger = evidence.FirstTerminalContacts(2)
    args['terminated'][:] = True
    with pytest.raises(ValueError, match='actual failure'): ledger.capture(**args)
    assert ledger.records() == [None, None]


def test_exact_timeout_and_proposed_torque_terminal_are_distinct(scene):
    args = ledger_inputs(scene); ledger = evidence.FirstTerminalContacts(2)
    args['state'].tilt.zero_(); args['qpos'][:, 3:7] = torch.tensor([1., 0, 0, 0])
    args['steps'] = torch.tensor([0, 2500]); args['timed_out'][1] = True
    proposal = torch.zeros(2, 14); proposal[0] = .37
    ledger.capture(**args, proposed_torque=proposal)
    records = ledger.records()
    assert records[0]['terminated'] and records[0]['rejected_proposed_torque_nm'] is not None
    assert records[1]['timed_out'] and records[1]['rejected_proposed_torque_nm'] is None
    assert not records[1]['trajectory_continuity_validated'] and not records[1]['checkpoint_admitted']


def test_nonfinite_contact_evidence_cannot_enter_ledger(scene):
    args = ledger_inputs(scene); ledger = evidence.FirstTerminalContacts(2)
    args['contacts']['force'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite contact'): ledger.capture(**args)
    assert ledger.records() == [None, None]


@pytest.mark.parametrize('invalid_address', [False, True])
def test_elliptic_contacts_and_negative_active_addresses(scene, monkeypatch, invalid_address):
    native, nd, _, _ = scene
    native.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mujoco.mj_forward(native, nd)
    with wp.ScopedDevice('cpu'):
        model = mjwarp.put_model(native)
        data = mjwarp.put_data(native, nd, nworld=2, nconmax=8, njmax=32)
        mjwarp.forward(model, data)
    if invalid_address:
        wp.to_torch(data.contact.efc_address)[0, 1] = -1
        monkeypatch.setattr(evidence.support, 'contact_force', lambda *a: pytest.fail('negative address decoded'))
        with pytest.raises(ValueError, match='elliptic rows'): evidence.read_contacts(model, data)
    else:
        table = evidence.read_contacts(model, data)
        loads, forbidden = evidence.contact_summary(table, 2, floor_id=0, foot_ids=(1, 2))
        expected = np.zeros(2)
        for i, c in enumerate(nd.contact):
            force = np.zeros(6); mujoco.mj_contactForce(native, nd, i, force)
            expected[int(c.geom[1])-1] += force[0]
        assert np.allclose(loads.numpy(), expected[None], rtol=1e-4, atol=1e-5)
        assert not forbidden.any()


def test_no_contacts_does_not_hide_nonfinite_active_constraint_force(scene):
    _, _, model, data = scene
    wp.to_torch(data.nacon)[0] = 0
    wp.to_torch(data.efc.force)[0, 0] = float('nan')
    with pytest.raises(ValueError, match='active constraint force'): evidence.read_contacts(model, data)


def test_invalid_ledger_contact_shape_cannot_partially_commit(scene):
    args = ledger_inputs(scene); ledger = evidence.FirstTerminalContacts(2)
    args['contacts']['force'] = torch.zeros(4, 5)
    with pytest.raises(ValueError, match='contact layout'): ledger.capture(**args)
    assert ledger.records() == [None, None]
