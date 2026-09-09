"""CPU matching-contract fixtures, not Duck loaded-trajectory qualification."""
from copy import deepcopy

import pytest
import torch

from mjlab_microduck import stance_contact_correspondence as audit
from test_stance_contact_evidence import scene


def table():
    return dict(worldid=torch.tensor([0, 0, 1], dtype=torch.int32),
        geom=torch.tensor([[0, 1], [0, 2], [0, 1]], dtype=torch.int32),
        dim=torch.full((3,), 3, dtype=torch.int32), dist=torch.full((3,), -.001),
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        frame=torch.eye(3).repeat(3, 1, 1), friction=torch.ones(3, 5),
        efc_address=torch.tensor([[0], [4], [0]], dtype=torch.int32),
        force=torch.tensor([[1., 0., 0., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], [3., 0., 0., 0., 0., 0.]]))


def ordered(value, indices):
    return {k: v[indices].clone() for k, v in value.items()}


def test_reordering_preserves_world_identity_and_raw_inputs():
    left=table(); right=ordered(left, [2, 0, 1]); original=deepcopy((left, right))
    r=audit.audit_contacts(left, right, 2)
    assert r['status']=='unique-key-candidates' and r['comparison']['bitwise_equal']
    assert [p['right_rows'] for p in r['candidates']]==[[1], [2], [0]]
    assert r['world_contact_counts']==dict(left=[2, 1], right=[2, 1])
    assert not r['physical_contact_identity_established'] and not r['graph_equivalence_established']
    assert not r['training_admitted'] and not r['physical_motion_authorized']
    for actual, saved in zip((left, right), original):
        for k in actual: assert torch.equal(actual[k], saved[k])


def test_different_positions_and_frames_are_reported_not_treated_as_identity_proof():
    left=table(); right=deepcopy(left)
    right['pos'][0, 0]=100.; right['frame'][0]=torch.diag(torch.tensor([1., -1., -1.]))
    r=audit.audit_contacts(left, right, 2)
    assert r['status']=='unique-key-candidates' and not r['physical_contact_identity_established']
    assert r['comparison']['fields']['pos']['max_abs_difference']==100.
    assert not r['comparison']['bitwise_equal'] and 'frame' in r['comparison']['fields']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'flipped_pair', 'changed_dimension', 'excluded'])
def test_missing_or_ambiguous_correspondence_blocks_whole_table_comparison(damage):
    left=table(); right=deepcopy(left)
    if damage=='missing': right=ordered(right, [0, 1])
    elif damage=='duplicate': right=ordered(right, [0, 0, 1, 2])
    elif damage=='flipped_pair': right['geom'][0]=right['geom'][0].flip(0)
    elif damage=='changed_dimension': right['dim'][0]=1
    else: right['efc_address'][0, 0]=-1
    r=audit.audit_contacts(left, right, 2)
    assert r['status']=='unresolved-correspondence' and r['comparison'] is None
    assert r['issues'] and not r['physical_contact_identity_established']
    if damage=='duplicate': assert 'ambiguous-right' in r['issues'][0]['reasons']


def test_distinct_points_on_the_same_geom_pair_are_not_guessed_by_nearest_neighbor():
    left=table(); left['geom'][1]=left['geom'][0]
    right=ordered(left, [1, 0, 2])
    r=audit.audit_contacts(left, right, 2)
    assert r['status']=='unresolved-correspondence' and r['comparison'] is None
    assert r['issues'][0]['reasons']==['ambiguous-left','ambiguous-right']


def test_empty_inputs_do_not_pass_a_loaded_contact_gate():
    empty=ordered(table(), [])
    r=audit.audit_contacts(empty, empty, 2)
    assert r['status']=='no-contact-evidence' and r['comparison'] is None
    assert not r['training_admitted'] and r['world_contact_counts']['left']==[0, 0]


def test_force_signed_zero_and_addresses_remain_visible_after_candidate_alignment():
    left=table(); right=deepcopy(left)
    right['force'][0, 1]=-0.; right['efc_address'][0, 0]=12
    r=audit.audit_contacts(left, right, 2)
    assert r['status']=='unique-key-candidates'
    assert not r['comparison']['bitwise_equal'] and not r['comparison']['discrete_equal']
    assert r['comparison']['fields']['force']['max_abs_difference']==0
    assert 'efc_address' in r['comparison']['fields']


@pytest.mark.parametrize('damage', ['nan', 'world', 'dim', 'sentinel', 'same_geom', 'extra', 'dtype'])
def test_invalid_tables_fail_before_producing_candidates(damage):
    left=table(); right=deepcopy(left)
    if damage=='nan': right['force'][0, 0]=float('nan')
    elif damage=='world': right['worldid'][0]=2
    elif damage=='dim': right['dim'][0]=2
    elif damage=='sentinel': right['efc_address'][0, 0]=-2
    elif damage=='same_geom': right['geom'][0]=0
    elif damage=='extra': right['extra']=torch.zeros(3)
    else: right['worldid']=right['worldid'].float()
    with pytest.raises(ValueError): audit.audit_contacts(left, right, 2)


def test_real_cpu_loaded_sphere_fixture_has_reorderable_structural_candidates(scene):
    from mjlab_microduck.stance_contact_evidence import read_contacts
    _, _, model, data = scene
    left=read_contacts(model, data); n=len(left['worldid'])
    assert n>=2 and left['force'][:, 0].sum()>0
    r=audit.audit_contacts(left, ordered(left, list(reversed(range(n)))), 2)
    assert r['status']=='unique-key-candidates' and r['comparison']['bitwise_equal']
    assert not r['training_admitted'] and not torch.cuda.is_initialized()


@pytest.mark.parametrize('worlds', [True, 0, -1, 65537])
def test_invalid_world_count_is_rejected(worlds):
    with pytest.raises(ValueError, match='world count'): audit.audit_contacts(table(), table(), worlds)


def test_oversized_or_non_cpu_tables_fail_before_comparison():
    value=table(); value['worldid']=torch.zeros(1, dtype=torch.int32).expand(1048577)
    with pytest.raises(ValueError, match='row count'): audit.audit_contacts(value, table(), 2)
    value=table(); value['force']=torch.empty(3, 6, device='meta')
    with pytest.raises(ValueError, match='CPU contact'): audit.audit_contacts(value, table(), 2)
