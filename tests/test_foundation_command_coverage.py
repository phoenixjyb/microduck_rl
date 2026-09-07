import copy

import pytest
import torch

from mjlab_microduck.foundation_command_coverage import command_coverage, audit_retained

BOUNDS=dict(lin_vel_x=[.3,.3],lin_vel_y=[0.,0.],ang_vel_z=[0.,0.])


def test_native_float32_bounds_and_all_samples_no_mutation():
    x=torch.tensor([.3,0.,0.]).expand(400,8,3).clone()
    original=x.clone();rng=torch.get_rng_state().clone()
    result=command_coverage(x.tolist(),BOUNDS)
    assert all(v['outside_samples']==0 for g in result['groups'].values() for v in g.values())
    assert torch.equal(original,x) and torch.equal(rng,torch.get_rng_state())
    assert not result['causal_failure_explained'] and not result['training_admitted']


def test_startup_and_settled_are_separate_and_tiny_yaw_not_hidden():
    x=torch.tensor([.3,0.,0.]).expand(400,8,3).clone()
    x[:100,:,2]=-.1;x[100,0,2]=1e-7
    r=command_coverage(x.tolist(),BOUNDS)
    assert r['groups']['all']['ang_vel_z']['outside_samples']==801
    assert r['groups']['settled']['ang_vel_z']['outside_samples']==1
    assert r['groups']['settled']['ang_vel_z']['samples']==2400


@pytest.mark.parametrize('bad',['nan','shape','bounds','columns','overflow'])
def test_invalid_or_ambiguous_coverage_fails_closed(bad):
    x=torch.tensor([.3,0.,0.]).expand(400,8,3).clone().tolist();b=copy.deepcopy(BOUNDS)
    if bad=='nan':x[0][0][0]=float('nan')
    if bad=='shape':x.pop()
    if bad=='bounds':b['ang_vel_z']=[1.,-1.]
    if bad=='columns':del b['lin_vel_y']
    if bad=='overflow':x[0][0][0]=1e100
    with pytest.raises((ValueError,RuntimeError)):command_coverage(x,b)


def test_changed_manifest_refused_before_loading_yaml_or_reports(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='exact closed'):audit_retained(tmp_path)
