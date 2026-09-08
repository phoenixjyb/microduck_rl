"""Known synthetic signals, no simulator, checkpoint inference or acceptance changes."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from mjlab_microduck import foundation_motion_decomposition as analysis
from mjlab_microduck import foundation_command_map as mapping
from test_foundation_command_map import trace,cell


def test_constant_bias_is_not_called_oscillation_and_cannot_be_promoted():
    t = trace();t.velocity[:,:,1] = .08;t.velocity[:,:,2] = .02
    report = analysis.decompose(cell(),t);g=report['groups']['settled']
    assert g['route_forward_bias_mps']==pytest.approx([-.02]*8)
    assert g['lateral_demeaned_rms_mps']==pytest.approx([0]*8,abs=1e-14)
    assert g['lateral_spectrum']['dominant_hz']==[None]*8
    assert g['explanatory_boxcar']['valid_windows']==275
    assert not report['policy_acceptance'] and not report['training_admitted']


def test_sinusoid_with_drift_has_correct_bias_variance_and_frequency():
    t = trace();x=torch.arange(400,dtype=torch.float64)*.02
    wave=torch.sin(2*torch.pi*2*x)[:,None]
    t.velocity[:,:,1]=.08+.03*wave;t.velocity[:,:,2]=.02+.1*wave
    g=analysis.decompose(cell(),t)['groups']['settled']
    assert g['route_forward_bias_mps']==pytest.approx([-.02]*8)
    assert g['lateral_demeaned_rms_mps']==pytest.approx([.1/2**.5]*8)
    assert g['lateral_demeaned_energy_fraction']==pytest.approx([.005/.0054]*8)
    assert g['lateral_spectrum']['dominant_hz']==pytest.approx([2.]*8)
    assert max(g['explanatory_boxcar']['absolute_lateral_mean_mps'])<.03
    assert min(g['absolute_lateral_mean_mps'])>.05
    assert g['explanatory_boxcar']['acceptance_evaluated'] is False
    for bias,rms,fluct in zip(g['route_forward_bias_mps'],g['route_forward_rms_error_mps'],g['route_forward_demeaned_rms_mps']):
        assert rms*rms==pytest.approx(bias*bias+fluct*fluct)


def test_named_motor_peak_preserves_environment_control_index_and_units():
    t=trace();j=list(t.joint_names).index('left_knee');t.pre_force[120,3,j]=.6
    report=analysis.decompose(cell(),t)
    joint=report['groups']['settled']['named_joint_load']['left_knee']
    assert joint['peak']==dict(utilization=1.,environment=3,control_step=120,control_interval_start_s=2.4)
    assert joint['soft_limit_fraction']==pytest.approx(1/2400)
    assert report['groups']['startup']['named_joint_load']['left_knee']['soft_limit_fraction']==0
    assert report['thermal_calibration_verified'] is False


def test_input_rng_and_historical_source_are_unchanged():
    t=trace();before={k:v.clone() for k,v in vars(t).items() if isinstance(v,torch.Tensor)}
    path=Path(mapping.__file__);sha=hashlib.sha256(path.read_bytes()).hexdigest();rng=torch.get_rng_state().clone()
    analysis.decompose(cell(),t)
    assert all(torch.equal(getattr(t,k),v) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state()) and hashlib.sha256(path.read_bytes()).hexdigest()==sha


@pytest.mark.parametrize('change',['partial','terminal','nonfinite','command','joint-order'])
def test_no_analysis_of_incomplete_corrupt_or_safety_failed_trace(change):
    from dataclasses import replace
    t=trace(steps=200 if change=='partial' else 400)
    if change=='terminal':t.dones[-1,0]=True
    elif change=='nonfinite':t.pre_force[0,0,0]=float('nan')
    elif change=='command':t.consumed[0,0,0]=.2
    elif change=='joint-order':t=replace(t,joint_names=t.joint_names[::-1])
    with pytest.raises(ValueError):analysis.decompose(cell(),t)


def test_result_pin_is_checked_before_any_reader(tmp_path):
    (tmp_path/'campaign-result.json').write_text('{}')
    with pytest.raises(ValueError,match='exact closed map'):
        analysis.analyze_map(tmp_path,reader=lambda *a:pytest.fail('unbound input'))


@pytest.mark.parametrize('corrupt',[False,True])
def test_campaign_uses_all_fixed_cells_and_rejects_changed_original_score(tmp_path,monkeypatch,corrupt):
    traces=[trace(c.speed_mps) for c in mapping.schedule()]
    scores=[mapping.score(c,t) for c,t in zip(mapping.schedule(),traces)]
    source=dict(verified_cell_count=18,summary=dict(decision='complete-descriptive-map',cells=scores))
    path=tmp_path/'campaign-result.json';path.write_text(json.dumps(source))
    monkeypatch.setattr(analysis,'RESULT_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    calls=[]
    def reader(directory,c):
        i=int(directory.name.split('-')[1]);assert c==mapping.schedule()[i];calls.append(i)
        score=dict(scores[i])
        if corrupt:score['classification']='changed'
        return traces[i],dict(score=score,manifest_sha256='a'*64)
    if corrupt:
        with pytest.raises(ValueError,match='original score'):analysis.analyze_map(tmp_path,reader=reader)
        assert calls==[0]
    else:
        report=analysis.analyze_map(tmp_path,reader=reader)
        assert calls==list(range(18)) and len(report['cells'])==18
        assert report['historical_gates_changed'] is report['policy_acceptance'] is False
        assert list(tmp_path.iterdir())==[path]
