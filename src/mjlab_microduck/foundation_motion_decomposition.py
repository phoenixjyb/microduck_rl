"""CPU-only explanatory decomposition of the closed map; never rescored admission."""

import torch

from mjlab_microduck import foundation_command_map as mapping
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'foundation-motion-decomposition-v1'
RESULT_SHA256 = '8dbc0f8fa5f41a3e35764b1e335003033f5cb72811fccc61bd14781efd7cce5d'
SMOOTH_SAMPLES = 26  # Exactly .50 s between first/last timestamps at50 Hz.


def spectrum(value):
    """Hann-windowed peak of demeaned velocity, not measured gait/contact phase."""
    n = len(value)
    centered = value-value.mean(0)
    window = torch.hann_window(n,periodic=False,dtype=torch.float64)[:,None]
    power = torch.fft.rfft(centered*window,dim=0).abs().square()[1:]
    total = power.sum(0)
    bins = power.argmax(0)
    return dict(bin_spacing_hz=1/(n*mapping.STEP_DT),window='symmetric Hann; DC excluded',
        dominant_hz=[float((bins[i]+1)/(n*mapping.STEP_DT)) if total[i]>1e-20 else None
                     for i in range(mapping.NUM_ENVS)],
        peak_power_fraction=[float(power[bins[i],i]/total[i]) if total[i]>1e-20 else None
                             for i in range(mapping.NUM_ENVS)],
        contact_phase_measured=False)


def decompose(cell,trace):
    """Pure arithmetic on a complete first-attempt trace; preserves original gates."""
    original = mapping.score(cell,trace)
    require(original['sample_steps']==400 and not original['safety_failures'],
            'complete no-safety-stop map cell for this decomposition')
    groups = {}
    for label,start,end in (('startup',0,100),('settled',100,400)):
        velocity = trace.velocity[start:end].double()
        forward,lateral = velocity[:,:,1],velocity[:,:,2]
        forward_bias = forward.mean(0)-cell.speed_mps
        forward_variance = (forward-forward.mean(0)).square().mean(0)
        lateral_mean = lateral.mean(0)
        lateral_variance = (lateral-lateral_mean).square().mean(0)
        lateral_ms = lateral.square().mean(0)
        smooth = velocity[:,:,:3].unfold(0,SMOOTH_SAMPLES,1).mean(-1)
        force = trace.pre_force[start:end].double()
        speed = trace.pre_speed[start:end].double()
        utilization = force.abs()/.6
        joints = {}
        for j,name in enumerate(trace.joint_names):
            u = utilization[:,:,j]
            peak = int(u.flatten().argmax()); row,env = divmod(peak,8)
            joints[name] = dict(utilization_p99=float(torch.quantile(u.flatten(),.99)),
                soft_limit_fraction=float((u>.7).double().mean()),
                mechanical_abs_power_mean_w=float((force[:,:,j]*speed[:,:,j]).abs().mean()),
                peak=dict(utilization=float(u[row,env]),environment=env,
                          control_step=start+row,control_interval_start_s=(start+row)*mapping.STEP_DT))
        groups[label] = dict(first_control_step=start,last_control_step=end-1,sample_count=end-start,
            route_forward_bias_mps=forward_bias.tolist(),
            route_forward_rms_error_mps=(forward-cell.speed_mps).square().mean(0).sqrt().tolist(),
            route_forward_demeaned_rms_mps=forward_variance.sqrt().tolist(),
            signed_lateral_mean_mps=lateral_mean.tolist(),
            absolute_lateral_mean_mps=lateral.abs().mean(0).tolist(),
            lateral_demeaned_rms_mps=lateral_variance.sqrt().tolist(),
            lateral_demeaned_energy_fraction=[float(lateral_variance[i]/lateral_ms[i])
                if lateral_ms[i]>1e-20 else None for i in range(8)],
            cross_route_displacement_m=(trace.position[end-1,:,1].double()-trace.position[start,:,1].double()).tolist(),
            forward_spectrum=spectrum(forward),lateral_spectrum=spectrum(lateral),
            explanatory_boxcar=dict(samples=SMOOTH_SAMPLES,timestamp_span_s=.5,
                valid_windows=len(smooth),boundary='valid only; no padding; no startup/settled mixing',
                signed_lateral_mean_mps=smooth[:,:,2].mean(0).tolist(),
                absolute_lateral_mean_mps=smooth[:,:,2].abs().mean(0).tolist(),
                route_forward_rms_error_mps=(smooth[:,:,1]-cell.speed_mps).square().mean(0).sqrt().tolist(),
                acceptance_evaluated=False),
            named_joint_load=joints)
    result = dict(protocol=PROTOCOL,cell=cell.identity(),original_classification=original['classification'],
        original_performance_failures=original['performance_failures'],groups=groups,
        timing='velocity pre-control; force/speed from corresponding pre-reset control interval, not simultaneous samples',
        torque_reference_nm=.6,thermal_calibration_verified=False,contact_phase_measured=False,
        causal_effect_established=False,policy_acceptance=False,training_admitted=False,physical_motion_authorized=False)
    canonical(result)
    return result


def analyze_map(directory,*,reader=None):
    """Reconcile the exact retained map; injected reader is only a CPU test seam."""
    from mjlab_microduck import foundation_command_campaign as campaign
    directory = campaign.native._plain_path(directory)
    raw = campaign.file_bytes(directory/'campaign-result.json')
    require(campaign.digest(raw)==RESULT_SHA256,'exact closed map bytes')
    source = campaign.parse(raw)
    require(source['verified_cell_count']==18 and source['summary']['decision']=='complete-descriptive-map'
            and len(source['summary']['cells'])==18,'complete map coverage')
    reader = campaign.verify_cell if reader is None else reader
    cells = []; hashes = {}
    for i,cell in enumerate(mapping.schedule()):
        trace,record = reader(directory/f'cell-{i:02d}',cell)
        require(record['score']==source['summary']['cells'][i],'original score remains unchanged')
        cells.append(decompose(cell,trace));hashes[f'cell-{i:02d}']=record['manifest_sha256']
    return dict(protocol=PROTOCOL,source_result_sha256=RESULT_SHA256,source_manifest_sha256=hashes,cells=cells,
                input_cells=18,explanatory_only=True,policy_acceptance=False,training_admitted=False,
                historical_gates_changed=False,physical_motion_authorized=False)


def main():
    import argparse
    import hashlib
    import os
    from pathlib import Path
    from mjlab_microduck import foundation_command_campaign as campaign
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only analysis')
    # Original-host exact arithmetic is authoritative. Cross-architecture last-bit
    # reductions can fail strict reconciliation; do not loosen the old reader.
    require(Path('/etc/machine-id').read_text().strip()=='0c79e415429b4933a400159bfa79a34d',
            'original Linux analysis host')
    output = campaign.native._plain_path(args.output)
    require(not output.exists() and not output.is_relative_to(args.input.resolve()),'new output outside raw map')
    report = analyze_map(args.input)
    report['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['selected_runtime'] = campaign.native.runtime_identity()
    require(not torch.cuda.is_initialized(),'analysis did not initialize CUDA')
    campaign.write_json(output,report)
    print(canonical(dict(output=str(output),sha256=campaign.digest(output.read_bytes()),cells=18,training_admitted=False)))


if __name__ == '__main__': main()
