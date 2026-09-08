"""Read-only sampled reset and RNG evidence; not a complete simulator replay state."""

import hashlib
import math
import random

import numpy as np
import torch

from mjlab_microduck.first_attempt_smoke import canonical,require

PROTOCOL = 'foundation-map-reset-rng-v1'
MODEL_FIELDS = ('body_mass','body_inertia','body_ipos','dof_armature','geom_friction',
                'actuator_gainprm','actuator_biasprm')
TIMING = 'after wrapper reset and saved common-time restore; before first policy input'


def tensor_record(value):
    require(isinstance(value,torch.Tensor) and value.layout == torch.strided and value.numel() <= 2_000_000,
            'bounded sampled tensor')
    data = value.detach().cpu().contiguous()
    require(bool(torch.isfinite(data).all()),'finite sampled reset state')
    return dict(shape=list(data.shape),dtype=str(data.dtype),values=data.tolist())


def rng_record(device):
    def packed(state):
        raw = state.detach().cpu().numpy().tobytes()
        return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),hex=raw.hex())
    numpy = np.random.get_state()
    require(numpy[0] == 'MT19937','declared NumPy RNG')
    cuda = None
    if device == 'cuda:0':
        require(torch.cuda.is_initialized() and torch.cuda.device_count() == 1,'one initialized visible CUDA device')
        cuda = packed(torch.cuda.get_rng_state(0))
    else:
        require(device == 'cpu','explicit RNG device')
    return dict(python=random.getstate(),numpy=dict(algorithm=numpy[0],keys=numpy[1].tolist(),
                position=numpy[2],has_gauss=numpy[3],cached_gauss=numpy[4]),torch_cpu=packed(torch.get_rng_state()),
                torch_cuda0=cuda,warp_state_exported=False,complete_replay_state=False)


def snapshot(env,cell,*,reset_common_step):
    """After wrapper reset and common-time restore, before the first actor call."""
    require(env.num_envs == 8 and env.step_dt == .02 and env.cfg.seed == cell.seed,'exact sampled session')
    fields = sorted(set(MODEL_FIELDS)|set(env.event_manager.domain_randomization_fields))
    ranges = {name:list(env.event_manager.get_term_cfg(name).params['ranges'])
              for name in ('randomize_com','randomize_head_com')}
    require(ranges == dict(randomize_com=[-.003,.003],randomize_head_com=[-.003,.003]),'live reset CoM ranges')
    result = dict(protocol=PROTOCOL,cell=cell.identity(),device=str(env.device),
        timing=TIMING,
        reset_common_step=int(reset_common_step),common_step_before_action=int(env.common_step_counter),
        event_ranges=ranges,randomized_model_fields=sorted(env.event_manager.domain_randomization_fields),
        model={field:tensor_record(getattr(env.sim.model,field)) for field in fields},
        qpos=tensor_record(env.sim.data.qpos),qvel=tensor_record(env.sim.data.qvel),
        encoder_bias=tensor_record(env.scene['robot'].data.encoder_bias),rng=rng_record(str(env.device)),
        complete_internal_state=False,policy_acceptance=False,physical_motion_authorized=False)
    canonical(result)
    return result


def validate(report,cell,*,device,common_step):
    require(report['protocol'] == PROTOCOL and report['cell'] == cell.identity() and report['device'] == device,
            'sampled reset identity')
    require(device in ('cpu','cuda:0') and report['timing'] == TIMING,'sampled reset timing/device')
    require(type(report['reset_common_step']) is int and report['reset_common_step'] == 0
            and type(report['common_step_before_action']) is int and report['common_step_before_action'] == common_step,
            'wrapper-reset then common-time restoration')
    require(report['event_ranges'] == dict(randomize_com=[-.003,.003],randomize_head_com=[-.003,.003]),'sampled domain')
    require(set(report['model']) == set(MODEL_FIELDS)|set(report['randomized_model_fields']),'sampled model fields')
    for item in (*report['model'].values(),report['qpos'],report['qvel'],report['encoder_bias']):
        require(item['dtype'] in ('torch.float32','torch.float64','torch.int32','torch.int64'),'sampled tensor dtype')
        shape = item['shape']
        require(isinstance(shape,list) and len(shape) <= 6
                and all(type(n) is int and 0 <= n <= 2_000_000 for n in shape)
                and math.prod(shape) <= 2_000_000,'bounded sampled shape')
        dtype = getattr(torch,item['dtype'].split('.')[1]); value = torch.tensor(item['values'],dtype=dtype)
        # JSON loses trailing empty dimensions; retain their explicit shape without
        # pretending zero-tendon model fields are missing samples.
        if math.prod(shape) == 0:
            require(item['values'] == torch.empty(shape).tolist(),'empty sampled shape')
            value = value.reshape(shape)
        require(list(value.shape) == shape and value.numel() <= 2_000_000
                and bool(torch.isfinite(value).all()),'sampled tensor shape/finite coverage')
    require(all(math.prod(report['model'][name]['shape']) > 0 for name in MODEL_FIELDS),'mandatory model coverage')
    for name in ('qpos','qvel','encoder_bias'):
        require(len(report[name]['shape']) == 2 and report[name]['shape'][0] == 8
                and report[name]['shape'][1] > 0,'per-environment reset coverage')
    rng = report['rng']
    require(rng['warp_state_exported'] is rng['complete_replay_state'] is False,'honest RNG scope')
    require((rng['torch_cuda0'] is None) == (device == 'cpu'),'CUDA RNG coverage')
    for state in (rng['torch_cpu'],*([rng['torch_cuda0']] if device == 'cuda:0' else [])):
        raw = bytes.fromhex(state['hex'])
        require(0 < len(raw) == state['bytes'] <= 16384 and hashlib.sha256(raw).hexdigest() == state['sha256'],
                'retained RNG bytes')
    require(len(rng['python']) == 3 and rng['python'][0] == 3 and len(rng['python'][1]) == 625
            and rng['numpy']['algorithm'] == 'MT19937' and len(rng['numpy']['keys']) == 624,'Python/NumPy RNG coverage')
    require(all(report[k] is False for k in ('complete_internal_state','policy_acceptance','physical_motion_authorized')),
            'reset evidence is not admission')
    canonical(report)
