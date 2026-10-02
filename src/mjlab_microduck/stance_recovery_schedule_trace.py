"""CPU-only frozen-parent scheduled first attempts and independent consistency replay.

Sibling to D1, never an expansion of its pulse/checkpoint protocols. This is
one fresh world, no reset or optimizer. Recorded state/control replay is not
fresh whole-trajectory physics re-simulation or a recovery/football admission.
"""
from copy import deepcopy
from hashlib import sha256
import io
import os
import time

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_matrix as matrix
from mjlab_microduck.stance_transition import PhysicsState, physical_failures
from mjlab_microduck.stance_evaluation import score_attempt
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-cpu-scheduled-first-attempt-v1'
EVALUATION_SEED = 671
LIMIT = 128*1024*1024
PULSE_KEYS = {'before_steps', 'accepted', 'pre_xfrc', 'pre_qfrc',
              'post_xfrc', 'post_qfrc', 'phases', 'schedule_sha256', 'binding_sha256'}


def binding(declaration, compiled_plant, cpu_profile):
    declaration = schedule.checked(declaration)
    require(declaration['worlds'] == 1 and declaration['split'] == 'held-out',
            'one explicitly held-out CPU scheduled attempt')
    baseline.force._binding(compiled_plant)
    require(type(compiled_plant) is dict and 'selected_plant' in compiled_plant,
            'actual complete scheduled compiled plant')
    profile.validate_receipt(cpu_profile)
    return dict(protocol=PROTOCOL, source=declaration['source'], worlds=1,
        capture_device='cpu', schedule_sha256=schedule.binding_sha256(declaration),
        plant_sha256=sha256(canonical(compiled_plant).encode()).hexdigest(),
        checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        checkpoint_identity=parent.expected_identity(), evaluation_seed=EVALUATION_SEED,
        cpu_math_profile=deepcopy(cpu_profile))


class ScheduledTrace(trace.FirstAttemptTrace):
    def __init__(self, value, initial, declaration, compiled_plant):
        require(canonical(value) == canonical(binding(declaration, compiled_plant, value['cpu_math_profile'])),
                'exact independent scheduled binding')
        self.binding = deepcopy(value); self.n = 1; self.faulted = False
        self._tensor_device = 'cpu'
        frame = trace.owned(initial); trace.validate_frame(frame, 1)
        require(str(initial['qpos'].device) == 'cpu' and not frame['physics_steps'].any()
                and not physical_failures(PhysicsState(**frame['state']), frame['physics_steps']).any(),
                'fresh valid CPU scheduled first attempt')
        self.initial = frame; self.last = frame; self.ticks = []; self.terminals = [None]

    def _initialize(self, *_):
        raise RuntimeError('scheduled trace requires its independent declaration constructor')


@torch.no_grad()
def collect(env, checkpoint_raw, *, deadline_monotonic, policy_tick_limit=250, clock=time.monotonic):
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CUDA-hidden CPU scheduled collection')
    require(type(env) is ScheduledRecoveryRuntime and env.n == 1 and env.live.all()
            and not env.steps.any() and str(env.device) == 'cpu' and not env.wp_device.is_cuda,
            'fresh one-world actual CPU scheduled runtime')
    require(env.forward_graph is None and env.solved_field_check == 'packed', 'eager packed scheduled collection')
    require(type(policy_tick_limit) is int and 1 <= policy_tick_limit <= 250
            and clock() < deadline_monotonic, 'bounded scheduled entry/ticks')
    declaration = schedule.checked(env.schedule_declaration)
    actual_profile = profile.checked_receipt()
    b = binding(declaration, env.binding, actual_profile)
    actor, _ = checkpoint.load_lean_replication_evaluation(
        checkpoint_raw, baseline.CHECKPOINT_SHA256, parent.expected_identity())
    recorder = ScheduledTrace(b, env.snapshot(), declaration, env.binding)
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(b), ticks=[])
    pulses = []; started = clock(); reason = 'policy-tick-limit'
    for _ in range(policy_tick_limit):
        if not env.live.any(): reason = 'all-first-attempts-complete'; break
        if clock() >= deadline_monotonic: reason = 'wall-budget-exhausted'; break
        obs = env.observations()['actor']; action = checkpoint.infer(actor, obs)
        result = env.step_with_schedule(action, capture_control=True)
        recorder.append(result, obs, action)
        controls['ticks'].append(trace.owned(result['control_evidence']))
        pulses.append(trace.owned(result['scheduled_pulse_evidence']))
    require(pulses, 'at least one complete scheduled tick before retention')
    if not env.live.any(): reason = 'all-first-attempts-complete'
    return dict(protocol=PROTOCOL, declaration=declaration, compiled_plant=deepcopy(env.binding),
        payload=recorder.payload(), control_evidence=controls, scheduled_pulse_evidence=pulses,
        collection=dict(policy_ticks=len(pulses), elapsed_seconds=clock()-started, stop_reason=reason),
        backend=dict(torch_device='cpu', warp_is_cuda=False), **baseline.FALSE_FLAGS)


def _phase(value, compiled_plant, declaration, steps, accepted, ctrl):
    require(type(value) is dict and set(value) == {'solved', 'inputs', 'motor_fields'}, 'complete scheduled phase')
    forward.validate_output(value['solved'], 1)
    inputs = value['inputs']
    require(set(inputs) == {'ctrl', 'xfrc_applied', 'qfrc_applied'}, 'complete scheduled phase inputs')
    trace.tensor(inputs['ctrl'], (1, 14), torch.float32, 'scheduled phase ctrl')
    require(torch.equal(inputs['ctrl'], ctrl), 'scheduled phase committed control')
    schedule.validate_wrenches(inputs['xfrc_applied'].tolist(), inputs['qfrc_applied'].tolist(),
        declaration, steps, accepted, compiled_plant['nbody'], compiled_plant['body_id'])
    fields = value['motor_fields']
    require(set(fields) == {'dof_frictionloss', 'dof_damping'}, 'complete scheduled motor fields')
    for key, item in fields.items():
        trace.tensor(item, (1, 20), torch.float32, key)
        require((item >= 0).all(), 'nonnegative scheduled motor fields')


def check_pulses(value):
    payload, declaration, compiled = value['payload'], value['declaration'], value['compiled_plant']
    pulses, ticks, controls = value['scheduled_pulse_evidence'], payload['ticks'], value['control_evidence']['ticks']
    require(type(pulses) is list and len(pulses) == len(ticks) == len(controls), 'scheduled tick coverage')
    checked = 0; windows = 0; delivered = 0
    nonzero = any(declaration['row_cells'][0]['force_world_newtons'])
    for entries, tick, ctl in zip(pulses, ticks, controls):
        frames = tick['boundaries']; proposals = [p for p in ctl['proposals'] if p['accepted'].any()]
        require(type(entries) is list and len(entries) == len(frames)-1 == len(proposals),
                'one scheduled force record per accepted physical substep')
        for row, before, after, proposal in zip(entries, frames, frames[1:], proposals):
            require(type(row) is dict and set(row) == PULSE_KEYS, 'exact scheduled substep record')
            require(row['schedule_sha256'] == schedule.binding_sha256(declaration)
                    and row['binding_sha256'] == sha256(canonical(compiled).encode()).hexdigest(),
                    'scheduled force record source/plant binding')
            trace.tensor(row['before_steps'], (1,), torch.int64, 'scheduled clocks')
            trace.tensor(row['accepted'], (1,), torch.bool, 'scheduled accepted mask')
            require(torch.equal(row['before_steps'], before['physics_steps'])
                    and torch.equal(row['accepted'], after['physics_steps']-before['physics_steps'] == 1)
                    and torch.equal(row['accepted'], proposal['accepted']), 'scheduled accepted clock binding')
            steps, accepted = row['before_steps'].tolist(), row['accepted'].tolist()
            for key, shape in (('pre_xfrc', (1, compiled['nbody'], 6)), ('post_xfrc', (1, compiled['nbody'], 6)),
                               ('pre_qfrc', (1, 20)), ('post_qfrc', (1, 20))):
                trace.tensor(row[key], shape, torch.float32, key)
            schedule.validate_wrenches(row['pre_xfrc'].tolist(), row['pre_qfrc'].tolist(), declaration,
                steps, accepted, compiled['nbody'], compiled['body_id'])
            schedule.validate_wrenches(row['post_xfrc'].tolist(), row['post_qfrc'].tolist(), declaration,
                steps, [False], compiled['nbody'], compiled['body_id'])
            window = schedule.window_mask(declaration, steps, accepted)[0]
            require((row['phases'] is not None) == window, 'exact scheduled phase-window coverage')
            if window:
                phases = row['phases']; windows += 1
                require(type(phases) is dict and set(phases) == {'forced_pre', 'integrated', 'unforced_post'},
                        'all scheduled force phases')
                for name in ('forced_pre', 'integrated', 'unforced_post'):
                    _phase(phases[name], compiled, declaration, steps,
                        accepted if name != 'unforced_post' else [False], proposal['committed']['ctrl'])
                pre, integrated, post = [phases[k] for k in ('forced_pre', 'integrated', 'unforced_post')]
                for name, frame in (('forced_pre', before), ('integrated', after), ('unforced_post', after)):
                    for key in ('qpos', 'qvel'):
                        require(torch.equal(phases[name]['solved']['kinematics'][key], frame[key]),
                                'scheduled phase physical boundary '+name+'/'+key)
                require(torch.equal(integrated['solved']['kinematics']['time'],
                    pre['solved']['kinematics']['time']+.002), 'one scheduled Euler clock increment')
                require(fixture.exact_tree(integrated['solved']['kinematics'], post['solved']['kinematics']),
                        'scheduled unforced post cannot integrate')
                for key in ('dynamics', 'solver', 'contacts', 'constraints'):
                    require(fixture.exact_tree(pre['solved'][key], integrated['solved'][key]),
                            'scheduled integrated fields retain pre-Euler solve '+key)
                require(all(fixture.exact_tree(pre['motor_fields'], p['motor_fields']) for p in (integrated, post)),
                        'fixed scheduled substep motor fields')
                delivered += int(nonzero)
            checked += 1
    return dict(checked_physics_steps=checked, pulse_window_steps=windows,
        delivered_nonzero_pulse_steps=delivered,
        complete_pulse_delivery=windows == declaration['row_cells'][0]['duration_steps'],
        complete_phase_checks=windows == declaration['row_cells'][0]['duration_steps'],
        recorded_phase_checks_valid=True, exact_full_force_arrays_checked=True, unforced_post_arrays_checked=True)


@torch.no_grad()
def score(value, checkpoint_raw, expected_schedule, expected_plant):
    require(type(value) is dict and set(value) == {'protocol', 'declaration', 'compiled_plant',
        'payload', 'control_evidence', 'scheduled_pulse_evidence', 'collection', 'backend'} | set(baseline.FALSE_FLAGS),
        'exact CPU scheduled artifact schema')
    require(value['protocol'] == PROTOCOL and canonical(value['declaration']) == canonical(schedule.checked(expected_schedule))
            and canonical(value['compiled_plant']) == canonical(expected_plant)
            and value['backend'] == dict(torch_device='cpu', warp_is_cuda=False)
            and all(value[k] is False for k in baseline.FALSE_FLAGS), 'exact non-admitting CPU scheduled artifact')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'independent CPU scheduled scorer')
    payload = value['payload']; require(set(payload) == {'binding', 'initial', 'ticks'}, 'complete scheduled trace')
    b = payload['binding']; profile.check_recorded(b['cpu_math_profile'])
    actor, restored = checkpoint.load_lean_replication_evaluation(checkpoint_raw, baseline.CHECKPOINT_SHA256, parent.expected_identity())
    recorder = ScheduledTrace(b, payload['initial'], expected_schedule, expected_plant)
    require(type(payload['ticks']) is list and payload['ticks'], 'ordered nonempty scheduled ticks')
    for tick in payload['ticks']:
        require(set(tick) == trace.TICK_KEYS, 'exact scheduled physical tick')
        require(torch.equal(checkpoint.infer(actor, tick['actor_input']), tick['actions']), 'exact scheduled actor replay')
        recorder.append(tick, tick['actor_input'], tick['actions'])
    collection = value['collection']
    require(set(collection) == {'policy_ticks', 'elapsed_seconds', 'stop_reason'}
            and type(collection['policy_ticks']) is int and collection['policy_ticks'] == len(recorder.ticks)
            and type(collection['elapsed_seconds']) is float and 0 < collection['elapsed_seconds'] < 120
            and collection['stop_reason'] in ('policy-tick-limit', 'all-first-attempts-complete', 'wall-budget-exhausted')
            and ((collection['stop_reason'] == 'all-first-attempts-complete') == (recorder.terminals[0] is not None)),
            'bounded scheduled collection and terminal reason')
    plant_receipt = plant.check_trace(payload, expected_plant['selected_plant'])
    control_receipt = control.replay(value['control_evidence'], payload, expected_plant['selected_plant'])
    pulse_receipt = check_pulses(value)
    frames = [recorder.initial]+[f for tick in recorder.ticks for f in tick['boundaries'][1:]]
    state = PhysicsState(**{k: torch.cat([f['state'][k] for f in frames]) for k in trace.STATE_KEYS})
    terminal = recorder.terminals[0]; proposed = None if terminal is None else terminal['rejected_proposed_torque_nm']
    numeric = score_attempt(state, torch.cat([f['qpos'][:, :3] for f in frames]),
        torch.cat([f['soft_limit_mask'] for f in frames]), torch.cat([f['physics_steps'] for f in frames]),
        rejected_proposed_torque=None if proposed is None else torch.tensor(proposed, dtype=torch.float32))
    require(numeric['complete_first_attempt'] == (terminal is not None), 'scheduled terminal/score agreement')
    return dict(protocol=PROTOCOL, cell=expected_schedule['cell_ids'][0], numerical_diagnostic=numeric,
        collection=deepcopy(collection), pulse=pulse_receipt, actor_replay_max_abs_error=0.,
        strict_actor_restore=restored['strict_actor_restore'], plant=plant_receipt, control=control_receipt,
        maximum_tilt_rad=float(state.tilt.max()), maximum_planar_speed_mps=float(state.root_velocity[:, :2].norm(dim=1).max()),
        maximum_motor_torque_nm=float(state.torque.abs().max()),
        maximum_abs_joint_power_w=float((state.torque*state.joint_velocity).abs().max()),
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False, **baseline.FALSE_FLAGS)


def prefix_hash(value, step):
    """Hash a previously independently scored pre-force prefix, even mid-tick.

    This structural helper does not authenticate its caller's input. Callers
    must first verify the whole capture bytes and independently score the case.
    """
    require(type(step) is int and 0 <= step < 2500, 'bounded scheduled prefix step')
    tick_index, substep = divmod(step, 10)
    ticks, controls = value['payload']['ticks'], value['control_evidence']['ticks']
    require(len(ticks) > tick_index and len(controls) == len(ticks), 'complete scheduled prefix')
    tick, ctl = ticks[tick_index], controls[tick_index]
    require(len(tick['boundaries']) > substep and len(ctl['proposals']) > substep
            and tick['boundaries'][substep]['physics_steps'].tolist() == [step]
            and ctl['proposals'][substep]['before_steps'].tolist() == [step], 'exact scheduled pre-force prefix clock')
    return matrix.prefix_hash(dict(initial=value['payload']['initial'], unforced_ticks=ticks[:tick_index],
        unforced_control=controls[:tick_index], onset=dict(actor_input=tick['actor_input'], actions=tick['actions'],
            boundaries=tick['boundaries'][:substep+1], initial_control=ctl['initial'], after_action=ctl['after_action'],
            proposals=ctl['proposals'][:substep+1])))


def encode(value):
    out = io.BytesIO(); torch.save(trace.owned(value), out); raw = out.getvalue()
    require(0 < len(raw) <= LIMIT, 'bounded complete scheduled capture')
    return raw


def verify(raw, expected_sha256, checkpoint_raw, expected_schedule, expected_plant):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == expected_sha256,
            'scheduled whole-byte hash before tensor loading')
    return score(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), checkpoint_raw, expected_schedule, expected_plant)
