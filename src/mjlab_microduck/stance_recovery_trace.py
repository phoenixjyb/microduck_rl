"""Separate disturbed first-attempt evidence and CPU actor/control rescore.

Does not re-simulate the whole physical trajectory or attest CUDA execution.
No nominal trace protocol is widened to accept disturbed trajectories.
"""
from copy import deepcopy
from hashlib import sha256
import io
import time

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_disturbance_fixture as fixture
from mjlab_microduck import stance_forward_probe as forward
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck.stance_transition import PhysicsState, physical_failures
from mjlab_microduck.stance_evaluation import score_attempt
from mjlab_microduck.first_attempt_smoke import require

PULSE_KEYS = {'before_steps', 'accepted', 'pre_xfrc', 'pre_qfrc',
              'post_xfrc', 'post_qfrc', 'phases'}


def validate_binding(binding, declaration):
    require(declaration == contract.declaration(declaration['source'], declaration['plant']),
            'exact independent recovery declaration')
    require(binding == contract.binding(declaration['source'], declaration['plant'],
        binding['case'], binding['capture_device'], binding['checkpoint_identity'],
        binding['cpu_math_profile']), 'exact independent recovery binding')
    profile.validate_receipt(binding['cpu_math_profile'])
    checkpoint.validate_identity(binding['checkpoint_identity'], evaluation='lean-replication')


class RecoveryTrace(trace.FirstAttemptTrace):
    """Reuse boundary/terminal checks, not nominal protocol admission."""
    def __init__(self, binding, initial, declaration, *, replay=False):
        require(type(replay) is bool, 'explicit recorded recovery replay mode')
        validate_binding(binding, declaration)
        self.binding = deepcopy(binding); self.n = 1; self.faulted = False
        self._tensor_device = 'cpu' if replay else binding['capture_device']
        require(str(initial['qpos'].device) == self._tensor_device, 'actual recovery capture device')
        frame = trace.owned(initial); trace.validate_frame(frame, self.n)
        require(not frame['physics_steps'].any() and
                not physical_failures(PhysicsState(**frame['state']), frame['physics_steps']).any(),
                'fresh valid recovery first attempt')
        self.initial = frame; self.last = frame; self.ticks = []; self.terminals = [None]

    def _initialize(self, *_):
        raise RuntimeError('recovery requires its independent declaration constructor')


@torch.no_grad()
def collect(env, actor, declaration, checkpoint_identity, *, deadline_monotonic,
            policy_tick_limit=contract.POLICY_TICKS, clock=time.monotonic):
    require(env.n == 1 and env.live.all() and not env.steps.any(), 'fresh one-world recovery attempt')
    require(type(policy_tick_limit) is int and 1 <= policy_tick_limit <= contract.POLICY_TICKS,
            'bounded recovery policy ticks')
    require(clock() < deadline_monotonic, 'recovery entry deadline')
    require(env.forward_graph is None and env.solved_field_check == 'packed',
            'unchanged eager packed recovery checks')
    require(all(p.device.type == 'cpu' and not p.requires_grad for p in actor.parameters()),
            'frozen restored CPU recovery actor')
    require(env.binding == declaration['plant'], 'actual compiled recovery body binding')
    receipt = profile.checked_receipt()
    binding = contract.binding(declaration['source'], env.binding, env.cases[0],
                               str(env.device), checkpoint_identity, receipt)
    recorder = RecoveryTrace(binding, env.snapshot(), declaration)
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(binding), ticks=[])
    pulses = []; reason = 'policy-tick-limit'; started = clock()
    for _ in range(policy_tick_limit):
        if not env.live.any(): reason = 'all-first-attempts-complete'; break
        if clock() >= deadline_monotonic: reason = 'wall-budget-exhausted'; break
        obs = env.observations()['actor']
        action = checkpoint.infer(actor, obs.detach().cpu()).to(env.device)
        result = env.step_with_pulse(action, capture_control=True)
        recorder.append(result, obs, action)
        controls['ticks'].append(trace.owned(result['control_evidence']))
        pulses.append(trace.owned(result['pulse_evidence']))
    require(controls['ticks'], 'no complete recovery policy tick retained')
    if not env.live.any(): reason = 'all-first-attempts-complete'
    elapsed = clock()-started
    return dict(protocol=contract.TRACE_PROTOCOL, declaration=deepcopy(declaration),
        payload=recorder.payload(), control_evidence=controls, pulse_evidence=pulses,
        collection=dict(stop_reason=reason, policy_ticks=len(pulses), elapsed_seconds=elapsed),
        backend=dict(torch_device=str(env.device), warp_is_cuda=env.wp_device.is_cuda),
        **contract.FALSE_FLAGS)


def _phase(value, nbody, body_id, case, steps, accepted, ctrl):
    require(set(value) == {'solved', 'inputs', 'motor_fields'}, 'complete recovery pulse phase')
    forward.validate_output(value['solved'], 1)
    inputs = value['inputs']
    require(set(inputs) == {'ctrl', 'xfrc_applied', 'qfrc_applied'}, 'complete phase inputs')
    trace.tensor(inputs['ctrl'], (1, 14), torch.float32, 'phase control')
    require(torch.equal(inputs['ctrl'], ctrl), 'pulse phase uses committed motor control')
    contract.validate_wrenches(inputs['xfrc_applied'].tolist(), inputs['qfrc_applied'].tolist(),
        [case], steps, accepted, nbody, body_id)
    fields = value['motor_fields']
    require(set(fields) == {'dof_frictionloss', 'dof_damping'}, 'complete pulse motor fields')
    for k, v in fields.items():
        trace.tensor(v, (1, 20), torch.float32, k)
        require((v >= 0).all(), 'nonnegative pulse motor field')


def check_pulses(value):
    payload, declaration = value['payload'], value['declaration']
    binding = payload['binding']; nbody = declaration['plant']['nbody']
    body_id = declaration['plant']['body_id']; case = binding['case']
    pulses, ticks, controls = value['pulse_evidence'], payload['ticks'], value['control_evidence']['ticks']
    require(type(pulses) is list and len(pulses) == len(ticks) == len(controls), 'complete pulse tick coverage')
    delivered = 0; checked = 0; windows = 0
    for entries, tick, ctl in zip(pulses, ticks, controls):
        frames = tick['boundaries']
        require(type(entries) is list and len(entries) == len(frames)-1, 'one force record per executed step')
        proposals = [p for p in ctl['proposals'] if p['accepted'].any()]
        require(len(proposals) == len(entries), 'force and accepted motor step coverage')
        for row, before, after, proposal in zip(entries, frames, frames[1:], proposals):
            require(type(row) is dict and set(row) == PULSE_KEYS, 'exact recovery pulse record')
            trace.tensor(row['before_steps'], (1,), torch.int64, 'pulse counters')
            trace.tensor(row['accepted'], (1,), torch.bool, 'pulse accepted mask')
            require(torch.equal(row['before_steps'], before['physics_steps'])
                    and torch.equal(row['accepted'], after['physics_steps']-before['physics_steps'] == 1)
                    and torch.equal(row['accepted'], proposal['accepted']), 'pulse accepted clock binding')
            steps, accepted = row['before_steps'].tolist(), row['accepted'].tolist()
            for key, shape in (('pre_xfrc', (1, nbody, 6)), ('post_xfrc', (1, nbody, 6)),
                               ('pre_qfrc', (1, 20)), ('post_qfrc', (1, 20))):
                trace.tensor(row[key], shape, torch.float32, key)
            contract.validate_wrenches(row['pre_xfrc'].tolist(), row['pre_qfrc'].tolist(),
                [case], steps, accepted, nbody, body_id)
            contract.validate_wrenches(row['post_xfrc'].tolist(), row['post_qfrc'].tolist(),
                [case], steps, [False], nbody, body_id)
            window = bool(row['accepted'][0]) and contract.ONSET_STEP <= steps[0] < contract.ONSET_STEP+contract.PULSE_STEPS
            require((row['phases'] is not None) == window, 'exact complete pulse-window phase coverage')
            if window:
                phases = row['phases']; windows += 1
                require(set(phases) == {'forced_pre', 'integrated', 'unforced_post'}, 'complete pulse phases')
                ctrl = proposal['committed']['ctrl']
                for name in ('forced_pre', 'integrated', 'unforced_post'):
                    _phase(phases[name], nbody, body_id, case, steps,
                           accepted if name != 'unforced_post' else [False], ctrl)
                pre, integrated, post = [phases[k] for k in ('forced_pre', 'integrated', 'unforced_post')]
                for name, frame in (('forced_pre', before), ('integrated', after), ('unforced_post', after)):
                    for k in ('qpos', 'qvel'):
                        require(torch.equal(phases[name]['solved']['kinematics'][k], frame[k]),
                                'pulse phase and physical boundary '+name+'/'+k)
                require(torch.equal(integrated['solved']['kinematics']['time'],
                    pre['solved']['kinematics']['time']+.002), 'pulse exactly one physical clock increment')
                require(fixture.exact_tree(integrated['solved']['kinematics'], post['solved']['kinematics']),
                        'unforced post-forward cannot integrate')
                for key in ('dynamics', 'solver', 'contacts', 'constraints'):
                    require(fixture.exact_tree(pre['solved'][key], integrated['solved'][key]),
                            'integrated fields retain pre-Euler solve '+key)
                require(all(fixture.exact_tree(pre['motor_fields'], phase['motor_fields'])
                            for phase in (integrated, post)), 'motor fields fixed throughout pulse substep')
                delivered += int(case != 'zero-wrench')
            checked += 1
    return dict(checked_physics_steps=checked, pulse_window_steps=windows,
        delivered_nonzero_pulse_steps=delivered, complete_pulse_delivery=windows == contract.PULSE_STEPS,
        exact_scheduled_forces_checked=True, unforced_post_arrays_checked=True,
        complete_pulse_window_phase_checks=True, whole_trajectory_physics_resimulated=False)


@torch.no_grad()
def score(value, checkpoint_raw, expected_declaration):
    require(type(value) is dict and set(value) == {'protocol', 'declaration', 'payload',
        'control_evidence', 'pulse_evidence', 'collection', 'backend'} | set(contract.FALSE_FLAGS),
        'exact independent recovery artifact schema')
    require(value['protocol'] == contract.TRACE_PROTOCOL and value['declaration'] == expected_declaration
            and all(value[k] is False for k in contract.FALSE_FLAGS), 'recovery evidence never admits training/skill')
    payload = value['payload']; require(set(payload) == {'binding', 'initial', 'ticks'}, 'complete recovery physical trace')
    b = payload['binding']; validate_binding(b, expected_declaration)
    require(value['backend'] == dict(torch_device=b['capture_device'], warp_is_cuda=b['capture_device']=='cuda:0'),
            'matching retained recovery backend metadata')
    profile.check_recorded(b['cpu_math_profile'])
    actor, restored = checkpoint.load_lean_replication_evaluation(
        checkpoint_raw, contract.CHECKPOINT_SHA256, b['checkpoint_identity'])
    recorder = RecoveryTrace(b, payload['initial'], expected_declaration, replay=True)
    require(type(payload['ticks']) is list and payload['ticks'], 'complete ordered recovery ticks')
    maximum_actor_error = 0.
    for tick in payload['ticks']:
        require(set(tick) == trace.TICK_KEYS, 'exact independent recovery tick')
        inferred = checkpoint.infer(actor, tick['actor_input'])
        require(torch.equal(inferred, tick['actions']), 'exact portable frozen actor replay')
        maximum_actor_error = max(maximum_actor_error, float((inferred-tick['actions']).abs().max()))
        recorder.append(tick, tick['actor_input'], tick['actions'])
    collection = value['collection']
    require(set(collection) == {'stop_reason', 'policy_ticks', 'elapsed_seconds'}
            and collection['policy_ticks'] == len(payload['ticks'])
            and type(collection['elapsed_seconds']) is float and 0 < collection['elapsed_seconds'] < contract.PROBE_CHILD_SECONDS
            and collection['stop_reason'] in ('policy-tick-limit', 'all-first-attempts-complete', 'wall-budget-exhausted'),
            'bounded retained recovery collection')
    require((collection['stop_reason'] == 'all-first-attempts-complete') == (recorder.terminals[0] is not None),
            'complete first-terminal collection reason')
    plant_receipt = plant.check_trace(payload, expected_declaration['plant']['selected_plant'])
    control_receipt = control.replay(value['control_evidence'], payload, expected_declaration['plant']['selected_plant'])
    pulse_receipt = check_pulses(value)
    frames = [recorder.initial]+[f for tick in recorder.ticks for f in tick['boundaries'][1:]]
    state = PhysicsState(**{k: torch.cat([f['state'][k] for f in frames]) for k in trace.STATE_KEYS})
    terminal = recorder.terminals[0]; proposed = None if terminal is None else terminal['rejected_proposed_torque_nm']
    numeric = score_attempt(state, torch.cat([f['qpos'][:, :3] for f in frames]),
        torch.cat([f['soft_limit_mask'] for f in frames]), torch.cat([f['physics_steps'] for f in frames]),
        rejected_proposed_torque=None if proposed is None else torch.tensor(proposed, dtype=torch.float32))
    require(numeric['complete_first_attempt'] == (terminal is not None), 'recovery complete terminal consistency')
    return dict(protocol=contract.PROTOCOL, case=b['case'], numerical_diagnostic=numeric,
        collection=deepcopy(collection),
        pulse=pulse_receipt, actor_replay_max_abs_error=maximum_actor_error,
        strict_actor_restore=restored['strict_actor_restore'], plant=plant_receipt, control=control_receipt,
        maximum_tilt_rad=float(state.tilt.max()), maximum_planar_speed_mps=float(state.root_velocity[:, :2].norm(dim=1).max()),
        maximum_motor_torque_nm=float(state.torque.abs().max()),
        maximum_abs_joint_power_w=float((state.torque*state.joint_velocity).abs().max()),
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **contract.FALSE_FLAGS)


def encode(value):
    buffer = io.BytesIO(); torch.save(trace.owned(value), buffer)
    raw = buffer.getvalue()
    require(0 < len(raw) <= contract.CAPTURE_LIMIT, 'bounded complete recovery artifact')
    return raw


def verify(raw, expected_sha256, checkpoint_raw, expected_declaration):
    require(type(raw) is bytes and 0 < len(raw) <= contract.CAPTURE_LIMIT
            and sha256(raw).hexdigest() == expected_sha256, 'independent recovery byte hash before tensor loading')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return score(value, checkpoint_raw, expected_declaration)
