"""Short source-bound scheduled CUDA capture with independent CPU replay.

This is a one-world, at-most-60-policy-tick integration trace.  It preserves
the CUDA/WarpCUDA capture claim in raw evidence, then checks the recorded
actor, plant, control, and force phases with CUDA hidden on the scorer.  It is
not a full-duration recovery or football acceptance protocol.
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
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_trace as cpu_schedule_trace
from mjlab_microduck import stance_recovery_matrix as matrix
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck.stance_transition import PhysicsState, physical_failures
from mjlab_microduck.stance_evaluation import score_attempt
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-cuda-scheduled-integration-trace-v1'
EVALUATION_SEED = 671
POLICY_TICK_LIMIT = 60
MAX_PHYSICS_STEPS = POLICY_TICK_LIMIT * 10
CAPTURE_LIMIT = 128 * 1024 * 1024
ALLOWED_CELLS = ('zero-wrench', '+x-2n-20steps-t250')


def binding(declaration, compiled_plant, cpu_profile):
    """Construct the exact independent CUDA scheduled-trace binding."""
    declaration = schedule.checked(declaration)
    require(declaration['worlds'] == 1 and declaration['split'] == 'held-out'
            and declaration['stage'] == 'dose' and declaration['cell_ids'] in
            ([ALLOWED_CELLS[0]], [ALLOWED_CELLS[1]]),
            'one declared held-out CUDA integration cell')
    require(declaration['parent_checkpoint_sha256'] == baseline.CHECKPOINT_SHA256,
            'unchanged frozen parent checkpoint SHA')
    baseline.force._binding(compiled_plant)
    require(type(compiled_plant) is dict and 'selected_plant' in compiled_plant,
            'actual complete scheduled compiled plant')
    profile.validate_receipt(cpu_profile)
    return dict(protocol=PROTOCOL, source=declaration['source'], worlds=1,
        capture_device='cuda:0', schedule_sha256=schedule.binding_sha256(declaration),
        plant_sha256=sha256(canonical(compiled_plant).encode()).hexdigest(),
        checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        checkpoint_identity=parent.expected_identity(), evaluation_seed=EVALUATION_SEED,
        cpu_math_profile=deepcopy(cpu_profile))


class CudaScheduledTrace(trace.FirstAttemptTrace):
    """Reuse first-attempt append validation with explicit capture/replay mode."""
    def __init__(self, value, initial, declaration, compiled_plant, *, replay=False):
        require(type(replay) is bool, 'explicit CUDA scheduled replay mode')
        expected = binding(declaration, compiled_plant, value['cpu_math_profile'])
        require(canonical(value) == canonical(expected), 'exact independent CUDA scheduled binding')
        self.binding = deepcopy(value); self.n = 1; self.faulted = False
        self._tensor_device = 'cpu' if replay else value['capture_device']
        require(str(initial['qpos'].device) == self._tensor_device,
                'actual CUDA scheduled capture device')
        frame = trace.owned(initial); trace.validate_frame(frame, 1)
        require(not frame['physics_steps'].any() and
                not physical_failures(PhysicsState(**frame['state']), frame['physics_steps']).any(),
                'fresh valid CUDA scheduled first attempt')
        self.initial = frame; self.last = frame; self.ticks = []; self.terminals = [None]

    def _initialize(self, *_):
        raise RuntimeError('CUDA scheduled trace requires its independent binding constructor')


def _require_cuda_capture_env(env):
    from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'CUDA_VISIBLE_DEVICES=0 capture')
    require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,
            'initialized visible CUDA device zero')
    require(type(env) is ScheduledRecoveryRuntime and env.n == 1 and env.live.all()
            and not env.steps.any() and str(env.device) == 'cuda:0' and env.wp_device.is_cuda,
            'fresh one-world actual CUDA/WarpCUDA scheduled runtime')
    require(env.forward_graph is None and env.solved_field_check == 'packed',
            'eager packed scheduled CUDA collection')


def _require_cpu_scorer():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'independent CPU-only CUDA scheduled scorer')


@torch.no_grad()
def collect(env, checkpoint_raw, *, deadline_monotonic, policy_tick_limit=POLICY_TICK_LIMIT,
            clock=time.monotonic):
    """Collect up to 60 one-world ticks; the caller owns lease and job caps."""
    _require_cuda_capture_env(env)
    require(type(policy_tick_limit) is int and 1 <= policy_tick_limit <= POLICY_TICK_LIMIT
            and clock() < deadline_monotonic, 'bounded CUDA scheduled entry/ticks')
    declaration = schedule.checked(env.schedule_declaration)
    actual_profile = profile.checked_receipt()
    b = binding(declaration, env.binding, actual_profile)
    actor, _ = checkpoint.load_lean_replication_evaluation(
        checkpoint_raw, baseline.CHECKPOINT_SHA256, parent.expected_identity())
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 and not p.requires_grad
                for p in actor.parameters()), 'frozen restored CPU parent actor')
    recorder = CudaScheduledTrace(b, env.snapshot(), declaration, env.binding)
    controls = dict(protocol=control.PROTOCOL, binding=deepcopy(b), ticks=[])
    pulses = []; reason = 'policy-tick-limit'; started = clock()
    for _ in range(policy_tick_limit):
        if not env.live.any():
            reason = 'all-first-attempts-complete'; break
        if clock() >= deadline_monotonic:
            reason = 'wall-budget-exhausted'; break
        obs = env.observations()['actor']
        action = checkpoint.infer(actor, obs.detach().cpu()).to(env.device)
        result = env.step_with_schedule(action, capture_control=True)
        recorder.append(result, obs, action)
        controls['ticks'].append(trace.owned(result['control_evidence']))
        pulses.append(trace.owned(result['scheduled_pulse_evidence']))
    require(pulses, 'at least one complete CUDA scheduled tick before retention')
    if not env.live.any():
        reason = 'all-first-attempts-complete'
    return dict(protocol=PROTOCOL, declaration=deepcopy(declaration),
        compiled_plant=deepcopy(env.binding), payload=recorder.payload(),
        control_evidence=controls, scheduled_pulse_evidence=pulses,
        collection=dict(policy_ticks=len(pulses), elapsed_seconds=float(clock()-started),
                        stop_reason=reason),
        backend=dict(torch_device=str(env.device), warp_is_cuda=bool(env.wp_device.is_cuda),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch_cuda_initialized=bool(torch.cuda.is_initialized()),
            torch_cuda_device=int(torch.cuda.current_device())),
        thermal_model_applied=False, whole_trajectory_physics_resimulated=False,
        **baseline.FALSE_FLAGS)


@torch.no_grad()
def score(value, checkpoint_raw, expected_schedule, expected_plant):
    """Independently validate CUDA-origin evidence with only CPU tensors/runtime."""
    schema = {'protocol', 'declaration', 'compiled_plant', 'payload', 'control_evidence',
        'scheduled_pulse_evidence', 'collection', 'backend', 'thermal_model_applied',
        'whole_trajectory_physics_resimulated'} | set(baseline.FALSE_FLAGS)
    require(type(value) is dict and set(value) == schema,
            'exact CUDA scheduled artifact schema')
    expected_schedule = schedule.checked(expected_schedule)
    require(value['protocol'] == PROTOCOL
            and canonical(value['declaration']) == canonical(expected_schedule)
            and canonical(value['compiled_plant']) == canonical(expected_plant)
            and value['backend'] == dict(torch_device='cuda:0', warp_is_cuda=True,
                cuda_visible_devices='0', torch_cuda_initialized=True, torch_cuda_device=0)
            and all(value[k] is False for k in baseline.FALSE_FLAGS)
            and value['thermal_model_applied'] is False
            and value['whole_trajectory_physics_resimulated'] is False,
            'preserved CUDA/WarpCUDA origin and non-admitting artifact')
    _require_cpu_scorer()
    payload = value['payload']
    require(type(payload) is dict and set(payload) == {'binding', 'initial', 'ticks'},
            'complete CUDA scheduled trace payload')
    b = payload['binding']; profile.check_recorded(b['cpu_math_profile'])
    require(canonical(b) == canonical(binding(expected_schedule, expected_plant,
                b['cpu_math_profile'])), 'source/checkpoint/schedule/plant/device binding')
    actor, restored = checkpoint.load_lean_replication_evaluation(
        checkpoint_raw, baseline.CHECKPOINT_SHA256, parent.expected_identity())
    recorder = CudaScheduledTrace(b, payload['initial'], expected_schedule, expected_plant, replay=True)
    require(type(payload['ticks']) is list and 1 <= len(payload['ticks']) <= POLICY_TICK_LIMIT,
            'bounded ordered CUDA scheduled ticks')
    for tick in payload['ticks']:
        require(set(tick) == trace.TICK_KEYS, 'exact CUDA scheduled physical tick')
        inferred = checkpoint.infer(actor, tick['actor_input'])
        require(torch.equal(inferred, tick['actions']), 'exact CUDA scheduled actor replay')
        recorder.append(tick, tick['actor_input'], tick['actions'])
    collection = value['collection']
    require(type(collection) is dict and set(collection) ==
            {'policy_ticks', 'elapsed_seconds', 'stop_reason'}
            and type(collection['policy_ticks']) is int
            and collection['policy_ticks'] == len(recorder.ticks)
            and type(collection['elapsed_seconds']) is float
            and 0 < collection['elapsed_seconds'] < 120
            and collection['stop_reason'] in
            ('policy-tick-limit', 'all-first-attempts-complete', 'wall-budget-exhausted')
            and ((collection['stop_reason'] == 'all-first-attempts-complete') ==
                 (recorder.terminals[0] is not None)),
            'bounded CUDA scheduled collection metadata')
    pulse_receipt = cpu_schedule_trace.check_pulses(value)
    plant_receipt = plant.check_trace(payload, expected_plant['selected_plant'])
    control_receipt = control.replay(value['control_evidence'], payload,
                                     expected_plant['selected_plant'])
    frames = [recorder.initial] + [f for tick in recorder.ticks for f in tick['boundaries'][1:]]
    state = PhysicsState(**{k: torch.cat([f['state'][k] for f in frames])
                            for k in trace.STATE_KEYS})
    terminal = recorder.terminals[0]
    proposed = None if terminal is None else terminal['rejected_proposed_torque_nm']
    numeric = score_attempt(state, torch.cat([f['qpos'][:, :3] for f in frames]),
        torch.cat([f['soft_limit_mask'] for f in frames]),
        torch.cat([f['physics_steps'] for f in frames]),
        rejected_proposed_torque=None if proposed is None else
            torch.tensor(proposed, dtype=torch.float32))
    require(numeric['complete_first_attempt'] == (terminal is not None),
            'CUDA scheduled terminal/score agreement')
    end_step = int(frames[-1]['physics_steps'][0])
    full_short_window = (len(recorder.ticks) == POLICY_TICK_LIMIT and end_step == MAX_PHYSICS_STEPS
        and terminal is None and collection['stop_reason'] == 'policy-tick-limit'
        and pulse_receipt['complete_pulse_delivery']
        and pulse_receipt['complete_phase_checks'])
    return dict(protocol=PROTOCOL, cell=expected_schedule['cell_ids'][0],
        numerical_diagnostic=numeric, collection=deepcopy(collection), pulse=pulse_receipt,
        actor_replay_max_abs_error=0., strict_actor_restore=restored['strict_actor_restore'],
        plant=plant_receipt, control=control_receipt,
        maximum_tilt_rad=float(state.tilt.max()),
        maximum_planar_speed_mps=float(state.root_velocity[:, :2].norm(dim=1).max()),
        maximum_motor_torque_nm=float(state.torque.abs().max()),
        maximum_abs_joint_power_w=float((state.torque*state.joint_velocity).abs().max()),
        integration_qualified=bool(full_short_window),
        integration_qualification=dict(exact_policy_ticks=len(recorder.ticks) == POLICY_TICK_LIMIT,
            exact_physics_steps=end_step == MAX_PHYSICS_STEPS,
            no_terminal=terminal is None, complete_declared_pulse_phases=bool(
                pulse_receipt['complete_pulse_delivery'] and pulse_receipt['complete_phase_checks']),
            full_duration_gate_passed=False),
        whole_trajectory_physics_resimulated=False, thermal_model_applied=False,
        **baseline.FALSE_FLAGS)


def prefix_hash(value, step):
    """Delegate to the existing independently checked scheduled prefix codec."""
    return cpu_schedule_trace.prefix_hash(value, step)


def encode(value):
    out = io.BytesIO(); torch.save(trace.owned(value), out); raw = out.getvalue()
    require(0 < len(raw) <= CAPTURE_LIMIT, 'bounded complete CUDA scheduled capture')
    return raw


def verify(raw, expected_sha256, checkpoint_raw, expected_schedule, expected_plant):
    """Hash exact serialized bytes before deserializing onto CPU for replay."""
    require(type(raw) is bytes and 0 < len(raw) <= CAPTURE_LIMIT
            and sha256(raw).hexdigest() == expected_sha256,
            'CUDA scheduled whole-byte hash before tensor loading')
    _require_cpu_scorer()
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return score(value, checkpoint_raw, expected_schedule, expected_plant)
