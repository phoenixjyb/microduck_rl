"""Separate frozen-policy D1 pulse declaration; no learner or skill admission."""
from copy import deepcopy
from hashlib import sha256
import json

from mjlab_microduck import stance_disturbance_contract as force

PROTOCOL = 'football-b1d-frozen-recovery-v1'
TRACE_PROTOCOL = 'football-b1d-frozen-recovery-trace-v1'
TRAINING_SOURCE = 'be2d59661af293b0d67ae20d2e16db50514cce14'
TRAINING_SEED, ITERATION, EVALUATION_SEED = 577, 255, 619
CHECKPOINT_SHA256 = '2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5'
CHECKPOINT_FILE = 'model_255.pt'
ONSET_STEP, PULSE_STEPS, TOTAL_STEPS = 500, 10, 2500
POLICY_TICKS = 250
PROBE_CASE = '+x'
CASE_NAMES = tuple(name for name, _ in force.CASES)
PROBE_CHILD_SECONDS, PROBE_SERVICE_SECONDS = 900, 960
CLOSEOUT_SECONDS, MARGIN_SECONDS = 180, 60
CUTOFF = force.CUTOFF
CAPTURE_LIMIT = 128*1024*1024
FALSE_FLAGS = dict(recovery_accepted=False, training_admitted=False,
    checkpoint_admitted=False, learned_stance_accepted=False,
    football_balance_accepted=False, physical_motion_authorized=False,
    independent_gpu_attestation=False, complete_binary_runtime_equivalence_verified=False)


def cases_checked(cases):
    if type(cases) not in (list, tuple) or not 1 <= len(cases) <= len(CASE_NAMES):
        raise ValueError('one to five explicit recovery cases required')
    if any(type(c) is not str or c not in CASE_NAMES for c in cases) or len(set(cases)) != len(cases):
        raise ValueError('unique declared recovery cases required')
    return tuple(cases)


def expected_wrenches(cases, steps, accepted, nbody, body_id):
    """Full per-world force matrices: only accepted pre-Euler steps 500..509."""
    cases = cases_checked(cases)
    if (type(steps) is not list or len(steps) != len(cases)
            or any(type(s) is not int or not 0 <= s <= TOTAL_STEPS for s in steps)):
        raise ValueError('bounded exact integer recovery counters required')
    if type(accepted) is not list or len(accepted) != len(cases) or any(type(v) is not bool for v in accepted):
        raise ValueError('exact accepted recovery mask required')
    xfrc, qfrc = [], []
    for name, step, live in zip(cases, steps, accepted):
        active = name if live and ONSET_STEP <= step < ONSET_STEP+PULSE_STEPS else 'zero-wrench'
        x, q = force.expected_wrench(active, nbody, body_id)
        xfrc.extend(x); qfrc.extend(q)
    return xfrc, qfrc


def validate_wrenches(xfrc, qfrc, cases, steps, accepted, nbody, body_id):
    expected_x, expected_q = expected_wrenches(cases, steps, accepted, nbody, body_id)
    force._numeric_tree(xfrc, (len(cases), nbody, 6), 'recovery xfrc_applied')
    force._numeric_tree(qfrc, (len(cases), force.NV), 'recovery qfrc_applied')
    if xfrc != expected_x or qfrc != expected_q:
        raise ValueError('exact declared recovery wrench required')
    return True


def declaration(source, plant):
    # Reuse the *binding* validator, not the D0 timing/admission protocol.
    base = force.plan(source, plant)
    return dict(protocol=PROTOCOL, source=source, plant=deepcopy(base['plant']),
        plant_sha256=base['plant_sha256'], source_plant_sha256=base['source_plant_sha256'],
        training_source=TRAINING_SOURCE, training_seed=TRAINING_SEED,
        checkpoint_iteration=ITERATION, checkpoint_file=CHECKPOINT_FILE,
        checkpoint_sha256=CHECKPOINT_SHA256, evaluation_seed=EVALUATION_SEED,
        cases=list(CASE_NAMES), worlds_per_case=1, actor_dim=44, action_dim=10,
        actor_device='cpu', dt=force.DT, policy_ticks=POLICY_TICKS,
        onset_physics_step=ONSET_STEP, pulse_physics_steps=PULSE_STEPS,
        pulse_newtons=2., impulse_newton_seconds=.04, total_physics_steps=TOTAL_STEPS,
        frame=force.FRAME, wrench_order=force.ORDER, application=force.APPLICATION,
        pre_forward_pulse_only=True, clear_before_unforced_post=True,
        optimizer_steps=0, first_attempt_only=True, auto_reset=False,
        actor_observation_expanded=False, raw_perception=False,
        final_tilt_gate_rad=.0873, final_planar_speed_gate_mps=.03,
        maximum_displacement_gate_m=.02, soft_limit_fraction_gate=.01,
        final_height_gate_m=.105, both_feet_support_fraction_gate=.99,
        probe_case=PROBE_CASE, probe_child_seconds=PROBE_CHILD_SECONDS,
        probe_service_seconds=PROBE_SERVICE_SECONDS, closeout_seconds=CLOSEOUT_SECONDS,
        margin_seconds=MARGIN_SECONDS, cutoff_unix=CUTOFF, **FALSE_FLAGS)


def binding(source, plant, case, device, checkpoint_identity, cpu_math_profile):
    cases_checked([case])
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('explicit recovery backend required')
    if (type(checkpoint_identity) is not dict
            or checkpoint_identity.get('source') != TRAINING_SOURCE
            or type(checkpoint_identity.get('training_seed')) is not int
            or checkpoint_identity['training_seed'] != TRAINING_SEED
            or type(checkpoint_identity.get('iteration')) is not int
            or checkpoint_identity['iteration'] != ITERATION
            or checkpoint_identity.get('purpose') != 'lean-replication'):
        raise ValueError('preselected frozen recovery checkpoint identity required')
    if type(cpu_math_profile) is not dict:
        raise ValueError('explicit portable actor profile required')
    d = declaration(source, plant)
    return dict(protocol=TRACE_PROTOCOL, source=source, worlds=1, capture_device=device,
        declaration_sha256=sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        case=case, checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_identity=deepcopy(checkpoint_identity), cpu_math_profile=deepcopy(cpu_math_profile))
