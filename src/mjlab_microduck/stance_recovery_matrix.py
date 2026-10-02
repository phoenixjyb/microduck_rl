"""CPU-only five-case frozen baseline comparison, never recovery admission.

Every receipt comes from byte-authenticated whole-case scoring. Only the exact
unforced approach and the first pre-force control proposal enter the matched
prefix hash. Reactive actions after the first pulse step are intentionally free
to differ. No fresh whole-trajectory physics re-simulation is claimed.
"""
from hashlib import sha256
import io
import math
import os
import struct

import torch

from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_trace as evidence
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1d-frozen-five-case-baseline-v1'
PREFIX_TICKS = contract.ONSET_STEP//10


def prefix_hash(value):
    """Typed, bit-exact CPU tree digest, independent of storage aliasing."""
    digest = sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            require(v.device.type == 'cpu' and v.dtype in (torch.float32, torch.long, torch.bool)
                    and bool(torch.isfinite(v).all()), 'finite typed CPU prefix tensor')
            digest.update(b'tensor\0'+str(v.dtype).encode()+b'\0'+canonical(list(v.shape)).encode()+b'\0')
            digest.update(v.detach().contiguous().numpy().tobytes())
        elif type(v) is dict:
            require(all(type(k) is str for k in v), 'named prefix fields')
            digest.update(b'dict\0'+str(len(v)).encode()+b'\0')
            for key in sorted(v): visit(key); visit(v[key])
        elif type(v) in (list, tuple):
            digest.update(type(v).__name__.encode()+b'\0'+str(len(v)).encode()+b'\0')
            for item in v: visit(item)
        elif v is None: digest.update(b'none\0')
        elif type(v) is bool: digest.update(b'bool1\0' if v else b'bool0\0')
        elif type(v) is int: digest.update(b'int\0'+str(v).encode()+b'\0')
        elif type(v) is float:
            require(math.isfinite(v), 'finite prefix float')
            digest.update(b'float\0'+struct.pack('>d', v))
        elif type(v) is str:
            raw = v.encode('utf8'); digest.update(b'str\0'+str(len(raw)).encode()+b'\0'+raw)
        else: raise ValueError('unsupported prefix field type')
    visit(value)
    return digest.hexdigest()


def checked_case(raw, expected_sha256, checkpoint_raw, declaration):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only independent five-case checker')
    require(type(raw) is bytes and 0 < len(raw) <= contract.CAPTURE_LIMIT
            and sha256(raw).hexdigest() == expected_sha256, 'whole case byte hash before tensor loading')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    # This scorer verifies the exact declaration, checkpoint/actor, every frame,
    # BAM/FIFO accounting, planned full force arrays and pulse phases first.
    score = evidence.score(value, checkpoint_raw, declaration)
    require(score['numerical_diagnostic']['gates']['full_duration'] is True,
            'full-length case before matched-prefix comparison')
    ticks = value['payload']['ticks']; controls = value['control_evidence']['ticks']
    require(len(ticks) > PREFIX_TICKS and len(controls) == len(ticks), 'complete unforced approach')
    onset = ticks[PREFIX_TICKS]; proposal = controls[PREFIX_TICKS]['proposals'][0]
    require(onset['boundaries'][0]['physics_steps'].tolist() == [contract.ONSET_STEP]
            and proposal['before_steps'].tolist() == [contract.ONSET_STEP], 'matched exact pre-force onset')
    prefix = dict(initial=value['payload']['initial'], unforced_ticks=ticks[:PREFIX_TICKS],
        unforced_control=controls[:PREFIX_TICKS],
        onset=dict(actor_input=onset['actor_input'], actions=onset['actions'],
            boundary=onset['boundaries'][0], initial_control=controls[PREFIX_TICKS]['initial'],
            after_action=controls[PREFIX_TICKS]['after_action'], first_proposal=proposal))
    return dict(protocol=PROTOCOL, source=declaration['source'], case=score['case'],
        declaration_sha256=sha256(canonical(declaration).encode()).hexdigest(),
        checkpoint_sha256=contract.CHECKPOINT_SHA256, evaluation_seed=contract.EVALUATION_SEED,
        capture_sha256=expected_sha256, capture_bytes=len(raw),
        prefix_sha256=prefix_hash(prefix), matched_through_physics_step=contract.ONSET_STEP,
        first_pre_force_action_included=True, post_push_actions_compared=False,
        score=score, whole_trajectory_physics_resimulated=False, **contract.FALSE_FLAGS)


def compare(receipts):
    require(type(receipts) is list and [r['case'] for r in receipts] == list(contract.CASE_NAMES),
            'all five ordered fresh frozen baseline cases')
    common = ('protocol', 'source', 'declaration_sha256', 'checkpoint_sha256', 'evaluation_seed',
        'prefix_sha256', 'matched_through_physics_step', 'first_pre_force_action_included',
        'post_push_actions_compared', 'whole_trajectory_physics_resimulated')
    first = receipts[0]
    require(first['protocol'] == PROTOCOL and first['checkpoint_sha256'] == contract.CHECKPOINT_SHA256
            and first['evaluation_seed'] == contract.EVALUATION_SEED
            and first['matched_through_physics_step'] == contract.ONSET_STEP
            and first['first_pre_force_action_included'] is True
            and first['post_push_actions_compared'] is False
            and first['whole_trajectory_physics_resimulated'] is False,
            'unchanged complete comparison scope')
    for receipt in receipts:
        require(all(receipt[k] == first[k] for k in common), 'identical binding and unforced state/action prefixes')
        require(all(receipt[k] is False for k in contract.FALSE_FLAGS), 'baseline never admits a skill or learner')
        score = receipt['score']
        require(score['protocol'] == contract.PROTOCOL and score['case'] == receipt['case']
                and all(score[k] is False for k in contract.FALSE_FLAGS)
                and score['numerical_diagnostic']['gates']['full_duration'] is True
                and type(score['numerical_diagnostic']['candidate_pass']) is bool
                and score['pulse']['complete_pulse_delivery'] is True
                and score['pulse']['checked_physics_steps'] == contract.TOTAL_STEPS
                and score['pulse']['pulse_window_steps'] == contract.PULSE_STEPS
                and score['pulse']['complete_pulse_window_phase_checks'] is True
                and score['pulse']['exact_scheduled_forces_checked'] is True
                and score['pulse']['unforced_post_arrays_checked'] is True
                and score['whole_trajectory_physics_resimulated'] is False
                and score['actor_replay_max_abs_error'] == 0., 'whole-case verified prefix receipt')
    passed = all(r['score']['numerical_diagnostic']['candidate_pass'] is True for r in receipts)
    return dict(protocol=PROTOCOL, source=first['source'], cases=list(contract.CASE_NAMES),
        decision='frozen-five-case-baseline-passed' if passed else 'frozen-five-case-baseline-rejected',
        all_prefixes_identical=True, zero_wrench_control_passed=receipts[0]['score']['numerical_diagnostic']['candidate_pass'],
        held_out_randomized_recovery_verified=False, cases_checked=len(receipts),
        prefix_sha256=first['prefix_sha256'], whole_trajectory_physics_resimulated=False,
        receipts=receipts, **contract.FALSE_FLAGS)
