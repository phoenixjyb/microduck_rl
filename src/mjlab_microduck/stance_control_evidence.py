"""Replay opt-in motor command/commit evidence without simulating or training.

Checks action slew, delay, proposal gates and state commits. Recorded BAM torque
and friction remain solver/model outputs, not independently recomputed here.
"""
from hashlib import sha256
import io

import torch

from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck.first_attempt_smoke import require
from mjlab_microduck.stance_lesson_contract import LEG_IDS
from mjlab_microduck.stance_transition import PhysicsState, physical_failures, EPISODE_STEPS

PROTOCOL = 'football-b1n-control-evidence-v1'
LIMIT = 512*1024*1024
SHAPES = dict(correction=(10,), target=(14,), queue=(3, 14), previous=(14,),
              voltage=(1,), kp=(1,), friction=(20,), damping=(20,), ctrl=(14,))
COMMAND = {'position_target', 'velocity_target', 'effort_target', 'pos', 'vel'}


def same(left, right, label, rows=None):
    if rows is not None: left, right = left[rows], right[rows]
    require(torch.equal(left, right), 'control evidence mismatch: '+label)


def close(left, right, label):
    require(torch.allclose(left, right, atol=1e-6, rtol=1e-5), 'control arithmetic mismatch: '+label)


def state(value, n):
    require(set(value) == set(SHAPES), 'exact control state fields')
    for key, shape in SHAPES.items(): trace.tensor(value[key], (n, *shape), torch.float32, key)
    require((value['friction'] >= 0).all() and (value['damping'] >= 0).all(), 'nonnegative motor fields')
    require((value['previous'].abs() <= .36).all() and (value['ctrl'].abs() <= .36).all(), 'committed torque gate')
    require(((value['voltage'] >= 6) & (value['voltage'] <= 7.5)).all(), 'committed voltage range')
    same(value['kp'], torch.full((n, 1), 200., device='cpu'), 'nominal gain')


def available(frame, live):
    return live & ~physical_failures(PhysicsState(**frame['state']), frame['physics_steps']) & (frame['physics_steps'] < EPISODE_STEPS)


def replay(evidence, payload, plant):
    """Requires the separately validated physical trace and compiled plant."""
    require(set(evidence) == {'protocol', 'binding', 'ticks'} and evidence['protocol'] == PROTOCOL
            and evidence['binding'] == payload['binding'], 'control binding')
    require(type(evidence['ticks']) is list and len(evidence['ticks']) == len(payload['ticks'])
            and 0 < len(evidence['ticks']) <= 250, 'control tick coverage')
    n = payload['binding']['worlds']
    nominal = torch.tensor(plant['initial_qpos'], dtype=torch.float32, device='cpu')[plant['qids']]
    ranges = torch.tensor(plant['ranges'], dtype=torch.float32, device='cpu')
    lo, hi = ranges.mean(1)-.45*(ranges[:, 1]-ranges[:, 0]), ranges.mean(1)+.45*(ranges[:, 1]-ranges[:, 0])
    ids = list(LEG_IDS)
    lower, upper = (lo[ids]-nominal[ids]).clamp(min=-.2), (hi[ids]-nominal[ids]).clamp(max=.2)
    last = {key: torch.zeros((n, *shape), dtype=torch.float32, device='cpu') for key, shape in SHAPES.items()}
    last.update(target=nominal.expand(n, 14).clone(), queue=nominal.expand(n, 3, 14).clone(),
                voltage=torch.full((n, 1), 7.5, device='cpu'), kp=torch.full((n, 1), 200., device='cpu'))
    live = torch.ones(n, dtype=torch.bool, device='cpu'); count = 0
    for tick, control in zip(payload['ticks'], evidence['ticks']):
        require(set(control) == {'initial', 'after_action', 'proposals', 'final'}, 'exact control tick fields')
        for key in ('initial', 'after_action', 'final'): state(control[key], n)
        for key in SHAPES: same(control['initial'][key], last[key], 'continuous '+key)
        close(tick['actor_input'][:, 34:], last['correction']/.2, 'previous actor correction')
        desired = (.2*tick['actions'].clamp(-1, 1)).clamp(lower, upper)
        correction = (last['correction']+(desired-last['correction']).clamp(-.02, .02)).clamp(lower, upper)
        correction = torch.where(live[:, None], correction, last['correction'])
        target = nominal.expand(n, 14).clone(); target[:, ids] += correction
        after = control['after_action']
        close(after['correction'], correction, 'realized slew/clipping')
        close(after['target'], target, 'nominal head and corrected leg target')
        for key in SHAPES:
            if key not in ('correction', 'target'): same(after[key], last[key], 'action does not change '+key)
            same(after[key], last[key], 'closed action '+key, ~live)
        for frame in tick['boundaries']:
            close(frame['observation']['actor'][:, 34:], after['correction']/.2, 'within-tick correction')
        last = after; index = 0
        require(type(control['proposals']) is list and len(control['proposals']) <= 10, 'bounded motor proposals')
        for proposal in control['proposals']:
            count += 1
            require(set(proposal) == {'before_steps', 'live', 'command', 'torque_nm', 'accepted', 'rejected', 'committed'},
                    'exact proposal fields')
            frame = tick['boundaries'][index]; current = available(frame, live)
            require(current.any(), 'no proposal after all worlds close')
            trace.tensor(proposal['before_steps'], (n,), torch.int64, 'proposal counters')
            same(proposal['before_steps'], frame['physics_steps'], 'proposal physical boundary')
            for key in ('live', 'accepted', 'rejected'): trace.tensor(proposal[key], (n,), torch.bool, key)
            same(proposal['live'], current, 'pre-proposal live mask')
            command = proposal['command']; require(set(command) == COMMAND, 'exact motor command')
            for key in COMMAND: trace.tensor(command[key], (n, 14), torch.float32, key)
            same(command['position_target'], last['queue'][:, 0], 'three-step delayed target')
            same(command['pos'], frame['qpos'][:, plant['qids']], 'motor position input')
            same(command['vel'], frame['qvel'][:, plant['dofs']], 'motor velocity input')
            require(not command['velocity_target'].any() and not command['effort_target'].any(), 'position-only motor command')
            trace.tensor(proposal['torque_nm'], (n, 14), torch.float32, 'proposed torque')
            rejected = current & (proposal['torque_nm'].abs().amax(1) > .36)
            accepted = current & ~rejected
            same(proposal['accepted'], accepted, 'accepted torque mask'); same(proposal['rejected'], rejected, 'rejected torque mask')
            committed = proposal['committed']; state(committed, n)
            for key in SHAPES: same(committed[key], last[key], 'unaccepted '+key, ~accepted)
            for key in ('correction', 'target'): same(committed[key], last[key], 'substep '+key)
            shifted = torch.cat((last['queue'][:, 1:], last['target'][:, None]), 1)
            same(committed['queue'], torch.where(accepted[:, None, None], shifted, last['queue']), 'accepted FIFO shift')
            for key in ('ctrl', 'previous'):
                same(committed[key], torch.where(accepted[:, None], proposal['torque_nm'], last[key]), 'accepted '+key)
            voltage = (7.5-.1*last['previous'].abs().sum(1, keepdim=True)).clamp(min=6.)
            close(committed['voltage'], torch.where(accepted[:, None], voltage, last['voltage']), 'voltage sag history')
            free = [j for j in range(20) if j not in plant['dofs']]
            for key in ('friction', 'damping'): same(committed[key][:, free], last[key][:, free], 'uncontrolled '+key)
            for row in rejected.nonzero().flatten().tolist():
                terminal = tick['terminal_records'][row]
                require(terminal is not None and terminal['physics_step'] == int(frame['physics_steps'][row])
                        and terminal['rejected_proposed_torque_nm'] == proposal['torque_nm'][row].tolist(), 'rejected proposal terminal binding')
            last = committed; live = accepted
            if accepted.any():
                require(index+1 < len(tick['boundaries']), 'accepted proposal requires physical step')
                next_frame = tick['boundaries'][index+1]
                same(next_frame['physics_steps']-frame['physics_steps'], accepted.long(), 'accepted physical increment')
                close(next_frame['state']['torque'][accepted], committed['ctrl'][accepted], 'applied torque/committed command')
                index += 1
                live = available(next_frame, live)
        require(index == len(tick['boundaries'])-1, 'every physical step has a motor proposal')
        live = available(tick['boundaries'][-1], live)
        same(live, tick['live'], 'final live mask')
        for key in SHAPES: same(control['final'][key], last[key], 'final control '+key)
    return dict(action_slew_checked=True, delayed_motor_targets_checked=True, motor_commit_masks_checked=True,
                voltage_history_checked=True, motor_proposals_checked=count, bam_outputs_recomputed=False)


def encode(evidence, payload, plant):
    receipt = replay(evidence, payload, plant)
    buffer = io.BytesIO(); torch.save(trace.owned(evidence), buffer)
    raw = buffer.getvalue(); require(len(raw) <= LIMIT, 'bounded control evidence')
    return raw, receipt


def verify(raw, expected_sha256, payload, plant):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == expected_sha256,
            'control artifact byte identity')
    return replay(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), payload, plant)
