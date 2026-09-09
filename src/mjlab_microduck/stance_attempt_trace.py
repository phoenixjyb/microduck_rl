"""Owned first-attempt trajectory capture/replay; not a checkpoint admission CLI.

Caller-supplied identity hashes bind a trace but do not attest model restoration,
live source, reset provenance or GPU supervision. Those remain launcher duties.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import re

import torch

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.stance_evaluation import score_attempt
from mjlab_microduck.stance_transition import PhysicsState, EPISODE_STEPS, physical_failures

PROTOCOL = 'football-b1n-first-attempt-trace-v1'
SEEDS = (541, 547, 557)
CHECKPOINTS = (128, 256, 384, 511)
MAX_TRACE_BYTES = 512*1024*1024
STATE_KEYS = set(PhysicsState.__dataclass_fields__)
FRAME_KEYS = {'physics_steps', 'qpos', 'qvel', 'soft_limit_mask', 'state', 'observation'}
TICK_KEYS = {'actor_input', 'actions', 'boundaries', 'terminal_records', 'reward',
             'terminated', 'timed_out', 'episode_steps', 'executed_steps', 'live'}


def owned(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, PhysicsState): return owned(asdict(value))
    if isinstance(value, dict): return {k: owned(v) for k, v in value.items()}
    if isinstance(value, list): return [owned(v) for v in value]
    return deepcopy(value)


def tensor(value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.shape == shape and value.dtype == dtype, 'trace tensor layout: '+label)
    if value.is_floating_point(): require(torch.isfinite(value).all(), 'nonfinite trace: '+label)


def validate_binding(binding):
    require(set(binding) == {'protocol', 'source', 'runtime_sha256', 'checkpoint_sha256',
        'launch_sha256', 'checkpoint_iteration', 'evaluation_seed', 'worlds', 'capture_device'},
        'exact trace binding fields')
    require(binding['protocol'] == PROTOCOL, 'trace binding protocol')
    for key in ('source', 'runtime_sha256', 'checkpoint_sha256', 'launch_sha256'):
        require(type(binding[key]) is str and re.fullmatch(
            '[0-9a-f]{'+('40' if key == 'source' else '64')+'}', binding[key]) is not None,
            'trace binding hash: '+key)
    require(type(binding['worlds']) is int and 1 <= binding['worlds'] <= 128, 'bounded trace worlds')
    require(type(binding['evaluation_seed']) is int and binding['evaluation_seed'] in SEEDS,
            'predeclared evaluation seed')
    require(type(binding['checkpoint_iteration']) is int and binding['checkpoint_iteration'] in CHECKPOINTS,
            'predeclared checkpoint iteration')
    require(binding['capture_device'] in ('cpu', 'cuda:0'), 'explicit capture device')


def validate_frame(frame, n):
    require(set(frame) == FRAME_KEYS and set(frame['state']) == STATE_KEYS
            and set(frame['observation']) == {'actor', 'critic'}, 'exact frame fields')
    tensor(frame['physics_steps'], (n,), torch.int64, 'steps')
    require(((frame['physics_steps'] >= 0) & (frame['physics_steps'] <= EPISODE_STEPS)).all(), 'bounded counters')
    for key, shape, dtype in (('qpos', (n, 21), torch.float32),
                             ('qvel', (n, 20), torch.float32),
                             ('soft_limit_mask', (n, 14), torch.bool)):
        tensor(frame[key], shape, dtype, key)
    state = PhysicsState(**frame['state']); state.validate(n, torch.device('cpu'))
    require(all(v.dtype in (torch.float32, torch.bool) for v in frame['state'].values()), 'float32 runtime state')
    require(torch.equal(state.height, frame['qpos'][:, 2])
            and torch.equal(state.root_velocity, frame['qvel'][:, :3]), 'physical frame consistency')
    for key, size in (('actor', 44), ('critic', 50)):
        tensor(frame['observation'][key], (n, size), torch.float32, key)
    require(torch.equal(frame['observation']['actor'], frame['observation']['critic'][:, :44]),
            'actor/critic prefix consistency')


def same_rows(left, right, rows, *, observation=False):
    for key in ('qpos', 'qvel', 'soft_limit_mask'):
        require(torch.equal(left[key][rows], right[key][rows]), 'unchanged physical boundary: '+key)
    for key in STATE_KEYS:
        require(torch.equal(left['state'][key][rows], right['state'][key][rows]), 'unchanged state boundary: '+key)
    if observation:
        for key in ('actor', 'critic'):
            require(torch.equal(left['observation'][key][rows], right['observation'][key][rows]),
                    'unchanged closed observation')


def terminal_matches(record, frame, row):
    canonical(record)  # All contact/observation fields must also be JSON-finite.
    require(record['world_id'] == row and type(record['world_id']) is int
            and record['physics_step'] == int(frame['physics_steps'][row])
            and type(record['physics_step']) is int, 'first terminal identity/counter')
    require(type(record['terminated']) is bool and type(record['timed_out']) is bool
            and record['terminated'] != record['timed_out'], 'exclusive terminal outcome')
    require(record['checkpoint_admitted'] is False and record['trajectory_continuity_validated'] is False,
            'raw terminal is not admission')
    for key in ('qpos', 'qvel'):
        require(record[key] == frame[key][row].tolist(), 'terminal physical evidence mismatch')
    require(record['state'] == {k: v[row].tolist() for k, v in frame['state'].items()}, 'terminal state mismatch')
    require(record['observation'] == {k: v[row].tolist() for k, v in frame['observation'].items()},
            'terminal observation mismatch')
    # Preserve full contact records. Geometry/force replay needs the separately
    # bound compiled-plant contract; do not pretend finite JSON proves contacts.
    require(type(record['contacts']) is dict and 'worldid' in record['contacts']
            and all(type(v) is int and v == row for v in record['contacts']['worldid']),
            'terminal contacts belong to their world')
    proposed = record['rejected_proposed_torque_nm']
    if proposed is not None:
        torque = torch.tensor(proposed, dtype=torch.float32)
        tensor(torque, (14,), torch.float32, 'proposed torque')
        require((torque.abs() > .36).any(), 'actual rejected torque')
    failed = bool(physical_failures(PhysicsState(**frame['state']), frame['physics_steps'])[row]) or proposed is not None
    require(record['terminated'] == failed and (failed or record['physics_step'] == EPISODE_STEPS),
            'terminal must be first physical/proposed failure or exact timeout')


class FirstAttemptTrace:
    """One unreset batch. Failed append is atomic and permanently faults capture.

    Keep the policy input BEFORE calling runtime.step; that method refreshes the
    realized correction in its initial boundary. Repeated physical boundaries
    across ticks are reconciled, not double-counted. No simulator calls occur here.
    """

    def __init__(self, binding, initial):
        validate_binding(binding)
        self.binding = deepcopy(binding); self.n = binding['worlds']; self.faulted = False
        require(str(initial['qpos'].device) == binding['capture_device'], 'observed capture device')
        frame = owned(initial); validate_frame(frame, self.n)
        require(not frame['physics_steps'].any(), 'first attempt must start at zero')
        require(not physical_failures(PhysicsState(**frame['state']), frame['physics_steps']).any(), 'valid initial state')
        self.initial = frame; self.last = frame; self.ticks = []
        self.terminals = [None]*self.n

    def append(self, result, actor_input, actions):
        require(not self.faulted, 'faulted trace requires closeout')
        try:
            require(len(self.ticks) < EPISODE_STEPS//10, 'bounded first-attempt tick count')
            require(any(t is None for t in self.terminals), 'all attempts already closed')
            require(str(result['boundaries'][0]['qpos'].device) == self.binding['capture_device'], 'observed capture device')
            tick = owned({k: result[k] for k in TICK_KEYS-{'actor_input', 'actions'}})
            tick.update(actor_input=owned(actor_input), actions=owned(actions))
            self._validate_tick(tick)
            self.ticks.append(tick); self.last = tick['boundaries'][-1]
            self.terminals = deepcopy(tick['terminal_records'])
        except Exception:
            self.faulted = True
            raise

    def _validate_tick(self, tick):
        n = self.n
        require(set(tick) == TICK_KEYS, 'exact tick fields')
        tensor(tick['actor_input'], (n, 44), torch.float32, 'policy input')
        tensor(tick['actions'], (n, 10), torch.float32, 'policy action')
        require(torch.equal(tick['actor_input'], self.last['observation']['actor']), 'actual pre-action policy input')
        for key in ('terminated', 'timed_out', 'live'):
            tensor(tick[key], (n,), torch.bool, key)
        for key in ('episode_steps', 'executed_steps'):
            tensor(tick[key], (n,), torch.int64, key)
        tensor(tick['reward'], (n,), torch.float32, 'reward')
        frames = tick['boundaries']; terminals = tick['terminal_records']
        require(type(frames) is list and 1 <= len(frames) <= 11 and len(terminals) == n, 'bounded tick evidence')
        for f in frames: validate_frame(f, n)
        require(torch.equal(frames[0]['physics_steps'], self.last['physics_steps']), 'tick boundary counter continuity')
        all_rows = torch.ones(n, dtype=torch.bool)
        same_rows(self.last, frames[0], all_rows)
        require(torch.equal(self.last['observation']['actor'][:, :34], frames[0]['observation']['actor'][:, :34])
                and torch.equal(self.last['observation']['critic'][:, 44:], frames[0]['observation']['critic'][:, 44:]),
                'only realized correction may change before physics')
        closed = torch.tensor([r is not None for r in self.terminals])
        same_rows(self.last, frames[0], closed, observation=True)
        for before, after in zip(frames, frames[1:]):
            delta = after['physics_steps']-before['physics_steps']
            require(((delta == 0) | (delta == 1)).all() and delta.any(), 'no skipped/reset/global duplicate substeps')
            same_rows(before, after, delta == 0, observation=True)
        end = frames[-1]
        require(torch.equal(tick['episode_steps'], end['physics_steps'])
                and torch.equal(tick['executed_steps'], end['physics_steps']-self.last['physics_steps']), 'executed counter agreement')
        failures = [physical_failures(PhysicsState(**f['state']), f['physics_steps']) for f in frames]
        for row, record in enumerate(terminals):
            old = self.terminals[row]
            if old is not None:
                require(record == old and tick['executed_steps'][row] == 0 and tick['reward'][row] == 0
                        and not tick['terminated'][row] and not tick['timed_out'][row], 'closed terminal immutable and unpaid')
            if record is not None:
                terminal_matches(record, end, row)
                require(not tick['live'][row], 'terminal cannot be live')
                if old is None:
                    require(bool(tick['terminated'][row]) == record['terminated']
                            and bool(tick['timed_out'][row]) == record['timed_out'], 'new terminal flags')
            else:
                require(tick['live'][row] and not tick['terminated'][row] and not tick['timed_out'][row]
                        and tick['executed_steps'][row] == 10, 'unfinished world must execute full tick')
            for f, failed in zip(frames, failures):
                if bool(failed[row]) or int(f['physics_steps'][row]) == EPISODE_STEPS:
                    require(record is not None and int(f['physics_steps'][row]) == record['physics_step'],
                            'no frames after first terminal')

    def payload(self):
        require(not self.faulted, 'faulted trace cannot publish success')
        return owned(dict(binding=self.binding, initial=self.initial, ticks=self.ticks))


def replay(payload, expected_binding):
    """CPU rescore against caller's independently selected binding, never admit PPO.

    CUDA provenance is declared metadata during replay, not CPU re-execution on
    CUDA. Hash verification of serialized bytes belongs to the artifact reader.
    """
    require(set(payload) == {'binding', 'initial', 'ticks'} and payload['binding'] == expected_binding,
            'expected trace binding mismatch')
    validate_binding(expected_binding)
    # Rehydrate the pure CPU checker without asserting that CPU tensors run CUDA.
    cpu_binding = {**expected_binding, 'capture_device': 'cpu'}
    trace = FirstAttemptTrace(cpu_binding, payload['initial'])
    require(type(payload['ticks']) is list, 'ordered tick list')
    for tick in payload['ticks']:
        require(set(tick) == TICK_KEYS, 'exact replay tick fields')
        trace.append(tick, tick['actor_input'], tick['actions'])
    frames = [trace.initial]+[f for t in trace.ticks for f in t['boundaries'][1:]]
    counters = torch.stack([f['physics_steps'] for f in frames])
    states = {k: torch.stack([f['state'][k] for f in frames]) for k in STATE_KEYS}
    positions = torch.stack([f['qpos'][:, :3] for f in frames])
    soft = torch.stack([f['soft_limit_mask'] for f in frames])
    attempts = []
    for row, terminal in enumerate(trace.terminals):
        keep = torch.cat((torch.ones(1, dtype=torch.bool), counters[1:, row] != counters[:-1, row]))
        proposed = None if terminal is None else terminal['rejected_proposed_torque_nm']
        scored = score_attempt(PhysicsState(**{k: v[keep, row] for k, v in states.items()}),
            positions[keep, row], soft[keep, row], counters[keep, row],
            rejected_proposed_torque=None if proposed is None else torch.tensor(proposed, dtype=torch.float32))
        require(scored['complete_first_attempt'] == (terminal is not None), 'terminal/complete score agreement')
        attempts.append(dict(world_id=row, **scored))
    return dict(protocol=PROTOCOL, binding=deepcopy(expected_binding), attempts=attempts,
        complete_attempts=sum(a['complete_first_attempt'] for a in attempts),
        numerical_passes=sum(a['candidate_pass'] for a in attempts),
        trajectory_continuity_validated=True, provenance_validated=False,
        checkpoint_admitted=False, learned_stance_accepted=False, physical_motion_authorized=False)


def encode(payload, expected_binding):
    """Create a CPU tensor artifact only after replay; caller durably stores bytes."""
    score = replay(payload, expected_binding)
    buffer = io.BytesIO()
    torch.save(owned(payload), buffer)
    raw = buffer.getvalue()
    require(len(raw) <= MAX_TRACE_BYTES, 'bounded serialized trace')
    return raw, dict(sha256=sha256(raw).hexdigest(), bytes=len(raw), score=score)


def verify(raw, expected_sha256, expected_binding):
    """Hash exact bytes BEFORE weights-only CPU deserialization and full replay.

    Expected digest and binding must come from an independently verified launch/
    artifact manifest, not from an untrusted receipt adjacent to this payload.
    No dynamic globals, CUDA deserialization, checkpoint restore or acceptance.
    """
    require(type(raw) is bytes and 0 < len(raw) <= MAX_TRACE_BYTES, 'bounded serialized trace')
    require(sha256(raw).hexdigest() == expected_sha256, 'retained trace hash mismatch')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return replay(value, expected_binding)
