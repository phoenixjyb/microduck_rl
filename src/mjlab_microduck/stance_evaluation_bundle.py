"""Exclusive hash-bound stance evidence bundles; no live launch or skill admission."""

from copy import deepcopy
from hashlib import sha256
import io
import os

import torch

from mjlab_microduck import foundation_command_campaign as files
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_control_evidence as control
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = 'football-b1n-evaluation-bundle-v3'
LAUNCH_PROTOCOL = 'football-b1n-evaluation-inputs-v3'
NAMES = {'launch.json', 'runtime.json', 'checkpoint.pt', 'trace.pt', 'control.pt', 'restore.json', 'score.json'}
ATOL, RTOL = 1e-6, 1e-5


def launch_bytes(binding_without_launch_sha, checkpoint_identity):
    """Caller binds the returned byte hash into the trace; no self-referential SHA."""
    require('launch_sha256' not in binding_without_launch_sha, 'launch hash assigned after encoding')
    trace.validate_binding({**binding_without_launch_sha, 'launch_sha256': '0'*64})
    return (canonical(dict(protocol=LAUNCH_PROTOCOL, binding=binding_without_launch_sha,
        checkpoint_identity=checkpoint_identity, purpose='diagnostic-evaluation-only',
        physical_motion_authorized=False))+'\n').encode()


def checked_inputs(binding, cp_raw, cp_identity, runtime_raw, launch_raw):
    trace.validate_binding(binding)
    for raw, expected in ((cp_raw, binding['checkpoint_sha256']),
                          (runtime_raw, binding['runtime_sha256']), (launch_raw, binding['launch_sha256'])):
        require(type(raw) is bytes and 0 < len(raw) <= checkpoint.LIMIT and sha256(raw).hexdigest() == expected,
                'bound input bytes mismatch')
    expected_launch = launch_bytes({k: v for k, v in binding.items() if k != 'launch_sha256'}, cp_identity)
    require(launch_raw == expected_launch, 'exact launch/checkpoint binding')
    runtime = files.parse(runtime_raw)
    compiled = plant.checked_runtime(runtime, binding['source'])
    require(cp_identity['iteration'] == binding['checkpoint_iteration'], 'checkpoint/trace iteration mismatch')
    loader = (checkpoint.load_eager_diagnostic if binding['protocol'] == trace.EAGER_PROTOCOL
              else checkpoint.load_evaluation)
    actor, receipt = loader(cp_raw, binding['checkpoint_sha256'], cp_identity)
    return actor, receipt, compiled


def actor_replay(payload, actor, score):
    maximum = 0.
    for tick in payload['ticks']:
        predicted = checkpoint.infer(actor, tick['actor_input'])
        recorded = tick['actions']
        require(torch.allclose(predicted, recorded, atol=ATOL, rtol=RTOL), 'recorded action disagrees with restored actor')
        maximum = max(maximum, float((predicted-recorded).abs().max()))
    return {**score, 'strict_checkpoint_checked': True, 'deterministic_actor_replay_checked': True,
            'actor_replay_max_abs_error': maximum, 'actor_replay_atol': ATOL, 'actor_replay_rtol': RTOL}


def write_bundle(directory, payload, *, control_evidence, binding, checkpoint_raw, checkpoint_identity, runtime_raw, launch_raw):
    """Publish manifest last, fsync every file; retain any partial failed directory.

    Caller must own the output path and retain the returned manifest hash outside
    the bundle. Never resumes/overwrites an existing directory or alters inputs.
    """
    actor, restore, compiled = checked_inputs(binding, checkpoint_raw, checkpoint_identity, runtime_raw, launch_raw)
    raw, receipt = trace.encode(payload, binding)
    score = {**actor_replay(payload, actor, receipt['score']), **plant.check_trace(payload, compiled)}
    control_raw, control_receipt = control.encode(control_evidence, payload, compiled)
    score.update(control_receipt)
    data = {'launch.json': launch_raw, 'runtime.json': runtime_raw, 'checkpoint.pt': checkpoint_raw,
            'trace.pt': raw, 'control.pt': control_raw,
            'restore.json': (canonical(restore)+'\n').encode(), 'score.json': (canonical(score)+'\n').encode()}
    directory = files.native._plain_path(directory)
    directory.mkdir(exist_ok=False)
    files.native._fsync_dir(directory.parent)
    manifest = dict(protocol=PROTOCOL, binding=deepcopy(binding), files={},
                    status='retained-diagnostic-only', checkpoint_admitted=False, physical_motion_authorized=False)
    for name, content in data.items():
        with (directory/name).open('xb') as target:
            target.write(content); target.flush(); os.fsync(target.fileno())
        files.native._fsync_dir(directory)
        manifest['files'][name] = dict(sha256=sha256(content).hexdigest(), bytes=len(content))
    files.write_json(directory/'manifest.json', manifest)
    return sha256(files.file_bytes(directory/'manifest.json')).hexdigest(), score


def verify_bundle(directory, expected_manifest_sha256, *, binding, checkpoint_identity):
    """Read-only rehash, strict restore, action replay and numerical rescoring."""
    directory = files.native._plain_path(directory)
    manifest_raw = files.file_bytes(directory/'manifest.json', limit=1024*1024)
    require(sha256(manifest_raw).hexdigest() == expected_manifest_sha256, 'independent manifest hash mismatch')
    manifest = files.parse(manifest_raw)
    require(set(manifest) == {'protocol', 'binding', 'files', 'status', 'checkpoint_admitted', 'physical_motion_authorized'}
            and manifest['protocol'] == PROTOCOL and manifest['binding'] == binding
            and manifest['status'] == 'retained-diagnostic-only'
            and manifest['checkpoint_admitted'] is False and manifest['physical_motion_authorized'] is False,
            'exact diagnostic manifest')
    require(set(manifest['files']) == NAMES and {p.name for p in directory.iterdir()} == NAMES | {'manifest.json'},
            'exact bundle inventory')
    data = {}
    for name in sorted(NAMES):
        limit = control.LIMIT if name == 'control.pt' else trace.MAX_TRACE_BYTES if name == 'trace.pt' else checkpoint.LIMIT
        data[name] = files.file_bytes(directory/name, limit=limit)
        require(manifest['files'][name] == dict(sha256=sha256(data[name]).hexdigest(), bytes=len(data[name])),
                'bundle byte identity mismatch: '+name)
    actor, restore, compiled = checked_inputs(binding, data['checkpoint.pt'], checkpoint_identity, data['runtime.json'], data['launch.json'])
    require(files.parse(data['restore.json']) == restore, 'strict restoration receipt mismatch')
    score = trace.verify(data['trace.pt'], manifest['files']['trace.pt']['sha256'], binding)
    payload = torch.load(io.BytesIO(data['trace.pt']), map_location='cpu', weights_only=True)
    score = {**actor_replay(payload, actor, score), **plant.check_trace(payload, compiled)}
    score.update(control.verify(data['control.pt'], manifest['files']['control.pt']['sha256'], payload, compiled))
    require(canonical(files.parse(data['score.json'])) == canonical(score), 'recomputed score mismatch')
    return score
