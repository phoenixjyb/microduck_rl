"""Owned temporary artifacts from a short CPU fixture, never trained acceptance."""
from copy import deepcopy
from hashlib import sha256
import pytest
import torch
from mjlab_microduck import stance_checkpoint as cp
from mjlab_microduck import stance_attempt_trace as trace
from mjlab_microduck import stance_evaluation_bundle as bundle
from mjlab_microduck.first_attempt_smoke import canonical


@pytest.fixture(scope='module')
def inputs():
    import mjlab
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    actor, critic = cp.fresh_models(521)
    meta = dict(protocol=cp.PROTOCOL, source='a'*40, runtime_sha256='b'*64,
        training_launch_sha256='c'*64, purpose='pilot', training_seed=521, worlds=512,
        iteration=128, initial_state_sha256=cp.state_hash(cp.states_of(actor, critic)),
        architecture=deepcopy(cp.ARCHITECTURE))
    raw = cp.encode(actor, critic, meta)
    # Source/runtime values below are explicit synthetic provenance fixtures.
    runtime = (canonical({'source': 'd'*40, 'synthetic': True})+'\n').encode()
    binding = dict(protocol=trace.PROTOCOL, source='d'*40, runtime_sha256=sha256(runtime).hexdigest(),
        checkpoint_sha256=sha256(raw).hexdigest(), checkpoint_iteration=128,
        evaluation_seed=541, worlds=2, capture_device='cpu')
    launch = bundle.launch_bytes(binding, meta); binding['launch_sha256'] = sha256(launch).hexdigest()
    env = WarpStanceRuntime(2, device='cpu')
    recorder = trace.FirstAttemptTrace(binding, env.snapshot())
    obs = env.observations()['actor']; actions = cp.infer(actor, obs)
    recorder.append(env.step(actions), obs, actions)
    return dict(payload=recorder.payload(), binding=binding, checkpoint_raw=raw,
        checkpoint_identity=meta, runtime_raw=runtime, launch_raw=launch)


def test_exclusive_durable_bundle_and_recomputed_receipts(tmp_path, inputs):
    directory = tmp_path/'owned'
    digest, score = bundle.write_bundle(directory, **inputs)
    assert bundle.verify_bundle(directory, digest, binding=inputs['binding'],
                                checkpoint_identity=inputs['checkpoint_identity']) == score
    assert score['strict_checkpoint_checked'] and score['deterministic_actor_replay_checked']
    assert score['actor_replay_max_abs_error'] == 0
    assert not score['provenance_validated'] and not score['checkpoint_admitted']
    assert not score['learned_stance_accepted'] and not score['physical_motion_authorized']
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError): bundle.write_bundle(directory, **inputs)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('name', ['trace.pt', 'checkpoint.pt', 'runtime.json', 'launch.json', 'score.json', 'restore.json'])
def test_tampered_file_refused_before_replay(tmp_path, inputs, name):
    directory = tmp_path/'owned'; digest, _ = bundle.write_bundle(directory, **inputs)
    (directory/name).write_bytes((directory/name).read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='byte identity mismatch'):
        bundle.verify_bundle(directory, digest, binding=inputs['binding'], checkpoint_identity=inputs['checkpoint_identity'])


def test_independent_manifest_and_inventory_required(tmp_path, inputs):
    directory = tmp_path/'owned'; digest, _ = bundle.write_bundle(directory, **inputs)
    with pytest.raises(ValueError, match='manifest hash'):
        bundle.verify_bundle(directory, '0'*64, binding=inputs['binding'], checkpoint_identity=inputs['checkpoint_identity'])
    (directory/'extra').write_bytes(b'foreign')
    with pytest.raises(ValueError, match='inventory'):
        bundle.verify_bundle(directory, digest, binding=inputs['binding'], checkpoint_identity=inputs['checkpoint_identity'])


def test_actor_disagreement_rejected_before_creating_directory(tmp_path, inputs):
    changed = deepcopy(inputs); changed['payload']['ticks'][0]['actions'][0, 0] += .01
    with pytest.raises(ValueError, match='restored actor'): bundle.write_bundle(tmp_path/'owned', **changed)
    assert not (tmp_path/'owned').exists()


@pytest.mark.parametrize('key', ['runtime_raw', 'launch_raw', 'checkpoint_raw'])
def test_input_hash_drift_refused_before_output(tmp_path, inputs, key):
    changed = deepcopy(inputs); changed[key] += b'changed'
    with pytest.raises(ValueError, match='input bytes mismatch'): bundle.write_bundle(tmp_path/'owned', **changed)
    assert not (tmp_path/'owned').exists()


def test_partial_write_has_no_manifest_and_is_not_overwritten(tmp_path, inputs, monkeypatch):
    directory = tmp_path/'owned'
    real_fsync = bundle.os.fsync; calls = 0
    def fail(fd):
        nonlocal calls
        calls += 1
        if calls == 3: raise OSError('synthetic disk failure')
        return real_fsync(fd)
    monkeypatch.setattr(bundle.os, 'fsync', fail)
    with pytest.raises(OSError, match='disk failure'): bundle.write_bundle(directory, **inputs)
    assert directory.exists() and not (directory/'manifest.json').exists()
    monkeypatch.setattr(bundle.os, 'fsync', real_fsync)
    with pytest.raises(FileExistsError): bundle.write_bundle(directory, **inputs)


def test_symlink_output_refused(tmp_path, inputs):
    elsewhere = tmp_path/'elsewhere'; elsewhere.mkdir()
    link = tmp_path/'link'; link.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError): bundle.write_bundle(link/'owned', **inputs)
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize('name', ['score.json', 'restore.json'])
def test_rehashed_false_receipt_still_fails_recomputation(tmp_path, inputs, name):
    import json
    directory = tmp_path/'owned'; bundle.write_bundle(directory, **inputs)
    receipt = json.loads((directory/name).read_bytes())
    receipt['checkpoint_admitted'] = True
    raw = (canonical(receipt)+'\n').encode(); (directory/name).write_bytes(raw)
    manifest = json.loads((directory/'manifest.json').read_bytes())
    manifest['files'][name] = dict(sha256=sha256(raw).hexdigest(), bytes=len(raw))
    raw_manifest = (canonical(manifest)+'\n').encode(); (directory/'manifest.json').write_bytes(raw_manifest)
    # Even supplying the modified outer hash cannot turn an invented score or
    # restoration claim into the deterministic result of the retained inputs.
    with pytest.raises(ValueError, match='receipt mismatch|score mismatch'):
        bundle.verify_bundle(directory, sha256(raw_manifest).hexdigest(),
            binding=inputs['binding'], checkpoint_identity=inputs['checkpoint_identity'])
