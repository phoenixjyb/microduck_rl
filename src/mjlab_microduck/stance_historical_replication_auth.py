"""Authenticate the three fixed packed-replication archives in original AVX2.

This proves bounded archive provenance only. It is not a live attestation, a
portable replay, a capability result, or evidence of GPU execution.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import os

from mjlab_microduck import stance_historical_training_auth as original
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = original.host, original.files
MODULE = 'mjlab_microduck.stance_historical_replication_auth'
PROTOCOL = 'football-b1n-historical-packed-replication-auth-v1'
SECONDS, OUTPUT_LIMIT = original.SECONDS, original.OUTPUT_LIMIT
TRAINING_SOURCE = 'be2d59661af293b0d67ae20d2e16db50514cce14'
SEEDS = (577, 587, 593)
REPORT_SHA256 = {
    577: '2a5a3ec503bcf39ed89d4f3a82cea8b2bd3974741313f3c7cd2ab8d97280b5de',
    587: '202d9b31aa27295c18a7c78086505ec7d821bc69d0568224def3bbe754ae9f05',
    593: '5ee3b60dab8c8e051d4fc4205b21302a1c85b12ebf25b4d4e57a6e00a6973d5c',
}
CHECKPOINTS = (64, 128, 192, 255)
EXEC_WRAPPER = '''import contextlib, sys
with contextlib.redirect_stdout(sys.stderr):
    from mjlab_microduck import stance_historical_replication_auth as auth
    receipt = auth.authenticate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
print(auth.canonical(receipt))
'''

# Reuse the original bounded machinery as function references. Nothing in the
# seed-577 legacy module or its protocol globals is changed by this module.
inventory_names = original.inventory_names
expected_runtime = original.expected_runtime
check_process = original.check_process
bounded_output = original.bounded_output


def _seed(seed):
    require(type(seed) is int and seed in SEEDS, 'one of the three fixed historical training seeds')
    return seed


def validate(receipt, source, retained):
    """Validate recorded provenance only; callers must rerun authenticate()."""
    files.hex_id(source, 40)
    require(type(receipt) is dict and set(receipt) == {
        'protocol', 'source', 'training_seed', 'cpu_runtime', 'settings',
        'timeout_seconds', 'archive_files', 'retained_training',
        'optimizer_steps', 'gpu_execution_performed', 'independent_attestation'},
        'exact historical replication authentication receipt')
    seed = _seed(receipt['training_seed'])
    require(receipt['protocol'] == PROTOCOL and receipt['source'] == source
            and canonical(receipt['cpu_runtime']) == canonical(expected_runtime())
            and receipt['settings'] == dict(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
            and type(receipt['timeout_seconds']) is int and receipt['timeout_seconds'] == SECONDS
            and type(receipt['optimizer_steps']) is int and receipt['optimizer_steps'] == 0
            and receipt['gpu_execution_performed'] is False
            and receipt['independent_attestation'] is False,
            'declared historical replication authentication provenance')
    require(receipt['retained_training'] == retained and type(retained) is dict
            and set(retained) == {'source', 'report_sha256', 'checkpoints'},
            'exact retained historical replication result')
    files.hex_id(retained['source'], 40)
    files.hex_id(retained['report_sha256'], 64)
    require(retained['source'] == TRAINING_SOURCE
            and retained['report_sha256'] == REPORT_SHA256[seed],
            'predeclared packed report hash for requested seed')
    checkpoints = retained['checkpoints']
    require(type(checkpoints) is list and len(checkpoints) == len(CHECKPOINTS)
            and all(type(saved) is dict and type(saved.get('identity')) is dict
                    for saved in checkpoints)
            and [saved['identity'].get('iteration') for saved in checkpoints] == list(CHECKPOINTS),
            'exact four common historical checkpoints')
    inventory = receipt['archive_files']
    require(type(inventory) is dict and set(inventory) == inventory_names(),
            'complete historical file inventory')
    for name, digest in inventory.items():
        require(type(name) is str and name == os.path.basename(name)
                and name not in ('.', '..', 'report.json'), 'plain historical archive filename')
        files.hex_id(digest, 64)
    for iteration, saved in zip(CHECKPOINTS, checkpoints):
        require(type(saved) is dict and set(saved) == {'file', 'sha256', 'identity'}
                and saved['file'] == f'model_{iteration}.pt',
                'exact retained historical checkpoint record')
        files.hex_id(saved['sha256'], 64)
        meta = saved['identity']
        require(type(meta) is dict and type(meta.get('training_seed')) is int
                and meta['training_seed'] == seed and meta.get('iteration') == iteration
                and meta.get('source') == TRAINING_SOURCE,
                'checkpoint identity matches requested historical training seed')
        evaluation.checkpoint.validate_identity(
            meta, evaluation=evaluation.lean.PACKED_REPLICATION['purpose'])
        require(inventory.get(saved['file']) == saved['sha256'],
                'selected checkpoint hash matches historical inventory')


def authenticate(source, training_source, seed):
    """Authenticate one fixed seed archive in the original CPU profile."""
    seed = _seed(seed)
    files.hex_id(training_source, 40)
    require(training_source == TRAINING_SOURCE,
            'predeclared original packed training source')
    runtime = check_process()
    host.identity(source)

    # Check the predeclared report digest before training_inputs can deserialize
    # any tensor. That unchanged verifier then hashes every archive file before
    # reconstructing the initializer and loading each selected checkpoint.
    root = evaluation.lean.output_path(
        training_source, evaluation.lean.PACKED_REPLICATION, seed)
    report_path = root/'report.json'
    raw = files.file_bytes(report_path)
    report_sha = sha256(raw).hexdigest()
    require(report_sha == REPORT_SHA256[seed],
            'predeclared original packed report hash for requested seed')
    report = files.parse(raw)
    retained = evaluation.training_inputs(evaluation.PORTABLE_PROBE, seed, training_source)
    require(retained['report_sha256'] == report_sha,
            'unchanged historical report after archive verification')
    require([saved['identity']['training_seed'] for saved in retained['checkpoints']]
            == [seed]*len(CHECKPOINTS),
            'every selected checkpoint has the requested historical seed')
    receipt = dict(protocol=PROTOCOL, source=source, training_seed=seed,
        cpu_runtime=runtime, settings=dict(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=''),
        timeout_seconds=SECONDS, archive_files=deepcopy(report['files']),
        retained_training=deepcopy(retained), optimizer_steps=0,
        gpu_execution_performed=False, independent_attestation=False)
    require(check_process() == runtime,
            'unchanged original historical CPU runtime after archive verification')
    validate(receipt, source, retained)
    return receipt


def run(source, training_source, seed):
    """Exec an isolated original-profile child; never alter the parent process."""
    files.hex_id(source, 40)
    files.hex_id(training_source, 40)
    require(training_source == TRAINING_SOURCE,
            'predeclared original packed training source')
    seed = _seed(seed)
    env = files.child_environment()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['OMP_NUM_THREADS'] = '1'
    env.pop('ATEN_CPU_CAPABILITY', None)
    env.pop('MKL_CBWR', None)
    command = [str(host.ROOT/'.venv/bin/python'), '-c', EXEC_WRAPPER,
               source, training_source, str(seed)]
    raw = bounded_output(command, cwd=host.ROOT, env=env)
    require(type(raw) is bytes and 0 < len(raw) <= OUTPUT_LIMIT,
            'bounded historical replication authentication output')
    receipt = files.parse(raw)
    require(raw == (canonical(receipt)+'\n').encode(),
            'one canonical historical replication authentication receipt')
    validate(receipt, source, receipt['retained_training'])
    require(receipt['retained_training']['source'] == training_source,
            'requested historical training source')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--training-source', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    print(canonical(authenticate(args.source, args.training_source, args.seed)))


if __name__ == '__main__':
    main()
