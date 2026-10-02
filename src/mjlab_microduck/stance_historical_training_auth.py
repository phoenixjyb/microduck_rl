"""Bounded CPU-only authentication of the original packed training archive.

Historical initializer reconstruction uses its original AVX2 math environment.
This is separate from portable actor inference, never a training/GPU fallback.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

import torch

from mjlab_microduck import stance_cpu_replay_profile as profile
from mjlab_microduck import stance_lean_evaluation as evaluation
from mjlab_microduck.first_attempt_smoke import canonical, require

host, files = evaluation.host, evaluation.files
MODULE = 'mjlab_microduck.stance_historical_training_auth'
PROTOCOL = 'football-b1n-historical-packed-training-auth-v1'
SECONDS, OUTPUT_LIMIT = 60, 1024*1024
SEED = 577
TRAINING_SOURCE = 'be2d59661af293b0d67ae20d2e16db50514cce14'
EXEC_WRAPPER = '''import contextlib, sys
with contextlib.redirect_stdout(sys.stderr):
    from mjlab_microduck import stance_historical_training_auth as auth
    receipt = auth.authenticate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
print(auth.canonical(receipt))
'''


def inventory_names():
    lean = evaluation.lean
    return ({'launch.json', 'runtime.json', 'parent.pt', 'child.log', 'completed.json', 'initial.pt'}
        | {f'model_{i}.pt' for i in range(lean.UPDATES)}
        | {f'update-{i:03d}.json' for i in range(lean.UPDATES)}
        | {f'tick-{i:03d}-{j:02d}.json' for i in range(lean.UPDATES) for j in range(lean.STEPS)})


def expected_runtime():
    result = {k: profile.expected_receipt()[k] for k in profile._RUNTIME_KEYS}
    return {**result, 'cpu_capability': 'AVX2'}


def check_process():
    for values in (profile.inspect_exec_environment(), os.environ):
        require(values.get('OMP_NUM_THREADS') == '1'
                and 'ATEN_CPU_CAPABILITY' not in values and 'MKL_CBWR' not in values,
                'original historical CPU profile at exec and currently')
    with Path('/proc/self/environ').open('rb') as stream:
        raw = stream.read(OUTPUT_LIMIT+1)
    require(len(raw) <= OUTPUT_LIMIT and
            [entry for entry in raw.split(b'\0') if entry.startswith(b'CUDA_VISIBLE_DEVICES=')]
            == [b'CUDA_VISIBLE_DEVICES='], 'CUDA hidden at historical authentication exec')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU-only historical archive authentication')
    runtime = profile.inspect_runtime()
    require(canonical(runtime) == canonical(expected_runtime()), 'exact original historical CPU runtime')
    return runtime


def validate(receipt, source, retained):
    """Recorded provenance only; a live caller must rerun authenticate()."""
    files.hex_id(source, 40)
    require(type(receipt) is dict and set(receipt) == {
        'protocol', 'source', 'training_seed', 'cpu_runtime', 'settings',
        'timeout_seconds', 'archive_files', 'retained_training',
        'optimizer_steps', 'gpu_execution_performed', 'independent_attestation'},
        'exact historical authentication receipt')
    require(receipt['protocol'] == PROTOCOL and receipt['source'] == source
            and type(receipt['training_seed']) is int and receipt['training_seed'] == SEED
            and canonical(receipt['cpu_runtime']) == canonical(expected_runtime())
            and receipt['settings'] == dict(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
            and type(receipt['timeout_seconds']) is int and receipt['timeout_seconds'] == SECONDS
            and type(receipt['optimizer_steps']) is int and receipt['optimizer_steps'] == 0
            and receipt['gpu_execution_performed'] is False
            and receipt['independent_attestation'] is False,
            'declared historical authentication provenance')
    require(receipt['retained_training'] == retained and type(retained) is dict
            and set(retained) == {'source', 'report_sha256', 'checkpoints'},
            'exact historical retained training result')
    files.hex_id(retained['source'], 40); files.hex_id(retained['report_sha256'], 64)
    require(retained['source'] == TRAINING_SOURCE, 'predeclared original packed training source')
    inventory = receipt['archive_files']
    require(type(inventory) is dict and set(inventory) == inventory_names(),
            'complete historical file inventory')
    for name, digest in inventory.items():
        require(type(name) is str and name == os.path.basename(name)
                and name not in ('.', '..', 'report.json'), 'plain historical archive filename')
        files.hex_id(digest, 64)
    for saved in retained['checkpoints']:
        require(inventory.get(saved['file']) == saved['sha256'], 'selected checkpoint in historical inventory')


def authenticate(source, training_source, seed):
    require(type(seed) is int and seed == SEED, 'fixed historical training seed')
    files.hex_id(training_source, 40)
    require(training_source == TRAINING_SOURCE, 'predeclared original packed training source')
    runtime = check_process()
    host.identity(source)
    # This unchanged verifier authenticates every byte before tensor loading,
    # reconstructs the original initializer and checks every completed update.
    retained = evaluation.training_inputs(evaluation.PACKED_PROBE, seed, training_source)
    root = evaluation.lean.output_path(training_source, evaluation.lean.PACKED_REPLICATION, seed)
    raw = files.file_bytes(root/'report.json')
    require(host.digest(root/'report.json') == retained['report_sha256'], 'unchanged historical report')
    report = files.parse(raw)
    receipt = dict(protocol=PROTOCOL, source=source, training_seed=seed, cpu_runtime=runtime,
        settings=dict(OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=''), timeout_seconds=SECONDS,
        archive_files=deepcopy(report['files']), retained_training=deepcopy(retained),
        optimizer_steps=0, gpu_execution_performed=False, independent_attestation=False)
    require(check_process() == runtime, 'unchanged historical CPU runtime after archive verification')
    validate(receipt, source, retained)
    return receipt


def run(source, training_source, seed):
    """Exec the isolated original profile; never alter the portable parent."""
    files.hex_id(source, 40); files.hex_id(training_source, 40)
    require(training_source == TRAINING_SOURCE, 'predeclared original packed training source')
    require(type(seed) is int and seed == SEED, 'fixed historical training seed')
    env = files.child_environment()
    env['CUDA_VISIBLE_DEVICES'] = ''; env['OMP_NUM_THREADS'] = '1'
    env.pop('ATEN_CPU_CAPABILITY', None); env.pop('MKL_CBWR', None)
    # Route package-registration/authentication diagnostics away from receipt
    # stdout before importing the package, not after Torch has initialized.
    command = [str(host.ROOT/'.venv/bin/python'), '-c', EXEC_WRAPPER,
               source, training_source, str(seed)]
    raw = bounded_output(command, cwd=host.ROOT, env=env)
    require(type(raw) is bytes and 0 < len(raw) <= OUTPUT_LIMIT, 'bounded historical authentication output')
    receipt = files.parse(raw)
    require(raw == (canonical(receipt)+'\n').encode(), 'one canonical historical authentication receipt')
    validate(receipt, source, receipt['retained_training'])
    require(receipt['retained_training']['source'] == training_source,
            'requested historical training source')
    return receipt


def bounded_output(command, *, cwd, env):
    """Cap total stdout/stderr while reading, without mixing the two streams.

    Only this new CPU process group is terminated on failure. The 60-second
    execution limit has at most two seconds of reap cleanup, no retry.
    """
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
    output = {'stdout': bytearray(), 'stderr': bytearray()}
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')
            while selector.get_map():
                remaining = SECONDS-(time.monotonic()-started)
                if remaining <= 0: raise subprocess.TimeoutExpired(command, SECONDS)
                for key, _ in selector.select(min(remaining, .25)):
                    size = sum(len(raw) for raw in output.values())
                    block = os.read(key.fd, min(65536, OUTPUT_LIMIT+1-size))
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    output[key.data].extend(block)
                    require(sum(len(raw) for raw in output.values()) <= OUTPUT_LIMIT,
                            'bounded historical authentication output')
        remaining = SECONDS-(time.monotonic()-started)
        if remaining <= 0: raise subprocess.TimeoutExpired(command, SECONDS)
        result = proc.wait(timeout=remaining)
        if result:
            error = subprocess.CalledProcessError(result, command)
            error.add_note('historical CPU authentication output: '+
                           bytes((output['stderr']+output['stdout'])[:16384]).decode('utf-8', errors='replace'))
            raise error
        diagnostics = output['stderr'].decode('utf-8', errors='strict').splitlines()
        require(host.BAD_LOG.search('\n'.join(line for line in diagnostics if line != host.STARTUP_INFO)) is None,
                'no numerical/backend warning in historical CPU authentication')
        return bytes(output['stdout'])
    except BaseException:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait(timeout=2)
        raise
    finally:
        proc.stdout.close()
        proc.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--training-source', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    print(canonical(authenticate(args.source, args.training_source, args.seed)))


if __name__ == '__main__': main()
