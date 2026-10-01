"""Explicit opt-in CPU math profile receipt for later replay processes.

This module only inspects the already-running process. It never selects the
profile, changes PyTorch state, initializes CUDA, or claims replay parity.
"""

from hashlib import sha256
from importlib.metadata import version
import os
import json
from pathlib import Path
import platform
import sys
import torch

PROTOCOL = 'football-b1n-portable-cpu-math-profile-v1'
TORCH_VERSION = '2.9.1'
LIBTORCH_CPU_SHA256 = '7918fc09ff644c9b667921100b924e33ea432c85737c2982b56283691b32a4d7'
_SETTINGS = {
    'ATEN_CPU_CAPABILITY': 'default',
    'MKL_CBWR': 'COMPATIBLE',
    'OMP_NUM_THREADS': '1',
}
_RUNTIME_KEYS = {
    'platform', 'architecture', 'torch_version', 'cpu_capability',
    'torch_num_threads', 'libtorch_cpu_sha256',
}


def settings():
    """Return the exact environment settings for a caller to opt into."""
    return dict(_SETTINGS)


def expected_receipt():
    """Pure expected bytes, not an observation or live runtime qualification."""
    return {
        'protocol': PROTOCOL, 'status': 'portable-cpu-math-profile-checked',
        'settings': settings(), 'platform': 'linux', 'architecture': 'x86_64',
        'torch_version': TORCH_VERSION, 'cpu_capability': 'DEFAULT',
        'torch_num_threads': 1, 'libtorch_cpu_sha256': LIBTORCH_CPU_SHA256,
        'full_replay_performed': False, 'actor_output_parity_verified': False,
        'capability_accepted': False, 'binary_equivalence_verified': False,
    }


def validate_receipt(receipt):
    """Validate recorded profile metadata without inspecting the current host."""
    require(type(receipt) is dict, 'exact recorded CPU math profile')
    require(json.dumps(receipt, sort_keys=True, allow_nan=False) ==
            json.dumps(expected_receipt(), sort_keys=True, allow_nan=False),
            'exact recorded CPU math profile')


def check_recorded(receipt):
    """Require archived and actual process receipts before any actor inference."""
    validate_receipt(receipt)
    require(checked_receipt() == receipt, 'actual recorded CPU math profile')


def _parse_exec_environment(raw):
    """Select only profile keys; never expose unrelated process environment."""
    require(type(raw) is bytes and len(raw) <= 1024 * 1024,
            'bounded exec environment')
    entries = raw.split(b'\0')
    if entries[-1] == b'':
        entries.pop()
    selected = {}
    names = {key.encode('ascii'): key for key in _SETTINGS}
    for entry in entries:
        require(b'=' in entry, 'well-formed exec environment')
        name, value = entry.split(b'=', 1)
        if name in names:
            key = names[name]
            require(key not in selected, 'unique profile key in exec environment')
            try:
                selected[key] = value.decode('ascii')
            except UnicodeDecodeError as error:
                raise ValueError('ASCII profile value in exec environment') from error
    return selected


def inspect_exec_environment():
    """Read Linux's initial-exec environment, not a late os.environ snapshot.

    /proc/self/environ normally reports the environment supplied to execve;
    this is a configuration check, not tamper-proof process attestation.
    """
    require(sys.platform == 'linux', 'Linux CPU replay platform')
    try:
        with Path('/proc/self/environ').open('rb') as source:
            raw = source.read(1024 * 1024 + 1)
    except OSError as error:
        raise ValueError('readable exec environment required') from error
    return _parse_exec_environment(raw)


def inspect_runtime():
    """Read the current CPU runtime without touching CUDA or global settings."""
    require(sys.platform == 'linux', 'Linux CPU replay platform')
    require(platform.machine().lower() == 'x86_64', 'x86_64 CPU replay architecture')
    library = Path(torch.__file__).resolve().parent / 'lib' / 'libtorch_cpu.so'
    digest = sha256()
    with library.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {
        'platform': sys.platform,
        'architecture': platform.machine().lower(),
        'torch_version': version('torch'),
        'cpu_capability': torch.backends.cpu.get_cpu_capability(),
        'torch_num_threads': torch.get_num_threads(),
        'libtorch_cpu_sha256': digest.hexdigest(),
    }


def validate_runtime(runtime, *, exec_env=None, current_env=None):
    """Check initial and current settings plus runtime; overrides are test seams."""
    exec_env = inspect_exec_environment() if exec_env is None else exec_env
    current_env = os.environ if current_env is None else current_env
    for label, values in (('at exec', exec_env), ('currently', current_env)):
        require(isinstance(values, dict) or hasattr(values, 'get'), 'environment snapshot '+label)
        for key, expected in _SETTINGS.items():
            require(values.get(key) == expected,
                    f'{key} must equal {expected!r} {label}')

    require(type(runtime) is dict and set(runtime) == _RUNTIME_KEYS, 'exact CPU runtime inspection')
    require(runtime['platform'] == 'linux', 'Linux CPU replay platform')
    require(runtime['architecture'] == 'x86_64', 'x86_64 CPU replay architecture')
    require(runtime['torch_version'] == TORCH_VERSION, 'pinned PyTorch metadata version')
    require(runtime['cpu_capability'] == 'DEFAULT', 'ATen default CPU capability')
    require(type(runtime['torch_num_threads']) is int and runtime['torch_num_threads'] == 1,
            'single PyTorch CPU thread')
    require(runtime['libtorch_cpu_sha256'] == LIBTORCH_CPU_SHA256,
            'pinned libtorch_cpu.so hash')
    return {
        'protocol': PROTOCOL,
        'status': 'portable-cpu-math-profile-checked',
        'settings': settings(),
        **runtime,
        'full_replay_performed': False,
        'actor_output_parity_verified': False,
        'capability_accepted': False,
        'binary_equivalence_verified': False,
    }


def checked_receipt():
    """Inspect and validate the current process; no profile settings are applied."""
    return validate_runtime(inspect_runtime())


def require(condition, message):
    if not condition:
        raise ValueError(message)
