"""Explicit host migration, not a caller-configurable GPU safety bypass.

The historical 100.100 profile remains the default. The opt-in WSL profile is
bound to one checkout, machine, GPU and driver and cooperates with FilmBrain's
existing lock. It changes no learner, physics, observation or acceptance gate.
"""
import os
from pathlib import Path

KEY = 'MICRODUCK_STANCE_PROFILE'
DEFAULT = 'linux-100100'
WSL = 'wsl-10098-20260930'
PROFILES = {
    DEFAULT: dict(root='/home/converge/work/microduck_rl-athletics-obstacle-curriculum',
        machine='0c79e415429b4933a400159bfa79a34d',
        gpu='GPU-f21e0304-3b55-b6eb-4993-946e7ee1f6dd', driver='595.84',
        smi='nvidia-smi', lock='/home/converge/.local/state/microduck-gpu0.lock',
        idle_memory_mib=100, temperature_c=80),
    WSL: dict(root='/home/yanbo/work/microduck_rl-stance-replication-20260930',
        machine='7d6778c98cb345788b8c1a410f19ad35',
        gpu='GPU-7d72b360-33bc-2cee-3ff4-a954474011b5', driver='595.95',
        smi='/usr/lib/wsl/lib/nvidia-smi',
        lock='/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock',
        idle_memory_mib=1024, temperature_c=75),
}


def select(name=None):
    name = os.environ.get(KEY, DEFAULT) if name is None else name
    if name not in PROFILES:
        raise ValueError('unknown explicit stance execution profile')
    return dict(name=name, **PROFILES[name])


PROFILE = select()
ROOT = Path(PROFILE['root'])
LOCK = Path(PROFILE['lock'])


def service_commands(services):
    # Both namespaces are checked on WSL; no existing service is changed.
    commands = [(s, ('systemctl', 'show', s, '-p', 'ActiveState', '--value')) for s in services]
    if PROFILE['name'] == WSL:
        commands += [('user:'+s, ('systemctl', '--user', 'show', s, '-p', 'ActiveState', '--value'))
                     for s in services]
    return commands


def child_settings():
    if PROFILE['name'] == DEFAULT:
        return {}
    return {KEY: PROFILE['name']}
