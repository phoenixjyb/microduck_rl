"""CPU-only checks for the explicit stance host profile and refusal bounds."""

import pytest

from mjlab_microduck import foundation_command_campaign as campaign
from mjlab_microduck import gpu_idle_gate as idle_gate
from mjlab_microduck import stance_execution_profile as execution


def use_profile(monkeypatch, name):
    profile = execution.select(name)
    monkeypatch.setattr(execution, 'PROFILE', profile)
    return profile


def test_default_profile_preserves_root_lock_services_and_idle_memory(monkeypatch):
    profile = use_profile(monkeypatch, execution.DEFAULT)
    assert profile['root'] == '/home/converge/work/microduck_rl-athletics-obstacle-curriculum'
    assert profile['lock'] == '/home/converge/.local/state/microduck-gpu0.lock'
    assert execution.ROOT.as_posix() == profile['root']
    assert execution.LOCK.as_posix() == profile['lock']
    assert profile['idle_memory_mib'] == 100 and profile['temperature_c'] == 80
    services = ('service-a.service', 'service-b.service')
    assert execution.service_commands(services) == [
        (service, ('systemctl', 'show', service, '-p', 'ActiveState', '--value'))
        for service in services]


def test_wsl_profile_is_bound_to_the_fixed_host_gpu_and_shared_lock(monkeypatch):
    profile = use_profile(monkeypatch, execution.WSL)
    assert profile['root'] == '/home/yanbo/work/microduck_rl-stance-replication-20260930'
    assert profile['machine'] == '7d6778c98cb345788b8c1a410f19ad35'
    assert profile['gpu'] == 'GPU-7d72b360-33bc-2cee-3ff4-a954474011b5'
    assert profile['driver'] == '595.95'
    assert profile['smi'] == '/usr/lib/wsl/lib/nvidia-smi'
    assert profile['lock'] == '/home/yanbo/data/recomo/film-brain/runtimes/wan22-fun-camera-3e8d686-py312-cu130/wan-gpu.lock'
    assert profile['idle_memory_mib'] == 1024 and profile['temperature_c'] == 75


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match='unknown explicit stance execution profile'):
        execution.select('user-chosen-profile')


def test_wsl_child_receives_profile_and_default_child_keeps_historical_environment(monkeypatch):
    monkeypatch.delenv(execution.KEY, raising=False)
    use_profile(monkeypatch, execution.WSL)
    assert campaign.child_environment()[execution.KEY] == execution.WSL
    use_profile(monkeypatch, execution.DEFAULT)
    assert execution.KEY not in campaign.child_environment()


def test_wsl_service_probe_checks_system_and_user_namespaces(monkeypatch):
    use_profile(monkeypatch, execution.WSL)
    services = ('service-a.service', 'service-b.service')
    assert execution.service_commands(services) == [
        (service, ('systemctl', 'show', service, '-p', 'ActiveState', '--value'))
        for service in services] + [
        ('user:' + service,
         ('systemctl', '--user', 'show', service, '-p', 'ActiveState', '--value'))
        for service in services]


def idle_reader(*, temperature=45, memory=32):
    def read(*args):
        if args[0] == 'systemctl':
            return 'inactive'
        if '--query-compute-apps=pid' in args:
            return ''
        return f'0, {temperature}, {memory}'
    return read


@pytest.mark.parametrize('temperature,memory', [(75, 32), (45, 1024)])
def test_wsl_idle_gate_refuses_temperature_and_memory_boundaries(monkeypatch, temperature, memory):
    profile = use_profile(monkeypatch, execution.WSL)
    with pytest.raises(ValueError, match='occupied/unsafe GPU'):
        idle_gate.wait_idle(reader=idle_reader(temperature=temperature, memory=memory),
                            sleep=lambda _: None, now=lambda: 0)


def test_wsl_idle_gate_accepts_values_below_both_boundaries(monkeypatch):
    profile = use_profile(monkeypatch, execution.WSL)
    result = idle_gate.wait_idle(reader=idle_reader(temperature=74, memory=1023),
                                 sleep=lambda _: None, now=lambda: 0)
    assert len(result['samples']) == 2
    assert result['samples'][-1]['memory_mib'] == 1023


def monitor_reader(*, used=5120, free=6144):
    reads = []
    def read(command, **kwargs):
        args = tuple(command)
        reads.append(args)
        if args[0] == 'systemctl':
            return 'inactive'
        if '--query-compute-apps=pid' in args:
            return '42'
        if '--query-gpu=temperature.gpu' in args:
            return '45'
        if '--query-gpu=memory.used,memory.free' in args:
            return f'{used}, {free}'
        raise AssertionError(f'unexpected monitor query: {args}')
    return read, reads


def test_wsl_live_monitor_allows_the_declared_memory_reserve(monkeypatch):
    use_profile(monkeypatch, execution.WSL)
    read, reads = monitor_reader()
    monkeypatch.setattr(campaign.subprocess, 'check_output', read)
    result = campaign.live_gpu(42)
    assert result['memory_used_mib'] == 5120
    assert result['memory_free_mib'] == 6144
    assert any(args[:3] == ('systemctl', '--user', 'show') for args in reads)
    assert any(args[0] == execution.PROFILE['smi'] for args in reads)


@pytest.mark.parametrize('used,free', [(5121, 6144), (5120, 6143)])
def test_wsl_live_monitor_refuses_memory_reserve_breach(monkeypatch, used, free):
    use_profile(monkeypatch, execution.WSL)
    read, _ = monitor_reader(used=used, free=free)
    monkeypatch.setattr(campaign.subprocess, 'check_output', read)
    with pytest.raises(ValueError, match='WSL GPU memory reserve'):
        campaign.live_gpu(42)
