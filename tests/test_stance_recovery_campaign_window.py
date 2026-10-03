import math
import pytest

from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_contract as expired
from mjlab_microduck import stance_recovery_cuda_schedule_probe as probe


def test_fixed_new_window_does_not_relabel_expired_protocol():
    assert window.CUTOFF == 1791028800
    assert expired.CUTOFF < window.START < window.CUTOFF
    assert window.declaration()['restore_protected_services'] is False
    assert probe.LAUNCH_RESERVE == 540
    assert (probe.PREPARE_SECONDS, probe.SERVICE_SECONDS, probe.CHILD_SECONDS, probe.CLOSEOUT_SECONDS) == (120, 240, 180, 120)
    window.check(now=window.START, reserve_seconds=540)


@pytest.mark.parametrize('now', [window.START-1, window.CUTOFF, window.CUTOFF-540, window.CUTOFF+1])
def test_reserves_the_entire_new_work_and_closeout(now):
    with pytest.raises(ValueError, match='fixed 20:00'):
        window.check(now=now, reserve_seconds=540)


@pytest.mark.parametrize('now,reserve', [(True, 0), (math.nan, 0), (math.inf, 0),
                                      (window.START, True), (window.START, -1), (window.START, math.inf)])
def test_refuses_nonfinite_or_bool_budget(now, reserve):
    with pytest.raises(ValueError): window.check(now=now, reserve_seconds=reserve)


def test_source_and_modes_are_bounded():
    assert probe.service_name('a'*40, 'prepare') == 'microduck-scheduled-cuda-prepare-aaaaaaaaaaaa.service'
    with pytest.raises(ValueError): probe.service_name('a'*40, 'train')
    with pytest.raises(ValueError): probe.output_path('a'*39)


def test_child_hash_only_mode_refuses_cpu_parent_and_late_cuda_initialization(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='hash-only child checks'):
        probe.checked('a'*40, 'b'*64, replay_cpu=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(probe.torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='hash-only child checks'):
        probe.checked('a'*40, 'b'*64, replay_cpu=False)


def test_exact_service_caps_cannot_be_widened(monkeypatch):
    expected = dict(MainPID=str(probe.os.getpid()), ActiveState='active', RuntimeMaxUSec='4min',
                    MemoryMax=str(3*1024**3), CPUQuotaPerSecUSec='2s', Nice='10', KillMode='control-group')
    monkeypatch.setattr(probe.base.host, 'read', lambda *args: expected[args[-2]])
    assert probe.service_properties('a'*40, 'supervise') == expected
    expected['RuntimeMaxUSec'] = '5min'
    with pytest.raises(ValueError, match='exact independently capped'):
        probe.service_properties('a'*40, 'supervise')


def test_numeric_warning_and_log_size_guards_preserve_owned_log(tmp_path):
    path = tmp_path/'child.log'
    path.write_text(probe.base.host.STARTUP_INFO+'\n')
    probe.check_log(tmp_path)
    path.write_text('Warning: nonfinite force\n')
    with pytest.raises(ValueError, match='numerical/backend warning'): probe.check_log(tmp_path)
    assert path.read_text() == 'Warning: nonfinite force\n'
    path.write_bytes(b'x'*(probe.LOG_LIMIT+1))
    with pytest.raises(ValueError, match='bounded owned'): probe.check_log(tmp_path)
