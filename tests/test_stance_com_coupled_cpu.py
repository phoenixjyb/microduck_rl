"""Fresh CPU authority checks; synthetic metadata, not native qualification."""

import pytest

from mjlab_microduck import stance_com_coupled_cpu as p
from mjlab_microduck import stance_com_entry_probe as old


def test_fresh_window_does_not_reopen_old_runner():
    assert p.CUTOFF == 1791349200 and old.CUTOFF == 1791262800
    assert old.EXPECTED_TESTS == 1879
    assert p.BASE == "8b032a2779d8bd743fab02bbc596db9bf6116987"
    assert p.ROOT == old.ROOT and len(p.OWN) == 10
    assert len(p.test_files()) == len(set(p.test_files())) == 60
    assert p.CAPS["LimitFSIZE"] == str(64 * 1024**2)
    assert old.SERVICE_CAPS["LimitFSIZE"] == str(1024**2)
    assert p.CAPS["MemoryMax"] == str(6 * 1024**3)


@pytest.mark.parametrize(
    "now", [float("nan"), float("inf"), -1, True, p.CUTOFF, p.CUTOFF - 660]
)
def test_window_rejects_invalid_or_insufficient_time(now):
    with pytest.raises(ValueError, match="October 7"):
        p.check_window(660, now)


@pytest.mark.parametrize("reserve", [True, 0, -1, 660.0])
def test_window_requires_positive_literal_reserve(reserve):
    with pytest.raises(ValueError):
        p.check_window(reserve, p.CUTOFF - 1000)


def test_window_accepts_only_complete_reserved_job():
    p.check_window(660, p.CUTOFF - 661)
    assert p.unit("a" * 40) == "microduck-com-coupled-cpu-aaaaaaaaaaaa.service"
    assert p.output("a" * 40).name == "com-coupled-cpu-aaaaaaaaaaaa"


@pytest.mark.parametrize("source", [None, "", "a" * 39, "A" * 40, "../x", True])
def test_source_unit_is_exact(source):
    with pytest.raises(ValueError):
        p.unit(source)


def test_junit_requires_exact_count():
    raw = (
        f'<testsuites><testsuite tests="{p.EXPECTED_TESTS}" '
        'errors="0" failures="0" skipped="0" /></testsuites>'
    ).encode()
    assert p.checked_junit(raw) == p.EXPECTED_TESTS


@pytest.mark.parametrize(
    "field,value",
    [("tests", "1"), ("errors", "1"), ("failures", "1"), ("skipped", "1")],
)
def test_junit_rejects_count_and_any_skips_errors_failures(field, value):
    fields = dict(tests=str(p.EXPECTED_TESTS), errors="0", failures="0", skipped="0")
    fields[field] = value
    raw = (
        "<testsuites><testsuite "
        + " ".join(f'{k}="{v}"' for k, v in fields.items())
        + " /></testsuites>"
    ).encode()
    with pytest.raises(ValueError):
        p.checked_junit(raw)


def test_live_service_requires_exact_caps_owner_and_invocation(monkeypatch):
    values = dict(
        p.CAPS,
        MainPID="123",
        InvocationID="b" * 32,
        ActiveState="active",
        SubState="running",
    )
    monkeypatch.setattr(
        old, "read", lambda *_: "\n".join(f"{k}={v}" for k, v in values.items())
    )
    assert p.service("a" * 40, 123)["MainPID"] == "123"
    values["NRestarts"] = "1"
    with pytest.raises(ValueError):
        p.service("a" * 40, 123)
