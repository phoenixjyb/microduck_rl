"""Fail-closed one-tick supervisor contracts; no native admission."""

from hashlib import sha256
from pathlib import Path

import pytest

from mjlab_microduck import stance_bam_load_probe as probe

SOURCE = "1" * 40


def test_distinct_source_units_paths_collection_and_caps():
    assert probe.unit(SOURCE, "run") == "microduck-bam-load-run-111111111111.service"
    assert (
        probe.unit(SOURCE, "tests") == "microduck-bam-load-tests-111111111111.service"
    )
    assert str(probe.output(SOURCE, "run")).endswith(
        "evaluations/bam-load-run-111111111111"
    )
    assert len(probe.OWN) == 7 and probe.LEAF_COUNT == 673
    assert probe.BASE == "4458450658ab3f0e12e17d393371003774298138"
    original = probe.step_prior.test_files()
    current = probe.test_files()
    assert current[:73] == original
    assert len(current) == len(set(current)) == 77
    assert current[73:] == ["tests/test_stance_serial_step_divergence.py"] + [
        f"tests/test_stance_bam_load_{n}.py" for n in ("observer", "probe", "receiver")
    ]
    assert probe.step_prior.receiver.EXPECTED_TESTS == 2250
    assert probe.SERVICE_SECONDS == probe.CLOSEOUT_SECONDS == 300
    assert probe.CHILD_SECONDS == 240 and probe.MARGIN == 60
    assert len(probe.receiver.CAPS) == 48
    assert probe.receiver.MOTOR_BYTES == 563200 and probe.receiver.MASK_BYTES == 7040
    assert all(v <= 2 * 1024**2 for v in probe.receiver.CAPS.values())
    assert probe.SERVICE_CAPS["LimitFSIZE"] == "2097152"
    assert probe.step_prior.SERVICE_CAPS["LimitFSIZE"] == "1048576"
    assert probe.old.TEST_SERVICE_CAPS["LimitFSIZE"] == "67108864"


@pytest.mark.parametrize(
    "source,mode",
    [(None, "run"), ("f" * 39, "run"), ("F" * 40, "run"), (SOURCE, "child")],
)
def test_nonliteral_units_refused(source, mode):
    with pytest.raises(ValueError):
        probe.unit(source, mode)


def test_oct7_reserve_strict_and_old_cutoffs_unchanged():
    cutoff = 1791349200
    probe.check_window(660, cutoff - 661)
    with pytest.raises(ValueError):
        probe.check_window(660, cutoff - 660)
    assert probe.old.CUTOFF == 1791262800


@pytest.mark.parametrize(
    "mutation", ["count", "skip", "error", "failure", "unfinalized", "empty"]
)
def test_actual_complete_junit_count_required(monkeypatch, mutation):
    monkeypatch.setattr(probe.receiver, "EXPECTED_TESTS", 3)
    fields = dict(tests=3, skipped=0, errors=0, failures=0)
    if mutation == "count":
        fields["tests"] = 2
    elif mutation in ("skip", "error", "failure"):
        fields[
            {"skip": "skipped", "error": "errors", "failure": "failures"}[mutation]
        ] = 1
    elif mutation == "unfinalized":
        monkeypatch.setattr(probe.receiver, "EXPECTED_TESTS", 0)
    raw = (
        "<testsuites><testsuite "
        + " ".join(f'{k}="{v}"' for k, v in fields.items())
        + "/></testsuites>"
    ).encode()
    if mutation == "empty":
        raw = b"<testsuites/>"
    with pytest.raises(ValueError):
        probe.checked_junit(raw)


def test_actual_service_caps_and_invocation_pid_required(monkeypatch):
    live = dict(
        probe.SERVICE_CAPS,
        MainPID="17",
        ActiveState="active",
        SubState="running",
        InvocationID="a" * 32,
    )
    monkeypatch.setattr(probe, "properties", lambda *_: dict(live))
    assert probe.service(SOURCE, "run", 17) == live
    with pytest.raises(ValueError):
        probe.service(SOURCE, "run", 18)
    live["MemoryMax"] = "8589934592"
    with pytest.raises(ValueError):
        probe.service(SOURCE, "run", 17)


def test_child_refuses_hidden_cuda_before_lease(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        probe.old, "inherited_lease", lambda *_: pytest.fail("early lease")
    )
    with pytest.raises(ValueError):
        probe.child(SOURCE, 5, 17, "0" * 64)


def test_predecessor_authentication_before_json(monkeypatch):
    monkeypatch.setattr(probe.old, "bounded", lambda *_args, **_kwargs: b"forged")
    monkeypatch.setattr(
        probe.json, "loads", lambda *_: pytest.fail("unauthenticated predecessor")
    )
    with pytest.raises(ValueError, match="whole native and Mac coupled proof"):
        probe.retained_rne()


def test_both_independent_predecessor_reports_retained(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(probe, "ROOT", root)
    value = probe.retained_rne()
    assert value == probe.receiver.PREDECESSOR
    assert value["full_window_qualified"] is False
    directory = root / "artifacts/tools/rne-coupled-run-closeout-50864a995f33"
    for name, anchor in probe.receiver.PREDECESSOR_FILES.items():
        raw = (directory / name).read_bytes()
        assert (
            len(raw) == anchor["bytes"] and sha256(raw).hexdigest() == anchor["sha256"]
        )


def test_complete_f32_producer_refuses_nonfinite_shape_and_dtype():
    import torch

    assert len(probe._f32(torch.zeros((64, 14)), (64, 14), torch)) == 3584
    for value in (
        torch.zeros((63, 14)),
        torch.zeros((64, 14), dtype=torch.float64),
        torch.full((64, 14), float("nan")),
    ):
        with pytest.raises(ValueError):
            probe._f32(value, (64, 14), torch)


def test_closed_negative_tick_is_authenticated_and_preserved(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(probe, "ROOT", root)
    value = probe.retained_tick()
    assert value == probe.receiver.TICK_PREDECESSOR
    assert value["decision"] == "fresh-one-step-serial-repeat-negative"
    assert value["full_window_qualified"] is False


def test_retained_tick_reproduction_must_equal_both_whole_files(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(probe, "ROOT", root)
    original = probe.old.bounded

    def changed(path, *args, **kwargs):
        if Path(path).name == "mac-analysis.json":
            return b"changed"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(probe.old, "bounded", changed)
    with pytest.raises(ValueError, match="both independent retained"):
        probe.retained_tick()
