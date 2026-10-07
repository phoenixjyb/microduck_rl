"""Fail-closed fresh supervisor contracts; synthetic checks are not native proof."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from mjlab_microduck import stance_rne_coupled_probe as probe
from mjlab_microduck.stance_com_entry_receiver import SERVICE_CAPS, canonical

SOURCE = "1" * 40


def test_distinct_units_outputs_and_exact_caps():
    assert (
        probe.unit(SOURCE, "tests")
        == "microduck-rne-coupled-tests-111111111111.service"
    )
    assert probe.unit(SOURCE, "run") == "microduck-rne-coupled-run-111111111111.service"
    assert str(probe.output(SOURCE, "tests")).endswith(
        "artifacts/tools/rne-coupled-tests-111111111111"
    )
    assert str(probe.output(SOURCE, "run")).endswith(
        "artifacts/evaluations/rne-coupled-run-111111111111"
    )
    assert probe.SERVICE_SECONDS == probe.CLOSEOUT_SECONDS == 300
    assert probe.CHILD_SECONDS == 240 and probe.MARGIN == 60
    assert SERVICE_CAPS["LimitFSIZE"] == "1048576"
    assert probe.old.TEST_SERVICE_CAPS["LimitFSIZE"] == "67108864"
    assert len(probe.OWN) == 7
    assert probe.BASE == "4888ceb2c8b40a762a12265cb12df3b1e3c93db4"


@pytest.mark.parametrize(
    "source,mode",
    [
        ("f" * 39, "run"),
        ("F" * 40, "run"),
        (None, "run"),
        (SOURCE, "child"),
        (SOURCE, None),
    ],
)
def test_unit_identity_refuses_nonliteral_source_or_mode(source, mode):
    with pytest.raises(ValueError, match="literal fresh RNE"):
        probe.unit(source, mode)


def test_fresh_cutoff_reserve_is_not_the_old_oct6_window():
    cutoff = probe.coupled.CUTOFF
    assert cutoff == 1791349200
    probe.check_window(660, cutoff - 661)
    with pytest.raises(ValueError):
        probe.check_window(660, cutoff - 660)
    with pytest.raises(ValueError):
        probe.check_window(1, float("nan"))
    assert probe.old.CUTOFF == 1791262800


def test_ordered_test_list_extends_but_does_not_mutate_original():
    prior = probe.prior.test_files()
    current = probe.test_files()
    assert current[:67] == prior
    assert len(current) == len(set(current)) == 70
    assert current[67:] == [
        "tests/test_stance_rne_coupled_control.py",
        "tests/test_stance_rne_coupled_probe.py",
        "tests/test_stance_rne_coupled_receiver.py",
    ]
    assert probe.prior.receiver.EXPECTED_TESTS == 2093
    assert probe.coupled.EXPECTED_TESTS == 2019
    assert probe.old.EXPECTED_TESTS == 1879
    assert probe.prior.test_files() == prior
    assert probe.LEAF_COUNT == 656
    assert tuple(probe.receiver.CASE_ORDER) == (
        "original0",
        "serial0",
        "original1",
        "serial1",
    )


@pytest.mark.parametrize(
    "mutation", ["count", "skip", "error", "failure", "empty", "unpredeclared"]
)
def test_whole_junit_refuses_omissions_and_unpredeclared_count(monkeypatch, mutation):
    monkeypatch.setattr(probe.receiver, "EXPECTED_TESTS", 3)
    fields = dict(tests=3, skipped=0, errors=0, failures=0)
    if mutation == "count":
        fields["tests"] = 2
    elif mutation in ("skip", "error", "failure"):
        fields[
            {"skip": "skipped", "error": "errors", "failure": "failures"}[mutation]
        ] = 1
    elif mutation == "unpredeclared":
        monkeypatch.setattr(probe.receiver, "EXPECTED_TESTS", 0)
    raw = (
        "<testsuites><testsuite "
        + " ".join(f'{key}="{value}"' for key, value in fields.items())
        + "/></testsuites>"
    ).encode()
    if mutation == "empty":
        raw = b"<testsuites/>"
    with pytest.raises(ValueError):
        probe.checked_junit(raw)


def test_complete_junit_positive_is_only_synthetic(monkeypatch):
    monkeypatch.setattr(probe.receiver, "EXPECTED_TESTS", 3)
    probe.checked_junit(
        b'<testsuites><testsuite tests="3" skipped="0" errors="0" failures="0"/></testsuites>'
    )


def test_actual_service_caps_and_pid_are_not_assumed(monkeypatch):
    live = dict(
        SERVICE_CAPS,
        MainPID="17",
        ActiveState="active",
        SubState="running",
        InvocationID="a" * 32,
    )
    monkeypatch.setattr(probe, "properties", lambda *_: dict(live))
    assert probe.service(SOURCE, "run", 17) == live
    live["LimitFSIZE"] = "67108864"
    with pytest.raises(ValueError):
        probe.service(SOURCE, "run", 17)
    live["LimitFSIZE"] = SERVICE_CAPS["LimitFSIZE"]
    with pytest.raises(ValueError):
        probe.service(SOURCE, "run", 18)


def test_cpu_service_uses_its_separate_file_cap(monkeypatch):
    live = dict(
        probe.old.TEST_SERVICE_CAPS,
        MainPID="17",
        ActiveState="active",
        SubState="running",
        InvocationID="a" * 32,
    )
    monkeypatch.setattr(probe, "properties", lambda *_: dict(live))
    assert probe.service(SOURCE, "tests", 17) == live
    with pytest.raises(ValueError):
        probe.service(SOURCE, "run", 17)


def test_child_refuses_hidden_cuda_before_lease_or_import(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        probe.old, "inherited_lease", lambda *_: pytest.fail("must refuse before lease")
    )
    with pytest.raises(ValueError, match="owned visible native child"):
        probe.child(SOURCE, 5, 17, "0" * 64)


def test_current_native_predecessor_is_independently_identical():
    # This CPU fixture authenticates retained bytes, not native execution.
    root = Path(__file__).resolve().parents[1]
    closeout = root / "artifacts/tools/com-coupled-run-closeout-6cd032915816"
    for name, digest in probe.COM_PROOF.items():
        assert sha256((closeout / name).read_bytes()).hexdigest() == digest
    raw = (
        root / "artifacts/tools/com-coupled-temporal-6cd032915816/native-analysis.json"
    ).read_bytes()
    assert len(raw) == 53898 and sha256(raw).hexdigest() == probe.TEMPORAL_SHA256
    value = json.loads(raw)
    assert value["decision"] == "temporal-serial-negative"
    assert value["negative_coordinate_count"] == 100
    assert all(flag is False for flag in value["flags"].values())


def test_retained_com_authenticates_all_whole_files_before_json(monkeypatch):
    monkeypatch.setattr(probe.old, "bounded", lambda *_args, **_kwargs: b"forged")
    monkeypatch.setattr(
        probe.json,
        "loads",
        lambda *_: pytest.fail("unverified predecessor JSON decoded"),
    )
    with pytest.raises(ValueError, match="whole immutable paired CoM"):
        probe.retained_com()


def test_retained_com_positive_is_not_temporal_acceptance(monkeypatch):
    monkeypatch.setattr(probe, "ROOT", Path(__file__).resolve().parents[1])
    value = probe.retained_com()
    assert value["paired_decision"] == "fresh-coupled-control-exact"
    assert value["temporal_decision"] == "temporal-serial-negative"
    assert value["temporal_sha256"] == probe.TEMPORAL_SHA256
    assert canonical(value) == canonical(dict(value))


def test_retained_rne_authenticates_all_whole_files_before_json(monkeypatch):
    monkeypatch.setattr(probe.old, "bounded", lambda *_args, **_kwargs: b"forged")
    monkeypatch.setattr(
        probe.json,
        "loads",
        lambda *_: pytest.fail("unverified RNE predecessor decoded"),
    )
    with pytest.raises(ValueError, match="whole independent actual-entry RNE"):
        probe.retained_rne()


def test_retained_rne_keeps_actual_temporal_negative_and_both_receiver_anchors(
    monkeypatch,
):
    monkeypatch.setattr(probe, "ROOT", Path(__file__).resolve().parents[1])
    proof = probe.retained_rne()
    assert proof["source"] == "3a50dea5d430ae942a48e81ba5e9391a6e7d3b83"
    assert proof["decision"] == "fresh-rne-serial-reference-exact"
    assert proof["separate_temporal_exact"] is False
    assert proof["closeout"]["receiver.json"] == proof["closeout"]["mac-receiver.json"]
    assert len(probe.receiver.CAPS) == 30
    assert not {"concurrent.bin", "serial.bin", "frames.bin"} & set(probe.receiver.CAPS)
