"""Shallow local checks for the bounded three-case frozen-parent dose screen.

The archive, profile, host, runtime, and traces are synthetic seams; no test
authenticates the selected parent or qualifies real physics/service execution.
"""

from hashlib import sha256 as real_sha256
import json
import os
from pathlib import Path

import pytest
import torch

from mjlab_microduck import stance_recovery_dose_screen as dose


SOURCE = "d" * 40
CP = b"synthetic selected checkpoint bytes"
CELLS = ("zero-wrench", "+x-2n-20steps-t250", "+x-4n-10steps-t250")


def _hash(raw):
    return real_sha256(raw).hexdigest()


def _flags():
    return dict(dose.base.baseline.FALSE_FLAGS)


def _install_closeout(monkeypatch, tmp_path, events):
    root = tmp_path / "completed-scheduled-closeout"
    root.mkdir(parents=True)
    artifacts = {"case-0.pt": b"synthetic completed case 0", "case-1.pt": b"synthetic completed case 1"}
    inventory = {}
    for name, raw in artifacts.items():
        (root / name).write_bytes(raw)
        inventory[name] = dict(bytes=len(raw), sha256=_hash(raw))
    closeout = dict(source=dose.CLOSED_SOURCE, whole_cpu_rescore_identical=True, cases_checked=2,
        decision="cpu-scheduled-diagonal-probe-passed", files_rehashed=inventory, **_flags())
    raw = json.dumps(closeout, sort_keys=True, separators=(",", ":")).encode()
    (root / "independent-closeout.json").write_bytes(raw)
    monkeypatch.setattr(dose.base, "output_path", lambda _source: root)
    monkeypatch.setattr(dose, "CLOSED_SHA256", _hash(raw))
    real_digest = dose.base.host.digest
    monkeypatch.setattr(dose.base.host, "digest", lambda path: (
        events.append(("digest", Path(path).name)), real_digest(path))[1])
    real_file_bytes = dose.base.files.file_bytes
    monkeypatch.setattr(dose.base.files, "file_bytes", lambda path, **kwargs: (
        events.append(("file-bytes", Path(path).name)), real_file_bytes(path, **kwargs))[1])
    monkeypatch.setattr(dose.base, "prerequisite", lambda: (
        events.append(("old-matrix-prerequisite", None)),
        ({"source": "synthetic-old-matrix"}, CP))[1])
    return root


def _install_run(monkeypatch, tmp_path, events, *, bad_property=None, collect_fault=None,
                 full=True, candidate=None, matched=True, delivered=True, zero_pass=True,
                 drift=False):
    closeout_root = _install_closeout(monkeypatch, tmp_path, events)
    monkeypatch.setattr(dose.base.host, "ROOT", tmp_path)
    (tmp_path / "artifacts/evaluations").mkdir(parents=True, exist_ok=True)
    props = dict(MainPID=str(os.getpid()), ActiveState="active", RuntimeMaxUSec="3min",
        MemoryMax=str(2 * 1024**3), CPUQuotaPerSecUSec="2s", Nice="10", KillMode="control-group")
    if bad_property:
        props[bad_property] = "unbounded"
    identity_values = iter([{"source": SOURCE, "clean": True},
        {"source": "e" * 40, "clean": True}] if drift else [{"source": SOURCE, "clean": True}] * 20)
    def read(*cmd):
        if cmd[:4] == ("systemctl", "--user", "show", dose.service_name(SOURCE)):
            events.append(("service-property", cmd[-2]))
            return props[cmd[-2]]
        events.append(("protected-service", cmd[0]))
        return "inactive"
    monkeypatch.setattr(dose.base.host, "read", read)
    monkeypatch.setattr(dose.base.host, "identity", lambda source: (
        events.append(("identity", source)), next(identity_values))[1])
    monkeypatch.setattr(dose.base, "protected_state", lambda: {"protected": "inactive"})
    monkeypatch.setattr(dose.base.retained.d0, "filmbrain_state", lambda: {"state": "unchanged"})
    monkeypatch.setattr(dose.base.profile, "checked_receipt", lambda: (
        events.append(("profile", None)) or {"test_only_synthetic_profile": True}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(dose.torch.cuda, "is_initialized", lambda: False)
    tick = [1000.]
    def monotonic():
        tick[0] += .01
        return tick[0]
    monkeypatch.setattr(dose.time, "monotonic", monotonic)
    monkeypatch.setattr(dose, "check_window", lambda **_kwargs: events.append(("window", None)))
    monkeypatch.setattr(dose.base.schedule, "declaration",
        lambda source, stage, split, cells: dict(source=source, stage=stage, split=split, cell_ids=list(cells)))
    import mjlab_microduck.stance_recovery_schedule_runtime as runtime_module
    class FakeRuntime:
        instances = []
        def __init__(self, declaration, *, device):
            events.append(("runtime", declaration["cell_ids"][0], device))
            self.binding = {"compiled": "synthetic-same"}
            self.declaration = declaration
            self.instances.append(self)
    monkeypatch.setattr(runtime_module, "ScheduledRecoveryRuntime", FakeRuntime)
    monkeypatch.setattr(dose.base, "seed_cpu", lambda: events.append(("seed", len(FakeRuntime.instances))))

    counters = {"collect": 0, "verify": 0, "prefix": 0}
    raws = [b"synthetic-zero", b"synthetic-2n", b"synthetic-4n"]
    def collect(env, _cp, *, deadline_monotonic):
        index = counters["collect"]
        events.append(("collect", index, env.declaration["cell_ids"][0]))
        counters["collect"] += 1
        if collect_fault and collect_fault[0] == index:
            raise collect_fault[1]
        if collect_fault == "overbudget":
            tick[0] += 61.
        return {"collection": {"policy_ticks": 250}}
    monkeypatch.setattr(dose.base.evidence, "collect", collect)
    monkeypatch.setattr(dose.base.evidence, "encode", lambda _value: raws[counters["verify"]])

    candidates = candidate if candidate is not None else [zero_pass, True, True]
    def verify(raw, expected, cp, declaration, compiled):
        index = counters["verify"]
        counters["verify"] += 1
        events.append(("verify", index, raw))
        is_full = full if type(full) is bool else full[index]
        return {"cell": declaration["cell_ids"][0],
            "collection": {"elapsed_seconds": .2},
            "numerical_diagnostic": {"complete_first_attempt": is_full,
                "candidate_pass": candidates[index]},
            "pulse": {"checked_physics_steps": 2500 if is_full else 500,
                "complete_pulse_delivery": delivered if isinstance(delivered, bool) else delivered[index],
                "complete_phase_checks": delivered if isinstance(delivered, bool) else delivered[index]},
            **_flags()}
    monkeypatch.setattr(dose.base.evidence, "verify", verify)
    def prefix_hash(_value, step):
        index = counters["prefix"]
        counters["prefix"] += 1
        events.append(("prefix", index, step))
        return "matched-prefix" if matched is True or (type(matched) is list and matched[index]) else f"different-{index}"
    monkeypatch.setattr(dose.base.evidence, "prefix_hash", prefix_hash)
    monkeypatch.setattr(dose.torch, "load", lambda *_a, **_k: {"synthetic": True})
    monkeypatch.setattr(dose.base.retained, "write_capture", lambda path, raw: (
        events.append(("capture", Path(path).name)), Path(path).open("xb").write(raw)))
    return FakeRuntime, counters, events, closeout_root


def test_launch_cutoff_requires_strictly_more_than_six_minutes():
    dose.check_window(launching=True, now=dose.base.baseline.CUTOFF - 361)
    with pytest.raises(ValueError, match="08:00 Shanghai cutoff"):
        dose.check_window(launching=True, now=dose.base.baseline.CUTOFF - 360)
    dose.check_window(now=dose.base.baseline.CUTOFF - 1)


def test_completed_closeout_and_inventory_are_verified_before_old_matrix_prerequisite(monkeypatch, tmp_path):
    events = []
    root = _install_closeout(monkeypatch, tmp_path, events)
    receipt, cp = dose.prerequisite()
    assert receipt["source"] == dose.CLOSED_SOURCE and cp == CP
    closeout_index = events.index(("file-bytes", "independent-closeout.json"))
    digest_indices = [i for i, item in enumerate(events) if item[0] == "digest"]
    old_index = events.index(("old-matrix-prerequisite", None))
    assert digest_indices and max(digest_indices) < old_index
    assert closeout_index < digest_indices[0] < old_index
    assert {path.name for path in root.iterdir()} == {"case-0.pt", "case-1.pt", "independent-closeout.json"}


@pytest.mark.parametrize("damage", ["hash", "extra", "artifact"])
def test_bad_closed_prerequisite_refuses_before_old_matrix_and_parent_bytes(monkeypatch, tmp_path, damage):
    events = []
    root = _install_closeout(monkeypatch, tmp_path, events)
    if damage == "hash":
        (root / "independent-closeout.json").write_bytes(b"tampered")
    elif damage == "extra":
        (root / "unlisted.bin").write_bytes(b"unlisted")
    else:
        (root / "case-0.pt").write_bytes(b"tampered")
    with pytest.raises((ValueError, KeyError)):
        dose.prerequisite()
    assert not any(item[0] == "old-matrix-prerequisite" for item in events)


def test_service_caps_and_cuda_gate_refuse_before_identity_or_prerequisite(monkeypatch, tmp_path):
    events = []
    _install_run(monkeypatch, tmp_path, events, bad_property="RuntimeMaxUSec")
    with pytest.raises(ValueError, match="exact capped dose-screen service"):
        dose.run(SOURCE)
    assert not any(item[0] in {"identity", "file-bytes", "old-matrix-prerequisite", "runtime"} for item in events)


def test_cuda_visible_entry_refuses_before_service_observation(monkeypatch, tmp_path):
    events = []
    _install_run(monkeypatch, tmp_path, events)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only dose screen"):
        dose.run(SOURCE)
    assert not any(item[0] in {"service-property", "identity", "old-matrix-prerequisite"} for item in events)


def test_three_exact_fresh_seeded_cases_and_verify_before_prefix(monkeypatch, tmp_path):
    events = []
    runtime, counters, _, _ = _install_run(monkeypatch, tmp_path, events)
    dose.run(SOURCE)
    assert [item[1] for item in events if item[0] == "runtime"] == list(CELLS)
    assert [item[0] for item in events if item[0] in {"seed", "runtime"}] == [
        "seed", "runtime", "seed", "runtime", "seed", "runtime"]
    assert counters == {"collect": 3, "verify": 3, "prefix": 3}
    for index in range(3):
        verify_i = next(i for i, item in enumerate(events) if item[:2] == ("verify", index))
        prefix_i = next(i for i, item in enumerate(events) if item[0] == "prefix" and i > verify_i)
        assert verify_i < prefix_i and events[prefix_i][2] == dose.PREFIX_STEP == 250
    assert len(runtime.instances) == 3
    root = dose.output_path(SOURCE)
    launch = json.loads((root / "launch.json").read_text())
    assert [d["cell_ids"][0] for d in launch["declarations"]] == list(CELLS)
    assert launch["evaluation_seed"] == 671 and launch["prefix_step"] == 250
    assert launch["service_properties"]["RuntimeMaxUSec"] == "3min"
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "frozen-dose-screen-no-deficit"
    assert report["optimizer_steps"] == 0 and report["genuine_cpu_first_attempts"] == 3
    assert all(report[key] is False for key in dose.base.baseline.FALSE_FLAGS)
    for index in range(3):
        case = json.loads((root / f"case-{index}.json").read_text())
        replay = json.loads((root / f"case-{index}-replay.json").read_text())
        assert all(case[key] is False for key in dose.base.baseline.FALSE_FLAGS)
        assert all(replay[key] is False for key in dose.base.baseline.FALSE_FLAGS)


def test_terminal_failure_may_continue_as_next_fresh_case_but_zero_must_pass(monkeypatch, tmp_path):
    events = []
    runtime, counters, _, _ = _install_run(monkeypatch, tmp_path, events,
        candidate=[True, False, True])
    dose.run(SOURCE)
    root = dose.output_path(SOURCE)
    assert counters["collect"] == 3 and len(runtime.instances) == 3
    assert (root / "case-1.pt").is_file() and (root / "case-2.pt").is_file()
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "frozen-dose-screen-measured-deficit"
    assert report["failed_cells"] == [CELLS[1]]
    assert all(report[key] is False for key in dose.base.baseline.FALSE_FLAGS)

    events2 = []
    runtime2, counters2, _, _ = _install_run(monkeypatch, tmp_path / "zero-fails", events2,
        zero_pass=False)
    with pytest.raises(ValueError, match="failed zero control cannot establish a dose deficit"):
        dose.run(SOURCE)
    assert counters2["collect"] == 3 and len(runtime2.instances) == 3
    report2 = json.loads((dose.output_path(SOURCE) / "report.json").read_text())
    assert "failed zero control" in report2["error"]
    assert all(report2[key] is False for key in dose.base.baseline.FALSE_FLAGS)


@pytest.mark.parametrize("matched,delivered", [(False, True), (True, [True, False, True])])
def test_unmatched_prefix_or_incomplete_force_delivery_is_inconclusive(monkeypatch, tmp_path, matched, delivered):
    events = []
    _install_run(monkeypatch, tmp_path, events, matched=matched, delivered=delivered,
        candidate=[True, False, True])
    dose.run(SOURCE)
    report = json.loads((dose.output_path(SOURCE) / "report.json").read_text())
    assert report["decision"] == "cpu-dose-screen-inconclusive"
    assert report["optimizer_steps"] == 0 and all(report[key] is False for key in dose.base.baseline.FALSE_FLAGS)


def test_terminal_failure_may_continue_but_wall_timeout_retains_failure_and_aborts(monkeypatch, tmp_path):
    events = []
    # This is a complete first attempt whose physics outcome is a terminal
    # failure; it may be scored and followed by the next fresh case, not retried.
    runtime, counters, _, _ = _install_run(monkeypatch, tmp_path, events,
        full=[True, True, True], candidate=[True, False, True])
    dose.run(SOURCE)
    assert counters["collect"] == 3 and len(runtime.instances) == 3
    report = json.loads((dose.output_path(SOURCE) / "report.json").read_text())
    assert report["decision"] == "frozen-dose-screen-measured-deficit"

    events2 = []
    runtime2, counters2, _, _ = _install_run(monkeypatch, tmp_path / "timeout", events2,
        collect_fault=(1, TimeoutError("synthetic wall timeout")))
    with pytest.raises(TimeoutError, match="synthetic wall timeout"):
        dose.run(SOURCE)
    root = dose.output_path(SOURCE)
    assert counters2["collect"] == 2 and len(runtime2.instances) == 2
    assert (root / "case-0.pt").is_file() and (root / "case-0-replay.json").is_file()
    failure = json.loads((root / "case-1-failure.json").read_text())
    assert failure["index"] == 1 and failure["current_case_trace_retained"] is False
    assert not (root / "case-2.pt").exists()
    report2 = json.loads((root / "report.json").read_text())
    assert report2["error_type"] == "TimeoutError" and all(report2[key] is False for key in dose.base.baseline.FALSE_FLAGS)


def test_returned_overbudget_capture_is_retained_scored_then_aborts_without_retry(monkeypatch, tmp_path):
    events = []
    runtime, counters, _, _ = _install_run(monkeypatch, tmp_path, events, collect_fault="overbudget")
    with pytest.raises(ValueError, match="over-budget dose trace retained and scored; no retry"):
        dose.run(SOURCE)
    root = dose.output_path(SOURCE)
    assert counters["collect"] == counters["verify"] == 1 and len(runtime.instances) == 1
    assert (root / "case-0.pt").is_file() and (root / "case-0-replay.json").is_file()
    assert not (root / "case-1.pt").exists()
    collect_i = next(i for i, item in enumerate(events) if item[0] == "collect")
    capture_i = next(i for i, item in enumerate(events) if item == ("capture", "case-0.pt"))
    verify_i = next(i for i, item in enumerate(events) if item[0] == "verify")
    assert collect_i < capture_i < verify_i
    report = json.loads((root / "report.json").read_text())
    assert "over-budget dose trace retained and scored" in report["error"]
    assert all(report[key] is False for key in dose.base.baseline.FALSE_FLAGS)
