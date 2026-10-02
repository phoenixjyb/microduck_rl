"""Shallow local orchestration checks for the bounded scheduled CPU probe.

Every checkpoint, host, service and runtime below is a synthetic test seam;
these tests are not evidence of the pinned archive or a WSL execution.
"""

from hashlib import sha256 as real_sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_recovery_schedule_probe as probe


SOURCE = "b" * 40
CP = b"synthetic immutable checkpoint bytes"
MATRIX_FILES = {f"matrix-{index:02}.bin": bytes([index + 1]) for index in range(18)}
PREFIX_FILES = {
    "launch.json": b"synthetic prefix launch",
    "qualification.json": b"synthetic CPU qualification",
    "prefix.pt": b"synthetic CPU prefix",
    "checkpoint.pt": CP,
}


def _hash(value):
    return real_sha256(value).hexdigest()


def _false_flags():
    return dict(probe.baseline.FALSE_FLAGS)


def _install_prerequisite(monkeypatch, tmp_path, events):
    """Make a locally rehashed fake archive to exercise prerequisite ordering."""
    matrix_root = tmp_path / "artifacts/evaluations/stance-wsl-d1-frozen-five-case-e9b9d4983637"
    # `retained.output_path` is patched below; make both roots ordinary temp files.
    prefix_root = tmp_path / "old-prefix"
    matrix_root.mkdir(parents=True)
    prefix_root.mkdir()
    report_files = {}
    closeout_files = {}
    for name, raw in MATRIX_FILES.items():
        (matrix_root / name).write_bytes(raw)
        report_files[name] = _hash(raw)
        closeout_files[name] = dict(bytes=len(raw), sha256=_hash(raw))
    report = dict(source=probe.MATRIX_SOURCE, decision="frozen-five-case-baseline-passed",
        files=report_files, cases_checked=5, **_false_flags())
    report_raw = (json.dumps(report, sort_keys=True, separators=(",", ":"))).encode()
    (matrix_root / "report.json").write_bytes(report_raw)
    closeout_files["report.json"] = dict(bytes=len(report_raw), sha256=_hash(report_raw))
    prefix_inventory = {}
    for name, raw in PREFIX_FILES.items():
        (prefix_root / name).write_bytes(raw)
        prefix_inventory[name] = dict(bytes=len(raw), sha256=_hash(raw))
    closeout = dict(source=probe.MATRIX_SOURCE, whole_cpu_rescore_identical=True, cases_checked=5,
        decision="frozen-five-case-baseline-passed", files_rehashed=closeout_files,
        prefix_files_rehashed=prefix_inventory, **_false_flags())
    closeout_raw = json.dumps(closeout, sort_keys=True, separators=(",", ":")).encode()
    (matrix_root / "independent-closeout.json").write_bytes(closeout_raw)

    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    monkeypatch.setattr(probe.retained, "output_path", lambda _source: prefix_root)
    monkeypatch.setattr(probe, "MATRIX_CLOSEOUT_SHA256", _hash(closeout_raw))
    monkeypatch.setattr(probe, "MATRIX_REPORT_SHA256", _hash(report_raw))
    monkeypatch.setattr(probe.baseline, "CHECKPOINT_SHA256", _hash(CP))
    original_file_bytes = probe.files.file_bytes

    def file_bytes(path, **kwargs):
        events.append(("file-bytes", Path(path).name))
        return original_file_bytes(path, **kwargs)

    original_digest = probe.host.digest

    def digest(path):
        events.append(("digest", Path(path).name))
        return original_digest(path)

    monkeypatch.setattr(probe.files, "file_bytes", file_bytes)
    monkeypatch.setattr(probe.host, "digest", digest)
    return matrix_root, prefix_root


def _install_host(monkeypatch, tmp_path, events, *, bad_property=None, identity_values=None):
    props = dict(MainPID=str(os.getpid()), ActiveState="active", RuntimeMaxUSec="4min",
        MemoryMax=str(2 * 1024**3), CPUQuotaPerSecUSec="2s", Nice="10", KillMode="control-group")
    if bad_property:
        props[bad_property] = "unbounded"
    def read(*cmd):
        if cmd[:4] == ("systemctl", "--user", "show", probe.service_name(SOURCE)):
            events.append(("service-property", cmd[-2]))
            return props[cmd[-2]]
        events.append(("protected-service", cmd[0]))
        return "inactive"
    monkeypatch.setattr(probe.host, "read", read)
    identities = iter(identity_values if identity_values is not None else [dict(source=SOURCE, clean=True)] * 10)
    monkeypatch.setattr(probe.host, "identity", lambda source: (
        events.append(("identity", source)), next(identities))[1])
    monkeypatch.setattr(probe.host.execution, "service_commands", lambda _services: [("protected", ("systemctl", "inactive"))])
    monkeypatch.setattr(probe.retained.d0, "filmbrain_state", lambda: {"state": "unchanged"})
    monkeypatch.setattr(probe.profile, "checked_receipt", lambda: (events.append(("profile", None)) or {"cpu": "synthetic"}))
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)


def _install_run_seams(monkeypatch, tmp_path, events, *, collect_values=None, collect_error=None,
                       score_full=True, identity_values=None, bad_property=None, prerequisites=True,
                       collection_delay=False):
    if prerequisites:
        _install_prerequisite(monkeypatch, tmp_path, events)
    _install_host(monkeypatch, tmp_path, events, bad_property=bad_property, identity_values=identity_values)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)

    tick = [1000.0]
    def monotonic():
        tick[0] += .01
        return tick[0]
    monkeypatch.setattr(probe.time, "monotonic", monotonic)
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: events.append(("window", None)))

    # The new parent loader and historical archive loader are mocked as an
    # explicitly synthetic checkpoint seam; actor shadow still must precede envs.
    actor = object()
    frozen = object()
    parent_result = {"actor": actor, "critic": object(), "receipt": {"synthetic": True}}
    monkeypatch.setattr(probe.parent, "load_parent", lambda _cp: (events.append(("parent-load", None)) or parent_result))
    monkeypatch.setattr(probe.retained.evaluation.checkpoint, "load_lean_replication_evaluation",
        lambda *_args: (events.append(("frozen-load", None)) or (frozen, {"synthetic": True})))
    def infer(model, obs):
        events.append(("actor-shadow", model is actor, model is frozen))
        return torch.zeros((2, 10))
    monkeypatch.setattr(probe.retained.evaluation.checkpoint, "infer", infer)

    class FakeRuntime:
        instances = []
        def __init__(self, declaration, *, device):
            events.append(("runtime", declaration["cell_ids"][0], device))
            self.declaration = declaration
            self.binding = {"plant": "same-compiled-synthetic"}
            self.instances.append(self)
    monkeypatch.setattr(probe.schedule, "declaration",
        lambda source, stage, split, cells: dict(source=source, stage=stage, split=split, cell_ids=list(cells)))
    import mjlab_microduck.stance_recovery_schedule_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "ScheduledRecoveryRuntime", FakeRuntime)
    monkeypatch.setattr(probe, "seed_cpu", lambda: events.append(("seed", len(FakeRuntime.instances))))

    raw_values = [b"synthetic complete trace zero", b"synthetic complete trace diagonal"]
    calls = {"collect": 0, "verify": 0, "prefix": 0}
    def collect(env, _cp, *, deadline_monotonic):
        index = calls["collect"]
        events.append(("collect", index, env.declaration["cell_ids"][0]))
        calls["collect"] += 1
        if collect_error:
            raise RuntimeError("synthetic collection failed")
        if collection_delay:
            tick[0] += 61.
        if collect_values is not None:
            return collect_values[index]
        return {"collection": {"policy_ticks": 250}}
    monkeypatch.setattr(probe.evidence, "collect", collect)
    monkeypatch.setattr(probe.evidence, "encode", lambda value: raw_values[calls["verify"]])
    def verify(raw, expected_sha, cp, declaration, compiled):
        index = calls["verify"]
        events.append(("verify", index, raw))
        calls["verify"] += 1
        full = score_full if isinstance(score_full, bool) else score_full[index]
        return {
            "numerical_diagnostic": {"gates": {"full_duration": full}, "complete_first_attempt": full,
                "candidate_pass": False},
            "collection": {"policy_ticks": 250 if full else 52, "elapsed_seconds": .25},
            "pulse": {"checked_physics_steps": 2500 if full else 520,
                "complete_pulse_delivery": full, "complete_phase_checks": full},
            **_false_flags(),
        }
    monkeypatch.setattr(probe.evidence, "verify", verify)
    monkeypatch.setattr(probe.evidence, "prefix_hash", lambda value, step: (
        events.append(("prefix", step)), calls.__setitem__("prefix", calls["prefix"] + 1), "same-prefix")[2])
    monkeypatch.setattr(probe.torch, "load", lambda *_args, **_kwargs: {"synthetic": True})
    monkeypatch.setattr(probe.retained, "write_capture", lambda path, raw: (
        events.append(("capture", Path(path).name)), Path(path).open("xb").write(raw)))
    monkeypatch.setattr(probe.gc, "collect", lambda: events.append(("gc", None)))
    return FakeRuntime, events, calls


def _completed_archive(monkeypatch, tmp_path, events):
    return _install_run_seams(monkeypatch, tmp_path, events)


def test_launch_reserve_is_strictly_greater_than_eight_minutes():
    probe.check_window(launching=True, now=probe.baseline.CUTOFF - 481)
    with pytest.raises(ValueError, match="08:00 Shanghai cutoff"):
        probe.check_window(launching=True, now=probe.baseline.CUTOFF - 480)
    probe.check_window(now=probe.baseline.CUTOFF - 1)


def test_prerequisite_rehashes_all_matrix_and_prefix_inventory_before_parent_bytes(monkeypatch, tmp_path):
    events = []
    _, prefix = _install_prerequisite(monkeypatch, tmp_path, events)
    pins, raw = probe.prerequisite()
    assert pins["source"] == probe.MATRIX_SOURCE and raw == CP
    assert len(MATRIX_FILES) + 1 == 19 and len(PREFIX_FILES) == 4  # Includes report.json.
    # The prefix inventory itself hashes checkpoint.pt first. The final bounded
    # parent-byte read must follow all inventory hashes, not that hash's read.
    parent_read = max(i for i, event in enumerate(events) if event == ("file-bytes", "checkpoint.pt"))
    assert events.index(("file-bytes", "independent-closeout.json")) < parent_read
    assert events.index(("file-bytes", "report.json")) < parent_read
    assert sum(1 for item in events[:parent_read] if item[0] == "digest") == len(MATRIX_FILES) + 1 + len(PREFIX_FILES)
    assert all(any(item == ("digest", name) for item in events[:parent_read])
        for name in (*MATRIX_FILES, "report.json", *PREFIX_FILES))
    assert prefix.joinpath("checkpoint.pt").read_bytes() == CP


@pytest.mark.parametrize("damage", ["closeout-hash", "report-hash", "matrix-file", "matrix-extra",
    "prefix-file", "prefix-extra", "checkpoint"])
def test_prerequisite_hash_or_inventory_mismatch_fails_before_parent_loader(monkeypatch, tmp_path, damage):
    events = []
    matrix_root, prefix = _install_prerequisite(monkeypatch, tmp_path, events)
    if damage == "closeout-hash":
        (matrix_root / "independent-closeout.json").write_bytes(b"tampered")
    elif damage == "report-hash":
        (matrix_root / "report.json").write_bytes(b"tampered")
    elif damage == "matrix-file":
        (matrix_root / next(iter(MATRIX_FILES))).write_bytes(b"tampered")
    elif damage == "matrix-extra":
        (matrix_root / "unlisted.bin").write_bytes(b"unlisted")
    elif damage == "prefix-file":
        (prefix / "qualification.json").write_bytes(b"tampered")
    elif damage == "prefix-extra":
        (prefix / "unlisted.bin").write_bytes(b"unlisted")
    else:
        (prefix / "checkpoint.pt").write_bytes(b"tampered")
    with pytest.raises((ValueError, KeyError)):
        probe.prerequisite()
    assert not any(item[0] == "parent-load" for item in events)


@pytest.mark.parametrize("bad_property", ["RuntimeMaxUSec", "MemoryMax", "CPUQuotaPerSecUSec", "KillMode"])
def test_service_cap_refusal_precedes_identity_parent_and_output_root(monkeypatch, tmp_path, bad_property):
    events = []
    _install_run_seams(monkeypatch, tmp_path, events, bad_property=bad_property, prerequisites=False)
    with pytest.raises(ValueError, match="capped genuine CPU scheduled service"):
        probe.run(SOURCE)
    assert not any(item[0] in {"identity", "parent-load", "runtime"} for item in events)
    assert not probe.output_path(SOURCE).exists()


@pytest.mark.parametrize("cuda_state", ["visible", "initialized"])
def test_cpu_entry_gate_refuses_before_service_identity_or_checkpoint(monkeypatch, tmp_path, cuda_state):
    events = []
    _install_run_seams(monkeypatch, tmp_path, events, prerequisites=False)
    if cuda_state == "visible":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    else:
        monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CPU-only scheduled probe entry"):
        probe.run(SOURCE)
    assert not any(item[0] in {"service-property", "identity", "parent-load"} for item in events)


def test_two_ordered_fresh_attempts_shadow_parent_before_runtime_and_verify_before_prefix(monkeypatch, tmp_path):
    events = []
    runtime, _, calls = _completed_archive(monkeypatch, tmp_path, events)
    probe.run(SOURCE)
    assert [item[1] for item in events if item[0] == "runtime"] == ["zero-wrench", "diagonal-++-2n-10steps-t375"]
    assert [item[0] for item in events if item[0] in ("seed", "runtime")][:4] == ["seed", "runtime", "seed", "runtime"]
    assert events.index(("parent-load", None)) < events.index(("frozen-load", None))
    assert events.index(("frozen-load", None)) < next(i for i, item in enumerate(events) if item[0] == "actor-shadow")
    assert next(i for i, item in enumerate(events) if item[0] == "actor-shadow") < next(i for i, item in enumerate(events) if item[0] == "runtime")
    for index in range(2):
        verify_index = next(i for i, item in enumerate(events) if item[:2] == ("verify", index))
        prefix_index = next(i for i, item in enumerate(events) if item[0] == "prefix" and i > verify_index)
        assert verify_index < prefix_index
    assert calls == {"collect": 2, "verify": 2, "prefix": 2}
    root = probe.output_path(SOURCE)
    assert (root / "case-0.pt").is_file() and (root / "case-1.pt").is_file()
    launch = json.loads((root / "launch.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert launch["actual_first_attempts_have_no_counter_support_hooks"] is True
    assert report["genuine_cpu_first_attempts"] == 2
    assert all(report[key] is False for key in probe.baseline.FALSE_FLAGS)
    assert len(runtime.instances) == 2


def test_partial_first_attempt_is_retained_and_stops_without_second_case(monkeypatch, tmp_path):
    events = []
    runtime, _, calls = _install_run_seams(monkeypatch, tmp_path, events, score_full=False)
    with pytest.raises(ValueError, match="partial CPU first attempt retained; stop without retry"):
        probe.run(SOURCE)
    root = probe.output_path(SOURCE)
    assert len(runtime.instances) == 1 and calls["collect"] == calls["verify"] == 1
    assert (root / "case-0.pt").is_file() and (root / "case-0-replay.json").is_file()
    assert not (root / "case-1.pt").exists() and not (root / "case-1.json").exists()
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "cpu-scheduled-diagonal-diagnostic-failed"
    assert "partial CPU first attempt retained" in report["error"]
    assert all(report[key] is False for key in probe.baseline.FALSE_FLAGS)


def test_collection_failure_writes_failure_note_and_never_retries(monkeypatch, tmp_path):
    events = []
    runtime, _, calls = _install_run_seams(monkeypatch, tmp_path, events, collect_error=True)
    with pytest.raises(RuntimeError, match="synthetic collection failed"):
        probe.run(SOURCE)
    root = probe.output_path(SOURCE)
    failure = json.loads((root / "case-0-failure.json").read_text())
    assert failure["current_case_trace_retained"] is False
    assert failure["index"] == 0 and calls["collect"] == 1 and len(runtime.instances) == 1
    assert not (root / "case-0.pt").exists() and not (root / "case-1.pt").exists()
    report = json.loads((root / "report.json").read_text())
    assert report["error_type"] == "RuntimeError" and all(report[key] is False for key in probe.baseline.FALSE_FLAGS)


def test_returned_overbudget_full_capture_is_retained_and_scored_before_stop(monkeypatch, tmp_path):
    events = []
    runtime, _, calls = _install_run_seams(monkeypatch, tmp_path, events, collection_delay=True)
    with pytest.raises(ValueError, match="over-budget CPU collection retained and scored; stop without retry"):
        probe.run(SOURCE)
    root = probe.output_path(SOURCE)
    assert len(runtime.instances) == 1 and calls["collect"] == calls["verify"] == 1
    assert (root / "case-0.pt").is_file() and (root / "case-0.json").is_file()
    assert (root / "case-0-replay.json").is_file()
    assert not (root / "case-1.pt").exists()
    collect_index = next(i for i, item in enumerate(events) if item[0] == "collect")
    capture_index = next(i for i, item in enumerate(events) if item == ("capture", "case-0.pt"))
    verify_index = next(i for i, item in enumerate(events) if item[0] == "verify")
    assert collect_index < capture_index < verify_index
    report = json.loads((root / "report.json").read_text())
    assert "over-budget CPU collection retained and scored" in report["error"]
    assert all(report[key] is False for key in probe.baseline.FALSE_FLAGS)


def test_source_drift_fails_before_first_collection_and_retains_failure_report(monkeypatch, tmp_path):
    events = []
    identities = [dict(source=SOURCE, clean=True), dict(source="c" * 40, clean=True)]
    runtime, _, calls = _install_run_seams(monkeypatch, tmp_path, events, identity_values=identities)
    with pytest.raises(ValueError, match="unchanged CPU source and unrelated/protected services"):
        probe.run(SOURCE)
    root = probe.output_path(SOURCE)
    assert len(runtime.instances) == 1 and calls["collect"] == 0
    report = json.loads((root / "report.json").read_text())
    assert "unchanged CPU source" in report["error"]
    assert all(report[key] is False for key in probe.baseline.FALSE_FLAGS)
