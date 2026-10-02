"""Synthetic boundaries for the bounded CUDA child / independent CPU replay."""

import hashlib
import json
import os

import pytest
import torch

from mjlab_microduck import stance_disturbance_contract as contract
from mjlab_microduck import stance_disturbance_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical


SOURCE = "d" * 40
LAUNCH_SHA = "e" * 64


def test_cutoff_is_fixed_08_shanghai_with_strict_service_closeout_reserve():
    reserve = contract.SERVICE_SECONDS + contract.CLOSEOUT_SECONDS + 60
    probe.check_window(launching=True, now=contract.CUTOFF - reserve - 1)
    with pytest.raises(ValueError):
        probe.check_window(launching=True, now=contract.CUTOFF - reserve)
    probe.check_window(now=contract.CUTOFF - 1)
    with pytest.raises(ValueError):
        probe.check_window(now=contract.CUTOFF)


def test_output_and_service_names_are_source_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    assert probe.output_path(SOURCE) == tmp_path / "artifacts/evaluations" / ("stance-wsl-d0-force-fixture-" + SOURCE[:12])
    assert probe.service_name(SOURCE) == "microduck-wsl-d0-force-fixture-" + SOURCE[:12] + ".service"
    for value in ("D" * 40, "f" * 39, "z" * 40, True):
        with pytest.raises(ValueError):
            probe.output_path(value)


def _closeout_rows():
    summary = {"cases": 36, "complete_attempts": 4608}
    return [
        {"source": probe.FULL_SOURCE},
        {"decision": "lean-replication-passed",
         "launch_sha256": probe.PREREQUISITES[next(iter(probe.PREREQUISITES))],
         "summary": dict(summary)},
        {"source": probe.FULL_SOURCE, "whole_plan_bundle_score_verified": True,
         "selected_compiled_plant_verified": True, "cuda_initialized": False,
         "summary": dict(summary)},
        {"source": probe.FULL_SOURCE, "file_count": 329, "total_bytes": 22352528834,
         "byte_inventory_only": True},
        {"separate_readonly_rehash_verified": True},
    ]


def _install_closeout(monkeypatch, rows, *, bad_hash=False):
    ordered_paths = list(probe.PREREQUISITES)
    encoded = {path: json.dumps(row, sort_keys=True).encode() for path, row in zip(ordered_paths, rows)}
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: encoded[str(path).removeprefix(str(probe.host.ROOT) + "/")])
    monkeypatch.setattr(probe.files, "parse", lambda raw: json.loads(raw))
    expected_hashes = list(probe.PREREQUISITES.values())
    calls = iter(["0" * 64 if bad_hash else value for value in expected_hashes])

    class Digest:
        def hexdigest(self):
            return next(calls)

    monkeypatch.setattr(probe, "sha256", lambda _raw: Digest())


def test_nominal_closeout_matches_original_full_schema_without_report_source(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    rows = _closeout_rows()
    assert "source" not in rows[1]
    _install_closeout(monkeypatch, rows)
    result = probe.nominal_closeout()
    assert result == probe.PREREQUISITES


def test_nominal_closeout_rejects_hash_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    _install_closeout(monkeypatch, _closeout_rows(), bad_hash=True)
    with pytest.raises(ValueError, match="pinned complete nominal closeout"):
        probe.nominal_closeout()


@pytest.mark.parametrize("mutate", [
    lambda rows: rows[2].update(source="1" * 40),
    lambda rows: rows[1]["summary"].update(cases=35),
    lambda rows: rows[1]["summary"].update(complete_attempts=4607),
    lambda rows: rows[2].update(whole_plan_bundle_score_verified=False),
    lambda rows: rows[2].update(cuda_initialized=True),
    lambda rows: rows[3].update(file_count=328),
    lambda rows: rows[4].update(separate_readonly_rehash_verified=False),
])
def test_nominal_closeout_rejects_wrong_source_incomplete_counts_and_failed_replay(monkeypatch, tmp_path, mutate):
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    rows = _closeout_rows()
    mutate(rows)
    _install_closeout(monkeypatch, rows)
    with pytest.raises(ValueError):
        probe.nominal_closeout()


def _capture_record(raw):
    return dict(protocol=contract.PROTOCOL, source=SOURCE, launch_sha256=LAUNCH_SHA,
        capture_sha256=hashlib.sha256(raw).hexdigest(), capture_bytes=len(raw),
        elapsed_seconds=0.25,
        backend={"torch_device": "cuda:0", "warp_is_cuda": True}, integration_steps=5,
        policy_inferences=0, optimizer_steps=0, **contract.NO_ADMISSION)


def _install_capture_io(monkeypatch, tmp_path, record, raw=b"serialized capture"):
    monkeypatch.setattr(probe.files, "parse", lambda _raw: record)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs:
                        raw if str(path).endswith("capture.pt") else b"capture record")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_replay_capture_authenticates_raw_length_and_digest_before_torch_load(monkeypatch, tmp_path):
    raw = b"serialized capture"
    record = _capture_record(raw)
    _install_capture_io(monkeypatch, tmp_path, record, raw)
    value = {"plan": {"source": SOURCE}, "backend": record["backend"]}
    load_calls, replay_calls = [], []
    monkeypatch.setattr(probe.torch, "load", lambda stream, **kwargs: (load_calls.append((stream.read(), kwargs)), value)[1])
    monkeypatch.setattr(probe.fixture, "replay", lambda captured: (replay_calls.append(captured), {"replayed": True})[1])
    result = probe.replay_capture(tmp_path, {"source": SOURCE}, LAUNCH_SHA)
    assert result == {"replayed": True} and replay_calls == [value]
    assert load_calls == [(raw, {"map_location": "cpu", "weights_only": True})]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("damage", ["digest", "length", "source", "launch", "protocol", "backend",
                                    "steps", "elapsed", "optimizer", "policy", "admission"])
def test_replay_capture_rejects_untrusted_record_before_replay(monkeypatch, tmp_path, damage):
    raw = b"serialized capture"
    record = _capture_record(raw)
    if damage == "digest": record["capture_sha256"] = "0" * 64
    if damage == "length": record["capture_bytes"] += 1
    if damage == "source": record["source"] = "1" * 40
    if damage == "launch": record["launch_sha256"] = "1" * 64
    if damage == "protocol": record["protocol"] = "other-protocol"
    if damage == "backend": record["backend"] = {"torch_device": "cpu", "warp_is_cuda": False}
    if damage == "steps": record["integration_steps"] = 4
    if damage == "elapsed": record["elapsed_seconds"] = 0.
    if damage == "optimizer": record["optimizer_steps"] = 1
    if damage == "policy": record["policy_inferences"] = 1
    if damage == "admission": record["training_admitted"] = True
    _install_capture_io(monkeypatch, tmp_path, record, raw)
    load_calls = []
    monkeypatch.setattr(probe.torch, "load", lambda *_args, **_kwargs: load_calls.append(True))
    monkeypatch.setattr(probe.fixture, "replay", lambda *_args: pytest.fail("untrusted payload replayed"))
    with pytest.raises(ValueError):
        probe.replay_capture(tmp_path, {"source": SOURCE}, LAUNCH_SHA)
    assert not load_calls


def test_replay_capture_rejects_payload_plan_mismatch_and_visible_cuda(monkeypatch, tmp_path):
    raw = b"serialized capture"
    record = _capture_record(raw)
    _install_capture_io(monkeypatch, tmp_path, record, raw)
    value = {"plan": {"source": "1" * 40}, "backend": record["backend"]}
    load_calls = []
    monkeypatch.setattr(probe.torch, "load", lambda *_args, **_kwargs: (load_calls.append(True), value)[1])
    replayed = []
    monkeypatch.setattr(probe.fixture, "replay", lambda value: replayed.append(value))
    with pytest.raises(ValueError, match="source-bound actual force payload"):
        probe.replay_capture(tmp_path, {"source": SOURCE}, LAUNCH_SHA)
    assert not replayed
    monkeypatch.setattr(probe.torch, "load", lambda *_args, **_kwargs: {
        "plan": {"source": SOURCE}, "backend": record["backend"]})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only force solve replay"):
        probe.replay_capture(tmp_path, {"source": SOURCE}, LAUNCH_SHA)
    assert not replayed


def test_checked_requires_exact_source_manifest_and_plan(tmp_path, monkeypatch):
    root = tmp_path / "artifacts/evaluations" / ("stance-wsl-d0-force-fixture-" + SOURCE[:12])
    root.mkdir(parents=True)
    cpu_plan = {"source": SOURCE, "plant_sha256": "f" * 64}
    raw_capture = b"verified cpu capture"
    replay = {"maximum_solved_field_error": 0., "cuda_initialized": False, "force_path_replayed": True}
    qualification = {
        "protocol": "football-b1d-cpu-force-qualification-v1", "source": SOURCE,
        "capture_sha256": hashlib.sha256(raw_capture).hexdigest(), "capture_bytes": len(raw_capture),
        "fixture_plan_sha256": hashlib.sha256(canonical(cpu_plan).encode()).hexdigest(),
        "cpu_math_profile": probe.cpu_profile.expected_receipt(), "replay": replay,
        "elapsed_seconds": 0.25,
    }
    qualification_path = root / "cpu-qualification.json"
    qualification_path.write_text(json.dumps(qualification, sort_keys=True))
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    launch = {**cpu_plan, "cpu_qualification_sha256": qualification_sha, "binding": "exact"}
    (root / "launch.json").write_text(json.dumps(launch, sort_keys=True))
    (root / "cpu-capture.pt").write_bytes(raw_capture)
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files, "parse", json.loads)
    monkeypatch.setattr(probe, "launch_plan", lambda source, qualification_sha: {
        **cpu_plan, "source": source, "cpu_qualification_sha256": qualification_sha, "binding": "exact"})
    value = {"plan": cpu_plan, "backend": {"torch_device": "cpu", "warp_is_cuda": False}}
    load_calls = []
    monkeypatch.setattr(probe.torch, "load", lambda *_args, **_kwargs: (load_calls.append(True), value)[1])
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    replay_calls = []
    monkeypatch.setattr(probe.fixture, "replay", lambda item: (replay_calls.append(item), replay)[1])
    launch_sha = probe.host.digest(root / "launch.json")
    assert probe.checked(SOURCE, launch_sha) == launch
    assert replay_calls == [value]
    qualification["replay"] = {**replay, "maximum_solved_field_error": 1.}
    qualification_path.write_text(json.dumps(qualification, sort_keys=True))
    with pytest.raises(ValueError, match="unchanged D0"):
        probe.checked(SOURCE, launch_sha)
    qualification["replay"] = replay
    qualification_path.write_text(json.dumps(qualification, sort_keys=True))
    (root / "cpu-capture.pt").write_bytes(b"tampered cpu capture")
    with pytest.raises(ValueError, match="authenticated exact CPU physics qualification"):
        probe.checked(SOURCE, launch_sha)
    assert load_calls == [True] and replay_calls == [value]
    (root / "cpu-capture.pt").write_bytes(raw_capture)
    ticks = iter((100., 130.))
    with monkeypatch.context() as timed:
        timed.setattr(probe.time, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError, match="bounded repeated CPU qualification"):
            probe.checked(SOURCE, launch_sha)
    with pytest.raises(ValueError):
        probe.checked("a" * 40, launch_sha)
    monkeypatch.setattr(probe, "launch_plan", lambda *_args: {"source": SOURCE, "binding": "changed"})
    with pytest.raises(ValueError, match="unchanged D0"):
        probe.checked(SOURCE, launch_sha)


def test_cpu_profile_settings_do_not_unset_cuda_and_are_added_to_child_environment():
    settings = probe.cpu_profile.settings()
    assert settings == {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE", "OMP_NUM_THREADS": "1"}
    assert "CUDA_VISIBLE_DEVICES" not in settings


def test_prepare_cpu_capture_and_replay_are_persisted_before_plan(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    cpu_plan = {"source": SOURCE, "plant_sha256": "f" * 64}
    capture = {"plan": cpu_plan, "backend": {"torch_device": "cpu", "warp_is_cuda": False}}
    replay = {"force_path_replayed": True, "maximum_solved_field_error": 0., "cuda_initialized": False}
    events = []
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files.native, "_plain_path", lambda path: path)
    monkeypatch.setattr(probe.files, "write_json", lambda path, value: path.write_text(json.dumps(value)))
    monkeypatch.setattr(probe.smoke, "write_bytes", lambda path, raw: path.write_bytes(raw))
    monkeypatch.setattr(probe.files, "gpu_lease", lambda: pytest.fail("prepare cannot acquire GPU lease"))
    monkeypatch.setattr(probe.cpu_profile, "checked_receipt", lambda: {"profile": "cpu-checked"})
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def capture_cases(source, device):
        assert source == SOURCE and device == "cpu"
        events.append("cpu-capture")
        return capture

    def replay_capture(value):
        assert value is capture
        events.append("cpu-replay")
        return replay

    def launch_plan(source, qualification_sha):
        assert source == SOURCE and len(qualification_sha) == 64
        if qualification_sha == "0" * 64:
            events.append("preparation-input-plan")
        else:
            assert (root / "cpu-capture.pt").is_file() and (root / "cpu-qualification.json").is_file()
            events.append("final-launch-plan")
        return {**cpu_plan, "cpu_qualification_sha256": qualification_sha}

    monkeypatch.setattr(probe.fixture, "capture_cases", capture_cases)
    monkeypatch.setattr(probe.fixture, "replay", replay_capture)
    monkeypatch.setattr(probe, "launch_plan", launch_plan)
    result = probe.prepare(SOURCE)
    assert events == ["preparation-input-plan", "cpu-capture", "cpu-replay", "final-launch-plan"]
    assert result["launch_sha256"] == probe.host.digest(root / "launch.json")
    assert {path.name for path in root.iterdir()} == {"cpu-capture.pt", "cpu-qualification.json", "launch.json"}
    qualification = json.loads((root / "cpu-qualification.json").read_text())
    assert qualification["source"] == SOURCE and qualification["replay"] == replay
    assert qualification["cpu_math_profile"] == {"profile": "cpu-checked"}
    assert qualification["capture_bytes"] == (root / "cpu-capture.pt").stat().st_size


def test_prepare_rejects_cpu_qualification_at_30_second_boundary(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files.native, "_plain_path", lambda path: path)
    monkeypatch.setattr(probe, "launch_plan", lambda source, _sha: {"source": source})
    monkeypatch.setattr(probe.fixture, "capture_cases", lambda source, _device: {
        "plan": {"source": source}, "backend": {"torch_device": "cpu", "warp_is_cuda": False}})
    monkeypatch.setattr(probe.fixture, "replay", lambda _value: {
        "maximum_solved_field_error": 0., "cuda_initialized": False})
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ticks = iter((100., 130.))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="bounded CPU physics qualification"):
        probe.prepare(SOURCE)


def test_child_exhausted_cpu_entry_reserve_stops_before_cuda_or_capture(monkeypatch):
    started = 100.
    ticks = iter((started + 1, started + contract.CHILD_SECONDS - contract.GPU_CAPTURE_RESERVE -
                  contract.CHILD_CLOSEOUT_RESERVE))
    events = []
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe.smoke, "inherited_lease", lambda fd: events.append(("lease", fd)))
    monkeypatch.setattr(probe, "checked", lambda *_args: events.append(("checked", True)) or {"source": SOURCE})
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: pytest.fail("CUDA availability queried"))
    monkeypatch.setattr(probe.fixture, "capture_cases", lambda *_args: pytest.fail("physics capture started"))
    with pytest.raises(ValueError, match="CPU entry leaves declared GPU and closeout reserves"):
        probe.child(SOURCE, LAUNCH_SHA, 73, started)
    assert events == [("lease", 73), ("checked", True)]


def test_supervisor_checks_service_lease_guard_and_retains_failed_report(tmp_path, monkeypatch):
    root = tmp_path / "artifacts/evaluations" / ("stance-wsl-d0-force-fixture-" + SOURCE[:12])
    root.mkdir(parents=True)
    (root / "launch.json").write_text("synthetic launch")
    (root / "cpu-capture.pt").write_bytes(b"synthetic cpu capture")
    (root / "cpu-qualification.json").write_text("synthetic qualification")
    launch = {"source": SOURCE, "preserved_filmbrain": {
        unit: {key: ("active" if key == "ActiveState" else "0") for key in ("ActiveState", "MainPID", "NRestarts")}
        for unit in ("recomo-filmbrain-observatory.service", "recomo-filmbrain-video-playground.service")}}
    props = {"MainPID": str(os.getpid()), "RuntimeMaxUSec": "3min", "KillMode": "control-group",
        "ActiveState": "active", "MemoryMax": str(2 * 1024**3), "CPUQuotaPerSecUSec": "2s", "Nice": "10"}
    monkeypatch.setattr(probe.host, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe, "service_name", lambda _source: "fixture-d0.service")
    monkeypatch.setattr(probe.host, "read", lambda *args: props[args[-2]] if "systemctl" in args
        else launch["source"] if args[0] == "git" and args[1] == "rev-parse"
        else "" if args[0] == "git" else "active" if args[-2] == "ActiveState" else "0")
    monkeypatch.setattr(probe, "filmbrain_state", lambda: launch["preserved_filmbrain"])
    order = []
    monkeypatch.setattr(probe, "checked", lambda *_args: (order.append("checked"), launch)[1])
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    idle_calls = []
    monkeypatch.setattr(probe.host, "wait_idle", lambda: (idle_calls.append(True), {"idle": True})[1])
    monkeypatch.setattr(probe.host, "check_log", lambda *_args: None)
    monkeypatch.setattr(probe.host.supervisor, "write_json", lambda path, value: path.write_text(json.dumps(value)))
    calls = []

    class Lease:
        def __enter__(self):
            assert order[-1] == "checked"
            order.append("lease")
            return 73
        def __exit__(self, *_args): return False

    monkeypatch.setattr(probe.files, "gpu_lease", lambda: Lease())
    monkeypatch.setattr(probe.files, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0", "BASE": "kept"})

    def supervised(command, log, **kwargs):
        calls.append((command, kwargs))
        assert command[1:4] == ["-m", probe.MODULE, "child"]
        assert command[command.index("--source") + 1] == SOURCE
        assert command[command.index("--launch-sha256") + 1] == LAUNCH_SHA
        assert command[command.index("--lock-fd") + 1] == "73"
        assert float(command[command.index("--started-monotonic") + 1]) > 0
        assert kwargs["timeout"] == contract.CHILD_SECONDS == 120
        assert kwargs["lock_fd"] == 73
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["BASE"] == "kept"
        assert all(kwargs["env"][key] == value for key, value in probe.cpu_profile.settings().items())
        kwargs["guard"]()
        log.write_text("synthetic child failed after guard\n")
        raise ValueError("synthetic child failure")

    monkeypatch.setattr(probe.files, "supervised_process", supervised)
    with pytest.raises(ValueError, match="synthetic child failure"):
        probe.supervise(SOURCE, LAUNCH_SHA)
    assert len(calls) == 1
    assert idle_calls == [True]
    assert order == ["checked", "lease"]
    report = json.loads((root / "report.json").read_text())
    assert report["decision"] == "force-fixture-failed"
    assert report["error"] == "synthetic child failure"
    assert "replay" not in report and "elapsed_seconds" not in report
    with pytest.raises(ValueError, match="one CPU-qualified fresh D0 attempt only"):
        probe.supervise(SOURCE, LAUNCH_SHA)
