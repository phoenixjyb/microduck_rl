"""Shallow ordering, deadline, and retention checks for the bounded D1 probe."""

from copy import deepcopy
from hashlib import sha256 as real_sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck import stance_historical_replication_auth as auth
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_probe as probe


SOURCE = "a" * 40
LAUNCH_SHA = "f" * 64
CPU_PROFILE = {"profile": "portable-cpu-checked"}


class FixedDigest:
    def __init__(self, value):
        self.value = value

    def hexdigest(self):
        return self.value


def fake_checkpoint_authentication(training_seed=577):
    checkpoints = []
    for iteration in (64, 128, 192, 255):
        digest = contract.CHECKPOINT_SHA256 if iteration == 255 else f"{iteration:064x}"
        checkpoints.append(dict(file=f"model_{iteration}.pt", sha256=digest,
            identity={"iteration": iteration, "training_seed": 577}))
    return dict(source=SOURCE, training_seed=training_seed,
        retained_training=dict(source=auth.TRAINING_SOURCE,
            report_sha256=auth.REPORT_SHA256[577], checkpoints=checkpoints))


def test_window_reserves_exact_1200_seconds_for_960_service_and_closeout(monkeypatch):
    assert contract.PROBE_SERVICE_SECONDS + contract.CLOSEOUT_SECONDS + contract.MARGIN_SECONDS == 1200
    probe.check_window(launching=True, now=contract.CUTOFF-1201)
    with pytest.raises(ValueError, match="fixed 08:00 Shanghai cutoff"):
        probe.check_window(launching=True, now=contract.CUTOFF-1200)
    probe.check_window(now=contract.CUTOFF-1)


def _d0_payloads():
    replay = {"cuda_initialized": False, "fresh_cpu_replay": True, "checked": 5}
    launch = {"source": probe.D0_SOURCE}
    report = dict(decision="single-substep-force-path-replayed", source=probe.D0_SOURCE,
        launch_sha256=probe.D0_FILES["launch.json"], child={"returncode": 0}, replay=replay,
        **contract.FALSE_FLAGS)
    values = {}
    for name in probe.D0_FILES:
        if name == "launch.json": value = launch
        elif name == "report.json": value = report
        elif name == "capture.pt": value = b"serialized D0 capture"
        else: value = {"file": name}
        values[name] = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return values, replay


def test_force_prerequisite_pins_code_and_d0_bytes_before_cpu_replay(monkeypatch):
    values, replay = _d0_payloads()
    events = []
    code_hashes = dict(zip(probe.D0_CODE, probe.D0_CODE.values()))
    file_hashes = dict(zip(probe.D0_FILES, probe.D0_FILES.values()))

    monkeypatch.setattr(probe.host, "ROOT", Path("/d1-test-root"))
    monkeypatch.setattr(probe.host, "digest", lambda path: (events.append(("code", Path(path).name)),
        code_hashes[Path(path).name])[1])
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: (
        events.append(("file", Path(path).name)), values[Path(path).name])[1])
    monkeypatch.setattr(probe.files, "parse", lambda raw: json.loads(raw))

    def digest(raw):
        for name, value in values.items():
            if raw == value:
                return FixedDigest(file_hashes[name])
        raise AssertionError("unexpected D0 bytes hashed")

    monkeypatch.setattr(probe, "sha256", digest)
    loaded = []
    monkeypatch.setattr(probe.torch, "load", lambda stream, **_kwargs: (
        events.append(("load", None)), loaded.append(stream.read()), {"serialized": True})[2])
    monkeypatch.setattr(probe.force_fixture, "replay", lambda value: (
        events.append(("replay", None)), replay)[1])
    ticks = iter((10., 11.))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))

    result = probe.force_prerequisite()
    assert result["source"] == probe.D0_SOURCE
    assert result["files"] == probe.D0_FILES and result["code"] == probe.D0_CODE
    assert result["replay"] == replay and loaded == [values["capture.pt"]]
    load_index = events.index(("load", None))
    assert all(events.index(("code", name)) < load_index for name in probe.D0_CODE)
    assert all(events.index(("file", name)) < load_index for name in probe.D0_FILES)
    assert events.index(("load", None)) < events.index(("replay", None))


@pytest.mark.parametrize("damage", ["code-hash", "artifact-hash", "admission", "slow-replay", "cuda-replay"])
def test_force_prerequisite_rejects_bad_pins_or_replay_before_return(monkeypatch, damage):
    values, replay = _d0_payloads()
    events = []
    code_hashes = dict(zip(probe.D0_CODE, probe.D0_CODE.values()))
    file_hashes = dict(zip(probe.D0_FILES, probe.D0_FILES.values()))
    if damage == "admission":
        report = json.loads(values["report.json"])
        report["training_admitted"] = True
        values["report.json"] = json.dumps(report, sort_keys=True).encode()
    monkeypatch.setattr(probe.host, "ROOT", Path("/d1-test-root"))

    def host_digest(path):
        name = Path(path).name
        events.append(("code", name))
        return "0"*64 if damage == "code-hash" and name == next(iter(code_hashes)) else code_hashes[name]

    monkeypatch.setattr(probe.host, "digest", host_digest)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: (
        events.append(("file", Path(path).name)), values[Path(path).name])[1])
    monkeypatch.setattr(probe.files, "parse", lambda raw: json.loads(raw))

    def digest(raw):
        for name, value in values.items():
            if raw == value:
                if damage == "artifact-hash" and name == "capture.pt":
                    return FixedDigest("0"*64)
                return FixedDigest(file_hashes[name])
        raise AssertionError("unexpected D0 bytes hashed")

    monkeypatch.setattr(probe, "sha256", digest)
    loads, replays = [], []
    monkeypatch.setattr(probe.torch, "load", lambda *_a, **_k: loads.append(True) or {"x": 1})
    replay_value = dict(replay)
    if damage == "cuda-replay": replay_value["cuda_initialized"] = True
    monkeypatch.setattr(probe.force_fixture, "replay", lambda *_a: replays.append(True) or replay_value)
    ticks = iter((10., 41. if damage == "slow-replay" else 11.))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))

    with pytest.raises(ValueError):
        probe.force_prerequisite()
    if damage in ("code-hash", "artifact-hash", "admission"):
        assert loads == [] and replays == []
    else:
        assert loads == [True] and replays == [True]


def test_selected_checkpoint_authenticates_metadata_before_reading_checkpoint_bytes(monkeypatch, tmp_path):
    authentication = fake_checkpoint_authentication()
    root = tmp_path / "training"
    raw = b"pinned checkpoint bytes behind hash seam"
    events = []
    monkeypatch.setattr(probe.auth, "validate", lambda receipt, source, retained: (
        events.append(("auth", source)), None)[1])
    monkeypatch.setattr(probe.evaluation.lean, "output_path", lambda *_args: root)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: (
        events.append(("raw", Path(path).name)), raw)[1])
    monkeypatch.setattr(probe, "sha256", lambda _raw: FixedDigest(contract.CHECKPOINT_SHA256))

    saved, loaded = probe.selected_checkpoint(authentication)
    assert saved["file"] == "model_255.pt" and saved["sha256"] == contract.CHECKPOINT_SHA256
    assert loaded == raw
    assert events == [("auth", SOURCE), ("raw", "model_255.pt")]


@pytest.mark.parametrize("damage", ["wrong-seed", "wrong-file", "wrong-checkpoint-hash"])
def test_selected_checkpoint_refuses_wrong_pinned_metadata_before_raw_read(monkeypatch, tmp_path, damage):
    authentication = fake_checkpoint_authentication(training_seed=578 if damage == "wrong-seed" else 577)
    selected = authentication["retained_training"]["checkpoints"][-1]
    if damage == "wrong-file": selected["file"] = "model_192.pt"
    if damage == "wrong-checkpoint-hash": selected["sha256"] = "1"*64
    events = []
    monkeypatch.setattr(probe.auth, "validate", lambda *_args: events.append("auth"))
    monkeypatch.setattr(probe.evaluation.lean, "output_path", lambda *_args: tmp_path)
    monkeypatch.setattr(probe.files, "file_bytes", lambda *_args, **_kw: events.append("raw"))
    with pytest.raises(ValueError):
        probe.selected_checkpoint(authentication)
    assert "raw" not in events


def test_checked_reauthenticates_then_checks_exact_launch_and_checkpoint_bytes_before_trace_verify(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    prefix = b"cpu-qualified recovery prefix"
    checkpoint_raw = b"frozen checkpoint behind pinned-hash seam"
    replay = {"pulse": {"complete_pulse_delivery": True}, "actor_replay_max_abs_error": 0.}
    q = dict(protocol="football-b1d-cpu-policy-pulse-qualification-v1", source=SOURCE,
        prefix_sha256=real_sha256(prefix).hexdigest(), prefix_bytes=len(prefix), elapsed_seconds=.5,
        checkpoint_sha256=contract.CHECKPOINT_SHA256, cpu_math_profile=probe.profile.expected_receipt(),
        replay=replay, cuda_initialized=False)
    qraw = json.dumps(q, sort_keys=True).encode()
    launch = {"source": SOURCE, "cpu_qualification_sha256": real_sha256(qraw).hexdigest(),
        "force_prerequisite": {"source": probe.D0_SOURCE, "files": probe.D0_FILES, "code": probe.D0_CODE},
        "declaration": {"plant": {"body_id": 2, "selected_plant": {"body": "bound"}}}}
    launch_raw = json.dumps(launch, sort_keys=True).encode()
    events, evidence_calls = [], []
    file_map = {"launch.json": launch_raw, "cpu-qualification.json": qraw,
                "cpu-prefix.pt": prefix, "checkpoint.pt": checkpoint_raw}
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.host, "digest", lambda path: real_sha256(launch_raw).hexdigest()
        if Path(path).name == "launch.json" else real_sha256(Path(path).read_bytes()).hexdigest())
    monkeypatch.setattr(probe.files, "parse", json.loads)
    monkeypatch.setattr(probe.auth, "run", lambda *args: (events.append(("auth", args)), fake_checkpoint_authentication())[1])
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: (
        events.append(("read", Path(path).name)), file_map[Path(path).name])[1])
    monkeypatch.setattr(probe, "launch_plan", lambda source, authentication, qualification_sha: (
        events.append(("plan", source, qualification_sha)), launch)[1])
    monkeypatch.setattr(probe, "sha256", lambda raw: FixedDigest(contract.CHECKPOINT_SHA256)
        if raw == checkpoint_raw else real_sha256(raw))
    monkeypatch.setattr(probe.evidence, "verify", lambda raw, digest, cp, declaration: (
        events.append(("verify", digest, cp, declaration)), evidence_calls.append(True), replay)[2])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)

    checked = probe.checked(SOURCE, real_sha256(launch_raw).hexdigest())
    assert checked == launch and evidence_calls == [True]
    assert events[0] == ("read", "launch.json")
    assert events.index(("auth", (SOURCE, auth.TRAINING_SOURCE, 577))) < events.index(("read", "checkpoint.pt"))
    assert events.index(("verify", q["prefix_sha256"], checkpoint_raw, launch["declaration"])) > events.index(("read", "checkpoint.pt"))
    assert q["prefix_sha256"] == real_sha256(prefix).hexdigest()


def test_prepare_requires_hidden_cuda_and_performs_cpu_archive_and_52_tick_qualification_before_final_plan(
        monkeypatch, tmp_path):
    root = tmp_path / "attempt"
    actor = object()
    checkpoint_raw = b"checkpoint"
    qualification_value = {"cpu": "prefix"}; encoded = b"serialized cpu prefix"
    replay = {"pulse": {"complete_pulse_delivery": True, "checked_physics_steps": 520},
              "actor_replay_max_abs_error": 0., "cuda_initialized": False}
    events = []; runtime_box = []
    monkeypatch.setattr(probe, "check_window", lambda **kwargs: events.append(("window", kwargs)))
    monkeypatch.setattr(probe.host, "identity", lambda source: (events.append(("identity", source)), {"source": source})[1])
    monkeypatch.setattr(probe.profile, "checked_receipt", lambda: (events.append(("profile", None)), CPU_PROFILE)[1])
    monkeypatch.setattr(probe.auth, "run", lambda *args: (events.append(("auth", args)), fake_checkpoint_authentication())[1])
    monkeypatch.setattr(probe, "launch_plan", lambda source, authentication, qualification_sha: (
        events.append(("plan", qualification_sha)), {"source": source, "selected_checkpoint": {"identity": {"iteration": 255}},
            "declaration": {"ok": True}})[1])
    monkeypatch.setattr(probe, "selected_checkpoint", lambda _auth: (
        events.append(("checkpoint", None)),
        {"file": "model_255.pt", "identity": {"iteration": 255}}, checkpoint_raw)[1:])
    monkeypatch.setattr(probe, "restored_actor", lambda raw, identity: (events.append(("actor", raw)), actor)[1])
    monkeypatch.setattr(probe.files.native, "_plain_path", lambda path: path)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files, "write_json", lambda path, value: (
        events.append(("write-json", Path(path).name)), path.write_text(json.dumps(value, sort_keys=True)))[1])
    monkeypatch.setattr(probe.smoke, "write_bytes", lambda path, raw: (
        events.append(("write-bytes", Path(path).name)), Path(path).write_bytes(raw))[1])
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    class FakeRuntime:
        def __init__(self, cases, *, device):
            runtime_box.append(self)
            events.append(("runtime", tuple(cases), device))
    monkeypatch.setattr("mjlab_microduck.stance_recovery_runtime.RecoveryRuntime", FakeRuntime)
    def collect(runtime_env, received_actor, declaration, identity, **kwargs):
        assert (runtime_env, received_actor) == (runtime_box[0], actor)
        assert kwargs["policy_tick_limit"] == probe.CPU_QUAL_TICKS == 52
        assert kwargs["deadline_monotonic"] > 0
        events.append(("collect", kwargs["policy_tick_limit"]))
        return qualification_value
    monkeypatch.setattr(probe.evidence, "collect", collect)
    monkeypatch.setattr(probe.evidence, "encode", lambda value: (events.append(("encode", None)), encoded)[1])
    monkeypatch.setattr(probe.evidence, "verify", lambda raw, digest, cp, declaration: (
        events.append(("verify-cpu", None)), replay)[1])
    monkeypatch.setattr(probe.files, "gpu_lease", lambda: pytest.fail("CPU preparation acquired GPU lease"))
    monkeypatch.setattr(probe.host, "digest", lambda path: real_sha256(Path(path).read_bytes()).hexdigest())

    result = probe.prepare(SOURCE)
    assert result["output"] == str(root)
    assert events.index(("auth", (SOURCE, auth.TRAINING_SOURCE, 577))) < events.index(("collect", 52))
    assert events.index(("collect", 52)) < events.index(("verify-cpu", None)) < events.index(("plan", real_sha256((root/"cpu-qualification.json").read_bytes()).hexdigest()))
    assert events.index(("verify-cpu", None)) < events.index(("write-json", "launch.json"))
    assert (root/"cpu-prefix.pt").read_bytes() == encoded


def test_prepare_rejects_visible_cuda_before_host_or_archive_auth(monkeypatch):
    events = []
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(probe.host, "identity", lambda *_args: events.append("identity"))
    monkeypatch.setattr(probe.auth, "run", lambda *_args: events.append("auth"))
    with pytest.raises(ValueError, match="CPU-only D1 preparation"):
        probe.prepare(SOURCE)
    assert events == []


def test_child_inherits_lease_and_rechecks_cpu_before_cuda_or_runtime(monkeypatch):
    started = 1000.
    times = iter((started+1, started+20, started+21))
    events = []
    launch = {"selected_checkpoint": {"identity": {"iteration": 255}}, "declaration": {"ok": True}}
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: events.append("window"))
    monkeypatch.setattr(probe.smoke, "inherited_lease", lambda fd: events.append(("lease", fd)))
    monkeypatch.setattr(probe, "checked", lambda *_args: (events.append("checked"), launch)[1])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: events.append("cuda-query") or False)
    monkeypatch.setattr(probe.host, "wait_idle", lambda: pytest.fail("continued after unavailable CUDA"))
    monkeypatch.setattr(probe, "restored_actor", lambda *_args: pytest.fail("loaded checkpoint before CUDA gate"))
    with pytest.raises(ValueError, match="only inherited-lease D1 child"):
        probe.child(SOURCE, LAUNCH_SHA, 73, started)
    assert events.index(("lease", 73)) < events.index("checked") < events.index("cuda-query")


def test_child_enforces_180_entry_reserve_and_fixed_capture_serialize_deadlines(monkeypatch, tmp_path):
    started = 1000.
    launch = {"selected_checkpoint": {"identity": {"iteration": 255}}, "declaration": {"ok": True}}
    events, writes = [], []
    # Entry reaches 179 s, then capture and serialization stay within their
    # explicit child-relative deadlines. No CUDA kernel or service is started.
    times = iter((started+1, started+179, started+179.5, started+180,
                  started+839, started+840, started+841, started+842))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe.smoke, "inherited_lease", lambda _fd: events.append("lease"))
    monkeypatch.setattr(probe, "checked", lambda *_args: events.append("checked") or launch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe.host, "wait_idle", lambda: {"idle": True})
    checkpoint_raw = b"child checkpoint"
    monkeypatch.setattr(probe.files, "file_bytes", lambda *_args, **_kwargs: checkpoint_raw)
    monkeypatch.setattr(probe, "restored_actor", lambda *_args: object())

    class FakeRuntime:
        def __init__(self, cases, *, device):
            assert cases == [contract.PROBE_CASE] and device == "cuda:0"
            events.append("runtime")
    monkeypatch.setattr("mjlab_microduck.stance_recovery_runtime.RecoveryRuntime", FakeRuntime)

    def collect(_env, _actor, _declaration, _identity, *, deadline_monotonic):
        events.append(("deadline", deadline_monotonic))
        return {"backend": {"torch_device": "cuda:0", "warp_is_cuda": True},
                "collection": {"stop_reason": "complete", "policy_ticks": contract.POLICY_TICKS,
                               "elapsed_seconds": 180.}}
    monkeypatch.setattr(probe.evidence, "collect", collect)
    raw = b"bounded encoded capture"
    monkeypatch.setattr(probe.evidence, "encode", lambda _value: raw)
    monkeypatch.setattr(probe.smoke, "write_bytes", lambda path, value: writes.append((Path(path).name, value)))
    monkeypatch.setattr(probe.files, "write_json", lambda path, value: writes.append((Path(path).name, value)))

    probe.child(SOURCE, LAUNCH_SHA, 73, started)
    assert ("deadline", started+contract.PROBE_CHILD_SECONDS-120) in events
    assert contract.PROBE_CHILD_SECONDS == 900
    assert probe.contract.PROBE_CHILD_SECONDS-120 == 780
    capture = next(value for name, value in writes if name == "capture.json")
    assert capture["serialization_seconds"] == pytest.approx(1.)
    assert capture["elapsed_seconds"] == pytest.approx(842.)


def test_child_refuses_when_cpu_entry_reaches_180_second_limit_before_cuda(monkeypatch):
    started = 1000.
    values = iter((started+1, started+180))
    events = []
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe.smoke, "inherited_lease", lambda fd: events.append(("lease", fd)))
    monkeypatch.setattr(probe, "checked", lambda *_args: events.append("checked") or {})
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: pytest.fail("CUDA queried at reserve edge"))
    with pytest.raises(ValueError, match="CPU entry leaves fixed capture and closeout reserves"):
        probe.child(SOURCE, LAUNCH_SHA, 73, started)
    assert events == [("lease", 73), "checked"]


def test_supervisor_checks_exact_service_and_live_guard_before_gpu_lease_and_retains_failure(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir()
    for name in ("launch.json", "checkpoint.pt", "cpu-prefix.pt", "cpu-qualification.json"):
        (root/name).write_bytes(b"fresh")
    preserved = {"filmbrain": "unchanged"}
    launch = {"preserved_filmbrain": preserved}
    props = dict(MainPID=str(os.getpid()), RuntimeMaxUSec="16min", KillMode="control-group",
        ActiveState="active", MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec="2s", Nice="10")
    events, reports = [], []
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe, "service_name", lambda _source: "d1-probe.service")
    monkeypatch.setattr(probe.host, "read", lambda *args: props[args[-2]] if "systemctl" in args
        else SOURCE if args[:2] == ("git", "rev-parse") else "")
    monkeypatch.setattr(probe.d0, "filmbrain_state", lambda: (events.append("filmbrain"), preserved)[1])
    monkeypatch.setattr(probe.host, "check_log", lambda *_args: events.append("log"))
    monkeypatch.setattr(probe, "check_window", lambda **_kwargs: events.append("window"))
    monkeypatch.setattr(probe, "checked", lambda *_args: (events.append("checked"), launch)[1])
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.host, "wait_idle", lambda: (events.append("idle"), {"idle": True})[1])
    monkeypatch.setattr(probe.host.supervisor, "write_json", lambda _path, value: reports.append(deepcopy(value)))

    class Lease:
        def __enter__(self):
            assert events.index("checked") < len(events)
            events.append("lease")
            return 73
        def __exit__(self, *_args): return False
    monkeypatch.setattr(probe.files, "gpu_lease", lambda: Lease())
    monkeypatch.setattr(probe.files, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0", "BASE": "kept"})

    def child(_command, log, **kwargs):
        assert contract.PROBE_CHILD_SECONDS == 900
        assert "timeout" not in kwargs  # The fixed stance wrapper owns the 900-second cap.
        assert kwargs["lock_fd"] == 73
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["env"]["BASE"] == "kept"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        kwargs["guard"]()
        log.write_text("synthetic child failed\n")
        raise ValueError("synthetic child failure")
    monkeypatch.setattr(probe.files, "supervised_stance_smoke", child)

    with pytest.raises(ValueError, match="synthetic child failure"):
        probe.supervise(SOURCE, LAUNCH_SHA)
    assert "checked" in events and events.index("checked") < events.index("lease")
    assert "filmbrain" in events and "log" in events
    assert reports[-1]["decision"] == "frozen-recovery-probe-failed"
    assert reports[-1]["error"] == "synthetic child failure"
    assert "score" not in reports[-1]
    assert all(reports[-1][key] is False for key in contract.FALSE_FLAGS)


@pytest.mark.parametrize("damage", ["runtime", "memory", "cpu", "nice", "killmode"])
def test_supervisor_refuses_service_property_drift_before_gpu_lease(tmp_path, monkeypatch, damage):
    root = tmp_path / "attempt"; root.mkdir()
    props = dict(MainPID=str(os.getpid()), RuntimeMaxUSec="16min", KillMode="control-group",
        ActiveState="active", MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec="2s", Nice="10")
    key = {"runtime": "RuntimeMaxUSec", "memory": "MemoryMax", "cpu": "CPUQuotaPerSecUSec",
           "nice": "Nice", "killmode": "KillMode"}[damage]
    props[key] = "wrong"
    reports = []
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe, "service_name", lambda _source: "d1-probe.service")
    monkeypatch.setattr(probe.host, "read", lambda *args: props[args[-2]] if "systemctl" in args else "")
    monkeypatch.setattr(probe.files, "gpu_lease", lambda: pytest.fail("service drift reached GPU lease"))
    monkeypatch.setattr(probe.host.supervisor, "write_json", lambda _path, value: reports.append(value))
    with pytest.raises(ValueError, match="independently capped D1 timing probe service"):
        probe.supervise(SOURCE, LAUNCH_SHA)
    assert reports[-1]["decision"] == "frozen-recovery-probe-failed"
    assert reports[-1]["error_type"] == "ValueError"


def test_parent_replay_requires_cuda_hidden_and_validates_capture_hash_before_rescore(tmp_path, monkeypatch):
    root = tmp_path / "attempt"; root.mkdir()
    raw = b"serialized CUDA capture"; checkpoint_raw = b"checkpoint bytes"
    capture = dict(protocol=contract.PROTOCOL, source=SOURCE, launch_sha256=LAUNCH_SHA,
        capture_sha256=real_sha256(raw).hexdigest(), capture_bytes=len(raw),
        backend={"torch_device": "cuda:0", "warp_is_cuda": True}, optimizer_steps=0,
        collection={"stop_reason": "complete", "policy_ticks": contract.POLICY_TICKS,
                    "elapsed_seconds": 800.}, construction_seconds=2.,
        serialization_seconds=1., elapsed_seconds=842.,
        **contract.FALSE_FLAGS)
    events = []
    file_map = {"capture.json": json.dumps(capture).encode(), "capture.pt": raw,
                "checkpoint.pt": checkpoint_raw}
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files, "parse", json.loads)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: file_map[Path(path).name])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    def verify(payload, digest, checkpoint_bytes, declaration):
        events.append(("verify", os.environ.get("CUDA_VISIBLE_DEVICES"), payload, digest, checkpoint_bytes))
        return {"case": contract.PROBE_CASE, "collection": capture["collection"], "rescored": True,
                "numerical_diagnostic": {"gates": {"full_duration": True}}}
    monkeypatch.setattr(probe.evidence, "verify", verify)
    launch = {"declaration": {"exact": True}}
    record, score, _elapsed = probe.replay_capture(SOURCE, launch, LAUNCH_SHA)
    assert record == capture and score["rescored"] is True
    assert events == [("verify", "", raw, capture["capture_sha256"], checkpoint_raw)]


def test_parent_replay_rejects_capture_hash_before_evidence_verify(tmp_path, monkeypatch):
    root = tmp_path / "attempt"; root.mkdir()
    raw = b"tampered capture"
    capture = dict(protocol=contract.PROTOCOL, source=SOURCE, launch_sha256=LAUNCH_SHA,
        capture_sha256="0"*64, capture_bytes=len(raw),
        backend={"torch_device": "cuda:0", "warp_is_cuda": True}, optimizer_steps=0,
        collection={"stop_reason": "complete", "policy_ticks": contract.POLICY_TICKS,
                    "elapsed_seconds": 1.}, construction_seconds=1.,
        serialization_seconds=1., elapsed_seconds=3.,
        **contract.FALSE_FLAGS)
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files, "parse", lambda _raw: capture)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs:
        raw if Path(path).name == "capture.pt" else b"record")
    monkeypatch.setattr(probe.evidence, "verify", lambda *_args: pytest.fail("unverified capture rescored"))
    with pytest.raises(ValueError, match="authenticated actual diagnostic CUDA capture"):
        probe.replay_capture(SOURCE, {"declaration": {}}, LAUNCH_SHA)


def test_parent_replay_must_not_accept_partial_capture_as_full_length_probe(tmp_path, monkeypatch):
    root = tmp_path / "attempt"; root.mkdir()
    raw = b"validly hashed but partial capture"; checkpoint_raw = b"checkpoint bytes"
    collection = {"stop_reason": "wall-budget-exhausted", "policy_ticks": 11, "elapsed_seconds": 779.}
    capture = dict(protocol=contract.PROTOCOL, source=SOURCE, launch_sha256=LAUNCH_SHA,
        capture_sha256=real_sha256(raw).hexdigest(), capture_bytes=len(raw),
        backend={"torch_device": "cuda:0", "warp_is_cuda": True}, optimizer_steps=0,
        collection=collection, construction_seconds=1., serialization_seconds=1., elapsed_seconds=840.,
        **contract.FALSE_FLAGS)
    file_map = {"capture.json": json.dumps(capture).encode(), "capture.pt": raw,
                "checkpoint.pt": checkpoint_raw}
    monkeypatch.setattr(probe, "output_path", lambda _source: root)
    monkeypatch.setattr(probe.files, "parse", json.loads)
    monkeypatch.setattr(probe.files, "file_bytes", lambda path, **_kwargs: file_map[Path(path).name])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.torch.cuda, "is_initialized", lambda: False)
    partial_score = {"case": contract.PROBE_CASE, "collection": collection,
        "numerical_diagnostic": {"gates": {"full_duration": False}}}
    monkeypatch.setattr(probe.evidence, "verify", lambda *_args: partial_score)

    with pytest.raises(ValueError, match="full-length") as raised:
        probe.replay_capture(SOURCE, {"declaration": {}}, LAUNCH_SHA)
    assert json.loads(raised.value.__notes__[0])["collection"] == collection
    assert json.loads(raised.value.__notes__[0])["numerical_diagnostic"]["gates"]["full_duration"] is False
