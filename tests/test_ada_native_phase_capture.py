"""Synthetic collector tests only; no native runtime or real solver calls."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import numpy as np
import pytest
from mjlab_microduck import ada_native_phase_capture as p
from test_ada_native_phase_getters import mock


def fake_case(tmp_path, monkeypatch):
    events = []
    data = [mock(w)[0] for w in (0, 1)]
    states = {k: np.stack([np.asarray(getattr(d, k)).reshape(-1) for d in data]) for k in p.native.STATE}
    files = []
    for w in (0, 1):
        raw = b"mock-mjb" + bytes([w])
        path = tmp_path / f"world{w}.mjb"
        path.write_bytes(raw)
        files.append(dict(file=path.name, bytes=len(raw), sha256=sha256(raw).hexdigest()))
    capture = dict(mjb=files, capture=dict(model_static=[{}, {}]))
    runtime = SimpleNamespace()
    runtime.MjModel = SimpleNamespace(from_binary_path=lambda path: int(path[-5]))
    runtime.MjData = lambda model: data[model]
    for name in p.native.CALLBACKS:
        setattr(runtime, name, lambda: None)
    def stage(name):
        def call(model, d):
            events.append((model, name))
            if name == "mj_fwdConstraint":
                d.efc_force[0] = 0.125
        return call
    for name in p.plan.BEFORE + p.plan.AFTER:
        setattr(runtime, name, stage(name))
    runtime.mj_step = lambda *a: pytest.fail("unguarded integration")
    runtime.mj_forward = lambda *a: pytest.fail("unguarded forward")
    runtime.mj_collision = lambda *a: pytest.fail("unguarded collision")
    monkeypatch.setattr(p.plan, "model_guards", lambda metas: None)
    monkeypatch.setattr(p, "fence", lambda *a: dict(mock_fence=True))
    original = p.getters.snapshot
    def snapshot(d, world, seven):
        events.append((world, "snapshot"))
        return original(d, world, seven)
    monkeypatch.setattr(p.getters, "snapshot", snapshot)
    return runtime, capture, states, events


def test_two_world_exact_order_boundaries_and_complete_owned_capture(tmp_path, monkeypatch):
    runtime, old, states, events = fake_case(tmp_path, monkeypatch)
    payload, result = p.capture_phases(runtime, old, {}, states, tmp_path)
    expected = [(w, n) for w in (0, 1) for n in p.plan.BEFORE + ("snapshot", "mj_fwdConstraint", "snapshot")]
    assert events == expected and result["public_stage_calls"] == [list(v) for v in p.CALL_ORDER]
    assert len(payload) == 272 and all(v.flags.owndata for v in payload.values())
    for record in result["records"]:
        assert len(record["comparison"]["differences"]) == 68
        assert record["comparison"]["differences"]["efc/force"]["max_abs"] == .125
    assert result["before_after_capture_executed"] and not result["simulator_qualified"]
    assert not result["internal_solver_dispatch_count_established"]


@pytest.mark.parametrize("name", ["mj_step", "mj_forward", "mj_collision", "mj_sensorAcc"])
def test_forbidden_public_calls_fail_and_guard_restores(name):
    runtime = SimpleNamespace(**{name: lambda: None})
    original = getattr(runtime, name)
    with p.public_stage_guard(runtime) as (calls, denied):
        with pytest.raises(ValueError, match="unexpected public native"):
            getattr(runtime, name)()
        assert denied == [name] and calls == []
    assert getattr(runtime, name) is original


@pytest.mark.parametrize("state_name", p.native.STATE)
def test_solver_mutated_seven_input_state_is_protocol_failure_not_masked(tmp_path, monkeypatch, state_name):
    runtime, old, states, events = fake_case(tmp_path, monkeypatch)
    def solve(model, data):
        if state_name == "time":
            data.time += .001
        else:
            getattr(data, state_name).flat[0] += .001
    runtime.mj_fwdConstraint = solve
    with pytest.raises(ValueError, match="state byte unchanged"):
        p.capture_phases(runtime, old, {}, states, tmp_path)


def test_mjb_changed_before_load_is_refused(tmp_path, monkeypatch):
    runtime, old, states, _ = fake_case(tmp_path, monkeypatch)
    (tmp_path / old["mjb"][0]["file"]).write_bytes(b"other")
    with pytest.raises(ValueError, match="MJB reauthentication"):
        p.capture_phases(runtime, old, {}, states, tmp_path)


def test_same_size_wrong_state_layout_fails_before_physics(tmp_path, monkeypatch):
    runtime, old, states, events = fake_case(tmp_path, monkeypatch)
    original = runtime.MjData
    def bad_data(model):
        data = original(model)
        data.xfrc_applied = data.xfrc_applied.reshape(6, 16)
        return data
    runtime.MjData = bad_data
    with pytest.raises(ValueError, match="explicit complete getter shape"):
        p.capture_phases(runtime, old, {}, states, tmp_path)
    assert events == []


def test_guard_setup_failure_restores_earlier_patch():
    class Runtime:
        def __init__(self):
            self.mj_a = lambda: None
            self.mj_b = lambda: None
        def __setattr__(self, name, value):
            if name == "mj_b" and hasattr(self, "mj_b"):
                raise ValueError("mock immutable function")
            object.__setattr__(self, name, value)
    runtime = Runtime(); original = runtime.mj_a
    with pytest.raises(ValueError, match="immutable"):
        with p.public_stage_guard(runtime):
            pytest.fail("must not enter")
    assert runtime.mj_a is original


@pytest.mark.parametrize("name", p.native.CALLBACKS)
def test_each_present_callback_refused(name):
    runtime = SimpleNamespace(**{n: lambda: None for n in p.native.CALLBACKS})
    setattr(runtime, name, lambda: object())
    with pytest.raises(ValueError, match="callback"):
        p.callbacks_absent(runtime)


def test_full_payload_decode_authentication_and_tamper_refusal(tmp_path, monkeypatch):
    runtime, old, states, _ = fake_case(tmp_path, monkeypatch)
    arrays, _ = p.capture_phases(runtime, old, {}, states, tmp_path)
    path = tmp_path / "payload.npz"
    descriptor = p.native.retain(path, arrays)
    decoded = p.decode_payload(path, descriptor)
    assert p.native.manifest(decoded) == p.native.manifest(arrays)
    changed = deepcopy(descriptor)
    changed["arrays"]["0/before/fields/qpos"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor"):
        p.decode_payload(path, changed)
    changed = deepcopy(descriptor); changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload bytes"):
        p.decode_payload(path, changed)


def test_public_model_and_serialization_fences(monkeypatch):
    arrays = {"model/qpos0": np.zeros(21)}
    monkeypatch.setattr(p.native, "model_snapshot", lambda model: (arrays, {}))
    runtime = SimpleNamespace(mj_sizeModel=lambda model: 3,
        mj_saveModel=lambda model, path, out: out.__setitem__(slice(None), np.frombuffer(b"abc", np.uint8)))
    bank = {"model/0/" + k: a.copy() for k, a in arrays.items()}
    p.fence(runtime, None, 0, bank, {}, b"abc")
    arrays["model/qpos0"][0] = 1e-15
    with pytest.raises(ValueError, match="Model/Option fence"):
        p.fence(runtime, None, 0, bank, {}, b"abc")
    arrays["model/qpos0"][0] = 0
    with pytest.raises(ValueError, match="MJB byte fence"):
        p.fence(runtime, None, 0, bank, {}, b"abd")


def test_full_receiver_recomputes_every_comparison_and_rejects_report_tamper(tmp_path, monkeypatch):
    runtime, old, states, _ = fake_case(tmp_path, monkeypatch)
    arrays, result = p.capture_phases(runtime, old, {}, states, tmp_path)
    bank = {f"model/{w}/model/qpos0": np.zeros(21) for w in (0, 1)}
    for w, record in enumerate(result["records"]):
        model = {"model/qpos0": bank[f"model/{w}/model/qpos0"]}
        record["fences"] = dict(public_numeric_fields=1,
            arrays_sha256=p.plan.own.audit.packet.digest(p.native.manifest(model)),
            static_sha256=p.plan.own.audit.packet.digest({}), mjb_sha256=old["mjb"][w]["sha256"])
    output = tmp_path / "capture.json"
    descriptor = p.native.retain(output.with_suffix(".npz"), arrays)
    module = p.Path(p.__file__).read_bytes()
    value = dict(protocol=p.PROTOCOL, decision=p.DECISION, source="a" * 40,
        module_sha256=sha256(module).hexdigest(), input_plan_sha256=p.projection.PLAN_SHA,
        capture=result, payload=descriptor,
        native_installed_provenance={k: dict(sha256=v, bytes=1, record_sha256_matches=True)
            for k, v in p.native.NATIVE_FILES.items()},
        protected_services={s + ":" + n: "inactive" for s in ("system", "user") for n in p.host.SERVICES},
        foreign_processes=[], telemetry={})
    monkeypatch.setattr(p, "inputs", lambda *a: (old, bank, states))
    old["native_installed_provenance"] = deepcopy(value["native_installed_provenance"])
    monkeypatch.setattr(p, "source_check", lambda *a: None)
    monkeypatch.setattr(p.host, "read", lambda *a, **k: module)
    output.write_text(json.dumps(value))
    assert p.receive(output, "a" * 40, tmp_path, tmp_path, tmp_path) == value
    value["capture"]["records"][1]["comparison"]["differences"]["efc/force"]["max_abs"] = 0
    output.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="comparison independently recomputed"):
        p.receive(output, "a" * 40, tmp_path, tmp_path, tmp_path)


def test_portable_source_check_binds_plan_dependency(monkeypatch):
    def refused(source):
        raise ValueError("mismatched phase plan module")
    monkeypatch.setattr(p.plan, "source_check", refused)
    monkeypatch.setattr(p, "committed_modules", lambda *a: pytest.fail("must check plan first"))
    with pytest.raises(ValueError, match="mismatched phase plan"):
        p.source_check("a" * 40)
