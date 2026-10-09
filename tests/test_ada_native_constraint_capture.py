"""Pure synthetic contracts only; no real native model/forward or GPU calls."""
from copy import deepcopy
from hashlib import sha256
import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mjlab_microduck import ada_native_constraint_capture as p


def fake_runtime():
    class Array:
        def _init_annotation(self): return "annotation"
    class Module:
        def load(self): return "loaded"
    for name in p.WARP_ARRAY_PATHS: setattr(Array, name, lambda *a: "concrete")
    wp = SimpleNamespace(array=Array, **{n: lambda *a: "warp" for n in p.WARP_FUNCTIONS})
    context = SimpleNamespace(Module=Module, **{n: lambda *a: "context" for n in p.WARP_FUNCTIONS})
    native = SimpleNamespace(**{n: lambda *a: "native" for n in p.NATIVE_FORBIDDEN})
    native.mj_forward = lambda *a: "forward"
    native.mj_contactForce = lambda *a: "force"
    return wp, context, native


@pytest.mark.parametrize("path", p.WARP_ARRAY_PATHS)
def test_concrete_array_paths_forbidden_but_annotations_permitted_and_restored(path):
    wp, context, native = fake_runtime(); original = getattr(wp.array, path)
    with p.forbidden_runtime(wp, context, native) as (attempts, calls):
        assert wp.array()._init_annotation() == "annotation"
        with pytest.raises(ValueError, match="forbidden runtime"):
            getattr(wp.array(), path)()
        assert attempts == ["array." + path] and calls == dict(forward=0, contact_force=0)
    assert getattr(wp.array, path) is original


@pytest.mark.parametrize("prefix", ["wp", "context"])
@pytest.mark.parametrize("path", p.WARP_FUNCTIONS)
def test_public_and_internal_launch_load_aliases_forbidden(prefix, path):
    wp, context, native = fake_runtime(); obj = wp if prefix == "wp" else context
    original = getattr(obj, path)
    with p.forbidden_runtime(wp, context, native) as (attempts, _):
        with pytest.raises(ValueError): getattr(obj, path)()
        assert attempts == [prefix + "." + path]
    assert getattr(obj, path) is original


@pytest.mark.parametrize("path", p.NATIVE_FORBIDDEN)
def test_unpredeclared_public_physics_calls_forbidden(path):
    wp, context, native = fake_runtime(); original = getattr(native, path)
    with p.forbidden_runtime(wp, context, native) as (attempts, _):
        with pytest.raises(ValueError): getattr(native, path)()
        assert attempts == [path]
    assert getattr(native, path) is original


def test_module_load_guard_restores_even_after_exception():
    wp, context, native = fake_runtime(); original = context.Module.load
    with pytest.raises(ValueError):
        with p.forbidden_runtime(wp, context, native): context.Module().load()
    assert context.Module.load is original


@pytest.mark.parametrize("name,key,limit", [("mj_forward", "forward", 2), ("mj_contactForce", "contact_force", 256)])
def test_native_call_limits_count_real_wrapped_calls(name, key, limit):
    wp, context, native = fake_runtime(); original = getattr(native, name)
    with p.forbidden_runtime(wp, context, native) as (_, calls):
        for _ in range(limit): getattr(native, name)()
        assert calls[key] == limit
        with pytest.raises(ValueError, match="bounded public native calls"): getattr(native, name)()
    assert getattr(native, name) is original


def test_partial_guard_install_failure_restores_every_prior_patch():
    wp, context, native = fake_runtime(); original = wp.launch
    delattr(context, "load_module")
    with pytest.raises(ValueError, match="frozen guard target"):
        with p.forbidden_runtime(wp, context, native): pass
    assert wp.launch is original


def test_all_public_numeric_model_and_option_fields_copied_no_method_calls():
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002, gravity=np.array([0., 0., -9.81])),
        nq=21, qpos0=np.zeros(21), names=b"duck\0", stat=object(), vis=object(),
        named=lambda: pytest.fail("named model methods must not be called"))
    schema = {"model/nq": {"kind": "int"}, "model/qpos0": {"kind": "array", "dtype": "float64"},
        "model/names": {"kind": "bytes"}, "option/timestep": {"kind": "float"},
        "option/gravity": {"kind": "array", "dtype": "float64"},
        **{k: {"kind": "nested"} for k in ("model/opt", "model/stat", "model/vis")},
        "model/named": {"kind": "method"}}
    arrays, meta = p.model_snapshot(model, schema)
    assert set(arrays) == {"model/qpos0", "model/names", "option/gravity"}
    assert meta["scalars"]["model/nq"] == dict(type="int", value=21)
    assert meta["exclusions"] == ["model/named:method", "model/opt:nested-serialized-in-MJB",
        "model/stat:nested-serialized-in-MJB", "model/vis:nested-serialized-in-MJB"]
    assert arrays["model/names"].tobytes() == b"duck\0"
    model.qpos0[0] = 9.
    assert arrays["model/qpos0"][0] == 0.


@pytest.mark.parametrize("value", [object(), "unexpected string", [1., 2.], np.array([object()], object), np.array([np.nan])])
def test_unknown_or_nonfinite_model_values_refused(value):
    with pytest.raises(ValueError): p.model_snapshot(SimpleNamespace(opt=SimpleNamespace(), field=value))


@pytest.mark.parametrize("damage", ["dtype", "shape", "bytes"])
def test_exact_comparison_retains_disagreement_without_tolerance(damage):
    expected = {"v": np.arange(4, dtype=np.float64)}
    actual = {"v": expected["v"].copy()}
    if damage == "dtype": actual["v"] = actual["v"].astype(np.float32)
    if damage == "shape": actual["v"] = actual["v"].reshape(2, 2)
    if damage == "bytes": actual["v"][0] = np.nextafter(0., 1.)
    result = p.compare(actual, expected)["v"]
    assert result[damage + "_equal"] is False
    assert any(not x for x in result.values())


def fixture():
    arrays = {"fields/" + k: np.zeros((2, p.base.WIDTHS[k]), np.float64) for k in p.base.FIELDS}
    active = {}
    counts = []
    for w, (ncon, nefc) in enumerate(((0, 14), (4, 30))):
        row = dict(ncon=ncon, ne=0, nf=14, nl=0, nefc=nefc, nJ=nefc * 20, nA=0, nisland=0)
        counts.append({k: row[k] for k in ("ncon", "ne", "nf", "nl", "nefc")} | {"solver_niter": w + 1})
        arrays[f"data/{w}/counters"] = np.array([row[k] for k in p.COUNTERS], np.int64)
        for k in p.DATA[:-1]: arrays[f"data/{w}/{k}"] = np.zeros(p.NWARNING if k.startswith("warning/") else p.NISLAND, np.int32)
        arrays[f"data/{w}/solver_niter"][0] = w + 1
        for k in p.SOLVER: arrays[f"solver/{w}/{k}"] = np.zeros((p.NISLAND, p.NSOLVER), np.int32 if k.startswith("n") else np.float64)
        for k in p.EFC:
            width = 20 if k in ("J", "J_colind") else 4 if k == "KBIP" else 1
            integer = k in ("type", "id", "J_rownnz", "J_rowadr", "J_rowsuper", "J_colind", "state")
            arrays[f"efc/{w}/{k}"] = np.zeros(nefc * width, np.int32 if integer else np.float64)
            if k == "KBIP": arrays[f"efc/{w}/{k}"] = arrays[f"efc/{w}/{k}"].reshape(nefc, 4)
        arrays[f"efc/{w}/type"][:14] = 1
        arrays[f"efc/{w}/type"][14:] = 6
        arrays[f"efc/{w}/id"][14:] = np.repeat(np.arange(ncon), 4)
        for k in p.CONTACT: arrays[f"contact/{w}/{k}"] = np.zeros((ncon,) + p.CONTACT_TAILS[k], np.int32 if k in p.CONTACT_INTS else np.float64)
        arrays[f"contact/{w}/geom"] = np.zeros((ncon, 2), np.int32)
        arrays[f"contact/{w}/solref"] = np.zeros((ncon, 2), np.float64)
        arrays[f"contact/{w}/solimp"] = np.zeros((ncon, 5), np.float64)
        arrays[f"contact/{w}/efc_address"] = 14 + np.arange(ncon, dtype=np.int32) * 4
        for k in p.contact.ROW_FIELDS:
            active[f"rows/{w}/{k}"] = arrays[f"efc/{w}/{k}"].reshape((nefc, 20) if k == "J" else (-1,)).copy()
        for k in ("dof_frictionloss", "dof_damping"): arrays[f"model/{w}/model/{k}"] = np.ones(20, np.float64)
    for k in p.contact.CONTACT_FIELDS:
        if k == "worldid": active["contacts/" + k] = np.ones(4, np.int32)
        elif k == "slot": active["contacts/" + k] = np.arange(4, dtype=np.int32)
        elif k == "force": active["contacts/" + k] = np.zeros((4, 6), np.float64)
        else: active["contacts/" + k] = arrays["contact/1/" + k].copy()
    arrays.update({"active/" + k: v for k, v in active.items()})
    report = {"child": {"counters": {"cpu": counts}}}
    banks = {"cpu-active.npz": deepcopy(active), "cpu-fields.npz": {k: arrays["fields/" + k].copy() for k in p.base.FIELDS},
             "motor.npz": {k: np.ones((2, 20), np.float32) for k in ("dof_frictionloss", "dof_damping")}}
    state = {k: arrays["fields/" + k].astype(np.float32) for k in p.STATE}
    banks.update({"prepared-inputs.npz": {"/data/" + k: np.frombuffer(v.tobytes(), np.uint8) for k, v in state.items()}, "gpu-fields.npz": state})
    report["child"]["input_manifest"] = {"/data/" + k: {"sha256": sha256(v.tobytes()).hexdigest()} for k, v in state.items()}
    return arrays, report, banks


def test_pure_full_summary_exact_historical_byte_comparison():
    arrays, report, banks = fixture(); result = p.summarize(arrays, report, banks)
    assert result["historical_outputs_equal"] is True
    assert result["counters"][1]["nefc"] == 30
    arrays["fields/qacc"][1, 0] = 1e-100
    assert p.summarize(arrays, report, banks)["historical_outputs_equal"] is False


@pytest.mark.parametrize("damage", ["missing", "extra", "dtype", "width", "count", "rowid", "address", "motor", "slots", "modelinventory"])
def test_invalid_capture_core_fail_closed(damage):
    arrays, report, banks = fixture()
    if damage == "missing": arrays.pop("efc/1/diagA")
    if damage == "extra": arrays["unbound"] = np.zeros(1)
    if damage == "dtype": arrays["efc/1/D"] = arrays["efc/1/D"].astype(np.float32)
    if damage == "width": arrays["efc/1/J"] = np.zeros(1)
    if damage == "count": arrays["data/1/counters"][0] = 999
    if damage == "rowid": arrays["efc/1/id"][14] = 3
    if damage == "address": arrays["contact/1/efc_address"][0] = 13
    if damage == "motor": arrays["model/1/model/dof_damping"][0] = 2
    if damage == "slots": arrays["active/contacts/slot"][0] = 9
    if damage == "modelinventory": arrays.pop("model/1/model/dof_damping")
    with pytest.raises(ValueError): p.summarize(arrays, report, banks)


def test_manifest_finite_nonobject_complete_bytes_including_empty_arrays():
    arrays = {"empty": np.zeros((0, 2), np.float64), "v": np.array([1., 2.])}
    result = p.manifest(arrays)
    assert result["empty"]["sha256"] == sha256(b"").hexdigest()
    assert result["v"]["bytes"] == 16
    assert p.manifest({"v": np.array([np.inf])})["v"]["nonfinite_count"] == 1


def test_capture_module_import_is_runtime_inert_in_fresh_process():
    code = "import sys; from mjlab_microduck import ada_native_constraint_capture; assert not any(k in sys.modules for k in ('mujoco','mujoco_warp','warp','torch'))"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert completed.returncode == 0, completed.stderr


def test_scope_does_not_claim_zero_native_solver_or_historical_identity():
    assert p.META["public_native_forward_calls"] == 2
    assert p.META["integration_steps"] == p.META["optimizer_steps"] == 0
    assert "new_solver_calls" not in p.META
    assert all(v is False for v in p.META["flags"].values())
    assert p.META["full_historical_model_identity_established"] is False


def header_fixture():
    from importlib.metadata import distribution
    dist = distribution("mujoco")
    assert dist.version == "3.10.0"
    headers = {}
    for name in ("mjtype.h", "mjmodel.h", "mjdata.h"):
        path = "mujoco/include/mujoco/" + name
        raw = Path(dist.locate_file(path)).read_bytes()
        assert sha256(raw).hexdigest() == p.NATIVE_FILES[path]
        headers[name] = raw.decode()
    return headers


def test_frozen_installed_header_statistics_schema_without_model_or_physics():
    assert p.statistics_header_schema(header_fixture()) == dict(warnings=7, islands=20, solver_iterations=200)


@pytest.mark.parametrize("damage", ["warning", "islands", "iterations", "dimension"])
def test_frozen_header_dimension_drift_fails_before_physics(damage):
    headers = header_fixture()
    if damage == "warning": headers["mjtype.h"] = headers["mjtype.h"].replace("mjWARN_BADCTRL", "mjWARN_UNKNOWN")
    if damage == "islands": headers["mjmodel.h"] = headers["mjmodel.h"].replace("mjNISLAND       20", "mjNISLAND       21")
    if damage == "iterations": headers["mjmodel.h"] = headers["mjmodel.h"].replace("mjNSOLVER       200", "mjNSOLVER       201")
    if damage == "dimension": headers["mjdata.h"] = headers["mjdata.h"].replace("warning[mjNWARNING]", "warning[8]")
    with pytest.raises(ValueError): p.statistics_header_schema(headers)


def test_synthetic_eight_slot_warning_layout_is_rejected_not_padded_or_truncated():
    arrays, report, banks = fixture()
    arrays["data/1/warning/number"] = np.zeros(8, np.int32)
    with pytest.raises(ValueError, match="native statistic layout: warning/number"):
        p.summarize(arrays, report, banks)


def public_model_fixture():
    arrays, scalar, excluded = {}, {}, []
    for k, row in p.SCHEMA.items():
        if row["kind"] == "array": arrays[k] = np.zeros(1, np.dtype(row["dtype"]))
        elif row["kind"] == "bytes": arrays[k] = np.zeros(1, np.uint8)
        elif row["kind"] in ("int", "float", "bool"):
            value = dict(int=0, float=0., bool=False)[row["kind"]]
            scalar[k] = dict(type=row["kind"], value=value)
        else: excluded.append(k + (":method" if row["kind"] == "method" else ":nested-serialized-in-MJB"))
    return arrays, dict(scalars=scalar, exclusions=excluded)


@pytest.mark.parametrize("damage", ["none", "arraymissing", "scalarmissing", "methodmissing", "dtype", "scalar_kind", "nan", "infinity"])
def test_complete627_public_schema_bilateral_omission_and_types(damage):
    arrays, meta = public_model_fixture()
    if damage == "arraymissing": arrays.pop("model/geom_bodyid")
    if damage == "scalarmissing": meta["scalars"].pop("model/nq")
    if damage == "methodmissing": meta["exclusions"].pop()
    if damage == "dtype": arrays["model/geom_bodyid"] = np.zeros(1, np.int64)
    if damage == "scalar_kind": meta["scalars"]["model/nq"]["value"] = 21.
    if damage == "nan": arrays["model/jnt_range"] = np.array([np.nan])
    if damage == "infinity": arrays["model/jnt_range"] = np.array([np.inf])
    if damage in ("none", "infinity"): p.validate_model(arrays, meta)
    else:
        with pytest.raises(ValueError): p.validate_model(arrays, meta)


@pytest.mark.parametrize("key", ["efc/1/KBIP", "contact/1/friction", "data/1/warning/number", "solver/1/nactive"])
def test_same_element_count_wrong_shape_refused(key):
    arrays, report, banks = fixture(); arrays[key] = arrays[key].reshape(-1, 1)
    with pytest.raises(ValueError): p.summarize(arrays, report, banks)


def test_nonfinite_computed_data_and_warning_are_retained_not_promoted(tmp_path):
    arrays, report, banks = fixture()
    arrays["data/1/warning/number"][0] = 1
    arrays["fields/qacc"][1, 0] = np.nan
    result = p.summarize(arrays, report, banks)
    assert result["warning_counts"] == [0, 1] and result["computed_arrays_finite"] is False
    assert result["historical_outputs_equal"] is False
    payload = p.retain(tmp_path / "failed-data.npz", arrays)
    assert payload["arrays"]["fields/qacc"]["nonfinite_count"] == 1
    with pytest.raises(FileExistsError): p.retain(tmp_path / "failed-data.npz", arrays)


def test_entity_and_mjlab_bam_guard_factory_blocks_runtime_methods():
    wp, context, native = fake_runtime()
    class FakeEntity:
        def initialize(self): pytest.fail("must not initialize")
    with p.forbidden_runtime(wp, context, native, lambda: [(FakeEntity, "initialize")]) as (attempts, _):
        with pytest.raises(ValueError): FakeEntity().initialize()
        assert attempts == ["FakeEntity.initialize"]


def receiver_fixture(tmp_path, monkeypatch):
    arrays, report, banks = fixture(); report["files"] = {}
    model, meta = public_model_fixture()
    for w in range(2):
        for k, v in model.items():
            if k not in ("model/dof_frictionloss", "model/dof_damping"): arrays[f"model/{w}/" + k] = v.copy()
    summary = p.summarize(arrays, report, banks, [meta, meta])
    summary.update(model_static=[meta, meta], model_fences_equal=True, guarded_paths_attempted=[], held_warp_executables=0,
        public_native_calls=dict(forward=2, contact_force=4), native_callback_hooks=list(p.CALLBACKS), static_bam_parameter_construction_allowed=True)
    monkeypatch.setattr(p.prior, "authenticated_banks", lambda root: (report, banks))
    monkeypatch.setattr(p.host, "read", lambda *args, **kwargs: Path(p.__file__).read_bytes() if args[-1].endswith(".py") else p.SCHEMA_PATH.read_bytes())
    output = tmp_path / "capture.json"
    stored, storage = p.pack_storage(arrays)
    value = dict(protocol=p.PROTOCOL, decision=p.DECISION, source={"commit": "a" * 40}, meta=deepcopy(p.META),
        module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest(), schema_sha256=sha256(p.SCHEMA_PATH.read_bytes()).hexdigest(),
        predecessor_report_sha256=p.prior.REPORT_SHA256, predecessor_files={}, capture=summary,
        protected_services={scope + ":" + service: "inactive" for scope in ("system", "user") for service in p.host.SERVICES},
        native_installed_provenance={k: dict(bytes=100, sha256=v, record_sha256_matches=True) for k, v in p.NATIVE_FILES.items()},
        payload=p.retain(output.with_suffix(".npz"), stored), array_storage=storage, mjb=[])
    for w in range(2):
        path = tmp_path / (output.stem + f"-world{w}.mjb")
        raw = b"synthetic MJB bytes, no native model"; path.write_bytes(raw)
        value["mjb"].append(dict(file=path.name, bytes=len(raw), sha256=sha256(raw).hexdigest()))
    return output, value


@pytest.mark.parametrize("damage", ["none", "scope", "summary", "provenance", "schema", "module", "payload", "mjb", "callback", "forcecalls", "protected"])
def test_portable_receiver_recomputes_full_packet_and_rejects_forged_claims(tmp_path, monkeypatch, damage):
    output, value = receiver_fixture(tmp_path, monkeypatch)
    if damage == "scope": value["meta"]["training_authorized"] = True
    if damage == "summary": value["capture"]["historical_outputs_equal"] = False
    if damage == "provenance": value["native_installed_provenance"][next(iter(p.NATIVE_FILES))]["sha256"] = "0" * 64
    if damage == "schema": value["schema_sha256"] = "0" * 64
    if damage == "module": value["module_sha256"] = "0" * 64
    if damage == "payload": value["payload"]["sha256"] = "0" * 64
    if damage == "mjb": value["mjb"][0]["sha256"] = "0" * 64
    if damage == "callback": value["capture"]["native_callback_hooks"].pop()
    if damage == "forcecalls": value["capture"]["public_native_calls"]["contact_force"] = 0
    if damage == "protected": value["protected_services"][next(iter(value["protected_services"]))] = "active"
    output.write_text(json.dumps(value))
    if damage == "none": assert p.receive(output, "a" * 40, tmp_path)[0]["capture"]["historical_outputs_equal"] is True
    else:
        with pytest.raises(ValueError): p.receive(output, "a" * 40, tmp_path)


@pytest.mark.parametrize("damage", ["none", "field", "dtype", "shape", "missing", "contract", "extra_world1"])
def test_exact_shared_storage_preserves_world_specific_motor_and_rejects_mismatch(damage):
    model, _ = public_model_fixture()
    arrays = {f"model/{w}/" + k: v.copy() for w in range(2) for k, v in model.items()}
    arrays["model/1/model/dof_frictionloss"][0] = 17.
    if damage == "field": arrays["model/1/model/body_mass"][0] = 1.
    if damage == "dtype": arrays["model/1/model/body_mass"] = np.zeros(1, np.float32)
    if damage == "shape": arrays["model/1/model/body_mass"] = np.zeros((1, 1), np.float64)
    if damage == "missing": arrays.pop("model/1/model/body_mass")
    if damage in ("field", "dtype", "shape", "missing"):
        with pytest.raises(ValueError): p.pack_storage(arrays)
        return
    stored, contract = p.pack_storage(arrays)
    assert contract["shared_numeric_field_count"] == 475
    assert len([k for k in stored if k.startswith("model/1/")]) == 2
    if damage == "contract": contract["shared_numeric_field_count"] = 474
    if damage == "extra_world1": stored["model/1/model/body_mass"] = np.zeros(1)
    if damage != "none":
        with pytest.raises(ValueError): p.expand_storage(stored, contract)
    else:
        expanded = p.expand_storage(stored, contract)
        assert all(all(row.values()) for row in p.compare(expanded, arrays).values())
        assert expanded["model/1/model/dof_frictionloss"][0] == 17.
