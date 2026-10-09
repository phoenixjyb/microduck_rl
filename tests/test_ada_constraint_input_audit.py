"""Synthetic arithmetic and reception contracts: no simulator initialization."""
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from mjlab_microduck import ada_constraint_input_audit as p
from test_ada_saved_pose_response_receiver import packet, SOURCE


def test_import_and_help_inert():
    code = "import sys;from mjlab_microduck import ada_constraint_input_audit;assert not {'numpy','warp','torch','mujoco','mujoco_warp'}&sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    subprocess.run([sys.executable, "-m", p.__name__, "--help"], check=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("damage", ["none", "head", "branch", "dirty", "module", "source"])
def test_clean_exact_arithmetic_source(monkeypatch, damage):
    def read(*args, binary=False):
        if binary: return b"different" if damage == "module" else Path(p.__file__).read_bytes()
        if args[-1] == "HEAD": return "2" * 40 if damage == "head" else SOURCE
        if args[-1] == "--show-current": return "main" if damage == "branch" else p.host.BRANCH
        if args[-1] == "--porcelain": return "?? user-work" if damage == "dirty" else ""
        raise AssertionError(args)
    monkeypatch.setattr(p.host, "read", read)
    if damage == "none": p.source_check(SOURCE)
    else:
        with pytest.raises(ValueError): p.source_check("bad" if damage == "source" else SOURCE)


def test_numeric_headers_are_record_bound_without_runtime_import():
    result = p.numeric_source()
    assert result["numeric"] == p.NUMERIC
    assert {k: v["sha256"] for k, v in result["headers"].items()} == p.HEADERS
    assert all(v["record_sha256_matches"] and v["bytes"] > 20000 for v in result["headers"].values())
    assert not {"warp", "torch", "mujoco", "mujoco_warp"} & sys.modules.keys()


@pytest.mark.parametrize("damage", ["version", "inventory", "hash"])
def test_numeric_source_refusals(monkeypatch, damage):
    dist = p.distribution("mujoco")
    if damage == "version":
        class Other:
            version = "3.9.0"
        monkeypatch.setattr(p, "distribution", lambda _: Other())
    if damage == "inventory":
        class Missing:
            version = dist.version
            files = []
        monkeypatch.setattr(p, "distribution", lambda _: Missing())
    if damage == "hash": monkeypatch.setattr(p, "HEADERS", {k: "0" * 64 for k in p.HEADERS})
    with pytest.raises(ValueError): p.numeric_source()


def model_packet():
    values = {p.MODEL_INPUTS[0]: np.array([.002], np.float32),
        p.MODEL_INPUTS[1]: np.ones(1, np.float32),
        p.MODEL_INPUTS[2]: np.ones((1, 16, 2), np.float32),
        p.MODEL_INPUTS[3]: np.zeros(82, np.int32)}
    values.update({"/model/padding" + str(i): np.array([i], np.float32) for i in range(343)})
    raw, manifest = {}, {}
    for k, v in values.items():
        vector = k == p.MODEL_INPUTS[2]
        shape = list(v.shape[:-1] if vector else v.shape)
        width = 8 if vector else v.dtype.itemsize
        strides = []
        for n in reversed(shape): strides.insert(0, width); width *= n
        raw[k] = np.frombuffer(v.tobytes(), np.uint8).copy()
        manifest[k] = dict(dtype="<class 'warp._src.types." + ("vec2f" if vector else str(v.dtype)) + "'>",
            shape=shape, strides=strides, bytes=v.nbytes, sha256=sha256(v.tobytes()).hexdigest())
    prior = dict(child=dict(input_manifest=manifest))
    child = dict(model_array_sha256={k: v["sha256"] for k, v in manifest.items()})
    banks = {"prepared-inputs.npz": dict(raw, **{"/data/old-poison": np.array([np.nan])})}
    return prior, banks, child, values


def test_all347_model_bytes_bound_without_historical_data():
    prior, banks, child, values = model_packet()
    result = p.bound_model_values(prior, banks, child)
    assert set(result) == set(p.MODEL_INPUTS)
    assert all(np.array_equal(result[k], values[k]) for k in result)
    banks["prepared-inputs.npz"]["/data/old-poison"] = object()
    assert p.bound_model_values(prior, banks, child).keys() == result.keys()


@pytest.mark.parametrize("damage", ["missing", "extra", "oldhash", "newhash", "byte", "dtype", "length", "shape", "nan", "time", "ratio"])
def test_full_model_binding_refusals(damage):
    prior, banks, child, _ = model_packet(); raw = banks["prepared-inputs.npz"]
    key = "/model/padding342"; m = prior["child"]["input_manifest"]
    if damage == "missing": raw.pop(key)
    if damage == "extra": raw["/model/extra"] = np.zeros(1, np.uint8)
    if damage == "oldhash": m[key]["sha256"] = "0" * 64
    if damage == "newhash": child["model_array_sha256"][key] = "0" * 64
    if damage == "byte": raw[key][0] ^= 1
    if damage == "dtype": raw[key] = raw[key].astype(np.int32)
    if damage == "length": raw[key] = raw[key][:-1]
    if damage == "shape": m[p.MODEL_INPUTS[2]]["shape"] = [1, 8, 2]
    if damage in {"nan", "time", "ratio"}:
        key = p.MODEL_INPUTS[1 if damage == "ratio" else 0]
        raw[key] = np.frombuffer(np.array([np.nan if damage == "nan" else 0], np.float32).tobytes(), np.uint8).copy()
        m[key]["sha256"] = child["model_array_sha256"][key] = sha256(raw[key]).hexdigest()
    with pytest.raises(ValueError): p.bound_model_values(prior, banks, child)


def recipe_inputs():
    return dict(timestep=.001, invsqrt=1., inverse_weight=1., friction=1.,
        solref=[.02, 1.], solimp=[.5, .9, .01, .5, 2.], distance=0., margin=0., velocity=.1)


def test_scalar_recipe_closed_positive_arm_and_known_values():
    args = recipe_inputs(); r = p.row_recipe(**args)
    assert r["impedance"] == .5 and r["effective_inverse_weight"] == 4. and r["D"] == .25
    assert r["aref"] == -r["damping"] * .1 and r["frictionloss"] == 0
    args.update(distance=.0025, margin=0., velocity=0.)
    assert p.row_recipe(**args)["impedance"] == .55
    args["distance"] = .0075
    assert p.row_recipe(**args)["impedance"] == .5 + (1. - 2. * .25 * .25) * (.9 - .5)
    args["distance"] = .02
    assert p.row_recipe(**args)["impedance"] == .9
    args.update(timestep=.02, solref=[.001, 1.])
    assert p.row_recipe(**args)["damping"] == 2. / (.9 * .04)


@pytest.mark.parametrize("damage", ["nan", "time", "ratio", "inverse", "friction", "ref0", "ref1", "power", "refshape", "impshape", "unordered"])
def test_closed_recipe_refusals(damage):
    args = recipe_inputs()
    if damage == "nan": args["velocity"] = float("nan")
    if damage == "time": args["timestep"] = 0
    if damage == "ratio": args["invsqrt"] = 0
    if damage == "inverse": args["inverse_weight"] = 0
    if damage == "friction": args["friction"] = 0
    if damage == "ref0": args["solref"][0] = -1
    if damage == "ref1": args["solref"][1] = 0
    if damage == "power": args["solimp"][4] = 3
    if damage == "refshape": args["solref"].pop()
    if damage == "impshape": args["solimp"].pop()
    if damage == "unordered": args["solimp"][:2] = [.9, .5]
    with pytest.raises(ValueError): p.row_recipe(**args)


def recipe_data():
    _, _, _, _, stages = packet(); data = stages["before_solve"]
    _, _, _, model = model_packet()
    data["/data/nacon"][:] = 8; data["/data/nefc"][:] = [14, 46]
    for slot in range(8):
        for field, value in {"worldid": 1, "type": 1, "dim": 3, "geom": [0, 29 if slot < 4 else 64],
            "friction": [1, 1, 0, 0, 0], "solref": [.02, 1.],
            "solimp": [.5, .9, .01, .5, 2.], "dist": -.0002, "includemargin": 0.}.items():
            data["/data/contact/" + field][slot] = value
        start = 14 + 4 * slot
        data["/data/contact/efc_address"][slot] = range(start, start + 4)
        data["/data/efc/type"][1, start:start+4] = 6
        data["/data/efc/id"][1, start:start+4] = slot
    for field in ("vel", "pos", "margin", "frictionloss", "Jqvel"):
        data["/data/efc/" + field] = np.zeros((2, 512), np.float32)
    manifest = dict(scalars={"/model/opt/disableflags": dict(kind="int", value=0)})
    return data, model, manifest


def test_all32_recipes_use_new_own_addresses():
    data, model, manifest = recipe_data(); rows = p.gpu_row_recipes(data, model, manifest)
    assert len(rows) == 32 and [r["efc_address"] for r in rows] == list(range(14, 46))
    assert {r["slot"] for r in rows} == set(range(8))
    assert all(r["recipe"]["D"] > 0 and r["bodies"] == [0, 0] for r in rows)
    data["/data/efc/Jqvel"][1, 14] = data["/data/efc/vel"][1, 14] = .125
    changed = p.gpu_row_recipes(data, model, manifest)
    assert changed[0]["recipe"]["aref"] != rows[0]["recipe"]["aref"]
    assert changed[1:] == rows[1:]


def test_constraint_bit_allows_independent_sensor_bit():
    d, m, s = recipe_data(); expected = p.gpu_row_recipes(d, m, s)
    d["/data/contact/type"][:8] = 3
    assert p.gpu_row_recipes(d, m, s) == expected


@pytest.mark.parametrize("damage", ["count", "dim", "world", "geom", "body", "address", "overlap", "type", "id", "vel", "flags", "nonfinite", "constraint", "sensor_only", "contact_type_dtype"])
def test_complete_contact_recipe_refusals(damage):
    d, m, s = recipe_data()
    if damage == "count": d["/data/nacon"][:] = 7
    if damage == "dim": d["/data/contact/dim"][0] = 4
    if damage == "world": d["/data/contact/worldid"][0] = -1
    if damage == "geom": d["/data/contact/geom"][0, 0] = 82
    if damage == "body": m[p.MODEL_INPUTS[3]][0] = 16
    if damage == "address": d["/data/contact/efc_address"][0, 3] -= 1
    if damage == "overlap": d["/data/contact/efc_address"][1] = d["/data/contact/efc_address"][0]
    if damage == "type": d["/data/efc/type"][1, 14] = 1
    if damage == "id": d["/data/efc/id"][1, 14] = 1
    if damage == "vel": d["/data/efc/vel"][1, 14] = .1
    if damage == "flags": s["scalars"]["/model/opt/disableflags"]["value"] = 4096
    if damage == "nonfinite": d["/data/contact/solref"][0, 0] = np.nan
    if damage == "constraint": d["/data/contact/type"][0] = 0
    if damage == "sensor_only": d["/data/contact/type"][0] = 2
    if damage == "contact_type_dtype": d["/data/contact/type"] = d["/data/contact/type"].astype(np.int64)
    with pytest.raises(ValueError): p.gpu_row_recipes(d, m, s)


def terms():
    J = np.zeros((3, 20), np.float64); J[:, :2] = [[1, 2], [-1, 3], [2, 1]]
    return J, np.array([2., 3., 4.]), np.array([.5, -2., 1.]), np.array([1, 2, 3], np.int32), [0, 1, 2]


def test_full_unmasked_nominal_terms_not_row_averages():
    r = p.nominal_terms(*terms())
    assert [v[:2] for v in r["H_JtDJ"][:2]] == [[21., 3.], [3., 39.]]
    assert r["h_JtDaref"][:2] == [15., -12.]
    assert r["c_half_aref_D_aref"] == 8.25
    assert r["final_state_histogram"] == {"1": 1, "2": 1, "3": 1}
    assert r["state_mask_applied"] is False and r["actual_solver_matrix_reconstructed"] is False
    J, D, ref, state, _ = terms()
    doubled = p.nominal_terms(np.repeat(J, 2, axis=0), np.repeat(D, 2), np.repeat(ref, 2), np.repeat(state, 2), list(range(6)))
    assert doubled["H_JtDJ"][0][0] == 42 and doubled["h_JtDaref"][0] == 30


def test_ascending_explicit_reduction_no_blas(monkeypatch):
    J, D, ref, state, rows = terms(); J[:] = 0; J[:, 0] = [2**60, 1, -2**60]; D[:] = 1; ref[:] = 1
    for name in ("dot", "einsum", "matmul"):
        monkeypatch.setattr(np, name, lambda *a, **k: pytest.fail("BLAS/reordering forbidden"))
    assert p.nominal_terms(J, D, ref, state, rows)["h_JtDaref"][0] == 0


@pytest.mark.parametrize("damage", ["dtype", "mix", "Jshape", "Dshape", "state", "nan", "zero", "negative", "duplicate", "unsorted", "range", "boolrow"])
def test_nominal_terms_refusals(damage):
    J, D, ref, state, rows = terms()
    if damage == "dtype": J = J.astype(np.int32)
    if damage == "mix": D = D.astype(np.float32)
    if damage == "Jshape": J = J[:, :19]
    if damage == "Dshape": D = D.reshape(3, 1)
    if damage == "state": state = state.astype(np.float32)
    if damage == "nan": J[0, 0] = np.nan
    if damage == "zero": D[0] = 0
    if damage == "negative": D[0] = -1
    if damage == "duplicate": rows = [0, 0, 1]
    if damage == "unsorted": rows = [1, 0, 2]
    if damage == "range": rows = [0, 1, 3]
    if damage == "boolrow": rows = [False, 1, 2]
    with pytest.raises(ValueError): p.nominal_terms(J, D, ref, state, rows)


def active():
    _, _, _, banks, _ = packet(); a = deepcopy(banks["cpu-active.npz"])
    for w in range(2): a[f"rows/{w}/D"][:] = 1
    return a


def test_aggregate_bins_are_not_physical_point_matching():
    a = active(); r = p.grouped_proxies(a)
    assert len(r) == 4 and sum(len(v["nominal"]["rows"]) for v in r) == 36
    assert all(not v["physical_point_correspondence_established"] for v in r)
    other = deepcopy(r)
    for row in other:
        row["nominal"]["rows"] *= 2; row["nominal"]["H_JtDJ"][0][0] += 1
    d = p.proxy_difference(r, other)
    assert all(v["gpu_rows"] == 2 * v["native_rows"] and v["gpu_minus_native_H_JtDJ"][0][0] == 1 for v in d)
    assert all(not v["cause_isolated"] for v in d)
    with pytest.raises(ValueError): p.proxy_difference(r, other[:-1])


def test_native_local_contact_ids_are_not_table_order():
    a = active(); a["contacts/slot"][:] = [7, 9]
    a["rows/1/id"][14:18] = 7; a["rows/1/id"][18:22] = 9
    assert p.grouped_proxies(a) == p.grouped_proxies(active())


@pytest.mark.parametrize("damage", ["overlap", "missing", "dim"])
def test_grouped_full_disjoint_partition(damage):
    a = active()
    if damage == "overlap": a["contacts/efc_address"][1] = a["contacts/efc_address"][0]
    if damage == "missing": a["rows/1/type"][14] = 0
    if damage == "dim": a["contacts/dim"][0] = 4
    with pytest.raises(ValueError): p.grouped_proxies(a)


def analysis_fixture(tmp_path, monkeypatch):
    """Mock only outer authenticated receivers; exercise complete internal wiring."""
    _, _, old, banks, _ = packet(); data, _, static = recipe_data()
    prior, model_banks, child, _ = model_packet()
    # New scalar arrays replace five synthetic padding leaves: exactly114 Data.
    for key in sorted(k for k in data if k.startswith("/data/padding"))[:len(data) - 114]: data.pop(key)
    data["/data/efc/D"][:] = 1
    for key in p.response.POSES: data["/data/" + key][:] = .125
    stages = {s: deepcopy(data) for s in p.response.STAGES}
    manifest = prior["child"]["input_manifest"]
    prepared = model_banks["prepared-inputs.npz"]
    for key, value in data.items():
        row = deepcopy(old["child"]["input_manifest"].get(key))
        if row is None:
            row = dict(dtype="<class 'warp._src.types.float32'>", shape=list(value.shape), strides=list(value.strides), bytes=value.nbytes)
        historical = value.copy()
        if key.removeprefix("/data/") in p.response.POSES: historical[:] = .75
        prepared[key] = np.frombuffer(historical.tobytes(), np.uint8).copy()
        row["sha256"] = sha256(prepared[key]).hexdigest(); manifest[key] = row
    descriptor = dict(old["child"]["plant"], floor=0, feet=[29, 64])
    prior["child"]["plant"] = descriptor; prior["files"] = dict(synthetic="old-full-packet")
    native = {k: v.astype(np.float64) if v.dtype == np.float32 else v.copy()
        for k, v in active().items() if not k.startswith("sidecar/")}
    banks = dict(model_banks, **{"cpu-active.npz": native})
    arrays = {stage + k: np.frombuffer(v.tobytes(), np.uint8).copy() for stage, d in stages.items() for k, v in d.items()}
    layout = {k: p.response.expanded_layout(v) for k, v in manifest.items() if k.startswith("/data/")}
    gpu = dict(child=child, files=dict(synthetic="new-full-packet"))
    root = tmp_path / "gpu"; root.mkdir(); report = root / "report.json"; report.write_bytes(b"{}")
    monkeypatch.setattr(p, "GPU_REPORT_SHA", sha256(report.read_bytes()).hexdigest())
    monkeypatch.setattr(p.measured, "receive", lambda *args: (gpu, arrays, layout, static, {}))
    monkeypatch.setattr(p.saved.p.prior, "authenticated_banks", lambda *args: (prior, banks))
    monkeypatch.setattr(p.response.source, "audit", lambda: dict(synthetic="source-audit"))
    return root, arrays, banks


def test_complete_analysis_uses_new_data_and_bound_old_model_only(tmp_path, monkeypatch):
    root, arrays, banks = analysis_fixture(tmp_path, monkeypatch)
    result = p.analyze(tmp_path, root)
    assert len(arrays) == 570 and len(result["all347_model_array_sha256"]) == 347
    assert len(result["gpu_contact_row_recipes"]) == 32
    assert len(result["native_nominal_proxies"]) == len(result["gpu_nominal_proxies"]) == 4
    assert [v["native_rows"] for v in result["nominal_proxy_differences"]] == [14, 14, 4, 4]
    assert [v["gpu_rows"] for v in result["nominal_proxy_differences"]] == [14, 14, 16, 16]
    # Old non-state/non-Model bytes are not substituted for NEW measured Data.
    for key, value in banks["prepared-inputs.npz"].items():
        if key.startswith("/data/") and key.removeprefix("/data/") not in p.saved.p.p.base.UNCHANGED:
            value[:] = 255
    assert p.analyze(tmp_path, root) == result


@pytest.mark.parametrize("field", ["J", "D", "aref"])
def test_nominal_gpu_inputs_must_be_solver_byte_unchanged(tmp_path, monkeypatch, field):
    root, arrays, _ = analysis_fixture(tmp_path, monkeypatch)
    arrays["after_solve/data/efc/" + field][0] ^= 1
    with pytest.raises(ValueError, match="unchanged by solver"): p.analyze(tmp_path, root)


@pytest.mark.parametrize("damage", ["none", "extra", "recipe", "proxy", "model", "source", "module", "flags", "numeric", "nan", "symlink", "oversize"])
def test_complete_recomputed_receiver(tmp_path, monkeypatch, damage):
    expected = dict(deepcopy(p.CONTRACT), protocol=p.PROTOCOL, decision=p.DECISION,
        gpu_contact_row_recipes=[dict(recipe=dict(D=1.))], nominal_proxy_differences=[dict(H=[[1.]])],
        all347_model_array_sha256={"/model/one": "a" * 64}, numeric_source=dict(minval=1e-15))
    monkeypatch.setattr(p, "source_check", lambda source: None)
    monkeypatch.setattr(p, "analyze", lambda *args: deepcopy(expected))
    value = dict(deepcopy(expected), source=SOURCE, module_sha256=sha256(Path(p.__file__).read_bytes()).hexdigest())
    if damage == "extra": value["extra"] = True
    if damage == "recipe": value["gpu_contact_row_recipes"][0]["recipe"]["D"] += 1
    if damage == "proxy": value["nominal_proxy_differences"][0]["H"][0][0] += 1
    if damage == "model": value["all347_model_array_sha256"] = {}
    if damage == "source": value["source"] = "2" * 40
    if damage == "module": value["module_sha256"] = "0" * 64
    if damage == "flags": value["training_authorized"] = 0
    if damage == "numeric": value["numeric_source"]["minval"] = 1e-10
    if damage == "nan": value["gpu_contact_row_recipes"][0]["recipe"]["D"] = float("nan")
    output = tmp_path / "result.json"; output.write_text(json.dumps(value))
    if damage == "symlink":
        link = tmp_path / "link.json"; link.symlink_to(output); output = link
    if damage == "oversize": output.write_bytes(b" " * (256 * 1024 + 1))
    if damage == "none": assert p.receive(output, SOURCE, tmp_path, tmp_path) == value
    else:
        with pytest.raises(ValueError): p.receive(output, SOURCE, tmp_path, tmp_path)


def test_runtime_and_acceptance_paths_absent():
    tree = ast.parse(Path(p.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.split(".")[0] in {"warp", "torch", "mujoco", "mujoco_warp"} for n in names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"solve", "forward", "kinematics", "collision", "make_constraint", "step", "dot", "matmul", "einsum"}
    assert not any(p.CONTRACT[k] for k in p.CONTRACT if k != "flags")
    assert not any(p.CONTRACT["flags"].values())
