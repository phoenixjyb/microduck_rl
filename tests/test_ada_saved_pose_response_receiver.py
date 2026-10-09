"""Complete synthetic response packet; no simulator or physics is initialized."""
from copy import deepcopy
from hashlib import sha256
import json
import shlex
import sys

import numpy as np
import pytest

from mjlab_microduck import ada_saved_pose_response as p

SOURCE = "1" * 40


def service():
    props = dict(p.SERVICE_CAPS, MainPID="123", InvocationID="2" * 32,
                 Environment=shlex.join([k + "=" + v for k, v in p.SERVICE_ENV.items()]))
    return dict(unit="microduck-ada-response-11111111.service", properties=props)


def packet():
    """Five complete allocated banks, two contacts, all14 friction rows/world."""
    values, manifest = {}, {}
    def add(key, value, name=None, tail=()):
        ns = "mujoco_warp._src.types" if name in {"vec5f", "vec10f"} else "warp._src.types"
        name = name or str(value.dtype)
        shape = list(value.shape[:-len(tail)] if tail else value.shape)
        width = value.dtype.itemsize * int(np.prod(tail or (1,)))
        strides = []
        for n in reversed(shape): strides.insert(0, width); width *= n
        raw = np.frombuffer(value.tobytes(), np.uint8).copy()
        values[key] = value
        manifest[key] = dict(dtype=f"<class '{ns}.{name}'>", shape=shape, strides=strides,
                             bytes=value.nbytes, sha256=sha256(raw).hexdigest())
    base = p.saved.p.p.base
    for k, width in base.WIDTHS.items(): add("/data/" + k, np.zeros((2, width), np.float32))
    for k in p.POSES:
        if "/data/" + k not in values: add("/data/" + k, np.zeros((2, 3), np.float32))
    for k in p.saved.p.COUNTERS + ("nacon", "ncollision"):
        add("/data/" + k, np.zeros((1 if k in {"nacon", "ncollision"} else 2,), np.int32))
    ints = {"dim", "geom", "flex", "vert", "efc_address", "worldid", "type", "geomcollisionid"}
    for k, tail in p.saved.CONTACT_TAILS.items():
        add("/data/contact/" + k, np.zeros((256,) + tail, np.int32 if k in ints else np.float32))
    for k in ("type", "id", "J", "D", "aref", "force", "state", "Ma"):
        add("/data/efc/" + k, np.zeros((2, 512, 20) if k == "J" else (2, 512),
                                    np.int32 if k in {"type", "id", "state"} else np.float32))
    while len(values) < 114: add("/data/padding" + str(len(values)), np.zeros((2, 1), np.float32))
    prepared = {k: np.frombuffer(v.tobytes(), np.uint8).copy() for k, v in values.items()}
    stages = {s: {k: v.copy() for k, v in values.items()} for s in p.STAGES}
    for stage in p.STAGES[1:]:
        data = stages[stage]
        data["/data/nacon"][:] = 2; data["/data/ncollision"][:] = 2
        data["/data/contact/worldid"][:2] = 1
        data["/data/contact/dim"][:2] = 3
        data["/data/contact/geom"][:2] = [(0, 29), (0, 64)]
        data["/data/contact/geomcollisionid"][:2] = [0, 1]
        data["/data/contact/type"][:2] = 1
        data["/data/contact/dist"][:2] = -.0002
        data["/data/contact/frame"][:2] = np.eye(3, dtype=np.float32)
        data["/data/contact/pos"][:2] = [[0, 1, 0], [0, -1, 0]]
        data["/data/contact/friction"][:2] = [2, 3, 0, 0, 0]
        data["/data/contact/efc_address"][:2] = -1
        if stage == "after_collision": continue
        data["/data/nf"][:] = 14; data["/data/nefc"][:] = [14, 22]
        for w in range(2):
            data["/data/efc/type"][w, :14] = 1
            data["/data/efc/id"][w, :14] = range(6, 20)
            data["/data/efc/J"][w, range(14), range(6, 20)] = 1
        for slot, first in enumerate((14, 18)):
            data["/data/contact/efc_address"][slot] = np.arange(first, first + 4)
            data["/data/efc/type"][1, first:first+4] = 6
            data["/data/efc/id"][1, first:first+4] = slot
            data["/data/efc/J"][1, first:first+4, 0] = 1
        if stage == "after_solve":
            data["/data/solver_niter"][:] = 1
            data["/data/efc/force"][1, 14:22] = [1, 2, 3, 4, 5, 6, 7, 8]
            data["/data/qfrc_constraint"][1, 0] = 36
    descriptor = dict(dofs=list(range(6, 20)), options=dict(iterations=10), selected_fields_sha256="a" * 64)
    report = dict(files={}, child=dict(input_manifest=manifest, plant=descriptor))
    arrays = {s + k: np.frombuffer(v.tobytes(), np.uint8).copy() for s, d in stages.items() for k, v in d.items()}
    layout = {k: p.expanded_layout(row) for k, row in manifest.items()}
    active = p.active_response(stages["after_solve"], descriptor)
    active.update({"sidecar/" + k: stages["after_solve"]["/data/contact/" + k][:2].copy()
                   for k in ("type", "geomcollisionid")})
    fields = {k: stages["after_solve"]["/data/" + k].reshape(2, -1).copy() for k in base.FIELDS}
    banks = {"prepared-inputs.npz": prepared, "gpu-active.npz": active, "cpu-active.npz": active,
             "gpu-fields.npz": fields, "cpu-fields.npz": {k: v.astype(np.float64) for k, v in fields.items()}}
    return arrays, layout, report, banks, stages


def test_complete_stage_reception_and_independent_force_arithmetic():
    arrays, layout, report, banks, stages = packet()
    assert len(arrays) == 570
    decoded = p.decode_stages(arrays, layout, report, banks)
    assert all(len(d) == 114 for d in decoded.values())
    active = p.active_response(decoded["after_solve"], report["child"]["plant"])
    assert active["contacts/force"].tolist() == [[10., -2., -3., 0., 0., 0.], [26., -2., -3., 0., 0., 0.]]
    result = p.analyze_stages(arrays, layout, report, banks)
    assert result["solver_changed_data_fields"] == ["/data/efc/force", "/data/qfrc_constraint", "/data/solver_niter"]
    assert result["complete_resultant_arithmetic"]["resultants"][0]["world_force_N"] == [10., -2., -3.]
    assert result["complete_resultant_arithmetic"]["resultants"][0]["world_torque_about_origin_Nm"] == [-3., 0., -10.]
    assert all(r["max_abs"] == 0 for r in result["complete_resultant_arithmetic"]["generalized"])
    assert all(not r["same_solver_inputs_established"] for r in result["descriptive_historical_resultants"])
    assert all("current_minus_historical" in v and "ada_minus_cpu" not in v
               for v in result["current_minus_historical_ada_fields"].values())


def test_fixed_generalized_reduction_order_without_blas(monkeypatch):
    active, fields = {}, {"qfrc_constraint": np.zeros((2, 20), np.float32)}
    for w in range(2):
        J = np.zeros((3, 20), np.float32); J[:, 0] = [2**60, 1, -2**60]
        active.update({f"rows/{w}/J": J, f"rows/{w}/force": np.ones(3, np.float32),
                       f"rows/{w}/type": np.array([1, 6, 1], np.int32)})
    monkeypatch.setattr(p.saved.p.prior, "reconstruction", lambda *args: pytest.fail("no backend-dependent reconstruction"))
    result = p.ordered_reconstruction(active, fields)
    # Fixed ascending-row IEEE64: (2**60 + 1) - 2**60 is0, not a regrouped1.
    assert all(r["total_generalized_force"][0] == 0 and r["contact_generalized_force"][0] == 1
               and r["friction_generalized_force"][0] == 0 for r in result)
    assert all(r["max_abs"] == 0 for r in result)


@pytest.mark.parametrize("damage", ["missing", "extra", "bytes", "dtype", "nan", "pose", "state", "layout", "stride"])
def test_complete_stage_packet_refusals(damage):
    arrays, layout, report, banks, _ = packet(); key = "after_solve/data/qpos"
    if damage == "missing": arrays.pop(key)
    if damage == "extra": arrays["extra"] = np.zeros(1, np.uint8)
    if damage == "bytes": arrays[key] = arrays[key][:-1]
    if damage == "dtype": arrays[key] = arrays[key].astype(np.int32)
    if damage == "nan": arrays["after_solve/data/qacc"].view(np.float32)[0] = np.nan
    if damage == "pose": arrays["after_solve/data/site_xpos"][0] ^= 1
    if damage == "state": arrays[key][0] ^= 1
    if damage == "layout": layout["/data/qpos"]["numpy_shape"] = [42]
    if damage == "stride": report["child"]["input_manifest"]["/data/qpos"]["strides"][0] += 4
    with pytest.raises(ValueError): p.decode_stages(arrays, layout, report, banks)


@pytest.mark.parametrize("damage", ["address", "overlap", "type", "id", "frictionJ", "uncovered", "negative_world", "large_world", "dim"])
def test_actual_active_row_linkage_refusals(damage):
    _, _, report, _, stages = packet(); d = stages["after_solve"]
    if damage == "address": d["/data/contact/efc_address"][0, 3] -= 1
    if damage == "overlap": d["/data/contact/efc_address"][1] = d["/data/contact/efc_address"][0]
    if damage == "type": d["/data/efc/type"][1, 14] = 1
    if damage == "id": d["/data/efc/id"][1, 14] = 1
    if damage == "frictionJ": d["/data/efc/J"][0, 0, 0] = 1
    if damage == "uncovered": d["/data/nefc"][1] += 1
    if damage == "negative_world": d["/data/contact/worldid"][0] = -1
    if damage == "large_world": d["/data/contact/worldid"][0] = 2
    if damage == "dim": d["/data/contact/dim"][0] = 4
    with pytest.raises(ValueError): p.active_response(d, report["child"]["plant"])


@pytest.mark.parametrize("damage", ["early_solver", "stale_before", "early_efc", "model_input", "candidate", "collision_counter", "too_many_iters"])
def test_downstream_stage_or_solver_write_refusals(damage):
    arrays, layout, report, banks, _ = packet()
    if damage == "early_solver": arrays["before_solve/data/solver_niter"].view(np.int32)[0] = 1
    if damage == "stale_before": arrays["before_collision/data/nacon"].view(np.int32)[0] = 1
    if damage == "early_efc": arrays["after_collision/data/nefc"].view(np.int32)[0] = 14
    if damage == "model_input": arrays["after_solve/data/efc/J"][0] ^= 1
    if damage == "candidate": arrays["before_solve/data/contact/pos"][0] ^= 1
    if damage == "collision_counter": arrays["before_solve/data/ncollision"].view(np.int32)[0] = 3
    if damage == "too_many_iters": arrays["after_solve/data/solver_niter"].view(np.int32)[0] = 11
    with pytest.raises(ValueError): p.analyze_stages(arrays, layout, report, banks)


@pytest.mark.parametrize("name,dtype,tail", [("float32", "float32", ()), ("int32", "int32", ()),
    ("bool", "bool", ()), ("vec2i", "int32", (2,)), ("vec2f", "float32", (2,)),
    ("vec3f", "float32", (3,)), ("quatf", "float32", (4,)), ("spatial_vectorf", "float32", (6,)),
    ("mat33f", "float32", (3, 3)), ("vec5f", "float32", (5,)), ("vec10f", "float32", (10,))])
def test_all_frozen_component_layouts(name, dtype, tail):
    ns = "mujoco_warp._src.types" if name in {"vec5f", "vec10f"} else "warp._src.types"
    width = int(np.prod(tail or (1,))) * np.dtype(dtype).itemsize
    row = dict(dtype=f"<class '{ns}.{name}'>", shape=[1, 2], strides=[0, width], bytes=2 * width)
    result = p.expanded_layout(row)
    assert result["numpy_shape"] == [1, 2] + list(tail) and result["numpy_dtype"] == dtype


@pytest.mark.parametrize("field", list(p.SERVICE_CAPS) + ["MainPID", "InvocationID", "Environment", "unit", "extra"])
def test_every_service_cap_or_identity_refuses(field):
    value = service()
    if field == "unit": value["unit"] = "foreign.service"
    elif field == "extra": value["properties"]["extra"] = "x"
    else: value["properties"][field] = "wrong"
    with pytest.raises(ValueError): p.validate_service(value, SOURCE)


def test_service_capture_and_cpu_environment_no_duplicate(monkeypatch):
    value = service(); p.validate_service(value, SOURCE, pid=123)
    monkeypatch.setattr(p.os, "getpid", lambda: 123)
    monkeypatch.setattr(p.saved.p.p.base.host, "read", lambda *args: "\n".join(k + "=" + v for k, v in value["properties"].items()))
    assert p.unit_identity(SOURCE) == value
    value["properties"]["Environment"] += " OMP_NUM_THREADS=1"
    with pytest.raises(ValueError): p.validate_service(value, SOURCE)


def receipt_fixture(tmp_path, monkeypatch):
    arrays, layout, report, banks, _ = packet()
    module = b"synthetic frozen module"
    monkeypatch.setattr(p.saved.p.p.base.host, "read", lambda *args, **kwargs: module)
    monkeypatch.setattr(p.saved.p.prior, "authenticated_banks", lambda root: (report, banks))
    monkeypatch.setattr(p.source, "audit", lambda: {"synthetic": "audit"})
    output = tmp_path / "response.json"
    result = dict(p.CONTRACT, protocol=p.PROTOCOL, decision=p.DECISION, source=SOURCE,
        module_sha256=sha256(module).hexdigest(), versions=p.saved.p.VERSIONS,
        predecessor_report_sha256=p.saved.p.prior.REPORT_SHA256, input_file_sha256=report["files"],
        plant=dict(selected_fields_sha256=report["child"]["plant"]["selected_fields_sha256"],
            options=report["child"]["plant"]["options"], collision_options=p.saved.p.COLLISION_OPTIONS,
            collision_flag_arm="unchanged-compiled-default"),
        options=dict(p.saved.p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11),
        model_array_sha256={}, data_layout=layout, stage_source_audit={"synthetic": "audit"},
        static_binding_sha256="3" * 64, static_binding_convention=p.STATIC_CONVENTION,
        private_cache_was_absent=True, loaded_binary_bytes_bound=False,
        private_cache_files={"cpu.o": dict(bytes=1, sha256="4" * 64)},
        existing_cpu_executables=[dict(module="synthetic", block_dim=1, opaque_handle="0x1", device="cpu",
            module_source_options_hash="5" * 64, meta={}, kernel_hook_count=1, loaded_binary_bytes_bound=False)],
        payload=p.saved.p.retain(output.with_suffix(".npz"), arrays),
        analysis=p.analyze_stages(arrays, layout, report, banks), running_service=service())
    result["solver_changed_data_fields"] = result["analysis"]["solver_changed_data_fields"]
    return output, result


def test_closed_receiver_recomputes_every_saved_stage(tmp_path, monkeypatch):
    output, result = receipt_fixture(tmp_path, monkeypatch)
    output.write_text(json.dumps(result))
    actual, arrays = p.receive(output, SOURCE, tmp_path)
    assert actual == result and len(arrays) == 570


@pytest.mark.parametrize("damage", ["extra", "source", "module", "flags", "boolean", "versions", "model", "analysis",
                                   "payload_hash", "payload_leaf", "payload_extra", "layout", "service", "binary", "static"])
def test_closed_receiver_refuses_unbound_or_promoted_receipt(tmp_path, monkeypatch, damage):
    output, result = receipt_fixture(tmp_path, monkeypatch)
    if damage == "extra": result["extra"] = True
    if damage == "source": result["source"] = "6" * 40
    if damage == "module": result["module_sha256"] = "6" * 64
    if damage == "flags": result["flags"] = dict(p.CONTRACT["flags"], training_authorized=True)
    if damage == "boolean": result["solver_qualified"] = 0
    if damage == "versions": result["versions"] = dict(p.saved.p.VERSIONS, unknown="1")
    if damage == "model": result["model_array_sha256"]["extra"] = "6" * 64
    if damage == "analysis": result["analysis"]["counters"]["after_solve"]["solver_niter"][0] = 2
    if damage == "payload_hash": result["payload"]["sha256"] = "6" * 64
    if damage == "payload_leaf": next(iter(result["payload"]["arrays"].values()))["sha256"] = "6" * 64
    if damage == "payload_extra": result["payload"]["extra"] = True
    if damage == "layout": result["data_layout"]["/data/qpos"]["numpy_dtype"] = "float64"
    if damage == "service": result["running_service"]["properties"]["MemoryMax"] = "max"
    if damage == "binary": result["loaded_binary_bytes_bound"] = True
    if damage == "static": result["static_binding_convention"] = "historical identity"
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError): p.receive(output, SOURCE, tmp_path)


@pytest.mark.parametrize("gpu", ["0", ""])
def test_cli_refuses_wrong_host_or_gpu_before_output(tmp_path, monkeypatch, gpu):
    output, cache = tmp_path / "never.json", tmp_path / "never-cache"
    monkeypatch.setattr(sys, "argv", ["response", "--input", str(tmp_path), "--output", str(output),
                                      "--cache", str(cache), "--source", SOURCE])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    with pytest.raises(ValueError): p.main()
    assert not output.exists() and not cache.exists()
