"""Frozen-pose CPU collision replay; stop before constraints, never admission."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct

from mjlab_microduck import ada_contact_diagnosis as prior

need = prior.need
p = prior.p
PROTOCOL = "microduck-ada-collision-only-cpu-v1"
KINEMATIC = ("xpos", "xquat", "xmat", "xanchor", "xaxis", "xipos", "ximat", "geom_xpos", "geom_xmat")
CONTACT = ("dist", "pos", "frame", "includemargin", "friction", "solref", "solreffriction",
           "solimp", "dim", "geom", "flex", "vert", "efc_address")
COUNTERS = ("ne", "nf", "nl", "nefc", "solver_niter")
VERSIONS = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
            "mujoco-warp": "3.8.1", "mjlab": "1.3.0"}
COLLISION_OPTIONS = dict(enableflags=0, disableflags=0, cone=0, ccd_iterations=35, ccd_tolerance=1e-6)
WARP_COLLISION_OPTIONS = dict(COLLISION_OPTIONS, ccd_tolerance=struct.unpack("<f", struct.pack("<f", 1e-6))[0])


def collision_options(opt, *, warp=False):
    values = {k: getattr(opt, k) for k in COLLISION_OPTIONS if k != "ccd_tolerance"}
    if warp:
        import numpy as np
        tolerance = opt.ccd_tolerance.numpy()
        need(tolerance.shape == (1,) and tolerance.dtype == np.float32
             and tolerance.tobytes() == np.asarray([COLLISION_OPTIONS["ccd_tolerance"]], dtype=np.float32).tobytes(),
             "unchanged Warp float32 CCD tolerance array bytes")
        values["ccd_tolerance"] = float(tolerance[0])
    else: values["ccd_tolerance"] = opt.ccd_tolerance
    need(values == (WARP_COLLISION_OPTIONS if warp else COLLISION_OPTIONS), "unchanged predeclared collision scalar options")
    return values


def state_inputs(report, banks):
    """Decode all seven prepared float32 state inputs, not a newly placed pose."""
    import numpy as np
    result = {}
    for name in p.base.UNCHANGED:
        key = "/data/" + name
        row = report["child"]["input_manifest"][key]
        raw = banks["prepared-inputs.npz"][key].tobytes()
        width = p.base.WIDTHS[name]
        need(len(raw) == 2 * width * 4 and sha256(raw).hexdigest() == row["sha256"], "exact prepared state bytes")
        value = np.frombuffer(raw, dtype=np.float32).reshape(2, width).copy()
        need(np.array_equal(value, banks["gpu-fields.npz"][name]), "unchanged predecessor state")
        result[name] = value
    return result


def bind_plant(model, descriptor):
    """Static binding only: deliberately never calls describe/place_on_floor."""
    from mjlab_microduck import stance_plant_evidence as plant
    from mjlab_microduck.first_attempt_smoke import canonical
    from mjlab_microduck.stance_lesson_contract import JOINTS
    from mjlab_microduck.football_contact_fixture import FEET
    values = {name: getattr(model, name).tolist() for name in plant.ARRAYS}
    need(descriptor["selected_fields"] == list(plant.ARRAYS)
         and sha256(canonical(values).encode()).hexdigest() == descriptor["selected_fields_sha256"], "same compiled selected fields")
    need(plant.asset_hashes() == descriptor["assets"]
         and [model.nq, model.nv, model.nu, model.ngeom, model.neq, model.nmocap] == descriptor["topology"]
         and [int(model.joint(n).qposadr[0]) for n in JOINTS] == descriptor["qids"]
         and [int(model.joint(n).dofadr[0]) for n in JOINTS] == descriptor["dofs"]
         and int(model.geom("hold_floor").id) == descriptor["floor"]
         and [int(model.geom(n).id) for n in FEET] == descriptor["feet"], "same assets and compiled topology")
    options = dict(timestep=model.opt.timestep, gravity=model.opt.gravity.tolist(),
                   integrator=int(model.opt.integrator), solver=int(model.opt.solver),
                   iterations=model.opt.iterations, tolerance=model.opt.tolerance)
    need(options == descriptor["options"] and model.nflex == 0, "same rigid plant options")
    return dict(selected_fields_sha256=descriptor["selected_fields_sha256"], options=options,
                collision_options=collision_options(model.opt), collision_flag_arm="unchanged-compiled-default")


def model_binding(model, report, banks):
    """Compare EVERY logical Warp model array, including meshes, before calls."""
    actual = p.all_arrays(model, "/model")
    expected = {k: v for k, v in report["child"]["input_manifest"].items() if k.startswith("/model/")}
    need(set(actual) == set(expected), "complete same Warp model array inventory")
    result = {}
    for key, array in actual.items():
        row = expected[key]
        raw = array.numpy().tobytes()
        need(str(array.dtype) == row["dtype"] and list(array.shape) == row["shape"]
             and list(array.strides) == row["strides"] and raw == banks["prepared-inputs.npz"][key].tobytes(),
             "same prepared Warp model array: " + key)
        result[key] = sha256(raw).hexdigest()
    return result


def native_candidates(datas):
    """All candidate fields; no mj_contactForce or synthesized solver forces."""
    import numpy as np
    bank = {}
    for world, data in enumerate(datas):
        for name in CONTACT + ("elem", "exclude", "mu", "H"):
            value = np.asarray(getattr(data.contact, name)).copy()
            if name == "frame": value = value.reshape(data.ncon, 3, 3)
            bank[f"{world}/{name}"] = value
    return bank


def warp_candidates(data):
    from dataclasses import fields
    count = int(data.nacon.numpy()[0])
    need(0 <= count <= data.naconmax, "no contact overflow")
    return {f.name: getattr(data.contact, f.name).numpy()[:count].copy() for f in fields(data.contact)}


def candidate_summary(native, warp):
    """Preserve raw slots and ordered pairs; eligibility is NOT an EFC address."""
    import numpy as np
    result = {}
    for side, tables in (("native", [{k: native[f"{w}/{k}"] for k in CONTACT} for w in range(2)]),
                         ("warp_cpu", [{k: v[warp["worldid"] == w] for k, v in warp.items()} for w in range(2)])):
        rows = []
        for w, table in enumerate(tables):
            count = len(table["dist"])
            need(all(np.isfinite(v).all() for v in table.values()), "finite candidate fields")
            need((table["efc_address"] == -1).all(), "no constructed contact address")
            eligible = table["dist"] < table["includemargin"]
            slots = np.arange(count) if side == "native" else np.flatnonzero(warp["worldid"] == w)
            groups = {}
            for i, (geom, dim) in enumerate(zip(table["geom"], table["dim"])):
                key = (int(geom[0]), int(geom[1]), int(dim), bool(eligible[i]))
                groups.setdefault(key, []).append(int(slots[i]))
            rows.append(dict(world=w, count=count, within_includemargin_count=int(eligible.sum()),
                             groups=[dict(ordered_key=list(k), raw_slots=v) for k, v in sorted(groups.items())]))
            if side == "warp_cpu":
                rows[-1]["constraint_type_bit_count"] = int(((table["type"] & 1) != 0).sum())
        result[side] = rows
    return result


def differences(left, right):
    import numpy as np
    need(set(left) == set(right), "complete kinematic fields")
    result = {}
    for name in left:
        a, b = left[name], right[name]
        need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "finite kinematic shapes")
        delta = b.astype(np.float64) - a.astype(np.float64)
        result[name] = dict(max_abs=float(np.abs(delta).max()),
                           float32_bit_mismatches=int(np.count_nonzero(a.astype(np.float32).view(np.uint32) != b.astype(np.float32).view(np.uint32))))
    return result


def replay(report, banks):
    import numpy as np
    import mujoco
    import mujoco_warp as mjwarp
    import warp as wp
    from mujoco_warp._src.smooth import kinematics
    from mjlab.sim.randomization import expand_model_fields
    from mjlab_microduck.stance_warp_runtime import build_entity
    state = state_inputs(report, banks)
    descriptor = report["child"]["plant"]
    models, datas, bindings = [], [], []
    for w in range(2):
        model = build_entity().compile()
        bindings.append(bind_plant(model, descriptor))
        for name in ("dof_frictionloss", "dof_damping"):
            getattr(model, name)[:] = banks["motor.npz"][name][w].astype(np.float64)
        data = mujoco.MjData(model)
        for name, value in state.items():
            if name == "time": data.time = float(value[w, 0])
            else: getattr(data, name)[:] = value[w].reshape(getattr(data, name).shape).astype(np.float64)
        models.append(model); datas.append(data)
    with wp.ScopedDevice("cpu"):
        model = mjwarp.put_model(models[0])
        from mujoco_warp._src.types import BroadphaseType, BroadphaseFilter
        warp_options = collision_options(model.opt, warp=True)
        need(model.opt.broadphase == BroadphaseType.NXN
             and model.opt.broadphase_filter == (BroadphaseFilter.PLANE | BroadphaseFilter.SPHERE | BroadphaseFilter.OBB), "unchanged Warp collision dispatch")
        warp_options.update(broadphase=int(model.opt.broadphase), broadphase_filter=int(model.opt.broadphase_filter))
        # make_data has ONE internal native kinematics call for static poses;
        # no forward/constraints/solve. Measured fixture kinematics follows below.
        data = mjwarp.make_data(models[0], nworld=2, nconmax=128, njmax=512)
        expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device="cpu"))
        for name, value in state.items():
            array = getattr(data, name)
            wp.copy(array, wp.array(value.reshape(array.numpy().shape), dtype=array.dtype, device="cpu"))
        model_hashes = model_binding(model, report, banks)
        for m, d in zip(models, datas): mujoco.mj_kinematics(m, d)
        kinematics(model, data)
        wp.synchronize_device("cpu")
        native_pose = {k: np.stack([np.asarray(getattr(d, k)).reshape(-1, 3, 3) if k.endswith("mat")
                                    else np.asarray(getattr(d, k)) for d in datas]) for k in KINEMATIC}
        warp_pose = {k: getattr(data, k).numpy().copy() for k in KINEMATIC}
        # These are the actual complete geometry inputs at each collision call.
        for m, d in zip(models, datas): mujoco.mj_collision(m, d)
        mjwarp.collision(model, data)
        wp.synchronize_device("cpu")
        native = native_candidates(datas)
        warp = warp_candidates(data)
        need(int(data.ncollision.numpy()[0]) <= data.naconmax, "no broadphase overflow")
        need(datas[0].ncon == 0 and datas[1].ncon > 0 and (warp["worldid"] == 0).sum() == 0
             and (warp["worldid"] == 1).sum() > 0, "declared clear and shallow-contact fixtures reached")
        counters = dict(native=[{k: int(np.asarray(getattr(d, k)).reshape(-1)[0]) for k in COUNTERS} for d in datas],
                        warp_cpu={k: getattr(data, k).numpy().tolist() for k in COUNTERS})
        need(all(v == 0 for row in counters["native"] for v in row.values())
             and all(v == 0 for row in counters["warp_cpu"].values() for v in row), "no EFC or solver iterations")
        need(all(not d.warning.number.any() and not d.solver_niter.any() for d in datas), "no native warning or hidden solver iteration")
        for name, value in state.items():
            need(getattr(data, name).numpy().tobytes() == value.reshape(getattr(data, name).numpy().shape).tobytes(), "unchanged Warp state bytes")
            need(all(np.asarray(getattr(d, name)).reshape(-1).tobytes() == value[w].astype(np.float64).tobytes()
                     for w, d in enumerate(datas)), "unchanged native float64 state bytes")
        need(model_binding(model, report, banks) == model_hashes, "static model unchanged after collision")
    saved_pose = {k: np.frombuffer(banks["prepared-inputs.npz"]["/data/" + k].tobytes(), dtype=np.float32)
                    .reshape(warp_pose[k].shape) for k in KINEMATIC}
    arrays = {f"native/contact/{k}": v for k, v in native.items()}
    arrays.update({f"warp_cpu/contact/{k}": v for k, v in warp.items()})
    arrays.update({f"{side}/pose/{k}": v for side, pose in (("native", native_pose), ("warp_cpu", warp_pose)) for k, v in pose.items()})
    arrays.update({"state/" + k: v for k, v in state.items()})
    # Retain exact local support geometry and hull graph inputs; these are not
    # observed selected vertex IDs (the collision kernels expose no such trace).
    for name in ("geom_dataid", "geom_rbound", "mesh_vertadr", "mesh_vertnum", "mesh_graphadr"):
        arrays["warp_cpu/model/" + name] = getattr(model, name).numpy().copy()
    for geom in descriptor["feet"]:
        mesh = int(models[0].geom_dataid[geom])
        start, count = int(models[0].mesh_vertadr[mesh]), int(models[0].mesh_vertnum[mesh])
        arrays[f"native/mesh/{geom}/vertices"] = models[0].mesh_vert[start:start + count].copy()
        arrays[f"warp_cpu/mesh/{geom}/vertices"] = model.mesh_vert.numpy()[start:start + count].copy()
    return dict(plant_bindings=bindings, warp_collision_options=warp_options, model_array_sha256=model_hashes, counters=counters,
                candidates=candidate_summary(native, warp),
                kinematic_warp_minus_native=differences(native_pose, warp_pose),
                kinematic_warp_cpu_minus_retained_ada=differences(saved_pose, {k: warp_pose[k] for k in saved_pose}),
                ncollision=data.ncollision.numpy().tolist(),
                support_selection_trace_available=False, support_geometry_retained=True,
                native_uncomputed_solver_scratch_fields=["mu", "H"],
                candidate_key_inclusion="dist < includemargin; not an EFC address or physical point identity",
                measured_native_kinematics_calls=2, allocation_native_kinematics_calls=1,
                measured_warp_kinematics_calls=1, native_collision_calls=2, warp_cpu_collision_calls=1,
                new_constraint_calls=0, new_solver_calls=0, new_integration_steps=0,
                flags=p.FLAGS, training_authorized=False, physical_motion_authorized=False), arrays


def retain(path, arrays):
    import numpy as np
    need(sum(v.nbytes for v in arrays.values()) < 15 * 1024**2, "bounded replay bank")
    need(all(np.isfinite(v).all() for v in arrays.values()), "finite complete replay bank")
    with path.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    raw = path.read_bytes()
    need(len(raw) < 16 * 1024**2, "bounded replay leaf")
    return dict(file=path.name, bytes=len(raw), sha256=sha256(raw).hexdigest(),
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()})


def receive(output, source, root):
    """CPU-only exact byte reception, recomputing summaries from the full bank."""
    import io
    import numpy as np
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024, "plain bounded replay JSON")
    result = json.loads(output.read_bytes())
    report, banks = prior.authenticated_banks(root)
    need(result["protocol"] == PROTOCOL and result["source"] == source
         and result["predecessor_report_sha256"] == prior.REPORT_SHA256 and result["predecessor_source"] == prior.SOURCE
         and result["decision"] == "collision-boundary-diagnostic-not-admission"
         and result["input_file_sha256"] == report["files"]
         and result["model_array_sha256"] == {k: v["sha256"] for k, v in report["child"]["input_manifest"].items() if k.startswith("/model/")}
         and result["flags"] == p.FLAGS and all(v is False for v in result["flags"].values())
         and result["training_authorized"] is result["physical_motion_authorized"] is False
         and result["new_constraint_calls"] == result["new_solver_calls"] == result["new_integration_steps"] == 0,
         "same unqualified replay envelope")
    module = p.base.host.read("git", "show", source + ":src/mjlab_microduck/ada_collision_replay.py", binary=True)
    need(sha256(module).hexdigest() == result["module_sha256"], "source-bound replay module")
    need(result["measured_native_kinematics_calls"] == result["native_collision_calls"] == 2
         and result["allocation_native_kinematics_calls"] == result["measured_warp_kinematics_calls"] == result["warp_cpu_collision_calls"] == 1
         and all(result["versions"][k].split("+")[0] == v for k, v in VERSIONS.items())
         and result["warp_collision_options"] == dict(WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
         and all(row["collision_options"] == COLLISION_OPTIONS
                 and row["selected_fields_sha256"] == report["child"]["plant"]["selected_fields_sha256"] for row in result["plant_bindings"])
         and all(v == 0 for row in result["counters"]["native"] for v in row.values())
         and all(v == 0 for row in result["counters"]["warp_cpu"].values() for v in row), "unchanged call/counter/options contract")
    path = output.with_suffix(".npz")
    payload = result["payload"]
    need(path.is_file() and not path.is_symlink() and path.name == payload["file"]
         and path.stat().st_size == payload["bytes"] < 16 * 1024**2, "plain complete replay bank")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == payload["sha256"], "exact complete replay bytes")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) and set(bank.files) == set(payload["arrays"]), "complete unique replay fields")
        arrays = {k: bank[k].copy() for k in bank.files}
    for key, array in arrays.items():
        row = payload["arrays"][key]
        need(list(array.shape) == row["shape"] and str(array.dtype) == row["dtype"]
             and sha256(array.tobytes()).hexdigest() == row["sha256"] and np.isfinite(array).all(), "exact finite replay array")
    native = {k.removeprefix("native/contact/"): v for k, v in arrays.items() if k.startswith("native/contact/")}
    warp = {k.removeprefix("warp_cpu/contact/"): v for k, v in arrays.items() if k.startswith("warp_cpu/contact/")}
    need(set(native) == {f"{w}/{k}" for w in range(2) for k in CONTACT + ("elem", "exclude", "mu", "H")}
         and set(warp) == set(CONTACT) | {"worldid", "type", "geomcollisionid"}, "complete native/Warp contact structs")
    need(len(native["0/dist"]) == 0 and len(native["1/dist"]) > 0
         and ((warp["worldid"] >= 0) & (warp["worldid"] < 2)).all()
         and (warp["worldid"] == 0).sum() == 0 and (warp["worldid"] == 1).sum() > 0, "received declared fixture occupancy")
    need(candidate_summary(native, warp) == result["candidates"], "recomputed candidate summary")
    expected = {"native/contact/" + k for k in native} | {"warp_cpu/contact/" + k for k in warp}
    expected |= {f"{s}/pose/{k}" for s in ("native", "warp_cpu") for k in KINEMATIC}
    expected |= {"state/" + k for k in p.base.UNCHANGED}
    expected |= {"warp_cpu/model/" + k for k in ("geom_dataid", "geom_rbound", "mesh_vertadr", "mesh_vertnum", "mesh_graphadr")}
    expected |= {f"{s}/mesh/{g}/vertices" for s in ("native", "warp_cpu") for g in report["child"]["plant"]["feet"]}
    need(set(arrays) == expected, "complete fixed replay array inventory")
    poses = {s: {k: arrays[f"{s}/pose/{k}"] for k in KINEMATIC} for s in ("native", "warp_cpu")}
    need(differences(poses["native"], poses["warp_cpu"]) == result["kinematic_warp_minus_native"], "recomputed kinematic residuals")
    for name, value in state_inputs(report, banks).items():
        need(arrays["state/" + name].tobytes() == value.tobytes(), "received exact seven-state inputs")
    saved_pose = {k: np.frombuffer(banks["prepared-inputs.npz"]["/data/" + k].tobytes(), dtype=np.float32)
                 .reshape(poses["warp_cpu"][k].shape) for k in KINEMATIC}
    need(differences(saved_pose, poses["warp_cpu"]) == result["kinematic_warp_cpu_minus_retained_ada"], "recomputed retained-Ada kinematic comparison")
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden collision replay")
    need(re.fullmatch(r"[0-9a-f]{40}", args.source) and Path.cwd().resolve() == Path(__file__).resolve().parents[2]
         and p.base.host.read("git", "rev-parse", "HEAD") == args.source
         and p.base.host.read("git", "branch", "--show-current") == p.base.host.BRANCH
         and not p.base.host.read("git", "status", "--porcelain"), "clean exact collision replay source")
    raw = Path(__file__).read_bytes()
    need(raw == p.base.host.read("git", "show", args.source + ":src/mjlab_microduck/ada_collision_replay.py", binary=True), "committed replay module bytes")
    need(not args.output.exists() and not args.output.with_suffix(".npz").exists()
         and not args.cache.exists() and args.cache.is_absolute() and args.output.is_absolute()
         and args.cache.parent.resolve(strict=True) == args.cache.parent
         and args.output.parent.resolve(strict=True) == args.output.parent, "fresh canonical replay output/cache paths")
    from importlib.metadata import version
    versions = {name: version(name) for name in VERSIONS}
    need(all(versions[k].split("+")[0] == v for k, v in VERSIONS.items()), "unchanged frozen packages")
    report, banks = prior.authenticated_banks(args.input)
    import warp as wp
    import torch
    args.cache.mkdir()
    from mjlab_microduck.ada_saved_pose_collision import configure_private_cpu_cache
    configure_private_cpu_cache(wp, args.cache)
    wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(), "CPU devices only")
    result, arrays = replay(report, banks)
    need(not torch.cuda.is_initialized(), "CUDA remains uninitialized")
    result.update(protocol=PROTOCOL, decision="collision-boundary-diagnostic-not-admission",
                  source=args.source, module_sha256=sha256(raw).hexdigest(), versions=versions,
                  predecessor_report_sha256=prior.REPORT_SHA256, predecessor_source=prior.SOURCE,
                  input_file_sha256=report["files"], payload=retain(args.output.with_suffix(".npz"), arrays))
    p.base.host.write_json(args.output, result)
    print(json.dumps(dict(decision=result["decision"], candidates=result["candidates"]), sort_keys=True))


if __name__ == "__main__": main()
