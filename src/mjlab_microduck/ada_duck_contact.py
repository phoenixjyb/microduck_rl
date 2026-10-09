"""Bounded paired motor-friction/shallow-contact forward, not training admission."""
import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from mjlab_microduck import ada_duck_forward as base

MODULE = "mjlab_microduck.ada_duck_contact"
PROTOCOL = "microduck-ada-motor-contact-forward-oct9-v1"
FLAGS = base.FLAGS
BOUNDS = base.BOUNDS
PAYLOAD_MAX = 15 * 1024**2
CHILD_JSON_LIMIT = 192 * 1024
PAYLOADS = ("cpu-fields.npz", "gpu-fields.npz", "cpu-active.npz", "gpu-active.npz", "prepared-inputs.npz", "motor.npz", "bam-source.npz")
FILES = ("launch.json", "child.log", "child.json") + PAYLOADS
COLLECTION_DECISION = "ada-motor-contact-forward-collected-pending-reception-not-training"
need = base.need
ROOT_OFFSETS = (0., -.0002)
ROW_FIELDS = ("type", "id", "J", "D", "aref", "force", "state")
CONTACT_FIELDS = ("worldid", "slot", "geom", "dim", "dist", "pos", "frame", "friction", "force", "efc_address")
INPUT_WIDTHS = {"<class '" + namespace + "." + name + "'>": width
                for namespace, names in (("warp._src.types", dict(int32=4, float32=4, vec3f=12, vec2f=8, vec2i=8,
                    bool=1, vec3i=12, spatial_vectorf=24, quatf=16, mat33f=36, vec4f=16)),
                    ("mujoco_warp._src.types", dict(vec5f=20, vec10f=40, vec11f=44, vec8i=32, vec8f=32, vec_pluginattr=512)))
                for name, width in names.items()}


def validate_input_layout(row):
    """Logical-byte consistency only, not compiled-device/origin authentication."""
    need(row["dtype"] in INPUT_WIDTHS and type(row["shape"]) is list and 1 <= len(row["shape"]) <= 4
         and type(row["strides"]) is list and len(row["strides"]) == len(row["shape"])
         and all(type(x) is int and 0 <= x <= PAYLOAD_MAX for x in row["shape"] + row["strides"]), "bounded declared input layout")
    size = INPUT_WIDTHS[row["dtype"]]
    for dimension, stride in zip(reversed(row["shape"]), reversed(row["strides"])):
        # put_model uses zero-stride singleton broadcast dimensions. Empty
        # axes are valid and contribute zero logical bytes, not padding.
        need(stride == size or (dimension == 1 and stride == 0), "consistent logical input strides")
        size *= dimension
    need(size == row["bytes"], "consistent logical input byte count")


def specification():
    return dict(protocol=PROTOCOL, worlds=2, root_height_offsets_m=list(ROOT_OFFSETS),
                motor_preparations=1, fixture_forwards_before_motor=1, measured_forwards=1,
                integration_steps=0, optimizer_steps=0, graph_calls=0,
                fields=list(base.FIELDS), row_fields=list(ROW_FIELDS), contact_fields=list(CONTACT_FIELDS),
                nominal_motor=dict(vin=7.5, vin_drop_gain=.1, kp=200., model="xl330-m6"),
                numerical_acceptance_tolerance=None, contact_key="world,ordered-geoms,dimension,included; duplicates unresolved",
                flags=FLAGS)


def directory(source):
    path = base.host.ROOT / "artifacts/evaluations" / ("ada-duck-contact-" + source[:12])
    need(path.parent.resolve(strict=True) == path.parent, "canonical evidence parent")
    return path


def identity(source):
    value = base.identity(source)
    raw = Path(__file__).read_bytes()
    need(raw == base.host.read("git", "show", source + ":src/mjlab_microduck/ada_duck_contact.py", binary=True), "committed contact diagnostic")
    return dict(**value, contact_module_sha256=sha256(raw).hexdigest())


def owned_service(source):
    return base.owned_service(source, prefix="microduck-ada-duck-contact-")


def retain(root, name, arrays):
    import numpy as np
    need(name in PAYLOADS and sum(x.nbytes for x in arrays.values()) <= PAYLOAD_MAX - 256 * 1024, "bounded raw payload plus ZIP-header reserve")
    with (root / name).open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    raw = (root / name).read_bytes()
    need(len(raw) <= PAYLOAD_MAX, "bounded NPZ file")
    return dict(file=name, bytes=len(raw), sha256=sha256(raw).hexdigest())


def all_arrays(value, path=""):
    from dataclasses import fields, is_dataclass
    import warp as wp
    if isinstance(value, wp.array): return {path: value}
    if is_dataclass(value): children = [(f.name, getattr(value, f.name)) for f in fields(value)]
    elif isinstance(value, dict): children = sorted(value.items())
    elif isinstance(value, (list, tuple)): children = list(enumerate(value))
    else: children = []
    result = {}
    for key, child in children: result.update(all_arrays(child, path + "/" + str(key)))
    return result


def input_snapshot(env):
    import numpy as np
    env._sync()
    result, manifest = {}, {}
    for name, array in all_arrays(dict(model=env.model, data=env.data)).items():
        raw = array.numpy().tobytes()
        result[name] = np.frombuffer(raw, dtype=np.uint8).copy()
        manifest[name] = dict(dtype=str(array.dtype), shape=list(array.shape), strides=list(array.strides),
                              bytes=len(raw), sha256=sha256(raw).hexdigest())
    need(len(manifest) > 400, "complete model/data array traversal")
    return result, manifest


def native_contacts(models, datas):
    import numpy as np
    import mujoco
    rows = {key: [] for key in CONTACT_FIELDS}
    for world, (model, data) in enumerate(zip(models, datas)):
        for slot, contact in enumerate(data.contact):
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, slot, force)
            values = dict(worldid=world, slot=slot, geom=contact.geom.copy(), dim=int(contact.dim),
                          dist=float(contact.dist), pos=contact.pos.copy(), frame=contact.frame.reshape(3, 3).copy(),
                          friction=contact.friction.copy(), force=force, efc_address=int(contact.efc_address))
            for key in rows: rows[key].append(values[key])
    tails = dict(geom=(2,), pos=(3,), frame=(3, 3), friction=(5,), force=(6,))
    ints = ("worldid", "slot", "geom", "dim", "efc_address")
    return {key: np.asarray(value, dtype=np.int32 if key in ints else np.float64).reshape((-1,) + tails.get(key, ()))
            for key, value in rows.items()}


def active_bank(env, datas, models):
    import numpy as np
    import warp as wp
    from mjlab_microduck.stance_contact_evidence import read_contacts
    cpu = {}
    gpu = {}
    for w, data in enumerate(datas):
        for key in ROW_FIELDS:
            value = getattr(data, "efc_" + key).copy()
            if key == "J": value = value.reshape(data.nefc, models[w].nv)
            cpu[f"rows/{w}/{key}"] = value.astype(np.int32 if key in ("type", "id", "state") else np.float64)
            gpu[f"rows/{w}/{key}"] = getattr(env.data.efc, key).numpy()[w, :int(env._view("nefc")[w])].copy()
    table = read_contacts(env.model, env.data)
    count = len(table["worldid"])
    for key in CONTACT_FIELDS:
        value = np.arange(count, dtype=np.int32) if key == "slot" else table[key].cpu().numpy().copy()
        if key == "efc_address": value = value[:, 0].copy()
        gpu["contacts/" + key] = value
    # Sidecars are recorded, not treated as cross-backend contact identities.
    for key in ("type", "geomcollisionid"):
        gpu["sidecar/" + key] = getattr(env.data.contact, key).numpy()[:count].copy()
    for key, value in native_contacts(models, datas).items(): cpu["contacts/" + key] = value
    return cpu, gpu


def contact_groups(table):
    groups = {}
    for i in range(len(table["worldid"])):
        key = (int(table["worldid"][i]), *map(int, table["geom"][i]), int(table["dim"][i]), bool(table["efc_address"][i] >= 0))
        groups.setdefault(key, []).append(i)
    return groups


def analyze_active(cpu, gpu, dofs, floor, feet):
    """Structural coverage and honest candidates; no row-order equivalence gate."""
    import numpy as np
    expected = {f"rows/{w}/{key}" for w in range(2) for key in ROW_FIELDS} | {"contacts/" + key for key in CONTACT_FIELDS}
    need(set(cpu) == expected and set(gpu) == expected | {"sidecar/type", "sidecar/geomcollisionid"}, "complete active inventory")
    for bank, floating in ((cpu, np.float64), (gpu, np.float32)):
        for key, array in bank.items():
            integer = key.endswith(("/type", "/id", "/state", "/worldid", "/slot", "/geom", "/dim", "/efc_address", "/geomcollisionid"))
            need(array.dtype == (np.int32 if integer else floating) and np.isfinite(array).all(), "full active dtype/finiteness: " + key)
        contacts = {key: bank["contacts/" + key] for key in CONTACT_FIELDS}
        counts = np.bincount(contacts["worldid"], minlength=2)
        need(counts.shape == (2,) and counts[0] == 0 and 0 < counts[1] <= 128, "air-gap and shallow-contact fixtures")
        n = len(contacts["worldid"])
        if bank is gpu:
            need(all(bank["sidecar/" + key].shape == (n,) for key in ("type", "geomcollisionid")), "complete contact sidecars")
        for key, tail in dict(worldid=(), slot=(), geom=(2,), dim=(), dist=(), pos=(3,), frame=(3, 3), friction=(5,), force=(6,), efc_address=()).items():
            need(contacts[key].shape == (n,) + tail, "complete contact shape: " + key)
        need((contacts["dim"] == 3).all() and (contacts["efc_address"] >= 0).all()
             and (contacts["dist"] < 0).all() and (contacts["friction"] >= 0).all(), "included penetrating pyramidal contacts")
        need({frozenset(map(int, pair)) for pair in contacts["geom"]} == {frozenset((floor, foot)) for foot in feet}, "both feet and no forbidden contact")
        for w in range(2):
            types, ids, J = (bank[f"rows/{w}/{key}"] for key in ("type", "id", "J"))
            count = len(types)
            need(count == 14 + 4 * counts[w] and count <= 512 and ids.shape == (count,) and J.shape == (count, 20), "full friction/contact rows")
            for key in ("D", "aref", "force", "state"): need(bank[f"rows/{w}/{key}"].shape == (count,), "complete row shape")
            friction = np.flatnonzero(types == 1)
            need(len(friction) == 14 and set(map(int, ids[friction])) == set(dofs), "one friction row per controlled DOF")
            for row in friction:
                unit = np.zeros(20, dtype=floating); unit[int(ids[row])] = 1
                need(np.array_equal(J[row], unit), "friction Jacobian addresses actual DOF")
            covered = set(map(int, friction))
            for slot in np.flatnonzero(contacts["worldid"] == w):
                address = int(contacts["efc_address"][slot])
                rows = set(range(address, address + 4))
                need(address >= 0 and address + 4 <= count and not covered.intersection(rows), "unambiguous local address group")
                need((types[address:address + 4] == 6).all()
                     and (ids[address:address + 4] == contacts["slot"][slot]).all(), "contact row slot linkage")
                covered.update(rows)
            need(covered == set(range(count)), "every active row structurally accounted for")
    a = contact_groups({k: cpu["contacts/" + k] for k in CONTACT_FIELDS})
    b = contact_groups({k: gpu["contacts/" + k] for k in CONTACT_FIELDS})
    candidates, issues = [], []
    for key in sorted(a.keys() | b.keys()):
        left, right = a.get(key, []), b.get(key, [])
        entry = dict(key=list(key), cpu_rows=left, gpu_rows=right)
        (candidates if len(left) == len(right) == 1 else issues).append(entry)
    friction_residuals = {}
    for w in range(2):
        ai = {int(x): i for i, x in enumerate(cpu[f"rows/{w}/id"]) if cpu[f"rows/{w}/type"][i] == 1}
        bi = {int(x): i for i, x in enumerate(gpu[f"rows/{w}/id"]) if gpu[f"rows/{w}/type"][i] == 1}
        friction_residuals[str(w)] = {key: float(np.max(np.abs(np.asarray([gpu[f"rows/{w}/{key}"][bi[d]] for d in dofs], dtype=np.float64)
                    - np.asarray([cpu[f"rows/{w}/{key}"][ai[d]] for d in dofs], dtype=np.float64)))) for key in ("J", "D", "aref", "force")}
    return dict(contact_status="unresolved-correspondence" if issues else "unique-key-candidates",
                candidates=candidates, issues=issues, friction_residuals=friction_residuals,
                physical_contact_identity_established=False, solver_qualified=False)


def physics(root, device="cuda:0"):
    import numpy as np
    import torch
    import mujoco
    from mjlab_microduck import stance_plant_evidence as plant
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime, ActuatorCmd
    from mjlab_microduck.stance_forward_graph import binding
    env = WarpStanceRuntime(2, device=device)
    descriptor = plant.describe(env.native)
    need(descriptor == plant.reference(), "fresh actual compiled plant")
    env._view("qpos")[1, 2] += ROOT_OFFSETS[1]
    # The motor budget reads the preceding solve's forces. Refresh at the
    # declared fixture pose BEFORE proposing, still without any integration.
    env._forward()
    initial = {key: env._view(key).cpu().numpy().copy() for key in base.UNCHANGED}
    bam_source = {key: env._view(key).cpu().numpy().copy() for key in
                  ("qpos", "qvel", "qfrc_bias", "qfrc_constraint", "qfrc_actuator", "nefc", "nf")}
    bam_source.update(previous=env.motor.previous.cpu().numpy().copy(),
                      voltage=env.motor.voltage.cpu().numpy().copy(), kp=env.motor.kp.cpu().numpy().copy(),
                      own_friction=env.actuator._dof_friction_force(20).cpu().numpy().copy(),
                      position_target=env.delay.peek().cpu().numpy().copy())
    position, velocity = env._view("qpos")[:, env.qids], env._view("qvel")[:, env.dofs]
    zeros = torch.zeros_like(position)
    need(torch.equal(env.delay.peek(), position), "genuine zero-error motor target")
    proposal = env.motor.compute(ActuatorCmd(env.delay.peek(), zeros, zeros, position, velocity), env.live.clone())
    need(proposal["accepted"].all() and not proposal["rejected"].any()
         and torch.isfinite(proposal["torque_nm"]).all() and not proposal["torque_nm"].any(), "one zero-error accepted zero-torque BAM preparation")
    env._view("ctrl")[:, env.ctrl_ids] = proposal["torque_nm"]
    motor = {key: value.cpu().numpy().copy() for key, value in env.motor.fields.items()}
    motor.update(torque_nm=proposal["torque_nm"].cpu().numpy().copy(), previous=env.motor.previous.cpu().numpy().copy(),
                 voltage=env.motor.voltage.cpu().numpy().copy(), kp=env.motor.kp.cpu().numpy().copy())
    dofs = descriptor["dofs"]
    need((motor["dof_frictionloss"][:, dofs] > 0).all() and (motor["dof_damping"] >= 0).all(), "genuine positive motor friction")
    raw_inputs, manifest = input_snapshot(env)
    static_binding = binding((env.model, env.data))
    inputs_file = retain(root, "prepared-inputs.npz", raw_inputs)
    datas, models = [], []
    for w in range(2):
        native = plant.build_entity().compile()
        need(plant.describe(native) == descriptor, "matched fresh per-world reference plant before motor fields")
        native.dof_frictionloss[:] = motor["dof_frictionloss"][w].astype(np.float64)
        native.dof_damping[:] = motor["dof_damping"][w].astype(np.float64)
        data = mujoco.MjData(native)
        for key in base.UNCHANGED:
            if key == "time": data.time = float(initial[key][w])
            else: getattr(data, key)[:] = initial[key][w].reshape(getattr(data, key).shape).astype(np.float64)
        before = {key: np.asarray(getattr(data, key)).copy() for key in base.UNCHANGED}
        mujoco.mj_forward(native, data)
        need(not data.warning.number.any() and all(np.asarray(getattr(data, key)).tobytes() == before[key].tobytes() for key in base.UNCHANGED), "native forward cannot integrate")
        datas.append(data)
        models.append(native)
    # Freeze proves the retained bytes are still the actual inputs at the call.
    need(input_snapshot(env)[1] == manifest and binding((env.model, env.data)) == static_binding, "complete unchanged measured input binding")
    env._forward()
    need(not env.steps.any() and all(env._view(key).cpu().numpy().tobytes() == initial[key].tobytes() for key in base.UNCHANGED), "motor preparation and forward cannot integrate")
    cpu = {key: np.stack([np.asarray(getattr(d, key), dtype=np.float64).reshape(-1) for d in datas]) for key in base.FIELDS}
    gpu = {key: env._view(key).cpu().numpy().reshape(2, -1).copy() for key in base.FIELDS}
    cpu_active, gpu_active = active_bank(env, datas, models)
    analyzed = analyze_active(cpu_active, gpu_active, dofs, descriptor["floor"], descriptor["feet"])
    return dict(plant=descriptor, initial_qpos=initial["qpos"].tolist(), input_manifest=manifest,
                input_static_binding_sha256=sha256(repr(static_binding).encode()).hexdigest(),
                residuals=base.comparison(cpu, gpu), active_analysis=analyzed,
                counters=dict(cpu=[dict(ne=d.ne, nf=d.nf, nl=d.nl, nefc=d.nefc, ncon=d.ncon, solver_niter=int(d.solver_niter[0])) for d in datas],
                              gpu={key: env._view(key).cpu().tolist() for key in ("ne", "nf", "nl", "nefc", "nacon", "ncollision", "solver_niter")}),
                payloads=[retain(root, "cpu-fields.npz", cpu), retain(root, "gpu-fields.npz", gpu),
                          retain(root, "cpu-active.npz", cpu_active), retain(root, "gpu-active.npz", gpu_active),
                          inputs_file, retain(root, "motor.npz", motor), retain(root, "bam-source.npz", bam_source)])


def validate_child(value):
    need(set(value) == {"specification", "gpu_uuid", "name", "capability", "torch_cuda", "warp_arch", "warp_precompiled_headers",
         "plant", "initial_qpos", "input_manifest", "input_static_binding_sha256", "residuals", "active_analysis", "counters", "payloads"}, "closed contact receipt")
    need(value["specification"] == specification() and all(x is False for x in value["specification"]["flags"].values())
         and value["gpu_uuid"] == base.host.GPU and value["name"] == base.host.NAME and value["capability"] == [8, 9]
         and value["torch_cuda"] == "12.8" and value["warp_arch"] == 89 and value["warp_precompiled_headers"] is False, "actual Ada contact scope")
    need([x["file"] for x in value["payloads"]] == list(PAYLOADS) and len(value["input_manifest"]) > 400, "complete retained contact files and input arrays")
    need(value["counters"]["gpu"]["nf"] == [14, 14] and all(row["nf"] == 14 for row in value["counters"]["cpu"]), "both motor-friction fixtures covered")
    need(set(value["residuals"]) == set(base.FIELDS) and value["active_analysis"]["solver_qualified"] is False
         and value["active_analysis"]["physical_contact_identity_established"] is False, "unqualified complete evidence")
    for key in base.UNCHANGED:
        row = value["residuals"][key]
        need(row["max_abs"] == row["rms"] == 0. and row["float32_bit_mismatches"] == 0, "unchanged paired state bytes")
    import math
    import re
    need(type(value["initial_qpos"]) is list and len(value["initial_qpos"]) == 2
         and all(type(row) is list and len(row) == 21 and all(type(x) is float and math.isfinite(x) for x in row)
                 for row in value["initial_qpos"]), "complete fixture positions")
    for name, row in value["input_manifest"].items():
        need(name.startswith(("/data/", "/model/")) and type(row) is dict
             and set(row) == {"dtype", "shape", "strides", "bytes", "sha256"}
             and type(row["bytes"]) is int and 0 <= row["bytes"] <= PAYLOAD_MAX
             and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "closed input manifest row")
        validate_input_layout(row)


def receive_payloads(root, value):
    import numpy as np
    validate_child(value)
    banks = {}
    for item in value["payloads"]:
        path = root / item["file"]
        need(not path.is_symlink() and path.stat().st_size == item["bytes"] <= PAYLOAD_MAX, "plain bounded payload")
        raw = path.read_bytes()
        need(sha256(raw).hexdigest() == item["sha256"], "authenticated payload bytes")
        with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
            need(len(bank.files) == len(set(bank.files)), "no duplicate NPZ fields")
            banks[item["file"]] = {key: bank[key].copy() for key in bank.files}
    need(base.comparison(banks["cpu-fields.npz"], banks["gpu-fields.npz"]) == value["residuals"], "every state/dynamic residual recomputed")
    p = value["plant"]
    need(analyze_active(banks["cpu-active.npz"], banks["gpu-active.npz"], p["dofs"], p["floor"], p["feet"]) == value["active_analysis"], "complete active structure recomputed")
    counters = value["counters"]
    for w in range(2):
        for side in ("cpu", "gpu"):
            bank = banks[side + "-active.npz"]
            counts = counters[side][w] if side == "cpu" else {key: counters[side][key][w] for key in ("ne", "nf", "nl", "nefc", "solver_niter")}
            need(counts["ne"] == counts["nl"] == 0 and counts["nf"] == 14
                 and counts["nefc"] == len(bank[f"rows/{w}/type"])
                 and type(counts["solver_niter"]) is int and 0 <= counts["solver_niter"] <= 100, "actual row/counter binding")
            if side == "cpu": need(counts["ncon"] == int((bank["contacts/worldid"] == w).sum()), "native contact counter binding")
    need(counters["gpu"]["nacon"] == [len(banks["gpu-active.npz"]["contacts/worldid"])]
         and len(counters["gpu"]["ncollision"]) == 1 and type(counters["gpu"]["ncollision"][0]) is int
         and 0 <= counters["gpu"]["ncollision"][0] <= 128, "shared contact/collision counters")
    inputs = banks["prepared-inputs.npz"]
    need(set(inputs) == set(value["input_manifest"]), "complete retained model/data inputs")
    for key, row in value["input_manifest"].items():
        raw = inputs[key]
        need(raw.dtype == np.uint8 and raw.ndim == 1 and raw.nbytes == row["bytes"]
             and sha256(raw.tobytes()).hexdigest() == row["sha256"], "every prepared input array verified")
    motor = banks["motor.npz"]
    need(set(motor) == {"dof_frictionloss", "dof_damping", "torque_nm", "previous", "voltage", "kp"}
         and all(x.dtype == np.float32 and np.isfinite(x).all() for x in motor.values()), "finite complete motor preparation")
    need(motor["dof_frictionloss"].shape == motor["dof_damping"].shape == (2, 20)
         and motor["torque_nm"].shape == motor["previous"].shape == (2, 14)
         and motor["voltage"].shape == motor["kp"].shape == (2, 1)
         and not motor["torque_nm"].any() and not motor["previous"].any()
         and (motor["voltage"] == 7.5).all() and (motor["kp"] == 200.).all()
         and (motor["dof_frictionloss"][:, p["dofs"]] > 0).all() and (motor["dof_damping"] >= 0).all(), "actual nominal motor parameters")
    for key in ("dof_frictionloss", "dof_damping"):
        need(inputs["/model/" + key].tobytes() == motor[key].tobytes(), "actual prepared model motor fields")
    q = banks["gpu-fields.npz"]["qpos"]
    expected = np.asarray(p["initial_qpos"], dtype=np.float32)
    expect_contact = expected.copy(); expect_contact[2] += np.float32(ROOT_OFFSETS[1])
    need(np.array_equal(q[0], expected) and np.array_equal(q[1], expect_contact)
         and np.array_equal(q, np.asarray(value["initial_qpos"], dtype=np.float32))
         and inputs["/data/qpos"].tobytes() == q.tobytes(), "declared root-offset initial states")
    source = banks["bam-source.npz"]
    need(set(source) == {"qpos", "qvel", "qfrc_bias", "qfrc_constraint", "qfrc_actuator", "nefc", "nf",
                         "previous", "voltage", "kp", "own_friction", "position_target"}
         and all(np.isfinite(x).all() for x in source.values()) and np.array_equal(source["qpos"], q)
         and not source["previous"].any() and not source["own_friction"].any() and not source["nf"].any()
         and source["nefc"].shape == (2,) and source["nefc"][0] == 0 and source["nefc"][1] > 0
         and not source["qfrc_constraint"][0].any() and source["qfrc_constraint"][1].any()
         and np.array_equal(source["position_target"], q[:, p["qids"]]), "fresh zero-error BAM source fixture")
    source_shapes = dict(qpos=(2, 21), qvel=(2, 20), qfrc_bias=(2, 20), qfrc_constraint=(2, 20), qfrc_actuator=(2, 20),
                         nefc=(2,), nf=(2,), previous=(2, 14), voltage=(2, 1), kp=(2, 1), own_friction=(2, 20), position_target=(2, 14))
    for key, shape in source_shapes.items():
        need(source[key].shape == shape and source[key].dtype == (np.int32 if key in ("nefc", "nf") else np.float32), "complete BAM source field")
        if "/data/" + key in inputs:
            need(source[key].tobytes() == inputs["/data/" + key].tobytes(), "BAM source equals actual prepared data: " + key)
    need(np.array_equal(source["voltage"], motor["voltage"]) and np.array_equal(source["kp"], motor["kp"])
         and np.array_equal(source["previous"], motor["previous"]), "BAM source/proposal parameter continuity")
    need(not banks["gpu-fields.npz"]["qfrc_applied"].any() and not banks["gpu-fields.npz"]["xfrc_applied"].any()
         and inputs["/data/qfrc_applied"].tobytes() == banks["gpu-fields.npz"]["qfrc_applied"].tobytes()
         and inputs["/data/xfrc_applied"].tobytes() == banks["gpu-fields.npz"]["xfrc_applied"].tobytes(), "no external assistance in retained inputs")
    return dict(payloads_verified=True, residuals_recomputed=True, active_structure_recomputed=True, flags=FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    if args.child:
        need(args.lease_fd is not None and args.lease_fd >= 3, "inherited lease")
        base.child(args.source, args.lease_fd, probe=sys.modules[__name__])
    else:
        need(args.lease_fd is None, "owner acquires lease")
        base.supervise(args.source, probe=sys.modules[__name__])


if __name__ == "__main__": main()
