"""Two native-only saved-state forward captures; diagnostic, never admission.

Current full MJB fences and public numeric Model/Option arrays are retained.
Neither output equality nor a current MJB authenticates a historical full model.
Warp is imported by the builder, but runtime array/launch paths are forbidden.
"""
import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import sys
import zipfile

from mjlab_microduck import ada_collision_replay as replay

prior, contact, base = replay.prior, replay.p, replay.p.base
host, need = base.host, replay.need
PROTOCOL = "microduck-current-native-constraint-capture-v1"
DECISION = "current-native-input-capture-not-historical-model-identity-or-admission"
CONTACT = replay.CONTACT + ("elem", "exclude", "mu", "H")
EFC = ("type", "id", "J_rownnz", "J_rowadr", "J_rowsuper", "J_colind", "J", "pos",
       "margin", "frictionloss", "diagA", "KBIP", "D", "R", "vel", "aref", "b", "state", "force")
COUNTERS = ("ncon", "ne", "nf", "nl", "nefc", "nJ", "nA", "nisland")
DATA = ("solver_niter", "solver_nnz", "warning/number", "warning/lastinfo", "counters")
SOLVER = ("improvement", "gradient", "lineslope", "nactive", "nchange", "neval", "nupdate")
STATE = base.UNCHANGED
META = dict(public_native_forward_calls=2, public_contact_force_calls_max=256,
    integration_steps=0, optimizer_steps=0, warp_runtime_array_calls=0, warp_launch_calls=0,
    mjlab_entity_initialize_calls=0, mjlab_bam_initialize_calls=0, mjlab_bam_compute_calls=0,
    full_historical_model_identity_established=False, physical_contact_identity_established=False,
    solver_qualified=False, simulator_qualified=False, training_authorized=False,
    physical_motion_authorized=False, flags=host.FLAGS)
WARP_ARRAY_PATHS = ("_init_from_data", "_init_from_ptr", "_init_new")
WARP_FUNCTIONS = ("launch", "launch_tiled", "capture_launch", "load_module")
NATIVE_FORBIDDEN = ("mj_step", "mj_step1", "mj_step2", "mj_forwardSkip", "mj_inverse",
    "mj_kinematics", "mj_collision", "mj_makeConstraint", "mj_projectConstraint",
    "mj_fwdPosition", "mj_fwdVelocity", "mj_fwdActuation", "mj_fwdAcceleration", "mj_fwdConstraint")
SCHEMA_PATH = Path(__file__).with_name("ada_native_constraint_schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_bytes())
CALLBACKS = ("get_mjcb_act_bias", "get_mjcb_act_dyn", "get_mjcb_act_gain", "get_mjcb_contactfilter",
    "get_mjcb_control", "get_mjcb_passive", "get_mjcb_sensor", "get_mjcb_time")
BANK_RAW_MAX = 96 * 1024**2
LEAF_MAX = 64 * 1024**2
MJB_MAX = 128 * 1024**2
WORLD_SPECIFIC = ("model/dof_frictionloss", "model/dof_damping")
NATIVE_FILES = {
    "mujoco/_functions.cpython-312-x86_64-linux-gnu.so": "05d8fcc40281e5278b37e8a5acaea736c5bac83ce436b3255ca75fca64184eef",
    "mujoco/include/mujoco/mjdata.h": "e2c7ac8a6011372eb1df4e9a53d63f8ddbc3fb940fd68d906480fe85ad0931d7",
    "mujoco/include/mujoco/mjmodel.h": "bcd51b20cb29b6aac7c8b9e1cf348f569b7fc2239f9226b844737c73511d90c8",
    "mujoco/include/mujoco/mjtype.h": "ec580ce2a4ef0c1f6a3e68b3c5b5eeaf61d03413827453e2d0a87c3bd92d16e4",
    "mujoco/include/mujoco/mujoco.h": "5f20d0b8f42ccd0b73eeb19ac7bdb6833093b47d9cd48c8b170adb000819b7f9",
    "mujoco/libmujoco.so.3.10.0": "872b2954e9760b1df122e3c807a907ff806e1980099b00e15789ae67e08b75d5",
}
CONTACT_TAILS = dict(dist=(), pos=(3,), frame=(3, 3), includemargin=(), friction=(5,), solref=(2,),
    solreffriction=(2,), solimp=(5,), dim=(), geom=(2,), flex=(2,), vert=(2,), efc_address=(),
    elem=(2,), exclude=(), mu=(), H=(6, 6))
CONTACT_INTS = ("dim", "geom", "flex", "vert", "efc_address", "elem", "exclude")


def storage_contract():
    names = sorted(k for k, v in SCHEMA.items() if v["kind"] in ("array", "bytes") and k not in WORLD_SPECIFIC)
    return dict(mode="exact-shared-static-model-bytes-v1", base_world=0, logical_worlds=2,
        world1_stored_numeric_paths=list(WORLD_SPECIFIC), shared_numeric_field_count=len(names),
        shared_numeric_paths_sha256=sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest())


def pack_storage(arrays):
    """Store identical model fields once; original two MJBs remain distinct."""
    numeric = {k for k, v in SCHEMA.items() if v["kind"] in ("array", "bytes")}
    need(all({k.removeprefix(f"model/{w}/") for k in arrays if k.startswith(f"model/{w}/")} == numeric
             for w in range(2)), "complete two-world model before storage sharing")
    shared = numeric - set(WORLD_SPECIFIC)
    for k in shared:
        a, b = arrays["model/0/" + k], arrays["model/1/" + k]
        need(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), "exact current shared storage: " + k)
    return {k: v for k, v in arrays.items() if k not in {"model/1/" + n for n in shared}}, storage_contract()


def expand_storage(stored, contract):
    """Logical decoding only, not a new native Model or physical reconstruction."""
    need(contract == storage_contract(), "fixed exact shared-storage contract")
    need({k.removeprefix("model/1/") for k in stored if k.startswith("model/1/")} == set(WORLD_SPECIFIC), "only world-specific numeric fields stored for world1")
    result = dict(stored)
    for k, row in SCHEMA.items():
        if row["kind"] in ("array", "bytes") and k not in WORLD_SPECIFIC:
            need("model/0/" + k in stored, "complete current shared numeric model storage")
            result["model/1/" + k] = stored["model/0/" + k]
    return result


def runtime_targets():
    """Spec/BAM parameter constructors allowed, mjlab actuator runtime forbidden."""
    from mjlab.entity import Entity
    from bam.mjlab import BamActuator
    from mjlab_microduck.actuator import FrictionDRBamActuator, BacklashEncoderBamActuator
    return ((Entity, "initialize"), (BamActuator, "initialize"), (BamActuator, "compute"),
            (FrictionDRBamActuator, "initialize"), (BacklashEncoderBamActuator, "initialize"))


def compare(actual, expected):
    """Exact dtype, shape and bytes; mismatch is retained, never tolerated."""
    need(set(actual) == set(expected), "complete historical comparison inventory")
    return {k: dict(dtype_equal=actual[k].dtype == expected[k].dtype,
                    shape_equal=actual[k].shape == expected[k].shape,
                    bytes_equal=actual[k].tobytes() == expected[k].tobytes()) for k in sorted(actual)}


def manifest(arrays):
    import numpy as np
    need(all(isinstance(a, np.ndarray) and a.dtype.kind in "biuf"
             for a in arrays.values()), "raw nonobject numeric arrays; nonfinite counts retained")
    return {k: dict(dtype=str(a.dtype), shape=list(a.shape), bytes=a.nbytes,
                    nonfinite_count=int(np.count_nonzero(~np.isfinite(a))),
                    sha256=sha256(a.tobytes()).hexdigest()) for k, a in sorted(arrays.items())}


def validate_model(arrays, meta, schema=SCHEMA):
    import numpy as np
    need(set(arrays) == {k for k, v in schema.items() if v["kind"] in ("array", "bytes")}
         and set(meta) == {"scalars", "exclusions"}
         and set(meta["scalars"]) == {k for k, v in schema.items() if v["kind"] in ("int", "float", "bool", "enum")}
         and meta["exclusions"] == [k + (":method" if v["kind"] == "method" else ":nested-serialized-in-MJB")
             for k, v in sorted(schema.items()) if v["kind"] in ("method", "nested")], "complete frozen627-field public Model/Option schema")
    for k, a in arrays.items():
        row = schema[k]
        need(a.dtype == np.dtype("uint8" if row["kind"] == "bytes" else row["dtype"])
             and (a.ndim == 1 if row["kind"] == "bytes" else True), "frozen native public field dtype")
        need(not np.isnan(a).any(), "model NaN refused; raw infinite sentinels retained")
    for k, row in meta["scalars"].items():
        need(set(row) == {"type", "value"} and type(row["value"]).__name__ == schema[k]["kind"]
             and np.isfinite(row["value"]), "frozen finite model scalar kind")


def summarize(arrays, report, banks, model_static=None):
    """Recompute core layout, own-slot row linkage and historical equality."""
    import numpy as np
    expected = {"fields/" + k for k in base.FIELDS}
    expected |= {f"contact/{w}/{k}" for w in range(2) for k in CONTACT}
    expected |= {f"efc/{w}/{k}" for w in range(2) for k in EFC}
    expected |= {f"data/{w}/{k}" for w in range(2) for k in DATA}
    expected |= {f"solver/{w}/{k}" for w in range(2) for k in SOLVER}
    expected |= {"active/" + k for k in banks["cpu-active.npz"]}
    numeric_model = {k for k in arrays if k.startswith("model/")}
    need(set(arrays) == expected | numeric_model and numeric_model, "complete core plus current numeric model inventory")
    names = [{k.removeprefix(f"model/{w}/") for k in numeric_model if k.startswith(f"model/{w}/")} for w in range(2)]
    need(names[0] == names[1] and all(k.startswith(("model/", "option/")) for k in names[0]), "same two current public model inventories")
    if model_static is not None:
        need(len(model_static) == 2 and model_static[0] == model_static[1], "same full current model scalar/options/exclusion inventory")
        for w in range(2): validate_model({k: arrays[f"model/{w}/" + k] for k in names[w]}, model_static[w])
        need(all(arrays["model/0/" + k].tobytes() == arrays["model/1/" + k].tobytes()
                 for k in names[0] if k not in ("model/dof_frictionloss", "model/dof_damping")), "all nonmotor public model arrays equal across worlds")
    for k in base.FIELDS:
        need(arrays["fields/" + k].dtype == np.float64 and arrays["fields/" + k].shape == (2, base.WIDTHS[k]), "complete native float64 fields")
    counts = []
    for w in range(2):
        v = arrays[f"data/{w}/counters"]
        need(v.dtype == np.int64 and v.shape == (len(COUNTERS),), "exact counter layout")
        row = dict(zip(COUNTERS, map(int, v))); counts.append(row)
        ncon, nefc = row["ncon"], row["nefc"]
        need(0 <= ncon <= 128 and 0 <= nefc <= 512 and row["nJ"] == nefc * 20, "bounded current dense native layout; not historical count equality")
        for k in EFC:
            value = arrays[f"efc/{w}/{k}"]
            width = 20 if k in ("J", "J_colind") else 4 if k == "KBIP" else 1
            shape = (nefc, 4) if k == "KBIP" else (nefc * width,)
            need(value.shape == shape and value.dtype == (np.int32 if k in
                 ("type", "id", "J_rownnz", "J_rowadr", "J_rowsuper", "J_colind", "state") else np.float64), "complete native EFC layout: " + k)
        for k in CONTACT:
            value = arrays[f"contact/{w}/{k}"]
            need(value.shape == (ncon,) + CONTACT_TAILS[k] and value.dtype == (np.int32 if k in CONTACT_INTS else np.float64), "every current native contact field layout")
        for k in DATA[:-1]:
            value = arrays[f"data/{w}/{k}"]
            need(value.shape == ((8,) if k.startswith("warning/") else (20,)) and value.dtype == np.int32, "all solver/warning counter layouts")
        for k in SOLVER:
            value = arrays[f"solver/{w}/{k}"]
            need(value.shape == (20, 200) and value.dtype == (np.int32 if k.startswith("n") else np.float64), "all4000 native solver statistic slots")
        slots = arrays["active/contacts/slot"][arrays["active/contacts/worldid"] == w]
        need(slots.tolist() == list(range(ncon)), "own native slots, not cross-backend identity")
        for k in contact.ROW_FIELDS:
            need(arrays[f"active/rows/{w}/{k}"].tobytes() == arrays[f"efc/{w}/{k}"].tobytes(), "active rows bound to current full EFC")
        selected = arrays["active/contacts/worldid"] == w
        for k in ("geom", "dim", "dist", "pos", "frame", "friction", "efc_address"):
            need(arrays["active/contacts/" + k][selected].tobytes() == arrays[f"contact/{w}/{k}"].tobytes(), "active contacts bound to full current candidate table")
        excluded = arrays[f"contact/{w}/efc_address"] < 0
        need(not arrays["active/contacts/force"][selected][excluded].any(), "zero marker for uninstantiated force")
        for slot, address in enumerate(arrays[f"contact/{w}/efc_address"]):
            if address < 0: continue
            need(14 <= address <= nefc - 4 and
                 (arrays[f"efc/{w}/type"][address:address + 4] == 6).all() and
                 (arrays[f"efc/{w}/id"][address:address + 4] == slot).all(), "own contact row type/id/address")
        for name in ("dof_frictionloss", "dof_damping"):
            need(arrays[f"model/{w}/model/{name}"].tobytes() == banks["motor.npz"][name][w].astype(np.float64).tobytes(), "actual restored motor bytes")
    fields = compare({k: arrays["fields/" + k] for k in base.FIELDS}, banks["cpu-fields.npz"])
    active = compare({k.removeprefix("active/"): v for k, v in arrays.items() if k.startswith("active/")}, banks["cpu-active.npz"])
    historical = ["ncon", "ne", "nf", "nl", "nefc", "solver_niter"]
    counters_equal = all((int(arrays[f"data/{w}/solver_niter"][0]) if k == "solver_niter" else counts[w][k])
        == report["child"]["counters"]["cpu"][w][k] for w in range(2) for k in historical)
    states = replay.state_inputs(report, banks)
    unchanged = all(arrays["fields/" + k].tobytes() == states[k].astype(np.float64).tobytes() for k in STATE)
    warnings = [int(arrays[f"data/{w}/warning/number"].sum()) for w in range(2)]
    computed_finite = all(np.isfinite(a).all() for k, a in arrays.items() if not k.startswith("model/"))
    return dict(counters=counts, current_fields_vs_historical=fields, current_active_vs_historical=active,
        seven_states_unchanged=unchanged, warning_counts=warnings, computed_arrays_finite=bool(computed_finite),
        historical_counter_fields=historical, current_counters_vs_historical=counters_equal,
        current_public_model_numeric_fields=sorted(names[0]),
        historical_outputs_equal=all(all(row.values()) for family in (fields, active) for row in family.values()) and counters_equal)


def model_snapshot(model, schema=SCHEMA):
    """All top-level public numeric arrays/scalars, Option fields and full MJB.

    stat/vis are explicitly not reflected; full MJB includes serialized fields.
    Public named-accessor methods are not called. Unknown public types fail shut.
    """
    import numpy as np
    arrays, scalar, excluded = {}, {}, []
    for prefix, obj in (("model", model), ("option", model.opt)):
        for name in sorted(n for n in dir(obj) if not n.startswith("_")):
            value = getattr(obj, name); key = prefix + "/" + name
            if callable(value): excluded.append(key + ":method"); continue
            if prefix == "model" and name in ("opt", "stat", "vis"):
                excluded.append(key + ":nested-serialized-in-MJB"); continue
            if isinstance(value, np.ndarray): arrays[key] = value.copy()
            elif isinstance(value, bytes): arrays[key] = np.frombuffer(value, np.uint8).copy()
            elif isinstance(value, (bool, int, float, np.generic)):
                scalar[key] = dict(type=type(value).__name__, value=value.item() if isinstance(value, np.generic) else value)
            elif type(value).__module__ == "mujoco._enums":
                scalar[key] = dict(type=type(value).__name__, value=int(value))
            else: raise ValueError("unsupported public model type: " + key + ":" + type(value).__name__)
    validate_model(arrays, dict(scalars=scalar, exclusions=excluded), schema)
    json.dumps(scalar, allow_nan=False)
    return arrays, dict(scalars=scalar, exclusions=excluded)


def decoded_contacts(models, datas):
    """Decode included contacts only; excluded contacts retain zero force marker."""
    import numpy as np
    import mujoco
    rows = {k: [] for k in contact.CONTACT_FIELDS}
    for w, (model, data) in enumerate(zip(models, datas)):
        for slot, c in enumerate(data.contact):
            force = np.zeros(6, np.float64)
            if c.efc_address >= 0: mujoco.mj_contactForce(model, data, slot, force)
            values = dict(worldid=w, slot=slot, geom=c.geom.copy(), dim=int(c.dim), dist=float(c.dist),
                pos=c.pos.copy(), frame=c.frame.reshape(3, 3).copy(), friction=c.friction.copy(),
                force=force, efc_address=int(c.efc_address))
            for k in rows: rows[k].append(values[k])
    tails = dict(geom=(2,), pos=(3,), frame=(3, 3), friction=(5,), force=(6,))
    return {k: np.asarray(v, np.int32 if k in ("worldid", "slot", "geom", "dim", "efc_address") else np.float64)
        .reshape((-1,) + tails.get(k, ())) for k, v in rows.items()}


@contextmanager
def forbidden_runtime(wp, context, mujoco, extra_factory=None):
    """Process-local fail-closed guards, installed before dynamic plant imports.

    Annotation-only wp.array constructors remain permitted. C internals of the
    two public mj_forward calls are deliberately not patched or counted as zero.
    """
    saved, attempts, calls = [], [], dict(forward=0, contact_force=0)
    def block(label):
        def refused(*args, **kwargs):
            attempts.append(label); raise ValueError("forbidden runtime path: " + label)
        return refused
    def patch(obj, name, replacement):
        need(hasattr(obj, name), "frozen guard target exists: " + name)
        saved.append((obj, name, getattr(obj, name))); setattr(obj, name, replacement)
    try:
        for name in WARP_ARRAY_PATHS: patch(wp.array, name, block("array." + name))
        for obj, prefix in ((wp, "wp"), (context, "context")):
            for name in WARP_FUNCTIONS: patch(obj, name, block(prefix + "." + name))
        patch(context.Module, "load", block("Module.load"))
        for name in NATIVE_FORBIDDEN: patch(mujoco, name, block(name))
        if extra_factory is not None:
            for cls, name in extra_factory(): patch(cls, name, block(cls.__name__ + "." + name))
        for name, key in (("mj_forward", "forward"), ("mj_contactForce", "contact_force")):
            original = getattr(mujoco, name)
            def counted(*args, _original=original, _key=key, **kwargs):
                calls[_key] += 1
                need(calls[_key] <= (2 if _key == "forward" else 256), "bounded public native calls")
                return _original(*args, **kwargs)
            patch(mujoco, name, counted)
        yield attempts, calls
    finally:
        for obj, name, original in reversed(saved): setattr(obj, name, original)


def native_provenance():
    """Installed file bytes/RECORD, not proof of loaded binary machine code."""
    need(sys.platform == "linux", "Linux-only native capture; portable receipt has no runtime imports")
    need(distribution("better-actuator-models").version == "1.0.1", "frozen BAM spec-construction version")
    dist = distribution("mujoco"); need(dist.version == "3.10.0", "frozen native version")
    selected = [f for f in dist.files if str(f).startswith("mujoco/") and
        (str(f).endswith(("mjdata.h", "mjmodel.h", "mjtype.h", "mujoco.h")) or
         ("_functions." in str(f) and str(f).endswith(".so")) or
         ("libmujoco.so" in str(f)))]
    need({str(f) for f in selected} == set(NATIVE_FILES), "exact four native headers, extension and shared library")
    result = {}
    for item in selected:
        path = Path(dist.locate_file(item))
        need(path.is_file() and not path.is_symlink() and path.stat().st_size < 16 * 1024**2,
             "plain bounded native installed provenance")
        raw = path.read_bytes(); digest = sha256(raw).digest()
        need(digest.hex() == NATIVE_FILES[str(item)] and item.hash and item.hash.mode == "sha256" and item.size == len(raw)
             and base64.urlsafe_b64encode(digest).decode().rstrip("=") == item.hash.value,
             "installed native bytes match wheel RECORD")
        result[str(item)] = dict(bytes=len(raw), sha256=digest.hex(), record_sha256_matches=True)
    return result


def capture(report, banks):
    import numpy as np
    import mujoco
    import torch
    import warp as wp
    from warp._src import context
    need(not torch.cuda.is_initialized(), "no initialized Torch CUDA")
    need(all(d.is_cpu for d in wp.get_devices()), "CPU-only Warp device inventory")
    callbacks = [n for n in dir(mujoco) if n.startswith("get_mjcb_")]
    need(tuple(callbacks) == CALLBACKS and all(getattr(mujoco, n)() is None for n in callbacks), "exact eight absent native global callback hooks")
    def no_executables():
        need(all(not m.execs for m in context.user_modules.values()), "no held Warp ModuleExec")
    no_executables()
    state = replay.state_inputs(report, banks)
    arrays, static, models, datas, bindings, counts, mjbs = {}, [], [], [], [], [], []
    with forbidden_runtime(wp, context, mujoco, runtime_targets) as (attempts, calls):
        # Static actuator/spec construction only: no Entity.initialize or BAM runtime.
        from mjlab_microduck.stance_warp_runtime import build_entity
        for w in range(2):
            model = build_entity().compile(); bindings.append(replay.bind_plant(model, report["child"]["plant"]))
            for name in ("dof_frictionloss", "dof_damping"):
                getattr(model, name)[:] = banks["motor.npz"][name][w].astype(np.float64)
            before, meta = model_snapshot(model)
            size = mujoco.mj_sizeModel(model)
            print("native compiled world", w, "MJB bytes", size, "numeric model bytes", sum(a.nbytes for a in before.values()), flush=True)
            need(0 < size < MJB_MAX, "bounded current MJB: " + str(size))
            mjb = np.empty(size, np.uint8); mujoco.mj_saveModel(model, None, mjb)
            data = mujoco.MjData(model)
            for name, value in state.items():
                if name == "time": data.time = float(value[w, 0])
                else: getattr(data, name)[:] = value[w].reshape(getattr(data, name).shape).astype(np.float64)
            mujoco.mj_forward(model, data)
            after, after_meta = model_snapshot(model)
            after_mjb = np.empty(size, np.uint8); mujoco.mj_saveModel(model, None, after_mjb)
            need(compare(before, after) == {k: dict(dtype_equal=True, shape_equal=True, bytes_equal=True) for k in before}
                 and meta == after_meta and mjb.tobytes() == after_mjb.tobytes(), "full current serialized model and public numeric fences")
            arrays.update({f"model/{w}/" + k: v for k, v in before.items()})
            static.append(meta); mjbs.append(mjb)
            counts.append({k: int(getattr(data, k)) for k in COUNTERS})
            arrays[f"data/{w}/counters"] = np.array([counts[-1][k] for k in COUNTERS], np.int64)
            need(0 <= data.nefc <= 512 and 0 <= data.ncon <= 128 and data.nJ == data.nefc * model.nv, "bounded current dense native EFC layout")
            for name in EFC:
                a = np.asarray(getattr(data, "efc_" + name)).copy()
                arrays[f"efc/{w}/{name}"] = a.reshape((data.nefc, 4) if name == "KBIP" else (-1,))
            for name in ("solver_niter", "solver_nnz", "warning/number", "warning/lastinfo"):
                obj, attr = (data.warning, name.split("/")[1]) if "/" in name else (data, name)
                arrays[f"data/{w}/{name}"] = np.asarray(getattr(obj, attr)).copy()
            for name in SOLVER:
                arrays[f"solver/{w}/{name}"] = np.asarray(getattr(data.solver, name)).copy().reshape(20, 200)
            models.append(model); datas.append(data)
        arrays.update({"fields/" + k: np.stack([np.asarray(getattr(d, k), dtype=np.float64).reshape(-1) for d in datas]) for k in base.FIELDS})
        arrays.update({"contact/" + k: v.reshape((datas[int(k.split('/')[0])].ncon,) + CONTACT_TAILS[k.split('/')[1]])
                       for k, v in replay.native_candidates(datas).items()})
        active = {f"rows/{w}/{k}": np.asarray(getattr(d, "efc_" + k)).copy().reshape((d.nefc, 20) if k == "J" else (-1,))
                  for w, d in enumerate(datas) for k in contact.ROW_FIELDS}
        active.update({"contacts/" + k: v for k, v in decoded_contacts(models, datas).items()})
        arrays.update({"active/" + k: v for k, v in active.items()})
        no_executables()
        need(not attempts and calls["forward"] == 2 and calls["contact_force"] == sum(int((d.contact.efc_address >= 0).sum()) for d in datas) and not torch.cuda.is_initialized()
             and all(getattr(mujoco, n)() is None for n in callbacks), "unchanged guarded native-only execution")
    result = dict(model_static=static, model_fences_equal=True, plant_bindings=bindings,
        native_callback_hooks=callbacks, guarded_paths_attempted=[], held_warp_executables=0,
        public_native_calls=calls, static_bam_parameter_construction_allowed=True,
        excluded_contact_force_representation="zero marker; no mj_contactForce call for efc_address<0",
        **summarize(arrays, report, banks, static))
    manifest(arrays)
    return result, arrays, mjbs


def receive(output, source, input_root):
    """Pure portable receipt: no native Model/Data creation or physics calls."""
    import numpy as np
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024, "plain bounded capture JSON")
    value = json.loads(output.read_bytes()); report, banks = prior.authenticated_banks(input_root)
    need(value["protocol"] == PROTOCOL and value["decision"] == DECISION and value["source"]["commit"] == source
         and value["meta"] == META and value["predecessor_report_sha256"] == prior.REPORT_SHA256
         and value["predecessor_files"] == report["files"], "same nonqualifying capture contract")
    module = host.read("git", "show", source + ":src/mjlab_microduck/ada_native_constraint_capture.py", binary=True)
    need(Path(__file__).read_bytes() == module and value["module_sha256"] == sha256(module).hexdigest(), "loaded receiver is the committed capture module")
    schema = host.read("git", "show", source + ":src/mjlab_microduck/ada_native_constraint_schema.json", binary=True)
    need(value["schema_sha256"] == sha256(schema).hexdigest() == sha256(SCHEMA_PATH.read_bytes()).hexdigest(), "complete committed627-field schema")
    payload = value["payload"]; path = output.with_suffix(".npz")
    need(path.is_file() and not path.is_symlink() and payload["file"] == path.name
         and path.stat().st_size == payload["bytes"] < LEAF_MAX, "plain bounded full capture bank")
    raw = path.read_bytes(); need(sha256(raw).hexdigest() == payload["sha256"], "capture payload hash")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        need(len(entries) <= 2048 and sum(e.file_size for e in entries) < BANK_RAW_MAX + 1024**2
             and all(e.file_size < BANK_RAW_MAX and e.compress_type == zipfile.ZIP_DEFLATED for e in entries), "bounded complete compressed bank before decoding")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)), "unique capture inventory")
        arrays = {k: bank[k].copy() for k in bank.files}
    need(manifest(arrays) == payload["arrays"], "all complete numeric capture bytes/layouts")
    need(sum(a.nbytes for a in arrays.values()) < BANK_RAW_MAX, "bounded complete uncompressed numeric bytes")
    arrays = expand_storage(arrays, value["array_storage"])
    result = value["capture"]
    summary = summarize(arrays, report, banks, result["model_static"])
    need(all(result[k] == v for k, v in summary.items()), "recomputed complete capture summary")
    need(result["public_native_calls"] == dict(forward=2, contact_force=sum(int((arrays[f"contact/{w}/efc_address"] >= 0).sum()) for w in range(2))), "actual bounded public forward/force counts")
    for w, row in enumerate(value["mjb"]):
        path = output.parent / row["file"]
        need(path.name == output.stem + f"-world{w}.mjb" and path.is_file() and not path.is_symlink()
             and path.stat().st_size == row["bytes"] < MJB_MAX
             and sha256(path.read_bytes()).hexdigest() == row["sha256"], "current full serialized MJB bytes")
    need(len(value["mjb"]) == 2 and result["model_fences_equal"] is True and not result["guarded_paths_attempted"]
         and result["held_warp_executables"] == 0, "native-only fences; not historical identity")
    need(result["native_callback_hooks"] == list(CALLBACKS)
         and result["static_bam_parameter_construction_allowed"] is True
         and value["protected_services"] == {scope + ":" + service: "inactive" for scope in ("system", "user") for service in host.SERVICES}, "native callback and protected service contract")
    need(set(value["native_installed_provenance"]) == set(NATIVE_FILES)
         and all(row["sha256"] == NATIVE_FILES[k] and row["record_sha256_matches"] is True
                 and type(row["bytes"]) is int and 0 < row["bytes"] < 16 * 1024**2
                 for k, row in value["native_installed_provenance"].items()), "exact predeclared Linux native provenance")
    return value, arrays


def retain(path, arrays):
    import numpy as np
    need(sum(a.nbytes for a in arrays.values()) < BANK_RAW_MAX, "bounded raw capture bank")
    rows = manifest(arrays)
    with path.open("xb") as stream: np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    raw = path.read_bytes(); need(len(raw) < LEAF_MAX, "bounded compressed capture leaf")
    return dict(file=path.name, bytes=len(raw), sha256=sha256(raw).hexdigest(), arrays=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output"): parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source", required=True); args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "CPU-hidden capture before closeout reserve")
    source = host.source_check(args.source)
    need(Path(__file__).read_bytes() == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_native_constraint_capture.py", binary=True), "exact committed capture bytes")
    need(SCHEMA_PATH.read_bytes() == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_native_constraint_schema.json", binary=True), "committed frozen public field schema")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
         and not any(args.output.with_suffix(s).exists() for s in (".json", ".npz"))
         and not any((args.output.parent / (args.output.stem + f"-world{w}.mjb")).exists() for w in range(2)), "fresh canonical capture paths")
    services = host.service_snapshot(); foreign = host.foreign_processes(); telemetry = host.telemetry()
    report, banks = prior.authenticated_banks(args.input); provenance = native_provenance()
    result, arrays, mjbs = capture(report, banks)
    stored, storage = pack_storage(arrays)
    payload = retain(args.output.with_suffix(".npz"), stored)
    files = []
    for w, mjb in enumerate(mjbs):
        path = args.output.parent / (args.output.stem + f"-world{w}.mjb")
        with path.open("xb") as stream: stream.write(mjb.tobytes()); stream.flush(); os.fsync(stream.fileno())
        files.append(dict(file=path.name, bytes=mjb.nbytes, sha256=sha256(mjb.tobytes()).hexdigest()))
    need(services == host.service_snapshot() and foreign == host.foreign_processes(), "preserved foreign workloads and protected services")
    host.source_check(args.source)
    value = dict(protocol=PROTOCOL, decision=DECISION, source=source, module_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        schema_sha256=sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        predecessor_report_sha256=prior.REPORT_SHA256, predecessor_files=report["files"],
        native_installed_provenance=provenance, meta=META, capture=result, payload=payload, mjb=files, array_storage=storage,
        protected_services=services, foreign_processes=foreign, telemetry_before=telemetry, telemetry_after=host.telemetry())
    host.write_json(args.output, value); receive(args.output, args.source, args.input)
    print(DECISION, "historical_outputs_equal=" + str(result["historical_outputs_equal"]))


if __name__ == "__main__": main()
