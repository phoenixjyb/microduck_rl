"""Guarded saved prepared-pose CPU response; never simulator admission.

Artificial prepared-pose intervention, not integrated forward or admission.
Importing this module initializes no simulator runtime.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import re
import shlex

from mjlab_microduck import ada_saved_pose_collision as saved
from mjlab_microduck import ada_solver_stage_audit as source

need = saved.need
POSES = saved.p.KINEMATIC + ("site_xpos", "site_xmat")
STAGES = ("before_collision", "after_collision", "after_construction", "before_solve", "after_solve")
CALLBACKS = ("passive", "control", "act_dyn", "act_gain", "act_bias", "sensor", "contactfilter")
SOLVER_OUTPUTS = {"/data/qacc", "/data/qfrc_constraint", "/data/solver_niter",
                  "/data/efc/Ma", "/data/efc/force", "/data/efc/state"}
PROTOCOL = "microduck-eleven-prepared-pose-cpu-response-v2"
DECISION = "saved-prepared-pose-cpu-response-diagnostic-not-admission"
NATIVE_ROOT = Path("/home/converge/work/microduck_rl-athletics-obstacle-curriculum")
MACHINE = "0c79e415429b4933a400159bfa79a34d"
CONTRACT = dict(allocation_native_kinematics_calls=1, fixture_native_kinematics_calls=0,
    fixture_warp_kinematics_calls=0, cpu_collision_calls=1, new_constraint_calls=1,
    new_solver_calls=1, new_integration_steps=0, acceleration_sensor_calls=0,
    force_decoder_kernel_calls=0,
    interpretation="Eleven prepared-pose intervention, CPU-generated manifold and descriptive CPU response",
    measured_gpu_collision_pose_complete=False, same_manifold_solver_control=False,
    flags=saved.p.p.FLAGS, solver_qualified=False, training_authorized=False, physical_motion_authorized=False)
STATIC_CONVENTION = "Within-process unchanged pointers, shapes, options and scalars; not historical execution identity"
SERVICE_CAPS = dict(Type="exec", RuntimeMaxUSec="3min", MemoryMax="6442450944", CPUQuotaPerSecUSec="2s",
    TasksMax="64", Nice="10", LimitFSIZE="16777216", LimitCORE="0", Restart="no", KillMode="control-group",
    TimeoutStopUSec="10s", RemainAfterExit="yes", ActiveState="active", SubState="running")
SERVICE_ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def topology_guard(native, model, data):
    """Reject unsupported paths before any fixture physics or callbacks."""
    import numpy as np
    need(all(getattr(native, k) == 0 and getattr(model, k) == 0
             for k in ("nflex", "neq", "nmocap", "na", "ntendon")),
         "no flex/equality/mocap/activation/tendon response paths")
    need((native.nq, native.nv, native.nu, native.nsite) == (21, 20, 14, 7)
         and (model.nq, model.nv, model.nu, model.nsite) == (21, 20, 14, 7)
         and not model.is_sparse and model.nacttrnbody == 0,
         "exact dense joint-only Duck response topology")
    callbacks = vars(model.callback)
    need(set(callbacks) == set(CALLBACKS) and all(callbacks[k] is None for k in CALLBACKS),
         "no custom control/passive/actuation/sensor/contact callbacks")
    trn = model.actuator_trntype.numpy()
    need(trn.dtype == np.int32 and trn.shape == (14,) and not trn.any()
         and not np.asarray(native.actuator_trntype).any(), "all fourteen transmissions are joint type0")
    for k, tail in (("act", (0,)), ("eq_active", (0,)), ("mocap_pos", (0, 3)), ("mocap_quat", (0, 4))):
        value = getattr(data, k).numpy()
        need(value.shape == (2,) + tail and value.nbytes == 0, "explicit empty unsupported state: " + k)
    need(data.qpos.device.is_cpu and not data.qpos.device.is_cuda
         and model.opt.run_collision_detection is True and model.opt.graph_conditional is True,
         "CPU-only default collision and solver dispatch")
    return saved.dispatch_guard(native, model, data)


def supplied_poses(data, report, banks):
    """Complete eleven-field prepared bundle, including site poses."""
    import numpy as np
    values = saved.supplied_poses(data, report, banks)
    for name in POSES[-2:]:
        key = "/data/" + name; row = report["child"]["input_manifest"][key]
        array = getattr(data, name); raw = banks["prepared-inputs.npz"][key].tobytes()
        need(str(array.dtype) == row["dtype"] and list(array.shape) == row["shape"]
             and list(array.strides) == row["strides"] and len(raw) == row["bytes"]
             and sha256(raw).hexdigest() == row["sha256"], "complete supplied site pose layout/bytes")
        tail = (3, 3) if name.endswith("mat") else (3,)
        value = np.frombuffer(raw, np.float32).reshape(array.numpy().shape).copy()
        need(value.shape == (2, 7) + tail and np.isfinite(value).all(), "finite complete site pose")
        values[name] = value
    return values


def exact_values(actual, expected, label):
    need(set(actual) == set(expected), "complete " + label + " field inventory")
    for name in expected:
        left, right = actual[name], expected[name]
        need(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(),
             "unchanged exact " + label + ": " + name)


def snapshot_data(data, report):
    """Every actual logical Data array, with byte and expanded NumPy layout.

    Full allocated banks include inactive capacity, explicitly not active rows.
    No hand-selected subset is substituted for a complete pre-solver bank.
    """
    import numpy as np
    actual = saved.p.p.all_arrays(data, "/data")
    expected = {k: v for k, v in report["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(len(expected) == 114 and set(actual) == set(expected), "complete frozen114 Data array inventory")
    arrays, layouts = {}, {}
    for key, array in actual.items():
        value = array.numpy().copy(); row = expected[key]
        need(str(array.dtype) == row["dtype"] and list(array.shape) == row["shape"]
             and list(array.strides) == row["strides"] and value.nbytes == row["bytes"],
             "unchanged complete Data logical layout: " + key)
        need(value.dtype.kind in "fibu" and np.isfinite(value).all(), "finite numeric allocated Data array")
        arrays[key] = np.frombuffer(value.tobytes(), np.uint8).copy()
        layouts[key] = dict(dtype=str(array.dtype), shape=list(array.shape), strides=list(array.strides),
                            numpy_dtype=str(value.dtype), numpy_shape=list(value.shape), bytes=value.nbytes)
    need(sum(v.nbytes for v in arrays.values()) < 2 * 1024**2, "bounded full Data stage bank")
    return arrays, layouts


def candidate_stability(before, after):
    """Only construction addresses may differ; no contact copying/injection."""
    need(set(before) == set(after) == set(saved.CONTACT_TAILS), "complete candidate fields across construction")
    exact_values({k: v for k, v in before.items() if k != "efc_address"},
                 {k: v for k, v in after.items() if k != "efc_address"}, "generated candidates")
    need(after["efc_address"].shape == before["efc_address"].shape
         and after["efc_address"].dtype == before["efc_address"].dtype,
         "same candidate address capacity; values are stage dependent")


def solver_write_fence(before, after):
    """Retain full bank and reject writes outside six frozen-source outputs."""
    need(set(before) == set(after) and SOLVER_OUTPUTS <= set(before), "complete solver-boundary banks")
    exact_values({k: v for k, v in before.items() if k not in SOLVER_OUTPUTS},
                 {k: v for k, v in after.items() if k not in SOLVER_OUTPUTS}, "non-solver-output Data bytes")
    return sorted(k for k in SOLVER_OUTPUTS if before[k].tobytes() != after[k].tobytes())


def staged_calls(model, data, modules, capture):
    """Exact frozen forward stage order minus kinematics; stop after solve.

    Accelerational sensors, integration and ordinary public forward are not
    called. Source guard/pose and full-boundary checks are the caller's duty.
    """
    smooth, forward = modules["smooth"], modules["forward"]
    smooth.com_pos(model, data)
    smooth.camlight(model, data)
    smooth.flex(model, data)
    smooth.tendon(model, data)
    smooth.crb(model, data)
    smooth.tendon_armature(model, data)
    capture("before_collision")
    modules["collision_driver"].collision(model, data)
    capture("after_collision")
    modules["constraint"].make_constraint(model, data)
    smooth.transmission(model, data)
    capture("after_construction")
    data.sensordata.zero_()
    modules["sensor"].sensor_pos(model, data)
    data.energy.zero_()  # Default enableflags0 is guarded; no energy callback.
    forward.fwd_velocity(model, data)
    modules["sensor"].sensor_vel(model, data)
    forward.fwd_actuation(model, data)  # All control/actuation callbacks absent.
    forward.fwd_acceleration(model, data, factorize=True)
    capture("before_solve")
    modules["solver"].solve(model, data)
    capture("after_solve")


def run(report, banks):
    """One CPU response intervention behind the closed native CLI guards."""
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden response primitives")
    import numpy as np
    import warp as wp
    import mujoco_warp as mjwarp
    from mujoco_warp._src import smooth, forward, collision_driver, constraint, sensor, solver
    from mjlab.sim.randomization import expand_model_fields
    from mjlab_microduck.stance_warp_runtime import build_entity
    from mjlab_microduck.stance_forward_graph import binding
    source_audit = source.audit()
    need(all(device.is_cpu for device in wp.get_devices()), "CPU-only enumerated response devices")
    need(not saved.passive_executables(), "fresh process before CPU response kernel loads")
    native = build_entity().compile()
    plant = saved.p.bind_plant(native, report["child"]["plant"])
    for name in ("dof_frictionloss", "dof_damping"):
        getattr(native, name)[:] = banks["motor.npz"][name][0].astype(np.float64)
    with wp.ScopedDevice("cpu"):
        model = mjwarp.put_model(native)
        data = mjwarp.make_data(native, nworld=2, nconmax=128, njmax=512)
        options = topology_guard(native, model, data)
        expand_model_fields(model, 2, ["dof_frictionloss", "dof_damping"])
        for name in ("dof_frictionloss", "dof_damping"):
            wp.copy(getattr(model, name), wp.array(banks["motor.npz"][name], dtype=wp.float32, device="cpu"))
        poses = supplied_poses(data, report, banks); states = saved.p.state_inputs(report, banks)
        states = {k: v.reshape(getattr(data, k).numpy().shape) for k, v in states.items()}
        for name, value in (states | poses).items():
            wp.copy(getattr(data, name), wp.array(value, dtype=getattr(data, name).dtype, device="cpu"))
        wp.synchronize_device("cpu")
        model_hashes = saved.p.model_binding(model, report, banks)
        static_binding = binding((model, data))
        need(all(not getattr(data, k).numpy().any() for k in saved.p.COUNTERS + ("nacon", "ncollision")),
             "no stale collision, constraints or solver before response")
        boundaries, layout, contacts = {}, None, {}

        def capture(stage):
            nonlocal layout
            need(stage == STAGES[len(boundaries)], "one exact ordered response boundary")
            wp.synchronize_device("cpu")
            exact_values({k: getattr(data, k).numpy().copy() for k in POSES}, poses, "eleven prepared poses")
            exact_values({k: getattr(data, k).numpy().copy() for k in states}, states, "seven state inputs")
            need(topology_guard(native, model, data) == options
                 and saved.p.model_binding(model, report, banks) == model_hashes
                 and binding((model, data)) == static_binding, "unchanged full model/static/callback binding at stage")
            raw, current_layout = snapshot_data(data, report)
            need(layout is None or current_layout == layout, "same full Data layout at every boundary")
            layout = current_layout; boundaries[stage] = raw
            if stage != "before_collision":
                candidates = saved.p.warp_candidates(data)
                need(0 < int(data.ncollision.numpy()[0]) <= 256, "bounded generated broadphase")
                if stage == "after_collision":
                    saved.contact_layout(candidates)
                    need(all(not getattr(data, k).numpy().any() for k in saved.p.COUNTERS),
                         "collision stops before construction and solve")
                else:
                    candidate_stability(contacts["after_collision"], candidates)
                    need((data.ne.numpy() == 0).all() and (data.nl.numpy() == 0).all()
                         and (data.nf.numpy() == 14).all()
                         and (candidates["dim"] == 3).all()
                         and np.array_equal(data.nefc.numpy(), 14 + 4 * np.bincount(candidates["worldid"], minlength=2))
                         and (data.nefc.numpy() <= 512).all(),
                         "finite-capacity friction/contact-only constraints")
                    if stage != "after_solve": need(not data.solver_niter.numpy().any(), "no solve before solver boundary")
                    else:
                        need(((data.solver_niter.numpy() >= 0) & (data.solver_niter.numpy() <= native.opt.iterations)).all(),
                             "solver iterations bounded by unchanged plant option")
                        solver_write_fence(boundaries["before_solve"], raw)
                contacts[stage] = candidates

        staged_calls(model, data, dict(smooth=smooth, forward=forward, collision_driver=collision_driver,
                                     constraint=constraint, sensor=sensor, solver=solver), capture)
        need(tuple(boundaries) == STAGES and source.audit() == source_audit, "complete unchanged-source response boundaries")
    return dict(plant=plant, options=options, model_array_sha256=model_hashes,
                data_layout=layout, stage_source_audit=source_audit,
                solver_changed_data_fields=solver_write_fence(boundaries["before_solve"], boundaries["after_solve"]),
                static_binding_sha256=sha256(repr(static_binding).encode()).hexdigest(),
                static_binding_convention=STATIC_CONVENTION, **CONTRACT), {
                    stage + key: value for stage, raw in boundaries.items() for key, value in raw.items()}


def expanded_layout(row):
    """Independently derive component dtype/shape from frozen logical types."""
    import numpy as np
    types = {"<class '" + ns + "." + name + "'>": (dtype, tail)
             for ns, entries in (
                 ("warp._src.types", {"float32": ("float32", ()), "int32": ("int32", ()), "bool": ("bool", ()),
                     "vec2i": ("int32", (2,)), "vec2f": ("float32", (2,)), "vec3f": ("float32", (3,)),
                     "quatf": ("float32", (4,)), "spatial_vectorf": ("float32", (6,)), "mat33f": ("float32", (3, 3))}),
                 ("mujoco_warp._src.types", {"vec5f": ("float32", (5,)), "vec10f": ("float32", (10,))}))
             for name, (dtype, tail) in entries.items()}
    need(row["dtype"] in types, "closed frozen Data component dtype")
    saved.p.p.validate_input_layout(row)
    dtype, tail = types[row["dtype"]]
    shape = list(row["shape"]) + list(tail)
    need(int(np.prod(shape)) * np.dtype(dtype).itemsize == row["bytes"], "independent component byte length")
    return dict(dtype=row["dtype"], shape=row["shape"], strides=row["strides"],
                numpy_dtype=dtype, numpy_shape=shape, bytes=row["bytes"])


def decode_stages(arrays, layout, report, banks):
    """Decode every raw stage byte with independently checked complete layout."""
    import numpy as np
    expected = {k: v for k, v in report["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(len(expected) == 114 and set(layout) == set(expected)
         and set(arrays) == {s + k for s in STAGES for k in expected}, "complete five-by114 stage bank")
    for k, row in expected.items(): need(layout[k] == expanded_layout(row), "independently bound expanded Data layout: " + k)
    result = {}
    for stage in STAGES:
        current = {}
        for k, row in layout.items():
            value = arrays[stage + k]
            need(value.dtype == np.uint8 and value.shape == (row["bytes"],), "complete raw uint8 Data leaf")
            current[k] = np.frombuffer(value.tobytes(), dtype=row["numpy_dtype"]).reshape(row["numpy_shape"]).copy()
            need(np.isfinite(current[k]).all(), "finite full decoded stage Data")
        for name in POSES + saved.p.p.base.UNCHANGED:
            k = "/data/" + name
            need(current[k].tobytes() == banks["prepared-inputs.npz"][k].tobytes(), "actual prepared pose/state bytes at stage: " + name)
        result[stage] = current
    return result


def stage_candidates(data):
    count = data["/data/nacon"]
    need(count.dtype.name == "int32" and count.shape == (1,) and 0 <= int(count[0]) <= 256, "actual bounded shared candidate counter")
    return {k: data["/data/contact/" + k][:int(count[0])].copy() for k in saved.CONTACT_TAILS}


def active_response(data, descriptor):
    """All active rows and diagnostic host decoding of dim3 pyramid outcomes.

    Float32 pair/add order follows frozen support._decode_pyramid. These are
    arithmetic decoded outcomes, not new kernel results or injected forces.
    """
    import numpy as np
    contact = stage_candidates(data); counts = data["/data/nefc"]
    need(counts.shape == (2,) and ((counts >= 14) & (counts <= 512)).all()
         and (contact["dim"] == 3).all(), "bounded friction/rigid dim3 response")
    active = {}
    covered = []
    for w, n in enumerate(map(int, counts)):
        for name in saved.p.p.ROW_FIELDS:
            value = data["/data/efc/" + name][w, :n].copy()
            if name == "J": value = value[:, :20]
            active[f"rows/{w}/{name}"] = value
        types, ids, J = (active[f"rows/{w}/{k}"] for k in ("type", "id", "J"))
        friction = np.flatnonzero(types == 1)
        need(n == 14 + 4 * int((contact["worldid"] == w).sum()) and J.shape == (n, 20)
             and len(friction) == 14 and set(map(int, ids[friction])) == set(descriptor["dofs"]), "complete friction/contact row coverage")
        for r in friction:
            unit = np.zeros(20, np.float32); unit[int(ids[r])] = 1
            need(J[r].tobytes() == unit.tobytes(), "actual friction Jacobian DOF linkage")
        covered.append(set(map(int, friction)))
    force = np.zeros((len(contact["dist"]), 6), np.float32)
    for slot, (world, addr) in enumerate(zip(contact["worldid"], contact["efc_address"])):
        w, first = int(world), int(addr[0])
        need(0 <= w < 2, "bounded generated contact world")
        n = int(counts[w])
        rows = set(range(first, first + 4))
        need(0 <= w < 2 and first >= 14 and first + 4 <= n
             and addr.tolist() == list(range(first, first + 4)) and not covered[w] & rows,
             "complete unambiguous pyramid address group")
        types, ids = active[f"rows/{w}/type"], active[f"rows/{w}/id"]
        need((types[first:first+4] == 6).all() and (ids[first:first+4] == slot).all(), "all four rows linked to generated raw slot")
        covered[w].update(rows)
        pyramid = active[f"rows/{w}/force"]
        for i in range(2):
            a, b = pyramid[first+2*i:first+2*i+2]
            force[slot, 0] = np.float32(force[slot, 0] + np.float32(a + b))
            force[slot, i+1] = np.float32(np.float32(a - b) * contact["friction"][slot, i])
    need(all(covered[w] == set(range(int(counts[w]))) for w in range(2))
         and np.isfinite(force).all(), "every active row and finite decoded force")
    for name in saved.p.p.CONTACT_FIELDS:
        value = np.arange(len(force), dtype=np.int32) if name == "slot" else force if name == "force" else contact[name]
        if name == "efc_address": value = value[:, 0]
        active["contacts/" + name] = value
    return active


def field_delta(historical, current):
    value = saved.metadata.field_difference(historical, current)
    value["current_minus_historical"] = value.pop("ada_minus_cpu")
    return value


def ordered_reconstruction(active, fields):
    """Explicit float64 products/sums in ascending retained row order.

    Do not dispatch a NumPy/BLAS dot: its reduction tree is backend-dependent.
    Every original active row remains included; no tolerance or normalization.
    """
    import numpy as np
    result = []
    for world in range(2):
        J, force, kinds = (active[f"rows/{world}/{k}"] for k in ("J", "force", "type"))
        need(J.shape == (len(force), 20) and kinds.shape == force.shape
             and np.isfinite(J).all() and np.isfinite(force).all(), "finite complete ordered row arithmetic")
        friction, contact, total = [0.] * 20, [0.] * 20, [0.] * 20
        for row in range(len(force)):
            need(int(kinds[row]) in (1, 6), "only guarded friction/contact rows")
            selected = friction if int(kinds[row]) == 1 else contact
            for col in range(20):
                product = float(J[row, col]) * float(force[row])
                selected[col] = selected[col] + product
                total[col] = total[col] + product
        stored = [float(v) for v in fields["qfrc_constraint"][world]]
        need(len(stored) == 20 and np.isfinite([friction, contact, total, stored]).all(), "finite ordered generalized result")
        delta = [a - b for a, b in zip(total, stored)]
        result.append(dict(world=world, friction_generalized_force=friction, contact_generalized_force=contact,
            total_generalized_force=total, stored_constraint_generalized_force=stored,
            reconstructed_minus_stored=delta, max_abs=max(map(abs, delta))))
    return result


def analyze_stages(arrays, layout, report, banks):
    """Recompute boundaries and all descriptive resultants without runtime."""
    import numpy as np
    decoded = decode_stages(arrays, layout, report, banks)
    contacts, counters = {}, {}
    descriptor = report["child"]["plant"]
    for stage, data in decoded.items():
        counts = {k: data["/data/" + k] for k in saved.p.COUNTERS + ("nacon", "ncollision")}
        counters[stage] = {k: v.tolist() for k, v in counts.items()}
        if stage == "before_collision":
            need(all(not v.any() for v in counts.values()), "no stale pre-collision work")
            continue
        contact = stage_candidates(data); contacts[stage] = contact
        need(counts["ncollision"].shape == (1,) and 0 < int(counts["ncollision"][0]) <= 256, "bounded unchanged generated collision count")
        if stage == "after_collision":
            saved.contact_layout(contact)
            need(all(not counts[k].any() for k in saved.p.COUNTERS), "candidate stage has zero EFC/solver work")
        else:
            candidate_stability(contacts["after_collision"], contact)
            need(all(counts[k].shape == (2,) for k in saved.p.COUNTERS)
                 and not counts["ne"].any() and not counts["nl"].any() and (counts["nf"] == 14).all()
                 and np.array_equal(counts["nefc"], 14 + 4 * np.bincount(contact["worldid"], minlength=2)),
                 "complete friction/contact stage counters")
            active_response(data, descriptor)
            if stage != "after_solve": need(not counts["solver_niter"].any(), "no solver before declared boundary")
            else: need(((counts["solver_niter"] >= 0) & (counts["solver_niter"] <= descriptor["options"]["iterations"])).all(), "bounded actual solver iterations")
        if stage != "after_collision":
            need(counters[stage]["nacon"] == counters["after_collision"]["nacon"]
                 and counters[stage]["ncollision"] == counters["after_collision"]["ncollision"], "unchanged collision counters downstream")
    before = {k: arrays["before_solve" + k] for k in layout}
    after = {k: arrays["after_solve" + k] for k in layout}
    changed = solver_write_fence(before, after)
    data = decoded["after_solve"]; active = active_response(data, descriptor)
    fields = {k: data["/data/" + k].reshape(2, -1) for k in saved.p.p.base.FIELDS}
    arithmetic = dict(resultants=saved.p.prior.resultants(active), generalized=ordered_reconstruction(active, fields))
    cross = []
    current = {tuple(x["key"]): x for x in arithmetic["resultants"]}
    for side in ("cpu", "gpu"):
        prior = saved.p.prior.resultants(banks[side + "-active.npz"])
        other = {tuple(x["key"]): x for x in prior}
        for key in sorted(current.keys() | other.keys()):
            a, b = current.get(key), other.get(key)
            cross.append(dict(historical_side=side, key=list(key), current_count=0 if a is None else a["count"],
                historical_count=0 if b is None else b["count"], current_minus_historical=None if a is None or b is None else {
                    k: (np.asarray(a[k]) - np.asarray(b[k])).tolist() for k in ("world_force_N", "world_torque_about_origin_Nm")},
                physical_point_identity_established=False, same_solver_inputs_established=False))
    return dict(counters=counters, solver_changed_data_fields=changed,
                candidate_metadata_comparison=saved.metadata_comparison(contacts["after_collision"], banks["gpu-active.npz"]),
                complete_resultant_arithmetic=arithmetic, descriptive_historical_resultants=cross,
                current_minus_native_fields=saved.p.p.base.comparison(banks["cpu-fields.npz"], fields),
                current_minus_historical_ada_fields={k: field_delta(banks["gpu-fields.npz"][k], v) for k, v in fields.items()},
                arithmetic_precision="Float32 pyramid pair/add decoding; generalized float64 scalar products and ascending raw-row additions (no BLAS); float64 aggregate wrenches; no tolerance",
                generated_manifold_comparison_not_same_manifold_solver_control=True)


def validate_service(service, source_sha, *, pid=None):
    need(type(service) is dict and set(service) == {"unit", "properties"}
         and service["unit"] == "microduck-ada-response-" + source_sha[:8] + ".service", "closed source-bound response service")
    row = service["properties"]
    need(type(row) is dict and set(row) == set(SERVICE_CAPS) | {"MainPID", "InvocationID", "Environment"}
         and all(type(row[k]) is str and row[k] == v for k, v in SERVICE_CAPS.items())
         and type(row["MainPID"]) is str and re.fullmatch(r"[1-9][0-9]*", row["MainPID"])
         and (pid is None or row["MainPID"] == str(pid))
         and type(row["InvocationID"]) is str and re.fullmatch(r"[0-9a-f]{32}", row["InvocationID"])
         and type(row["Environment"]) is str, "all declared running service caps and identity")
    env = shlex.split(row["Environment"])
    need(len(env) == len(SERVICE_ENV) and all("=" in x for x in env)
         and dict(x.split("=", 1) for x in env) == SERVICE_ENV,
         "actual literal CPU-hidden single-thread response environment")
    return service


def unit_identity(source_sha):
    host = saved.p.p.base.host
    unit = "microduck-ada-response-" + source_sha[:8] + ".service"
    props = tuple(SERVICE_CAPS) + ("MainPID", "InvocationID", "Environment")
    raw = host.read("systemctl", "--user", "show", unit, *["--property=" + k for k in props])
    row = dict(line.split("=", 1) for line in raw.splitlines())
    return validate_service(dict(unit=unit, properties=row), source_sha, pid=os.getpid())


def receive(output, source_sha, root, *, cache=None):
    """Closed pure-CPU reception; no solver, loader or runtime import.

    Native reception supplies cache to rehash every cache file. Portable
    reception without it checks the declared inventory, not those file bytes.
    Neither mode authenticates the bytes of an actually loaded binary.
    """
    import numpy as np
    host = saved.p.p.base.host; output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size <= 256 * 1024, "bounded plain response JSON")
    result = json.loads(output.read_bytes())
    fixed = {"plant", "options", "model_array_sha256", "data_layout", "stage_source_audit", "solver_changed_data_fields",
             "static_binding_sha256", "static_binding_convention", "protocol", "decision", "source", "module_sha256",
             "versions", "predecessor_report_sha256", "input_file_sha256", "private_cache_was_absent", "private_cache_files",
             "existing_cpu_executables", "loaded_binary_bytes_bound", "payload", "analysis", "running_service"}
    need(set(result) == fixed | set(CONTRACT) and all(type(result[k]) is type(v) and result[k] == v for k, v in CONTRACT.items())
         and set(result["flags"]) == set(CONTRACT["flags"]) and all(v is False for v in result["flags"].values())
         and result["protocol"] == PROTOCOL and result["decision"] == DECISION and result["source"] == source_sha
         and re.fullmatch(r"[0-9a-f]{40}", source_sha)
         and result["module_sha256"] == sha256(host.read("git", "show", source_sha + ":src/mjlab_microduck/ada_saved_pose_response.py", binary=True)).hexdigest(),
         "closed exact-source unqualified response receipt")
    report, banks = saved.p.prior.authenticated_banks(root)
    need(result["predecessor_report_sha256"] == saved.p.prior.REPORT_SHA256 and result["input_file_sha256"] == report["files"]
         and result["stage_source_audit"] == source.audit()
         and set(result["versions"]) == set(saved.p.VERSIONS)
         and all(result["versions"][k].split("+")[0] == v for k, v in saved.p.VERSIONS.items())
         and result["model_array_sha256"] == {k: v["sha256"] for k, v in report["child"]["input_manifest"].items() if k.startswith("/model/")}
         and result["options"] == dict(saved.p.WARP_COLLISION_OPTIONS, broadphase=0, broadphase_filter=11)
         and result["plant"] == dict(selected_fields_sha256=report["child"]["plant"]["selected_fields_sha256"],
              options=report["child"]["plant"]["options"], collision_options=saved.p.COLLISION_OPTIONS, collision_flag_arm="unchanged-compiled-default")
         and result["static_binding_convention"] == STATIC_CONVENTION
         and re.fullmatch(r"[0-9a-f]{64}", result["static_binding_sha256"]), "same frozen sources/inputs/model/options with within-process identity only")
    saved.provenance_layout(result)
    validate_service(result["running_service"], source_sha)
    path = output.with_suffix(".npz"); payload = result["payload"]
    need(type(payload) is dict and set(payload) == {"file", "bytes", "sha256", "arrays"}
         and type(payload["bytes"]) is int and type(payload["arrays"]) is dict
         and len(payload["arrays"]) == len(STAGES) * 114
         and re.fullmatch(r"[0-9a-f]{64}", payload["sha256"])
         and path.is_file() and not path.is_symlink() and path.name == payload["file"]
         and path.stat().st_size == payload["bytes"] < 16 * 1024**2, "complete plain response stage NPZ")
    raw = path.read_bytes(); need(sha256(raw).hexdigest() == payload["sha256"], "exact response NPZ bytes")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        need(len(bank.files) == len(set(bank.files)) and set(bank.files) == set(payload["arrays"]), "complete unique response stage arrays")
        arrays = {k: bank[k].copy() for k in bank.files}
    for k, v in arrays.items():
        row = payload["arrays"][k]
        need(row == dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha256(v.tobytes()).hexdigest()), "fully byte-bound response stage leaf")
    analysis = analyze_stages(arrays, result["data_layout"], report, banks)
    need(analysis == result["analysis"] and result["solver_changed_data_fields"] == analysis["solver_changed_data_fields"], "every actual stage invariant and resultant recomputed")
    if cache is not None: need(saved.cache_files(Path(cache)) == result["private_cache_files"], "every retained native CPU cache byte")
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "cache"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args(); host = saved.p.p.base.host
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden response launch")
    need(datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "before campaign closeout reserve")
    need(re.fullmatch(r"[0-9a-f]{40}", args.source) and Path.cwd().resolve() == NATIVE_ROOT
         and Path(__file__).resolve().parents[2] == NATIVE_ROOT and host.read("cat", "/etc/machine-id") == MACHINE
         and host.read("git", "rev-parse", "HEAD") == args.source
         and host.read("git", "branch", "--show-current") == host.BRANCH
         and not host.read("git", "status", "--porcelain"), "clean exact native response source/machine")
    module = Path(__file__).read_bytes()
    need(module == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_saved_pose_response.py", binary=True), "committed response bytes")
    need(args.output.is_absolute() and args.cache.is_absolute() and not args.output.exists()
         and not args.output.with_suffix(".npz").exists() and not args.cache.exists()
         and args.output.parent.resolve(strict=True) == args.output.parent
         and args.cache.parent.resolve(strict=True) == args.cache.parent, "fresh canonical output and private response cache")
    running_service = unit_identity(args.source)
    versions = {k: version(k) for k in saved.p.VERSIONS}
    need(all(versions[k].split("+")[0] == v for k, v in saved.p.VERSIONS.items()), "unchanged frozen response packages")
    audit = source.audit(); report, banks = saved.p.prior.authenticated_banks(args.input)
    import warp as wp
    import torch
    args.cache.mkdir(); saved.configure_private_cpu_cache(wp, args.cache)
    wp.init()
    need(all(d.is_cpu for d in wp.get_devices()) and not torch.cuda.is_initialized(), "CPU-only response devices")
    result, arrays = run(report, banks)
    need(source.audit() == audit and not torch.cuda.is_initialized(), "unchanged sources and no initialized CUDA after response")
    result.update(protocol=PROTOCOL, decision=DECISION, source=args.source, module_sha256=sha256(module).hexdigest(),
        versions=versions, predecessor_report_sha256=saved.p.prior.REPORT_SHA256, input_file_sha256=report["files"],
        private_cache_was_absent=True, private_cache_files=saved.cache_files(args.cache),
        existing_cpu_executables=saved.passive_executables(), loaded_binary_bytes_bound=False,
        payload=saved.p.retain(args.output.with_suffix(".npz"), arrays), analysis=analyze_stages(arrays, result["data_layout"], report, banks),
        running_service=running_service)
    host.write_json(args.output, result)
    receive(args.output, args.source, args.input, cache=args.cache)
    print(DECISION, result["analysis"]["counters"]["after_solve"], result["analysis"]["descriptive_historical_resultants"])


if __name__ == "__main__": main()
