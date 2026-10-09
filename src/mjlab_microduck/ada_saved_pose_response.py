"""Guarded staged CPU response primitives; CLI/reception gate not yet complete.

Artificial prepared-pose intervention, not integrated forward or admission.
Importing this module initializes no simulator runtime. No standalone launch.
"""
from hashlib import sha256
import os

from mjlab_microduck import ada_saved_pose_collision as saved
from mjlab_microduck import ada_solver_stage_audit as source

need = saved.need
POSES = saved.p.KINEMATIC + ("site_xpos", "site_xmat")
STAGES = ("before_collision", "after_collision", "after_construction", "before_solve", "after_solve")
CALLBACKS = ("passive", "control", "act_dyn", "act_gain", "act_bias", "sensor", "contactfilter")
SOLVER_OUTPUTS = {"/data/qacc", "/data/qfrc_constraint", "/data/solver_niter",
                  "/data/efc/Ma", "/data/efc/force", "/data/efc/state"}


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
    """One CPU response intervention; not reachable from a CLI yet."""
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
                static_binding_convention="Within-process unchanged pointers, shapes, options and scalars; not historical execution identity",
                allocation_native_kinematics_calls=1, fixture_native_kinematics_calls=0,
                fixture_warp_kinematics_calls=0, cpu_collision_calls=1, new_constraint_calls=1,
                new_solver_calls=1, new_integration_steps=0, acceleration_sensor_calls=0,
                interpretation="Eleven prepared-pose intervention, CPU-generated manifold and descriptive CPU response",
                measured_gpu_collision_pose_complete=False, same_manifold_solver_control=False,
                flags=saved.p.p.FLAGS, solver_qualified=False, training_authorized=False,
                physical_motion_authorized=False), {stage + key: value for stage, raw in boundaries.items() for key, value in raw.items()}
