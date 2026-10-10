"""Bounded native saved-state phase capture, never simulator/PPO admission.

Two retained MJBs, two fresh Data objects, no integration. Only the explicit
public forward stages in the frozen phase plan are permitted. Internal native
dispatch and loaded machine-code identity are not established by Python counts.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from mjlab_microduck import ada_native_phase_getters as getters

projection, plan = getters.projection, getters.projection.plan
native, host, need = plan.native, plan.host, plan.need
PROTOCOL = "microduck-native-saved-state-phase-capture-v1"
DECISION = "native-before-after-constraint-projection-retained-not-admission"
CALL_ORDER = tuple((w, n) for w in (0, 1) for n in plan.BEFORE + plan.AFTER)
CONTRACT = dict(integration_steps=0, optimizer_steps=0, gpu_workloads=0,
    full_native_Data_inventory=False, internal_solver_dispatch_count_established=False,
    compiled_binary_identity_established=False, full_historical_model_identity_established=False,
    contact_point_matching=False, solver_qualified=False, simulator_qualified=False,
    training_authorized=False, physical_motion_authorized=False, flags=host.FLAGS)


@contextmanager
def public_stage_guard(runtime):
    """Block unexpected public mj_* calls; not a transitive C call counter."""
    permitted = set(plan.BEFORE + plan.AFTER) | {"mj_sizeModel", "mj_saveModel"}
    originals, calls, denied = {}, [], []
    def blocked(name):
        def call(*a, **k):
            denied.append(name)
            raise ValueError("unexpected public native call: " + name)
        return call
    try:
        for name in dir(runtime):
            if name.startswith("mj_") and callable(getattr(runtime, name)) and name not in permitted:
                original = getattr(runtime, name)
                setattr(runtime, name, blocked(name))
                originals[name] = original
        yield calls, denied
    finally:
        for name, fn in originals.items():
            setattr(runtime, name, fn)


def callbacks_absent(runtime):
    names = tuple(n for n in dir(runtime) if n.startswith("get_mjcb_"))
    need(names == native.CALLBACKS and all(getattr(runtime, n)() is None for n in names),
         "exact absent native callback hooks")


def committed_modules(source):
    for name in ("ada_native_phase_capture.py", "ada_native_phase_getters.py", "ada_native_phase_projection.py"):
        path = Path(__file__).with_name(name)
        need(path.read_bytes() == host.read("git", "show", source + ":src/mjlab_microduck/" + name, binary=True),
             "exact committed collector and getter adapter")


def source_check(source):
    plan.source_check(source)
    committed_modules(source)


def inputs(input_root, measured, tools):
    # Independently regenerate the entire frozen preflight before runtime import.
    old = Path(tools) / "8382487e-linux-native-phase-plan.json"
    plan.own.hybrid.same.exact_file(old, projection.PLAN_SHA, 256 * 1024)
    expected = plan.analyze(input_root, measured, tools)
    expected.update(source=projection.PLAN_SOURCE,
        module_sha256=plan.own.hybrid.archived_module(plan, projection.PLAN_SOURCE))
    need(plan.own.audit.packet.digest(expected) == plan.own.audit.packet.digest(json.loads(old.read_bytes())),
         "entire predeclared phase plan regenerated")
    capture, bank = native.receive(Path(tools) / "f8540785-linux-native-constraint.json",
        plan.own.recipe.NATIVE_SOURCE, input_root)
    report, banks = native.prior.authenticated_banks(input_root)
    state = native.replay.state_inputs(report, banks)
    return capture, bank, state


def fence(runtime, model, world, bank, static, mjb):
    arrays, meta = native.model_snapshot(model)
    expected = {k.removeprefix(f"model/{world}/"): a for k, a in bank.items()
                if k.startswith(f"model/{world}/")}
    need(native.compare(arrays, expected) == {k: dict(dtype_equal=True, shape_equal=True,
        bytes_equal=True) for k in expected} and meta == static, "full public Model/Option fence")
    size = runtime.mj_sizeModel(model)
    need(size == len(mjb) and 0 < size < native.MJB_MAX, "bounded exact serialized Model size")
    raw = np.empty(size, np.uint8)
    runtime.mj_saveModel(model, None, raw)
    need(raw.tobytes() == mjb, "full MJB byte fence")
    return dict(public_numeric_fields=len(arrays), arrays_sha256=plan.own.audit.packet.digest(native.manifest(arrays)),
        static_sha256=plan.own.audit.packet.digest(meta), mjb_sha256=sha256(mjb).hexdigest())


def capture_phases(runtime, capture, bank, states, tools):
    """Runtime supplied by fresh Linux main; tests use explicit mock objects."""
    callbacks_absent(runtime)
    payload, records = {}, []
    with public_stage_guard(runtime) as (calls, denied):
        for world in (0, 1):
            row = capture["mjb"][world]
            path = Path(tools) / row["file"]
            raw = path.read_bytes()
            need(sha256(raw).hexdigest() == row["sha256"] and len(raw) == row["bytes"], "fresh MJB reauthentication")
            model = runtime.MjModel.from_binary_path(str(path))
            meta = capture["capture"]["model_static"][world]
            initial = fence(runtime, model, world, bank, meta, raw)
            plan.model_guards([meta, meta])
            data = runtime.MjData(model)
            seven = {k: states[k][world].astype(np.float64).reshape(-1) for k in native.STATE}
            for name, value in seven.items():
                if name == "time":
                    need(type(data.time) is float and np.isfinite(data.time), "plain native time before restore")
                    data.time = float(value[0])
                else:
                    target = getters.attribute(data, name)
                    shape = (native.base.WIDTHS[name],)
                    alias = (getters.FIELD_GETTER_SHAPES[name],) if name in getters.FIELD_GETTER_SHAPES else ()
                    getters.copy_leaf(target, np.dtype("float64"), shape, alias)
                    target[:] = value.reshape(target.shape)
            callbacks_absent(runtime)
            for name in plan.BEFORE:
                getattr(runtime, name)(model, data)
                calls.append((world, name))
            before, _ = getters.snapshot(data, world, seven)
            before_fence = fence(runtime, model, world, bank, meta, raw)
            callbacks_absent(runtime)
            runtime.mj_fwdConstraint(model, data)
            calls.append((world, "mj_fwdConstraint"))
            after, _ = getters.snapshot(data, world, seven)
            after_fence = fence(runtime, model, world, bank, meta, raw)
            callbacks_absent(runtime)
            need(initial == before_fence == after_fence, "all three model fences unchanged")
            comparison = projection.compare(before, after, world, seven)
            records.append(dict(world=world, fences=initial, comparison=comparison))
            for phase, arrays in (("before", before), ("after", after)):
                payload.update({f"{world}/{phase}/{k}": a for k, a in arrays.items()})
        need(tuple(calls) == CALL_ORDER and not denied, "exact two-world public stage order")
    return payload, dict(records=records, public_stage_calls=[list(c) for c in calls],
        model_allocations=2, data_allocations=2, before_after_capture_executed=True,
        public_stage_identity_recorded=True, native_callback_hooks=list(native.CALLBACKS), **CONTRACT)


def decode_payload(path, descriptor):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == descriptor["bytes"] < 4 * 1024**2,
         "plain bounded phase payload")
    raw = path.read_bytes()
    need(descriptor["file"] == path.name and sha256(raw).hexdigest() == descriptor["sha256"], "phase payload bytes")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows = z.infolist()
        expected = {f"{w}/{p}/{k}.npy" for w in (0, 1) for p in ("before", "after") for k in projection.layout(w)}
        need(len(rows) == len(expected) and {r.filename for r in rows} == expected
             and sum(r.file_size for r in rows) < 4 * 1024**2
             and all(r.compress_type == zipfile.ZIP_DEFLATED for r in rows), "closed bounded archive before decode")
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        arrays = {k: bank[k].copy() for k in bank.files}
    need(native.manifest(arrays) == descriptor["arrays"], "every complete payload descriptor")
    return arrays


def receive(output, source, input_root, measured, tools):
    """Pure full-packet reception: no native runtime import or calls."""
    need(not {"mujoco", "warp", "torch", "mujoco_warp"} & sys.modules.keys(), "fresh pure receiver")
    source_check(source)
    output = Path(output)
    need(output.is_file() and not output.is_symlink() and output.stat().st_size < 256 * 1024, "bounded plain report")
    value = json.loads(output.read_bytes())
    need(set(value) == {"protocol", "decision", "source", "module_sha256", "input_plan_sha256", "capture",
        "payload", "native_installed_provenance", "protected_services", "foreign_processes", "telemetry"}, "closed report fields")
    module = host.read("git", "show", source + ":src/mjlab_microduck/ada_native_phase_capture.py", binary=True)
    need(value["source"] == source and module == Path(__file__).read_bytes()
         and value["module_sha256"] == sha256(module).hexdigest()
         and value["protocol"] == PROTOCOL and value["decision"] == DECISION
         and value["input_plan_sha256"] == projection.PLAN_SHA, "exact source and declared input plan")
    old, bank, state = inputs(input_root, measured, tools)
    arrays = decode_payload(output.with_suffix(".npz"), value["payload"])
    records = []
    for w in (0, 1):
        seven = {k: state[k][w].astype(np.float64).reshape(-1) for k in native.STATE}
        phase = {p: {k: arrays[f"{w}/{p}/{k}"] for k in projection.layout(w)} for p in ("before", "after")}
        models = {k.removeprefix(f"model/{w}/"): a for k, a in bank.items() if k.startswith(f"model/{w}/")}
        fence_row = dict(public_numeric_fields=len(models), arrays_sha256=plan.own.audit.packet.digest(native.manifest(models)),
            static_sha256=plan.own.audit.packet.digest(old["capture"]["model_static"][w]), mjb_sha256=old["mjb"][w]["sha256"])
        records.append(dict(world=w, fences=fence_row, comparison=projection.compare(phase["before"], phase["after"], w, seven)))
    expected = dict(records=records, public_stage_calls=[list(c) for c in CALL_ORDER], model_allocations=2,
        data_allocations=2, before_after_capture_executed=True, public_stage_identity_recorded=True,
        native_callback_hooks=list(native.CALLBACKS), **CONTRACT)
    need(plan.own.audit.packet.digest(value["capture"]) == plan.own.audit.packet.digest(expected), "complete comparison independently recomputed")
    provenance = value["native_installed_provenance"]
    need(provenance == old["native_installed_provenance"] and set(provenance) == set(native.NATIVE_FILES) and all(row["sha256"] == native.NATIVE_FILES[k]
         and row["record_sha256_matches"] is True and type(row["bytes"]) is int
         and 0 < row["bytes"] < 16 * 1024**2 for k, row in provenance.items()), "frozen installed native provenance")
    need(value["protected_services"] == {s + ":" + n: "inactive" for s in ("system", "user") for n in host.SERVICES}, "protected scopes inactive")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("input", "measured", "tools", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--receive-only", action="store_true")
    a = p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(n + "_NUM_THREADS") == "1"
         for n in ("OMP", "MKL", "OPENBLAS", "NUMEXPR")), "CPU hidden/four thread1 settings")
    if a.receive_only:
        receive(a.output, a.source, a.input, a.measured, a.tools)
        print("full phase packet received; not admission")
        return
    host.source_check(a.source)
    source_check(a.source)
    need(a.output.is_absolute() and a.output.parent.resolve(strict=True) == a.tools.resolve(strict=True)
         and a.output.name == a.source[:8] + "-linux-native-phase-capture.json"
         and not any(a.output.with_suffix(s).exists() for s in (".json", ".npz")), "fresh exact source-prefix outputs")
    services, foreign = host.service_snapshot(), host.foreign_processes()
    old, bank, state = inputs(a.input, a.measured, a.tools)
    provenance = native.native_provenance()
    need(not {"mujoco", "warp", "torch", "mujoco_warp"} & sys.modules.keys(), "fresh no simulator imports before capture")
    import mujoco
    payload, result = capture_phases(mujoco, old, bank, state, a.tools)
    need(not {"warp", "torch", "mujoco_warp"} & sys.modules.keys(), "native-only runtime")
    need(services == host.service_snapshot() and foreign == host.foreign_processes(), "foreign workloads unchanged")
    host.source_check(a.source)
    source_check(a.source)
    descriptor = native.retain(a.output.with_suffix(".npz"), payload)
    value = dict(protocol=PROTOCOL, decision=DECISION, source=a.source,
        module_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), input_plan_sha256=projection.PLAN_SHA,
        capture=result, payload=descriptor, native_installed_provenance=provenance,
        protected_services=services, foreign_processes=foreign, telemetry=host.telemetry())
    need(len(json.dumps(value, allow_nan=False).encode()) < 256 * 1024, "bounded final report")
    host.write_json(a.output, value)
    print(DECISION, [sum(not v["bytes_equal"] for v in r["comparison"]["differences"].values()) for r in result["records"]])


if __name__ == "__main__":
    main()
