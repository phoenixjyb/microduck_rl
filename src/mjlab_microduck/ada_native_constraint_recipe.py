"""Pure current-native constraint recipes; no physics or cause qualification.

Uses the complete f854 capture, not a reconstructed historical native Model.
The independent C source recipe is not loaded-binary or FP32 emulation proof.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

from mjlab_microduck import ada_constraint_input_audit as audit
from mjlab_microduck import ada_native_constraint_capture as native

need, host = native.need, native.host
PROTOCOL = "microduck-current-native-constraint-recipe-v1"
DECISION = "current-native-constraint-arithmetic-not-cause-isolation-or-admission"
NATIVE_SOURCE = "f8540785a4ff5ee96d8186962c1934a9a15c0bb7"
NATIVE_REPORT_SHA = "52d76742ab534af2434d4dd698124f485c15a3e2ff3ee6e7171e93d5485961f4"
C_COMMIT = "28009f9105cd92784b7b0b30c0605a5e29107a77"
C_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco/" + C_COMMIT + "/src/engine/engine_core_constraint.c"
C_SHA = "71bcfcc6e3518846ee3b65e6835b14565aacf4027ea3c522b26498ea9ccc58cd"
C_BYTES = 102904
MODEL_COMPARE = ("body_invweight0", "geom_bodyid", "dof_invweight0", "dof_solref", "dof_solimp",
                 "geom_solref", "geom_solimp", "geom_friction")
CONTRACT = dict(runtime_imported=False, new_model_data_allocations=0, new_forward_calls=0,
    new_collision_calls=0, new_constraint_calls=0, new_solver_calls=0, integration_steps=0,
    optimizer_steps=0, historical_complete_native_model_identity=False,
    native_capture_is_before_solve=False, native_input_phase_replay_executed=False,
    contact_point_matching=False, row_count_normalization=False, actual_solver_matrix_reconstructed=False,
    exclusive_cause_established=False, loaded_native_binary_identity=False, fp32_machine_emulation=False,
    solver_qualified=False, simulator_qualified=False, training_authorized=False,
    physical_motion_authorized=False, flags=host.FLAGS)


def bind_c_source(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == C_BYTES, "plain exact declared upstream native C source")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == C_SHA, "exact declared upstream native C source bytes")
    return dict(url=C_URL, tag="3.10.0", commit=C_COMMIT, bytes=C_BYTES, sha256=C_SHA,
        recipe_functions=["mj_diagApprox", "getsolparam", "getposdim", "getimpedance", "mj_makeImpedance", "mj_referenceConstraint"],
        source_recipe_only=True, wheel_binary_correspondence_established=False)


def impedance(solimp, relative):
    """Ordered float64 scalar recipe restricted to captured power2 inputs."""
    dmin, dmax, width, mid, power = map(float, solimp)
    need(all(math.isfinite(v) for v in (dmin, dmax, width, mid, power, relative))
         and power == 2., "finite power2 native impedance recipe")
    lo, hi = audit.NUMERIC["minimp"], audit.NUMERIC["maximp"]
    dmin, dmax = min(max(dmin, lo), hi), min(max(dmax, lo), hi)
    width, mid = max(0., width), min(max(mid, lo), hi)
    need(dmin <= dmax, "ordered native impedance interval")
    if dmin == dmax or width <= audit.NUMERIC["minval"]:
        return .5 * (dmin + dmax), 0., dmax
    signed = relative / width; x = abs(signed)
    if x >= 1. or x <= 0.:
        return (dmax if x >= 1. else dmin), 0., dmax
    if x <= mid:
        scale = 1. / mid; y = scale * (x * x); dy = (2. * scale) * x
    else:
        scale = 1. / (1. - mid); z = 1. - x
        y = 1. - scale * (z * z); dy = (2. * scale) * z
    value = dmin + y * (dmax - dmin)
    derivative = ((dy * (-1. if signed < 0. else 1.)) * (dmax - dmin)) / width
    return value, derivative, dmax


def scalar_recipe(*, row_type, timestep, impratio, inverse_weight, solref, solimp,
                  position, margin, velocity, frictionloss, friction=None):
    """Captured rigid dim3 pyramids or DOF friction, no other row types.

    diagA output is the post-impedance adjusted diagonal, not diagApprox.
    Inputs never include captured D/R/KBIP/aref or final solver state.
    """
    need(row_type in (1, 6) and len(solref) == 2 and len(solimp) == 5,
         "closed native friction/pyramid row inventory")
    values = [timestep, impratio, inverse_weight, *solref, *solimp, position, margin, velocity, frictionloss]
    need(all(math.isfinite(float(v)) for v in values) and timestep > 0 and impratio > 0
         and inverse_weight > 0 and frictionloss >= 0, "finite positive native scalar inputs")
    ref0, ref1 = map(float, solref)
    need((ref0 > 0 and ref1 > 0) or (ref0 <= 0 and ref1 <= 0), "no mixed native reference format")
    if ref0 > 0: ref0 = max(ref0, 2. * float(timestep))
    I, P, dmax = impedance(solimp, float(position) - float(margin))
    floor = audit.NUMERIC["minval"]
    if row_type == 1:
        need(position == margin == 0. and friction is None, "DOF friction zero position/margin, no contact parameter")
        K, diagonal, mu = 0., float(inverse_weight), None
    else:
        need(friction is not None and len(friction) == 5 and all(math.isfinite(float(v)) and v > 0 for v in friction)
             and float(friction[0]) == float(friction[1]) and frictionloss == 0., "closed isotropic dim3 pyramid parameters")
        mu0 = float(friction[0]); diagonal = float(inverse_weight) + (mu0 * mu0) * float(inverse_weight)
        K = 1. / max(floor, ((((dmax * dmax) * ref0) * ref0) * ref1) * ref1) if ref0 > 0 else -ref0 / max(floor, dmax * dmax)
        mu = mu0
    B = 2. / max(floor, dmax * ref0) if ref1 > 0 else -ref1 / max(floor, dmax)
    R = max(floor, ((1. - I) * diagonal) / I)
    if row_type == 6:
        friction_R = R / max(floor, float(impratio))
        mu = float(friction[0]) * math.sqrt(friction_R / R)
        R = ((2. * mu) * mu) * R
    D = 1. / R; adjusted = (R * I) / (1. - I)
    aref = -B * float(velocity) - (K * I) * (float(position) - float(margin))
    result = dict(K=K, B=B, I=I, P=P, R=R, D=D, diagA=adjusted, aref=aref,
        vel=float(velocity), pos=float(position), margin=float(margin), frictionloss=float(frictionloss),
        pre_impedance_diagonal=diagonal, mu=mu)
    need(all(v is None or math.isfinite(v) for v in result.values()), "finite complete native scalar recipe")
    return result


def native_recipes(arrays, capture):
    import numpy as np
    rows, contacts, covered = [], [], [[], []]
    for w in range(2):
        scalars = capture["model_static"][w]["scalars"]
        value = lambda name: scalars["option/" + name]["value"]
        need(value("disableflags") == value("enableflags") == value("cone") == 0
             and value("solver") == 2 and value("noslip_iterations") == 0,
             "frozen no-override/no-diagexact dense Newton pyramidal branch")
        model = lambda name: arrays[f"model/{w}/model/" + name]
        efc = lambda name: arrays[f"efc/{w}/" + name]
        con = lambda name: arrays[f"contact/{w}/" + name]
        ncon, nefc = len(con("dist")), len(efc("type"))
        need((ncon, nefc) == ((0, 14) if w == 0 else (4, 30)) and
             efc("J").shape == (nefc * 20,) and arrays["fields/qvel"].shape == (2, 20), "fixed complete captured native layout")
        for name, shape, kind in (("dof_invweight0", (20,), "float64"), ("dof_solref", (20, 2), "float64"),
            ("dof_solimp", (20, 5), "float64"), ("dof_frictionloss", (20,), "float64"),
            ("geom_bodyid", (82,), "int32"), ("body_invweight0", (16, 2), "float64")):
            need(model(name).shape == shape and model(name).dtype == np.dtype(kind), "complete raw native recipe Model layout")
        need((efc("type")[:14] == 1).all() and (efc("type")[14:] == 6).all()
             and len(set(map(int, efc("id")[:14]))) == 14, "all native rows are partitioned own friction/contact rows")
        for slot in range(ncon):
            address = int(con("efc_address")[slot]); geoms = list(map(int, con("geom")[slot]))
            need(con("dim")[slot] == 3 and address == 14 + 4 * slot and
                 all(0 <= g < 82 for g in geoms) and (con("flex")[slot] == -1).all(), "own complete included rigid dim3 contact slot")
            bodies = [int(model("geom_bodyid")[g]) for g in geoms]
            need(all(0 <= b < 16 for b in bodies), "complete native geom-body linkage")
            inputs = dict(world=w, slot=slot, geom=geoms, bodies=bodies,
                inverse_weight=sum(float(model("body_invweight0")[b, 0]) for b in bodies),
                friction=con("friction")[slot].tolist(), solref=con("solref")[slot].tolist(),
                solimp=con("solimp")[slot].tolist(), position=float(con("dist")[slot]),
                margin=float(con("includemargin")[slot]), efc_addresses=list(range(address, address + 4)))
            contacts.append(inputs)
        for r in range(nefc):
            kind, rowid = int(efc("type")[r]), int(efc("id")[r])
            velocity = 0.
            for j in range(20): velocity = velocity + float(efc("J")[r * 20 + j]) * float(arrays["fields/qvel"][w, j])
            if kind == 1:
                need(0 <= rowid < 20, "own friction dof index")
                inputs = dict(inverse_weight=float(model("dof_invweight0")[rowid]),
                    solref=model("dof_solref")[rowid].tolist(), solimp=model("dof_solimp")[rowid].tolist(),
                    position=0., margin=0., frictionloss=float(model("dof_frictionloss")[rowid]), friction=None)
            else:
                need(0 <= rowid < ncon and r in list(range(int(con("efc_address")[rowid]), int(con("efc_address")[rowid]) + 4)), "native own contact row linkage")
                c = next(c for c in contacts if c["world"] == w and c["slot"] == rowid)
                inputs = {k:c[k] for k in ("inverse_weight", "solref", "solimp", "position", "margin", "friction")}
                inputs["frictionloss"] = 0.
            recipe = scalar_recipe(row_type=kind, timestep=value("timestep"), impratio=value("impratio"), velocity=velocity, **inputs)
            captured = {k:float(efc(k)[r]) for k in ("R", "D", "diagA", "aref", "vel", "pos", "margin", "frictionloss")}
            captured.update(dict(zip(("K", "B", "I", "P"), map(float, efc("KBIP")[r]))))
            rows.append(dict(world=w, row=r, row_type=kind, own_id=rowid, inputs=inputs, recipe=recipe, captured=captured,
                recipe_minus_captured={k:recipe[k] - v for k,v in captured.items()},
                recipe_fp64_bytes_equal={k:np.float64(recipe[k]).tobytes() == np.float64(v).tobytes() for k,v in captured.items()}))
            covered[w].append(r)
    need(covered == [list(range(14)), list(range(30))], "all44 native rows covered once, no masking or normalization")
    return rows, contacts


def model_precision_brackets(arrays, prior, banks, gpu_child):
    import numpy as np
    # Establish every raw historical Model binding to the actually measured GPU.
    audit.bound_model_values(prior, banks, gpu_child)
    result = []
    for name in MODEL_COMPARE:
        key = "/model/" + name
        layout = audit.response.expanded_layout(prior["child"]["input_manifest"][key])
        gpu = np.frombuffer(banks["prepared-inputs.npz"][key].tobytes(), layout["numpy_dtype"]).reshape(layout["numpy_shape"])
        if name != "geom_bodyid":
            need(gpu.shape[0] == 1, "explicit broadcast Model world axis")
            gpu = gpu[0]
        for w in range(2):
            actual = arrays[f"model/{w}/model/" + name]
            need(actual.shape == gpu.shape and np.isfinite(actual).all() and np.isfinite(gpu).all(), "same selected numeric model topology")
            result.append(dict(world=w, native_path="model/" + name, measured_gpu_path=key,
                native_dtype=str(actual.dtype), gpu_dtype=str(gpu.dtype), shape=list(actual.shape),
                native_cast_to_gpu_dtype_bytes_equal=actual.astype(gpu.dtype).tobytes() == gpu.tobytes(),
                native_minus_widened_gpu_max_abs=float(np.max(np.abs(actual.astype(np.float64) - gpu.astype(np.float64))))))
    return result


def analyze(input_root, measured_root, native_packet, c_source):
    need(not any(n in sys.modules for n in ("mujoco", "mujoco_warp", "warp", "torch")), "pure fresh arithmetic process")
    native_path = Path(native_packet)
    need(native_path.is_file() and not native_path.is_symlink() and native_path.stat().st_size <= 256 * 1024
         and sha256(native_path.read_bytes()).hexdigest() == NATIVE_REPORT_SHA, "anchored complete current native report")
    capture, native_arrays = native.receive(native_packet, NATIVE_SOURCE, input_root)
    need(capture["capture"]["historical_outputs_equal"] and capture["capture"]["seven_states_unchanged"]
         and capture["capture"]["warning_counts"] == [0, 0] and capture["capture"]["computed_arrays_finite"], "captured historical output agreement, not Model identity")
    need(sha256((Path(measured_root) / "report.json").read_bytes()).hexdigest() == audit.GPU_REPORT_SHA, "anchored measured GPU report")
    gpu, raw, layout, static, _ = audit.measured.receive(measured_root, audit.GPU_SOURCE, input_root)
    prior, banks = native.prior.authenticated_banks(input_root)
    reference = audit.packet.measured_reference_banks(raw, layout, prior, banks)
    decoded = audit.response.decode_stages(raw, layout, prior, reference)
    before, after = decoded["before_solve"], decoded["after_solve"]
    for k in ("J", "D", "aref"):
        need(before["/data/efc/" + k].tobytes() == after["/data/efc/" + k].tobytes(), "measured GPU construction inputs unchanged by solve")
    models = audit.bound_model_values(prior, banks, gpu["child"])
    gpu_rows = audit.gpu_row_recipes(before, models, static)
    native_rows, native_contacts = native_recipes(native_arrays, capture["capture"])
    gpu_contacts = audit.response.stage_candidates(before)
    gpu_table = [dict(world=int(gpu_contacts["worldid"][i]), slot=i,
        **{k:gpu_contacts[k][i].tolist() for k in ("geom", "friction", "solref", "solimp", "dist", "includemargin", "efc_address")}) for i in range(8)]
    numeric = audit.numeric_source(); source = bind_c_source(c_source)
    result = dict(protocol=PROTOCOL, decision=DECISION, native_source=NATIVE_SOURCE,
        native_report_sha256=NATIVE_REPORT_SHA, native_payload_sha256=capture["payload"]["sha256"], native_mjb_files=capture["mjb"],
        predecessor_report_sha256=native.prior.REPORT_SHA256, predecessor_files=prior["files"],
        measured_gpu_source=audit.GPU_SOURCE, measured_gpu_report_sha256=audit.GPU_REPORT_SHA, measured_gpu_files=gpu["files"],
        native_recipe_source=source, numeric_header_source=numeric,
        gpu_stage_source_audit_sha256=audit.packet.digest(audit.response.source.audit()),
        native_stage="Current post-forward capture; J/state/contact/Model arithmetic, no native stage replay",
        gpu_stage="Retained before_solve construction inputs, byte-unchanged after_solve",
        arithmetic="Ordered Python float64 arithmetic over each backend's own indices; no BLAS or tolerance",
        native_row_recipes=native_rows, native_contact_inputs=native_contacts, measured_gpu_contact_inputs=gpu_table,
        measured_gpu_row_recipes=gpu_rows, selected_model_precision_brackets=model_precision_brackets(native_arrays, prior, banks, gpu["child"]),
        native_recipe_max_abs={k:max(abs(row["recipe_minus_captured"][k]) for row in native_rows) for k in native_rows[0]["captured"]},
        native_recipe_byte_disagreements={k:sum(not row["recipe_fp64_bytes_equal"][k] for row in native_rows) for k in native_rows[0]["captured"]},
        native_row_count=44, native_contact_row_count=16, native_dof_friction_row_count=28,
        measured_gpu_contact_row_count=32, native_contact_count=4, measured_gpu_contact_count=8,
        **CONTRACT)
    need(not any(n in sys.modules for n in ("mujoco", "mujoco_warp", "warp", "torch")), "arithmetic remained runtime-inert")
    return result


def source_check(source):
    audit.source_check(source)
    need(Path(__file__).read_bytes() == host.read("git", "show", source + ":src/mjlab_microduck/ada_native_constraint_recipe.py", binary=True), "exact committed native arithmetic module")


def receive(path, source, input_root, measured_root, native_packet, c_source):
    source_check(source); path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024, "bounded complete native arithmetic result")
    actual = json.loads(path.read_bytes()); expected = analyze(input_root, measured_root, native_packet, c_source)
    expected.update(source=source, module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    need(audit.packet.digest(actual) == audit.packet.digest(expected), "every arithmetic result completely recomputed")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "measured", "native", "c-source", "output"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source", required=True); args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "CPU-hidden native arithmetic before closeout reserve")
    need(all(os.environ.get(n + "_NUM_THREADS") == "1" for n in ("OMP", "MKL", "OPENBLAS", "NUMEXPR")), "four literal single-thread CPU settings")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(), "fresh canonical arithmetic output")
    source_check(args.source)
    result = analyze(args.input, args.measured, args.native, args.c_source)
    result.update(source=args.source, module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source_check(args.source); host.write_json(args.output, result)
    receive(args.output, args.source, args.input, args.measured, args.native, args.c_source)
    print(DECISION, result["native_recipe_max_abs"], result["native_recipe_byte_disagreements"])


if __name__ == "__main__": main()
