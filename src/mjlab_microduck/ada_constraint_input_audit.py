"""Pure arithmetic audit of measured contact inputs and unlike row-sum proxies.

New GPU Data only; old numeric Model bytes only after all347 exact bindings.
No runtime imports, reconstruction of native parameters, or point matching.
"""
import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distribution
import json
import math
import os
from pathlib import Path
import re

from mjlab_microduck import ada_measured_gpu_boundary as measured

packet, response, saved, host, need = (measured.packet, measured.response, measured.saved, measured.host, measured.need)
PROTOCOL = "microduck-measured-constraint-input-arithmetic-v1"
DECISION = "measured-constraint-input-arithmetic-not-cause-isolation-or-admission"
GPU_SOURCE = "daebeb6ee97b1ce0c4a665331306bcb78a94204c"
GPU_REPORT_SHA = "5146e17ace38e5b0e2897cb0a3febf1202b66d9d69b757ac0398e2c581d267de"
HEADERS = {
    "mujoco/include/mujoco/mjmodel.h": "bcd51b20cb29b6aac7c8b9e1cf348f569b7fc2239f9226b844737c73511d90c8",
    "mujoco/include/mujoco/mjtype.h": "ec580ce2a4ef0c1f6a3e68b3c5b5eeaf61d03413827453e2d0a87c3bd92d16e4",
}
NUMERIC = dict(minval=1e-15, minimp=.0001, maximp=.9999, refsafe_bit=4096)
# ContactType.CONSTRAINT in the independently RECORD-bound frozen types.py.
CONTACT_CONSTRAINT_BIT = 1
MODEL_INPUTS = ("/model/opt/timestep", "/model/opt/impratio_invsqrt", "/model/body_invweight0", "/model/geom_bodyid")
CONTRACT = dict(runtime_imported=False, new_allocations=0, new_kinematics_calls=0, new_collision_calls=0,
    new_constraint_calls=0, new_solver_calls=0, new_integration_steps=0, optimizer_steps=0,
    native_parameter_reconstruction=False, contact_point_matching=False, row_count_normalization=False,
    actual_solver_matrix_reconstructed=False, exclusive_cause_established=False,
    compiled_binary_identity_established=False, solver_qualified=False, simulator_qualified=False,
    training_authorized=False, physical_motion_authorized=False, flags=host.FLAGS)


def source_check(source):
    root = Path(__file__).resolve().parents[2]
    need(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source)
         and host.read("git", "-C", str(root), "rev-parse", "HEAD") == source
         and host.read("git", "-C", str(root), "branch", "--show-current") == host.BRANCH
         and not host.read("git", "-C", str(root), "status", "--porcelain"), "clean exact arithmetic source revision")
    need(Path(__file__).read_bytes() == host.read("git", "-C", str(root), "show",
         source + ":src/mjlab_microduck/ada_constraint_input_audit.py", binary=True), "current arithmetic module bytes")


def numeric_source():
    dist = distribution("mujoco"); need(dist.version == "3.10.0", "frozen MuJoCo numeric header version")
    files = {str(f): f for f in dist.files}; raw, rows = {}, {}
    for name, expected in HEADERS.items():
        need(name in files, "numeric header in wheel inventory")
        item = files[name]; path = Path(dist.locate_file(item))
        need(path.is_file() and not path.is_symlink() and path.stat().st_size < 256 * 1024, "plain bounded numeric header")
        raw[name] = path.read_bytes(); digest = sha256(raw[name]).digest()
        need(digest.hex() == expected and item.hash and item.hash.mode == "sha256"
             and base64.urlsafe_b64encode(digest).decode().rstrip("=") == item.hash.value,
             "exact numeric header bytes and wheel RECORD")
        rows[name] = dict(sha256=expected, bytes=len(raw[name]), record_sha256_matches=True)
    model, types = (raw[n].decode() for n in HEADERS)
    need(re.search(r"#define\s+mjMINIMP\s+0\.0001\s", model)
         and re.search(r"#define\s+mjMAXIMP\s+0\.9999\s", model)
         and re.search(r"#define\s+mjMINVAL\s+1E-15\s", types)
         and re.search(r"mjDSBL_REFSAFE\s*=\s*1<<12", types), "source-backed numeric constants")
    return dict(headers=rows, numeric=NUMERIC)


def bound_model_values(prior, banks, child):
    """Historical MODEL bytes, never historical Data, with exact new binding."""
    import numpy as np
    expected = {k: v for k, v in prior["child"]["input_manifest"].items() if k.startswith("/model/")}
    raw = {k: v for k, v in banks["prepared-inputs.npz"].items() if k.startswith("/model/")}
    need(len(raw) == 347 and set(raw) == set(expected) == set(child["model_array_sha256"]), "all347 exact model input paths")
    for k, v in raw.items():
        need(v.dtype == np.uint8 and v.shape == (expected[k]["bytes"],)
             and sha256(v.tobytes()).hexdigest() == expected[k]["sha256"] == child["model_array_sha256"][k],
             "every old numeric Model byte bound to actual measured GPU model")
    values = {}
    for k in MODEL_INPUTS:
        row = response.expanded_layout(expected[k])
        values[k] = np.frombuffer(raw[k].tobytes(), row["numpy_dtype"]).reshape(row["numpy_shape"]).copy()
        need(np.isfinite(values[k]).all(), "finite exact model numeric input")
    need(values[MODEL_INPUTS[0]].shape == values[MODEL_INPUTS[1]].shape == (1,)
         and values[MODEL_INPUTS[2]].shape == (1, 16, 2) and values[MODEL_INPUTS[3]].shape == (82,)
         and values[MODEL_INPUTS[0]][0] > 0 and values[MODEL_INPUTS[1]][0] > 0,
         "unchanged numeric topology and positive model time/impedance")
    return values


def row_recipe(*, timestep, invsqrt, inverse_weight, friction, solref, solimp, distance, margin, velocity):
    """Ordered float64 source recipe for this frozen positive-solref/power2 arm.

    Decimal header constants and FP32 inputs are widened, not an emulation of
    Warp FP32 rounding/fusion or loaded machine code. No acceptance tolerance.
    """
    inputs = [timestep, invsqrt, inverse_weight, friction, *solref, *solimp, distance, margin, velocity]
    need(len(solref) == 2 and len(solimp) == 5 and all(math.isfinite(float(v)) for v in inputs)
         and timestep > 0 and invsqrt > 0 and inverse_weight > 0 and friction > 0
         and solref[0] > 0 and solref[1] > 0 and solimp[4] == 2., "closed positive-ref/power2 scalar recipe")
    tc = max(float(solref[0]), 2. * float(timestep))
    dr = float(solref[1]); dmin = min(max(float(solimp[0]), NUMERIC["minimp"]), NUMERIC["maximp"])
    dmax = min(max(float(solimp[1]), NUMERIC["minimp"]), NUMERIC["maximp"])
    width = max(float(solimp[2]), NUMERIC["minval"])
    mid = min(max(float(solimp[3]), NUMERIC["minimp"]), NUMERIC["maximp"])
    need(dmin <= dmax, "ordered impedance interval")
    pos = float(distance) - float(margin); x = abs(pos) / width
    if x < mid: y = (1. / mid) * (x * x)
    else:
        z = 1. - x; y = 1. - (1. / (1. - mid)) * (z * z)
    imp = min(max(dmin + y * (dmax - dmin), dmin), dmax)
    if x > 1.: imp = dmax
    inv = float(inverse_weight); mu = float(friction); ratio = float(invsqrt)
    inv = inv + (mu * mu) * inv
    inv = ((inv * 2.) * mu) * mu  # literal left-to-right source multiplications
    inv = (inv * ratio) * ratio
    k = 1. / ((((dmax * dmax) * tc) * tc) * dr * dr)
    b = 2. / (dmax * tc)
    result = dict(D=1. / max(inv * (1. - imp) / imp, NUMERIC["minval"]),
        aref=-(k * imp) * pos - b * float(velocity), vel=float(velocity),
        pos=pos + float(margin), margin=float(margin), frictionloss=0.,
        impedance=imp, effective_inverse_weight=inv, stiffness=k, damping=b)
    need(all(math.isfinite(v) for v in result.values()), "finite complete recipe results")
    return result


def gpu_row_recipes(data, model, manifest):
    import numpy as np
    need(manifest["scalars"]["/model/opt/disableflags"] == dict(kind="int", value=0), "frozen REFSAFE enabled, no disabled constraints")
    contacts = response.stage_candidates(data); count = len(contacts["dist"])
    need(count == 8 and (contacts["dim"] == 3).all(), "complete frozen eight dim3 contacts")
    need(contacts["type"].dtype == np.int32 and contacts["type"].shape == (8,)
         and ((contacts["type"] & CONTACT_CONSTRAINT_BIT) != 0).all(),
         "every contact enters the frozen CONSTRAINT update branch")
    rows = []
    for slot in range(count):
        w = int(contacts["worldid"][slot]); geoms = contacts["geom"][slot]
        need(w == 1 and all(0 <= int(g) < 82 for g in geoms), "frozen rigid shallow-world pairs")
        bodies = [int(model[MODEL_INPUTS[3]][int(g)]) for g in geoms]
        need(all(0 <= b < 16 for b in bodies), "complete actual geom/body mapping")
        inv = sum(float(model[MODEL_INPUTS[2]][0, b, 0]) for b in bodies)
        addresses = contacts["efc_address"][slot].tolist()
        need(addresses == list(range(addresses[0], addresses[0] + 4)), "four actual contiguous pyramid addresses")
        for dimid, addr in enumerate(addresses):
            need(14 <= addr < int(data["/data/nefc"][w])
                 and int(data["/data/efc/id"][w, addr]) == slot and int(data["/data/efc/type"][w, addr]) == 6,
                 "every recipe row linked to its own raw contact slot, not another backend")
            result = row_recipe(timestep=float(model[MODEL_INPUTS[0]][0]), invsqrt=float(model[MODEL_INPUTS[1]][0]),
                inverse_weight=inv, friction=float(contacts["friction"][slot, 0]), solref=contacts["solref"][slot].tolist(),
                solimp=contacts["solimp"][slot].tolist(), distance=float(contacts["dist"][slot]),
                margin=float(contacts["includemargin"][slot]), velocity=float(data["/data/efc/Jqvel"][w, addr]))
            captured = {k: float(data["/data/efc/" + k][w, addr]) for k in ("D", "aref", "vel", "pos", "margin", "frictionloss")}
            need(np.float32(captured["vel"]).tobytes() == data["/data/efc/Jqvel"][w, addr].tobytes(), "captured vel exactly equals own Jqvel")
            rows.append(dict(slot=slot, world=w, dimid=dimid, efc_address=addr, geom=geoms.tolist(), bodies=bodies,
                recipe=result, captured=captured, recipe_minus_captured={k: result[k] - v for k, v in captured.items()}))
    need(len(rows) == 32 and len({r["efc_address"] for r in rows}) == 32, "complete unique32-contact-row recipe coverage")
    return rows


def nominal_terms(J, D, aref, state, rows):
    """Unmasked nominal .5*a'Ha - h'a + c; NOT an actual solver Hessian."""
    import numpy as np
    n = len(D)
    need(J.dtype in (np.dtype("float32"), np.dtype("float64")) and D.dtype == aref.dtype == J.dtype
         and J.shape == (n, 20) and D.shape == aref.shape == state.shape == (n,) and state.dtype == np.int32
         and all(np.isfinite(v).all() for v in (J, D, aref)) and (D > 0).all()
         and type(rows) is list and rows == sorted(set(rows)) and all(type(r) is int and 0 <= r < n for r in rows),
         "finite positive complete nominal row inputs and ordered unique selection")
    H, h, c = [[0.] * 20 for _ in range(20)], [0.] * 20, 0.
    for r in rows:
        d, ref = float(D[r]), float(aref[r])
        c = c + (.5 * d * ref) * ref
        for i in range(20):
            x = float(J[r, i]); h[i] = h[i] + (x * d) * ref
            for j in range(20): H[i][j] = H[i][j] + (x * d) * float(J[r, j])
    need(np.isfinite(H).all() and np.isfinite(h).all() and math.isfinite(c), "finite ordered nominal proxy")
    return dict(H_JtDJ=H, h_JtDaref=h, c_half_aref_D_aref=c, rows=rows,
        D=[float(D[r]) for r in rows], aref=[float(aref[r]) for r in rows],
        final_state_histogram={str(int(s)): int(sum(int(state[r]) == s for r in rows)) for s in sorted(set(map(int, state[rows])))},
        state_mask_applied=False, actual_solver_matrix_reconstructed=False)


def grouped_proxies(active):
    import numpy as np
    table = {k: active["contacts/" + k] for k in saved.p.p.CONTACT_FIELDS}
    selections = {(w, 1, -1, -1, 0, False): list(map(int, np.flatnonzero(active[f"rows/{w}/type"] == 1))) for w in range(2)}
    for key, slots in saved.p.p.contact_groups(table).items():
        w, g0, g1, dim, included = key
        need(dim == 3 and included, "included dim3 aggregate bins")
        addresses = [int(table["efc_address"][s]) for s in slots]
        for slot, address in zip(slots, addresses):
            need(0 <= address <= len(active[f"rows/{w}/type"]) - 4
                 and (active[f"rows/{w}/type"][address:address + 4] == 6).all()
                 and (active[f"rows/{w}/id"][address:address + 4] == int(table["slot"][slot])).all(),
                 "own-contact type/id/address linkage for every aggregate row")
        selections[(w, 6, g0, g1, dim, included)] = sorted(r for a in addresses for r in range(a, a + 4))
    result, covered = [], [[], []]
    for key, rows in sorted(selections.items()):
        w = key[0]; covered[w].extend(rows)
        args = [active[f"rows/{w}/{k}"] for k in ("J", "D", "aref", "state")]
        result.append(dict(key=list(key), nominal=nominal_terms(*args, rows),
            physical_point_correspondence_established=False))
    need(all(sorted(covered[w]) == list(range(len(active[f"rows/{w}/type"]))) for w in range(2)), "exact disjoint full active-row partition")
    return result


def proxy_difference(native, gpu):
    a, b = ({tuple(x["key"]): x["nominal"] for x in side} for side in (native, gpu))
    need(set(a) == set(b), "same aggregate bins, not point sets")
    result = []
    for key in sorted(a):
        left, right = a[key], b[key]
        H = [[right["H_JtDJ"][i][j] - left["H_JtDJ"][i][j] for j in range(20)] for i in range(20)]
        h = [r - l for r, l in zip(right["h_JtDaref"], left["h_JtDaref"])]
        result.append(dict(key=list(key), native_rows=len(left["rows"]), gpu_rows=len(right["rows"]),
            gpu_minus_native_H_JtDJ=H, gpu_minus_native_h_JtDaref=h,
            gpu_minus_native_c=right["c_half_aref_D_aref"] - left["c_half_aref_D_aref"],
            physical_point_correspondence_established=False, cause_isolated=False))
    return result


def analyze(input_root, gpu_root):
    import numpy as np
    exact = Path(gpu_root) / "report.json"
    need(exact.is_file() and not exact.is_symlink() and exact.stat().st_size <= 256 * 1024
         and sha256(exact.read_bytes()).hexdigest() == GPU_REPORT_SHA, "anchored exact new measured GPU report")
    gpu, arrays, layout, manifest, _ = measured.receive(gpu_root, GPU_SOURCE, input_root)
    prior, banks = saved.p.prior.authenticated_banks(input_root)
    models = bound_model_values(prior, banks, gpu["child"])
    reference = packet.measured_reference_banks(arrays, layout, prior, banks)
    decoded = response.decode_stages(arrays, layout, prior, reference)
    before, after = decoded["before_solve"], decoded["after_solve"]
    for k in ("J", "D", "aref"):
        need(before["/data/efc/" + k].tobytes() == after["/data/efc/" + k].tobytes(), "GPU nominal inputs unchanged by solver")
    active = response.active_response(after, prior["child"]["plant"])
    saved.p.p.analyze_active(banks["cpu-active.npz"], dict(active, **{
        "sidecar/type": response.stage_candidates(after)["type"],
        "sidecar/geomcollisionid": response.stage_candidates(after)["geomcollisionid"]}),
        prior["child"]["plant"]["dofs"], prior["child"]["plant"]["floor"], prior["child"]["plant"]["feet"])
    recipes = gpu_row_recipes(before, models, manifest)
    native, current = (grouped_proxies(v) for v in (banks["cpu-active.npz"], active))
    return dict(protocol=PROTOCOL, decision=DECISION, predecessor_source=saved.p.prior.SOURCE,
        predecessor_report_sha256=saved.p.prior.REPORT_SHA256, predecessor_files=prior["files"],
        gpu_source=GPU_SOURCE, gpu_report_sha256=GPU_REPORT_SHA, gpu_files=gpu["files"],
        all347_model_array_sha256=gpu["child"]["model_array_sha256"],
        model_input_layout={k: response.expanded_layout(prior["child"]["input_manifest"][k]) for k in MODEL_INPUTS},
        numeric_source=numeric_source(), stage_source_audit_sha256=packet.digest(response.source.audit()),
        data_source="Only new measured114-field GPU stages; no historical Data used for GPU recipes",
        native_stage="Retained native post-solve active rows; native scalar inputs unavailable",
        gpu_stage="New measured before-solve J/D/aref, byte-unchanged afterward; final state is descriptive only",
        arithmetic="Ordered Python float64 sums/products over original row indices; no BLAS, RMS, tolerance or normalization",
        gpu_contact_row_recipes=recipes, native_nominal_proxies=native, gpu_nominal_proxies=current,
        nominal_proxy_differences=proxy_difference(native, current),
        recipe_max_abs={k: max(abs(r["recipe_minus_captured"][k]) for r in recipes) for k in recipes[0]["captured"]},
        **CONTRACT)


def receive(path, source, input_root, gpu_root):
    source_check(source); path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 * 1024, "bounded plain arithmetic result")
    value = json.loads(path.read_bytes()); expected = analyze(input_root, gpu_root)
    expected.update(source=source, module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    need(packet.digest(value) == packet.digest(expected), "every complete exact arithmetic result recomputed")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "measured", "output"): parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source", required=True); args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026, 10, 9, 22, 50, tzinfo=timezone.utc), "CPU-hidden arithmetic before closeout reserve")
    source_check(args.source)
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(), "fresh canonical arithmetic output")
    value = analyze(args.input, args.measured)
    value.update(source=args.source, module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    source_check(args.source); host.write_json(args.output, value); receive(args.output, args.source, args.input, args.measured)
    print(DECISION, value["recipe_max_abs"])


if __name__ == "__main__": main()
