"""CPU-only arithmetic diagnosis of retained Ada contacts; never admission.

No collision, solver, integration or device call. Aggregate by ordered geometry
key without inventing point correspondence. Evaluate J.T @ force in float64;
its residual against a stored float32 sum is descriptive, not a new tolerance.
"""
import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re

from mjlab_microduck import ada_duck_contact as p

PROTOCOL = "microduck-ada-contact-arithmetic-diagnosis-v1"
SOURCE = "0ce8c9d0a8203f5645f6dcd9ad02d04f18a173ac"
REPORT_SHA256 = "5fa41d7edeb79cffee87093e1255367c2e01af1b7ca6c468f104aab5ae408aa9"
need = p.need


def authenticated_banks(root):
    """Authenticate the frozen predecessor, then decode those exact bytes."""
    import numpy as np
    root = Path(root)
    need(root.is_dir() and not root.is_symlink(), "plain retained evidence directory")
    expected = set(p.FILES) | {"report.json"}
    need({x.name for x in root.iterdir()} in (expected, expected | {"private-cache"}), "complete retained inventory")
    raw = {}
    for name in expected:
        file = root / name
        limit = 256 * 1024 if name.endswith(".json") else 16 * 1024**2
        need(file.is_file() and not file.is_symlink() and file.stat().st_size <= limit, "plain bounded evidence file")
        raw[name] = file.read_bytes()
    need(sha256(raw["report.json"]).hexdigest() == REPORT_SHA256, "exact predeclared predecessor report bytes")
    report = json.loads(raw["report.json"])
    need(report["source"]["commit"] == SOURCE and report["decision"] == p.COLLECTION_DECISION
         and report["specification"] == p.specification() and report["bounds"] == p.BOUNDS
         and report["child_exit"] == 0 and set(report["files"]) == set(p.FILES), "same frozen diagnostic")
    need(all(sha256(raw[name]).hexdigest() == digest for name, digest in report["files"].items()), "every manifest file authenticated")
    child = json.loads(raw["child.json"])
    need(child == report["child"], "separate child/report binding")
    p.receive_payloads(root, child)  # Full predecessor schema/semantics, still CPU-only.
    banks = {}
    for item in child["payloads"]:
        payload = raw[item["file"]]
        need(len(payload) == item["bytes"] and sha256(payload).hexdigest() == item["sha256"], "same authenticated decoding bytes")
        with np.load(io.BytesIO(payload), allow_pickle=False) as bank:
            need(len(bank.files) == len(set(bank.files)), "unique serialized field names")
            banks[item["file"]] = {k: bank[k].copy() for k in bank.files}
    return report, banks


def resultants(active):
    """All reported forces/torques transformed as row axes, about world origin.

    No sign flip, nearest-point matching, averaging or manifold normalization.
    The wrench is a diagnostic of the retained frame convention, not net force
    on the entire robot (which also has gravity, motors and inertia).
    """
    import numpy as np
    table = {k: active["contacts/" + k] for k in p.CONTACT_FIELDS}
    groups = p.contact_groups(table)
    rows = []
    for key, indices in sorted(groups.items()):
        pos = table["pos"][indices].astype(np.float64)
        frame = table["frame"][indices].astype(np.float64)
        force = table["force"][indices].astype(np.float64)
        world_force = np.einsum("ni,nij->nj", force[:, :3], frame)
        world_torque = np.einsum("ni,nij->nj", force[:, 3:], frame) + np.cross(pos, world_force)
        distances = [float(np.linalg.norm(pos[i] - pos[j])) for i in range(len(pos)) for j in range(i)]
        rows.append(dict(key=list(key), raw_slots=table["slot"][indices].tolist(), count=len(indices),
                         exact_distinct_positions=len({x.tobytes() for x in pos}),
                         minimum_pairwise_position_distance_m=min(distances) if distances else None,
                         penetration_distances_m=table["dist"][indices].tolist(),
                         frame_orthonormality_max_abs=float(np.max(np.abs(frame @ frame.transpose(0, 2, 1) - np.eye(3)))),
                         world_force_N=world_force.sum(0).tolist(),
                         world_torque_about_origin_Nm=world_torque.sum(0).tolist()))
    return rows


def reconstruction(active, fields):
    import numpy as np
    result = []
    for world in range(2):
        J = active[f"rows/{world}/J"].astype(np.float64)
        force = active[f"rows/{world}/force"].astype(np.float64)
        kinds = active[f"rows/{world}/type"]
        friction = J[kinds == 1].T @ force[kinds == 1]
        contact = J[kinds == 6].T @ force[kinds == 6]
        total = J.T @ force
        stored = fields["qfrc_constraint"][world].astype(np.float64)
        need(all(np.isfinite(x).all() for x in (friction, contact, total, stored)), "finite full generalized arithmetic")
        result.append(dict(world=world, friction_generalized_force=friction.tolist(),
                           contact_generalized_force=contact.tolist(), total_generalized_force=total.tolist(),
                           stored_constraint_generalized_force=stored.tolist(),
                           reconstructed_minus_stored=(total - stored).tolist(),
                           max_abs=float(np.max(np.abs(total - stored)))))
    return result


def diagnose(root):
    import numpy as np
    report, banks = authenticated_banks(root)
    p.analyze_active(banks["cpu-active.npz"], banks["gpu-active.npz"], report["child"]["plant"]["dofs"],
                     report["child"]["plant"]["floor"], report["child"]["plant"]["feet"])
    sides = {}
    for side in ("cpu", "gpu"):
        active, fields = banks[side + "-active.npz"], banks[side + "-fields.npz"]
        sides[side] = dict(resultants=resultants(active), generalized=reconstruction(active, fields))
    cross = []
    a = {tuple(x["key"]): x for x in sides["cpu"]["resultants"]}
    b = {tuple(x["key"]): x for x in sides["gpu"]["resultants"]}
    for key in sorted(a.keys() | b.keys()):
        left, right = a.get(key), b.get(key)
        delta = None if left is None or right is None else {
            name: (np.asarray(right[name]) - np.asarray(left[name])).tolist()
            for name in ("world_force_N", "world_torque_about_origin_Nm")}
        cross.append(dict(key=list(key), cpu_count=0 if left is None else left["count"],
                          gpu_count=0 if right is None else right["count"], gpu_minus_cpu=delta,
                          point_correspondence_established=False))
    return dict(protocol=PROTOCOL, decision="retained-contact-arithmetic-diagnosis-not-admission",
                execution_source=SOURCE, report_sha256=REPORT_SHA256,
                input_file_sha256=report["files"], sides=sides, aggregate_differences=cross,
                difference_convention="GPU minus CPU; sums, never averaged or normalized by contact count",
                reconstruction_precision="float64 arithmetic over all retained active rows; no acceptance tolerance",
                source_flags=p.FLAGS, physical_contact_identity_established=False, solver_qualified=False,
                new_collision_calls=0, new_solver_calls=0, new_integration_steps=0,
                training_authorized=False, physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden diagnosis")
    need(re.fullmatch(r"[0-9a-f]{40}", args.source) and Path.cwd().resolve() == Path(__file__).resolve().parents[2]
         and p.base.host.read("git", "rev-parse", "HEAD") == args.source
         and p.base.host.read("git", "branch", "--show-current") == p.base.host.BRANCH
         and not p.base.host.read("git", "status", "--porcelain"), "clean exact diagnosis source")
    module = Path(__file__).read_bytes()
    need(module == p.base.host.read("git", "show", args.source + ":src/mjlab_microduck/ada_contact_diagnosis.py", binary=True),
         "executed committed diagnostic bytes")
    result = diagnose(args.input)
    result.update(diagnosis_source=args.source, diagnostic_module_sha256=sha256(module).hexdigest())
    p.base.host.write_json(args.output, result)  # Exclusive-create, finite bounded JSON.
    print(json.dumps(dict(decision=result["decision"], aggregate_differences=result["aggregate_differences"]), sort_keys=True))


if __name__ == "__main__": main()
