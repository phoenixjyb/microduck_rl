"""Authenticated contact-generator metadata audit; no physics or admission.

Exact generator labels align rows, not physical points. Positions are never a
join key. The collision-only CPU and post-solve Ada records are different stages.
"""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import re

from mjlab_microduck import ada_collision_replay as replay

need = replay.need
prior = replay.prior
PROTOCOL = "microduck-ada-contact-generator-metadata-audit-v1"
REPLAY_SOURCE = "2d0da53fd8ab4e6b9aed366a3ff925a1b7c2a0a4"
REPLAY_SHA256 = "00a4e2bbd7c43a1376c97883ce274e3640b2c395a50f751561c96c5d5f8ad45b"
REPLAY_BANK_SHA256 = "c85afc7f17721c6b4d7343df587e0cb22b744261afdeb54f7623f957d93c93a1"
KEY = ("worldid", "geom0", "geom1", "dim", "type", "geomcollisionid")
FLOAT_FIELDS = dict(dist=(), pos=(3,), frame=(3, 3), friction=(5,))
INT_FIELDS = dict(worldid=(), slot=(), geom=(2,), dim=(), type=(), geomcollisionid=())


def metadata_index(table):
    """Require a complete unique label index; never assign spatial identity."""
    import numpy as np
    need(set(table) == set(INT_FIELDS) | set(FLOAT_FIELDS), "complete comparison fields")
    n = len(table["worldid"])
    need(0 < n <= 256, "bounded nonempty metadata table")
    for name, tail in (INT_FIELDS | FLOAT_FIELDS).items():
        value = table[name]
        need(value.shape == (n,) + tail and value.dtype == (np.int32 if name in INT_FIELDS else np.float32)
             and np.isfinite(value).all(), "exact finite metadata table layout: " + name)
    need(((table["worldid"] >= 0) & (table["worldid"] < 2)).all()
         and (table["geom"] >= 0).all() and (table["dim"] == 3).all()
         and (table["type"] >= 0).all() and (table["geomcollisionid"] >= 0).all()
         and sorted(table["slot"].tolist()) == list(range(n)), "valid fixture labels and raw slots")
    index = {}
    for i in range(n):
        key = (int(table["worldid"][i]), *map(int, table["geom"][i]), int(table["dim"][i]),
               int(table["type"][i]), int(table["geomcollisionid"][i]))
        need(key not in index, "unique complete generator metadata keys")
        index[key] = i
    return index


def field_difference(left, right):
    """Ada minus CPU in float64 over float32 inputs; exact bits, no tolerance."""
    import numpy as np
    need(left.shape == right.shape and left.dtype == right.dtype == np.float32
         and np.isfinite(left).all() and np.isfinite(right).all(), "complete finite float32 field")
    delta = right.astype(np.float64) - left.astype(np.float64)
    return dict(ada_minus_cpu=delta.tolist(), max_abs=float(np.abs(delta).max()),
                float32_bit_mismatches=int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))),
                bytes_equal=left.tobytes() == right.tobytes())


def compare(cpu, ada):
    """Full key-set equality is mandatory; slots merely preserve provenance."""
    a, b = metadata_index(cpu), metadata_index(ada)
    need(a.keys() == b.keys(), "exact complete generator metadata key set")
    rows = []
    for key in sorted(a):
        i, j = a[key], b[key]
        fields = {name: field_difference(cpu[name][i], ada[name][j]) for name in FLOAT_FIELDS}
        rows.append(dict(generator_key=list(key), cpu_raw_slot=int(cpu["slot"][i]),
                         ada_raw_slot=int(ada["slot"][j]), common_fields=fields,
                         physical_point_identity_established=False))
    return dict(key_fields=list(KEY), row_count=len(rows), key_sets_equal=True, keys_unique=True,
                rows=rows, maximum_common_field_residual={name: max(x["common_fields"][name]["max_abs"] for x in rows)
                                                          for name in FLOAT_FIELDS},
                physical_point_identity_established=False,
                alignment_convention="Exact generator metadata only; no spatial pairing or tolerance")


def audit(root, collision_json):
    import numpy as np
    collision_json = Path(collision_json)
    need(collision_json.is_file() and not collision_json.is_symlink()
         and collision_json.stat().st_size <= 256 * 1024
         and sha256(collision_json.read_bytes()).hexdigest() == REPLAY_SHA256, "fixed collision replay JSON bytes")
    receipt, arrays = replay.receive(collision_json, REPLAY_SOURCE, root)
    need(receipt["payload"]["sha256"] == REPLAY_BANK_SHA256, "fixed complete collision replay bank")
    report, banks = prior.authenticated_banks(root)
    active = banks["gpu-active.npz"]
    cpu = {name: arrays["warp_cpu/contact/" + name] for name in (set(INT_FIELDS) | set(FLOAT_FIELDS)) - {"slot"}}
    cpu["slot"] = np.arange(len(cpu["worldid"]), dtype=np.int32)
    ada = {name: active[("sidecar/" if name in ("type", "geomcollisionid") else "contacts/") + name]
           for name in set(INT_FIELDS) | set(FLOAT_FIELDS)}
    comparison = compare(cpu, ada)
    poses = {}
    for name in replay.KINEMATIC:
        current = arrays["warp_cpu/pose/" + name]
        saved = np.frombuffer(banks["prepared-inputs.npz"]["/data/" + name].tobytes(), dtype=np.float32).reshape(current.shape)
        residual = field_difference(current, saved)
        # Complete pose arrays already exist in authenticated input banks. Keep
        # hashes and bit counts here rather than duplicating all pose numbers.
        residual.pop("ada_minus_cpu")
        poses[name] = dict(residual, cpu_array_sha256=sha256(current.tobytes()).hexdigest(),
                           ada_array_sha256=sha256(saved.tobytes()).hexdigest(), shape=list(current.shape))
    margin = arrays["warp_cpu/contact/dist"] < arrays["warp_cpu/contact/includemargin"]
    return dict(protocol=PROTOCOL, decision="generator-metadata-diagnostic-not-admission",
                replay_source=REPLAY_SOURCE, replay_json_sha256=REPLAY_SHA256,
                replay_bank_sha256=REPLAY_BANK_SHA256, predecessor_source=prior.SOURCE,
                predecessor_report_sha256=prior.REPORT_SHA256, predecessor_file_sha256=report["files"],
                comparison=comparison, complete_actual_pose_fields=poses,
                actual_pose_bytes_equal=all(x["bytes_equal"] for x in poses.values()),
                stage_binding=dict(cpu="collision-only, before constraint construction",
                                   ada="retained complete motor/contact forward, after solve",
                                   cpu_efc_addresses=arrays["warp_cpu/contact/efc_address"].tolist(),
                                   ada_efc_addresses=active["contacts/efc_address"].tolist(),
                                   cpu_inside_includemargin_raw_slots=np.flatnonzero(margin).tolist(),
                                   cpu_constraint_type_bit_raw_slots=np.flatnonzero((cpu["type"] & 1) != 0).tolist(),
                                   ada_includemargin_retained=False,
                                   common_float_fields=list(FLOAT_FIELDS),
                                   force_or_efc_address_compared_as_common_fields=False),
                native_complete_generator_metadata_available=False,
                compiled_generator_execution_identity_established=False,
                same_pose_collision_response_isolated=False, solver_qualified=False,
                new_collision_calls=0, new_solver_calls=0, new_integration_steps=0,
                flags=replay.p.FLAGS, training_authorized=False, physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-hidden metadata audit")
    host = replay.p.base.host
    need(re.fullmatch(r"[0-9a-f]{40}", args.source)
         and Path.cwd().resolve() == Path(__file__).resolve().parents[2]
         and host.read("git", "rev-parse", "HEAD") == args.source
         and host.read("git", "branch", "--show-current") == host.BRANCH
         and not host.read("git", "status", "--porcelain"), "clean exact metadata audit source")
    raw = Path(__file__).read_bytes()
    need(raw == host.read("git", "show", args.source + ":src/mjlab_microduck/ada_contact_metadata_audit.py", binary=True),
         "committed metadata audit module bytes")
    result = audit(args.input, args.replay)
    result.update(audit_source=args.source, audit_module_sha256=sha256(raw).hexdigest())
    host.write_json(args.output, result)
    print(result["decision"], result["comparison"]["row_count"], result["actual_pose_bytes_equal"])


if __name__ == "__main__": main()
