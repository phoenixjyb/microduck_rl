"""Pure native-row-family hybrid packet, not a solver launch or admission.

The complete measured GPU surrounding context is held. Native own contact
slots are concatenated, never paired with GPU points. Synthetic collision
metadata prohibit sending this packet to collision/construction/sensors.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import zipfile

from mjlab_microduck import ada_native_constraint_recipe as recipe

need, host, packet, response = recipe.need, recipe.host, recipe.audit.packet, recipe.audit.response
PROTOCOL = "microduck-current-native-row-family-packet-v1"
DECISION = "native-row-family-hybrid-packet-not-solver-execution-or-admission"
RECIPE_SOURCE = "ec81e2bb2ffd25f6e847036cb0deb1ab1836478d"
RECIPE_SHA = "674c6d7779bad3daef7c54aa761745d26df03cf44051753811f3dfcaf05490ae"
COUNTS = ("ne", "nf", "nl", "nefc", "nacon")
EFC = ("J", "D", "aref", "frictionloss", "type", "id", "pos", "margin", "vel", "Jqvel")
CONTACT = tuple(response.saved.CONTACT_TAILS)
OVERRIDES = tuple("/data/" + k for k in COUNTS) + tuple("/data/efc/" + k for k in EFC) + tuple("/data/contact/" + k for k in CONTACT)
CONTRACT = dict(runtime_imported=False, new_model_data_allocations=0, new_forward_calls=0,
    new_collision_calls=0, new_constraint_calls=0, new_solver_calls=0, integration_steps=0,
    optimizer_steps=0, encoded_complete_data_fields=114, overridden_data_fields=31,
    held_data_fields=83, native_capture_is_before_solve=False, native_phase_replay_executed=False,
    native_final_solver_outputs_transplanted=False, physical_contact_point_matching=False,
    native_end_to_end_context_identity=False, collision_metadata_regenerated=False,
    synthetic_collision_metadata=True, actual_solver_read_set_proved=False,
    subsequent_solver_execution_authorized=False, exclusive_cause_established=False,
    loaded_binary_identity=False, solver_qualified=False, simulator_qualified=False,
    training_authorized=False, physical_motion_authorized=False, flags=host.FLAGS)


def full_bank(raw, layout, prior):
    """Decode every source leaf only after its complete canonical layout check."""
    import numpy as np
    expected = {k:v for k,v in prior["child"]["input_manifest"].items() if k.startswith("/data/")}
    need(len(expected) == 114 and set(raw) == set(layout) == set(expected) and set(OVERRIDES) <= set(raw), "complete114-field canonical hybrid bank")
    result = {}
    for k, row in expected.items():
        need(layout[k] == response.expanded_layout(row), "exact complete hybrid layout: " + k)
        value = raw[k]
        need(value.dtype == np.uint8 and value.shape == (row["bytes"],), "complete raw hybrid leaf: " + k)
        result[k] = np.frombuffer(value.tobytes(), layout[k]["numpy_dtype"]).reshape(layout[k]["numpy_shape"]).copy()
        need(np.isfinite(result[k]).all(), "finite complete hybrid leaf: " + k)
    need(sum(v.nbytes for v in result.values()) < 2 * 1024**2, "bounded full114-field hybrid packet")
    return result


def encode(arrays, capture, before, layout, prior):
    """All31 declared replacements, all83 held leaves; no runtime operations.

    This is a joint input intervention. Native post-solve force/state/b are
    never inputs. Dense sparse metadata remain the authentic empty GPU leaves.
    """
    import numpy as np
    original = full_bank(before, layout, prior)
    decoded = {k:v.copy() for k,v in original.items()}
    native_rows, _ = recipe.native_recipes(arrays, capture)
    need(len(native_rows) == 44 and len(set(OVERRIDES)) == 31 and not set(OVERRIDES) & response.SOLVER_OUTPUTS, "closed input family excludes all native solver outputs")
    expected_counts = ((0,0,14,0,14), (4,0,14,0,30))
    for w in range(2):
        counters = arrays[f"data/{w}/counters"]
        need(counters.dtype == np.int64 and counters.shape == (len(recipe.native.COUNTERS),), "complete native int64 counter source")
        count = dict(zip(recipe.native.COUNTERS, map(int, counters)))
        need(tuple(count[k] for k in ("ncon", "ne", "nf", "nl", "nefc")) == expected_counts[w], "fixed closed native count family")
        for name in COUNTS[:-1]:
            target = decoded["/data/" + name]
            need(target.dtype == np.int32 and target.shape == (2,), "exact native count target")
            target[w] = count[name]
    target = decoded["/data/nacon"]
    need(target.dtype == np.int32 and target.shape == (1,), "exact native global contact target")
    target[:] = 4
    ints = {"type", "id"}
    for name in EFC:
        target = decoded["/data/efc/" + name]
        need(target.dtype == (np.int32 if name in ints else np.float32) and target.shape == ((2,512,20) if name == "J" else (2,512)), "exact native dense EFC target: " + name)
        target.fill(0)
        native_name = "vel" if name == "Jqvel" else name
        for w, (_,_,_,_,nefc) in enumerate(expected_counts):
            source = arrays[f"efc/{w}/" + native_name]
            need(source.dtype == (np.int32 if name in ints else np.float64) and source.shape == ((nefc*20,) if name == "J" else (nefc,)) and np.isfinite(source).all(), "complete native EFC source: " + name)
            value = source.reshape((nefc,20) if name == "J" else (nefc,)).astype(target.dtype)
            need(np.isfinite(value).all(), "finite explicit native EFC cast")
            target[w,:nefc] = value
    # Explicit own global concatenation: world0 contributes no contact slots;
    # world1 contributes slots0..3. This is NOT GPU point correspondence.
    for name, tail in response.saved.CONTACT_TAILS.items():
        target = decoded["/data/contact/" + name]
        integer = name in {"dim","geom","flex","vert","efc_address","worldid","type","geomcollisionid"}
        need(target.shape == (256,) + tail and target.dtype == (np.int32 if integer else np.float32), "exact full contact target: " + name)
        target.fill(-1 if name in {"geom","flex","vert","efc_address","worldid","geomcollisionid"} else 0)
        if name == "worldid": target[:4] = 1
        elif name == "type": target[:4] = 1  # frozen ContactType.CONSTRAINT
        elif name == "geomcollisionid": pass  # synthetic -1, no collision provenance
        elif name == "efc_address":
            address = arrays["contact/1/efc_address"]
            need(address.dtype == np.int32 and address.tolist() == [14,18,22,26], "own complete native address partition")
            target[:4] = address[:,None] + np.arange(4,dtype=np.int32)
        else:
            source = arrays["contact/1/" + name]
            need(source.shape == (4,) + tail and source.dtype == (np.int32 if integer else np.float64) and np.isfinite(source).all(), "complete native contact source: " + name)
            value = source.astype(target.dtype)
            need(np.isfinite(value).all(), "finite explicit native contact cast")
            target[:4] = value
    # Every own native DOF and contact row remains covered exactly once.
    links = []
    for w, (_,_,_,_,nefc) in enumerate(expected_counts):
        kinds, ids = decoded["/data/efc/type"][w], decoded["/data/efc/id"][w]
        need(kinds[:14].tolist() == [1]*14 and ids[:14].tolist() == list(range(6,20)) and kinds[14:nefc].tolist() == [6]*(nefc-14), "fixed native DOF friction and dim3 pyramid partition")
        for slot in range(4 if w else 0):
            address = decoded["/data/contact/efc_address"][slot]
            need(address.tolist() == list(range(14+4*slot,18+4*slot)) and ids[address].tolist() == [slot]*4 and decoded["/data/contact/dim"][slot] == 3 and decoded["/data/contact/worldid"][slot] == w, "own complete global contact row linkage")
            need(decoded["/data/contact/geom"][slot].tolist() in ([0,29],[0,79]) and (decoded["/data/contact/flex"][slot] == -1).all() and (decoded["/data/contact/vert"][slot] == -1).all() and decoded["/data/contact/dist"][slot] < decoded["/data/contact/includemargin"][slot], "included rigid own foot contact, no flex")
            links.append(dict(world=w,native_slot=slot,hybrid_global_slot=slot,geom=decoded["/data/contact/geom"][slot].tolist(),rows=address.tolist()))
        need((decoded["/data/efc/D"][w,:nefc] > 0).all() and (decoded["/data/efc/frictionloss"][w,:nefc] >= 0).all(), "positive native active D and nonnegative friction loss")
    encoded = {k:np.frombuffer(v.tobytes(),np.uint8).copy() for k,v in decoded.items()}
    full_bank(encoded,layout,prior)
    held = set(encoded) - set(OVERRIDES)
    response.exact_values({k:encoded[k] for k in held},{k:before[k] for k in held}, "all83 held measured GPU context leaves")
    leaves = {k:dict(policy="native-family-override" if k in OVERRIDES else "held-measured-gpu-context", bytes=v.nbytes, before_sha256=sha256(before[k].tobytes()).hexdigest(), encoded_sha256=sha256(v.tobytes()).hexdigest(), bytes_equal=v.tobytes()==before[k].tobytes()) for k,v in encoded.items()}
    return encoded, dict(data_layout=layout,data_leaves=leaves,own_contact_links=links,
        counts=dict(native_ncon=[0,4],ne=[0,0],nf=[14,14],nl=[0,0],nefc=[14,30],nacon=[4]),
        declared_override_fields=sorted(OVERRIDES),held_context_fields=sorted(held),
        changed_fields=sorted(k for k,v in encoded.items() if v.tobytes()!=before[k].tobytes()),
        cast="Explicit NumPy float64-to-float32; not FP32 constructor emulation",
        padding="Zero all overridden EFC tails; inactive contact indices/address/flex/vert/collisionid=-1; other contact tails=0",
        synthetic_fields={"/data/contact/type":"1=CONSTRAINT for four active rigid slots;0 inactive", "/data/contact/geomcollisionid":"-1 throughout; no generated collision provenance"},
        prohibited_consumers=["collision","constraint construction","sensor","force decoder","ordinary forward","integration","optimizer"])


def analyze(input_root, measured_root, native_path, c_source, recipe_path):
    need(not any(n in sys.modules for n in ("mujoco","mujoco_warp","warp","torch")), "fresh pure hybrid encoder")
    recipe_path = Path(recipe_path)
    need(recipe_path.is_file() and not recipe_path.is_symlink() and recipe_path.stat().st_size == 79729 and sha256(recipe_path.read_bytes()).hexdigest() == RECIPE_SHA, "exact accepted descriptive native recipe report")
    expected = recipe.analyze(input_root,measured_root,native_path,c_source)
    expected.update(source=RECIPE_SOURCE,module_sha256=sha256(Path(recipe.__file__).read_bytes()).hexdigest())
    need(packet.digest(expected) == packet.digest(json.loads(recipe_path.read_bytes())), "complete native arithmetic recomputation before encoding")
    capture, arrays = recipe.native.receive(native_path,recipe.NATIVE_SOURCE,input_root)
    gpu, raw, layout, static, _ = recipe.audit.measured.receive(measured_root,recipe.audit.GPU_SOURCE,input_root)
    prior, _ = recipe.native.prior.authenticated_banks(input_root)
    before = {k:raw["before_solve"+k] for k in layout}
    encoded, description = encode(arrays,capture["capture"],before,layout,prior)
    need(not any(n in sys.modules for n in ("mujoco","mujoco_warp","warp","torch")), "hybrid encoding remained runtime-inert")
    result = dict(protocol=PROTOCOL,decision=DECISION,native_report_sha256=recipe.NATIVE_REPORT_SHA,
        native_recipe_report_sha256=RECIPE_SHA,measured_gpu_report_sha256=recipe.audit.GPU_REPORT_SHA,
        original_report_sha256=recipe.native.prior.REPORT_SHA256,original_files=prior["files"],measured_gpu_files=gpu["files"],
        measured_model_array_sha256=gpu["child"]["model_array_sha256"],measured_static_manifest_sha256=packet.digest(static),
        source_audit_sha256=expected["gpu_stage_source_audit_sha256"],**description,**CONTRACT)
    return result, encoded


def source_check(source):
    recipe.source_check(source)
    need(Path(__file__).read_bytes() == host.read("git","show",source+":src/mjlab_microduck/ada_native_row_family_plan.py",binary=True), "exact committed pure hybrid encoder")


def receive(path, source, input_root, measured_root, native_path, c_source, recipe_path):
    import numpy as np
    source_check(source); path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256*1024, "plain bounded hybrid report")
    actual = json.loads(path.read_bytes()); expected, encoded = analyze(input_root,measured_root,native_path,c_source,recipe_path)
    expected.update(source=source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    payload_path = path.with_suffix(".npz"); payload = actual["payload"]
    need(payload_path.is_file() and not payload_path.is_symlink() and payload_path.stat().st_size < 2*1024**2, "plain bounded full hybrid bank")
    raw = payload_path.read_bytes()
    need(payload == dict(file=payload_path.name,bytes=len(raw),sha256=sha256(raw).hexdigest()), "exact full hybrid payload binding")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        need(len(entries) == len({v.filename for v in entries}) == 114 and all(0 <= v.file_size < 2*1024**2 for v in entries) and sum(v.file_size for v in entries) < 2*1024**2+114*1024, "bounded complete nonduplicate hybrid archive")
    with np.load(io.BytesIO(raw),allow_pickle=False) as bank:
        need(len(bank.files) == 114 and set(bank.files) == set(encoded), "complete114-field hybrid archive inventory")
        response.exact_values({k:bank[k] for k in bank.files},encoded,"all114 independently regenerated hybrid leaves")
    expected["payload"] = payload
    need(packet.digest(actual) == packet.digest(expected), "every hybrid descriptor independently regenerated")
    return actual, encoded


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input","measured","native","c-source","recipe","output"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source",required=True); args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and datetime.now(timezone.utc) < datetime(2026,10,9,22,50,tzinfo=timezone.utc), "CPU-hidden hybrid packet before closeout reserve")
    need(all(os.environ.get(n+"_NUM_THREADS") == "1" for n in ("OMP","MKL","OPENBLAS","NUMEXPR")), "four literal single-thread CPU settings")
    need(args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists() and not args.output.with_suffix(".npz").exists(), "fresh canonical hybrid packet prefix")
    source_check(args.source)
    result, arrays = analyze(args.input,args.measured,args.native,args.c_source,args.recipe)
    source_check(args.source); path = args.output.with_suffix(".npz")
    with path.open("xb") as stream: np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
    raw = path.read_bytes(); need(len(raw) < 2*1024**2, "bounded complete hybrid compressed bank")
    result.update(source=args.source,module_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),payload=dict(file=path.name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    host.write_json(args.output,result)
    receive(args.output,args.source,args.input,args.measured,args.native,args.c_source,args.recipe)
    print(DECISION,len(OVERRIDES),len(result["held_context_fields"]),result["counts"])


if __name__ == "__main__": main()
