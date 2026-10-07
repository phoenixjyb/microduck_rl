"""Independent CPU whole-byte receiver for a single nominal serial physics tick.

This authenticates bounded retained data; it does not reexecute BAM/physics,
qualify the historical full window, or admit learning or physical motion.
"""

from hashlib import sha256
import math
import os
from pathlib import Path
import re
import stat

import numpy as np
from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_com_coupled_receiver as coupled
from mjlab_microduck import stance_com_entry_receiver as old
from mjlab_microduck import stance_rne_coupled_receiver as prior
from mjlab_microduck import stance_serial_step_receiver as step_receiver
from mjlab_microduck import stance_serial_step_divergence as divergence

PROTOCOL = "microduck-bam-load-receiver-oct7-v1"
DATA_PROTOCOL = "microduck-bam-load-probe-oct7-v1"
SERVICE_CAPS = {**old.SERVICE_CAPS, "LimitFSIZE": "2097152"}
TICK_ANALYSIS_SHA256 = (
    "5c5d3959acfacea057bc3729a4f0a0270544e47189a7e8b164355f116fadf5b8"
)
TICK_PREDECESSOR = dict(
    source="26e2ee8244b7bd05fb3290d655113e538bd1a989",
    decision="fresh-one-step-serial-repeat-negative",
    analysis_sha256=TICK_ANALYSIS_SHA256,
    first_observed_stream="motor",
    first_observed_substep=2,
    first_observed_field="committed.friction",
    full_window_qualified=False,
)
EXPECTED_TESTS = 2322
TEST_FILES_SHA256 = "4793266dc5bde8c0bb5da8dfa626c699f4d80034060707212a91cc989c33d6bd"
DESCRIPTOR_SHA256 = prior.DESCRIPTOR_SHA256
SELECTED_FIELDS_SHA256 = prior.SELECTED_FIELDS_SHA256
DESCRIPTOR_FIXTURE = dict(prior.DESCRIPTOR_FIXTURE)
RETAINED_TEST_FIXTURES = dict(prior.RETAINED_TEST_FIXTURES)
PHASE_A, PHASE_A_SOURCE, PHASE_A_TERMINAL = (
    prior.PHASE_A,
    prior.PHASE_A_SOURCE,
    prior.PHASE_A_TERMINAL,
)
PREDECESSORS, COM_PREDECESSOR = prior.PREDECESSORS, prior.COM_PREDECESSOR
PREDECESSOR_FILES = {
    "inventory.json": {
        "bytes": 3429,
        "sha256": "822da5698dac708b59597728ace88543a1c58a3ca982f5a9a087327bfe051a59",
    },
    "terminal.json": {
        "bytes": 390,
        "sha256": "d756fcca0d879e22c5f0d8dbed18cfca86762b8d4f61afce999822847a7f95d8",
    },
    "receiver.json": {
        "bytes": 104815,
        "sha256": "f7b1eb1a2aa16c24274c2ead81726aa206bc435dfa885705618b026313a191c1",
    },
    "verification.json": {
        "bytes": 1916,
        "sha256": "e7af4ace6d7f0419938404e79b6d39c512c4a9132d02fb7813374922eb4c0888",
    },
    "mac-receiver.json": {
        "bytes": 104815,
        "sha256": "f7b1eb1a2aa16c24274c2ead81726aa206bc435dfa885705618b026313a191c1",
    },
}
PREDECESSOR = dict(
    source="50864a995f338e379196d43e8446c8f2aa6aca2d",
    closeout=PREDECESSOR_FILES,
    decision="fresh-coupled-rne-control-exact",
    paired_decision="serial-paired-frames-exact",
    temporal_decision="serial-temporal-frames-exact",
    full_window_qualified=False,
)
CASE_ORDER = ("serial0", "serial1")
RNG_KEYS = coupled.RNG_KEYS
# "forward_end" is a wire-key inherited only for byte-state validation. The child
# explicitly labels the physical boundary after-one-nominal-step.
FORWARDS, SUBSTEPS, WORLDS = 21, 10, 64
RNE_BYTES, COM_BYTES = 24576, 12288
MOTOR_FIELDS = {
    "command.position_target": (14,),
    "command.velocity_target": (14,),
    "command.effort_target": (14,),
    "command.pos": (14,),
    "command.vel": (14,),
    "torque": (14,),
    "committed.correction": (10,),
    "committed.target": (14,),
    "committed.queue": (3, 14),
    "committed.previous": (14,),
    "committed.voltage": (1,),
    "committed.kp": (1,),
    "committed.friction": (20,),
    "committed.damping": (20,),
    "committed.ctrl": (14,),
}
MOTOR_BYTES = SUBSTEPS * WORLDS * sum(math.prod(s) for s in MOTOR_FIELDS.values()) * 4
MASK_BYTES = SUBSTEPS * WORLDS * (8 + 3)
CASE_CAPS = {
    "frames.bin": 2 * frames.FRAME_BYTES,
    "rne.entries.bin": FORWARDS * RNE_BYTES,
    "rne.outputs.bin": FORWARDS * RNE_BYTES,
    "com.entries.bin": FORWARDS * COM_BYTES,
    "com.weighted.bin": FORWARDS * COM_BYTES,
    "rng.bin": 128 * 1024,
    "motor.bin": MOTOR_BYTES,
    "masks.bin": MASK_BYTES,
}
LOAD_FIELDS = {
    **{
        name: ("<f4", (20,))
        for name in ("qfrc_bias", "qfrc_constraint", "qfrc_actuator", "qfrc_friction")
    },
    **{name: ("<i4", (512,)) for name in ("efc_type", "efc_id")},
    "efc_force": ("<f4", (512,)),
    "nefc": ("<i4", ()),
    **{
        name: ("<f4", (14,))
        for name in (
            "budget_motor",
            "budget_external",
            "budget_stribeck",
            "budget_output",
        )
    },
    "friction_scale": ("<f4", (1,)),
}
CASE_CAPS.update(
    {
        "load." + name + ".bin": SUBSTEPS
        * WORLDS
        * math.prod(shape)
        * np.dtype(dtype).itemsize
        for name, (dtype, shape) in LOAD_FIELDS.items()
    }
)
CAPS = {
    case + "." + name: cap for case in CASE_ORDER for name, cap in CASE_CAPS.items()
}
CAPS.update(
    {
        "declaration.json": 128 * 1024,
        "child.json": 1024**2,
        "report.json": 256 * 1024,
        "child.log": 1024**2,
        "tests.receipt.json": 128 * 1024,
        "tests.terminal.json": 4096,
    }
)
FLAGS = dict(old.FLAGS)
RECEIVER_OUTPUT_CAP = 4 * 1024**2
GROUPS = [[6, 15], [5, 10, 14], [4, 9, 13], [3, 8, 12], [2], [7], [11], [1], [0]]
SMOOTH_SHA = "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
FORWARD_SHA = "c764b6da0b55c05f97b9368f7c77d4826cbafafe93a15f682a878eef7f9e3de3"
_need = old.need

# Literal layouts audited on the frozen CPU plant. The native bound layout
# must match independently; CPU inspection is not CUDA qualification. Static
# one-row model arrays genuinely have leading stride0, not fabricated stride64.
LAYOUTS = {
    "body_parentid": ([16], [4], "warp._src.types.int32"),
    **{
        f"body_tree_{i}": ([n], [4], "warp._src.types.int32")
        for i, n in enumerate((1, 1, 3, 3, 3, 3, 2))
    },
    "body_mass": ([1, 16], [0, 4], "warp._src.types.float32"),
    "body_subtreemass": ([1, 16], [0, 4], "warp._src.types.float32"),
    "body_rootid": ([16], [4], "warp._src.types.int32"),
    "body_inertia": ([1, 16], [0, 12], "warp._src.types.vec3f"),
    "dof_bodyid": ([20], [4], "warp._src.types.int32"),
    "dof_parentid": ([20], [4], "warp._src.types.int32"),
    "dof_armature": ([1, 20], [0, 4], "warp._src.types.float32"),
    **{
        name: ([15], [4], "warp._src.types.int32")
        for name in ("jnt_type", "jnt_dofadr", "jnt_bodyid")
    },
    **{
        name: ([64, 16], [640, 40], "mujoco_warp._src.types.vec10f")
        for name in ("crb", "cinert")
    },
    "qM": ([64, 20, 20], [1600, 80, 4], "warp._src.types.float32"),
    "cdof": ([64, 20], [480, 24], "warp._src.types.spatial_vectorf"),
    **{
        name: ([64, 16], [192, 12], "warp._src.types.vec3f")
        for name in ("subtree_com", "xipos")
    },
    **{
        name: ([64, 16], [576, 36], "warp._src.types.mat33f")
        for name in ("ximat", "xmat")
    },
    **{
        name: ([64, 15], [180, 12], "warp._src.types.vec3f")
        for name in ("xanchor", "xaxis")
    },
    **{
        name: ([64, 16], [384, 24], "warp._src.types.spatial_vectorf")
        for name in ("cfrc_int", "cacc", "cvel")
    },
}


def read_inventory(root, anchors):
    """Read and authenticate every exact artifact before JSON/NumPy decoding."""
    root = Path(root)
    _need(root.is_dir() and not root.is_symlink(), "regular fresh artifact directory")
    _need(
        type(anchors) is dict
        and set(anchors) == set(CAPS)
        and {path.name for path in root.iterdir()} == set(CAPS),
        "exact forty-eight-file one-step artifact inventory",
    )
    result = {}
    for name in sorted(CAPS):
        anchor = anchors[name]
        _need(
            type(anchor) is dict
            and set(anchor) == {"bytes", "sha256"}
            and type(anchor["bytes"]) is int
            and 0 < anchor["bytes"] <= CAPS[name],
            "bounded externally anchored whole artifact",
        )
        descriptor = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            _need(
                stat.S_ISREG(before.st_mode) and before.st_size == anchor["bytes"],
                "regular whole-length artifact",
            )
            chunks = []
            total = 0
            while total <= CAPS[name]:
                chunk = os.read(descriptor, min(65536, CAPS[name] + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(descriptor)
            _need(
                total == before.st_size
                and (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ),
                "stable whole artifact read",
            )
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        old._sha(raw, anchor["sha256"])
        result[name] = raw
    _need(sum(map(len, result.values())) < 32 * 1024**2, "whole evidence below 32 MiB")
    return result


def checked_binding(value, source, expected_tree, expected_leaves_sha256):
    _need(
        type(value) is dict
        and set(value) == {"source", "branch", "tree", "leaf_count", "leaves_sha256"}
        and value["source"] == source
        and value["branch"] == "feat/athletics-obstacle-curriculum"
        and type(value["tree"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", value["tree"])
        and value["tree"] == expected_tree
        and type(value["leaf_count"]) is int
        and value["leaf_count"] == 673
        and type(value["leaves_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["leaves_sha256"])
        and value["leaves_sha256"] == expected_leaves_sha256,
        "fresh whole committed source binding",
    )


def _check_test_proof(raw, declaration, binding, packages, caps):
    _need(
        EXPECTED_TESTS > 0
        and re.fullmatch(r"[0-9a-f]{64}", TEST_FILES_SHA256) is not None,
        "exact current-source CPU test collection constants are finalized",
    )
    tests = declaration["tests"]
    receipt = old._json(raw["tests.receipt.json"])
    terminal = old._json(raw["tests.terminal.json"])
    test_caps = {**caps, "LimitFSIZE": "67108864"}
    _need(
        declaration["phase_a"]
        == {"source": PHASE_A_SOURCE, "files": PHASE_A, "terminal": PHASE_A_TERMINAL}
        and declaration["predecessors"] == PREDECESSORS
        and declaration["com_predecessor"] == COM_PREDECESSOR
        and declaration["rne_predecessor"] == PREDECESSOR
        and declaration["tick_predecessor"] == TICK_PREDECESSOR
        and declaration["case_order"] == list(CASE_ORDER),
        "exact retained Phase A, immutable predecessors and separate CoM negatives",
    )
    _need(
        type(tests) is dict
        and old.exact_int(tests["count"], EXPECTED_TESTS)
        and old.exact_int(declaration["expected_tests"], EXPECTED_TESTS)
        and old.exact_int(tests["files"], len(receipt["test_files"]))
        and tests["source_binding"] == binding
        and type(tests["receipt_sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", tests["receipt_sha256"])
        and tests["terminal"] == terminal
        and terminal["InvocationID"]
        != declaration["service_properties"]["InvocationID"]
        and terminal["Result"] == "success"
        and terminal["MainPID"] == "0"
        and terminal["ExecMainStatus"] == "0"
        and terminal["NRestarts"] == "0",
        "completed independent current-source CPU prerequisite",
    )
    files = receipt["test_files"]
    _need(
        receipt["protocol"] == DATA_PROTOCOL + ":tests"
        and receipt["source_binding"] == binding
        and receipt["packages"] == packages
        and old.exact_int(receipt["tests"], EXPECTED_TESTS)
        and type(files) is list
        and len(files) == len(set(files))
        and sha256(old.canonical(files)).hexdigest() == TEST_FILES_SHA256
        and receipt["retained_descriptor_fixture"] == DESCRIPTOR_FIXTURE
        and receipt["retained_test_fixtures"] == RETAINED_TEST_FIXTURES
        and re.fullmatch(r"[0-9a-f]{64}", receipt["junit_sha256"])
        and re.fullmatch(r"[0-9a-f]{64}", receipt["log_sha256"])
        and old.false_flags(receipt["flags"])
        and sha256(raw["tests.receipt.json"]).hexdigest() == tests["receipt_sha256"]
        and all(terminal.get(key) == value for key, value in test_caps.items())
        and all(
            receipt["service_properties"].get(key) == value
            for key, value in test_caps.items()
        )
        and terminal["ActiveState"] == "active"
        and terminal["SubState"] == "exited"
        and terminal["InvocationID"] == receipt["service_properties"]["InvocationID"]
        and re.fullmatch(r"[0-9a-f]{32}", terminal["InvocationID"])
        and receipt["service_properties"]["ActiveState"] == "active"
        and receipt["service_properties"]["SubState"] == "running"
        and re.fullmatch(r"[1-9][0-9]*", receipt["service_properties"]["MainPID"]),
        "whole completed CPU receipt, exact ordered suite and capped invocation",
    )
    _need(
        receipt["test_environment"]
        == {
            "CUDA_VISIBLE_DEVICES": "",
            "MICRODUCK_STANCE_PROFILE": "wsl-10098-20260930",
            "ATEN_CPU_CAPABILITY": "default",
            "MKL_CBWR": "COMPATIBLE",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
        "exact recorded CPU prerequisite environment",
    )


def _chunks(raw, size, count, label):
    _need(type(raw) is bytes and len(raw) == size * count, label + " complete raw bank")
    return [raw[i * size : (i + 1) * size] for i in range(count)]


# Reuse pure, unchanged shape/motor/reduction validators with their original
# literal one-tick counts. No historical protocol, cap or guard is mutated.
checked_motor = step_receiver.checked_motor
checked_frames = step_receiver.checked_frames
checked_ledger = step_receiver.checked_ledger
checked_scope = step_receiver.checked_scope


M6_PARAMETERS = {
    "kt": 0.36601349688984386,
    "R": 2.8113923539223227,
    "armature": 0.0018077432831600838,
    "q_offset": 0.0271132870444849,
    "friction_base": 0.004771183165566,
    "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698,
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,
    "model": "m6",
    "actuator": "xl330",
}
M6_SHA256 = "61c699362fb3fabdde93eeba5e1ad3bf4ef9ca2f71d03e316b1924ff005b20d3"


def decode_load_packets(raw, case):
    """Decode only complete already-authenticated per-case banks."""
    result = {}
    for name, (dtype, shape) in LOAD_FIELDS.items():
        packet = raw[case + ".load." + name + ".bin"]
        _need(
            type(packet) is bytes and len(packet) == CASE_CAPS["load." + name + ".bin"],
            "complete ten-call load packet " + name,
        )
        values = np.frombuffer(packet, dtype=dtype).reshape((SUBSTEPS, WORLDS, *shape))
        if np.dtype(dtype).kind == "f":
            _need(bool(np.isfinite(values).all()), "finite full load bank " + name)
        result[name] = values
    _need(
        bool(((result["nefc"] >= 0) & (result["nefc"] <= 512)).all()),
        "complete bounded active constraint counts",
    )
    _need(
        bool((result["friction_scale"] == np.float32(1.0)).all())
        and bool(
            ((result["budget_stribeck"] >= 0) & (result["budget_stribeck"] <= 1)).all()
        )
        and bool((result["budget_output"] >= 0).all()),
        "nominal scale, bounded Stribeck and nonnegative budget",
    )
    return result


def friction_scan_reference(types, ids, forces, counts):
    """Independent float32 ascending-row scan, not CUDA scatter replay."""
    _need(
        types.shape == ids.shape == forces.shape == (WORLDS, 512)
        and types.dtype == ids.dtype == np.dtype("<i4")
        and forces.dtype == np.dtype("<f4")
        and counts.shape == (WORLDS,)
        and counts.dtype == np.dtype("<i4")
        and bool(np.isfinite(forces).all())
        and bool(((counts >= 0) & (counts <= 512)).all()),
        "literal finite complete constraint provenance",
    )
    result = np.zeros((WORLDS, 20), dtype="<f4")
    duplicates = np.zeros((WORLDS, 20), dtype=np.int32)
    for world in range(WORLDS):
        for row in range(int(counts[world])):
            if types[world, row] != 1:  # pinned mjCNSTR_FRICTION_DOF
                continue
            dof = int(ids[world, row])
            _need(0 <= dof < 20, "active friction row has a real DOF address")
            result[world, dof] = np.float32(result[world, dof] + forces[world, row])
            duplicates[world, dof] += 1
    return result, duplicates


def budget_reference(motor, external, stribeck, scale):
    """Independent pointwise m6 budget using captured inputs, not BAM import."""
    _need(
        motor.shape == external.shape == stribeck.shape == (WORLDS, 14)
        and scale.shape == (WORLDS, 1)
        and all(
            v.dtype == np.dtype("<f4") and np.isfinite(v).all()
            for v in (motor, external, stribeck, scale)
        ),
        "literal finite complete m6 budget inputs",
    )
    p = {
        name: np.float32(value)
        for name, value in M6_PARAMETERS.items()
        if type(value) is float
    }
    output = np.full_like(motor, p["friction_base"])
    output = output + stribeck * p["friction_stribeck"]
    gearbox = np.abs(
        external * p["load_friction_external"] - motor * p["load_friction_motor"]
    )
    output = output + gearbox
    gearbox_stribeck = np.abs(
        external * p["load_friction_external_stribeck"]
        - motor * p["load_friction_motor_stribeck"]
    )
    output = output + stribeck * gearbox_stribeck
    abs_external, abs_motor = np.abs(external), np.abs(motor)
    drive = (abs_motor > abs_external).astype("<f4")
    backdrive = np.float32(1.0) - drive
    quad = (
        drive * p["load_friction_external_quad"] * abs_external**2
        + backdrive * p["load_friction_motor_quad"] * abs_motor**2
    )
    output = (output + stribeck * quad) * scale
    _need(
        output.dtype == np.dtype("<f4") and bool(np.isfinite(output).all()),
        "finite independent m6 reference",
    )
    return output


def checked_load_receipt(receipt, record, raw, case):
    _need(
        type(receipt) is dict
        and receipt["protocol"] == "microduck-bam-load-observer-oct7-v1"
        and receipt["status"] == "complete"
        and receipt["fault"] is None
        and old.exact_int(receipt["calls"], 10)
        and old.exact_int(receipt["expected_calls"], 10)
        and old.exact_int(receipt["worlds"], 64)
        and receipt["device"] == "cuda:0"
        and receipt["m6_json_sha256"] == M6_SHA256
        and receipt["m6_parameters"] == M6_PARAMETERS
        and receipt["installed_bam_sha256"]
        == "af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440"
        and receipt["adapter_sha256"]
        == "e9b349a4be9910ea2c452cacfa78ccfc7fadb9f60e2f5ab8b54eed835e09b2b4"
        and receipt["state_commit_sha256"]
        == "7974953ded3c96e4a05ba05d15bfc82e12ea3b8255b150c8636ac07f580d26b9"
        and all(
            receipt[key] is True
            for key in (
                "original_compute_called_once_per_proposal",
                "original_friction_scan_called_once_per_proposal",
                "original_budget_called_once_per_proposal",
                "runtime_parameters_checked_each_budget",
                "actual_bam_inputs_share_bound_raw_storage",
            )
        )
        and all(
            receipt[key] is False
            for key in (
                "runtime_cause_proven",
                "training_authorized",
                "physical_acceptance",
                "original_run_entry_captured",
                "native_qualified",
                "full_window_qualified",
            )
        ),
        "closed actual ten-call pinned nominal BAM observer",
    )
    _need(
        receipt["data_id"] == record["control_scope"]["calls"][0]["data_id"]
        and all(
            type(receipt[key]) is int and receipt[key] > 0
            for key in ("runtime_id", "data_id", "motor_id", "bridge_data_id")
        )
        and type(receipt["staged_actuator_ids"]) is list
        and len(receipt["staged_actuator_ids"]) == 10
        and all(
            type(value) is int and value > 0 for value in receipt["staged_actuator_ids"]
        ),
        "observed data identity tied to live reduction scope",
    )
    _need(
        receipt["source_pins"]
        == {
            "bam.mjlab": receipt["installed_bam_sha256"],
            "stance_control_state.py": receipt["state_commit_sha256"],
            "friction_dr_bam.py": receipt["adapter_sha256"],
        }
        and receipt["model_flags"]
        == {
            "name": "m6",
            "actuator": "xl330",
            "stribeck": True,
            "load_dependent": True,
            "directional": True,
            "quadratic": True,
        }
        and all(
            receipt["model_flags"][key] is True
            for key in ("stribeck", "load_dependent", "directional", "quadratic")
        )
        and receipt["runtime_parameters"]
        == [
            {
                name: value
                for name, value in M6_PARAMETERS.items()
                if type(value) is float
            }
        ]
        * 10,
        "actual fixed model flags, source pins and every proposal's parameters",
    )
    names = {
        "BamStateCommit.compute": "context_decorator.<locals>.decorate_context",
        "BamActuator.compute": "BamActuator.compute",
        "BamActuator._dof_friction_force": "BamActuator._dof_friction_force",
        "FrictionDRBamActuator._compute_friction_budget": "FrictionDRBamActuator._compute_friction_budget",
        "BamActuator._compute_friction_budget": "BamActuator._compute_friction_budget",
    }
    _need(
        type(receipt["method_pins"]) is dict
        and set(receipt["method_pins"]) == set(names)
        and all(
            type(receipt["method_pins"][name]) is dict
            and set(receipt["method_pins"][name]) == {"object_id", "code_name"}
            and type(receipt["method_pins"][name]["object_id"]) is int
            and receipt["method_pins"][name]["object_id"] > 0
            and receipt["method_pins"][name]["code_name"] == code
            for name, code in names.items()
        ),
        "exact original method identity/code receipts",
    )
    _need(
        receipt["call_order"] == list(range(10))
        and all(type(v) is int for v in receipt["call_order"])
        and receipt["raw_data_call_ids"] == [receipt["data_id"]] * 10
        and receipt["bridge_data_call_ids"] == [receipt["bridge_data_id"]] * 10
        and type(receipt["call_records"]) is list
        and len(receipt["call_records"]) == 10,
        "complete actual ascending proposal/bridge/raw-data sequence",
    )
    for index, row in enumerate(receipt["call_records"]):
        _need(
            row
            == dict(
                proposal_index=index,
                steps_before=[index] * 64,
                runtime_data_id=receipt["data_id"],
                bridge_data_id=receipt["bridge_data_id"],
                staged_actuator_id=receipt["staged_actuator_ids"][index],
                compute_calls=index + 1,
                friction_scan_calls=index + 1,
                budget_calls=index + 1,
            )
            and all(
                type(row[name]) is int
                for name in (
                    "proposal_index",
                    "runtime_data_id",
                    "bridge_data_id",
                    "staged_actuator_id",
                    "compute_calls",
                    "friction_scan_calls",
                    "budget_calls",
                )
            )
            and all(type(v) is int for v in row["steps_before"]),
            "literal actual one compute/scan/budget at each substep",
        )
    layout_names = (
        "qfrc_bias",
        "qfrc_constraint",
        "qfrc_actuator",
        "nefc",
        "efc.type",
        "efc.id",
        "efc.force",
    )
    layouts = receipt["raw_array_layouts"]
    _need(
        type(layouts) is list and [r["name"] for r in layouts] == list(layout_names),
        "complete raw load input layouts",
    )
    for row in layouts:
        name = row["name"]
        shape = (
            [64]
            if name == "nefc"
            else [64, 512]
            if name.startswith("efc.")
            else [64, 20]
        )
        strides = [4] if name == "nefc" else [shape[-1] * 4, 4]
        dtype = "int32" if name in ("nefc", "efc.type", "efc.id") else "float32"
        _need(
            set(row)
            == {
                "name",
                "object_id",
                "ptr",
                "shape",
                "strides",
                "dtype",
                "device",
                "contiguous",
            }
            and type(row["object_id"]) is int
            and row["object_id"] > 0
            and type(row["ptr"]) is int
            and row["ptr"] > 0
            and row["ptr"] % 4 == 0
            and row["shape"] == shape
            and row["strides"] == strides
            and all(type(v) is int for v in row["shape"] + row["strides"])
            and row["dtype"] == f"<class 'warp._src.types.{dtype}'>"
            and row["device"] == "cuda:0"
            and row["contiguous"] is True,
            "literal actual stable raw load storage " + name,
        )
    _need(
        type(receipt["calls_sha256"]) is dict
        and set(receipt["calls_sha256"]) == set(LOAD_FIELDS)
        and type(receipt["field_lengths"]) is dict
        and set(receipt["field_lengths"]) == set(LOAD_FIELDS),
        "complete observer call/field inventory",
    )
    for name in LOAD_FIELDS:
        size = CASE_CAPS["load." + name + ".bin"] // SUBSTEPS
        packet = raw[case + ".load." + name + ".bin"]
        _need(
            receipt["field_lengths"][name] == size
            and receipt["calls_sha256"][name]
            == [
                sha256(part).hexdigest()
                for part in _chunks(packet, size, SUBSTEPS, name)
            ],
            "actual complete per-call observer snapshots " + name,
        )


def checked_load_analysis(raw, child):
    banks = {case: decode_load_packets(raw, case) for case in CASE_ORDER}
    cases = {}
    for case in CASE_ORDER:
        record = child["cases"][case]
        checked_load_receipt(record["load_observer"], record, raw, case)
        motor_rows = divergence._motor_views(raw[case + ".motor.bin"])
        rows = []
        for step in range(SUBSTEPS):
            fields = {name: value[step] for name, value in banks[case].items()}
            reference, row_counts = friction_scan_reference(
                fields["efc_type"],
                fields["efc_id"],
                fields["efc_force"],
                fields["nefc"],
            )
            expected_external = (
                -fields["qfrc_bias"][:, 6:20] + fields["qfrc_constraint"][:, 6:20]
            ) - fields["qfrc_friction"][:, 6:20]
            references = {
                "previous_solved_actuator_input": (
                    fields["qfrc_actuator"][:, 6:20],
                    fields["budget_motor"],
                ),
                "actual_external_torque_input": (
                    expected_external,
                    fields["budget_external"],
                ),
                "actual_budget_committed": (
                    fields["budget_output"],
                    motor_rows[step]["committed.friction"][:, 6:20],
                ),
                "friction_row_scan": (reference, fields["qfrc_friction"]),
                "m6_budget_reference": (
                    budget_reference(
                        fields["budget_motor"],
                        fields["budget_external"],
                        fields["budget_stribeck"],
                        fields["friction_scale"],
                    ),
                    fields["budget_output"],
                ),
            }
            comparisons = {
                name: divergence._float_stats(left, right, label=name)
                for name, (left, right) in references.items()
            }
            rows.append(
                dict(
                    substep=step,
                    references=comparisons,
                    multiple_active_friction_rows_per_dof=int((row_counts > 1).sum()),
                    active_friction_rows=int(row_counts.sum()),
                )
            )
        cases[case] = rows
    paired = []
    for step in range(SUBSTEPS):
        for name, (dtype, _shape) in LOAD_FIELDS.items():
            left, right = (banks[case][name][step] for case in CASE_ORDER)
            if np.dtype(dtype).kind == "f":
                stats = divergence._float_stats(left, right, label=name)
            else:
                unequal = left != right
                stats = dict(
                    exact_raw_bits=left.tobytes() == right.tobytes(),
                    mismatch_scalars=int(unequal.sum()),
                    world_mismatch_scalars=unequal.reshape(WORLDS, -1)
                    .sum(1)
                    .astype(int)
                    .tolist(),
                    coordinate_samples=[
                        dict(
                            world=int(c[0]),
                            index=[int(v) for v in c[1:]],
                            left=int(left[tuple(c)]),
                            right=int(right[tuple(c)]),
                        )
                        for c in np.argwhere(unequal)[:4]
                    ],
                )
            paired.append(
                dict(
                    stream="bam-load",
                    index=step,
                    field=name,
                    stats=stats,
                    boundary="actual-friction-scan-or-budget-call",
                )
            )
    reference_exact = all(
        stat["exact_raw_bits"]
        for rows in cases.values()
        for row in rows
        for stat in row["references"].values()
    )
    input_commit_exact = all(
        row["references"][name]["exact_raw_bits"]
        for rows in cases.values()
        for row in rows
        for name in (
            "previous_solved_actuator_input",
            "actual_external_torque_input",
            "actual_budget_committed",
        )
    )
    return dict(
        cases=cases,
        paired_field_comparisons=paired,
        first_observed_load_difference=divergence._first_event(paired),
        reference_decision="bam-load-reference-exact"
        if reference_exact
        else "bam-load-reference-negative",
        paired_observed_loads_exact=all(
            row["stats"]["exact_raw_bits"] for row in paired
        ),
        actual_input_commit_consistency_exact=input_commit_exact,
        stribeck_function_reexecution=False,
        constructor_and_solver_reexecution=False,
        inactive_constraint_rows_retained_but_not_used_in_friction_scan=True,
        scan_order_is_cpu_ascending_not_observed_cuda_atomic_order=True,
        tolerance_applied=False,
        runtime_cause_proven=False,
        flags=dict(FLAGS),
    )


def repeat_decision(arithmetic, paired, load_analysis):
    return (
        "fresh-bam-load-repeat-exact"
        if (
            arithmetic
            and paired
            and load_analysis["paired_observed_loads_exact"]
            and load_analysis["actual_input_commit_consistency_exact"]
            and load_analysis["reference_decision"] == "bam-load-reference-exact"
        )
        else "fresh-bam-load-repeat-negative"
    )


def verify_directory(
    root,
    anchors,
    terminal_raw,
    terminal_sha256,
    *,
    source,
    expected_tree,
    expected_leaves_sha256,
):
    _need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and type(expected_tree) is str
        and re.fullmatch(r"[0-9a-f]{40}", expected_tree)
        and type(expected_leaves_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_leaves_sha256)
        and type(terminal_raw) is bytes
        and 0 < len(terminal_raw) <= 4096,
        "external exact source/tree/leaves and terminal anchors",
    )
    raw = read_inventory(root, anchors)
    old._sha(terminal_raw, terminal_sha256)
    declaration = old._json(raw["declaration.json"])
    report = old._json(raw["report.json"])
    child = old._json(raw["child.json"])
    terminal = old._json(terminal_raw)
    for value, suffix in (
        (declaration, ":declaration"),
        (child, ":child"),
    ):
        _need(
            type(value) is dict
            and value["protocol"] == DATA_PROTOCOL + suffix
            and value["source"] == source
            and old.false_flags(value["flags"]),
            "fresh protocol/source and permanently false admission flags",
        )
        checked_binding(
            value["source_binding"], source, expected_tree, expected_leaves_sha256
        )
    _need(
        type(report) is dict
        and report["protocol"] == DATA_PROTOCOL + ":report"
        and report["source"] == source
        and old.false_flags(report["flags"]),
        "fresh report protocol/source and false flags",
    )
    checked_binding(
        report["source_binding"], source, expected_tree, expected_leaves_sha256
    )
    binding = declaration["source_binding"]
    _need(
        declaration["source_binding"]
        == child["source_binding"]
        == report["source_binding"]
        and child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest()
        and report["child_sha256"] == sha256(raw["child.json"]).hexdigest(),
        "whole declaration/child/report source binding",
    )

    packages = declaration["packages"]
    coupled.checked_packages(packages)
    _need(child["packages"] == packages, "same actual native child package binding")
    live = declaration["service_properties"]
    caps = SERVICE_CAPS
    _need(
        type(live) is dict
        and all(
            live.get(key) == value and terminal.get(key) == value
            for key, value in caps.items()
        )
        and live["ActiveState"] == "active"
        and live["SubState"] == "running"
        and re.fullmatch(r"[1-9][0-9]*", live["MainPID"])
        and re.fullmatch(r"[0-9a-f]{32}", live["InvocationID"])
        and terminal["InvocationID"] == live["InvocationID"]
        and terminal["MainPID"] == "0"
        and terminal["Result"] == "success"
        and terminal["ExecMainStatus"] == "0"
        and terminal["ActiveState"] == "active"
        and terminal["SubState"] == "exited",
        "completed native run with exact retained service caps",
    )
    _need(
        old.exact_int(report["returncode"], 0)
        and report["gpu_child_observed"] is True
        and type(report["observed_child_pid"]) is int
        and report["observed_child_pid"] > 0
        and report["observed_child_ppid"] == int(live["MainPID"])
        and child["child_pid"] == report["observed_child_pid"]
        and type(child["child_pid"]) is int
        and type(child["owner_pid"]) is int
        and child["owner_pid"] == int(live["MainPID"])
        and type(child["child_ppid"]) is int
        and child["child_ppid"] == child["owner_pid"]
        and report["child_sha256"] == sha256(raw["child.json"]).hexdigest()
        and child["declaration_sha256"] == sha256(raw["declaration.json"]).hexdigest(),
        "observed native owner/child chain and whole child hash",
    )
    old.valid_host(declaration["host"], child["child_pid"], idle=True)
    monitor = report["monitor"]
    _need(type(monitor) is list and 0 < len(monitor) <= 480, "bounded real GPU monitor")
    seen = False
    previous = -1.0
    for row in monitor:
        _need(
            type(row["elapsed"]) in (int, float)
            and math.isfinite(row["elapsed"])
            and previous <= row["elapsed"] < 240
            and old.exact_int(row["child_pid"], child["child_pid"]),
            "bounded monitor timeline and exact child PID",
        )
        seen |= old.valid_host(row["host"], child["child_pid"])
        _need(
            row["host"]["used_mib"] <= 12288 and row["host"]["free_mib"] >= 10240,
            "bounded whole-GPU memory exposure",
        )
        previous = row["elapsed"]
    _need(seen, "actual GPU child observed, not inferred")
    old.valid_host(report["closure_host"], child["child_pid"], idle=True)

    _check_test_proof(raw, declaration, binding, packages, caps)
    log_text = raw["child.log"].decode("utf-8")
    filtered_log = "\n".join(
        line
        for line in log_text.splitlines()
        if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
    )
    _need(
        not re.search(
            r"\b(?:nan|nonfinite|overflow|warning|traceback)\b|CUDA error",
            filtered_log,
            re.I,
        ),
        "finite warning-free owned native child log",
    )

    _need(
        child["case_order"] == list(CASE_ORDER)
        and type(child["cases"]) is dict
        and set(child["cases"]) == set(CASE_ORDER),
        "exact two fresh serial physical ticks",
    )
    case_rows = {}
    for case in CASE_ORDER:
        record = child["cases"][case]
        _need(
            old.exact_int(record["physics_steps"], 10)
            and old.exact_int(record["ntendon"], 0)
            and old.exact_int(record["explicit_reset_calls"], 0)
            and all(
                record[key] is False
                for key in (
                    "graph_created",
                    "actor_model_created",
                    "optimizer_created",
                    "storage_created",
                )
            )
            and record["control_capture"] is True
            and record["rng_end_boundary"] == "after-one-nominal-step"
            and record["control_ids"] == list(range(14))
            and record["nominal_parameters"]
            == {
                "voltage": 7.5,
                "drop_gain": 0.1,
                "kp_scale": 1.0,
                "kd_scale": 1.0,
                "friction_scale": 1.0,
            },
            "literal nominal one-step/no-reset/no-learner case",
        )
        descriptor = record["compiled_descriptor"]
        _need(
            sha256(old.canonical(descriptor)).hexdigest() == DESCRIPTOR_SHA256
            and descriptor["selected_fields_sha256"] == SELECTED_FIELDS_SHA256
            and descriptor["qids"] == list(range(7, 21)),
            "whole exact moving plant descriptor",
        )
        _initial, final = checked_frames(raw[case + ".frames.bin"], descriptor)
        banks = {
            name: _chunks(raw[case + "." + name], size, FORWARDS, name)
            for name, size in (
                ("com.entries.bin", COM_BYTES),
                ("com.weighted.bin", COM_BYTES),
                ("rne.entries.bin", RNE_BYTES),
                ("rne.outputs.bin", RNE_BYTES),
            )
        }
        checked_scope(record["control_scope"], banks)
        com_rows = [
            coupled.weighted_comparison(e, o)
            for e, o in zip(
                banks["com.entries.bin"], banks["com.weighted.bin"], strict=True
            )
        ]
        rne_rows = [
            prior._field_stats(
                prior._recurrence(prior._rne_array(e, "actual evolving RNE input")),
                prior._rne_array(o, "actual evolving RNE output"),
                body_axis=1,
            )
            for e, o in zip(
                banks["rne.entries.bin"], banks["rne.outputs.bin"], strict=True
            )
        ]
        states = coupled.checked_rng(raw[case + ".rng.bin"], record["rng_metadata"])
        _need(
            all(
                states[f"private_{dev}_ctor_end"]
                == states[f"private_{dev}_forward_end"]
                for dev in ("cpu", "cuda")
            ),
            "no RNG draws during nominal zero-action physics",
        )
        case_rows[case] = dict(
            com_arithmetic=com_rows,
            rne_arithmetic=rne_rows,
            com_arithmetic_exact=all(v["exact_raw_bits"] for v in com_rows),
            rne_arithmetic_exact=all(v["exact_raw_bits"] for v in rne_rows),
            motor=checked_motor(
                raw[case + ".motor.bin"], raw[case + ".masks.bin"], descriptor, final
            ),
            ledger=checked_ledger(record["ledger"]),
            rng_states_authenticated=True,
        )
    comparisons = []
    for name in CASE_CAPS:
        left, right = raw["serial0." + name], raw["serial1." + name]
        comparisons.append(
            dict(
                packet=name,
                bytes_left=len(left),
                bytes_right=len(right),
                left_sha256=sha256(left).hexdigest(),
                right_sha256=sha256(right).hexdigest(),
                exact_raw_bits=left == right,
            )
        )
    field_rows = []
    left = frames._decode_packet(raw["serial0.frames.bin"], "serial0")
    right = frames._decode_packet(raw["serial1.frames.bin"], "serial1")
    for index, (lframe, rframe) in enumerate(zip(left, right, strict=True)):
        for name, _shape in frames.FRAME_FIELDS:
            field_rows.append(
                dict(
                    boundary="constructor" if index == 0 else "after-one-tick",
                    field=name,
                    **prior._field_stats(lframe[name], rframe[name]),
                )
            )
    arithmetic = all(
        row["com_arithmetic_exact"] and row["rne_arithmetic_exact"]
        for row in case_rows.values()
    )
    ledgers = [child["cases"][case]["ledger"] for case in CASE_ORDER]
    ledger_comparison = dict(
        left_sha256=sha256(old.canonical(ledgers[0])).hexdigest(),
        right_sha256=sha256(old.canonical(ledgers[1])).hexdigest(),
        exact_retained_values=old.canonical(ledgers[0]) == old.canonical(ledgers[1]),
        independent_reward_reexecution=False,
    )
    ledger_fields = []
    for name in ("reward", *ledgers[0]["term_sums"]):
        values = [
            v["reward"] if name == "reward" else v["term_sums"][name] for v in ledgers
        ]
        ledger_fields.append(
            dict(
                field=name,
                **prior._field_stats(
                    np.asarray(values[0], dtype="<f4"),
                    np.asarray(values[1], dtype="<f4"),
                ),
            )
        )
    paired = (
        ledger_comparison["exact_retained_values"]
        and all(row["exact_raw_bits"] for row in comparisons)
        and all(row["exact_raw_bits"] for row in field_rows)
    )
    load_analysis = checked_load_analysis(raw, child)
    # Independent execution/consistency checks above fail closed, not a tolerance.
    result = dict(
        protocol=PROTOCOL,
        source=source,
        case_order=list(CASE_ORDER),
        cases=case_rows,
        load_boundary_analysis=load_analysis,
        arithmetic_decision="moving-reductions-reference-exact"
        if arithmetic
        else "moving-reductions-reference-negative",
        execution_decision="one-nominal-tick-load-consistent"
        if load_analysis["actual_input_commit_consistency_exact"]
        else "one-nominal-tick-load-consistency-negative",
        paired_decision="one-tick-paired-exact"
        if paired
        else "one-tick-paired-negative",
        decision=repeat_decision(arithmetic, paired, load_analysis),
        packet_comparisons=comparisons,
        field_comparisons=field_rows,
        ledger_comparison=ledger_comparison,
        ledger_field_comparisons=ledger_fields,
        simulated_seconds=0.02,
        physics_reexecution=False,
        bam_reexecution=False,
        full_policy_capture_replay=False,
        receiver_files=anchors,
        terminal_sha256=terminal_sha256,
        flags=dict(FLAGS),
    )
    _need(
        len(old.canonical(result)) <= RECEIVER_OUTPUT_CAP,
        "complete bounded one-step receiver",
    )
    return result
