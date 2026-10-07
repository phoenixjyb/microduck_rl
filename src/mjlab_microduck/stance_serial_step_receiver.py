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

PROTOCOL = "microduck-serial-one-step-receiver-oct7-v1"
DATA_PROTOCOL = "microduck-serial-one-step-probe-oct7-v1"
EXPECTED_TESTS = 2250
TEST_FILES_SHA256 = "1f341edaf05efe261b6888b4e649aaca71a4ebb4b12cf00b8d33e0e6689f68f5"
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
        "exact twenty-two-file one-step artifact inventory",
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
    _need(sum(map(len, result.values())) < 16 * 1024**2, "whole evidence below 16 MiB")
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
        and value["leaf_count"] == 663
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


def checked_motor(raw, mask_raw, descriptor, final):
    """Check complete actual motor/control records, not recompute BAM."""
    _need(
        type(raw) is bytes
        and len(raw) == MOTOR_BYTES
        and type(mask_raw) is bytes
        and len(mask_raw) == MASK_BYTES,
        "exact complete motor and mask packets",
    )
    nominal = np.asarray(descriptor["initial_qpos"], dtype="<f4")[descriptor["qids"]]
    _need(nominal.shape == (14,), "descriptor-bound ordered nominal targets")
    offset, mask_offset, peak = 0, 0, 0.0
    for step in range(SUBSTEPS):
        row = {}
        for name, shape in MOTOR_FIELDS.items():
            length = WORLDS * math.prod(shape) * 4
            value = np.frombuffer(raw[offset : offset + length], dtype="<f4").reshape(
                WORLDS, *shape
            )
            _need(np.isfinite(value).all(), "finite full motor field " + name)
            row[name] = value
            offset += length
        steps = np.frombuffer(
            mask_raw[mask_offset : mask_offset + WORLDS * 8], dtype="<i8"
        )
        mask_offset += WORLDS * 8
        _need((steps == step).all(), "literal accepted substep counters")
        for name, expected in (("live", 1), ("accepted", 1), ("rejected", 0)):
            mask = np.frombuffer(
                mask_raw[mask_offset : mask_offset + WORLDS], dtype="u1"
            )
            mask_offset += WORLDS
            _need((mask == expected).all(), "all rows " + name + " as declared")
        for name in ("command.position_target", "committed.target"):
            _need(
                np.array_equal(
                    row[name].view("<u4"),
                    np.broadcast_to(nominal.view("<u4"), (WORLDS, 14)),
                ),
                "zero-action exact nominal target " + name,
            )
        _need(
            np.array_equal(
                row["committed.queue"].view("<u4"),
                np.broadcast_to(nominal.view("<u4"), (WORLDS, 3, 14)),
            ),
            "zero-action exact three-slot FIFO",
        )
        for name in (
            "command.velocity_target",
            "command.effort_target",
            "committed.correction",
        ):
            _need(not row[name].view("<u4").any(), "literal positive-zero " + name)
        torque = row["torque"]
        _need(
            (np.abs(torque) <= np.float32(0.36)).all(),
            "predeclared motor torque ceiling",
        )
        peak = max(peak, float(np.max(np.abs(torque))))
        for name in ("committed.previous", "committed.ctrl"):
            _need(
                np.array_equal(torque.view("<u4"), row[name].view("<u4")),
                "actual committed torque/history " + name,
            )
        _need(
            (
                (row["committed.voltage"] >= 6.0) & (row["committed.voltage"] <= 7.5)
            ).all()
            and ((row["committed.kp"] >= 0.0) & (row["committed.kp"] <= 200.0)).all()
            and (row["committed.friction"] >= 0.0).all()
            and (row["committed.damping"] >= 0.0).all(),
            "bounded actual firmware/motor fields",
        )
        if step == 0:
            _need(
                np.array_equal(
                    row["command.pos"].view("<u4"),
                    np.broadcast_to(nominal.view("<u4"), (WORLDS, 14)),
                )
                and not row["command.vel"].view("<u4").any(),
                "first actual command reads initial physical joints",
            )
    _need(
        offset == len(raw) and mask_offset == len(mask_raw),
        "no motor/mask prefix or tail",
    )
    _need(
        np.array_equal(final["ctrl"].view("<u4"), row["committed.ctrl"].view("<u4")),
        "final actual native control matches last committed motor proposal",
    )
    return dict(
        substeps=SUBSTEPS,
        worlds=WORLDS,
        peak_torque_nm=peak,
        complete_position_only_zero_action_controls_consistent=True,
        independent_bam_reexecution=False,
        thermal_simulation=False,
    )


def checked_frames(raw, descriptor):
    initial, final = frames._decode_packet(raw, "serial-step")
    for label, frame in (("initial", initial), ("final", final)):
        _need(
            all(np.isfinite(frame[name]).all() for name, _ in frames.FRAME_FIELDS),
            "finite complete " + label + " physical frame",
        )
        _need(
            not frame["xfrc_applied"].view("<u4").any()
            and not frame["qfrc_applied"].view("<u4").any(),
            "literal zero external/generalized assistance",
        )
    home = np.asarray(descriptor["initial_qpos"], dtype="<f4")
    _need(
        home.shape == (21,)
        and np.array_equal(
            initial["qpos"].view("<u4"), np.broadcast_to(home.view("<u4"), (WORLDS, 21))
        ),
        "literal initial home pose",
    )
    for name in frames.FIXED_INPUT_FIELDS:
        if name != "qpos":
            _need(
                not initial[name].view("<u4").any(),
                "literal initial positive-zero " + name,
            )
    clock = np.float32(0.0)
    for _ in range(SUBSTEPS):
        clock = np.float32(clock + np.float32(0.002))
    _need(
        np.array_equal(
            final["time"].view("<u4"),
            np.full((WORLDS,), clock, dtype="<f4").view("<u4"),
        ),
        "ten independently accumulated float32 2-ms physical clocks",
    )
    return initial, final


def checked_ledger(value):
    _need(
        type(value) is dict
        and set(value)
        == {
            "reward",
            "terminated",
            "timed_out",
            "episode_steps",
            "executed_steps",
            "live",
            "term_sums",
        },
        "complete retained tick ledger",
    )
    for name in ("reward",):
        a = np.asarray(value[name], dtype=np.float64)
        _need(
            type(value[name]) is list
            and all(type(v) is float for v in value[name])
            and a.shape == (WORLDS,)
            and np.isfinite(a).all()
            and np.array_equal(a, a.astype("<f4").astype(np.float64)),
            "finite complete retained " + name,
        )
    for name, expected in (("terminated", False), ("timed_out", False), ("live", True)):
        _need(
            type(value[name]) is list
            and len(value[name]) == WORLDS
            and all(type(v) is bool and v is expected for v in value[name]),
            "exact tick " + name,
        )
    for name in ("episode_steps", "executed_steps"):
        _need(
            type(value[name]) is list
            and len(value[name]) == WORLDS
            and all(type(v) is int and v == SUBSTEPS for v in value[name]),
            "accepted complete " + name,
        )
    terms = value["term_sums"]
    _need(
        type(terms) is dict
        and set(terms)
        == {
            "upright",
            "stillness",
            "height",
            "support",
            "motor",
            "joint_speed",
            "correction",
            "correction_change",
        },
        "complete retained dense-term ledger",
    )
    for name, values in terms.items():
        a = np.asarray(values, dtype=np.float64)
        _need(
            type(values) is list
            and all(type(v) is float for v in values)
            and a.shape == (WORLDS,)
            and np.isfinite(a).all()
            and np.array_equal(a, a.astype("<f4").astype(np.float64)),
            "finite retained dense term " + name,
        )
    return dict(
        reward_min=min(value["reward"]),
        reward_max=max(value["reward"]),
        independent_reward_reexecution=False,
    )


def _layout_check(layout):
    _need(
        type(layout) is list and len(layout) == 31, "complete bound live array layouts"
    )
    names = (
        "body_parentid",
        *(f"body_tree_{i}" for i in range(7)),
        "body_mass",
        "body_subtreemass",
        "body_rootid",
        "body_inertia",
        "dof_bodyid",
        "dof_parentid",
        "dof_armature",
        "jnt_type",
        "jnt_dofadr",
        "jnt_bodyid",
        "crb",
        "cinert",
        "qM",
        "cdof",
        "subtree_com",
        "xipos",
        "ximat",
        "xmat",
        "xanchor",
        "xaxis",
        "cfrc_int",
        "cacc",
        "cvel",
    )
    _need([v["name"] for v in layout] == list(names), "exact live array order")
    for row in layout:
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
            and row["device"] == "cuda:0"
            and row["contiguous"] is True
            and type(row["shape"]) is list
            and row["shape"]
            and all(type(v) is int and v > 0 for v in row["shape"])
            and type(row["strides"]) is list
            and len(row["strides"]) == len(row["shape"])
            and all(type(v) is int and v >= 0 and v % 4 == 0 for v in row["strides"])
            and type(row["dtype"]) is str
            and len(row["dtype"]) < 128,
            "typed actual live CUDA layout",
        )
        shape, strides, dtype = LAYOUTS[row["name"]]
        _need(
            row["shape"] == shape
            and row["strides"] == strides
            and row["dtype"] == f"<class '{dtype}'>",
            "exact frozen array shape/strides/dtype " + row["name"],
        )
    by_name = {r["name"]: r for r in layout}
    _need(
        by_name["cfrc_int"]["shape"] == [WORLDS, 16]
        and by_name["cfrc_int"]["strides"] == [384, 24]
        and by_name["subtree_com"]["shape"] == [WORLDS, 16]
        and by_name["subtree_com"]["strides"] == [192, 12]
        and by_name["qM"]["shape"] == [WORLDS, 20, 20]
        and by_name["qM"]["strides"] == [1600, 80, 4],
        "literal full RNE/CoM/dense mass-matrix layouts",
    )
    return layout


def checked_scope(scope, banks):
    _need(
        type(scope) is dict
        and scope["protocol"] == "microduck-serial-one-step-control-oct7-v1"
        and scope["status"] == "complete"
        and scope["fault"] is None
        and old.exact_int(scope["forward_calls"], FORWARDS)
        and old.exact_int(scope["constructor_forward_calls"], 1)
        and old.exact_int(scope["step_calls"], 1)
        and old.exact_int(scope["worlds"], WORLDS)
        and scope["device"] == "cuda:0"
        and type(scope["flags"]) is dict
        and scope["flags"] == dict(FLAGS, native_qualified=False)
        and all(v is False for v in scope["flags"].values()),
        "complete exact one-step control",
    )
    _need(
        scope["entry_counts"]
        == {k: FORWARDS for k in ("com", "crb", "rne_bias", "rne_post")}
        and scope["dispatch"]
        == {
            "forward_cap": 21,
            "com_original_logical_launches": 11,
            "com_underlying_launches": 13,
            "com_split_index": 5,
            "crb_logical_accumulation_launches": 7,
            "crb_underlying_accumulation_launches": 9,
            "crb_qM_launches": 1,
            "crb_split_index": 4,
            "rne_bias_logical_launches": 7,
            "rne_bias_underlying_launches": 9,
            "rne_split_index": 4,
            "split_body_order": [2, 7, 11],
            "sensory_rne_untouched_passthrough_calls": 21,
            "constructor_plus_one_step_only": True,
            "all_observation_readbacks_perturb_timing": True,
            "closed": True,
        },
        "exact closed aggregate dispatch, not a reused cap",
    )
    for key, bank in (
        ("com_initialized_sha256", "com.entries.bin"),
        ("com_weighted_sha256", "com.weighted.bin"),
        ("rne_input_sha256", "rne.entries.bin"),
        ("rne_output_sha256", "rne.outputs.bin"),
    ):
        _need(
            scope[key] == [sha256(v).hexdigest() for v in banks[bank]],
            "actual per-call whole snapshot hashes",
        )
    calls = scope["calls"]
    trace = [
        dict(forward_index=i, entry=entry, body_ids=group)
        for i in range(FORWARDS)
        for entry in ("com", "crb", "rne_bias")
        for group in GROUPS
    ]
    _need(
        scope["dispatch_trace"] == trace,
        "complete ordered actual accumulation-ID trace",
    )
    _need(
        type(calls) is list and len(calls) == FORWARDS,
        "complete bounded per-forward records",
    )
    layout, identity = None, None
    for i, row in enumerate(calls):
        expected_phase = (
            "constructor" if i == 0 else "step-pre" if i % 2 else "step-post"
        )
        _need(
            old.exact_int(row["forward_index"], i)
            and row["phase"] == expected_phase
            and row["device"] == "cuda:0"
            and type(row["stream"]) is int
            and row["stream"] >= 0
            and row["smooth_sha256"] == SMOOTH_SHA
            and row["forward_sha256"] == FORWARD_SHA
            and row["rne_sensory"] == "untouched-original-passthrough"
            and type(row["model_id"]) is int
            and row["model_id"] > 0
            and type(row["data_id"]) is int
            and row["data_id"] > 0,
            "actual ordered bound forward, source and sensory receipts",
        )
        current = _layout_check(row["array_layouts"])
        ids = (row["model_id"], row["data_id"], row["stream"])
        if layout is None:
            layout, identity = current, ids
        _need(
            current == layout and ids == identity,
            "same live model/data arrays and stream across physical steps",
        )
        for name, count in (("com", 11), ("rne_bias", 7)):
            receipt = row[name]
            _need(
                old.exact_int(receipt["logical_launches"], count)
                and old.exact_int(receipt["underlying_launches"], count + 2)
                and receipt["split_index"] == (5 if name == "com" else 4)
                and receipt["split_body_order"] == [2, 7, 11]
                and receipt["underlying_body_groups"] == GROUPS
                and receipt["observed_body_groups"] == GROUPS,
                "complete literal accumulation groups/counts " + name,
            )
        crb = row["crb"]
        _need(
            crb
            == {
                "logical_accumulation_launches": 7,
                "underlying_accumulation_launches": 9,
                "qM_launches": 1,
                "split_index": 4,
                "split_body_order": [2, 7, 11],
                "underlying_body_groups": GROUPS,
                "observed_body_groups": GROUPS,
            },
            "complete literal CRB dispatch and qM",
        )
        _need(
            row["com"]["initialized_sha256"]
            == sha256(banks["com.entries.bin"][i]).hexdigest()
            and row["com"]["weighted_sha256"]
            == sha256(banks["com.weighted.bin"][i]).hexdigest()
            and row["rne_bias"]["input_sha256"]
            == sha256(banks["rne.entries.bin"][i]).hexdigest()
            and row["rne_bias"]["output_sha256"]
            == sha256(banks["rne.outputs.bin"][i]).hexdigest()
            and row["rne_bias"]["input_output_alias"] is True
            and row["rne_bias"]["readback_and_device_sync_perturbs_timing"] is True
            and row["rne_bias"]["input_boundary"]
            == "after-original-cfrc-before-first-backward"
            and row["rne_bias"]["output_boundary"]
            == "after-original-backward-before-original-qfrc-bias",
            "actual initialized live boundaries, alias and complete hashes",
        )
        force = {v["name"]: v for v in current}["cfrc_int"]
        _need(
            old.exact_int(row["rne_bias"]["bytes_per_snapshot"], RNE_BYTES)
            and row["rne_bias"]["snapshot_layout"]
            == {
                "object_id": force["object_id"],
                "ptr": force["ptr"],
                "shape": force["shape"],
                "strides": force["strides"],
                "dtype": "spatial_vectorf",
                "device": "cuda:0",
                "stream": row["stream"],
                "contiguous": True,
            },
            "complete RNE snapshot bound to actual live force storage",
        )
    return True


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
    caps = old.SERVICE_CAPS
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
    # Independent execution/consistency checks above fail closed, not a tolerance.
    result = dict(
        protocol=PROTOCOL,
        source=source,
        case_order=list(CASE_ORDER),
        cases=case_rows,
        arithmetic_decision="moving-reductions-reference-exact"
        if arithmetic
        else "moving-reductions-reference-negative",
        execution_decision="one-nominal-tick-consistent",
        paired_decision="one-tick-paired-exact"
        if paired
        else "one-tick-paired-negative",
        decision="fresh-one-step-serial-repeat-exact"
        if arithmetic and paired
        else "fresh-one-step-serial-repeat-negative",
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
