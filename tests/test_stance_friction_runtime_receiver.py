"""Pure CPU checks for the independent one-tick runtime packet receiver."""

from hashlib import sha256
import struct

import pytest

from mjlab_microduck import stance_friction_runtime_receiver as receiver
from mjlab_microduck import stance_friction_runtime_probe as producer
from mjlab_microduck import stance_friction_prefix_cuda_receiver as component


def _anchor(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def _owner_window():
    start = int((producer.CUTOFF - 1000) * 10**9)
    declaration = {
        "owner_pid": 123,
        "started_utc_ns": start,
        "child_timeout_seconds": 540,
    }
    owner = {
        "child_observed": True,
        "child_ppid": 123,
        "finished_utc_ns": start + 100 * 10**9,
        "samples": [{"elapsed_seconds": 2.0, "child_pid": 456}],
    }
    return declaration, owner


def test_owner_window_binds_observed_child_and_bounded_time():
    declaration, owner = _owner_window()
    receiver.owner_window(declaration, owner, 456)


@pytest.mark.parametrize(
    "mutation",
    [
        "unobserved",
        "bool_parent",
        "foreign_parent",
        "duration",
        "reserve",
        "timeout",
        "nonfinite",
        "foreign_sample",
    ],
)
def test_owner_window_refuses_unobserved_unbounded_or_foreign_child(mutation):
    declaration, owner = _owner_window()
    if mutation == "unobserved":
        owner["child_observed"] = 1
    elif mutation == "bool_parent":
        owner["child_ppid"] = True
    elif mutation == "foreign_parent":
        owner["child_ppid"] = 122
    elif mutation == "duration":
        owner["finished_utc_ns"] = declaration["started_utc_ns"] + 601 * 10**9
    elif mutation == "reserve":
        declaration["started_utc_ns"] = int((producer.CUTOFF - 899) * 10**9)
        owner["finished_utc_ns"] = declaration["started_utc_ns"] + 10**9
    elif mutation == "timeout":
        declaration["child_timeout_seconds"] = True
    elif mutation == "nonfinite":
        owner["samples"][0]["elapsed_seconds"] = float("nan")
    else:
        owner["samples"][0]["child_pid"] = 789
    with pytest.raises(ValueError):
        receiver.owner_window(declaration, owner, 456)


def _write_inventory(root, values):
    root.mkdir()
    for name, raw in values.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return {name: _anchor(raw) for name, raw in values.items()}


def test_authenticate_returns_exact_external_leaf_bytes(tmp_path):
    root = tmp_path / "run"
    values = {"arm/entry.bin": b"\x00\x01payload", "receipt.json": b'{"ok":true}'}
    inventory = _write_inventory(root, values)
    assert receiver.authenticate(root, inventory) == values


@pytest.mark.parametrize("mutation", ["extra", "missing", "symlink", "special"])
def test_authenticate_refuses_nonexact_leaf_sets_and_nodes(tmp_path, mutation):
    root = tmp_path / "run"
    values = {"packet.bin": b"packet"}
    inventory = _write_inventory(root, values)
    if mutation == "extra":
        (root / "unexpected.bin").write_bytes(b"extra")
    elif mutation == "missing":
        (root / "packet.bin").unlink()
    elif mutation == "symlink":
        (root / "packet.bin").unlink()
        (root / "packet.bin").symlink_to(root / "target.bin")
        (root / "target.bin").write_bytes(b"packet")
        inventory["target.bin"] = _anchor(b"packet")
    else:
        (root / "socket").mkdir()
        (root / "socket" / "nested").symlink_to(root / "packet.bin")
    with pytest.raises((ValueError, OSError)):
        receiver.authenticate(root, inventory)


def test_authenticate_enforces_inventory_file_and_total_caps(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _write_inventory(root, {"a.bin": b"12", "b.bin": b"34"})
    monkeypatch.setattr(receiver, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="complete inventory"):
        receiver.authenticate(root, inventory)
    monkeypatch.setattr(receiver, "MAX_FILES", 10)
    monkeypatch.setattr(receiver, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError):
        receiver.authenticate(root, inventory)
    monkeypatch.setattr(receiver, "MAX_FILE_BYTES", 10)
    monkeypatch.setattr(receiver, "MAX_TOTAL_BYTES", 3)
    with pytest.raises(ValueError, match="bounded total"):
        receiver.authenticate(root, inventory)


@pytest.mark.parametrize("mutation", ["hash", "size"])
def test_authenticate_refuses_wrong_whole_file_hash_or_size(tmp_path, mutation):
    root = tmp_path / "run"
    inventory = _write_inventory(root, {"packet.bin": b"whole packet"})
    if mutation == "hash":
        inventory["packet.bin"]["sha256"] = "0" * 64
    else:
        inventory["packet.bin"]["bytes"] += 1
    with pytest.raises(ValueError):
        receiver.authenticate(root, inventory)


def test_verify_run_authenticates_every_leaf_before_json_decode(tmp_path, monkeypatch):
    root = tmp_path / "run"
    values = {
        "declaration.json": b"not-json",
        "child.json": b"child-bytes",
        "owner.json": b"owner-bytes",
    }
    inventory = _write_inventory(root, values)
    read_bytes = 0
    original_read = receiver.os.read
    original_json_packet = receiver.json_packet

    def counted_read(fd, size):
        nonlocal read_bytes
        chunk = original_read(fd, size)
        read_bytes += len(chunk)
        return chunk

    def check_then_decode(raw):
        assert read_bytes == sum(len(value) for value in values.values())
        return original_json_packet(raw)

    monkeypatch.setattr(receiver.os, "read", counted_read)
    monkeypatch.setattr(receiver, "json_packet", check_then_decode)
    with pytest.raises(ValueError):
        receiver.verify_run(
            root, inventory, expected_source={}, expected_tests_sha="0" * 64
        )
    assert read_bytes == sum(len(value) for value in values.values())


def test_json_packet_is_bounded_and_rejects_duplicate_or_nonfinite_json(monkeypatch):
    monkeypatch.setattr(receiver, "MAX_JSON_BYTES", 7)
    assert receiver.json_packet(b'{"x":1}') == {"x": 1}
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b"12345678", "{}"):
        with pytest.raises(ValueError):
            receiver.json_packet(raw)


def _section(raw, fields):
    return {
        "path": "candidate0/entry-00.bin",
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
        "fields": fields,
    }


def test_unpack_packet_uses_literal_order_offsets_shapes_and_raw_bytes():
    raw = struct.pack("<ff", 1.25, -2.5) + struct.pack("<ii", 11, 12)
    section = _section(
        raw,
        {
            "first": {"offset": 0, "bytes": 8, "shape": [2], "dtype": "<f4"},
            "second": {"offset": 8, "bytes": 8, "shape": [2], "dtype": "<i4"},
        },
    )
    values = receiver.unpack_packet(
        section, {section["path"]: raw}, order=("first", "second")
    )
    assert values["first"] == {"shape": [2], "dtype": "<f4", "raw": raw[:8]}
    assert values["second"] == {"shape": [2], "dtype": "<i4", "raw": raw[8:]}


@pytest.mark.parametrize(
    "change",
    [
        {"offset": True},
        {"bytes": True},
        {"shape": [True]},
        {"dtype": " <f4"},
        {"offset": 4},
    ],
)
def test_unpack_packet_rejects_boolints_wrong_offsets_and_dtype_whitespace(change):
    raw = struct.pack("<f", 3.0)
    fields = {"x": {"offset": 0, "bytes": 4, "shape": [1], "dtype": "<f4"}}
    fields["x"].update(change)
    section = _section(raw, fields)
    with pytest.raises(ValueError):
        receiver.unpack_packet(section, {section["path"]: raw}, order=("x",))


def test_unpack_packet_rejects_trailing_unclaimed_bytes_and_bad_section_binding():
    raw = struct.pack("<f", 3.0) + b"tail"
    fields = {"x": {"offset": 0, "bytes": 4, "shape": [1], "dtype": "<f4"}}
    section = _section(raw, fields)
    with pytest.raises(ValueError, match="trailing"):
        receiver.unpack_packet(section, {section["path"]: raw}, order=("x",))
    section["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        receiver.unpack_packet(section, {section["path"]: raw}, order=("x",))


def _carrier(shape, dtype="<f4"):
    size = 4
    for dimension in shape:
        size *= dimension
    return {"shape": list(shape), "dtype": dtype, "raw": bytes(size)}


def _layout_value(
    carrier,
    *,
    logical_shape=None,
    strides=None,
    pointer=4096,
    span=None,
    context=77,
    warp_dtype=None,
):
    logical_shape = (
        list(carrier["shape"]) if logical_shape is None else list(logical_shape)
    )
    if strides is None:
        item_size = 4
        for dimension in carrier["shape"][len(logical_shape) :]:
            item_size *= dimension
        strides = []
        width = item_size
        for dimension in reversed(logical_shape):
            strides.insert(0, width)
            width *= dimension
    if span is None:
        if any(dimension == 0 for dimension in logical_shape):
            span = 0
        else:
            item_size = 4
            for dimension in carrier["shape"][len(logical_shape) :]:
                item_size *= dimension
            span = item_size + sum(
                (dimension - 1) * stride
                for dimension, stride in zip(logical_shape, strides)
            )
    warp_dtype = (
        warp_dtype
        or {
            "<f4": "<class 'warp._src.types.float32'>",
            "<i4": "<class 'warp._src.types.int32'>",
        }[carrier["dtype"]]
    )
    return {
        "object_id": 5,
        "pointer": pointer,
        "span": span,
        "device": "cuda:0",
        "context": context,
        "warp_dtype": warp_dtype,
        "shape": logical_shape,
        "strides": strides,
        "host_shape": carrier["shape"],
        "host_dtype": "float32" if carrier["dtype"] == "<f4" else "int32",
        "bytes": len(carrier["raw"]),
    }


def test_layout_accepts_contiguous_and_leading_zero_stride_broadcast():
    device = {"context": 77}
    contiguous = _carrier([2, 3])
    assert receiver.layout(_layout_value(contiguous), contiguous, device) == (
        4096,
        4120,
    )
    broadcast = _carrier([2, 3])
    assert receiver.layout(
        _layout_value(broadcast, strides=[0, 4]), broadcast, device
    ) == (4096, 4108)


def test_layout_accepts_empty_zero_pointer_and_float_vectors():
    device = {"context": 77}
    empty = _carrier([0])
    assert receiver.layout(_layout_value(empty, pointer=0, span=0), empty, device) == (
        0,
        0,
    )
    null_layout = _layout_value(empty, pointer=None, span=0)
    assert receiver.layout(null_layout, empty, device) == (0, 0)
    assert null_layout["pointer"] is None
    vector = _carrier([2, 3, 2])
    value = _layout_value(
        vector,
        logical_shape=[2, 3],
        strides=[24, 8],
        span=48,
        warp_dtype="<class 'warp._src.types.vec2f'>",
    )
    assert receiver.layout(value, vector, device) == (4096, 4144)
    vector5 = _carrier([2, 3, 5])
    value5 = _layout_value(
        vector5,
        logical_shape=[2, 3],
        strides=[60, 20],
        span=120,
        warp_dtype="<class 'mujoco_warp._src.types.vec5f'>",
    )
    assert receiver.layout(value5, vector5, device) == (4096, 4216)


@pytest.mark.parametrize(
    "mutation",
    ["context", "span", "dtype", "pointer", "null", "stride", "shape"],
)
def test_layout_refuses_wrong_context_dtype_pointer_span_or_layout(mutation):
    carrier = _carrier([2, 3])
    value = _layout_value(carrier)
    if mutation == "context":
        value["context"] = 78
    elif mutation == "span":
        value["span"] += 4
    elif mutation == "dtype":
        value["warp_dtype"] = "<class 'warp._src.types.int32'>"
    elif mutation == "pointer":
        value["pointer"] = 0
    elif mutation == "null":
        value["pointer"] = None
    elif mutation == "stride":
        value["strides"] = [4, 12]
    else:
        value["shape"] = [3, 2]
    with pytest.raises(ValueError):
        receiver.layout(value, carrier, {"context": 77})


def _unit(mode="run", source="a" * 40):
    return {
        "Id": f"microduck-friction-runtime-tick-{mode}-{source[:12]}.service",
        "MainPID": "456",
        "RuntimeMaxUSec": "10min" if mode == "run" else "11min",
        "MemoryMax": str(6 * 1024**3),
        "CPUQuotaPerSecUSec": "2s",
        "TasksMax": "64",
        "Nice": "10",
        "LimitFSIZE": str(16 * 1024**2 if mode == "run" else 64 * 1024**2),
        "Restart": "no",
        "KillMode": "control-group",
        "InvocationID": "b" * 32,
    }


@pytest.mark.parametrize("mode", ["run", "tests"])
def test_bounded_unit_accepts_exact_source_derived_service_caps(mode):
    assert receiver.bounded_unit(_unit(mode), "a" * 40, mode, 456) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("TasksMax", True),
        ("InvocationID", True),
        ("MainPID", 456),
        ("Restart", "always"),
    ],
)
def test_bounded_unit_rejects_truthy_wrong_types_and_unbounded_properties(key, value):
    unit = _unit()
    unit[key] = value
    with pytest.raises(ValueError):
        receiver.bounded_unit(unit, "a" * 40, "run", 456)


def test_bounded_unit_rejects_bool_owner_pid():
    unit = _unit()
    unit["MainPID"] = "True"
    with pytest.raises(ValueError):
        receiver.bounded_unit(unit, "a" * 40, "run", True)


def test_bounded_unit_rejects_unpinned_source_and_mode():
    with pytest.raises(ValueError):
        receiver.bounded_unit(_unit("run", "a" * 40), "b" * 40, "run", 456)
    with pytest.raises(ValueError):
        receiver.bounded_unit(_unit("other"), "a" * 40, "other", 456)


def _source_binding():
    return {
        "source": "a" * 40,
        "tree": "b" * 40,
        "branch": component.BRANCH,
        "leaves": {
            "src/example.py": {
                "bytes": 3,
                "git_blob": "c" * 40,
                "sha256": "d" * 64,
            }
        },
    }


def _environment_alias():
    return {
        "path": component.ROOT + "/.venv",
        "target": component.VENV_TARGET,
        "device": 1,
        "inode": 1794097,
        "bytes": len(component.VENV_TARGET.encode()),
        "mtime_ns": 2,
        "ctime_ns": 3,
    }


def _junit(tests=2, *, skipped=0, errors=0, failures=0):
    return (
        f'<testsuites><testsuite tests="{tests}" skipped="{skipped}" '
        f'errors="{errors}" failures="{failures}"/></testsuites>'
    ).encode()


def _cpu_proof_fixture(monkeypatch):
    monkeypatch.setattr(producer, "EXPECTED_TESTS", 2)
    files = [f"tests/test_explicit_fixture_{index:02d}.py" for index in range(90)]
    monkeypatch.setattr(
        producer, "TEST_FILES_SHA256", sha256(receiver.canonical(files)).hexdigest()
    )
    source = _source_binding()
    env = _environment_alias()
    tests_proof = {
        "source_binding": source,
        "flags": dict(receiver.FLAGS),
        "packages": dict(receiver.pins.VERSIONS),
        "files": files,
        "tests": 2,
        "environment_alias": env,
        "unit": _unit("tests"),
        "host_before": {},
        "host_after": {},
    }
    mac_proof = {
        "protocol": receiver.DATA_PROTOCOL + ":mac-tests",
        "source_binding": source,
        "flags": dict(receiver.FLAGS),
        "packages": dict(receiver.pins.VERSIONS),
        "files": files,
        "tests": 2,
        "python": "3.12.12",
        "machine": "arm64",
    }
    raw = {}
    inventories = {}
    for mode, proof in (("tests", tests_proof), ("mac-tests", mac_proof)):
        packets = {
            "receipt.json": receiver.canonical(proof),
            "pytest.log": b"2 passed\n",
            "junit.xml": _junit(),
        }
        inventory = {name: _anchor(packet) for name, packet in packets.items()}
        inventories[mode] = inventory
        raw.update({mode + "/" + name: packet for name, packet in packets.items()})
    declaration = {
        "tests_inventory": inventories["tests"],
        "tests_inventory_sha256": sha256(
            receiver.canonical(inventories["tests"])
        ).hexdigest(),
        "tests": tests_proof,
        "tests_terminal": {
            "MainPID": "0",
            "SubState": "exited",
            "Result": "success",
            "ExecMainStatus": "0",
            "InvocationID": tests_proof["unit"]["InvocationID"],
        },
        "mac_tests_inventory": inventories["mac-tests"],
        "mac_tests_inventory_sha256": sha256(
            receiver.canonical(inventories["mac-tests"])
        ).hexdigest(),
    }
    expected_tests_sha = declaration["tests_inventory_sha256"]
    return declaration, raw, source, expected_tests_sha


def test_cpu_proof_accepts_explicit_fixture_with_zero_skips_and_bound_receipt(
    monkeypatch,
):
    declaration, raw, source, expected_tests_sha = _cpu_proof_fixture(monkeypatch)
    receiver.cpu_proof(declaration, raw, source, expected_tests_sha)


def test_cpu_proof_missing_top_level_field_is_a_value_error(monkeypatch):
    declaration, raw, source, expected_tests_sha = _cpu_proof_fixture(monkeypatch)
    del declaration["tests_inventory_sha256"]
    with pytest.raises(ValueError):
        receiver.cpu_proof(declaration, raw, source, expected_tests_sha)


def test_verify_run_malformed_json_schema_is_a_value_error(tmp_path):
    root = tmp_path / "run"
    values = {name: b"{}" for name in ("declaration.json", "child.json", "owner.json")}
    inventory = _write_inventory(root, values)
    with pytest.raises(ValueError):
        receiver.verify_run(
            root, inventory, expected_source={}, expected_tests_sha="0" * 64
        )


@pytest.mark.parametrize("mutation", ["source", "flags", "collection", "zero-skips"])
def test_cpu_proof_rejects_source_flags_collection_and_junit_mutations(
    monkeypatch, mutation
):
    declaration, raw, source, expected_tests_sha = _cpu_proof_fixture(monkeypatch)
    if mutation == "source":
        proof = dict(declaration["tests"])
        proof["source_binding"] = {**source, "source": "e" * 40}
        declaration["tests"] = proof
        raw["tests/receipt.json"] = receiver.canonical(proof)
        inventory = declaration["tests_inventory"]
        inventory["receipt.json"] = _anchor(raw["tests/receipt.json"])
        declaration["tests_inventory_sha256"] = sha256(
            receiver.canonical(inventory)
        ).hexdigest()
        expected_tests_sha = declaration["tests_inventory_sha256"]
    elif mutation == "flags":
        proof = dict(declaration["tests"])
        proof["flags"] = {**proof["flags"], "training_authorized": 1}
        declaration["tests"] = proof
        raw["tests/receipt.json"] = receiver.canonical(proof)
        inventory = declaration["tests_inventory"]
        inventory["receipt.json"] = _anchor(raw["tests/receipt.json"])
        declaration["tests_inventory_sha256"] = sha256(
            receiver.canonical(inventory)
        ).hexdigest()
        expected_tests_sha = declaration["tests_inventory_sha256"]
    elif mutation == "collection":
        proof = dict(declaration["tests"])
        proof["files"] = list(proof["files"])
        proof["files"][-1] = "tests/unexpected.py"
        declaration["tests"] = proof
        raw["tests/receipt.json"] = receiver.canonical(proof)
        inventory = declaration["tests_inventory"]
        inventory["receipt.json"] = _anchor(raw["tests/receipt.json"])
        declaration["tests_inventory_sha256"] = sha256(
            receiver.canonical(inventory)
        ).hexdigest()
        expected_tests_sha = declaration["tests_inventory_sha256"]
    else:
        packet = _junit(skipped=1)
        raw["tests/junit.xml"] = packet
        declaration["tests_inventory"]["junit.xml"] = _anchor(packet)
        declaration["tests_inventory_sha256"] = sha256(
            receiver.canonical(declaration["tests_inventory"])
        ).hexdigest()
        expected_tests_sha = declaration["tests_inventory_sha256"]
    with pytest.raises(ValueError):
        receiver.cpu_proof(declaration, raw, source, expected_tests_sha)


def _binary_packet(path, order, *, dtypes=None):
    dtypes = dtypes or {}
    raw, fields = bytearray(), {}
    for name in order:
        dtype = dtypes.get(name, "<f4")
        fields[name] = {
            "offset": len(raw),
            "bytes": 4,
            "shape": [1],
            "dtype": dtype,
        }
        raw.extend(struct.pack("<i" if dtype == "<i4" else "<f", 0))
    packet = bytes(raw)
    return {
        "path": path,
        "bytes": len(packet),
        "sha256": sha256(packet).hexdigest(),
        "fields": fields,
    }, packet


def _observer_fixture(arm="candidate0"):
    input_order = receiver.numerical.INPUT_NAMES
    bank_order = receiver.numerical.BANK_NAMES
    int_fields = {
        "nf",
        "nefc",
        "efc_nnz",
        "type",
        "id",
        "row_nnz",
        "row_adr",
        "col_ind",
    }
    layouts = {}
    for prefix, names in (("input.", input_order), ("bank.", bank_order)):
        for name in names:
            dtype = "<i4" if name in int_fields else "<f4"
            carrier = _carrier([1], dtype)
            layouts[prefix + name] = _layout_value(
                carrier, pointer=0x1000 + len(layouts) * 16
            )
    selected_name = "original" if arm == "original" else "candidate"
    symbol = "_friction_dof" if arm == "original" else "ascending_friction_dof"
    selected = {
        "binding": {"observed_object_ids": {"kernel": 101}, "symbol": symbol},
        "runtime_entries": {"launch": {"object_id": 202, "code_id": 203}},
    }
    compiled = {selected_name: selected}
    recipe = {"control_scope": {"calls": []}}
    rows, raw = [], {}
    fixed_ids = {
        "kernel_object_id": 101,
        "kernel_function_id": 102,
        "kernel_code_id": 103,
        "launch_function_id": 202,
        "launch_code_id": 203,
        "make_function_id": 204,
        "make_code_id": 205,
        "model_object_id": 301,
        "data_object_id": 302,
    }
    for index in range(21):
        recipe["control_scope"]["calls"].append(
            {"model_id": 301, "data_id": 302, "stream": 404}
        )
        packets = {}
        for label, order in (("inputs", input_order), ("bank", bank_order)):
            for phase in ("before", "after"):
                path = f"{arm}/entry-{index:02d}.{label}.{phase}.bin"
                packet, packet_raw = _binary_packet(
                    path,
                    order,
                    dtypes={name: "<i4" for name in order if name in int_fields},
                )
                packets[label + "." + phase] = packet
                raw[path] = packet_raw
        rows.append(
            {
                "index": index,
                "arm": arm,
                "packets": packets,
                "layouts": layouts,
                "dim": [64, 20],
                "scalars": {
                    "nv": 20,
                    "disableflags": 0,
                    "is_sparse": False,
                    "njmax": 512,
                    "njmax_nnz": 10240,
                },
                "kernel": symbol,
                **fixed_ids,
                "stream": 404,
                "device_context": 77,
                "record_tape": False,
            }
        )
    record = {
        "protocol": receiver.CONTROL_PROTOCOL,
        "arm": arm,
        "entries": rows,
        "other_launches": [
            {"forward": index, "kernel": "unrelated"} for index in range(21)
        ],
        "owned_hooks_restored": True,
    }
    device = {"stream": 404, "context": 77}
    return record, raw, device, compiled, recipe


@pytest.mark.parametrize("arm", ["original", "candidate0", "candidate1"])
def test_observer_accepts_ordered_21_entry_receipt_tied_to_recipe_and_layouts(arm):
    record, raw, device, compiled, recipe = _observer_fixture(arm)
    entries = receiver.observer(record, raw, arm, device, compiled, recipe)
    assert len(entries) == 21
    assert list(entries[0]) == [
        "inputs",
        "inputs_after",
        "before",
        "after",
        "scalars",
        "dim",
    ]
    assert entries[0]["dim"] == [64, 20]


@pytest.mark.parametrize(
    "mutation", ["count", "order", "receipt-tie", "packet-hash", "hook"]
)
def test_observer_rejects_wrong_call_count_order_recipe_tie_or_restoration(mutation):
    record, raw, device, compiled, recipe = _observer_fixture()
    if mutation == "count":
        record["entries"].pop()
    elif mutation == "order":
        record["entries"][0]["index"] = True
    elif mutation == "receipt-tie":
        recipe["control_scope"]["calls"][0]["model_id"] += 1
    elif mutation == "packet-hash":
        record["entries"][0]["packets"]["inputs.before"]["sha256"] = "0" * 64
    else:
        record["owned_hooks_restored"] = 1
    with pytest.raises(ValueError):
        receiver.observer(record, raw, "candidate0", device, compiled, recipe)
