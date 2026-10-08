"""Synthetic standard-library packet and restorer checks; no runtime imports."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import textwrap

import pytest

from mjlab_microduck import stance_solver_init_control as control
from mjlab_microduck import stance_solver_scratch as scratch


def _packet():
    raw = bytearray()
    fields, layouts = {}, {}
    pointer = 0x100000
    for name in control.SOLVER_INIT_ORDER:
        logical, host_shape, host_dtype = control.SOLVER_INIT_SPECS[name]
        width = 1 if host_dtype == "bool" else 4
        size = width
        for dimension in host_shape:
            size *= dimension
        fields[name] = {
            "offset": len(raw),
            "bytes": size,
            "shape": list(host_shape),
            "dtype": {"bool": "|b1", "int32": "<i4", "float32": "<f4"}[host_dtype],
        }
        payload = bytearray(size)
        if name == "data.nefc":
            payload[:] = struct.pack("<64i", *([46] * 64))
        elif name == "contact.nacon":
            payload[:] = struct.pack("<i", 512)
        elif name == "efc.J":
            payload[:4] = struct.pack("<f", 1.0)
        raw.extend(payload)
        item_width = width
        tail = host_shape[len(logical):]
        for dimension in tail:
            item_width *= dimension
        strides, stride = [], item_width
        for dimension in reversed(logical):
            strides.insert(0, stride)
            stride *= dimension
        layouts[name] = {
            "object_id": pointer,
            "pointer": pointer if size else None,
            "span": size,
            "device": "cuda:0",
            "context": 9,
            "warp_dtype": scratch._warp_dtype(host_dtype, tail),
            "shape": list(logical),
            "strides": strides,
            "host_shape": list(host_shape),
            "host_dtype": host_dtype,
            "bytes": size,
        }
        pointer += max(size, 4) + 64
    digest = sha256(raw).hexdigest()
    snapshot = {
        "path": "solver-init/original/forward-04.initialized.bin",
        "bytes": len(raw),
        "sha256": digest,
        "fields": fields,
        "layouts": layouts,
        "forward": 4,
        "phase": "initialized-before-search",
        "context_object_id": 777,
    }
    return bytes(raw), snapshot, digest


class _Host:
    def __init__(self, shape, dtype, raw):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.nbytes = len(raw)
        self.raw = raw

    def tobytes(self, order="C"):
        assert order == "C"
        return self.raw


class _Array:
    def __init__(self, shape, dtype, device, pointer, *, host_shape=None,
                 host_dtype=None, raw=b""):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ptr = pointer
        self.host_shape = host_shape or shape
        self.host_dtype = host_dtype or dtype
        self.raw = raw
        width = 1 if "bool" in dtype else 4
        stride = width
        strides = []
        for dimension in reversed(shape):
            strides.insert(0, stride)
            stride *= dimension
        self.strides = tuple(strides)

    def numpy(self):
        return _Host(self.host_shape, self.host_dtype, self.raw)


def _decoded(elliptic_type=7):
    raw, snapshot, digest = _packet()
    return scratch.decode_packet(
        raw, snapshot, digest, control.SOLVER_RECIPE, elliptic_type
    )


def _elliptic_packet(*, contact_id=0, dimension=1, address=0):
    raw, snapshot, _ = _packet()
    mutated = bytearray(raw)
    patches = (
        ("efc.type", struct.pack("<i", 7)),
        ("efc.id", struct.pack("<i", contact_id)),
        ("contact.nacon", struct.pack("<i", 1)),
        ("contact.dim", struct.pack("<i", dimension)),
        ("contact.efc_address", struct.pack("<i", address)),
    )
    for name, value in patches:
        offset = snapshot["fields"][name]["offset"]
        mutated[offset:offset + len(value)] = value
    digest = sha256(mutated).hexdigest()
    snapshot["sha256"] = digest
    return bytes(mutated), snapshot, digest


def test_import_is_inert_and_packet_schema_is_frozen():
    assert len(control.SOLVER_INIT_ORDER) == 66
    code = (
        "import sys; from mjlab_microduck import stance_solver_scratch; "
        "assert not any(name in sys.modules for name in "
        "('warp', 'torch', 'mujoco', 'mujoco_warp._src.solver'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr.decode()


def test_isolated_real_warp_cpu_staging_and_restore_without_solver(tmp_path):
    raw, snapshot, digest = _packet()
    script = textwrap.dedent("""\
        import json, sys
        import numpy as np
        import warp as wp
        from mjlab_microduck import stance_solver_scratch as scratch
        from mjlab_microduck.stance_solver_init_control import SOLVER_RECIPE
        snapshot, digest = json.loads(sys.stdin.buffer.readline())
        raw = sys.stdin.buffer.read()
        packet = scratch.decode_packet(raw, snapshot, digest, SOLVER_RECIPE, 7)
        wp.init()
        device = wp.get_device('cpu')
        def forbidden(*args, **kwargs):
            raise AssertionError('CPU allocation/copy test must not launch a kernel')
        wp.launch = forbidden
        arrays = {}
        for name in scratch.RESTORE_ORDER:
            logical, host, dtype = packet.specs[name]
            arrays[name] = wp.empty(logical, dtype=getattr(wp, dtype), device=device)
        def stage(name, raw, logical, host, dtype, warp_dtype):
            data = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(host).copy()
            return wp.array(data, dtype=arrays[name].dtype, device=device)
        def guard():
            assert wp.launch is forbidden
            assert 'mujoco_warp._src.solver' not in sys.modules
        receipt = scratch.DenseSolverScratchRestorer(
            packet, arrays, device, None, stage=stage, copy=wp.copy,
            synchronize=lambda stream: wp.synchronize_device(device), guard=guard,
        ).restore()
        for name, value in arrays.items():
            assert value.numpy().tobytes(order='C') == packet.fields[name], name
        assert all(value is False for value in receipt['flags'].values())
        print(json.dumps({'fields': len(arrays), 'device': str(device),
                          'solver_imported': False, 'kernel_launches': 0}))
        """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               WARP_CACHE_PATH=str(tmp_path / "private-cpu-cache"))
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            input=json.dumps([snapshot, digest]).encode() + b"\n" + raw,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    evidence = json.loads(result.stdout.decode().strip().splitlines()[-1])
    assert evidence == {"fields": 24, "device": "cpu", "solver_imported": False,
                        "kernel_launches": 0}


def test_decode_retains_whole_raw_packet_and_all_flags_false():
    decoded = _decoded()
    assert decoded.raw
    assert decoded.sha256 == sha256(decoded.raw).hexdigest()
    assert set(scratch.RESTORE_ORDER) <= set(decoded.fields)
    assert all(value is False for value in scratch.FLAGS.values())


def test_captured_mat33_dtype_uses_warp_metadata_spelling():
    # Independent literal from the retained capture convention, not the helper
    # used by the synthetic fixture builder.
    assert scratch._warp_dtype("float32", (3, 3)) == "<class 'warp._src.types.mat33f'>"
    raw, snapshot, digest = _packet()
    assert snapshot["layouts"]["context.frame"]["warp_dtype"] == (
        "<class 'warp._src.types.mat33f'>"
    )
    snapshot["layouts"]["context.frame"]["warp_dtype"] = (
        "<class 'mujoco_warp._src.types.mat33f'>"
    )
    with pytest.raises(ValueError, match="retained allocation layout matches frozen schema context.frame"):
        scratch.decode_packet(raw, snapshot, digest, control.SOLVER_RECIPE, 7)


def test_decode_refuses_bad_whole_packet_anchor_before_fields():
    raw, snapshot, _ = _packet()
    with pytest.raises(ValueError, match="whole supplied"):
        scratch.decode_packet(raw, snapshot, "0" * 64, control.SOLVER_RECIPE, 7)


def test_decode_requires_exact_recipe_and_no_active_elliptic_rows():
    raw, snapshot, digest = _packet()
    wrong = dict(control.SOLVER_RECIPE, cone=1)
    with pytest.raises(ValueError, match="exact dense pyramidal"):
        scratch.decode_packet(raw, snapshot, digest, wrong, 7)

    # Make the first active row elliptic with an in-range contact, dimension,
    # and EFC address. It must still be refused because friction is not retained.
    raw, snapshot, digest = _elliptic_packet()
    with pytest.raises(ValueError, match="active elliptic rows require"):
        scratch.decode_packet(
            raw, snapshot, digest, control.SOLVER_RECIPE, 7
        )


@pytest.mark.parametrize("changes", [
    {"contact_id": 1}, {"dimension": 6}, {"address": 512},
])
def test_decode_rejects_elliptic_rows_without_dereferencing_contact_fields(changes):
    raw, snapshot, digest = _elliptic_packet(**changes)
    with pytest.raises(ValueError, match="active elliptic rows require"):
        scratch.decode_packet(raw, snapshot, digest, control.SOLVER_RECIPE, 7)


def test_decode_rejects_nonmeaningful_zero_count_or_zero_jacobian():
    raw, snapshot, digest = _packet()
    mutated = bytearray(raw)
    count_offset = snapshot["fields"]["data.nefc"]["offset"]
    mutated[count_offset:count_offset + 256] = bytes(256)
    digest = sha256(mutated).hexdigest()
    snapshot["sha256"] = digest
    with pytest.raises(ValueError, match="active dense constraint"):
        scratch.decode_packet(bytes(mutated), snapshot, digest, control.SOLVER_RECIPE, 7)

    raw, snapshot, digest = _packet()
    mutated = bytearray(raw)
    j_offset = snapshot["fields"]["efc.J"]["offset"]
    mutated[j_offset:j_offset + snapshot["fields"]["efc.J"]["bytes"]] = bytes(
        snapshot["fields"]["efc.J"]["bytes"]
    )
    digest = sha256(mutated).hexdigest()
    snapshot["sha256"] = digest
    with pytest.raises(ValueError, match="nonzero active dense Jacobian"):
        scratch.decode_packet(bytes(mutated), snapshot, digest, control.SOLVER_RECIPE, 7)


def test_one_shot_restore_copies_exact_fields_on_explicit_stream():
    packet = _decoded()
    device, stream = object(), object()
    arrays = {}
    pointer = 0x400000
    for name in scratch.RESTORE_ORDER:
        logical, _host, _dtype = packet.specs[name]
        arrays[name] = _Array(
            tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer
        )
        pointer += packet.layouts[name]["span"] + 64
    staged, copies, syncs, guards = [], [], [], []

    stage_pointer = [0x800000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        value = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                       host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        staged.append((name, raw, host_shape, host_dtype))
        return value

    def copy(destination, source, *, stream):
        copies.append((destination, source, stream))

    def synchronize(actual_stream):
        syncs.append(actual_stream)

    def guard():
        guards.append(True)

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, stream, stage=stage, copy=copy,
        synchronize=synchronize, guard=guard,
    )
    receipt = restorer.restore()
    assert receipt["restored_fields"] == list(scratch.RESTORE_ORDER)
    assert receipt["solver_called"] is False
    assert receipt["forward_called"] is False
    assert len(staged) == len(copies) == len(scratch.RESTORE_ORDER)
    assert all(row[2] is stream for row in copies)
    assert syncs == [stream]
    assert guards
    with pytest.raises(AttributeError, match="bindings are frozen"):
        restorer.stream = object()
    with pytest.raises(AttributeError, match="bindings are frozen"):
        restorer.copy = lambda *args, **kwargs: None
    with pytest.raises(AttributeError, match="bindings are frozen"):
        del restorer.guard
    with pytest.raises(ValueError, match="single-use"):
        restorer.restore()


def test_copy_failure_has_no_success_receipt():
    packet = _decoded()
    device = object()
    arrays, pointer = {}, 0x500000
    for name in scratch.RESTORE_ORDER:
        logical = packet.specs[name][0]
        arrays[name] = _Array(tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer)
        pointer += packet.layouts[name]["span"] + 64

    stage_pointer = [0x900000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        source = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                      host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        return source

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, object(), stage=stage,
        copy=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("copy failed")),
        synchronize=lambda stream: None,
        guard=lambda: None,
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        restorer.restore()
    with pytest.raises(ValueError, match="all scratch copies"):
        restorer.receipt()


def test_packet_cannot_be_forged_or_mutated_after_authentication():
    packet = _decoded()
    with pytest.raises((AttributeError, TypeError)):
        packet.sha256 = "0" * 64
    with pytest.raises(TypeError):
        packet.recipe["block_dim"]["ray"] = 1
    with pytest.raises(ValueError, match="only from authentication"):
        scratch.DecodedPacket(b"x", "0" * 64, {}, {}, {}, {})


def test_staging_bytes_and_identity_are_checked_before_any_copy():
    packet = _decoded()
    device, stream = object(), object()
    arrays, pointer = {}, 0xA00000
    for name in scratch.RESTORE_ORDER:
        logical = packet.specs[name][0]
        arrays[name] = _Array(tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer)
        pointer += packet.layouts[name]["span"] + 64
    sources, copies, stage_pointer = [], [], [0xB00000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        source = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                        host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        sources.append(source)
        if len(sources) == 2:
            sources[0].raw = b"x" * len(sources[0].raw)
        return source

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, stream, stage=stage,
        copy=lambda *args, **kwargs: copies.append(args),
        synchronize=lambda actual_stream: None,
        guard=lambda: None,
    )
    with pytest.raises(ValueError, match="whole CPU staging bytes"):
        restorer.restore()
    assert copies == []
    with pytest.raises(ValueError, match="all scratch copies"):
        restorer.receipt()


def test_staging_pointer_mutation_is_rejected_before_any_copy():
    packet = _decoded()
    device, stream = object(), object()
    arrays, pointer = {}, 0xE00000
    for name in scratch.RESTORE_ORDER:
        logical = packet.specs[name][0]
        arrays[name] = _Array(tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer)
        pointer += packet.layouts[name]["span"] + 64
    sources, copies, stage_pointer = [], [], [0xF00000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        source = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                        host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        sources.append(source)
        if len(sources) == 2:
            sources[0].ptr += 4096
        return source

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, stream, stage=stage,
        copy=lambda *args, **kwargs: copies.append(args),
        synchronize=lambda actual_stream: None,
        guard=lambda: None,
    )
    with pytest.raises(ValueError, match="identity/pointer/strides unchanged"):
        restorer.restore()
    assert copies == []


def test_staging_stride_mutation_is_rejected_before_any_copy():
    packet = _decoded()
    device, stream = object(), object()
    arrays, pointer = {}, 0x1100000
    for name in scratch.RESTORE_ORDER:
        logical = packet.specs[name][0]
        arrays[name] = _Array(tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer)
        pointer += packet.layouts[name]["span"] + 64
    sources, copies, stage_pointer = [], [], [0x1200000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        source = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                        host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        sources.append(source)
        if len(sources) == 2:
            sources[0].strides = tuple(stride + 4 for stride in sources[0].strides)
        return source

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, stream, stage=stage,
        copy=lambda *args, **kwargs: copies.append(args),
        synchronize=lambda actual_stream: None,
        guard=lambda: None,
    )
    with pytest.raises(ValueError, match="identity/pointer/strides unchanged"):
        restorer.restore()
    assert copies == []


def test_original_destination_mapping_cannot_change_after_binding():
    packet = _decoded()
    device = object()
    arrays, pointer = {}, 0xC00000
    for name in scratch.RESTORE_ORDER:
        logical = packet.specs[name][0]
        arrays[name] = _Array(tuple(logical), packet.layouts[name]["warp_dtype"], device, pointer)
        pointer += packet.layouts[name]["span"] + 64

    stage_pointer = [0xD00000]

    def stage(name, raw, logical, host_shape, host_dtype, warp_dtype):
        source = _Array(logical, warp_dtype, "cpu", stage_pointer[0],
                        host_shape=host_shape, host_dtype=host_dtype, raw=raw)
        stage_pointer[0] += len(raw) + 64
        return source

    restorer = scratch.DenseSolverScratchRestorer(
        packet, arrays, device, object(), stage=stage,
        copy=lambda *args, **kwargs: None,
        synchronize=lambda actual_stream: None,
        guard=lambda: None,
    )
    arrays["data.ne"] = object()
    with pytest.raises(ValueError, match="array mapping identity"):
        restorer.restore()
