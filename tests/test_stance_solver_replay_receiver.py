"""Synthetic packet/inventory fixtures only; never run CUDA or native tools."""

from hashlib import sha256
from copy import deepcopy
import json
import struct

import pytest

from mjlab_microduck import stance_solver_packets as packet_contract
from mjlab_microduck import stance_solver_disassembly_scope as disassembly
from mjlab_microduck import stance_solver_replay_receiver as receiver
from mjlab_microduck import stance_solver_scratch as scratch_contract
from mjlab_microduck import stance_solver_target_binding as target_contract


def packet_pair():
    before = bytearray(receiver.PACKET_BYTES)
    after = bytearray(receiver.PACKET_BYTES)
    offsets = receiver._FIELD_LAYOUT
    # World zero has two active rows. The full padded J/force banks remain zero.
    nefc_offset = offsets["nefc"][2]
    struct.pack_into("<i", before, nefc_offset, 2)
    struct.pack_into("<i", after, nefc_offset, 2)
    j_offset = offsets["J"][2]
    force_offset = offsets["force"][2]
    struct.pack_into("<f", before, j_offset + 0 * 20 * 4, 2.0)
    struct.pack_into("<f", before, j_offset + 1 * 20 * 4, -1.0)
    struct.pack_into("<f", before, force_offset, 3.0)
    struct.pack_into("<f", before, force_offset + 4, 4.0)
    struct.pack_into("<f", after, j_offset + 0 * 20 * 4, 2.0)
    struct.pack_into("<f", after, j_offset + 1 * 20 * 4, -1.0)
    struct.pack_into("<f", after, force_offset, 3.0)
    struct.pack_into("<f", after, force_offset + 4, 4.0)
    # qfrc[0,0] = 2*3 + (-1)*4 = 2; a done world preserves its sentinel.
    struct.pack_into("<f", after, offsets["qfrc_constraint"][2], 2.0)
    struct.pack_into("<b", before, offsets["done"][2] + 1, 1)
    struct.pack_into("<b", after, offsets["done"][2] + 1, 1)
    struct.pack_into("<f", before, offsets["qfrc_constraint"][2] + 20 * 4, 9.5)
    struct.pack_into("<f", after, offsets["qfrc_constraint"][2] + 20 * 4, 9.5)
    return bytes(before), bytes(after)


def phase_record(raw):
    fields = []
    for name, shape, _dtype, wire, width in packet_contract.SPECS:
        _, _, offset, size = receiver._FIELD_LAYOUT[name]
        chunk = raw[offset:offset + size]
        fields.append({"name": name, "shape": list(shape), "dtype": wire,
                       "offset": offset, "bytes": size,
                       "sha256": sha256(chunk).hexdigest()})
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest(), "fields": fields}


def packet_record(before, after):
    return {
        "protocol": packet_contract.PROTOCOL,
        "decision": "guarded-packets-only-not-qualification",
        "packets": {"before": phase_record(before), "after": phase_record(after)},
        "complete_pair": True,
        "input_bytes_unchanged": True,
        "packet_bytes": receiver.PACKET_BYTES,
        "pair_bytes": 2 * receiver.PACKET_BYTES,
        "copy_stream_handle": 71,
        "timing_changed_by_readback": True,
        "numerical_acceptance": False,
        "driver_loaded_code_observed": False,
        "flags": dict(target_contract.FLAGS),
    }


def test_cpu_reference_handles_ascending_rows_done_world_and_whole_banks():
    before, after = packet_pair()
    result = receiver.compare_packets(before, after, packet_record(before, after))
    assert result["input_banks_unchanged"] is True
    assert result["numerical_match"] is True
    assert result["worlds"][0]["result"] == "match"
    assert result["worlds"][0]["nefc"] == 2
    assert result["worlds"][1]["result"] == "done-row-preserved"
    assert result["worlds"][1]["done"] is True
    assert result["numerical_acceptance"] is False


def test_numerical_mismatch_is_reported_without_becoming_acceptance():
    before, after = packet_pair()
    damaged = bytearray(after)
    output_offset = receiver._FIELD_LAYOUT["qfrc_constraint"][2]
    struct.pack_into("<f", damaged, output_offset, 42.0)
    damaged = bytes(damaged)
    result = receiver.compare_packets(before, damaged, packet_record(before, damaged))
    assert result["numerical_match"] is False
    assert result["worlds"][0]["mismatches"] == 1
    assert result["worlds"][1]["result"] == "done-row-preserved"
    assert result["numerical_acceptance"] is False


def test_any_input_bank_change_fails_closed_even_when_packet_hashes_are_updated():
    before, after = packet_pair()
    changed = bytearray(after)
    force_offset = receiver._FIELD_LAYOUT["force"][2]
    struct.pack_into("<f", changed, force_offset, 8.0)
    changed = bytes(changed)
    with pytest.raises(ValueError, match="input banks unchanged"):
        receiver.compare_packets(before, changed, packet_record(before, changed))


def test_inventory_authenticates_exact_regular_leaf_set(tmp_path):
    leaf = tmp_path / "receipt.json"
    leaf.write_bytes(b"{}")
    inventory = {"receipt.json": {"bytes": 2, "sha256": sha256(b"{}").hexdigest()}}
    assert receiver.authenticate(tmp_path, inventory) == {"receipt.json": b"{}"}
    (tmp_path / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="exact complete artifact leaf inventory"):
        receiver.authenticate(tmp_path, inventory)


def test_inventory_rejects_symlinked_leaf(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"{}")
    (tmp_path / "receipt.json").symlink_to(outside)
    inventory = {"receipt.json": {"bytes": 2, "sha256": sha256(b"{}").hexdigest()}}
    with pytest.raises(ValueError, match="symlink"):
        receiver.authenticate(tmp_path, inventory)


def test_inventory_accepts_hashed_empty_regular_leaf(tmp_path):
    (tmp_path / "cache.lock").write_bytes(b"")
    inventory = {"cache.lock": {"bytes": 0, "sha256": sha256(b"").hexdigest()}}
    assert receiver.authenticate(tmp_path, inventory) == {"cache.lock": b""}


def test_complete_packet_size_and_schema_are_literal():
    assert receiver.PACKET_BYTES == 2_757_952
    assert receiver.MAX_TOTAL_BYTES == 128 * 1024**2
    assert receiver.MAX_FILES == 256


def _elf(symbol, code):
    section_names = b"\0.shstrtab\0.strtab\0.symtab\0.text\0"
    strings = b"\0" + symbol.encode("ascii") + b"\0"
    symbols = b"\0" * 24 + struct.pack(
        "<IBBHQQ", strings.index(symbol.encode("ascii") + b"\0"), 18, 16, 4, 0, len(code)
    )
    data = bytearray(b"\0" * 64)
    sections = [(0,) * 10]
    for name, kind, flags, payload, link, entry in (
        (b".shstrtab", 3, 0, section_names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, symbols, 2, 24),
        (b".text", 1, 6, code, 0, 0),
    ):
        sections.append((section_names.index(name + b"\0"), kind, flags, 0,
                         len(data), len(payload), link, 0, 1, entry))
        data.extend(payload)
    section_offset = len(data)
    data.extend(b"".join(struct.pack("<IIQQQQIIQQ", *row) for row in sections))
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into("<16sHHIQQQIHHHHHH", data, 0, ident, 2, 190, 1, 0, 0,
                     section_offset, 0, 64, 0, 0, 64, 5, 1)
    return bytes(data)


def _sass(symbol, code):
    words = [int.from_bytes(code[i:i + 8], "little") for i in (0, 8)]
    body = (f"/*0000*/ MOV R1, R2 ; /* 0x{words[0]:016x} */\n"
            f"                    /* 0x{words[1]:016x} */\n")
    return ("\n\tcode for sm_120\n"
            f"\t\tFunction : {symbol}\n"
            '\t.headerflags\t@"EF_CUDA_64BIT_ADDRESS EF_CUDA_SM120 '
            'EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"\n'
            + body + "\t\t..........\n").encode("ascii")


def _write_bundle(root, receipt, declaration, files):
    root.mkdir(parents=True)
    (root / "declaration.json").write_bytes(receiver._canonical(declaration))
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    receipt["declaration_sha256"] = sha256((root / "declaration.json").read_bytes()).hexdigest()
    (root / "receipt.json").write_bytes(receiver._canonical(receipt))
    inventory = {}
    for path in root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            inventory[str(path.relative_to(root))] = {
                "bytes": len(raw), "sha256": sha256(raw).hexdigest(),
            }
    return inventory


def _runtime_record(monkeypatch):
    warp_leaf_count, warp_total = 460, 9_292_904
    warp_base, warp_extra = divmod(warp_total, warp_leaf_count)
    warp_leaves = {
        f"source-{index:03d}.py": {
            "bytes": warp_base + (1 if index < warp_extra else 0), "sha256": "0" * 64,
        }
        for index in range(warp_leaf_count)
    }
    warp_sha = sha256(receiver._canonical(warp_leaves)).hexdigest()
    monkeypatch.setattr(receiver, "WARP_SOURCE_TREE_SHA256", warp_sha)
    mw_leaves = {f"module-{index:02d}.py": "1" * 64 for index in range(69)}
    mw_sha = sha256(receiver._canonical(mw_leaves)).hexdigest()
    monkeypatch.setattr(receiver, "MUJOCO_WARP_SOURCE_TREE_SHA256", mw_sha)
    alias = {"path": receiver.NATIVE_ROOT + "/.venv", "target": receiver.FROZEN_ENV,
             "device": 2096, "inode": 1_794_097,
             "bytes": len(receiver.FROZEN_ENV.encode()), "mtime_ns": 1, "ctime_ns": 2}
    libraries = {
        name: {"path": receiver.FROZEN_WARP_ROOT + "/bin/" + name,
               "bytes": size, "sha256": digest}
        for name, (size, digest) in receiver.RUNTIME_LIBRARY_PINS.items()
    }
    return {
        "environment_alias": alias,
        "packages": dict(receiver.PACKAGE_PINS),
        "libraries": libraries,
        "warp_sources": {"root": receiver.FROZEN_WARP_ROOT,
                         "leaves": warp_leaves, "sha256": warp_sha},
        "mujoco_warp_sources": {"root": receiver.FROZEN_MW_ROOT,
                                "leaves": mw_leaves, "sha256": mw_sha},
        "tool": {"path": receiver.FROZEN_ENV + "/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump",
                 "bytes": receiver.TOOL_BYTES, "sha256": receiver.TOOL_SHA256,
                 "version": "Cuda compilation tools, release 12.8, V12.8.55"},
    }


def _complete_bundle(root, monkeypatch):
    before, after = packet_pair()
    packet_pair_record = packet_record(before, after)
    packet_pair_record["copy_stream_handle"] = 99
    symbol = receiver.binary_scope.TARGET + "_1234abcd_cuda_kernel_forward"
    code = bytes(range(16))
    cubin = _elf(symbol, code)
    sass = _sass(symbol, code)
    selected = disassembly.verify_target_disassembly(cubin, sass)
    source_sha = "a" * 40
    native_root = ("/home/yanbo/work/microduck_rl-com-entry-20261006/artifacts/evaluations/"
                   "dense-solver-replay-" + source_sha[:12])
    compile_dir = native_root + "/compiled-dense-solver"
    leaves = {
        "compiled-dense-solver/target.cubin": cubin,
        "compiled-dense-solver/other.cubin": cubin,
        "compiled-dense-solver/target.meta": json.dumps(
            {symbol + "_smem_bytes": 0}, separators=(",", ":")
        ).encode(),
        "compiled-dense-solver/target.cu": (
            'extern "C" __global__ void ' + symbol + '() {}\n'
        ).encode(),
        "target.sass": sass,
        "packets/before.bin": before,
        "packets/after.bin": after,
        "supervision.json": json.dumps({
            "result": {
                "decision": "reviewed-owned-root-exited-no-native-qualification",
                "returncode": 0, "root": [202, 555], "elapsed": 1.0,
                "observed_session_members": [], "process_tree_retirement_proven": False,
                "native_qualified": False, "training_authorized": False,
            },
            "kernel_cgroup_only_owner": True,
            "unit_retirement_independently_required": True,
            "flags": receiver.target_contract.FLAGS,
        }, sort_keys=True, separators=(",", ":")).encode(),
    }
    def generated(name):
        raw = leaves["compiled-dense-solver/" + name]
        return {"path": compile_dir + "/" + name, "bytes": len(raw),
                "sha256": sha256(raw).hexdigest(), "identity": [1, 2, len(raw), 3, 4]}
    module_hash = "b" * 64
    object_ids = {"kernel": 11, "module": 12, "device": 13, "executable": 14, "hooks": 15}
    binding = {
        "protocol": "microduck-retained-cuda-artifact-binding-oct7-v1",
        "artifact_format": "cubin", "binary_sha256": sha256(cubin).hexdigest(),
        "binary_path": compile_dir + "/target.cubin", "binary_bytes": len(cubin),
        "metadata_sha256": sha256(leaves["compiled-dense-solver/target.meta"]).hexdigest(),
        "metadata_path": compile_dir + "/target.meta",
        "metadata_bytes": len(leaves["compiled-dense-solver/target.meta"]),
        "module_hash": module_hash, "context": 31, "module_handle": 32,
        "forward_handle": 33, "forward_smem_bytes": 0, "symbol": symbol,
        "device": "cuda:0", "device_arch": 120, "observed_object_ids": object_ids,
        "driver_jit_machine_code_observed": False, "loaded_binary_bytes_observed": False,
        "native_execution_qualified": False, "training_authorized": False,
        "physical_acceptance": False,
    }
    executable = {
        "protocol": receiver.executable_contract.PROTOCOL,
        "decision": "dense-solver-executable-prepared-not-dispatch-or-qualification",
        "source": {"path": receiver.FROZEN_ENV + "/lib/python3.12/site-packages/mujoco_warp/_src/solver.py",
                   "sha256": target_contract.SOLVER_SHA256, "identity": [1, 2, 3, 4, 5]},
        "target": receiver.binary_scope.TARGET,
        "device": {"alias": "cuda:0", "arch": 120, "context": 31, "object_id": 13},
        "compile": {"performed": True, "output_arch": 120, "use_ptx": False,
                    "module_hash": module_hash,
                    "generated": {"binary": generated("target.cubin"),
                                  "metadata": generated("target.meta"),
                                  "source": generated("target.cu")},
                    "target_compile_input": disassembly.select_target_cubin(cubin)},
        "offline_disassembly": selected, "loaded_binding": binding,
        "explicit_load": {"binary_path": compile_dir + "/target.cubin",
                          "metadata_path": compile_dir + "/target.meta", "output_arch": 120,
                          "block_dim": 256, "module_object_id": 12,
                          "executable_object_id": 14, "device_object_id": 13,
                          "returned_executable_is_cache_entry": True},
        "actual_dispatch_observed": False, "loaded_binary_bytes_observed": False,
        "driver_jit_machine_code_observed": False, "numerical_cause_proven": False,
        "flags": dict(receiver.target_contract.FLAGS),
    }
    shapes = ([64], [64, 512, 20], [64, 512], [64], [64, 20])
    widths = (4, 4, 4, 1, 4)
    dtype_ids = (200, 201, 201, 202, 201)
    layouts = [{"object_id": 100 + i, "pointer": 100_000 + i * 10_000_000,
                "shape": list(shape), "dtype_object_id": dtype_ids[i]}
               for i, shape in enumerate(shapes)]
    guard = {
        "protocol": "microduck-dense-solver-dispatch-guard-oct8-v1",
        "decision": "dense-solver-dispatch-guard-complete-not-qualification",
        "observed_target_calls": 1, "caller": "_update_constraint",
        "dispatch_line": 2197, "dimensions": [64, 20],
        "input_order": ["d.nefc", "d.efc.J", "d.efc.force", "d.njmax", "ctx.done"],
        "output_order": ["d.qfrc_constraint"], "layouts": layouts,
        "stream_object_id": 44, "stream_handle": 99, "binding": binding,
        "packets": packet_pair_record, "numerical_acceptance": False,
        "driver_loaded_code_observed": False, "flags": dict(receiver.target_contract.FLAGS),
    }
    service_name = "microduck-dense-solver-replay-" + source_sha[:12] + ".service"
    service = {"name": service_name, **receiver.UNIT_CAPS,
               "Id": service_name, "MainPID": "101", "ActiveState": "active",
               "InvocationID": "f" * 32,
               "ControlGroup": "/user.slice/user-1000.slice/user@1000.service/app.slice/" + service_name}
    services = {
        "system:recomo-ai-mission-vllm.service": "inactive",
        "user:recomo-ai-mission-vllm.service": "inactive",
        "system:recomo-ai-mission-subject-model-worker.service": "inactive",
        "user:recomo-ai-mission-subject-model-worker.service": "inactive",
        "recomo-filmbrain-observatory.service": {
            "ActiveState": "active", "MainPID": "521", "NRestarts": "0",
        },
        "recomo-filmbrain-video-playground.service": {
            "ActiveState": "active", "MainPID": "298048", "NRestarts": "0",
        },
    }
    telemetry = {"uuid": receiver.PINNED_GPU_UUID, "driver": receiver.GPU_DRIVER,
                 "total_mib": 24467, "used_mib": 1000, "free_mib": 23000,
                 "utilization_percent": 1, "temperature_c": 32}
    declaration = {
        "protocol": "microduck-dense-solver-replay-probe-oct8-v1", "source": source_sha,
        "owner_pid": 101, "native_root": native_root, "runtime": _runtime_record(monkeypatch),
        "service": service, "lease": {"device": 2096, "inode": 35886, "bytes": 0},
        "services": services, "bounds": dict(receiver.REPLAY_BOUNDS),
        "deadline_unix": 4_102_444_800.0, "exclusive_gpu_claimed": False,
        "baseline": {"wall_time_unix": 1.0, "wsl": dict(telemetry),
                     "windows": dict(telemetry), "windows_sampled_at": "2026-10-08T00:00:00+00:00",
                     "windows_active_engines": [], "counters_simultaneous": False},
        "flags": dict(receiver.target_contract.FLAGS),
        "source_binding": {"source": source_sha, "tree": "c" * 40,
                           "branch": "feat/athletics-obstacle-curriculum",
                           "leaves": [{"path": name, "bytes": 0 if index == 0 else 1,
                                       "git_blob": format(index + 1, "040x"),
                                       "sha256": format(index + 1, "064x")}
                                      for index, name in enumerate(sorted(receiver.SOURCE_SCOPE))]},
        "cpu_tests": {
            "path": native_root + "/cpu-evidence.json", "bytes": 1, "sha256": "e" * 64,
            "evidence": {
                "source": source_sha,
                "test_files": list(receiver.REPLAY_TESTS),
                "flags": dict(receiver.target_contract.FLAGS),
                "mac": {"path": receiver.NATIVE_ROOT + "/artifacts/tools/mac.xml",
                        "bytes": 1, "sha256": "1" * 64, "tests": 400},
                "native": {"path": receiver.NATIVE_ROOT + "/artifacts/tools/native.xml",
                           "bytes": 1, "sha256": "2" * 64, "tests": 400},
            },
        },
        "replay_input": {"inventory_sha256": "8dcb44a422530c4169e4133a0dfb858d24909dc9d59ac90d163f5037a014b625",
                         "metadata_sha256": "d19e16bcbf8152629b18c514ff627ff8417de1c4b4ed760d9ab0add30f867d91",
                         "raw_sha256": receiver.HISTORICAL_REPLAY_SHA256,
                         "raw_bytes": receiver.HISTORICAL_REPLAY_BYTES},
    }
    receipt = {
        "protocol": "microduck-dense-solver-replay-probe-oct8-v1", "source": source_sha,
        "declaration_sha256": "", "owner_pid": 101, "child_pid": 202,
        "device": {"alias": "cuda:0", "arch": 120, "context": 31,
                   "object_id": 13, "gpu_uuid": receiver.PINNED_GPU_UUID},
        "native_root": native_root, "executable": executable, "guard": guard,
        "scratch": {"protocol": scratch_contract.PROTOCOL,
                    "decision": "dense-solver-scratch-restored-no-dispatch-or-qualification",
                    "source_sha256": receiver.HISTORICAL_REPLAY_SHA256,
                    "source_bytes": receiver.HISTORICAL_REPLAY_BYTES,
                    "restored_fields": list(scratch_contract.RESTORE_ORDER),
                    "retained_source_bytes": True, "solver_called": False,
                    "forward_called": False, "flags": dict(scratch_contract.FLAGS)},
        "files": {"cubin": "compiled-dense-solver/target.cubin",
                  "metadata": "compiled-dense-solver/target.meta",
                  "generated_source": "compiled-dense-solver/target.cu",
                  "sass": "target.sass", "packet_before": "packets/before.bin",
                  "packet_after": "packets/after.bin"},
        "flags": dict(receiver.target_contract.FLAGS),
    }
    inventory = _write_bundle(root, receipt, declaration, leaves)
    return inventory


def test_receive_authenticates_and_binds_a_complete_synthetic_replay(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _complete_bundle(root, monkeypatch)
    result = receiver.receive(root, inventory)
    assert result["numerical"]["numerical_match"] is True
    assert result["executable_binding"]["offline_sass_matches_target_cubin"] is True
    assert result["qualification"] == receiver.FLAGS


def test_receive_rejects_wrong_pinned_gpu_even_with_reauthenticated_receipt(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _complete_bundle(root, monkeypatch)
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["device"]["gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
    raw = receiver._canonical(receipt)
    (root / "receipt.json").write_bytes(raw)
    inventory["receipt.json"] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="receipt device alias"):
        receiver.receive(root, inventory)


def test_receive_rejects_native_role_path_that_does_not_match_compile_record(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _complete_bundle(root, monkeypatch)
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["files"]["cubin"] = "compiled-dense-solver/other.cubin"
    raw = receiver._canonical(receipt)
    (root / "receipt.json").write_bytes(raw)
    inventory["receipt.json"] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="role path matches native-relative"):
        receiver.receive(root, inventory)


def test_receive_rejects_substituted_user_cgroup(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _complete_bundle(root, monkeypatch)
    declaration = json.loads((root / "declaration.json").read_bytes())
    declaration["service"]["ControlGroup"] = "/user.slice/other.service"
    raw = receiver._canonical(declaration)
    (root / "declaration.json").write_bytes(raw)
    inventory["declaration.json"] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    receipt = json.loads((root / "receipt.json").read_bytes())
    receipt["declaration_sha256"] = sha256(raw).hexdigest()
    raw = receiver._canonical(receipt)
    (root / "receipt.json").write_bytes(raw)
    inventory["receipt.json"] = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="bounded service caps"):
        receiver.receive(root, inventory)


def control_pair(reference_before, reference_after):
    offset = receiver._FIELD_LAYOUT["qfrc_constraint"][2]
    canary = receiver.overwrite_canary()
    before = reference_before[:offset] + canary
    after = bytearray(reference_after)
    done_offset = receiver._FIELD_LAYOUT["done"][2]
    for world in range(64):
        if before[done_offset + world]:
            after[offset + world * 80:offset + (world + 1) * 80] = canary[world * 80:(world + 1) * 80]
    return before, bytes(after)


def test_literal_finite_canary_and_complete_active_overwrite_with_done_preservation():
    raw = receiver.overwrite_canary()
    assert len(raw) == 5120 and struct.unpack_from("<4f", raw) == (4096, -4097, 4098, -4099)
    assert receiver.overwrite_manifest()["sha256"] == sha256(raw).hexdigest()
    rb, ra = packet_pair()
    cb, ca = control_pair(rb, ra)
    result = receiver.compare_overwrite_packets(rb, ra, packet_record(rb, ra), cb, ca, packet_record(cb, ca))
    assert result["active_worlds"] == 63 and result["done_worlds"] == 1
    assert result["overwritten_dofs"] == 1260
    assert result["qualification"] == receiver.FLAGS


@pytest.mark.parametrize("fault", ["no-op", "partial-write", "wrong-output", "nan-output",
    "canary-tamper", "cross-arm-input", "post-input", "done-row"])
def test_control_rejects_false_passes_even_with_updated_packet_hashes(fault):
    rb, ra = packet_pair()
    cb, ca = control_pair(rb, ra)
    offset = receiver._FIELD_LAYOUT["qfrc_constraint"][2]
    before, after = bytearray(cb), bytearray(ca)
    if fault == "no-op":
        after = bytearray(cb)
    elif fault == "partial-write":
        after[offset:offset + 4] = before[offset:offset + 4]
    elif fault == "wrong-output":
        struct.pack_into("<f", after, offset, 33.0)
    elif fault == "nan-output":
        struct.pack_into("<f", after, offset, float("nan"))
    elif fault == "canary-tamper":
        struct.pack_into("<f", before, offset, 4095.0)
    elif fault in ("cross-arm-input", "post-input"):
        position = receiver._FIELD_LAYOUT["J"][2] + (511 * 20 + 19) * 4  # inactive padding still bound
        struct.pack_into("<f", after, position, 7.0)
        if fault == "cross-arm-input":
            struct.pack_into("<f", before, position, 7.0)
    else:
        struct.pack_into("<f", after, offset + 20 * 4, 9.5)
    cb, ca = bytes(before), bytes(after)
    with pytest.raises(ValueError):
        receiver.compare_overwrite_packets(rb, ra, packet_record(rb, ra), cb, ca, packet_record(cb, ca))


def _refresh(root):
    return {str(path.relative_to(root)): dict(bytes=len(path.read_bytes()),
            sha256=sha256(path.read_bytes()).hexdigest()) for path in root.rglob("*") if path.is_file()}


def _complete_control_bundle(root, monkeypatch):
    _complete_bundle(root, monkeypatch)
    receipt = json.loads((root / "receipt.json").read_bytes())
    declaration = json.loads((root / "declaration.json").read_bytes())
    rb, ra = packet_pair()
    cb, ca = control_pair(rb, ra)
    control_guard = deepcopy(receipt["guard"])
    control_guard["packets"] = packet_record(cb, ca)
    control_guard["packets"]["copy_stream_handle"] = 99
    receipt["protocol"] = declaration["protocol"] = receiver.OVERWRITE_PROBE_PROTOCOL
    declaration["overwrite_control"] = receiver.overwrite_manifest()
    layout = control_guard["layouts"][4]
    receipt["overwrite_control"] = dict(manifest=receiver.overwrite_manifest(),
        guard=control_guard, scratch=deepcopy(receipt["scratch"]),
        files=dict(packet_before="control-before.bin", packet_after="control-after.bin"),
        copy=dict(bytes=5120, sha256=sha256(receiver.overwrite_canary()).hexdigest(),
            source_object_id=5000, source_pointer=7000,
            destination_object_id=layout["object_id"], destination_pointer=layout["pointer"],
            device_object_id=13, stream_object_id=44, stream_handle=99,
            dtype_object_id=layout["dtype_object_id"], shape=[64, 20],
            source_cpu_pinned=True, synchronized=True, flags=dict(target_contract.FLAGS)))
    (root / "control-before.bin").write_bytes(cb)
    (root / "control-after.bin").write_bytes(ca)
    raw = receiver._canonical(declaration)
    (root / "declaration.json").write_bytes(raw)
    receipt["declaration_sha256"] = sha256(raw).hexdigest()
    (root / "receipt.json").write_bytes(receiver._canonical(receipt))
    return _refresh(root)


def test_full_v2_receiver_binds_both_arms_without_qualifying_training(tmp_path, monkeypatch):
    root = tmp_path / "run"
    result = receiver.receive(root, _complete_control_bundle(root, monkeypatch))
    assert result["decision"] == "authenticated-two-arm-output-overwrite-diagnostic-only"
    assert result["protocol"] == receiver.OVERWRITE_RECEIVER_PROTOCOL
    assert result["overwrite_control"]["overwritten_dofs"] == 1260
    assert result["qualification"] == receiver.FLAGS


@pytest.mark.parametrize("fault", ["v1-masquerade", "missing-control", "wrong-copy-destination",
    "wrong-copy-stream", "wrong-copy-hash", "wrong-control-stream", "aliased-packets", "promoted-manifest",
    "same-copy-pointer"])
def test_full_control_envelope_rejects_substitutions(tmp_path, monkeypatch, fault):
    root = tmp_path / "run"
    _complete_control_bundle(root, monkeypatch)
    receipt = json.loads((root / "receipt.json").read_bytes())
    control = receipt["overwrite_control"]
    if fault == "v1-masquerade":
        receipt["protocol"] = receiver.PROBE_PROTOCOL
    elif fault == "missing-control":
        del receipt["overwrite_control"]
    elif fault == "wrong-copy-destination":
        control["copy"]["destination_pointer"] += 4
    elif fault == "wrong-copy-stream":
        control["copy"]["stream_handle"] += 1
    elif fault == "wrong-copy-hash":
        control["copy"]["sha256"] = "0" * 64
    elif fault == "wrong-control-stream":
        control["guard"]["stream_object_id"] += 1
    elif fault == "aliased-packets":
        control["files"]["packet_before"] = receipt["files"]["packet_before"]
    elif fault == "same-copy-pointer":
        control["copy"]["source_pointer"] = control["copy"]["destination_pointer"]
    else:
        control["manifest"]["qualification"]["training_authorized"] = True
    (root / "receipt.json").write_bytes(receiver._canonical(receipt))
    with pytest.raises(ValueError):
        receiver.receive(root, _refresh(root))


def test_control_packet_inventory_tampering_is_rejected_before_any_json(tmp_path, monkeypatch):
    root = tmp_path / "run"
    inventory = _complete_control_bundle(root, monkeypatch)
    (root / "control-after.bin").write_bytes(b"tampered")
    monkeypatch.setattr(receiver, "_json", lambda *args: pytest.fail("decode preceded authentication"))
    with pytest.raises(ValueError):
        receiver.receive(root, inventory)


def test_all_done_control_is_not_an_overwrite_demonstration():
    rb, _ = packet_pair()
    before = bytearray(rb)
    offset = receiver._FIELD_LAYOUT["done"][2]
    before[offset:offset + 64] = b"\x01" * 64
    rb = ra = bytes(before)
    cb, ca = control_pair(rb, ra)
    with pytest.raises(ValueError, match="nonempty complete active overwrite coverage"):
        receiver.compare_overwrite_packets(rb, ra, packet_record(rb, ra), cb, ca, packet_record(cb, ca))


def test_valid_reference_equal_to_canary_does_not_discriminate_noop():
    rb, ra = packet_pair()
    before, after = bytearray(rb), bytearray(ra)
    for raw in (before, after):
        struct.pack_into("<i", raw, receiver._FIELD_LAYOUT["nefc"][2], 1)
        struct.pack_into("<f", raw, receiver._FIELD_LAYOUT["J"][2], 4096.0)
        struct.pack_into("<f", raw, receiver._FIELD_LAYOUT["force"][2], 1.0)
    struct.pack_into("<f", after, receiver._FIELD_LAYOUT["qfrc_constraint"][2], 4096.0)
    rb, ra = bytes(before), bytes(after)
    cb, ca = control_pair(rb, ra)
    with pytest.raises(ValueError, match="discriminates a no-op"):
        receiver.compare_overwrite_packets(rb, ra, packet_record(rb, ra), cb, ca, packet_record(cb, ca))
