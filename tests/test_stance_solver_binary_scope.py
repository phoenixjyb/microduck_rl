"""Synthetic byte-level refusal tests; no retained capture or CUDA dependency."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
from mjlab_microduck import stance_solver_binary_scope as audit

NATIVE = "/home/yanbo/work/microduck_rl-com-entry-20261006/artifacts/evaluations/solver-init-tick-run-e45a59c412bf"


def elf(symbols):
    names = b"\0.shstrtab\0.strtab\0.symtab\0.text\0"
    strings = b"\0" + b"".join(s.encode() + b"\0" for s in symbols)
    symbols_raw = b"\0" * 24
    for s in symbols:
        symbols_raw += struct.pack(
            "<IBBHQQ", strings.index(s.encode() + b"\0"), 18, 16, 4, 0, 16
        )
    data = bytearray(b"\0" * 64)
    sections = [(0,) * 10]
    for name, kind, flags, payload, link, entry in (
        (b".shstrtab", 3, 0, names, 0, 0),
        (b".strtab", 3, 0, strings, 0, 0),
        (b".symtab", 2, 0, symbols_raw, 2, 24),
        (b".text", 1, 6, b"X" * 16, 0, 0),
    ):
        sections.append(
            (
                names.index(name + b"\0"),
                kind,
                flags,
                0,
                len(data),
                len(payload),
                link,
                0,
                1,
                entry,
            )
        )
        data.extend(payload)
    shoff = len(data)
    data.extend(b"".join(struct.pack("<IIQQQQIIQQ", *s) for s in sections))
    ident = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        data,
        0,
        ident,
        2,
        190,
        1,
        0,
        0,
        shoff,
        0,
        64,
        0,
        0,
        64,
        5,
        1,
    )
    return bytes(data)


def change(raw, fmt, offset, value):
    b = bytearray(raw)
    struct.pack_into(fmt, b, offset, value)
    return bytes(b)


def fixture():
    child, raw, inventory = {"compiled": {}}, {}, {}
    for role, families in audit.FAMILIES.items():
        symbols = sorted(f + "_123abcff_cuda_kernel_forward" for f in families)
        meta = {
            s + suffix: 0
            for s in symbols
            for suffix in ("_smem_bytes", "_smem_bytes_backward_unused")
        }
        meta = {k: v for k, v in meta.items() if not k.endswith("_unused")}
        meta.update(
            {s.replace("_forward", "_backward") + "_smem_bytes": 0 for s in symbols}
        )
        payloads = {
            "binary": elf(symbols),
            "metadata": json.dumps(meta).encode(),
            "generated_source": b"\n".join(
                b'extern "C" __global__ void ' + s.encode() + b"(int x) {}"
                for s in symbols
            ),
        }
        item = {
            "binding": {"symbol": symbols[0], "device_arch": 120},
            "explicit_load": {"fresh_cache_before": True},
            "module_options": {"fuse_fp": True},
            "output_arch": 120,
        }
        for kind, payload in payloads.items():
            name = f"compiled-{role}/" + (
                role + ".cubin"
                if kind == "binary"
                else "module.meta"
                if kind == "metadata"
                else "module.cu"
            )
            anchor = {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            raw[name], inventory[name] = payload, anchor
            item[kind] = {"path": NATIVE + "/" + name, **anchor}
            if kind in ("binary", "metadata"):
                item["binding"].update(
                    {kind + "_" + k: v for k, v in item[kind].items()}
                )
        item["binding"].update(
            {
                k: False
                for k in (
                    "driver_jit_machine_code_observed",
                    "loaded_binary_bytes_observed",
                    "native_execution_qualified",
                    "training_authorized",
                    "physical_acceptance",
                )
            }
        )
        child["compiled"][role] = item
    return child, raw, inventory


def replace_payload(f, role, kind, payload):
    child, raw, inventory = f
    info = child["compiled"][role][kind]
    name = str(Path(info["path"]).relative_to(NATIVE))
    anchor = {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
    raw[name], inventory[name] = payload, anchor
    info.update(anchor)
    if kind in ("binary", "metadata"):
        child["compiled"][role]["binding"].update(
            {kind + "_" + k: v for k, v in anchor.items()}
        )


def test_functions_and_section_hash_are_exact():
    p = audit.elf_functions(elf(["one_cuda_kernel_forward", "helper"]))
    assert p["machine"] == 190 and p["section_count"] == 5
    assert [x["symbol"] for x in p["functions"]] == [
        "helper",
        "one_cuda_kernel_forward",
    ]
    assert all(
        x["section_sha256"] == sha256(b"X" * 16).hexdigest() for x in p["functions"]
    )


@pytest.mark.parametrize(
    "fmt,offset,value",
    [
        ("B", 0, 0),
        ("B", 4, 1),
        ("B", 5, 2),
        ("B", 6, 2),
        ("H", 16, 1),
        ("H", 18, 62),
        ("I", 20, 2),
        ("H", 52, 63),
        ("Q", 40, 2**63),
        ("H", 58, 63),
        ("H", 60, 0),
        ("H", 60, 65535),
        ("H", 62, 0),
        ("H", 62, 65535),
        ("H", 54, 55),
        ("H", 56, 4097),
        ("H", 56, 1),
    ],
)
def test_bad_headers_are_refused(fmt, offset, value):
    with pytest.raises(ValueError):
        audit.elf_functions(change(elf(["one"]), fmt, offset, value))


@pytest.mark.parametrize(
    "section,field,value",
    [
        (0, 1, 1),
        (1, 1, 1),
        (1, 4, 2**63),
        (1, 5, 2**63),
        (1, 0, 999999),
        (1, 8, 3),
        (3, 1, 1),
        (3, 6, 99),
        (3, 6, 4),
        (3, 9, 23),
        (3, 5, 25),
        (4, 1, 8),
        (4, 2, 2),
        (4, 5, 8),
    ],
)
def test_bad_sections_are_refused(section, field, value):
    b = bytearray(elf(["one"]))
    shoff = struct.unpack_from("<Q", b, 40)[0]
    s = list(struct.unpack_from("<IIQQQQIIQQ", b, shoff + section * 64))
    s[field] = value
    struct.pack_into("<IIQQQQIIQQ", b, shoff + section * 64, *s)
    with pytest.raises(ValueError):
        audit.elf_functions(bytes(b))


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "duplicate",
        "unterminated",
        "bad_ascii",
        "bad_index",
        "zero_size",
        "null_symbol",
    ],
)
def test_bad_symbols_are_refused(case):
    b = bytearray(elf(["one", "one"] if case == "duplicate" else ["one"]))
    shoff = struct.unpack_from("<Q", b, 40)[0]
    sym = struct.unpack_from("<IIQQQQIIQQ", b, shoff + 3 * 64)
    string = struct.unpack_from("<IIQQQQIIQQ", b, shoff + 2 * 64)
    if case == "empty":
        struct.pack_into("<I", b, sym[4] + 24, 0)
    if case == "unterminated":
        b[string[4] + string[5] - 1] = 88
    if case == "bad_ascii":
        b[string[4] + 1] = 255
    if case == "bad_index":
        struct.pack_into("<H", b, sym[4] + 24 + 6, 65535)
    if case == "zero_size":
        struct.pack_into("<Q", b, sym[4] + 24 + 16, 0)
    if case == "null_symbol":
        b[sym[4]] = 1
    with pytest.raises(ValueError):
        audit.elf_functions(bytes(b))


def test_scope_complete_and_nonmutating():
    f = fixture()
    before = deepcopy(f)
    r = audit.compiled_scope(*f, NATIVE)
    assert f == before
    assert r["forward_symbol_count"] == 12 and r["compiled_leaf_count"] == 9
    assert r["target_binary_bound"] is False and r["instructions_disassembled"] is False
    assert (
        not any(r["flags"].values())
        and r["decision"] == "missing-retained-solver-binary-binding"
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_role",
        "extra_role",
        "wrong_anchor",
        "raw_flip",
        "extra_leaf",
        "path_escape",
        "wrong_symbol",
        "wrong_arch",
        "not_fresh",
        "loaded_claim",
        "fuse_changed",
        "source_disagrees",
        "source_duplicate",
        "metadata_bool",
        "metadata_duplicate",
        "metadata_missing_backward",
        "wrong_family",
    ],
)
def test_bad_scope_refused(case):
    f = fixture()
    child, raw, inventory = f
    item = child["compiled"]["candidate"]
    if case == "missing_role":
        del child["compiled"]["contact"]
    if case == "extra_role":
        child["compiled"]["solver"] = item
    if case == "wrong_anchor":
        item["binary"]["sha256"] = "0" * 64
    if case == "raw_flip":
        raw["compiled-candidate/candidate.cubin"] += b"X"
    if case == "extra_leaf":
        inventory["compiled-solver/solver.cubin"] = {"bytes": 0, "sha256": "0" * 64}
    if case == "path_escape":
        item["binary"]["path"] = NATIVE + "/compiled-candidate/../candidate.cubin"
    if case == "wrong_symbol":
        item["binding"]["symbol"] = audit.TARGET + "_cuda_kernel_forward"
    if case == "wrong_arch":
        item["output_arch"] = 89
    if case == "not_fresh":
        item["explicit_load"]["fresh_cache_before"] = 1
    if case == "loaded_claim":
        item["binding"]["loaded_binary_bytes_observed"] = True
    if case == "fuse_changed":
        item["module_options"]["fuse_fp"] = False
    if case == "source_disagrees":
        replace_payload(f, "candidate", "generated_source", b"unrelated source")
    if case == "source_duplicate":
        replace_payload(
            f, "candidate", "generated_source", raw["compiled-candidate/module.cu"] * 2
        )
    if case.startswith("metadata_"):
        meta = json.loads(raw["compiled-candidate/module.meta"])
        key = next(iter(meta))
        if case == "metadata_bool":
            meta[key] = False
        if case == "metadata_missing_backward":
            meta = {key: 0}
        payload = json.dumps(meta).encode()
        if case == "metadata_duplicate":
            payload = b'{"' + key.encode() + b'":0,"' + key.encode() + b'":0}'
        replace_payload(f, "candidate", "metadata", payload)
    if case == "wrong_family":
        symbol = audit.TARGET + "_123abcff_cuda_kernel_forward"
        replace_payload(f, "candidate", "binary", elf([symbol]))
        replace_payload(
            f,
            "candidate",
            "metadata",
            json.dumps(
                {
                    symbol + "_smem_bytes": 0,
                    symbol.replace("_forward", "_backward") + "_smem_bytes": 0,
                }
            ).encode(),
        )
        replace_payload(
            f,
            "candidate",
            "generated_source",
            b'extern "C" __global__ void ' + symbol.encode() + b"(){}",
        )
        item["binding"]["symbol"] = symbol
    with pytest.raises(ValueError):
        audit.compiled_scope(*f, NATIVE)


def test_core_imports_no_gpu_stack():
    code = "import sys;from mjlab_microduck import stance_solver_binary_scope;assert not any(k.split('.')[0] in {'warp','torch','numpy','mujoco','mujoco_warp'} for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_source_fence_wrong_root():
    with pytest.raises(ValueError, match="worktree"):
        audit.analysis_binding(Path("/tmp"), "0" * 40)


@pytest.mark.parametrize("raw", [b"", b"\x7fELF", bytearray(b"X" * 64)])
def test_literal_whole_elf_required(raw):
    with pytest.raises(ValueError, match="bounded ELF"):
        audit.elf_functions(raw)


@pytest.mark.parametrize("optimized", [False, True])
def test_cli_refuses_visible_cuda_or_optimized_python(optimized):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if optimized else "0")
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-m",
            "mjlab_microduck.stance_solver_binary_scope",
            "--source",
            "0" * 40,
        ],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert b"CUDA-hidden unoptimized scope audit" in result.stderr
