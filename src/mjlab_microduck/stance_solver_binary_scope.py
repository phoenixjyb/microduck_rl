"""CPU-only scope of authenticated retained CUBINs, never loaded GPU code.

This deliberately narrow ELF64 profile inventories defined function symbols.
It neither disassembles instructions nor infers a solver binary from siblings.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct
import subprocess

PROTOCOL = "microduck-solver-binary-scope-oct8-v1"
BASE = "a0455e2b27584cbec7850b5ed19a3d67dbfe13b1"
OWN = {
    "src/mjlab_microduck/stance_solver_binary_scope.py",
    "tests/test_stance_solver_binary_scope.py",
    "docs/experiments/2026-10-08-solver-binary-scope.md",
}
TARGET = "update_constraint_init_qfrc_constraint_dense"
FLAGS = {
    "native_qualified": False,
    "full_window_qualified": False,
    "runtime_cause_proven": False,
    "training_authorized": False,
    "physical_acceptance": False,
}
FAMILIES = {
    "original": {
        "_equality_joint",
        "_zero_constraint_counts",
        "_friction_dof",
        "_equality_tendon",
        "_friction_tendon",
        "_equality_weld",
        "_limit_tendon",
        "_equality_connect",
        "_limit_ball",
        "_limit_slide_hinge",
    },
    "candidate": {"ascending_friction_dof"},
    "contact": {"_efc_contact_init__locals__kernel"},
}
SYMBOL = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)_[0-9a-f]{8}_cuda_kernel_forward")


def need(value, message):
    if not value:
        raise ValueError(message)


def elf_functions(raw):
    """Bounded ELF64 little-endian CUDA ET_EXEC symbol profile, not ISA proof."""
    need(type(raw) is bytes and 64 <= len(raw) <= 16 * 1024**2, "bounded ELF bytes")
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    need(h[0][:7] == b"\x7fELF\x02\x01\x01", "ELF64 little endian version one")
    need(h[1:4] == (2, 190, 1) and h[8] == 64, "CUDA executable ELF header")
    phoff, shoff, phsize, phnum, shsize, shnum, shstr = (
        h[5],
        h[6],
        h[9],
        h[10],
        h[11],
        h[12],
        h[13],
    )
    need(
        shsize == 64 and 1 <= shnum <= 4096 and 0 < shstr < shnum,
        "ordinary bounded section table",
    )
    need(shoff >= 64 and shoff + shnum * 64 <= len(raw), "whole section table")
    need(phnum <= 4096 and phsize in (0, 56), "bounded program table profile")
    need(
        (phnum == 0 and phoff == 0)
        or (phsize == 56 and phoff >= 64 and phoff + phnum * 56 <= len(raw)),
        "whole program table",
    )
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", raw, shoff + i * 64) for i in range(shnum)
    ]
    need(sections[0] == (0,) * 10, "ordinary null section; no extended numbering")
    for s in sections:
        need(s[8] == 0 or s[8] & (s[8] - 1) == 0, "power-of-two section alignment")
        if s[1] != 8:  # NOBITS occupies memory but no file payload.
            need(
                s[4] <= len(raw) and s[5] <= len(raw) - s[4],
                "whole file section payload",
            )
    need(sections[shstr][1] == 3, "section-name string table")

    def table(index):
        s = sections[index]
        return raw[s[4] : s[4] + s[5]]

    def string(strings, offset):
        need(0 <= offset < len(strings), "string offset within whole table")
        end = strings.find(b"\0", offset, min(len(strings), offset + 1025))
        need(end >= offset, "bounded terminated string")
        value = strings[offset:end]
        need(all(32 <= x < 127 for x in value), "printable ASCII symbol/section name")
        return value.decode("ascii")

    names = [string(table(shstr), s[0]) for s in sections]
    symtabs = [s for s in sections if s[1] == 2]
    need(len(symtabs) == 1, "one complete static symbol table")
    sym = symtabs[0]
    need(
        sym[9] == 24 and sym[5] % 24 == 0 and 1 <= sym[5] // 24 <= 65536,
        "bounded ELF64 symbols",
    )
    need(
        0 < sym[6] < shnum and sections[sym[6]][1] == 3,
        "symbol table linked to string table",
    )
    need(raw[sym[4] : sym[4] + 24] == b"\0" * 24, "null symbol")
    strings = table(sym[6])
    functions = []
    for off in range(sym[4], sym[4] + sym[5], 24):
        name, info, other, index, value, size = struct.unpack_from("<IBBHQQ", raw, off)
        text = string(strings, name)
        if info & 15 != 2:
            continue
        need(text and 0 < index < shnum, "defined nonempty function symbol")
        section = sections[index]
        need(section[1] == 1 and section[2] & 4, "function in executable PROGBITS")
        need(
            section[3] <= value
            and size > 0
            and value - section[3] + size <= section[5],
            "whole function extent in code section",
        )
        functions.append(
            {
                "symbol": text,
                "section": names[index],
                "value": value,
                "bytes": size,
                "section_bytes": section[5],
                "section_sha256": sha256(table(index)).hexdigest(),
                "binding": info >> 4,
                "other": other,
            }
        )
    need(
        functions and len({x["symbol"] for x in functions}) == len(functions),
        "nonempty unique defined functions",
    )
    return {
        "machine": 190,
        "elf_flags_hex": f"{h[7]:08x}",
        "section_count": shnum,
        "functions": sorted(functions, key=lambda x: x["symbol"]),
    }


def compiled_scope(child, raw, inventory, native_directory):
    """Scope-only helper; caller must first authenticate the full old receiver."""
    need(
        type(child) is dict and type(raw) is dict and type(inventory) is dict,
        "literal authenticated inputs",
    )
    compiled = child["compiled"]
    need(
        type(compiled) is dict and set(compiled) == set(FAMILIES),
        "three historical compiled roles",
    )
    need(
        type(native_directory) is str
        and re.fullmatch(
            r"/home/yanbo/work/microduck_rl-com-entry-20261006/artifacts/evaluations/solver-init-tick-run-[0-9a-f]{12}",
            native_directory,
        ),
        "historical native artifact root",
    )
    roles, leaves = {}, set()
    for role in sorted(FAMILIES):
        item = compiled[role]
        payloads, anchors = {}, {}
        for kind in ("binary", "metadata", "generated_source"):
            info = item[kind]
            need(
                type(info) is dict and set(info) == {"path", "bytes", "sha256"},
                "whole retained artifact anchor",
            )
            need(
                type(info["path"]) is str
                and type(info["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", info["sha256"]),
                "literal artifact path/hash",
            )
            p = Path(info["path"])
            need(
                p.parent == Path(native_directory) / ("compiled-" + role),
                "exact role artifact parent",
            )
            name = str(p.relative_to(native_directory))
            need(
                name in raw and name in inventory and name not in leaves,
                "unique inventoried artifact",
            )
            payload = raw[name]
            cap = {
                "binary": 16 * 1024**2,
                "metadata": 128 * 1024,
                "generated_source": 8 * 1024**2,
            }[kind]
            need(
                type(payload) is bytes and len(payload) <= cap,
                "bounded artifact payload",
            )
            anchor = {"bytes": len(payload), "sha256": sha256(payload).hexdigest()}
            need(
                type(info["bytes"]) is int
                and anchor == inventory[name] == {k: info[k] for k in anchor},
                "whole authenticated artifact bytes",
            )
            leaves.add(name)
            payloads[kind], anchors[kind] = payload, {"path": name, **anchor}
        need(
            anchors["binary"]["path"] == f"compiled-{role}/{role}.cubin",
            "literal CUBIN role",
        )
        parsed = elf_functions(payloads["binary"])

        def unique_object(pairs):
            need(len({k for k, _ in pairs}) == len(pairs), "no duplicate metadata keys")
            return dict(pairs)

        meta = json.loads(payloads["metadata"], object_pairs_hook=unique_object)
        need(type(meta) is dict and meta, "nonempty metadata symbol map")
        need(
            all(
                type(k) is str
                and k.endswith(
                    (
                        "_cuda_kernel_forward_smem_bytes",
                        "_cuda_kernel_backward_smem_bytes",
                    )
                )
                and type(v) is int
                and 0 <= v <= 1024**2
                for k, v in meta.items()
            ),
            "bounded metadata entries",
        )
        forwards = {
            k.removesuffix("_smem_bytes")
            for k in meta
            if k.endswith("_cuda_kernel_forward_smem_bytes")
        }
        need(
            len(meta) == 2 * len(forwards)
            and all(
                k.replace("_forward", "_backward") + "_smem_bytes" in meta
                for k in forwards
            ),
            "paired forward/backward metadata entries",
        )
        binary_forwards = {
            f["symbol"]
            for f in parsed["functions"]
            if f["symbol"].endswith("_cuda_kernel_forward")
        }
        source_forwards = re.findall(
            rb'extern "C" __global__ void ([A-Za-z_][A-Za-z_0-9]*_cuda_kernel_forward)\s*\(',
            payloads["generated_source"],
        )
        need(
            len(source_forwards) == len(set(source_forwards)),
            "unique generated entrypoints",
        )
        need(
            forwards == binary_forwards == {s.decode() for s in source_forwards},
            "whole metadata/source/CUBIN forward symbol agreement",
        )
        matches = [SYMBOL.fullmatch(s) for s in forwards]
        need(
            len(forwards) == len(FAMILIES[role])
            and all(matches)
            and {m.group(1) for m in matches} == FAMILIES[role],
            "exact retained role kernel families",
        )
        binding, load = item["binding"], item["explicit_load"]
        need(
            binding["symbol"] in forwards
            and binding["device_arch"] == item["output_arch"] == 120,
            "bound role symbol and recorded sm120 target",
        )
        need(
            type(load["fresh_cache_before"]) is bool
            and load["fresh_cache_before"]
            and item["module_options"]["fuse_fp"] is True,
            "recorded role fresh-cache/fuse settings",
        )
        for kind in ("binary", "metadata"):
            need(
                all(
                    binding[kind + "_" + key] == item[kind][key]
                    for key in ("path", "bytes", "sha256")
                ),
                "bound loaded-input file anchors",
            )
        need(
            all(
                binding[k] is False
                for k in (
                    "driver_jit_machine_code_observed",
                    "loaded_binary_bytes_observed",
                    "native_execution_qualified",
                    "training_authorized",
                    "physical_acceptance",
                )
            ),
            "opaque runtime bytes and non-admission",
        )
        roles[role] = {
            "artifacts": anchors,
            "elf": parsed,
            "forward_symbols": sorted(forwards),
            "bound_role_symbol": binding["symbol"],
            "recorded_fuse_fp": True,
            "recorded_fresh_cache_before": True,
            "target_present": any(TARGET in s for s in forwards),
        }
    need(
        {n for n in inventory if n.startswith("compiled-")} == leaves,
        "closed nine-leaf compiled inventory",
    )
    need(
        len(leaves) == 9 and not any(r["target_present"] for r in roles.values()),
        "target absent from all retained compiled roles",
    )
    return {
        "protocol": PROTOCOL,
        "target_module": "mujoco_warp._src.solver",
        "target_kernel": TARGET,
        "roles": roles,
        "compiled_leaf_count": 9,
        "forward_symbol_count": sum(len(r["forward_symbols"]) for r in roles.values()),
        "target_binary_bound": False,
        "instructions_disassembled": False,
        "loaded_machine_code_observed": False,
        "flags": dict(FLAGS),
        "decision": "missing-retained-solver-binary-binding",
        "interpretation": "absence from these authenticated retained roles only; not absence from the historical GPU context or proof of compiled FMA",
    }


def analysis_binding(root, source):
    from mjlab_microduck import stance_solver_init_row_view as rows

    need(
        root in rows.ROOTS
        and Path.cwd().resolve() == root
        and Path(__file__).resolve().parents[2] == root,
        "exact scope worktree",
    )
    need(
        rows._git(root, "rev-parse", "HEAD").decode().strip() == source
        and rows._git(root, "branch", "--show-current").decode().strip() == rows.BRANCH
        and not rows._git(root, "status", "--porcelain"),
        "clean exact scope source",
    )
    subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE, source],
        check=True,
        timeout=15,
    )
    need(
        set(rows._git(root, "diff", "--name-only", BASE, source).decode().splitlines())
        == OWN,
        "three new scope paths only",
    )
    return rows.git_binding(root, source, worktree=True)


def analyze_retained(root, source):
    from mjlab_microduck import stance_solver_init_row_view as rows

    binding = analysis_binding(root, source)
    short = rows.CAPTURE_SOURCE[:12]
    closeout = root / "artifacts/tools" / ("solver-init-tick-closeout-" + short)
    raw_root = root / "artifacts/evaluations" / ("solver-init-tick-run-" + short)
    inventory = rows.receiver.decode(
        rows.anchored(closeout / "inventory.json", 128 * 1024, rows.INVENTORY_SHA)
    )
    need(
        len(inventory) == rows.RAW_FILES
        and sum(x["bytes"] for x in inventory.values()) == rows.RAW_BYTES,
        "whole retained capture",
    )
    historical = rows.git_binding(root, rows.CAPTURE_SOURCE, worktree=False)
    original = rows.receiver.verify_run(
        raw_root,
        inventory,
        expected_source=historical,
        expected_tests_sha=rows.TESTS_SHA,
        historical_root=root,
    )
    need(
        rows.receiver.canonical(original)
        == rows.anchored(closeout / "receiver.json", 8 * 1024**2, rows.RECEIVER_SHA),
        "whole historical receiver reproduced",
    )
    raw = rows.receiver.authenticate(raw_root, inventory)
    child = rows.receiver.json_packet(raw["child.json"])
    scope = compiled_scope(
        child,
        raw,
        inventory,
        "/home/yanbo/work/microduck_rl-com-entry-20261006/artifacts/evaluations/solver-init-tick-run-"
        + short,
    )
    need(binding == analysis_binding(root, source), "scope source unchanged")
    return {
        "protocol": PROTOCOL,
        "analysis_source_binding": binding,
        "capture_source": rows.CAPTURE_SOURCE,
        "raw_inventory_sha256": rows.INVENTORY_SHA,
        "historical_receiver_sha256": rows.RECEIVER_SHA,
        "whole_receiver_recomputed": True,
        "raw_files": rows.RAW_FILES,
        "raw_bytes": rows.RAW_BYTES,
        "compiled_scope": scope,
        "flags": dict(FLAGS),
        "decision": scope["decision"],
    }


def main():
    import argparse
    import sys
    from mjlab_microduck import stance_solver_init_row_view as rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not sys.flags.optimize,
        "CUDA-hidden unoptimized scope audit",
    )
    print(
        rows.receiver.canonical(
            analyze_retained(Path.cwd().resolve(), args.source)
        ).decode()
    )


if __name__ == "__main__":
    main()
