#!/usr/bin/env python3
"""Hash a complete portable full-evaluation capture as bytes only.

This module intentionally has no project imports and never deserializes tensors.
Its result is a transport inventory, not a numerical or model acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys


PROTOCOL = "football-b1n-wsl-portable-packed-full-evaluation-v1"
TRACE_PROTOCOL = "football-b1n-portable-packed-full-evaluation-trace-v1"
INVENTORY_SCHEMA = "microduck-portable-full-byte-inventory-v1"
TRAINING_SEEDS = (577, 587, 593)
ITERATIONS = (64, 128, 192, 255)
EVALUATION_SEEDS = (541, 547, 557)
TOP_FILES = ("launch.json", "runtime.json", "child.log", "comparison.json", "report.json")
BUNDLE_FILES = ("launch.json", "runtime.json", "checkpoint.pt", "trace.pt", "control.pt",
                "restore.json", "score.json", "manifest.json")
MANIFEST_PAYLOADS = tuple(name for name in BUNDLE_FILES if name != "manifest.json")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_BINARY = 512 * 1024 * 1024
MAX_METADATA = 16 * 1024 * 1024
MAX_TOTAL_METADATA = 64 * 1024 * 1024
CHUNK = 1024 * 1024
FALSE_FLAGS = ("checkpoint_admitted", "learned_stance_accepted", "football_balance_accepted",
               "physical_motion_authorized", "independent_gpu_attestation",
               "complete_binary_runtime_equivalence_verified")
SEED_DECISIONS = ("lean-replication-seed-passed", "lean-replication-seed-rejected")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise ValueError("non-finite JSON number")


def _finite_float(raw):
    value = float(raw)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def _parse(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=_bad_constant, parse_float=_finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    _require(type(value) is dict, f"{label} must be a JSON object")
    return value


def _sha(value, label):
    _require(type(value) is str and HEX64.fullmatch(value) is not None,
             f"invalid {label} SHA256")
    return value


def _exact_int(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def _root(path):
    _require(isinstance(path, (str, os.PathLike)), "root path required")
    supplied = os.fspath(path)
    _require(os.path.isabs(supplied), "root must be absolute")
    absolute = os.path.abspath(supplied)
    current = os.path.sep
    for component in absolute.split(os.path.sep):
        if not component:
            continue
        current = os.path.join(current, component)
        info = os.lstat(current)
        _require(not stat.S_ISLNK(info.st_mode), "symlink ancestor refused")
    info = os.stat(absolute, follow_symlinks=False)
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(),
             "root must be an existing user-owned directory")
    _require(os.path.realpath(absolute) == absolute, "root realpath mismatch")
    return absolute


def _read_hash(path, *, limit, metadata, label):
    """Stream a no-follow regular file and return (sha256, byte length)."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        _require(before.st_uid == os.getuid(), f"{label} must be user-owned")
        _require(0 < before.st_size <= limit, f"{label} size outside byte bound")
        digest = hashlib.sha256()
        count = 0
        while True:
            block = os.read(fd, min(CHUNK, limit + 1 - count))
            if not block:
                break
            count += len(block)
            _require(count <= limit, f"{label} exceeds byte bound")
            digest.update(block)
        after = os.fstat(fd)
        _require(count == before.st_size and
                 (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 f"{label} changed while hashing")
        return digest.hexdigest(), count
    finally:
        os.close(fd)


def _read_bounded(path, *, limit, label):
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(),
                 f"{label} must be a user-owned regular file")
        _require(0 < before.st_size <= limit, f"{label} size outside byte bound")
        chunks = []
        count = 0
        while True:
            block = os.read(fd, min(CHUNK, limit + 1 - count))
            if not block:
                break
            chunks.append(block)
            count += len(block)
            _require(count <= limit, f"{label} exceeds byte bound")
        after = os.fstat(fd)
        _require(count == before.st_size and
                 (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 f"{label} changed while reading")
        raw = b"".join(chunks)
        return raw, hashlib.sha256(raw).hexdigest(), count
    finally:
        os.close(fd)


def _read_json(root, key, *, budget):
    raw, digest, size = _read_bounded(os.path.join(root, key), limit=MAX_METADATA, label=key)
    budget[0] += size
    _require(budget[0] <= MAX_TOTAL_METADATA, "aggregate metadata exceeds byte bound")
    return raw, digest, size, _parse(raw, key)


def _expected_top_files(case_names):
    return set(TOP_FILES) | {name + ".json" for name in case_names}


def _expected_cases():
    result = []
    for training_seed in TRAINING_SEEDS:
        for iteration in ITERATIONS:
            for evaluation_seed in EVALUATION_SEEDS:
                result.append((f"packed-full-train-{training_seed}-cp-{iteration}-eval-{evaluation_seed}",
                               training_seed, iteration, evaluation_seed))
    return result


def _safe_inventory(root, case_names):
    expected_top = _expected_top_files(case_names) | set(case_names)
    actual_top = set(os.listdir(root))
    _require(actual_top == expected_top, "unexpected or missing top-level capture paths")
    for name in case_names:
        directory = os.path.join(root, name)
        info = os.lstat(directory)
        _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                 and info.st_uid == os.getuid(), "case directory must be user-owned real directory")
        _require(set(os.listdir(directory)) == set(BUNDLE_FILES),
                 "unexpected or missing case bundle paths")


def _binding(case, source, runtime_sha, checker_sha, case_tuple):
    name, training_seed, iteration, evaluation_seed = case_tuple
    _require(type(case) is dict and set(case) == {"name", "binding", "checkpoint"}
             and case["name"] == name, "ordered full matrix case declaration")
    binding = case["binding"]
    required = {"protocol", "source", "runtime_sha256", "checkpoint_sha256", "launch_sha256",
                "checkpoint_iteration", "evaluation_seed", "worlds", "capture_device",
                "solved_field_check", "checker_sha256", "cpu_math_profile", "training_seed"}
    _require(type(binding) is dict and set(binding) == required,
             "exact full trace binding fields")
    _require(binding["protocol"] == TRACE_PROTOCOL and binding["source"] == source
             and binding["runtime_sha256"] == runtime_sha and binding["checker_sha256"] == checker_sha,
             "source, runtime, trace protocol and checker binding")
    _sha(binding["checkpoint_sha256"], "checkpoint")
    _sha(binding["launch_sha256"], "case launch")
    _require(type(binding["training_seed"]) is int and binding["training_seed"] == training_seed
             and type(binding["checkpoint_iteration"]) is int and binding["checkpoint_iteration"] == iteration
             and type(binding["evaluation_seed"]) is int and binding["evaluation_seed"] == evaluation_seed,
             "exact ordered training/checkpoint/evaluation binding")
    _require(binding["worlds"] == 128 and type(binding["worlds"]) is int
             and binding["capture_device"] == "cuda:0" and binding["solved_field_check"] == "packed",
             "exact full evaluation runtime binding")
    _require(type(binding["cpu_math_profile"]) is dict, "recorded portable CPU profile required")
    checkpoint = case["checkpoint"]
    _require(type(checkpoint) is dict and set(checkpoint) == {"file", "sha256", "identity"}
             and type(checkpoint["file"]) is str and checkpoint["file"]
             and os.path.basename(checkpoint["file"]) == checkpoint["file"]
             and checkpoint["file"] not in (".", "..")
             and checkpoint["sha256"] == binding["checkpoint_sha256"],
             "checkpoint source hash and basename binding")
    identity = checkpoint.get("identity")
    _require(type(identity) is dict and type(identity.get("iteration")) is int
             and identity["iteration"] == iteration and type(identity.get("training_seed")) is int
             and identity["training_seed"] == training_seed,
             "checkpoint identity binding")
    _sha(identity.get("training_launch_sha256"), "training launch")
    _sha(identity.get("runtime_sha256"), "checkpoint runtime")
    return binding


def _validate(root, launch_sha_expected, report_sha_expected):
    launch_sha_expected = _sha(launch_sha_expected, "expected launch")
    report_sha_expected = _sha(report_sha_expected, "expected report")
    budget = [0]
    inventory_files = {}
    launch_raw, launch_sha, launch_size, launch = _read_json(root, "launch.json", budget=budget)
    inventory_files["launch.json"] = {"sha256": launch_sha, "bytes": launch_size}
    _require(launch_sha == launch_sha_expected, "independent launch SHA256 mismatch")
    report_raw, report_sha, report_size, report = _read_json(root, "report.json", budget=budget)
    inventory_files["report.json"] = {"sha256": report_sha, "bytes": report_size}
    _require(report_sha == report_sha_expected, "independent report SHA256 mismatch")
    _, runtime_sha, runtime_size, _runtime = _read_json(root, "runtime.json", budget=budget)
    inventory_files["runtime.json"] = {"sha256": runtime_sha, "bytes": runtime_size}
    _require(type(launch.get("protocol")) is str and launch["protocol"] == PROTOCOL,
             "full evaluation launch protocol")
    source = launch.get("source")
    _require(type(source) is str and HEX40.fullmatch(source) is not None, "40-hex source required")
    _require(launch.get("runtime_sha256") == runtime_sha, "launch/runtime byte binding")
    _sha(launch.get("checker_sha256"), "checker")
    cases = launch.get("cases")
    expected_cases = _expected_cases()
    _require(type(cases) is list and len(cases) == 36, "exact 36 full evaluation cases")
    case_names = [row[0] for row in expected_cases]
    _safe_inventory(root, case_names)

    comparison_raw, comparison_sha, comparison_size, comparison = _read_json(root, "comparison.json", budget=budget)
    inventory_files["comparison.json"] = {"sha256": comparison_sha, "bytes": comparison_size}
    _require(comparison.get("protocol") == PROTOCOL and
             comparison.get("launch_sha256") == launch_sha, "comparison launch hash chain")
    _require(type(comparison.get("cases")) is list and len(comparison["cases"]) == 36,
             "complete comparison case receipts")
    _require(type(report) is dict and report.get("protocol") == PROTOCOL
             and report.get("launch_sha256") == launch_sha
             and report.get("comparison_sha256") == comparison_sha,
             "completed report launch/comparison hash chain")
    decisions = ("lean-replication-passed", "lean-replication-seed-dependent",
                 "lean-replication-rejected")
    _require(report.get("decision") in decisions and "error" not in report
             and "error_type" not in report and report.get("optimizer_steps") == 0
             and type(report.get("optimizer_steps")) is int,
             "successful complete three-seed numerical decision required")
    _require(all(report.get(flag) is False for flag in FALSE_FLAGS),
             "report contains an explicit non-admission flag")
    summary = comparison.get("summary")
    _require(type(summary) is dict and report.get("summary") == summary
             and summary.get("decision") == report["decision"]
             and all(summary.get(flag) is False for flag in FALSE_FLAGS),
             "comparison and report summary/decision chain")
    per_seed = summary.get("per_seed")
    _require(type(per_seed) is dict and set(per_seed) == {str(seed) for seed in TRAINING_SEEDS}
             and all(type(per_seed[str(seed)]) is dict and "error" not in per_seed[str(seed)]
                     and "error_type" not in per_seed[str(seed)]
                     and per_seed[str(seed)].get("decision") in SEED_DECISIONS
                     for seed in TRAINING_SEEDS),
             "three completed replication seed verdicts without errors")

    for index, (case_tuple, case, receipt) in enumerate(zip(expected_cases, cases, comparison["cases"])):
        name = case_tuple[0]
        binding = _binding(case, source, runtime_sha, launch["checker_sha256"], case_tuple)
        receipt_raw, receipt_sha, receipt_size, receipt_file = _read_json(root, name + ".json", budget=budget)
        inventory_files[name + ".json"] = {"sha256": receipt_sha, "bytes": receipt_size}
        _require(receipt_file == receipt, "case receipt differs from comparison receipt")
        _require(type(receipt) is dict and receipt.get("case") == name,
                 "ordered case receipt name")
        _sha(receipt.get("manifest_sha256"), "manifest")
        collection = receipt.get("collection")
        _require(type(collection) is dict and set(collection) == {"stop_reason", "policy_ticks",
                 "actor_device", "physics_device", "seed_initialization_validated",
                 "independent_gpu_supervision_validated", "checkpoint_admitted"}
                 and collection.get("stop_reason") == "all-first-attempts-complete"
                 and collection.get("actor_device") == "cpu" and collection.get("physics_device") == "cuda:0"
                 and all(collection.get(key) is False for key in
                         ("seed_initialization_validated", "independent_gpu_supervision_validated",
                          "checkpoint_admitted")), "complete capture receipt and explicit false flags")
        _require(1 <= _exact_int(collection.get("policy_ticks"), "policy ticks", 1) <= 250,
                 "bounded complete capture policy ticks")

        case_dir = os.path.join(root, name)
        manifest_raw, manifest_sha, manifest_size, manifest = _read_json(case_dir, "manifest.json", budget=budget)
        inventory_files[name + "/manifest.json"] = {"sha256": manifest_sha, "bytes": manifest_size}
        _require(manifest_sha == receipt["manifest_sha256"], "case manifest hash differs from receipt")
        _require(set(manifest) == {"protocol", "binding", "files", "status", "checkpoint_admitted",
                                  "physical_motion_authorized"}
                 and manifest["protocol"] == "football-b1n-evaluation-bundle-v3"
                 and manifest["binding"] == binding
                 and manifest["status"] == "retained-diagnostic-only"
                 and manifest["checkpoint_admitted"] is False
                 and manifest["physical_motion_authorized"] is False,
                 "exact retained diagnostic bundle manifest")
        file_rows = manifest["files"]
        _require(type(file_rows) is dict and set(file_rows) == set(MANIFEST_PAYLOADS),
                 "exact seven-file manifest coverage")
        for filename in MANIFEST_PAYLOADS:
            limit = MAX_METADATA if filename.endswith(".json") else MAX_BINARY
            key = name + "/" + filename
            if filename.endswith(".json"):
                payload_raw, digest, size, payload_json = _read_json(case_dir, filename, budget=budget)
            else:
                payload_raw = None
                payload_json = None
                digest, size = _read_hash(os.path.join(case_dir, filename), limit=limit,
                                          metadata=False, label=key)
            inventory_files[key] = {"sha256": digest, "bytes": size}
            row = file_rows[filename]
            _require(type(row) is dict and set(row) == {"sha256", "bytes"}
                     and row["sha256"] == digest and _exact_int(row["bytes"], "manifest byte count") == size,
                     "manifest byte digest/length mismatch")
            if filename == "checkpoint.pt":
                _require(digest == binding["checkpoint_sha256"],
                         "checkpoint byte digest differs from trace binding")
            elif filename == "runtime.json":
                _require(digest == binding["runtime_sha256"],
                         "runtime byte digest differs from trace binding")
            elif filename == "launch.json":
                checkpoint_identity = case["checkpoint"]["identity"]
                expected_case_binding = {key: value for key, value in binding.items()
                                         if key != "launch_sha256"}
                _require(set(payload_json) == {"protocol", "binding", "checkpoint_identity",
                         "purpose", "physical_motion_authorized"}
                         and payload_json["protocol"] == "football-b1n-evaluation-inputs-v3"
                         and payload_json["binding"] == expected_case_binding
                         and payload_json["checkpoint_identity"] == checkpoint_identity
                         and payload_json["purpose"] == "diagnostic-evaluation-only"
                         and payload_json["physical_motion_authorized"] is False,
                         "exact case launch schema and checkpoint identity")
                _require(payload_raw == _canonical(payload_json) + b"\n",
                         "case launch bytes must use exact canonical encoding")
                _require(digest == binding["launch_sha256"],
                         "case launch byte digest differs from trace binding")

    files_map = report.get("files")
    expected_top_files = _expected_top_files(case_names)
    _require(type(files_map) is dict and set(files_map) == expected_top_files - {"report.json"},
             "report exact top-level file digest coverage")
    hashes = {}
    for name in sorted(expected_top_files - {"report.json"}):
        if name in inventory_files:
            digest, size = inventory_files[name]["sha256"], inventory_files[name]["bytes"]
        else:
            digest, size = _read_hash(os.path.join(root, name), limit=MAX_METADATA,
                                      metadata=True, label=name)
            inventory_files[name] = {"sha256": digest, "bytes": size}
            budget[0] += size
            _require(budget[0] <= MAX_TOTAL_METADATA, "aggregate metadata exceeds byte bound")
        hashes[name] = digest
        _require(files_map[name] == digest, "report top-level file digest mismatch")
    _require(report["summary"] == summary and report["decision"] == summary["decision"],
             "report decision differs from comparison summary")
    _require(len(inventory_files) == 329, "complete 329-file capture required")
    return source, launch_sha, report_sha, inventory_files


def create(root, expected_launch_sha256, expected_report_sha256):
    """Return canonical inventory bytes after validating every captured file byte."""
    root = _root(root)
    source, launch_sha, report_sha, files = _validate(
        root, expected_launch_sha256, expected_report_sha256)
    total = sum(row["bytes"] for row in files.values())
    inventory = {"schema": INVENTORY_SCHEMA, "protocol": PROTOCOL, "source": source,
        "launch_sha256": launch_sha, "report_sha256": report_sha,
        "files": dict(sorted(files.items())), "file_count": len(files), "total_bytes": total,
        "byte_inventory_only": True, "tensor_replay_performed": False,
        "numerical_acceptance": False, "motion_authorized": False}
    return _canonical(inventory) + b"\n"


def verify(root, inventoryraw, expected_inventorysha):
    """Re-inventory the capture and require an exact canonical, root-free payload."""
    _require(type(inventoryraw) is bytes and 0 < len(inventoryraw) <= MAX_METADATA,
             "bounded inventory bytes required")
    expected_inventorysha = _sha(expected_inventorysha, "expected inventory")
    _require(hashlib.sha256(inventoryraw).hexdigest() == expected_inventorysha,
             "inventory SHA256 mismatch")
    supplied = _parse(inventoryraw, "inventory")
    _require(_canonical(supplied) + b"\n" == inventoryraw,
             "inventory must use canonical JSON encoding")
    _require(supplied.get("byte_inventory_only") is True
             and supplied.get("tensor_replay_performed") is False
             and supplied.get("numerical_acceptance") is False
             and supplied.get("motion_authorized") is False,
             "inventory must preserve byte-only non-admission flags")
    expected = create(root, supplied.get("launch_sha256"), supplied.get("report_sha256"))
    _require(expected == inventoryraw, "inventory does not exactly match capture bytes")
    return expected


def _exclusive_output(path, raw, capture_root):
    _require(os.path.isabs(path) and path == os.path.abspath(path),
             "inventory output path must be absolute and normalized")
    parent = os.path.abspath(os.path.dirname(path))
    _require(os.path.realpath(parent) == parent, "output parent symlink refused")
    parent_info = os.stat(parent, follow_symlinks=False)
    _require(stat.S_ISDIR(parent_info.st_mode) and parent_info.st_uid == os.getuid(),
             "output parent must be an owned real directory")
    _require(os.path.commonpath((capture_root, os.path.abspath(path))) != capture_root,
             "inventory output must be outside capture root")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                 "inventory output must be an owned regular file")
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("--root", required=True)
    create_parser.add_argument("--launch-sha256", required=True)
    create_parser.add_argument("--report-sha256", required=True)
    create_parser.add_argument("--output", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--root", required=True)
    verify_parser.add_argument("--inventory", required=True)
    verify_parser.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        capture = _root(args.root)
        _require(os.path.isabs(args.output), "inventory output path must be absolute")
        raw = create(capture, args.launch_sha256, args.report_sha256)
        _exclusive_output(args.output, raw, capture)
        print(hashlib.sha256(raw).hexdigest())
    else:
        root = _root(args.root)
        inventory_path = os.path.abspath(args.inventory)
        raw, digest, size = _read_bounded(inventory_path, limit=MAX_METADATA, label="inventory")
        _require(digest == args.inventory_sha256, "inventory SHA256 mismatch")
        _require(len(raw) == size, "inventory changed while reading")
        verify(root, raw, args.inventory_sha256)
        print(digest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"microduck portable full inventory: {exc}", file=sys.stderr)
        raise SystemExit(2)
