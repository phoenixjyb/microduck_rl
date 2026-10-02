import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "microduck_portable_full_inventory.py"
SPEC = importlib.util.spec_from_file_location("portable_full_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _capture(root):
    root.mkdir()
    source = "a" * 40
    runtime = _json({"runtime": "synthetic"})
    (root / "runtime.json").write_bytes(runtime)
    cases = []
    receipts = []
    summary = {"decision": "lean-replication-rejected", "complete_attempts": 4608,
               "cases": 36,
               "per_seed": {str(seed): {"decision": "lean-replication-seed-rejected"}
                            for seed in inventory.TRAINING_SEEDS},
               **{key: False for key in inventory.FALSE_FLAGS}}
    payload_rows = {}
    for name, train, iteration, evaluation_seed in inventory._expected_cases():
        checkpoint_raw = b"checkpoint bytes"
        binding = {"protocol": inventory.TRACE_PROTOCOL, "source": source,
            "runtime_sha256": _sha(runtime), "checkpoint_sha256": _sha(checkpoint_raw),
            "checkpoint_iteration": iteration,
            "evaluation_seed": evaluation_seed, "worlds": 128, "capture_device": "cuda:0",
            "solved_field_check": "packed", "checker_sha256": "e" * 64,
            "cpu_math_profile": {"profile": "synthetic"}, "training_seed": train}
        identity = {"iteration": iteration, "training_seed": train,
                    "training_launch_sha256": "f" * 64, "runtime_sha256": "b" * 64}
        case_launch = _json({"protocol": "football-b1n-evaluation-inputs-v3",
            "binding": dict(binding), "checkpoint_identity": identity,
            "purpose": "diagnostic-evaluation-only", "physical_motion_authorized": False})
        binding["launch_sha256"] = _sha(case_launch)
        cases.append({"name": name, "binding": binding,
                      "checkpoint": {"file": f"checkpoint-{iteration}.pt", "sha256": _sha(checkpoint_raw),
                                     "identity": identity}})
        case_payload = {
            "launch.json": case_launch, "runtime.json": runtime,
            "checkpoint.pt": checkpoint_raw, "trace.pt": b"trace bytes",
            "control.pt": b"control bytes", "restore.json": _json({"restore": True}),
            "score.json": _json({"score": "synthetic"}),
        }
        manifest = {"protocol": "football-b1n-evaluation-bundle-v3", "binding": binding,
            "files": {key: {"sha256": _sha(data), "bytes": len(data)}
                      for key, data in case_payload.items()},
            "status": "retained-diagnostic-only", "checkpoint_admitted": False,
            "physical_motion_authorized": False}
        manifest_raw = _json(manifest)
        receipt = {"case": name, "manifest_sha256": _sha(manifest_raw),
            "collection": {"stop_reason": "all-first-attempts-complete", "policy_ticks": 1,
                "actor_device": "cpu", "physics_device": "cuda:0",
                "seed_initialization_validated": False,
                "independent_gpu_supervision_validated": False, "checkpoint_admitted": False},
            "timings": {"synthetic": True}}
        receipts.append(receipt)
        case_dir = root / name
        case_dir.mkdir()
        for key, data in case_payload.items():
            (case_dir / key).write_bytes(data)
        (case_dir / "manifest.json").write_bytes(manifest_raw)
        (root / (name + ".json")).write_bytes(_json(receipt))
        payload_rows[name] = case_payload

    launch = {"protocol": inventory.PROTOCOL, "source": source,
        "runtime_sha256": _sha(runtime), "checker_sha256": "e" * 64, "cases": cases}
    launch_raw = _json(launch)
    launch_sha = _sha(launch_raw)
    (root / "launch.json").write_bytes(launch_raw)
    (root / "child.log").write_bytes(b"synthetic successful child log\n")
    comparison = {"protocol": inventory.PROTOCOL, "launch_sha256": launch_sha,
                  "cases": receipts, "summary": summary}
    comparison_raw = _json(comparison)
    (root / "comparison.json").write_bytes(comparison_raw)
    top_names = inventory._expected_top_files([row[0] for row in inventory._expected_cases()]) - {"report.json"}
    report = {"protocol": inventory.PROTOCOL, "launch_sha256": launch_sha,
        "comparison_sha256": _sha(comparison_raw), "decision": summary["decision"],
        "optimizer_steps": 0, "summary": summary, **{key: False for key in inventory.FALSE_FLAGS},
        "files": {name: _sha((root / name).read_bytes()) for name in sorted(top_names)}}
    report_raw = _json(report)
    (root / "report.json").write_bytes(report_raw)
    return _sha(launch_raw), _sha(report_raw), cases, receipts, payload_rows


def _reseal(root, *, refresh_manifest_receipts=True):
    """Recompute outer evidence hashes after a semantic fixture mutation."""
    launch_raw = (root / "launch.json").read_bytes()
    launch_sha = _sha(launch_raw)
    launch = json.loads(launch_raw)
    receipts = []
    for case in launch["cases"]:
        name = case["name"]
        receipt_path = root / (name + ".json")
        receipt = json.loads(receipt_path.read_bytes())
        if refresh_manifest_receipts:
            receipt["manifest_sha256"] = _sha((root / name / "manifest.json").read_bytes())
        receipt_path.write_bytes(_json(receipt))
        receipts.append(receipt)
    comparison = json.loads((root / "comparison.json").read_bytes())
    comparison["launch_sha256"] = launch_sha
    comparison["cases"] = receipts
    comparison_raw = _json(comparison)
    (root / "comparison.json").write_bytes(comparison_raw)
    report = json.loads((root / "report.json").read_bytes())
    report["launch_sha256"] = launch_sha
    report["comparison_sha256"] = _sha(comparison_raw)
    top_names = inventory._expected_top_files([row[0] for row in inventory._expected_cases()]) - {"report.json"}
    report["files"] = {name: _sha((root / name).read_bytes()) for name in sorted(top_names)}
    report_raw = _json(report)
    (root / "report.json").write_bytes(report_raw)
    return launch_sha, _sha(report_raw)


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "capture"
    launch_sha, report_sha, cases, receipts, payload_rows = _capture(root)
    return root, launch_sha, report_sha, cases, receipts, payload_rows


def test_standalone_create_and_verify_inventory_329_streamed_files(completed):
    root, launch_sha, report_sha, *_ = completed
    raw = inventory.create(root, launch_sha, report_sha)
    parsed = json.loads(raw)
    assert parsed["file_count"] == 329
    assert len(parsed["files"]) == 329
    assert parsed["byte_inventory_only"] is True
    assert parsed["tensor_replay_performed"] is False
    assert parsed["numerical_acceptance"] is False
    assert parsed["motion_authorized"] is False
    assert str(root) not in raw.decode()
    assert inventory.verify(root, raw, _sha(raw)) == raw


def test_seed_verdict_names_match_unchanged_evaluator_declaration():
    source = SCRIPT.parents[1] / "src/mjlab_microduck/stance_lean_evaluation.py"
    tree = ast.parse(source.read_text())
    declaration = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "REPLICATION"
                               for target in node.targets))
    values = {item.arg: ast.literal_eval(item.value) for item in declaration.keywords
              if item.arg in {"passed", "rejected"}}
    assert inventory.SEED_DECISIONS == (values["passed"], values["rejected"])


@pytest.mark.parametrize("aggregate, per_seed", [
    ("lean-replication-passed", ["lean-replication-seed-passed"] * 3),
    ("lean-replication-seed-dependent", ["lean-replication-seed-passed",
       "lean-replication-seed-rejected", "lean-replication-seed-passed"]),
])
def test_completed_seed_verdict_schemas_are_retained_as_bytes(completed, aggregate, per_seed):
    root, *_ = completed
    comparison = json.loads((root / "comparison.json").read_bytes())
    comparison["summary"]["decision"] = aggregate
    comparison["summary"]["per_seed"] = {
        str(seed): {"decision": verdict}
        for seed, verdict in zip(inventory.TRAINING_SEEDS, per_seed)}
    (root / "comparison.json").write_bytes(_json(comparison))
    report = json.loads((root / "report.json").read_bytes())
    report["decision"] = aggregate
    report["summary"] = comparison["summary"]
    (root / "report.json").write_bytes(_json(report))
    launch_sha, report_sha = _reseal(root)
    raw = inventory.create(root, launch_sha, report_sha)
    assert json.loads(raw)["numerical_acceptance"] is False


def test_aggregate_verdict_is_not_a_per_seed_verdict(completed):
    root, *_ = completed
    comparison = json.loads((root / "comparison.json").read_bytes())
    comparison["summary"]["per_seed"]["577"]["decision"] = "lean-replication-passed"
    (root / "comparison.json").write_bytes(_json(comparison))
    report = json.loads((root / "report.json").read_bytes())
    report["summary"] = comparison["summary"]
    (root / "report.json").write_bytes(_json(report))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="three completed replication seed verdicts"):
        inventory.create(root, launch_sha, report_sha)


@pytest.mark.parametrize("target", ["runtime.json", "packed-full-train-577-cp-64-eval-541/trace.pt"])
def test_hash_and_size_truncation_are_rejected(completed, target):
    root, launch_sha, report_sha, *_ = completed
    with (root / target).open("ab") as stream:
        stream.write(b"truncated-or-appended")
    with pytest.raises(ValueError):
        inventory.create(root, launch_sha, report_sha)


def test_missing_extra_and_reordered_cases_fail_closed(completed):
    root, launch_sha, report_sha, cases, receipts, _ = completed
    (root / "child.log").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        inventory.create(root, launch_sha, report_sha)
    (root / "child.log").write_bytes(b"log")
    (root / "unexpected").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected or missing"):
        inventory.create(root, launch_sha, report_sha)
    (root / "unexpected").unlink()
    comparison = json.loads((root / "comparison.json").read_bytes())
    comparison["cases"].reverse()
    raw = _json(comparison)
    (root / "comparison.json").write_bytes(raw)
    report = json.loads((root / "report.json").read_bytes())
    report["comparison_sha256"] = _sha(raw)
    report["files"]["comparison.json"] = _sha(raw)
    (root / "report.json").write_bytes(_json(report))
    with pytest.raises(ValueError):
        inventory.create(root, launch_sha, _sha((root / "report.json").read_bytes()))


@pytest.mark.parametrize("target, raw", [
    ("launch.json", b'{"protocol":"x","protocol":"y"}'),
    ("launch.json", b'{"protocol":NaN}'),
    ("launch.json", b'{"protocol":"x","large":1e999}'),
    ("packed-full-train-577-cp-64-eval-541/manifest.json", b'{"files":{}}'),
])
def test_duplicate_keys_and_nonfinite_json_rejected(completed, target, raw):
    root, launch_sha, report_sha, *_ = completed
    (root / target).write_bytes(raw)
    with pytest.raises(ValueError):
        inventory.create(root, launch_sha, report_sha)


def test_symlink_fifo_and_bundle_binding_mismatch_rejected(completed):
    root, launch_sha, report_sha, *_ = completed
    target = root / "packed-full-train-577-cp-64-eval-541" / "trace.pt"
    original = target.read_bytes()
    target.unlink()
    target.symlink_to(root / "runtime.json")
    with pytest.raises((ValueError, OSError)):
        inventory.create(root, launch_sha, report_sha)
    target.unlink()
    target.write_bytes(original)
    manifest_path = target.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["binding"]["source"] = "9" * 40
    manifest_path.write_bytes(_json(manifest))
    new_launch_sha, new_report_sha = _reseal(root)
    with pytest.raises(ValueError):
        inventory.create(root, new_launch_sha, new_report_sha)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO not supported")
def test_fifo_is_rejected_without_blocking(completed):
    root, launch_sha, report_sha, *_ = completed
    runtime = root / "runtime.json"
    runtime.unlink()
    os.mkfifo(runtime)
    with pytest.raises((ValueError, OSError)):
        inventory.create(root, launch_sha, report_sha)


@pytest.mark.parametrize("field,value", [
    ("physical_motion_authorized", True), ("checkpoint_admitted", True),
    ("independent_gpu_attestation", True),
])
def test_non_admission_flag_mismatch_rejected(completed, field, value):
    root, launch_sha, report_sha, *_ = completed
    report = json.loads((root / "report.json").read_bytes())
    report[field] = value
    raw = _json(report)
    (root / "report.json").write_bytes(raw)
    with pytest.raises(ValueError):
        inventory.create(root, launch_sha, _sha(raw))


def test_source_and_case_binding_mismatch_rejected(completed):
    root, launch_sha, report_sha, *_ = completed
    launch = json.loads((root / "launch.json").read_bytes())
    launch["source"] = "9" * 40
    raw = _json(launch)
    (root / "launch.json").write_bytes(raw)
    new_launch_sha, new_report_sha = _reseal(root)
    with pytest.raises(ValueError):
        inventory.create(root, new_launch_sha, new_report_sha)


@pytest.mark.parametrize("filename, mutate", [
    ("checkpoint.pt", lambda old: old + b" altered"),
    ("runtime.json", lambda old: _json({"runtime": "altered"})),
])
def test_bundle_payload_hashes_must_match_trace_bindings(completed, filename, mutate):
    root, *_ = completed
    case_dir = root / "packed-full-train-577-cp-64-eval-541"
    target = case_dir / filename
    target.write_bytes(mutate(target.read_bytes()))
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    changed = target.read_bytes()
    manifest["files"][filename] = {"sha256": _sha(changed), "bytes": len(changed)}
    manifest_path.write_bytes(_json(manifest))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="differs from trace binding"):
        inventory.create(root, launch_sha, report_sha)


def test_case_launch_digest_must_match_trace_binding(completed):
    root, *_ = completed
    name = "packed-full-train-577-cp-64-eval-541"
    launch = json.loads((root / "launch.json").read_bytes())
    launch["cases"][0]["binding"]["launch_sha256"] = "0" * 64
    (root / "launch.json").write_bytes(_json(launch))
    manifest_path = root / name / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["binding"]["launch_sha256"] = "0" * 64
    manifest_path.write_bytes(_json(manifest))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="case launch byte digest differs from trace binding"):
        inventory.create(root, launch_sha, report_sha)


def test_case_launch_requires_exact_schema_and_bound_identity(completed):
    root, *_ = completed
    launch = json.loads((root / "launch.json").read_bytes())
    launch["cases"][0]["checkpoint"]["identity"]["training_seed"] = True
    (root / "launch.json").write_bytes(_json(launch))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="checkpoint identity"):
        inventory.create(root, launch_sha, report_sha)


@pytest.mark.parametrize("change", [
    {"purpose": "training"},
    {"physical_motion_authorized": True},
    {"protocol": "wrong-protocol"},
    {"unexpected": "field"},
    {"checkpoint_identity": {"iteration": 999}},
])
def test_case_launch_semantics_fail_after_its_byte_hash_is_resealed(completed, change):
    root, *_ = completed
    name = "packed-full-train-577-cp-64-eval-541"
    payload_path = root / name / "launch.json"
    payload = json.loads(payload_path.read_bytes())
    payload.update(change)
    raw = _json(payload)
    payload_path.write_bytes(raw)
    launch = json.loads((root / "launch.json").read_bytes())
    launch["cases"][0]["binding"]["launch_sha256"] = _sha(raw)
    (root / "launch.json").write_bytes(_json(launch))
    manifest_path = root / name / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["binding"]["launch_sha256"] = _sha(raw)
    manifest["files"]["launch.json"] = {"sha256": _sha(raw), "bytes": len(raw)}
    manifest_path.write_bytes(_json(manifest))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="exact case launch schema"):
        inventory.create(root, launch_sha, report_sha)


def test_helper_import_does_not_load_tensor_or_project_libraries():
    code = ("import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert not any(m.split('.')[0] in "
            "{'torch','numpy','mjlab','mjlab_microduck','pickle'} for m in sys.modules)")
    result = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_manifest_receipt_sha_binding_checked_after_outer_hashes_are_resealed(completed):
    root, *_ = completed
    name = "packed-full-train-577-cp-64-eval-541"
    receipt_path = root / (name + ".json")
    receipt = json.loads(receipt_path.read_bytes())
    receipt["manifest_sha256"] = "0" * 64
    receipt_path.write_bytes(_json(receipt))
    launch_sha, report_sha = _reseal(root, refresh_manifest_receipts=False)
    with pytest.raises(ValueError, match="manifest hash differs from receipt"):
        inventory.create(root, launch_sha, report_sha)


def test_checkpoint_filename_traversal_rejected_after_resealing(completed):
    root, *_ = completed
    launch = json.loads((root / "launch.json").read_bytes())
    launch["cases"][0]["checkpoint"]["file"] = "../../outside.pt"
    (root / "launch.json").write_bytes(_json(launch))
    launch_sha, report_sha = _reseal(root)
    with pytest.raises(ValueError, match="basename binding"):
        inventory.create(root, launch_sha, report_sha)


def test_verify_rejects_malformed_inventory_and_hash_mismatch(completed):
    root, launch_sha, report_sha, *_ = completed
    raw = inventory.create(root, launch_sha, report_sha)
    with pytest.raises(ValueError, match="SHA256"):
        inventory.verify(root, raw, "0" * 64)
    changed = json.loads(raw)
    changed["files"]["runtime.json"]["bytes"] += 1
    malformed = _json(changed)
    with pytest.raises(ValueError):
        inventory.verify(root, malformed, _sha(malformed))


def test_cli_create_output_is_exclusive_and_verify_is_read_only(completed, tmp_path):
    root, launch_sha, report_sha, *_ = completed
    output = tmp_path / "inventory.json"
    command = [sys.executable, str(SCRIPT), "create", "--root", str(root),
        "--launch-sha256", launch_sha, "--report-sha256", report_sha, "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    digest = _sha(raw)
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2
    verify_cmd = [sys.executable, str(SCRIPT), "verify", "--root", str(root),
        "--inventory", str(output), "--inventory-sha256", digest]
    checked = subprocess.run(verify_cmd, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert output.read_bytes() == raw
    inside = root / "bad-inventory.json"
    inside_cmd = command[:-1] + [str(inside)]
    refused = subprocess.run(inside_cmd, capture_output=True, text=True)
    assert refused.returncode == 2
    assert not inside.exists()


def test_cli_requires_absolute_output_and_verify_refuses_symlink_inventory(completed, tmp_path):
    root, launch_sha, report_sha, *_ = completed
    relative_cmd = [sys.executable, str(SCRIPT), "create", "--root", str(root),
        "--launch-sha256", launch_sha, "--report-sha256", report_sha,
        "--output", "relative-inventory.json"]
    relative = subprocess.run(relative_cmd, capture_output=True, text=True, cwd=tmp_path)
    assert relative.returncode == 2
    assert not (tmp_path / "relative-inventory.json").exists()

    output = tmp_path / "inventory.json"
    inventory_raw = inventory.create(root, launch_sha, report_sha)
    output.write_bytes(inventory_raw)
    alias = tmp_path / "inventory-link.json"
    alias.symlink_to(output)
    verify_cmd = [sys.executable, str(SCRIPT), "verify", "--root", str(root),
        "--inventory", str(alias), "--inventory-sha256", _sha(inventory_raw)]
    nofollow = subprocess.run(verify_cmd, capture_output=True, text=True)
    assert nofollow.returncode == 2


def test_capture_root_symlink_is_rejected(completed, tmp_path):
    root, launch_sha, report_sha, *_ = completed
    alias = tmp_path / "capture-link"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises((ValueError, OSError), match="symlink"):
        inventory.create(alias, launch_sha, report_sha)
