"""Offline consistency audit for retained C1-S evidence.

This module reads retained files only. It never creates a policy session, loads
MuJoCo, runs a campaign, or authenticates simulator observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

from . import community_hop_diagnostic as runner
from . import community_hop_rehearsal as c1
from .community_hop_baseline import score_walk_only_trace


MANIFEST_SCHEMA = "community-hop-c1s-artifact-manifest-v1"
DECLARATION_PATH = "docs/experiments/2026-10-01-community-hop-c1s-predeclaration.json"
DECLARATION_SHA256 = "236d7ca25e1a0508e6a913944f822a44244fb032362bd4ee3a38e5315c9eb700"
FALSE_FLAGS = (
    "policy_execution_by_audit", "simulation_execution_by_audit",
    "trace_authenticated", "author_replication", "training",
    "physical_motion", "behavioral_acceptance",
)
IDENTITY_KEYS = (
    "experiment_id", "measurement_protocol", "measurement_scorer_sha256",
    "predeclaration_sha256", "runner_source_commit", "runner_source_sha256",
)
CASE_IDS = ("walk-only-control", "walk-hop-transition")


class AuditError(ValueError):
    pass


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: bytes, label: str):
    def invalid(value):
        raise AuditError(f"{label}: non-standard JSON constant {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=invalid, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON: {exc}") from exc


def _regular_bytes(path: Path, label: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise AuditError(f"{label}: missing or unreadable") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AuditError(f"{label}: expected regular non-symlink file")
        if info.st_size > max_bytes:
            raise AuditError(f"{label}: exceeds audit size bound")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise AuditError(f"{label}: exceeds audit size bound")
        return payload
    except OSError as exc:
        raise AuditError(f"{label}: unreadable") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _hash_regular(path: Path, label: str, *, max_bytes: int) -> tuple[int, str]:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise AuditError(f"{label}: missing or unreadable") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise AuditError(f"{label}: expected regular non-symlink file")
        if info.st_size > max_bytes:
            raise AuditError(f"{label}: exceeds audit size bound")
        digest = hashlib.sha256()
        count = 0
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > max_bytes:
                    raise AuditError(f"{label}: exceeds audit size bound")
                digest.update(block)
        return count, digest.hexdigest()
    except OSError as exc:
        raise AuditError(f"{label}: unreadable") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _safe_relative(name: str) -> PurePosixPath:
    if type(name) is not str or not name or "\\" in name:
        raise AuditError("manifest contains unsafe relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise AuditError("manifest contains unsafe relative path")
    return path


def _verify_manifest(root: Path) -> tuple[dict, dict[str, bytes], dict[str, str], str]:
    if root.is_symlink() or not root.is_dir():
        raise AuditError("campaign path must be a real directory")
    manifest_path = root / "manifest.json"
    raw_manifest = _regular_bytes(manifest_path, "manifest.json", max_bytes=4 * 1024 * 1024)
    manifest = _json_bytes(raw_manifest, "manifest.json")
    if type(manifest) is not dict or set(manifest) != {"schema", "artifacts"} \
            or manifest.get("schema") != MANIFEST_SCHEMA or type(manifest.get("artifacts")) is not dict:
        raise AuditError("manifest schema mismatch")

    declared: dict[str, tuple[int, str]] = {}
    for name, metadata in manifest["artifacts"].items():
        rel = _safe_relative(name)
        if type(metadata) is not dict or set(metadata) != {"bytes", "sha256"}:
            raise AuditError(f"manifest entry schema mismatch: {name}")
        size, digest = metadata["bytes"], metadata["sha256"]
        if type(size) is not int or size < 0 or type(digest) is not str \
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AuditError(f"manifest entry values invalid: {name}")
        declared[rel.as_posix()] = (size, digest)

    actual: dict[str, Path] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        here = Path(current)
        for directory in list(dirs):
            item = here / directory
            if item.is_symlink() or not item.is_dir():
                raise AuditError(f"symlink or non-directory path in campaign: {item.relative_to(root)}")
        for filename in files:
            item = here / filename
            rel = item.relative_to(root).as_posix()
            if rel == "manifest.json":
                continue
            if item.is_symlink() or not item.is_file():
                raise AuditError(f"non-regular artifact in campaign: {rel}")
            actual[rel] = item
    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        raise AuditError(f"manifest file set mismatch; missing={missing}; extra={extra}")

    if sum(size for size, _ in declared.values()) > 512 * 1024 * 1024:
        raise AuditError("campaign exceeds total audit size bound")
    contents, hashes = {}, {}
    for name in sorted(actual):
        size, digest = declared[name]
        bound = 128 * 1024 * 1024 if name.endswith(".mjb") else 64 * 1024 * 1024
        actual_size, actual_digest = _hash_regular(actual[name], name, max_bytes=bound)
        if actual_size != size or actual_digest != digest:
            raise AuditError(f"manifest bytes/hash mismatch: {name}")
        hashes[name] = actual_digest
        if not name.endswith(".mjb"):
            payload = _regular_bytes(actual[name], name, max_bytes=bound)
            if _sha(payload) != digest:
                raise AuditError(f"artifact changed during audit: {name}")
            contents[name] = payload
    return manifest, contents, hashes, _sha(raw_manifest)


def _record(contents: dict[str, bytes], name: str) -> dict:
    if name not in contents:
        raise AuditError(f"required artifact missing: {name}")
    value = _json_bytes(contents[name], name)
    if type(value) is not dict:
        raise AuditError(f"{name}: expected JSON object")
    return value


def _check_declaration(contents: dict[str, bytes], name: str) -> dict:
    if name not in contents:
        raise AuditError(f"required artifact missing: {name}")
    payload = contents[name]
    if _sha(payload) != DECLARATION_SHA256:
        raise AuditError(f"copied declaration hash mismatch: {name}")
    return _json_bytes(payload, name)


def _close(a, b) -> bool:
    """JSON-roundtrip equality, rejecting NaN/Infinity instead of equating them."""
    try:
        return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
            b, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _finite_vector(value, size: int, label: str) -> bool:
    return (type(value) is list and len(value) == size
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _journal_events(payload: bytes) -> list[dict]:
    events = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n"):
            raise AuditError(f"journal has partial line at {line_number}")
        record = _json_bytes(raw, f"journal line {line_number}")
        if type(record) is not dict or set(record) != {"kind", "value"} \
                or type(record["kind"]) is not str or type(record["value"]) is not dict:
            raise AuditError(f"journal record schema invalid at line {line_number}")
        events.append(record)
    return events


def _replay_journal(case_id: str, payload: bytes, trace: dict) -> dict:
    events = _journal_events(payload)
    samples = trace.get("samples")
    if type(samples) is not list or len(samples) != c1.TOTAL_SAMPLES:
        raise AuditError("incomplete trace cannot pass event replay")
    if len(events) != c1.TOTAL_SAMPLES + 2 * 600:
        raise AuditError("journal event count does not match a complete fixed attempt")
    raw_history = [0.0] * 14
    applied_history = [0.0] * 14
    cursor = 0
    control_count = 0
    for index, sample in enumerate(samples):
        t_s = index * c1.PHYSICS_DT_S
        terminal = index == c1.TOTAL_SAMPLES - 1
        if index % 4 == 0 and not terminal:
            intent, control = events[cursor:cursor + 2]
            cursor += 2
            kind = runner.policy_kind(case_id, t_s)
            for event, expected_kind in ((intent, "inference-intent"), (control, "control")):
                if event["kind"] != expected_kind:
                    raise AuditError(f"journal event order mismatch at sample {index}")
                value = event["value"]
                expected_fields = ({"t_s", "policy", "obs"} if expected_kind == "inference-intent"
                                   else {"t_s", "policy", "obs", "action_raw", "action_applied"})
                if set(value) != expected_fields:
                    raise AuditError(f"journal event fields mismatch at sample {index}")
                if type(value.get("t_s")) not in (int, float) or not math.isfinite(value["t_s"]) \
                        or value.get("t_s") != t_s or value.get("policy") != kind:
                    raise AuditError(f"journal time or policy mismatch at sample {index}")
            iv, cv = intent["value"], control["value"]
            obs = iv.get("obs")
            if not _finite_vector(obs, 61, "observation") or not _close(cv.get("obs"), obs):
                raise AuditError(f"journal observation invalid or changed at sample {index}")
            if not _close(obs[48:61], c1.command_at(t_s)):
                raise AuditError(f"observation command mismatch at sample {index}")
            if not _close(obs[34:48], raw_history):
                raise AuditError(f"raw action history mismatch at sample {index}")
            if index == 0 and not _close(obs[20:34], [0.0] * 14):
                raise AuditError("initial joint velocity lag is not zero")
            raw = cv.get("action_raw")
            applied = cv.get("action_applied")
            if not _finite_vector(raw, 14, "raw action") or not _finite_vector(applied, 14, "applied action"):
                raise AuditError(f"journal action invalid at sample {index}")
            if not _close(applied, raw_history):
                raise AuditError(f"action delay mismatch at sample {index}")
            raw_history, applied_history = list(raw), list(raw_history)
            control_count += 1

        event = events[cursor]
        cursor += 1
        if event["kind"] == "sample" and set(event["value"]) != c1.SAMPLE_FIELDS:
            raise AuditError(f"journal sample fields mismatch at sample {index}")
        if event["kind"] != "sample" or not _close(event["value"], sample):
            raise AuditError(f"journal sample differs from trace at sample {index}")
        if type(sample) is not dict or type(sample.get("t_s")) not in (int, float) \
                or not math.isfinite(sample["t_s"]) or sample.get("t_s") != t_s:
            raise AuditError(f"trace timestamp mismatch at sample {index}")
        if not _close(sample.get("command"), c1.command_at(t_s)):
            raise AuditError(f"trace command mismatch at sample {index}")
        if not _close(sample.get("action_raw"), raw_history):
            raise AuditError(f"trace raw action hold mismatch at sample {index}")
        if not _close(sample.get("action_applied"), applied_history):
            raise AuditError(f"trace applied action hold mismatch at sample {index}")
    if cursor != len(events) or control_count != 600:
        raise AuditError("journal has extra events or wrong control count")
    return {"complete": True, "sample_count": len(samples), "control_count": control_count,
            "physics_sample_period_s": c1.PHYSICS_DT_S, "control_period_s": c1.CONTROL_DT_S}


def _phase_descriptives(trace: dict) -> dict:
    samples = trace["samples"]
    phases = (("walk", 0.0, 3.0), ("entry_settle", 3.0, 4.0), ("hop", 4.0, 7.0),
              ("landing_settle", 7.0, 9.0), ("resume_walk", 9.0, 12.0000001))
    result = {}
    for name, start, end in phases:
        rows = [row for row in samples if start <= row["t_s"] < end]
        vx = [row["base_velocity_world_m_s"][0] for row in rows]
        displacement = None
        if rows:
            displacement = [rows[-1]["base_xy_m"][axis] - rows[0]["base_xy_m"][axis]
                            for axis in range(2)]
        controls = [row["action_raw"] for row in rows if abs(row["t_s"] / c1.CONTROL_DT_S
                                                               - round(row["t_s"] / c1.CONTROL_DT_S)) < 1e-8
                    and row["t_s"] < c1.TOTAL_TIME_S]
        amplitudes = [abs(value) for action in controls for value in action]
        result[name] = {
            "sample_count": len(rows),
            "forward_velocity_m_s": ({"mean": sum(vx) / len(vx), "min": min(vx), "max": max(vx)}
                                      if vx else None),
            "base_displacement_xy_m": displacement,
            "double_foot_contact_fraction": (sum(all(row["foot_contact"]) for row in rows) / len(rows)
                                             if rows else None),
            "raw_action_abs_amplitude": ({"mean": sum(amplitudes) / len(amplitudes),
                                           "max": max(amplitudes)} if amplitudes else None),
        }
    return result


def _audit_case(case_id: str, prefix: str, contents: dict[str, bytes], hashes: dict[str, str],
                setup: dict, declaration: dict) -> tuple[dict, dict]:
    case_pre = _record(contents, f"{prefix}/preflight.json")
    case_declaration = _check_declaration(contents, f"{prefix}/predeclaration.json")
    if not _close(case_declaration, declaration):
        raise AuditError(f"{case_id}: copied declaration differs from setup declaration")
    for key in runner.BINDING_KEYS:
        if key not in setup or case_pre.get(key) != setup[key]:
            raise AuditError(f"{case_id}: setup/case binding mismatch: {key}")
    if any(case_pre.get(key) is not False or setup.get(key) is not False
           for key in declaration["not_claimed"]):
        raise AuditError(f"{case_id}: setup/case contains an unsupported positive claim")
    compiled_hash = hashes.get(f"{prefix}/compiled-plant.mjb")
    plant = setup.get("plant", {})
    if compiled_hash is None or type(plant) is not dict or plant.get("compiled_mjb_sha256") != compiled_hash:
        raise AuditError(f"{case_id}: compiled plant hash mismatch")

    report = _record(contents, f"{prefix}/report.json")
    for key in IDENTITY_KEYS:
        if report.get(key) != setup.get(key):
            raise AuditError(f"{case_id}: report identity mismatch: {key}")
    if report.get("case_id") != case_id or report.get("setup_runner_source_commit") != setup.get("runner_source_commit"):
        raise AuditError(f"{case_id}: report case/setup identity mismatch")

    trace_obj = _json_bytes(contents.get(f"{prefix}/trace.json", b""), f"{case_id} trace")
    try:
        trace = runner.decode_numbers(trace_obj)
    except ValueError as exc:
        raise AuditError(f"{case_id}: invalid nonfinite trace tag") from exc
    if type(trace) is not dict or trace.get("schema") != c1.TRACE_SCHEMA:
        raise AuditError(f"{case_id}: trace schema mismatch")
    replay = _replay_journal(case_id, contents.get(f"{prefix}/journal.jsonl", b""), trace)
    scored = score_walk_only_trace(trace) if case_id == "walk-only-control" else c1.score_trace(trace)

    report_gates = report.get("gates")
    if type(report_gates) is not dict:
        raise AuditError(f"{case_id}: report gates missing")
    measured_gates = {key: value for key, value in report_gates.items()
                      if key != "runtime_completed_without_failure"}
    if not _close(measured_gates, scored.get("gates")):
        raise AuditError(f"{case_id}: scorer gate drift")
    for key in ("metrics", "scored_samples", "ignored_samples", "first_failure"):
        if not _close(report.get(key), scored.get(key)):
            raise AuditError(f"{case_id}: scorer {key} drift")
    runtime = report.get("runtime", {})
    worker = _record(contents, f"{prefix}/worker-result.json")
    child = _record(contents, f"{prefix}/child-exit.json")
    recovery = report.get("recovery")
    if type(recovery) is not dict or type(report.get("child")) is not dict:
        raise AuditError(f"{case_id}: runtime receipt schema invalid")
    runtime_ok = (type(runtime) is dict and runtime == worker
                  and report.get("child") == child
                  and type(child.get("returncode")) is int and child["returncode"] == 0
                  and child.get("timed_out") is False
                  and worker.get("provider") == "CPUExecutionProvider"
                  and type(worker.get("intra_threads")) is int and worker["intra_threads"] == 1
                  and type(worker.get("inter_threads")) is int and worker["inter_threads"] == 1
                  and worker.get("first_runtime_error") is None
                  and worker.get("first_runtime_failure") is None
                  and runtime.get("policy_inferences") == 600 and runtime.get("physics_steps") == 2400
                  and report_gates.get("runtime_completed_without_failure") is True
                  and report.get("binding_error") is None and report.get("validation_error") is None
                  and recovery.get("journal_error") is None
                  and recovery.get("partial_lines_dropped") == 0
                  and recovery.get("complete_records_recovered") == c1.TOTAL_SAMPLES)
    if not runtime_ok:
        raise AuditError(f"{case_id}: incomplete or inconsistent runtime evidence")
    if report.get("decision") != scored.get("decision"):
        raise AuditError(f"{case_id}: report decision differs from frozen scorer")
    report_false_flags = (*declaration["not_claimed"], "trace_authenticated",
                          "plant_runtime_binding_verified", "training_authorized",
                          "transition_authorized", "impact_limit_calibrated",
                          "thermal_model_verified", "multi_seed_promotion")
    if any(flag in report and report[flag] is not False for flag in report_false_flags):
        raise AuditError(f"{case_id}: report contains an unsupported positive claim")
    return report, {"case_id": case_id, "replay": replay, "decision": report["decision"],
                    "scored_decision": scored["decision"], "metrics_match": True,
                    "runtime_complete": runtime_ok, "phase_descriptives": _phase_descriptives(trace)}


def audit_campaign(campaign_path: Path) -> dict:
    result = {"audit_passed": False, "status": "rejected", "errors": [],
              **{flag: False for flag in FALSE_FLAGS}}
    try:
        root = campaign_path.absolute()
        manifest, contents, hashes, manifest_sha = _verify_manifest(root)
        declaration = _check_declaration(contents, "preflight/predeclaration.json")
        local_declaration_path = Path(__file__).resolve().parents[2] / DECLARATION_PATH
        local_payload = _regular_bytes(local_declaration_path, DECLARATION_PATH)
        if _sha(local_payload) != DECLARATION_SHA256:
            raise AuditError("locally pinned declaration hash mismatch")
        local_declaration = _json_bytes(local_payload, DECLARATION_PATH)
        if not _close(declaration, local_declaration):
            raise AuditError("retained declaration differs from local pinned declaration")
        setup = _record(contents, "preflight/preflight.json")
        expected = {
            "experiment_id": declaration["experiment_id"],
            "measurement_protocol": c1.PROTOCOL,
            "measurement_scorer_sha256": declaration["measurement"]["scorer_sha256"],
            "predeclaration_sha256": DECLARATION_SHA256,
        }
        for key, value in expected.items():
            if setup.get(key) != value:
                raise AuditError(f"campaign setup identity mismatch: {key}")
        if any(setup.get(key) is not False for key in declaration["not_claimed"]):
            raise AuditError("campaign setup contains an unsupported positive claim")
        scorer_path = Path(__file__).resolve().parents[2] / declaration["measurement"]["scorer_path"]
        if _sha(_regular_bytes(scorer_path, declaration["measurement"]["scorer_path"])) \
                != declaration["measurement"]["scorer_sha256"]:
            raise AuditError("locally available frozen scorer bytes do not match pinned SHA256")
        commit = setup.get("runner_source_commit")
        sources = setup.get("runner_source_sha256")
        if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None \
                or type(sources) is not dict or not sources \
                or any(type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in sources.values()):
            raise AuditError("historical runner identity/hash fields invalid")
        policies = setup.get("policies")
        if type(policies) is not dict or set(policies) != set(declaration["policies"]):
            raise AuditError("campaign policy identity set mismatch")
        for name, pinned in declaration["policies"].items():
            artifact = policies[name].get("artifact") if type(policies[name]) is dict else None
            if type(artifact) is not dict or artifact.get("sha256") != pinned["sha256"] \
                    or artifact.get("bytes") != pinned["bytes"]:
                raise AuditError(f"campaign policy identity/hash mismatch: {name}")
        asset_closure = setup.get("asset_closure")
        if type(asset_closure) is not dict or asset_closure.get("asset_tree_sha1") != declaration["plant"]["asset_tree_sha1"]:
            raise AuditError("campaign stock asset-tree identity mismatch")
        plant = setup.get("plant")
        if type(plant) is not dict or not re.fullmatch(r"[0-9a-f]{64}", plant.get("compiled_mjb_sha256", "")):
            raise AuditError("campaign compiled-plant identity/hash invalid")
        setup_mjb_hash = hashes.get("preflight/compiled-plant.mjb")
        if setup_mjb_hash is None or setup_mjb_hash != plant["compiled_mjb_sha256"]:
            raise AuditError("setup compiled-plant hash mismatch")
        bam_settings = setup.get("bam_settings")
        expected_bam = {key: declaration["plant"][key] for key in (
            "vin_v", "kp_fw", "max_current_a", "voltage_drop_gain",
            "firmware_position_view", "back_emf_and_friction_velocity_view")}
        if not _close(bam_settings, expected_bam):
            raise AuditError("campaign BAM settings differ from pinned declaration")
        if type(setup.get("bam_parameters")) is not dict:
            raise AuditError("campaign BAM parameter receipt missing")

        reports, summaries = {}, {}
        for case_id in CASE_IDS:
            prefix = case_id
            if f"{prefix}/report.json" in contents:
                reports[case_id], summaries[case_id] = _audit_case(
                    case_id, prefix, contents, hashes, setup, declaration)
            elif any(name.startswith(prefix + "/") for name in contents):
                raise AuditError(f"{case_id}: case artifacts exist without a report")
        decision = _record(contents, "decision.json")
        for key, value in runner.identity(setup, "campaign").items():
            if decision.get(key) != value:
                raise AuditError(f"campaign decision identity mismatch: {key}")
        expected_decision = runner.sequential_decision(
            reports.get("walk-only-control"), reports.get("walk-hop-transition"))
        if decision.get("decision") != expected_decision:
            raise AuditError("campaign sequential decision drift")
        expected_case_decisions = {key: value["decision"] for key, value in reports.items()}
        if decision.get("case_decisions") != expected_case_decisions:
            raise AuditError("campaign case decision map drift")
        if "walk-hop-transition" in reports and reports.get("walk-only-control", {}).get("decision") \
                != runner.BASELINE_DECISION:
            raise AuditError("transfer evidence exists although walker-only control did not pass")
        for flag in ("author_entry_replication", "author_training_plant_binding", "training_authorized_by_predeclaration",
                     "physical_motion_authorized", "behavioral_acceptance"):
            if decision.get(flag) is not False:
                raise AuditError(f"campaign has unsupported positive claim: {flag}")
        if any(decision.get(key) is not False for key in declaration["not_claimed"]):
            raise AuditError("campaign decision contains an unsupported positive claim")
        result.update({"audit_passed": True, "status": "evidence-consistent",
                       "manifest_sha256": manifest_sha, "manifest_artifact_count": len(manifest["artifacts"]),
                       "experiment_id": declaration["experiment_id"], "runner_source_commit": commit,
                       "decision": expected_decision, "cases": summaries,
                       "interpretation": "Offline consistency only; no inference, simulator authenticity, or normalizer parity is established."})
    except (AuditError, AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError, OSError) as exc:
        result["errors"] = [str(exc)]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="retained C1-S campaign folder")
    parser.add_argument("--output", type=Path, help="write JSON exclusively outside the campaign")
    args = parser.parse_args(argv)
    result = audit_campaign(args.campaign)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        root = args.campaign.resolve()
        target = args.output.resolve()
        if target == root or root in target.parents:
            parser.error("--output must be outside the input campaign")
        runner.save_json(args.output, result)
    else:
        sys.stdout.write(payload)
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
