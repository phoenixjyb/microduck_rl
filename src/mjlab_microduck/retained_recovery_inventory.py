"""CPU inventory of the closed seed-379 baseline failure; no rollout or retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from mjlab_microduck.skill_compatibility import _require, _verify_reference


MANIFEST = "docs/experiments/2026-09-08-recovery-retained-inventory.json"
MANIFEST_SHA = "6d49fa4a5af7ef18018c8fc8a25c5f175f7ca03203ad5660ca75b1588e3be43f"
SOURCE = "bd2a20a6232f647e656a2dfe6788b037e29a06b1"
EVAL = "artifacts/evaluations/recovery-cap-specialist-s379-v1"
REPORT = f"{EVAL}/00-cell0-a1/hierarchical-teacher-evaluation.json"
ACTOR = "logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt"
NEAR = "artifacts/checkpoints/hc4r2-bc-796634d-s42/supervisor.pt"
FAR = "artifacts/checkpoints/hc4lh-11002cc-center002/supervisor.pt"
HOST_ROOT = "/home/converge/work/microduck_rl-athletics-obstacle-curriculum/"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_receipts(launch, child, runtime, child_runtime, expected_models, expected_command):
    """Exactly the executed A1 prefix, not the unexecuted A2/B/far plan."""
    _require(launch["source"] == SOURCE and launch["protocol"] == "recovery-cap-specialist-s379-v1"
             and launch["model_sha256"] == expected_models, "frozen source/model declaration differs")
    _require(type(runtime["runs"]) is list and len(runtime["runs"]) == 1
             and canonical(runtime["runs"][0]) == canonical(child_runtime), "exact one-child receipt required")
    _require(child["index"] == child_runtime["index"] == 0
             and child["returncode"] is None and type(child_runtime["returncode"]) is int
             and child_runtime["returncode"] == 0, "completed baseline receipt required")
    _require(child["command"] == child_runtime["command"] == expected_command,
             "exact uncapped near A1 command required")


def audit(root: Path, repo: Path) -> dict:
    root, repo = Path(root).resolve(strict=True), Path(repo).resolve(strict=True)
    path = repo / MANIFEST
    _require(not path.is_symlink(), "manifest symlink refused")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "pinned recovery inventory required")
    manifest = json.loads(raw)
    _require(manifest["historical_protocol_source"] == SOURCE, "exact historical source required")
    references = {name: _verify_reference(root, name, ref) for name, ref in manifest["payloads"].items()}
    paths = list((root / EVAL).rglob("*"))
    _require(not any(p.is_symlink() for p in paths), "closed directory symlink refused")
    _require({p.relative_to(root).as_posix() for p in paths if p.is_file()}
             == {name for name in references if name.startswith(EVAL + "/")},
             "exact closed eight-file experiment inventory required")
    sources = {}
    for name, ref in manifest["protocol_sources"].items():
        checked = _verify_reference(repo, name, {key: ref[key] for key in ("path", "sha256", "bytes")})
        historical = subprocess.check_output(["git", "show", f"{SOURCE}:{name}"], cwd=repo)
        blob = hashlib.sha1(b"blob " + str(len(historical)).encode() + b"\0" + historical).hexdigest()
        _require(blob == ref["git_blob"]
                 and hashlib.sha256(historical).hexdigest() == checked["sha256"],
                 "historical selected source differs")
        sources[name] = dict(checked, git_blob=blob)

    from mjlab_microduck import recovery_ab
    _require(Path(recovery_ab.__file__).resolve() == repo / "src/mjlab_microduck/recovery_ab.py",
             "evaluator import outside verified repository")
    def read(name):
        return json.loads((root / EVAL / name).read_text())

    expected_models = {HOST_ROOT + name: references[name]["sha256"] for name in (ACTOR, NEAR, FAR)}
    # This helper constructs argv only. Neither its CLI nor any child launcher
    # is called. Pin the recorded interpreter, not this audit's local Python.
    command = recovery_ab.command_for(0, Path(HOST_ROOT + EVAL + "/00-cell0-a1"))
    command[0] = HOST_ROOT + ".venv/bin/python"
    verify_receipts(read("launch.json"), read("launch-00.json"), read("runtime.json"),
                    read("runtime-00.json"), expected_models, command)
    decision = recovery_ab.evaluate_paths([root / REPORT])
    _require(canonical(decision) == canonical(read("decision.json"))
             == canonical(read("decision-00.json")), "closed decision differs")
    _require(decision["decision"] == "numerical-gate-stop"
             and decision["failures"] == ["recovery-window"], "original prerequisite failure required")
    report = read("00-cell0-a1/hierarchical-teacher-evaluation.json")
    case = report["cases"][0]
    return dict(
        protocol="recovery-retained-inventory-v1",
        decision="historical-baseline-prerequisite-failure-reproduced",
        inventory_sha256=MANIFEST_SHA,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_verifier_sha256=hashlib.sha256(
            Path(__file__).with_name("skill_compatibility.py").read_bytes()).hexdigest(),
        historical_protocol_source=SOURCE, protocol_sources=sources,
        payloads=references, selected_payload_count=len(references),
        selected_payload_bytes=sum(ref["bytes"] for ref in references.values()),
        closed_directory_exact=True,
        declared_mechanics_group="rigid-locomotion-not-sprung-hop",
        model_roles={ACTOR: "executed-frozen-locomotion", NEAR: "executed-near-supervisor",
                     FAR: "declared-reference-only-not-executed"},
        historical_decision=decision,
        phase_speeds={phase: dict(mean_mps=case[f"{phase}_route_speed_mps"],
                                 samples=case[f"{phase}_samples"])
                      for phase in recovery_ab.PHASES},
        recovery_measurement=case["recovery_speed_measurement"],
        sensor_envelope=report["obstacle_sensor_model"],
        descriptor=None,
        full_runtime_binding_verified=False, effective_mechanics_binding_verified=False,
        raw_rollout_reconstruction_verified=False, behavioral_retention_verified=False,
        capped_treatment_executed=False, far_case_executed=False,
        stop_stance_restart_evaluated=False, stale_missing_sensor_evaluated=False,
        policy_acceptance=False, transition_authorized=False,
        training_authorized=False, physical_motion_authorized=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicit CPU-only environment required")
    result = audit(args.root, args.repo)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"recovery_inventory_decision={result['decision']}")


if __name__ == "__main__":
    main()
