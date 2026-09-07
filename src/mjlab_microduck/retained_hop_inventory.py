"""Reconcile immutable H1-T summaries on CPU; do not rerun hop rollouts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from mjlab_microduck.skill_compatibility import _require, _verify_reference


MANIFEST = "docs/experiments/2026-09-08-h1t-retained-inventory.json"
MANIFEST_SHA = "dc2d351c73e1b16c2b75d3ff2a76a04dea8e66d8748986ec0c14510b5b65a4cb"
SOURCE = "aa48f178cc9cb5791ee95e520cf6ef9fc14b2456"
EVAL = "artifacts/evaluations/h1t-k3900-117c881-s67-diagnostic"
BASELINE = "artifacts/evaluations/h1p-k3900-e7b15bb-s67-diagnostic/iteration-5999/hop-checkpoint-evaluation.json"
TRAIN = "logs/rsl_rl/hop_k3900_h1t/2026-09-05_15-47-06_h1t-k3900-117c881-s67-6000x256"
ITERATIONS = (500, 1000, 2000, 3000, 4000, 5999)
HOST_ROOT = "/home/converge/work/microduck_rl-athletics-obstacle-curriculum/"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalized_locations(value: dict) -> dict:
    """Replace only the two reader-local report paths, not checkpoint metadata."""
    result = deepcopy(value)
    for key, path in (("baseline", BASELINE),
                      ("candidate", f"{EVAL}/iteration-5999/hop-checkpoint-evaluation.json")):
        _require(type(result[key]["path"]) is str and bool(result[key]["path"]),
                 "explicit reader report location required")
        result[key]["path"] = path
    return result


def reconcile_causal(stored: dict, computed: dict) -> dict:
    # The pinned historical record must retain its original exact host paths.
    for key, path in (("baseline", BASELINE),
                      ("candidate", f"{EVAL}/iteration-5999/hop-checkpoint-evaluation.json")):
        _require(stored[key]["path"] == HOST_ROOT + path, "historical report path changed")
    stored, computed = normalized_locations(stored), normalized_locations(computed)
    _require(canonical(stored) == canonical(computed), "causal summary decision differs")
    _require(stored["decision"] == "stop", "closed H1-T rejection required")
    return stored


def audit(root: Path, repo: Path) -> dict:
    """Verify selected bytes and reproduce decisions from retained summaries.

Not a full training-directory inventory, checkpoint deserialization, raw trace
reconstruction, historical dynamics binding or cross-skill retention test.
"""
    root, repo = Path(root).resolve(strict=True), Path(repo).resolve(strict=True)
    manifest_path = repo / MANIFEST
    _require(not manifest_path.is_symlink(), "manifest symlink refused")
    raw = manifest_path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "pinned hop inventory required")
    manifest = json.loads(raw)
    _require(manifest["historical_protocol_source"] == SOURCE, "exact protocol source required")
    references = {name: _verify_reference(root, name, reference)
                  for name, reference in manifest["payloads"].items()}
    sources = {}
    for name, reference in manifest["protocol_sources"].items():
        checked = _verify_reference(repo, name, {k: reference[k] for k in ("path", "sha256", "bytes")})
        historical = subprocess.check_output(["git", "show", f"{SOURCE}:{name}"], cwd=repo)
        blob = hashlib.sha1(b"blob " + str(len(historical)).encode() + b"\0" + historical).hexdigest()
        _require(blob == reference["git_blob"]
                 and hashlib.sha256(historical).hexdigest() == checked["sha256"],
                 "historical evaluator source differs")
        sources[name] = dict(checked, git_blob=blob)

    # Import only after current source bytes match the historical protocol.
    from mjlab_microduck import hop_evaluation, hop_revision_gate
    for module in (hop_evaluation, hop_revision_gate):
        _require(Path(module.__file__).resolve().parent == repo / "src/mjlab_microduck",
                 "evaluator import outside verified repository")
    rows = []
    for iteration in ITERATIONS:
        path = f"{EVAL}/iteration-{iteration}/hop-checkpoint-evaluation.json"
        report = json.loads((root / path).read_text())
        canonical(report)  # refuse nonfinite numeric report fields
        _require(report["task"] == hop_revision_gate.H1T_TASK
                 and report["seeds"] == [211, 223, 227]
                 and report["num_envs"] == 128 and report["cycles"] == 6
                 and Path(report["checkpoint"]).name == f"model_{iteration}.pt",
                 "unchanged H1 held-out report identity required")
        _require([case["seed"] for case in report["cases"]] == [211, 223, 227]
                 and all(case["num_envs"] == 128 and case["cycles_per_env"] == 6
                         for case in report["cases"]), "complete ordered held-out cases required")
        decision = hop_evaluation.h1_decision(report["cases"])
        _require(canonical({key: report[key] for key in decision}) == canonical(decision),
                 "stored H1 summary gates differ")
        _require(decision["decision"] == "rejected", "closed H1 rejection required")
        if iteration == 5999:
            _require(report["checkpoint_sha256"] == references[f"{TRAIN}/model_5999.pt"]["sha256"],
                     "final checkpoint binding differs")
        rows.append(dict(iteration=iteration, reference=references[path], **decision))

    computed = hop_revision_gate.compare_h1t_to_h1p(
        root / BASELINE, root / EVAL / "iteration-5999/hop-checkpoint-evaluation.json")
    stored = json.loads((root / EVAL / "h1t-vs-h1p-causal-gate.json").read_text())
    comparison = reconcile_causal(stored, computed)
    return dict(
        protocol="h1t-retained-summary-inventory-v1",
        decision="historical-hop-rejection-reproduced",
        inventory_sha256=MANIFEST_SHA,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_verifier_sha256=hashlib.sha256(
            Path(__file__).with_name("skill_compatibility.py").read_bytes()).hexdigest(),
        historical_protocol_source=SOURCE, protocol_sources=sources,
        payloads=references, selected_payload_count=len(references),
        selected_payload_bytes=sum(item["bytes"] for item in references.values()),
        declared_mechanics_group="sprung-k3900-not-rigid-locomotion",
        evaluations=rows, causal_comparison=comparison,
        descriptor=None,
        full_training_inventory_verified=False,
        baseline_and_intermediate_checkpoint_bytes_verified=False,
        saved_config_historical_identity_verified=False,
        effective_mechanics_binding_verified=False,
        raw_rollout_reconstruction_verified=False,
        behavioral_retention_verified=False,
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
    print(f"hop_inventory_decision={result['decision']}")


if __name__ == "__main__":
    main()
