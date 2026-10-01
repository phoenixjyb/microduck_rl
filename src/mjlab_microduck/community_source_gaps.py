"""Reproducible source-gap inventory for the two pinned C1-S releases.

Static graph/card claims only, not an author-config validator or a new trial.
Reuse the existing interface and skill-descriptor vocabulary; never manufacture
a complete descriptor from matching shapes, a branch name or an upstream base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .community_hop_surrogate import load_declaration, verified_policy
from .community_policy_inspection import _read_payload
from .community_policy_normalizer import inspect_affine_prefix
from .skill_compatibility import STATIC_FIELDS


PROTOCOL = "community-c1-source-gap-inventory-v1"
CARD_HASHES = {
    "walk": {
        "README.md": "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74",
        "manifest.json": "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94",
    },
    "hop": {
        "README.md": "108c65ce78acfdeee87ca115e6bf3e638fbb18b4bcbf42ba058c89da2a128198",
        "manifest.json": "c3e351d46a0401d437916aa7520127505b50065cc0cd3225fc8408357f345089",
    },
}
NEEDED = {
    "mechanics": "Effective compiled plant, asset closure, collision overrides and BAM parameter/current bindings",
    "actor_interface": "Resolved column meanings, output-side views, units/scales, sensor lags/noise and export-normalizer parity",
    "action_interface": "Resolved servo order, exact HOME, units, scaling/clipping, filtering and queue initialization",
    "control_timing": "Effective physics/control cadence and action/sensor delays from the producing run",
    "command_sensor_envelope": "Training/evaluation command meanings, admissible sensor envelope and reset distribution",
    "runtime": "Exact training overlay, saved environment/agent config, export source and dependency/runtime receipts",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_card(payload: bytes) -> dict:
    require(type(payload) is bytes and 0 < len(payload) <= 128 * 1024,
            "bounded manifest bytes required")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate manifest key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite manifest constant: " + value)

    try:
        value = json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)
        json.dumps(value, allow_nan=False)  # Reject numeric overflow too.
    except (UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("invalid manifest JSON") from exc
    require(type(value) is dict, "manifest object required")
    return value


def published_claims(kind, manifest):
    """Keep published provenance distinct from exact producing-source identity."""
    require(kind in CARD_HASHES, "known pinned candidate required")
    require(all(type(manifest.get(key)) is int and manifest[key] == expected
                for key, expected in (("schema_version", 2), ("model_api", 1),
                                      ("obs_len", 61), ("action_len", 14))),
            "published API1 declaration required")
    if kind == "walk":
        policies = manifest.get("policies")
        require(type(policies) is list, "official policy list required")
        matches = [row for row in policies if type(row) is dict and row.get("file") == "velstand.onnx"]
        require(len(matches) == 1, "one official velstand entry required")
        training = matches[0].get("training")
        require(type(training) is dict, "published walker training claims required")
        return dict(training=training, entry_pose=matches[0].get("entry_pose"),
                    robot=manifest.get("robot"), exact_training_commit=None,
                    provenance="published-claims-not-producing-run-binding")
    require(manifest.get("name") == "happy-hop" and type(manifest.get("training")) is dict,
            "published Happy Hop claims required")
    return dict(training=manifest["training"], robot=manifest.get("robot"),
                command=manifest.get("command"), deployment=manifest.get("deployment"),
                entry_pose=manifest.get("entry_pose"), duration_s=manifest.get("duration_s"),
                status=manifest.get("status"), exact_training_commit=None,
                provenance="published-claims-not-producing-run-binding")


def build_gap_report(root: Path) -> dict:
    """Inspect the pinned bytes without sessions, compilation or rollouts.

    No author run receipts are present in this fixed intake. This tool has no
    option to 'resolve' those gaps: a later receipt needs a separate byte-bound
    descriptor/effective-runtime review, not a self-attested boolean here.
    """
    root = Path(root).resolve(strict=True)
    declaration, declaration_bytes = load_declaration(root)
    candidates = {}
    for kind in ("walk", "hop"):
        record = declaration["policies"][kind]
        payload, inspection = verified_policy(root, record, declaration["interface"])
        folder = root / Path(record["local_path"]).parent
        cards, manifest = {}, None
        for name, expected in CARD_HASHES[kind].items():
            raw = _read_payload(folder / name)
            require(0 < len(raw) <= 128 * 1024, "bounded supporting card required")
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == expected, "pinned supporting card SHA256 mismatch: " + kind + "/" + name)
            cards[name] = dict(path=(folder / name).relative_to(root).as_posix(),
                               sha256=digest, bytes=len(raw))
            if name == "manifest.json":
                manifest = parse_card(raw)
        prefix = inspect_affine_prefix(payload, record["sha256"])
        metadata = inspection["metadata"]
        candidates[kind] = dict(
            policy={"path": record["local_path"], **inspection["artifact"]},
            release=dict(repo=record["repo"], revision=record["revision"]),
            supporting_cards=cards,
            published=published_claims(kind, manifest),
            graph=dict(interface=inspection["interface_classification"],
                       inputs=inspection["runtime_inputs"], outputs=inspection["outputs"],
                       onnx=inspection["onnx"], metadata=metadata,
                       affine_prefix=dict(
                           decision=prefix["decision"], formula=prefix["formula"],
                           values_sha256=hashlib.sha256(json.dumps(
                               prefix["affine_values"], sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                           normalizer_parity_verified=False,
                           actor_forward_executed=False)),
            gaps=[dict(field=field, status="effective-author-receipt-missing", needed=NEEDED[field])
                  for field in STATIC_FIELDS],
            training_source_binding_verified=False,
            descriptor=None,
        )
    return dict(
        protocol=PROTOCOL, decision="author-source-handoff-required-no-policy-trial",
        declaration_sha256=hashlib.sha256(declaration_bytes).hexdigest(),
        candidates=candidates,
        handoff=[
            dict(field="entry_policy", status="exact-artifact-and-source-missing",
                 published_filename="walking_backlash_model_5000.onnx",
                 needed="Exact ONNX SHA256/bytes and producing run/checkpoint/export source; official velstand is not this identity"),
            dict(field="training_overlay", status="exact-overlay-and-run-config-missing",
                 published_upstream_base="d424a0c899f6b33cbd3daeb279913134349c0b63",
                 published_run="2026-08-31_20-39-21_happy_hop_clearance_35mm_stage1",
                 published_checkpoint_iteration=1255,
                 needed="Actual HappyHop task patch/commit and saved environment/agent config; upstream base is not the missing overlay"),
            dict(field="transition_state", status="author-handoff-implementation-missing",
                 needed="Previous raw action and delayed action queue carry/reset across walk-hop-walk, entry/exit conditions and filter state"),
        ],
        complete_skill_descriptor_available=False,
        descriptor_to_runtime_binding_verified=False,
        normalizer_parity_verified=False,
        effective_command_sensitivity_verified=False,
        author_matched_replication_verified=False,
        behavioral_retention_verified=False,
        policy_acceptance=False, training_authorized=False, transition_authorized=False,
        physical_motion_authorized=False,
        policy_inferences=0, simulation_steps=0, optimizer_updates=0,
        previous_c1s_attempt_reopened=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="repository containing the immutable declaration and pinned intake")
    args = parser.parse_args()
    try:
        result = build_gap_report(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
