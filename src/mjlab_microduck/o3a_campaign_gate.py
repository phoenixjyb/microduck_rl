"""Aggregate frozen O3a per-seed decisions without hiding local failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mjlab_microduck.o3a_gate import _reject_nonfinite

SOURCE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_PROTOCOLS = {
    "O3a-HC4LH-seed-271-range-noise-prescreen-v1": (
        "hc4lh",
        271,
        "continue_hc4r2_predeclaration",
    ),
    "O3a-HC4LH-seed-281-range-noise-prescreen-v1": (
        "hc4lh",
        281,
        "continue_campaign",
    ),
    "O3a-HC4LH-seed-283-range-noise-prescreen-v1": (
        "hc4lh",
        283,
        "continue_campaign",
    ),
    "O3a-HC4R2-seed-277-range-noise-prescreen-v1": (
        "hc4r2",
        277,
        "continue_multi_seed_predeclaration",
    ),
    "O3a-HC4R2-seed-281-range-noise-prescreen-v1": (
        "hc4r2",
        281,
        "continue_campaign",
    ),
    "O3a-HC4R2-seed-283-range-noise-prescreen-v1": (
        "hc4r2",
        283,
        "continue_campaign",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact(entry: dict[str, Any], name: str) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"{name} artifact is missing")
    path = Path(entry.get("path", "")).expanduser().resolve(strict=True)
    observed_sha256 = _sha256(path)
    if entry.get("sha256") != observed_sha256:
        raise ValueError(f"{name} report hash mismatch: {path}")
    return {"path": str(path), "sha256": observed_sha256}


def _pooled_phase_speed(report_paths: list[str], phase: str) -> float:
    value_name = f"{phase}_route_speed_mps"
    sample_name = f"{phase}_samples"
    weighted_sum = 0.0
    sample_count = 0
    for raw_path in report_paths:
        report = json.loads(Path(raw_path).read_text())
        for case in report["cases"]:
            samples = int(case[sample_name])
            weighted_sum += float(case[value_name]) * samples
            sample_count += samples
    if sample_count <= 0:
        raise ValueError(f"{phase} has no campaign samples")
    return weighted_sum / sample_count


def compare_o3a_campaign(
    decision_paths: tuple[Path, ...], source_commit: str
) -> dict[str, Any]:
    """Require all six predeclared specialist/seed decisions to pass."""
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    retained: dict[str, dict[str, Any]] = {}
    for raw_path in decision_paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text())
        _reject_nonfinite(payload)
        protocol = payload.get("protocol")
        if protocol not in EXPECTED_PROTOCOLS:
            raise ValueError(f"unexpected campaign protocol in {path}")
        if protocol in retained:
            raise ValueError(f"duplicate campaign protocol in {path}")
        specialist, seed, expected_decision = EXPECTED_PROTOCOLS[protocol]
        if payload.get("decision") != expected_decision:
            raise ValueError(f"local O3a gate did not pass in {path}")
        if payload.get("physical_motion_authorized") is not False:
            raise ValueError(f"decision expands physical authority: {path}")
        checks = payload.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"decision checks are missing in {path}")
        if any(check.get("status") != "pass" for check in checks):
            raise ValueError(f"decision contains a failed check in {path}")
        retained[protocol] = {
            "specialist": specialist,
            "physics_seed": seed,
            "decision_path": str(path),
            "decision_sha256": _sha256(path),
            "baseline": _verified_artifact(payload.get("baseline"), "baseline"),
            "noisy": _verified_artifact(payload.get("noisy"), "noisy"),
            "baseline_totals": payload["baseline_totals"],
            "noisy_totals": payload["noisy_totals"],
            "pooled_clean_rate_delta": payload["pooled_clean_rate_delta"],
            "pooled_phase_speed_deltas_mps": payload[
                "pooled_phase_speed_deltas_mps"
            ],
            "noisy_max_torque_utilization_p99": payload[
                "noisy_max_torque_utilization_p99"
            ],
        }

    if set(retained) != set(EXPECTED_PROTOCOLS):
        missing = sorted(set(EXPECTED_PROTOCOLS) - set(retained))
        raise ValueError(f"campaign decisions are incomplete: {missing}")

    ordered = [retained[protocol] for protocol in EXPECTED_PROTOCOLS]
    specialist_summaries = {}
    for specialist in ("hc4lh", "hc4r2"):
        entries = [entry for entry in ordered if entry["specialist"] == specialist]
        pooled_phase_speed_deltas = {
            phase: _pooled_phase_speed(
                [entry["noisy"]["path"] for entry in entries], phase
            )
            - _pooled_phase_speed(
                [entry["baseline"]["path"] for entry in entries], phase
            )
            for phase in ("approach", "recovery")
        }
        specialist_summaries[specialist] = {
            "physics_seeds": [entry["physics_seed"] for entry in entries],
            "baseline_totals": {
                name: sum(entry["baseline_totals"][name] for entry in entries)
                for name in (
                    "clean_pass_events",
                    "collision_events",
                    "attempt_timeout_events",
                )
            },
            "noisy_totals": {
                name: sum(entry["noisy_totals"][name] for entry in entries)
                for name in (
                    "clean_pass_events",
                    "collision_events",
                    "attempt_timeout_events",
                )
            },
            "worst_seed_clean_rate_delta": min(
                entry["pooled_clean_rate_delta"] for entry in entries
            ),
            "max_noisy_torque_utilization_p99": max(
                entry["noisy_max_torque_utilization_p99"] for entry in entries
            ),
            "pooled_phase_speed_deltas_mps": pooled_phase_speed_deltas,
        }

    return {
        "schema_version": 1,
        "protocol": "O3a-two-specialist-three-seed-range-noise-campaign-v1",
        "source_commit": source_commit,
        "range_noise_distribution": "bounded-uniform",
        "range_noise_bound_m": 0.02,
        "seed_decisions": ordered,
        "specialist_summaries": specialist_summaries,
        "decision": "simulation_pass_pending_measured_sensor",
        "physical_motion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", nargs=6, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_o3a_campaign(tuple(args.decisions), args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"o3a_campaign_decision={result['decision']}")
    print(f"o3a_campaign_retained={args.output}")


if __name__ == "__main__":
    main()
