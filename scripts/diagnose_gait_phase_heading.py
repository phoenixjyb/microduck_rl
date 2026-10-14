"""Authenticate the three retained steering packets and diagnose contact-phase drift.

CPU only. This reports associations from sealed traces; it never changes a gate,
checkpoint, service, training recipe, or policy-acceptance decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from mjlab_microduck import gait_path_qualification as route
from mjlab_microduck import gait_phase_heading_diagnosis as diagnosis

PROTOCOL = "microduck-retained-steering-phase-diagnosis-v1"
REFERENCES = {
    "steady": dict(run="v8-steady-steering-7f55ded7-20261013",
        source="7f55ded7c615e0adda4f859e6e05dde40c447e66", stem="steady-steering",
        checkpoint="d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f",
        summary="182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155",
        manifest="8e0954b7853d9e810a9e13127ba0c9f86e2be1d16896da81cceffbae0b9c6de6"),
    "cumulative-heading": dict(run="v8-projected-heading-fe23cc80-20261013",
        source="fe23cc80fa9f4d624c8d79d792e0f45281239580", stem="projected-heading",
        checkpoint="95dde4bcf89c13462efa9c0f599b20239a13058dd596718360e09668d2b97440",
        summary="8a64c7b07c410f4f4adc01c72422532e2be1c843a6ac8cb3be25eeb5d59ffcad",
        manifest="30aa449966a0efef0ac0ec8d50795564f72234cd2e4cf2f214deee61681e5e97"),
    "projected-rate": dict(run="v8-projected-yaw-bias-db9a77ae-20261014",
        source="db9a77ae1baf27efa19a9fc9c16baacbc58f30dc", stem="projected-yaw-bias",
        checkpoint="dc9f4861910c91ed8a58a11f62ad785908b9e5b152716ab2a775c27fa6b89fed",
        summary="5c411b9d72a988a66245e34deda1dc53875180d414200479b85b21e366dd90fc",
        manifest="36e4a90ea17cea80fccf20adf951e927143ed5f37d6b4f20acc0facb808cadc7"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), "regular non-symlink evidence file")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_bound(path, digest):
    require(sha(path) == digest, f"sealed evidence hash mismatch: {path.name}")
    return json.loads(path.read_text())


def group_reports(cases, cycle_periods):
    """Equal-world endpoint means, duration-weighted phases; never a new gate."""
    groups = []
    for speed, yaw in ((0., 0.), (.3, 0.), (.2, -.2), (.2, .2)):
        chosen = [c for c in cases if (c["speed"], c["yaw"]) == (speed, yaw)]
        require(len(chosen) == 3, "three held-out seeds per command")
        worlds = [w for c in chosen for w in c["worlds"]]
        require(len(worlds) == 24, "24 worlds per command")
        duration = math.fsum(w["settled_duration_s"] for w in worlds)
        phases = {}
        for phase in diagnosis.PHASES:
            bins = [w["phases"][phase] for w in worlds]
            span = math.fsum(b["duration_s"] for b in bins)
            signed = math.fsum(b["signed_residual_rad"] for b in bins)
            absolute = math.fsum(b["absolute_residual_rad"] for b in bins)
            phases[phase] = dict(interval_count=sum(b["interval_count"] for b in bins),
                duration_s=span, fraction_of_measured_duration=span / duration,
                signed_residual_rad=signed, absolute_interval_residual_rad=absolute,
                signed_residual_rate_rad_s=signed / span if span else None,
                absolute_interval_residual_rate_rad_s=absolute / span if span else None)
        periods = sorted(cycle_periods[(speed, yaw)])
        groups.append(dict(command=[speed, yaw], worlds=24,
            interval_count=sum(p["interval_count"] for p in phases.values()),
            pooled_duration_s=duration,
            mean_signed_endpoint_rate_rad_s=math.fsum(
                w["signed_endpoint_rate_rad_s"] for w in worlds) / 24,
            mean_absolute_endpoint_rate_rad_s=math.fsum(
                w["absolute_endpoint_rate_rad_s"] for w in worlds) / 24,
            phases=phases, physical_cycles=len(periods), cycle_duration_s=(dict(
                median=statistics.median(periods), p10=periods[int(len(periods)*.1)],
                p90=periods[int(len(periods)*.9)]) if periods else None)))
    return groups


def diagnose_packet(root, state_root, reference):
    packet = root / "artifacts/training" / reference["run"]
    state = state_root / (reference["run"] + "-handoff")
    manifest = read_bound(state / (reference["stem"] + "-manifest.json"), reference["manifest"])
    summary = read_bound(state / (reference["stem"] + "-summary.json"), reference["summary"])
    require(manifest["source"] == summary["source"] == reference["source"], "sealed source identity")
    require(summary["evaluations"]["final-eval"]["checkpoint_sha256"] == reference["checkpoint"],
            "sealed final checkpoint identity")
    require(summary["policy_acceptance"] is False and summary["physical_motion_authorized"] is False,
            "retained reference is not an accepted policy")
    result = read_bound(packet / "final-eval/result.json",
                        summary["evaluations"]["final-eval"]["result_sha256"])
    require(result["source"] == reference["source"] and result["checkpoint_sha256"] == reference["checkpoint"],
            "bound evaluation source and checkpoint")
    require(result["policy_acceptance"] is False and result["physical_motion_authorized"] is False,
            "evaluation authorization remains false")
    require(len(result["raw_case_files"]) == 12, "exact 12-case retained matrix")
    reports, periods = [], {cmd: [] for cmd in ((0., 0.), (.3, 0.), (.2, -.2), (.2, .2))}
    for index, ((seed, speed, yaw), entry) in enumerate(zip(route.IDENTITIES, result["raw_case_files"], strict=True)):
        filename = f"s{seed}-c{index % 4}.json"
        require(entry["file"] == filename, "fixed ordered trace filename")
        item = manifest["files"]["final-eval/" + filename]
        require(item["sha256"] == entry["sha256"], "result and manifest agree on trace hash")
        path = packet / "final-eval" / filename
        require(path.stat().st_size == item["bytes"], "trace byte count")
        case = read_bound(path, item["sha256"])
        require((case["seed"], case["speed"], case["yaw"]) == (seed, speed, yaw)
                and case["checkpoint_sha256"] == reference["checkpoint"]
                and case["complete"] is True and not case["terminal_worlds"], "complete bound case")
        report = diagnosis.summarize_case(case, case["gait_path_trace"])
        report["input_file"] = filename
        report["input_sha256"] = item["sha256"]
        reports.append(report)
        route_report = route.summarize_case(case, case["gait_path_trace"])
        periods[(speed, yaw)].extend(c["duration_s"] for w in route_report["worlds"] for c in w["cycles"])
    return dict(reference=reference, cases=reports, groups=group_reports(reports, periods),
                unchanged_route_decision=result["route_gate"],
                policy_acceptance=False, physical_motion_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "new output path required; do not overwrite evidence")
    report = dict(protocol=PROTOCOL, diagnostic_protocol=diagnosis.PROTOCOL,
        implementation_sha256=sha(Path(diagnosis.__file__)), script_sha256=sha(Path(__file__)),
        references={name: diagnose_packet(args.root, args.state_root, value)
                    for name, value in REFERENCES.items()},
        interpretation="descriptive association only; neither causal proof nor a qualification gate",
        policy_acceptance=False, physical_motion_authorized=False)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(payload)
    print("DIAGNOSIS_SHA256", sha(args.output))
    print("DIAGNOSIS_BYTES", args.output.stat().st_size)


if __name__ == "__main__":
    main()
