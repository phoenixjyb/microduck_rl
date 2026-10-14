"""CPU-only authenticated cycle diagnosis; no gate, policy or service changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import statistics

from mjlab_microduck import gait_path_qualification as route
from mjlab_microduck.community_policy_inspection import _read_payload

PROTOCOL = "microduck-retained-cycle-heading-diagnosis-v1"
REFERENCE = dict(run="v8-motor-headroom-06c2b709-20261014",
    source="06c2b709dc1c536d83edf7f6f2ab5f33422cb99f",
    summary="61391992bf577974883c454155f01367d8faccad916734719aef06a30e574671",
    manifest="f5e7d7a15b6796172d57af354c022c21ec6f7a7281ce40ac9049b8d304341bbf",
    checkpoints=dict(
        control="dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a",
        headroom="24677f3b4ba15a8631221f8fcdab44d655c597b68ea520ae42e893dfcf65a489"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path, expected_sha):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
            "owned regular non-symlink evidence file")
    raw = _read_payload(path)
    require(hashlib.sha256(raw).hexdigest() == expected_sha,
            "sealed evidence hash mismatch: " + path.name)
    return raw


def read_bound(path, expected_sha):
    return json.loads(read_bytes(path, expected_sha))


def load_core(path=None, expected_sha=None):
    """Optional authenticated standalone core avoids changing an active worktree."""
    require((path is None) == (expected_sha is None), "core file and hash must be paired")
    if path is None:
        from mjlab_microduck import cycle_heading_diagnosis as core
        core.authenticated_source_sha256 = hashlib.sha256(_read_payload(Path(core.__file__))).hexdigest()
        return core
    payload = read_bytes(path, expected_sha)
    require(len(payload) <= 256*1024, "bounded trusted diagnostic implementation")
    # Execute the authenticated bytes, not a second unauthenticated path read.
    name = "_microduck_authenticated_cycle_diagnostic"
    import types
    core = types.ModuleType(name)
    core.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), core.__dict__)
    core.authenticated_source_sha256 = hashlib.sha256(payload).hexdigest()
    return core


def group_cases(cases):
    """Cycle-weighted descriptive summaries; no acceptance or causal inference."""
    groups = []
    for speed, yaw in ((0., 0.), (.3, 0.), (.2, -.2), (.2, .2)):
        selected = [case for case in cases if (case["speed"], case["yaw"]) == (speed, yaw)]
        worlds = [world for case in selected for world in case["worlds"]]
        cycles = [cycle for world in worlds for cycle in world["cycles"]]
        phases = sorted({p for c in cycles for p in c["phase_time_s"]})
        exposure = {p: sum(c["phase_time_s"].get(p, 0.) for c in cycles) for p in phases}
        peaks = {p: sum(c["peak_interval_phase"] == p and c["peak_only"] for c in cycles)
                 for p in phases}
        statistics_by_field = {}
        for field in ("duration_s", "signed_endpoint_residual_rad", "max_abs_residual_rad",
                      "max_detrended_abs_residual_rad", "peak_elapsed_fraction",
                      "head_yaw_velocity_at_peak_rad_s"):
            values = [c[field] for c in cycles]
            statistics_by_field[field] = dict(
                mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                max=max(values) if values else None)
        groups.append(dict(speed=speed, yaw=yaw, cases=len(selected), worlds=len(worlds),
            cycle_count=len(cycles),
            **{key: sum(c[key] for c in cycles) for key in ("endpoint_fail", "max_fail", "peak_only")},
            cycle_weighted_statistics=statistics_by_field,
            phase_exposure_s=exposure, peak_only_phase_counts=peaks,
            peak_only_events_per_phase_second={p: peaks[p]/exposure[p] if exposure[p] else None
                                               for p in phases},
            phase_interpretation="contact association only; duration-normalized counts are not causal evidence"))
    return groups


def diagnose_packet(root, state_root, core):
    packet = Path(root) / "artifacts/training" / REFERENCE["run"]
    state = Path(state_root) / (REFERENCE["run"] + "-handoff")
    summary = read_bound(state / "headroom-summary.json", REFERENCE["summary"])
    manifest = read_bound(state / "headroom-manifest.json", REFERENCE["manifest"])
    require(summary["source"] == manifest["source"] == REFERENCE["source"]
            and summary["status"] == "verified-evidence-not-policy-acceptance"
            and summary["policy_acceptance"] is False
            and summary["physical_motion_authorized"] is False, "sealed packet identity and non-acceptance")
    arms = {}
    for arm, checkpoint in REFERENCE["checkpoints"].items():
        sealed = summary["evaluations"][arm]
        require(sealed["checkpoint_sha256"] == checkpoint, "sealed arm checkpoint identity")
        result = read_bound(packet/arm/"final-eval/result.json", sealed["result_sha256"])
        require(result["source"] == REFERENCE["source"] and result["arm"] == arm
                and result["evaluation"]["checkpoint_sha256"] == checkpoint
                and result["policy_acceptance"] is False
                and result["physical_motion_authorized"] is False, "bound arm evaluation identity")
        raw_files = result["evaluation"]["raw_case_files"]
        require(len(raw_files) == 12, "complete fixed twelve-case matrix")
        cases = []
        for index, (identity, entry) in enumerate(zip(route.IDENTITIES,raw_files,strict=True)):
            seed,speed,yaw = identity
            filename = f"s{seed}-c{index%4}.json"
            require(entry["file"] == filename, "fixed ordered trace filename")
            key = f"{arm}/final-eval/{filename}"
            item = manifest["files"][key]
            require(entry["sha256"] == item["sha256"], "manifest and result trace hashes agree")
            payload = read_bytes(packet/arm/"final-eval"/filename, item["sha256"])
            require(len(payload) == item["bytes"], "sealed raw trace byte count")
            row = json.loads(payload)
            require((row["seed"],row["speed"],row["yaw"]) == identity
                    and row["checkpoint_sha256"] == checkpoint
                    and row["complete"] is True and row["terminal_worlds"] == [],
                    "exact complete first-episode raw trace")
            case = core.summarize_case(row,row["gait_path_trace"])
            case.update(input_file=key,input_sha256=item["sha256"])
            cases.append(case)
        arms[arm] = dict(checkpoint_sha256=checkpoint, cases=cases, command_groups=group_cases(cases),
            original_route_gate=result["evaluation"]["route_gate"],
            original_legacy_gate=result["evaluation"]["legacy_gate"],
            policy_acceptance=False,physical_motion_authorized=False)
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--state-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--core-file",type=Path)
    parser.add_argument("--core-sha256")
    args = parser.parse_args()
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "offline diagnostic requires hidden uninitialized CUDA")
    require(not args.output.exists() and not args.output.is_symlink(), "new output path required")
    core = load_core(args.core_file,args.core_sha256)
    script_bytes = _read_payload(Path(__file__))
    report = dict(protocol=PROTOCOL, diagnostic_protocol=core.PROTOCOL,
        core_sha256=core.authenticated_source_sha256,
        script_sha256=hashlib.sha256(script_bytes).hexdigest(), reference=REFERENCE,
        route_implementation_sha256=hashlib.sha256(_read_payload(Path(route.__file__))).hexdigest(),
        phase_implementation_sha256=hashlib.sha256(_read_payload(Path(core.phase_diagnosis.__file__))).hexdigest(),
        arms=diagnose_packet(args.root,args.state_root,core),
        interpretation="descriptive only; detrending and contact association are not acceptance or causal evidence",
        policy_acceptance=False,physical_motion_authorized=False)
    require(not torch.cuda.is_initialized(), "diagnostic must not initialize CUDA")
    payload = (json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    print("DIAGNOSIS_SHA256",hashlib.sha256(payload).hexdigest())
    print("DIAGNOSIS_BYTES",len(payload))


if __name__ == "__main__":
    main()
