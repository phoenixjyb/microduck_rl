"""One source-bound, evaluation-only F1-M motor-history diagnostic."""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

from mjlab_microduck import foundation_motor_experiment as exp
from mjlab_microduck.first_attempt_smoke import require, sha256, canonical
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.motor_trace_audit import audit_motor_trace
from mjlab_microduck.recovery_ab import verify_source, write_new

PROTOCOL = "f1m-motor-timing-s503-v1"
OUTPUT = exp.OUTPUT.parent / PROTOCOL
ORIGINAL = exp.OUTPUT
ORIGINAL_SOURCE = "fbc0b73ca1a61016ea8e4002f1af65e24990e892"
MANIFEST_SHA = "cf0954316a0523f54104e4ddf0b1982f4e3dde9ef53e5352c75aad0883c0a788"
DECISION_SHA = "1c5bb1f83bbd86a2a8eea8a9d65e71bdad43e0f8e041df4f484018e07aecfe7f"
ARMS = ("parent", "control", "motor")
SEED, CHILD_SECONDS, SERVICE_SECONDS = 503, 120, 600


def verify_original():
    require(sha256(ORIGINAL / "manifest.json") == MANIFEST_SHA
            and sha256(ORIGINAL / "decision.json") == DECISION_SHA, "immutable F1-M identity")
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    require(manifest["source"] == ORIGINAL_SOURCE and manifest["protocol"] == exp.PROTOCOL,
            "original source and protocol")
    require(len(manifest["files"]) == 83, "complete retained payload count")
    for name, row in manifest["files"].items():
        path = ORIGINAL / name
        require(path.resolve().is_relative_to(ORIGINAL.resolve()), "retained path stays inside original")
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                "immutable payload: "+name)
    require(set(manifest["files"]) == {str(p.relative_to(ORIGINAL)) for p in ORIGINAL.rglob("*")
                                     if p.is_file() and p.name != "manifest.json"}, "exact original file coverage")
    reports = {}
    for arm in ARMS:
        report = json.loads((ORIGINAL / f"{arm}-s{SEED}.json").read_text())
        _, identity = exp.checkpoint(arm, ORIGINAL_SOURCE)
        exp.validate_controller_trace(report, True, protocol=exp.PROTOCOL, seeds=(SEED,), checkpoint_sha=identity)
        require(not report["safety_failures"], "absolute-safe entry gate for timing measurement")
        reports[arm] = report
    old = json.loads((ORIGINAL / "decision.json").read_text())
    require(old["decision"] == "numerical-gate-stop"
            and canonical(exp.paired_decision(*[reports[a] for a in ARMS])) == canonical(old["pairs"][0]),
            "unchanged original numerical rejection")
    exp.preserved()
    return reports


def compare_historical(report, original):
    # Bit-exact replay is a measurement-validity gate, not a new acceptance gate.
    # A mismatch is retained and stopped; do not select/retry a favorable replay.
    strip = lambda r: {k: v for k, v in r.items() if k not in ("source", "motor_trace")}
    require(canonical(strip(report)) == canonical(strip(original)), "motor instrumentation replay differs from retained case")


def run_child(arm, source, end):
    require(time.monotonic()+CHILD_SECONDS+40 < end, "bounded sequential timing budget")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS+40) < exp.DEADLINE,
            "timing child fits overnight deadline")
    write_new(OUTPUT / f"{arm}-idle.json", wait_idle())
    with (OUTPUT / f"{arm}.log").open("x") as log:
        result = subprocess.run([sys.executable, "-m", "mjlab_microduck.foundation_motor_timing",
            "--source", source, "--child", arm], stdout=log, stderr=subprocess.STDOUT,
            timeout=CHILD_SECONDS, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                                      "PYTHONUNBUFFERED": "1"})
    require(result.returncode == 0, f"{arm} timing child failed")


def campaign(source):
    verify_source(source)
    require(dt.datetime.now(dt.timezone.utc) < exp.LAST_START, "no new GPU work after06:00")
    originals = verify_original()
    runtime = exp.training.runtime_identity()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    end = time.monotonic()+SERVICE_SECONDS
    rows = {}
    decision = dict(protocol=PROTOCOL, source=source, original_source=ORIGINAL_SOURCE,
        decision="runtime-failure-stop", arms=rows, optimizer_updates=0, policy_acceptance=False,
        physical_motion_authorized=False, curriculum_promotion=False)
    try:
        write_new(OUTPUT / "launch.json", dict(source=source, protocol=PROTOCOL, original_source=ORIGINAL_SOURCE,
            original_manifest_sha256=MANIFEST_SHA, original_decision_sha256=DECISION_SHA,
            runtime=runtime, idle=wait_idle(), arms=ARMS, seed=SEED, optimizer_updates=0,
            child_seconds=CHILD_SECONDS, service_seconds=SERVICE_SECONDS))
        for arm in ARMS:
            run_child(arm, source, end)
            path = OUTPUT / f"{arm}.json"
            report = json.loads(path.read_text())
            rows[arm] = dict(report_sha256=sha256(path), safety_failures=report["safety_failures"])
            if report["safety_failures"]:
                decision.update(decision="safety-stop", failures=report["safety_failures"])
                break
            compare_historical(report, originals[arm])
            analysis = audit_motor_trace(report)
            require(analysis["decision"] == "timing-measurement-only", "complete motor timing evidence")
            write_new(OUTPUT / f"{arm}-timing.json", analysis)
            rows[arm]["analysis_sha256"] = sha256(OUTPUT / f"{arm}-timing.json")
        else:
            decision["decision"] = "timing-measurement-only"
        verify_original()
    except Exception as error:
        decision.update(decision="runtime-failure-stop", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT / "decision.json", decision)
        files = {str(p.relative_to(OUTPUT)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                 for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT / "manifest.json", dict(source=source, protocol=PROTOCOL, files=files))
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", choices=ARMS)
    args = parser.parse_args()
    if not args.child:
        campaign(args.source)
        return
    verify_source(args.source); exp.training.runtime_identity(); verify_original(); exp.check_host()
    launch = json.loads((OUTPUT / "launch.json").read_text())
    require(launch["source"] == args.source and not (OUTPUT / "decision.json").exists(), "open exact timing run")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS) < exp.DEADLINE,
            "timing deadline")
    path, identity = exp.checkpoint(args.child, ORIGINAL_SOURCE)
    report = exp.run_control(checkpoint=path, seed=SEED, protocol=exp.PROTOCOL,
        retain_route_trace=True, command_adapter=exp.HeadingHold(True), retain_motor_trace=True)
    report.update(source=args.source, retained_checkpoint_hashes=exp.preserved())
    # Preserve raw evidence before reporting/consistency checks can fail.
    write_new(OUTPUT / f"{args.child}.json", report)
    exp.validate_controller_trace(report, True, protocol=exp.PROTOCOL, seeds=(SEED,), checkpoint_sha=identity)


if __name__ == "__main__":
    main()
