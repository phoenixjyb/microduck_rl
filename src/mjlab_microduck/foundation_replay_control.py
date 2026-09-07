"""Two same-source parent503 evaluations with motor-history recording OFF."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from mjlab_microduck import foundation_motor_timing as timing
from mjlab_microduck.first_attempt_smoke import canonical, require, sha256
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import differences

exp = timing.exp
PROTOCOL = "f1m-replay-off-off-s503-v1"
OUTPUT = exp.OUTPUT.parent / PROTOCOL
MODES = ("first", "second")
CHILD_SECONDS, SERVICE_SECONDS = 120, 420
TIMING_MANIFEST = "001ed5d5d9060a687a206af50d11f00cf26019b62d50d9f34b331a5beee51037"
TIMING_DECISION = "2661a4c0b7e7e6c4d7547d603905b69a0ca7fe7b3a09f24ed351114e44da5160"
HELPERS = {
    "utils/random.py": "5ba445435efa5f064aac4b572d035c6a6e98ca3ce459faf1f00dc6f73cbabbc8",
    "utils/torch.py": "87e0673c850ae6cd5306c9533c928ddb570396323b904f7bbbab589990bc44b1",
    "rl/vecenv_wrapper.py": "d458aa421d72c979d0269a24e51a4b49a02e4ec8cb5a9d10e47290aa5d86a3e2",
}
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_LAUNCH_BLOCKING")


def fingerprint():
    """Read settings only; do not initialize CUDA or change backend/RNG state."""
    return dict(environment={k: os.environ.get(k) for k in ENV_KEYS},
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_enabled=torch.backends.cudnn.enabled,
        matmul_fp32_precision=torch.backends.cuda.matmul.fp32_precision,
        cudnn_fp32_precision=torch.backends.cudnn.fp32_precision,
        default_dtype=str(torch.get_default_dtype()), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), cuda_initialized=torch.cuda.is_initialized(),
        python_hash_probe=hash("microduck-repeatability-probe-v1"))


def verify_history():
    timing.verify_original()
    p = timing.OUTPUT
    require(sha256(p / "manifest.json") == TIMING_MANIFEST
            and sha256(p / "decision.json") == TIMING_DECISION, "immutable failed timing run")
    manifest = json.loads((p / "manifest.json").read_text())
    require(len(manifest["files"]) == 5, "one timing case retained")
    for name, row in manifest["files"].items():
        f = p / name
        require(f.resolve().is_relative_to(p.resolve()) and f.is_file()
                and f.stat().st_size == row["bytes"] and sha256(f) == row["sha256"], "unchanged timing payload")
    require(set(manifest["files"]) == {str(f.relative_to(p)) for f in p.rglob("*")
                                     if f.is_file() and f.name != "manifest.json"}, "exact timing file coverage")
    import mjlab
    actual = {name: sha256(Path(mjlab.__file__).parent / name) for name in HELPERS}
    require(actual == HELPERS, "inspected runtime helpers unchanged")
    return actual


def validate_case(report, source):
    require(report["source"] == source and "motor_trace" not in report, "same-source recording OFF")
    exp.validate_controller_trace(report, True, protocol=exp.PROTOCOL, seeds=(503,), checkpoint_sha=exp.MODEL_SHA)


def compare_pair(first, second, states):
    canonical([first, second, states])
    validate_case(first, first["source"]); validate_case(second, first["source"])
    require(not first["safety_failures"] and not second["safety_failures"], "absolute-safe repeatability controls")
    settings = lambda r: {k: v for k, v in r.items() if k != "python_hash_probe"}
    for phase in ("before", "after"):
        require(canonical(settings(states["first"][phase])) == canonical(settings(states["second"][phase])),
                "numerical settings/initialization fingerprint mismatch")
    diff = differences(first, second)
    require((not diff) == (canonical(first) == canonical(second)), "exact comparison accounting")
    trace = {}
    for field, columns in (("position", "position_columns"), ("velocity", "velocity_columns")):
        a, b = (torch.tensor(r["route_trace"][field], dtype=torch.float64) for r in (first, second))
        delta = (a-b).abs(); indices = (delta != 0).nonzero()
        trace[field] = dict(first_changed_index=indices[0].tolist() if len(indices) else None,
            max_absolute_by_column={name: float(delta[:,:,i].max())
                                    for i, name in enumerate(first["route_trace"][columns])})
    return dict(decision="recording-disabled-same-source-divergence" if diff else "recording-disabled-exact-match-in-this-pair",
        exact_reports_equal=not diff, difference_count=len(diff), differences=diff, trace_differences=trace,
        python_hash_probe_equal=states["first"]["before"]["python_hash_probe"] == states["second"]["before"]["python_hash_probe"],
        policy_acceptance=False, physical_motion_authorized=False, optimizer_updates=0,
        specific_numerical_cause_established=False, third_case_admitted=False)


def run_child(mode, source, end):
    require(time.monotonic()+CHILD_SECONDS+40 < end, "bounded repeatability budget")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS+40) < exp.DEADLINE,
            "repeatability closeout before07:00")
    write_new(OUTPUT / f"{mode}-idle.json", wait_idle())
    with (OUTPUT / f"{mode}.log").open("x") as log:
        result = subprocess.run([sys.executable, "-m", "mjlab_microduck.foundation_replay_control",
            "--source", source, "--child", mode], stdout=log, stderr=subprocess.STDOUT,
            timeout=CHILD_SECONDS, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                                      "PYTHONUNBUFFERED": "1"})
    require(result.returncode == 0, f"{mode} replay child failed")


def campaign(source):
    verify_source(source)
    require(dt.datetime.now(dt.timezone.utc) < exp.LAST_START, "no new GPU work after06:00")
    helpers, runtime = verify_history(), exp.training.runtime_identity()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    end = time.monotonic()+SERVICE_SECONDS
    decision = dict(protocol=PROTOCOL, source=source, decision="runtime-failure-stop", reports=[],
        optimizer_updates=0, policy_acceptance=False, physical_motion_authorized=False, third_case_admitted=False)
    reports, states = {}, {}
    try:
        write_new(OUTPUT / "launch.json", dict(protocol=PROTOCOL, source=source, runtime=runtime, helpers=helpers,
            idle=wait_idle(), modes=MODES, seed=503, checkpoint_sha256=exp.MODEL_SHA,
            child_seconds=CHILD_SECONDS, service_seconds=SERVICE_SECONDS, motor_history_enabled=False,
            numerical_environment={k: os.environ.get(k) for k in ENV_KEYS}))
        for mode in MODES:
            run_child(mode, source, end)
            path = OUTPUT / f"{mode}.json"
            report = json.loads(path.read_text()); reports[mode] = report
            decision["reports"].append(dict(mode=mode, sha256=sha256(path)))
            validate_case(report, source)
            if report["safety_failures"]:
                decision.update(decision="absolute-safety-stop", failures=report["safety_failures"])
                break
            states[mode] = json.loads((OUTPUT / f"{mode}-fingerprint.json").read_text())
        else:
            decision.update(compare_pair(reports["first"], reports["second"], states))
        verify_history()
    except Exception as error:
        decision.update(decision="runtime-failure-stop", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT / "decision.json", decision)
        files = {str(p.relative_to(OUTPUT)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                 for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT / "manifest.json", dict(source=source, protocol=PROTOCOL, files=files))
    print(json.dumps({k: v for k, v in decision.items() if k != "differences"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--child", choices=MODES)
    args = parser.parse_args()
    if not args.child:
        campaign(args.source); return
    verify_source(args.source); verify_history(); exp.training.runtime_identity(); exp.check_host()
    launch = json.loads((OUTPUT / "launch.json").read_text())
    require(launch["source"] == args.source and not (OUTPUT / "decision.json").exists(), "open exact-source control")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS) < exp.DEADLINE, "replay deadline")
    before = fingerprint()
    report = exp.run_control(checkpoint=exp.MODEL, seed=503, protocol=exp.PROTOCOL,
        retain_route_trace=True, command_adapter=exp.HeadingHold(True), retain_motor_trace=False)
    after = fingerprint()
    report.update(source=args.source, retained_checkpoint_hashes=exp.preserved())
    write_new(OUTPUT / f"{args.child}.json", report)
    write_new(OUTPUT / f"{args.child}-fingerprint.json", dict(before=before, after=after))


if __name__ == "__main__":
    main()
