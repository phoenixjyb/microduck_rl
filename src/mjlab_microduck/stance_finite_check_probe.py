"""Source-separated WSL finite-check microbenchmark; no runtime integration.

An archived reviewed tool reads the unchanged runtime's solved tensor views.
Only checker calls are measured. No policy tick, Euler step or optimizer runs.
"""

import argparse
from hashlib import sha256
import io
import math
import os
from pathlib import Path
import subprocess
import time
from types import ModuleType

import torch

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_plant_evidence as plant
from mjlab_microduck import stance_training_smoke as smoke
from mjlab_microduck import stance_lean_throughput as throughput
from mjlab_microduck.first_attempt_smoke import canonical, require


PROTOCOL = "football-b1n-wsl-finite-check-probe-v1"
RUNTIME_SOURCE = "987b452dbfdb4cba7c7fef79f419790bd602ae3a"
TOOL_PATH = "src/mjlab_microduck/stance_finite_check_probe.py"
CHECK_PATH = "src/mjlab_microduck/stance_solved_field_check.py"
ORDER = ("legacy", "packed", "packed", "legacy", "legacy", "packed")
WARMUP, ITERATIONS = 100, 1000
CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT = 120, 180, 600
INTEGER_FIELDS = ("deadline_unix", "worlds", "warmup_checks", "measured_checks", "child_seconds",
                  "service_seconds", "closeout_seconds", "memory_max_bytes", "cpu_quota_per_sec_usec", "nice")
NO_ADMISSION = dict(runtime_integrated=False, training_admitted=False,
                    timing_qualified=False, policy_acceptance=False,
                    learned_stance=False, football_balance=False,
                    physical_motion_authorized=False)


def check_window(deadline, now=None):
    now = time.time() if now is None else now
    require(type(deadline) is int and type(now) in (int, float) and math.isfinite(now)
            and CHILD_SECONDS + 60 + CLOSEOUT < deadline - now <= 3600,
            "bounded service and closeout window required")


def output_path(source):
    host.supervisor.hex_id(source, 40)
    return host.ROOT / "artifacts/evaluations" / ("stance-finite-check-probe-" + source[:12])


def service_name(source):
    host.supervisor.hex_id(source, 40)
    return "microduck-finite-check-probe-" + source[:12] + ".service"


def tool_identity(source, runner_sha, checker_sha):
    host.supervisor.hex_id(source, 40)
    result = {}
    for name, path, expected in (("runner.py", TOOL_PATH, runner_sha),
                                 ("checker.py", CHECK_PATH, checker_sha)):
        host.supervisor.hex_id(expected, 64)
        raw = subprocess.check_output(["git", "show", source + ":" + path], cwd=host.ROOT, timeout=5)
        require(sha256(raw).hexdigest() == expected, "independently pinned tool Git bytes")
        result[name] = dict(path=path, sha256=expected, bytes=len(raw))
    require(host.digest(Path(__file__).resolve()) == runner_sha, "executing pinned runner bytes")
    return dict(source=source, files=result, runtime_source=RUNTIME_SOURCE)


def plan(source, runner_sha, checker_sha, deadline):
    require(execution.PROFILE["name"] == execution.WSL, "explicit fixed WSL profile")
    tool = tool_identity(source, runner_sha, checker_sha)
    raw = subprocess.check_output(["git", "show", source + ":" + CHECK_PATH], cwd=host.ROOT, timeout=5)
    preflight = cpu_preflight(checker_from_payload(raw, checker_sha, CHECK_PATH))
    return dict(protocol=PROTOCOL, tool=tool,
        inputs=host.identity(RUNTIME_SOURCE), deadline_unix=deadline,
        child_seconds=CHILD_SECONDS, service_seconds=SERVICE_SECONDS, closeout_seconds=CLOSEOUT,
        memory_max_bytes=6 * 1024**3, cpu_quota_per_sec_usec=2_000_000, nice=10,
        worlds=64, order=list(ORDER), warmup_checks=WARMUP, measured_checks=ITERATIONS,
        forward_graph=False, policy_inferences=0, physics_steps=0, optimizer_updates=0,
        cpu_preflight=preflight, physics_step_meaning="Euler-integration-ticks-not-initialization-forward",
        **NO_ADMISSION)


def prepare(source, runner_sha, checker_sha, deadline):
    check_window(deadline)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CPU-only preparation")
    launch = plan(source, runner_sha, checker_sha, deadline)
    runtime = plant.runtime_bytes(RUNTIME_SOURCE, plant.build_entity().compile())
    root = output_path(source)
    root.mkdir(exist_ok=False)
    for name, record in launch["tool"]["files"].items():
        raw = subprocess.check_output(["git", "show", source + ":" + record["path"]], cwd=host.ROOT, timeout=5)
        require(sha256(raw).hexdigest() == record["sha256"], "fresh tool archive hash")
        smoke.write_bytes(root / name, raw)
    smoke.write_bytes(root / "runtime.json", runtime)
    launch["runtime_sha256"] = sha256(runtime).hexdigest()
    host.supervisor.write_json(root / "launch.json", launch)
    return dict(output=str(root), service=service_name(source), launch_sha256=host.digest(root / "launch.json"))


def checked(source, runner_sha, checker_sha, launch_sha):
    host.supervisor.hex_id(launch_sha, 64)
    root = output_path(source)
    require(host.digest(root / "launch.json") == launch_sha, "independent launch hash")
    launch = host.supervisor.parse(host.supervisor.file_bytes(root / "launch.json"))
    require(all(type(launch[key]) is int for key in INTEGER_FIELDS),
            "integer declaration counts and resource bounds")
    expected = plan(source, runner_sha, checker_sha, launch["deadline_unix"])
    expected["runtime_sha256"] = host.digest(root / "runtime.json")
    require(canonical(launch) == canonical(expected), "unchanged tool, host, runtime and launch")
    for name, record in launch["tool"]["files"].items():
        require(host.digest(root / name) == record["sha256"], "retained tool hash")
    plant.checked_runtime(host.supervisor.parse(host.supervisor.file_bytes(root / "runtime.json")), RUNTIME_SOURCE)
    return launch


def checker_from_payload(raw, expected_sha, filename):
    require(sha256(raw).hexdigest() == expected_sha, "exact immutable checker payload")
    module = ModuleType("reviewed_stance_finite_checker")
    module.__file__ = str(filename)
    exec(compile(raw, str(filename), "exec"), module.__dict__)
    return module


def load_checker(root, expected_sha):
    return checker_from_payload(host.supervisor.file_bytes(root / "checker.py"), expected_sha, root / "checker.py")


def cpu_preflight(checker):
    """Deterministic checker fixtures, not simulator values or capability."""
    values = {name: torch.arange(64 * (index + 1), dtype=torch.float32, device="cpu").reshape(64, -1)
              for index, name in enumerate(checker.FIELDS)}
    before = {name: value.view(torch.uint8).clone() for name, value in values.items()}
    checker.legacy_check(values)
    checker.packed_check(values)
    faults = errors(checker, values)
    require(all(torch.equal(before[name], value.view(torch.uint8)) for name, value in values.items()),
            "synthetic CPU fixture bits unchanged")
    return dict(schema="finite-check-cpu-fixture-v1", decision="synthetic-cpu-predicates-equivalent-only",
                device="cpu", fields=list(checker.FIELDS), fault_count=len(faults),
                faults_sha256=sha256(canonical(faults).encode()).hexdigest(), input_bits_unchanged=True,
                physics_executed=False, policy_inferences=0, **NO_ADMISSION)


def errors(checker, values):
    result = []
    for name in checker.FIELDS:
        for label, bad in (("not-a-number", float("nan")), ("positive-infinity", float("inf")),
                           ("negative-infinity", -float("inf"))):
            copied = {key: value.clone(memory_format=torch.contiguous_format) for key, value in values.items()}
            copied[name].reshape(-1)[-1] = bad
            messages = []
            for check in (checker.legacy_check, checker.packed_check):
                try:
                    check(copied)
                except ValueError as exc:
                    messages.append(str(exc))
                else:
                    raise ValueError("injected finite-check fault was accepted")
            require(messages == ["nonfinite solved stance field: " + name] * 2,
                    "identical original first-field error")
            result.append(dict(field=name, fault=label, error=messages[0]))
    return result


def decide(rows):
    require(type(rows) is list and len(rows) == len(ORDER), "all declared timing blocks")
    for index, (variant, row) in enumerate(zip(ORDER, rows)):
        require(set(row) == {"block", "variant", "warmup_checks", "measured_checks", "seconds"}
                and type(row["block"]) is int and row["block"] == index
                and row["variant"] == variant
                and type(row["warmup_checks"]) is int and row["warmup_checks"] == WARMUP
                and type(row["measured_checks"]) is int and row["measured_checks"] == ITERATIONS
                and type(row["seconds"]) is float and math.isfinite(row["seconds"]) and row["seconds"] > 0,
                "exact ordered finite timing block")
    pairs = []
    for index in range(0, len(rows), 2):
        pair = {row["variant"]: row["seconds"] for row in rows[index:index + 2]}
        ratio = pair["packed"] / pair["legacy"]
        require(math.isfinite(ratio), "finite derived timing ratio")
        pairs.append(dict(blocks=[index, index + 1], legacy_seconds=pair["legacy"],
                          packed_seconds=pair["packed"], packed_to_legacy_ratio=ratio))
    return dict(decision="finite-check-probe-complete-not-runtime-equivalence",
                paired_blocks=pairs,
                packed_faster_in_all_three_pairs=all(row["packed_to_legacy_ratio"] < 1 for row in pairs),
                scope="checker-only-including-fresh-packing-not-end-to-end-collection",
                **NO_ADMISSION)


def child(source, runner_sha, checker_sha, launch_sha, fd):
    smoke.inherited_lease(fd)
    launch = checked(source, runner_sha, checker_sha, launch_sha)
    check_window(launch["deadline_unix"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "explicit CUDA child")
    host.wait_idle()
    root = output_path(source)
    checker = load_checker(root, checker_sha)
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    env = WarpStanceRuntime(64, device="cuda:0")
    require(env.forward_graph is None and env.wp_device.is_cuda, "unchanged eager CUDA runtime")
    require(plant.describe(env.native) == launch_plant(root), "actual compiled plant")
    with torch.no_grad():
        values = {name: env._view(name) for name in checker.FIELDS}
        checker.legacy_check(values)
        original = {name: value.detach().cpu().clone() for name, value in values.items()}
        buffer = io.BytesIO()
        torch.save(original, buffer)
        smoke.write_bytes(root / "inputs.pt", buffer.getvalue())
        faults = errors(checker, values)  # Mutated copies, never simulator views.
        rows = []
        for index, variant in enumerate(ORDER):
            check = getattr(checker, variant + "_check")
            for _ in range(WARMUP):
                check(values)
            torch.cuda.synchronize(env.device)
            started = time.monotonic()
            for _ in range(ITERATIONS):
                check(values)
            torch.cuda.synchronize(env.device)
            row = dict(block=index, variant=variant, warmup_checks=WARMUP,
                       measured_checks=ITERATIONS, seconds=time.monotonic() - started)
            rows.append(row)
            host.supervisor.write_json(root / ("block-" + str(index) + ".json"), row)
            print("Finite checker completed timing block " + str(index), flush=True)
        require(all(torch.equal(original[name].contiguous().view(torch.uint8), value.detach().cpu().contiguous().view(torch.uint8))
                    for name, value in values.items()), "input bits unchanged by every checker")
        require(not env.steps.any() and not env._view("time").any(), "zero physics steps and elapsed simulation time")
        summary = decide(rows)
        summary.update(faults=faults, input_bits_unchanged=True, physics_steps=0,
                       policy_inferences=0, optimizer_updates=0, worlds=64,
                       device=str(env.device), input_sha256=host.digest(root / "inputs.pt"))
        host.supervisor.write_json(root / "summary.json", summary)
    checked(source, runner_sha, checker_sha, launch_sha)


def launch_plant(root):
    return host.supervisor.parse(host.supervisor.file_bytes(root / "runtime.json"))["plant"]


def verify_evidence(root, report_sha, launch_sha, runner_sha, checker_sha, *, checker=None):
    """CPU hash/schema replay; does not authenticate a CUDA run or its timings."""
    root = Path(root)
    for digest in (report_sha, launch_sha, runner_sha, checker_sha):
        host.supervisor.hex_id(digest, 64)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CPU-only evidence replay")
    require(host.digest(root / "report.json") == report_sha, "independent report hash")
    report = host.supervisor.parse(host.supervisor.file_bytes(root / "report.json"))
    summary = replay_payload(root, report, launch_sha, runner_sha, checker_sha,
                             extra={"report.json"}, checker=checker)
    return dict(report_sha256=report_sha, summary=summary, live_host_rechecked=False,
                cuda_run_authenticated=False, measured_timings_independently_reproduced=False,
                **NO_ADMISSION)


def replay_payload(root, report, launch_sha, runner_sha, checker_sha, *, extra=frozenset(), checker=None):
    """Replay before publishing a successful report, or after its hash check."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CPU-only payload replay")
    names = {"runner.py", "checker.py", "runtime.json", "launch.json", "inputs.pt",
             "summary.json", "child.log", *("block-" + str(i) + ".json" for i in range(6))}
    require(set(report["files"]) == names and {p.name for p in root.iterdir()} == names | set(extra),
            "exact closed probe inventory")
    require(all(host.digest(root / name) == digest for name, digest in report["files"].items()),
            "all retained probe hashes")
    require(host.digest(root / "launch.json") == launch_sha
            and host.digest(root / "runner.py") == runner_sha
            and host.digest(root / "checker.py") == checker_sha,
            "independent launch and tool hashes")
    require(report["protocol"] == PROTOCOL and report["launch_sha256"] == launch_sha
            and report["decision"] == "finite-check-probe-complete-not-runtime-equivalence"
            and all(report[key] is False for key in NO_ADMISSION), "completed non-admitting probe")
    child_record = report["child"]
    require(type(child_record["returncode"]) is int and child_record["returncode"] == 0
            and type(child_record["elapsed_s"]) is float and math.isfinite(child_record["elapsed_s"])
            and 0 < child_record["elapsed_s"] <= CHILD_SECONDS, "bounded successful child")
    launch = host.supervisor.parse(host.supervisor.file_bytes(root / "launch.json"))
    require(all(type(launch[key]) is int for key in INTEGER_FIELDS),
            "integer declaration counts and resource bounds")
    require(launch["protocol"] == PROTOCOL
            and launch["tool"]["runtime_source"] == launch["inputs"]["source"] == RUNTIME_SOURCE
            and launch["order"] == list(ORDER) and type(launch["worlds"]) is int and launch["worlds"] == 64
            and (launch["warmup_checks"], launch["measured_checks"]) == (WARMUP, ITERATIONS)
            and (launch["child_seconds"], launch["service_seconds"], launch["closeout_seconds"]) ==
                (CHILD_SECONDS, SERVICE_SECONDS, CLOSEOUT)
            and (launch["memory_max_bytes"], launch["cpu_quota_per_sec_usec"], launch["nice"]) ==
                (6 * 1024**3, 2_000_000, 10)
            and launch["forward_graph"] is False and all(launch[key] is False for key in NO_ADMISSION)
            and all(type(launch[key]) is int and launch[key] == 0
                    for key in ("physics_steps", "policy_inferences", "optimizer_updates")),
            "fixed source-separated checker-only declaration")
    host.supervisor.hex_id(launch["tool"]["source"], 40)
    require(set(launch["tool"]["files"]) == {"runner.py", "checker.py"}, "both retained tools")
    for name, path, expected in (("runner.py", TOOL_PATH, runner_sha), ("checker.py", CHECK_PATH, checker_sha)):
        require(launch["tool"]["files"][name] == dict(path=path, sha256=expected, bytes=(root / name).stat().st_size),
                "exact tool reference")
    require(host.digest(root / "runtime.json") == launch["runtime_sha256"], "runtime receipt hash")
    runtime = host.supervisor.parse(host.supervisor.file_bytes(root / "runtime.json"))
    require(runtime["source"] == RUNTIME_SOURCE, "runtime receipt source")
    host.check_log(root / "child.log")
    # Use the current reviewed checker for replay, never execute retained source
    # supplied by an evidence directory. Bind it to the independent expected hash.
    if checker is None:
        from mjlab_microduck import stance_solved_field_check as checker
    require(host.digest(Path(checker.__file__)) == checker_sha, "reviewed replay checker hash")
    require(canonical(launch["cpu_preflight"]) == canonical(cpu_preflight(checker))
            and launch["physics_step_meaning"] == "Euler-integration-ticks-not-initialization-forward",
            "retained independent CPU preflight and zero-step meaning")
    raw = host.supervisor.file_bytes(root / "inputs.pt", limit=64 * 1024 * 1024)
    values = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    require(type(values) is dict and tuple(values) == checker.FIELDS
            and all(isinstance(v, torch.Tensor) and v.device.type == "cpu"
                    and v.dtype == torch.float32 and v.ndim >= 1 and v.shape[0] == 64
                    for v in values.values()), "complete CPU float32 solved-field snapshot")
    checker.legacy_check(values)
    checker.packed_check(values)
    rows = [host.supervisor.parse(host.supervisor.file_bytes(root / ("block-" + str(i) + ".json"))) for i in range(6)]
    summary = decide(rows)
    summary.update(faults=errors(checker, values), input_bits_unchanged=True, physics_steps=0,
                   policy_inferences=0, optimizer_updates=0, worlds=64, device="cuda:0",
                   input_sha256=sha256(raw).hexdigest())
    require(canonical(host.supervisor.parse(host.supervisor.file_bytes(root / "summary.json"))) == canonical(summary),
            "deterministic timing summary and identical CPU fault replay")
    return summary


def supervise(source, runner_sha, checker_sha, launch_sha):
    root = output_path(source)
    launch = checked(source, runner_sha, checker_sha, launch_sha)
    check_window(launch["deadline_unix"])
    expected = dict(MainPID=str(os.getpid()), RuntimeMaxUSec="3min", KillMode="control-group",
                    ActiveState="active", MemoryMax=str(6 * 1024**3), CPUQuotaPerSecUSec="2s", Nice="10")
    actual = {key: host.read("systemctl", "--user", "show", service_name(source), "-p", key, "--value")
              for key in expected}
    require(actual == expected, "independently bounded user service")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only supervisor")
    require({p.name for p in root.iterdir()} == {"runner.py", "checker.py", "runtime.json", "launch.json"},
            "fresh probe evidence")
    report = dict(protocol=PROTOCOL, launch_sha256=launch_sha, decision="failed", **NO_ADMISSION)
    try:
        with host.supervisor.gpu_lease() as fd:
            report["idle_before"] = host.wait_idle()
            def guard():
                require(time.time() + CLOSEOUT < launch["deadline_unix"], "probe closeout reserve")
                host.check_log(root / "child.log")
                require(checked(source, runner_sha, checker_sha, launch_sha) == launch, "live drift check")
            report["child"] = host.supervisor.supervised_process(
                [str(host.ROOT / ".venv/bin/python"), str(root / "runner.py"), "child",
                 "--tool-source", source, "--runner-sha256", runner_sha,
                 "--checker-sha256", checker_sha, "--launch-sha256", launch_sha, "--lock-fd", str(fd)],
                root / "child.log", cwd=host.ROOT, env=host.supervisor.child_environment(),
                lock_fd=fd, timeout=CHILD_SECONDS, guard=guard)
            checked(source, runner_sha, checker_sha, launch_sha)
            report["idle_after"] = host.wait_idle()
            report["decision"] = "finite-check-probe-complete-not-runtime-equivalence"
            report["files"] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
            replay_payload(root, report, launch_sha, runner_sha, checker_sha,
                           checker=load_checker(root, checker_sha))
    except Exception as exc:
        report.update(decision="failed", error_type=type(exc).__name__, error=str(exc),
                      error_notes=getattr(exc, "__notes__", []))
        raise
    finally:
        report["files"] = {p.name: host.digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        host.supervisor.write_json(root / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "supervise", "child", "verify"))
    parser.add_argument("--tool-source", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--checker-sha256", required=True)
    parser.add_argument("--deadline-unix", type=int)
    parser.add_argument("--launch-sha256")
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args()
    common = (args.tool_source, args.runner_sha256, args.checker_sha256)
    if args.mode == "prepare":
        print(canonical(prepare(*common, args.deadline_unix)))
    elif args.mode == "supervise":
        supervise(*common, args.launch_sha256)
    elif args.mode == "child":
        child(*common, args.launch_sha256, args.lock_fd)
    else:
        checker = None
        if execution.PROFILE["name"] == execution.WSL:
            # The old runtime package intentionally does not contain the new
            # checker. Rebind the reviewed Git tool/host before loading its
            # independently hash-pinned owned archive for Linux replay.
            checked(*common, args.launch_sha256)
            require(args.input_root.resolve() == output_path(args.tool_source).resolve(),
                    "exact live WSL probe evidence root")
            checker = load_checker(args.input_root, args.checker_sha256)
        print(canonical(verify_evidence(args.input_root, args.report_sha256, args.launch_sha256,
                                        args.runner_sha256, args.checker_sha256, checker=checker)))


if __name__ == "__main__":
    main()
