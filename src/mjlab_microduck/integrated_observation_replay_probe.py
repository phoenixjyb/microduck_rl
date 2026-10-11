"""Four repeated same-policy cases; descriptive replay spread, never admission."""

import argparse
import json
import os
import subprocess
from importlib.metadata import version

import torch

from mjlab_microduck import integrated_observation_timing_audit as timing

base, retained, motor, recipe = timing.base, timing.retained, timing.motor, timing.recipe
PROTOCOL = "integrated-angular-observation-replay-probe-v1"
TIMING_SOURCE = "dbb5df4f24c891f2a5b6459336bc36b127988087"
TIMING_SHA = "921d0edda0595878e9f52cd91eefd562161df8a982ea0a6f3e2eff5eff706fa1"
TIMING_LAUNCH_SHA = "522fea006e0915c8512315ca47f9810cd32a50a5c3849f96e429dc6749a21dce"
TIMING_INPUT = base.ROOT / "artifacts/evaluations/obs-timing-dbb5df4f-20261012/result.json"
ORDER = ("baseline", "trace", "trace", "baseline")
CASE = (.3, 0.)
IDENTITY = ("seed", "speed", "yaw", "worlds", "steps", "complete", "terminal_worlds",
            "checkpoint_sha256", "motor_stream")
SCALARS = ("torque_p99", "soft_limit_fraction", "rated_speed_exceed_fraction",
           "mean_abs_mechanical_power_w", "thermal_load_proxy")


def validate_timing(prior):
    base.require(TIMING_INPUT.is_file() and not TIMING_INPUT.is_symlink()
                 and base.sha256(TIMING_INPUT) == TIMING_SHA, "exact completed timing aggregate bytes")
    evidence = timing.read_json(TIMING_INPUT)
    launch_path = TIMING_INPUT.parent / "launch.json"
    base.require(launch_path.is_file() and not launch_path.is_symlink()
                 and base.sha256(launch_path) == TIMING_LAUNCH_SHA,
                 "exact completed timing launch bytes")
    launch = timing.read_json(launch_path)
    base.require(launch["source"] == TIMING_SOURCE and launch["protocol"] == timing.PROTOCOL
                 and launch["runtime"] == timing.RUNTIME, "unchanged timing launch runtime")
    base.require(evidence["source"] == TIMING_SOURCE and evidence["protocol"] == timing.PROTOCOL
                 and evidence["input_source"] == timing.INPUT_SOURCE
                 and evidence["checkpoint_sha256"] == prior["checkpoint_sha256"]
                 and evidence["original_decisions"] == prior["decisions"]
                 and evidence["evaluation_sha256"] == base.sha256(timing.INPUT / "evaluate/result.json")
                 and evidence["diagnostic_case_count"] == len(evidence["cases"]) == 8
                 and all(row["complete"] for row in evidence["cases"])
                 and evidence["decision"] == "diagnostic-only-not-admission"
                 and evidence["policy_acceptance"] is False
                 and evidence["simulator_qualified"] is False
                 and evidence["physical_motion_authorized"] is False,
                 "completed non-admitting timing diagnostic identity")
    for index, row in enumerate(evidence["cases"]):
        name = f"{timing.ARMS[index // 4]}-s{timing.SEED}-c{index % 4}.json"
        path = TIMING_INPUT.parent / name
        base.require(row["raw_json"] == name and path.is_file() and not path.is_symlink()
                     and base.sha256(path) == row["raw_sha256"], "exact complete timing raw case bytes")
    return evidence


def compare(left, right):
    """No tolerance or causal/policy decision is inferred from four replays."""
    base.require(all(key in left and key in right and left[key] == right[key] for key in IDENTITY)
                 and (left["seed"], left["speed"], left["yaw"], left["worlds"], left["steps"])
                 == (timing.SEED, *CASE, 8, 240)
                 and left["complete"] is True and left["terminal_worlds"] == []
                 and base.finite_tree((left, right)), "identical finite replay case identity")
    scalars = {}
    for key in SCALARS:
        base.require(type(left[key]) in (int, float) and type(right[key]) in (int, float),
                     "scalar replay metric")
        scalars[key] = right[key] - left[key]
    worlds = {}
    for key in ("speed_mae", "yaw_mae"):
        a, b = torch.tensor(left[key], dtype=torch.float64), torch.tensor(right[key], dtype=torch.float64)
        base.require(a.shape == b.shape == (8,), "eight-world replay metrics")
        delta = b-a
        worlds[key] = dict(signed_delta_per_world=delta.tolist(),
                           max_abs_delta=float(delta.abs().max()),
                           mean_abs_delta=float(delta.abs().mean()))
    return dict(common_fields_exact=all(right.get(key) == value for key, value in left.items()
                                       if key != "observation_timing_trace"),
                observation_trace_exact_if_both=(left["observation_timing_trace"] == right["observation_timing_trace"]
                    if "observation_timing_trace" in left and "observation_timing_trace" in right else None),
                scalar_delta=scalars, world_metrics=worlds,
                interpretation="descriptive spread only; no causal conclusion or admission",
                policy_acceptance=False, physical_motion_authorized=False)


def comparisons(rows):
    base.require(type(rows) is list and len(rows) == 4
                 and tuple(row["mode"] for row in rows) == ORDER, "complete declared ABBA order")
    cases = [row["case"] for row in rows]
    return {name: compare(cases[a], cases[b]) for name, a, b in (
        ("baseline_repeat", 0, 3), ("trace_repeat", 1, 2),
        ("first_trace_minus_baseline", 0, 1), ("second_trace_minus_baseline", 3, 2))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    args = parser.parse_args(argv)
    head, status = retained._git_identity(args.source)
    timing.audit.validate_invocation(args.source, args.run_id, args.evaluation_sha256,
        head=head, status=status, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    base.require(subprocess.run(("git", "-C", str(base.ROOT), "merge-base", "--is-ancestor",
                                 TIMING_SOURCE, args.source)).returncode == 0,
                 "probe source descends from completed timing source")
    output = base.ROOT / "artifacts/evaluations" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    lease = None
    try:
        lease = retained._validate_gpu_lease()
        runtime = retained._runtime()
        base.require(version("rsl-rl-lib") == "5.0.1", "unchanged policy runner")
        prior = timing.validated_input(args.evaluation_sha256, runtime)
        validate_timing(prior)
        checkpoint = motor.checkpoint_path(timing.INPUT, "control", motor.training_contract(recipe, "control"))
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
            run_id=args.run_id, runtime=runtime, host=retained.shared_host(),
            input_evaluation_sha256=args.evaluation_sha256, timing_sha256=TIMING_SHA,
            checkpoint_sha256=prior["checkpoint_sha256"]["control"], order=list(ORDER),
            seed=timing.SEED, command=list(CASE), angular_diagnostics_in_every_case=True,
            policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False))
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        rows = []
        for index, mode in enumerate(ORDER):
            case = base.evaluate_case(checkpoint, timing.SEED, *CASE,
                lambda: retained._health(output), diagnostics=True,
                angular_diagnostics=True, observation_trace=(mode == "trace"))
            path = output / f"{index}-{mode}.json"
            row = dict(index=index, mode=mode, case=case,
                       policy_acceptance=False, physical_motion_authorized=False)
            base.write_new(path, row)
            base.require(case["complete"] and case["steps"] == 240 and not case["terminal_worlds"],
                         "complete first-episode replay; stop on terminal before next case")
            if mode == "trace":
                timing.capture.summarize(case["observation_timing_trace"])
            rows.append(row)
        result = dict(protocol=PROTOCOL, source=args.source, run_id=args.run_id,
            checkpoint_sha256=prior["checkpoint_sha256"]["control"], timing_sha256=TIMING_SHA,
            original_decisions=prior["decisions"], comparisons=comparisons(rows),
            cases=[dict(index=i, mode=mode, json=f"{i}-{mode}.json",
                        sha256=base.sha256(output / f"{i}-{mode}.json")) for i, mode in enumerate(ORDER)],
            decision="descriptive-repeat-spread-not-admission", policy_acceptance=False,
            simulator_qualified=False, physical_motion_authorized=False)
        base.write_new(output / "result.json", result)
        retained._health(output)
        return 0
    except Exception as exc:
        base.write_new(output / "failure.json", dict(protocol=PROTOCOL, source=args.source,
            type=type(exc).__name__, error=str(exc), policy_acceptance=False, physical_motion_authorized=False))
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
