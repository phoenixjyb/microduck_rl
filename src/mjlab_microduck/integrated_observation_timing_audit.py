"""Bounded cached-observation timing diagnostic; no policy or physical admission."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from mjlab_microduck import integrated_body_yaw_trace_audit as retained
from mjlab_microduck import integrated_gait_yaw_audit as audit
from mjlab_microduck import integrated_low_yaw_rehearsal_refinement as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import observation_timing_capture as capture

PROTOCOL = "integrated-angular-observation-timing-diagnostic-v1"
INPUT_SOURCE = "aee0a31e394ad6c81e614f0d2dda4863439b47c0"
INPUT = base.ROOT / "artifacts/training/low-yaw-aee0a31e-20261012"
ARMS = recipe.ARMS
SEED = 839
RUNTIME = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
           "mujoco-warp": "3.8.1", "mjlab": "1.3.0", "better-actuator-models": "1.0.1"}



def read_json(path):
    path = Path(path)
    base.require(path.is_file() and not path.is_symlink(), "regular retained JSON file")
    value = json.loads(path.read_text())
    base.require(type(value) is dict, "retained JSON object")
    return value


def validated_input(evaluation_sha256, runtime):
    """Authenticate the complete original matrix and final weights, not a cherry pick."""
    base.require(runtime == RUNTIME, "unchanged diagnostic runtime pins")
    evaluation = INPUT / "evaluate/result.json"
    base.require(evaluation.is_file() and not evaluation.is_symlink(), "regular evaluation aggregate")
    retained.validate_evaluation_bytes(evaluation.read_bytes(), evaluation_sha256)
    prior = read_json(evaluation)
    launch = read_json(INPUT / "evaluate/launch.json")
    base.require(prior["source"] == launch["source"] == INPUT_SOURCE
                 and prior["protocol"] == launch["protocol"] == recipe.PROTOCOL
                 and launch["mode"] == "evaluate" and launch["runtime"] == runtime
                 and launch["parent_sha256"] == recipe.PARENT_SHA
                 and prior["selection"] == motor.evaluation_plan(recipe)["selection"]
                 and prior.get("final_iterations") == {arm: 499 for arm in ARMS},
                 "exact final499 evaluation source/protocol/runtime/parent")
    base.require(prior["policy_acceptance"] is False and prior["simulator_qualified"] is False
                 and prior["physical_motion_authorized"] is False
                 and set(prior["cases"]) == set(prior["decisions"])
                 == set(prior["checkpoint_sha256"]) == set(ARMS),
                 "complete two-arm non-admitting aggregate")
    for arm in ARMS:
        contract = motor.training_contract(recipe, arm)
        checkpoint = motor.checkpoint_path(INPUT, arm, contract)
        base.require(checkpoint.is_file() and not checkpoint.is_symlink(), "regular final499 checkpoint")
        trained = read_json(INPUT / arm / "result.json")
        started = read_json(INPUT / arm / "launch.json")
        motor.validate_training_receipt(trained, recipe, INPUT_SOURCE, contract, checkpoint)
        base.require(Path(trained["checkpoint"]).resolve() == checkpoint.resolve()
                     and started["source"] == INPUT_SOURCE and started["protocol"] == recipe.PROTOCOL
                     and started["mode"] == arm and started["runtime"] == runtime
                     and started["parent_sha256"] == recipe.PARENT_SHA
                     and prior["checkpoint_sha256"][arm] == base.sha256(checkpoint),
                     "exact training launch/receipt/checkpoint identity")
        rows = prior["cases"][arm]
        base.require(len(rows) == 12 and recipe.decision(rows) == prior["decisions"][arm],
                     "unchanged complete deterministic held-out decision")
        for index, row in enumerate(rows):
            seed, (speed, yaw) = base.SEEDS[index // 4], base.CASES[index % 4]
            base.require((row["seed"], row["speed"], row["yaw"], row["checkpoint_sha256"])
                         == (seed, speed, yaw, prior["checkpoint_sha256"][arm])
                         and row["worlds"] == 8 and base.finite_tree(row)
                         and read_json(INPUT / "evaluate" / f"{arm}-s{seed}-c{index % 4}.json") == row,
                         "exact ordered original per-case evidence")
    base.require(prior["paired_differences"] == motor.paired_differences(
        prior["cases"][ARMS[0]], prior["cases"][ARMS[1]], label=ARMS[1]+"_minus_control"),
        "unchanged original paired differences")
    return prior


def retain_case(output, arm, index, observed, original, checkpoint_sha256):
    """Keep partial terminal observations before deriving any summary."""
    base.require(arm in ARMS and type(index) is int and index in range(4),
                 "predeclared timing diagnostic case")
    raw = output / f"{arm}-s{SEED}-c{index}.json"
    base.write_new(raw, dict(observed_case=observed, original_case=original,
        policy_acceptance=False, physical_motion_authorized=False))
    speed, yaw = base.CASES[index]
    base.require((observed["seed"], observed["speed"], observed["yaw"], observed["worlds"],
                  observed["checkpoint_sha256"]) == (SEED, speed, yaw, 8, checkpoint_sha256),
                 "exact timing diagnostic checkpoint/command/seed")
    trace = observed["observation_timing_trace"]
    base.require(trace["steps"] == observed["steps"] and trace["checkpoint_sha256"] == checkpoint_sha256,
                 "retained timing trace case identity")
    summary = capture.summarize(trace)
    terminal = trace["samples"][-1]["post"]["terminal"]
    base.require([i for i, value in enumerate(terminal) if value] == observed["terminal_worlds"]
                 and observed["complete"] == (observed["steps"] == 240 and not any(terminal)),
                 "first-episode terminal trace identity")
    baseline_equal = all(observed.get(key) == value for key, value in original.items())
    evidence = dict(arm=arm, seed=SEED, speed=speed, yaw=yaw,
        checkpoint_sha256=checkpoint_sha256, complete=observed["complete"],
        original_fields_replay_exact=baseline_equal, summary=summary,
        raw_json=raw.name, raw_sha256=base.sha256(raw),
        policy_acceptance=False, physical_motion_authorized=False)
    base.write_new(output / f"{arm}-s{SEED}-c{index}-summary.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    args = parser.parse_args(argv)
    head, status = retained._git_identity(args.source)
    audit.validate_invocation(args.source, args.run_id, args.evaluation_sha256,
                              head=head, status=status,
                              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    base.require(subprocess.run(("git", "-C", str(base.ROOT), "merge-base", "--is-ancestor",
                                 INPUT_SOURCE, args.source)).returncode == 0,
                 "diagnostic source descends from retained training source")
    output = base.ROOT / "artifacts/evaluations" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    lease = None
    try:
        lease = retained._validate_gpu_lease()
        runtime = retained._runtime()
        from importlib.metadata import version
        base.require(version("rsl-rl-lib") == "5.0.1", "reviewed policy runner")
        host = retained.shared_host()
        prior = validated_input(args.evaluation_sha256, runtime)
        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
            input_source=INPUT_SOURCE, input_protocol=recipe.PROTOCOL, run_id=args.run_id,
            parent_sha256=recipe.PARENT_SHA, evaluation_sha256=args.evaluation_sha256,
            runtime=runtime, host=host, checkpoint_sha256=prior["checkpoint_sha256"],
            seed=SEED, commands=list(base.CASES), diagnostic_case_count=8,
            policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False))
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        cases = []
        for arm in ARMS:
            checkpoint = motor.checkpoint_path(INPUT, arm, motor.training_contract(recipe, arm))
            for index, (speed, yaw) in enumerate(base.CASES):
                observed = base.evaluate_case(checkpoint, SEED, speed, yaw,
                    lambda: retained._health(output), diagnostics=True,
                    angular_diagnostics=True, observation_trace=True)
                cases.append(retain_case(output, arm, index, observed, prior["cases"][arm][index],
                                         prior["checkpoint_sha256"][arm]))
        base.write_new(output / "result.json", dict(protocol=PROTOCOL, source=args.source,
            input_source=INPUT_SOURCE, run_id=args.run_id, parent_sha256=recipe.PARENT_SHA,
            evaluation_sha256=args.evaluation_sha256, checkpoint_sha256=prior["checkpoint_sha256"],
            original_decisions=prior["decisions"], cases=cases, diagnostic_case_count=8,
            decision="diagnostic-only-not-admission", policy_acceptance=False,
            simulator_qualified=False, physical_motion_authorized=False))
        retained._health(output)
        return 0
    except Exception as exc:
        base.write_new(output / "failure.json", dict(protocol=PROTOCOL, source=args.source,
            type=type(exc).__name__, error=str(exc), policy_acceptance=False,
            physical_motion_authorized=False))
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
