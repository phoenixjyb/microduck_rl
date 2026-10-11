"""Bounded nonzero head-yaw diagnostic; no policy or physical admission."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from mjlab_microduck import integrated_body_yaw_trace_audit as retained
from mjlab_microduck import integrated_gait_yaw_audit as audit
from mjlab_microduck import integrated_entropy_consolidation_refinement as recipe
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import body_yaw_trace, gait_yaw_trace

PROTOCOL = "integrated-head-yaw-command-diagnostic-v1"
INPUT_SOURCE = "f5f112bf43b8e09f274e291249a5f66626b0137f"
INPUT = base.ROOT / "artifacts/training/entropy-f5f112bf-20261012"
ARMS = recipe.ARMS
SEED, SPEED, BODY_YAW = 839, .3, 0.
HEAD_YAWS = (-.2, .2)
RUNTIME = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
           "mujoco-warp": "3.8.1", "mjlab": "1.3.0", "better-actuator-models": "1.0.1"}
VIEW = "named plain-servo qpos minus HOME"


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
                 and "final_iterations" not in prior,
                 "exact final999 evaluation source/protocol/runtime/parent")
    base.require(prior["policy_acceptance"] is False and prior["simulator_qualified"] is False
                 and prior["physical_motion_authorized"] is False
                 and set(prior["cases"]) == set(prior["decisions"])
                 == set(prior["checkpoint_sha256"]) == set(ARMS),
                 "complete two-arm non-admitting aggregate")
    for arm in ARMS:
        contract = motor.training_contract(recipe, arm)
        checkpoint = motor.checkpoint_path(INPUT, arm, contract)
        base.require(checkpoint.is_file() and not checkpoint.is_symlink(), "regular final999 checkpoint")
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
                         and read_json(INPUT / "evaluate" / f"{arm}-s{seed}-c{index % 4}.json") == row,
                         "exact ordered original per-case evidence")
    base.require(prior["paired_differences"] == motor.paired_differences(
        prior["cases"][ARMS[0]], prior["cases"][ARMS[1]], label=ARMS[1]+"_minus_control"),
        "unchanged original paired differences")
    return prior


def head_summary(case, trace, head_yaw):
    """Raw physical HOME-relative errors, not a new acceptance threshold."""
    base.require(case["head_measurement_view"] == VIEW
                 and case["head_yaw_command"] == head_yaw and head_yaw in HEAD_YAWS,
                 "plain-servo head measurement and fixed diagnostic command")
    # Validate every original aligned trace channel before deriving extra errors.
    gait_yaw_trace.summarize_trace(case, trace)
    positions = torch.tensor(trace["head_position_offset"], dtype=torch.float64)
    command = torch.tensor(trace["head_command"], dtype=torch.float64)
    # Live commands are float32; preserve their exact serialized numeric values.
    target = torch.tensor([0., 0., head_yaw, 0.], dtype=torch.float32).double()
    expected = target.expand_as(command)
    base.require(torch.equal(command, expected), "constant requested four-axis head command")
    error = positions[60:] - command[60:]
    return dict(head_joints=list(gait_yaw_trace.HEAD_JOINTS), startup_steps=60,
                measurement_view=VIEW,
                mae_rad_per_world_by_joint=error.abs().mean(0).tolist(),
                signed_error_rad_per_world_by_joint=error.mean(0).tolist(),
                rms_rad_per_world_by_joint=error.square().mean(0).sqrt().tolist(),
                pooled_mae_rad_by_joint=error.abs().mean((0, 1)).tolist(),
                policy_acceptance=False, physical_motion_authorized=False)


def retain_case(output, arm, index, observed, original, checkpoint_sha256):
    """Keep raw and partial samples before deriving descriptive summaries."""
    base.require(arm in ARMS and type(index) is int and index in (0, 1),
                 "predeclared diagnostic case")
    head_yaw = HEAD_YAWS[index]
    raw = output / f"{arm}-s{SEED}-h{index}.json"
    base.write_new(raw, dict(observed_case=observed, zero_head_baseline=original,
                            baseline_source=INPUT_SOURCE,
                            baseline_is_not_rng_identical_replay=True,
                            policy_acceptance=False, physical_motion_authorized=False))
    base.require((observed["seed"], observed["speed"], observed["yaw"], observed["worlds"],
                  observed["checkpoint_sha256"], observed["head_yaw_command"])
                 == (SEED, SPEED, BODY_YAW, 8, checkpoint_sha256, head_yaw),
                 "exact diagnostic checkpoint/command/seed identity")
    base.require((original["seed"], original["speed"], original["yaw"], original["checkpoint_sha256"])
                 == (SEED, SPEED, BODY_YAW, checkpoint_sha256),
                 "retained zero-head baseline identity")
    for key in ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w"):
        base.require(observed["body_yaw_trace"][key] == observed["gait_yaw_trace"][key],
                     "identical aligned shared head-probe trace channels")
    if not observed["complete"]:
        base.require(observed["terminal_worlds"] and 0 < observed["steps"] <= 240,
                     "retained bounded first-episode terminal")
        summary = dict(status="partial-first-episode-terminal", steps=observed["steps"],
                       terminal_worlds=observed["terminal_worlds"], head_tracking=None)
    else:
        summary = dict(status="complete-descriptive-only",
                       head_tracking=head_summary(observed, observed["gait_yaw_trace"], head_yaw),
                       body_yaw=body_yaw_trace.summarize_trace(observed, observed["body_yaw_trace"]))
    path = output / f"{arm}-s{SEED}-h{index}-summary.json"
    summary.update(protocol=PROTOCOL, arm=arm, head_yaw_command=head_yaw,
                   raw_sha256=base.sha256(raw), policy_acceptance=False,
                   simulator_qualified=False, physical_motion_authorized=False)
    base.write_new(path, summary)
    return dict(arm=arm, head_yaw_command=head_yaw, status=summary["status"],
                raw_file=raw.name, raw_sha256=base.sha256(raw),
                summary_file=path.name, summary_sha256=base.sha256(path))


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
            parent_sha256=recipe.PARENT_SHA,
            evaluation_sha256=args.evaluation_sha256, runtime=runtime, host=host,
            checkpoint_sha256=prior["checkpoint_sha256"], seed=SEED,
            speed=SPEED, body_yaw=BODY_YAW, head_yaws=list(HEAD_YAWS), diagnostic_case_count=4,
            policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False))
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        cases = []
        for arm in ARMS:
            checkpoint = motor.checkpoint_path(INPUT, arm, motor.training_contract(recipe, arm))
            old = prior["cases"][arm][1]
            for index, head_yaw in enumerate(HEAD_YAWS):
                observed = base.evaluate_case(checkpoint, SEED, SPEED, BODY_YAW,
                    lambda: retained._health(output), diagnostics=True, yaw_trace=True,
                    gait_trace=True, head_yaw_command=head_yaw)
                cases.append(retain_case(output, arm, index, observed, old,
                                         prior["checkpoint_sha256"][arm]))
        base.write_new(output / "result.json", dict(protocol=PROTOCOL, source=args.source,
            input_source=INPUT_SOURCE, run_id=args.run_id, parent_sha256=recipe.PARENT_SHA,
            evaluation_sha256=args.evaluation_sha256,
            checkpoint_sha256=prior["checkpoint_sha256"], original_decisions=prior["decisions"],
            cases=cases, diagnostic_case_count=4, decision="diagnostic-only-not-admission",
            policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False))
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
