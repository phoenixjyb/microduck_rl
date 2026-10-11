"""Bounded retained-checkpoint gait/head/body-yaw audit; no policy admission."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import torch

from mjlab_microduck import integrated_body_yaw_trace_audit as prior_audit
from mjlab_microduck import integrated_motor_refinement as motor
from mjlab_microduck import integrated_training_budget_refinement as recipe
from mjlab_microduck import integrated_response_audit as replay
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import body_yaw_trace, gait_yaw_trace

PROTOCOL = "integrated-gait-yaw-diagnostic-v1"
INPUT = base.ROOT / "artifacts/training/training-duration-7a9c37da-20261012"
INPUT_SOURCE = "7a9c37da8a11695180572c982ebcdf1358d84d30"
ARMS = ("control", "budget")
SEED = 839
CASE_COUNT = len(base.CASES)
RUNTIME_PACKAGES = prior_audit.RUNTIME_PACKAGES
EVALUATION_SELECTION = "declared final checkpoints only; unchanged held-out protocol"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_invocation(source: Any, run_id: Any, evaluation_sha256: Any, *,
                        head: str, status: str, cuda_visible_devices: str | None) -> None:
    require(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
            "source must be a lowercase 40-character commit")
    require(type(run_id) is str and re.fullmatch(r"[a-z0-9-]{1,64}", run_id) is not None,
            "safe unique run id")
    require(type(evaluation_sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", evaluation_sha256) is not None,
            "explicit lowercase evaluation SHA256 required")
    require(head == source and not status, "exact clean audit source")
    require(cuda_visible_devices == "0", "literal GPU0 selection")


def _runtime_matches(actual: Any, expected: dict[str, str]) -> bool:
    return type(actual) is dict and actual == expected


def validate_input_bundle(prior: Any, evaluation_sha256: str, *, source: str,
                          runtime: dict[str, str], arm_records: dict[str, dict[str, Any]],
                          evaluation_launch: dict[str, Any]) -> dict[str, Any]:
    """Validate the full 24-case final-checkpoint bundle before diagnostics."""
    require(type(evaluation_sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", evaluation_sha256) is not None,
            "explicit aggregate SHA256")
    require(source == INPUT_SOURCE, "pinned duration training source")
    require(type(prior) is dict and prior.get("source") == INPUT_SOURCE
            and prior.get("protocol") == recipe.PROTOCOL,
            "exact retained duration evaluation source/protocol")
    require(prior.get("selection") == EVALUATION_SELECTION,
            "evaluation must select declared final checkpoints only")
    expected_iterations = {arm: motor.training_contract(recipe, arm)["final_iteration"] for arm in ARMS}
    require(type(prior.get("final_iterations")) is dict
            and prior["final_iterations"] == expected_iterations,
            "exact declared final iterations")
    require(type(evaluation_launch) is dict
            and evaluation_launch.get("source") == INPUT_SOURCE
            and evaluation_launch.get("protocol") == recipe.PROTOCOL
            and evaluation_launch.get("mode") == "evaluate"
            and evaluation_launch.get("parent_sha256") == recipe.PARENT_SHA
            and _runtime_matches(evaluation_launch.get("runtime"), runtime),
            "same-source final evaluation launch/runtime/parent")
    require(type(arm_records) is dict and set(arm_records) == set(ARMS),
            "exact control/budget training receipts")
    require(type(prior.get("cases")) is dict and set(prior["cases"]) == set(ARMS)
            and type(prior.get("decisions")) is dict and set(prior["decisions"]) == set(ARMS)
            and type(prior.get("checkpoint_sha256")) is dict
            and set(prior["checkpoint_sha256"]) == set(ARMS),
            "retained full evaluation aggregate schema")
    if "evaluation_sha256" in prior:
        require(prior["evaluation_sha256"] == evaluation_sha256,
                "aggregate embedded SHA identity")

    hashes: dict[str, str] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    recomputed: dict[str, Any] = {}
    for arm in ARMS:
        contract = motor.training_contract(recipe, arm)
        record = arm_records[arm]
        require(type(record) is dict and type(record.get("launch")) is dict
                and type(record.get("result")) is dict,
                f"{arm} launch/result receipts")
        launch, trained = record["launch"], record["result"]
        require(launch.get("source") == INPUT_SOURCE
                and launch.get("protocol") == recipe.PROTOCOL
                and launch.get("mode") == arm
                and launch.get("parent_sha256") == recipe.PARENT_SHA
                and _runtime_matches(launch.get("runtime"), runtime),
                f"{arm} exact launch source/protocol/mode/parent/runtime")
        checkpoint = INPUT / arm / f"model_{contract['final_iteration']}.pt"
        require(checkpoint.is_file() and not checkpoint.is_symlink(),
                f"{arm} final checkpoint must be regular nonsymlink file")
        motor.validate_training_receipt(trained, recipe, INPUT_SOURCE, contract, checkpoint)
        try:
            receipt_checkpoint = (Path(trained["checkpoint"]).resolve(strict=True)
                                 if type(trained.get("checkpoint")) is str else None)
        except OSError:
            receipt_checkpoint = None
        require(receipt_checkpoint == checkpoint.resolve(strict=True),
                f"{arm} receipt names exact resolved final checkpoint")
        actual_hash = prior_audit.sha256(checkpoint)
        receipt_hash = trained.get("checkpoint_sha256")
        aggregate_hash = prior["checkpoint_sha256"].get(arm)
        supplied_hash = record.get("actual_checkpoint_sha256")
        require(type(actual_hash) is str and re.fullmatch(r"[0-9a-f]{64}", actual_hash) is not None
                and actual_hash == receipt_hash == aggregate_hash == supplied_hash,
                f"{arm} checkpoint bytes/receipt/aggregate identity")
        hashes[arm] = actual_hash

        rows = prior["cases"][arm]
        expected_count = len(base.SEEDS) * CASE_COUNT
        require(type(rows) is list and len(rows) == expected_count,
                f"{arm} retained full {expected_count}-case matrix")
        for index, row in enumerate(rows):
            seed_index, case_index = divmod(index, CASE_COUNT)
            seed = base.SEEDS[seed_index]
            speed, yaw = base.CASES[case_index]
            require(type(row) is dict and (row.get("seed"), row.get("speed"), row.get("yaw"))
                    == (seed, speed, yaw),
                    f"{arm} case ordering/command identity at index {index}")
            require(row.get("checkpoint_sha256") == actual_hash,
                    f"{arm} case checkpoint identity at index {index}")
        recomputed[arm] = recipe.decision(rows)
        require(recomputed[arm] == prior["decisions"].get(arm),
                f"{arm} original deterministic decision changed")
        arm_selected = [row for row in rows if row.get("seed") == SEED]
        require(len(arm_selected) == CASE_COUNT and rows[:CASE_COUNT] == arm_selected
                and all(row.get("seed") == SEED for row in arm_selected),
                f"{arm} declared seed839 cases must be first")
        selected[arm] = arm_selected
    return dict(decisions=recomputed, cases=selected, checkpoint_sha256=hashes,
                final_iterations=expected_iterations)


def case_jobs(validated: dict[str, Any]):
    require(type(validated) is dict and type(validated.get("cases")) is dict
            and set(validated["cases"]) == set(ARMS), "validated fixed arm cases")
    for arm in ARMS:
        rows = validated["cases"][arm]
        require(type(rows) is list and len(rows) == CASE_COUNT, f"{arm} exactly four audit cases")
        for index, (row, command) in enumerate(zip(rows, base.CASES, strict=True)):
            require(type(row) is dict and (row.get("seed"), row.get("speed"), row.get("yaw"))
                    == (SEED, *command), f"{arm} declared audit case {index}")
            yield arm, index, row


def retain_case(output: Path, arm: str, index: int, original: dict[str, Any],
                observed: dict[str, Any], checkpoint_sha256: str) -> dict[str, Any]:
    """Persist all raw observed channels before validating either trace."""
    require(arm in ARMS and type(index) is int and 0 <= index < CASE_COUNT,
            "declared diagnostic case identity")
    require(type(observed) is dict and "body_yaw_trace" in observed
            and "gait_yaw_trace" in observed,
            "evaluator returned both requested traces")
    body_trace, gait_trace = observed["body_yaw_trace"], observed["gait_yaw_trace"]
    case = {key: value for key, value in observed.items()
            if key not in ("body_yaw_trace", "gait_yaw_trace")}
    raw_path = output / f"{arm}-s{SEED}-c{index}.json"
    prior_audit.base.write_new(raw_path, dict(original_case=original, observed_case=case,
                                               body_yaw_trace=body_trace,
                                               gait_yaw_trace=gait_trace))
    raw_digest = prior_audit.sha256(raw_path)

    speed, yaw = base.CASES[index]
    for label, row in (("original", original), ("observed", case)):
        require(type(row) is dict and (row.get("seed"), row.get("speed"), row.get("yaw"),
                                       row.get("checkpoint_sha256"))
                == (SEED, speed, yaw, checkpoint_sha256),
                f"{label} declared arm/checkpoint/command identity")
    require(type(body_trace) is dict and type(gait_trace) is dict
            and all(key in body_trace and key in gait_trace
                    and body_trace[key] == gait_trace[key]
                    for key in ("command", "actor_command", "quat_wxyz", "angular_b", "angular_w")),
            "body and gait traces have identical aligned shared pre-action samples")
    body_summary = body_yaw_trace.summarize_trace(case, body_trace)
    gait_summary = gait_yaw_trace.summarize_trace(case, gait_trace)
    differences = replay.replay_differences(case, original)
    summary_path = output / f"{arm}-s{SEED}-c{index}-summary.json"
    prior_audit.base.write_new(summary_path, dict(
        protocol=PROTOCOL, raw_observation_sha256=raw_digest,
        original_checkpoint_sha256=checkpoint_sha256,
        recomputed_checkpoint_sha256=case["checkpoint_sha256"], arm=arm,
        seed=SEED, case_index=index, command=dict(speed=speed, yaw=yaw),
        body_yaw_summary=body_summary, gait_yaw_summary=gait_summary,
        original_case_differences=differences, replay_verified=not differences,
        policy_acceptance=False, simulator_qualified=False,
        physical_motion_authorized=False))
    return dict(arm=arm, seed=SEED, case_index=index, raw_file=raw_path.name,
                raw_sha256=raw_digest, summary_file=summary_path.name,
                summary_sha256=prior_audit.sha256(summary_path),
                replay_verified=not differences, differences=differences)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(type(value) is dict, f"JSON object required: {path.name}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    args = parser.parse_args(argv)
    validate_invocation(args.source, args.run_id, args.evaluation_sha256,
                        head=args.source, status="",
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    output = base.ROOT / "artifacts/evaluations" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    lease = None
    try:
        head, status = prior_audit._git_identity(args.source)
        validate_invocation(args.source, args.run_id, args.evaluation_sha256,
                            head=head, status=status,
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        lease = prior_audit._validate_gpu_lease()
        host, runtime = prior_audit.shared_host(), prior_audit._runtime()

        evaluation_path = INPUT / "evaluate" / "result.json"
        evaluation_bytes = evaluation_path.read_bytes()
        evaluation_sha256 = prior_audit.validate_evaluation_bytes(
            evaluation_bytes, args.evaluation_sha256)
        prior = json.loads(evaluation_bytes)
        require(type(prior) is dict, "evaluation aggregate JSON object")
        evaluation_launch = _load_json(INPUT / "evaluate" / "launch.json")
        arm_records = {}
        for arm in ARMS:
            arm_dir = INPUT / arm
            contract = motor.training_contract(recipe, arm)
            checkpoint = arm_dir / f"model_{contract['final_iteration']}.pt"
            arm_records[arm] = dict(
                launch=_load_json(arm_dir / "launch.json"),
                result=_load_json(arm_dir / "result.json"),
                actual_checkpoint_sha256=prior_audit.sha256(checkpoint)
                if checkpoint.is_file() and not checkpoint.is_symlink() else None)
        validated = validate_input_bundle(prior, evaluation_sha256, source=INPUT_SOURCE,
                                          runtime=runtime, arm_records=arm_records,
                                          evaluation_launch=evaluation_launch)

        prior_audit.base.write_new(output / "launch.json", dict(
            protocol=PROTOCOL, source=args.source, run_id=args.run_id,
            input_source=INPUT_SOURCE, input_protocol=recipe.PROTOCOL,
            evaluation_sha256=evaluation_sha256,
            checkpoint_sha256=validated["checkpoint_sha256"],
            parent_sha256=recipe.PARENT_SHA, final_iterations=validated["final_iterations"],
            runtime=runtime, host=host, arms=list(ARMS), selected_seed=SEED,
            selected_commands=[list(case) for case in base.CASES],
            selected_checkpoints={arm: f"model_{validated['final_iterations'][arm]}.pt"
                                  for arm in ARMS},
            diagnostic_case_count=len(ARMS) * CASE_COUNT,
            policy_acceptance=False, simulator_qualified=False,
            physical_motion_authorized=False))
        prior_audit._health(output)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()

        cases = []
        for arm, index, old in case_jobs(validated):
            contract = motor.training_contract(recipe, arm)
            checkpoint = INPUT / arm / f"model_{contract['final_iteration']}.pt"
            observed = base.evaluate_case(
                checkpoint, SEED, old["speed"], old["yaw"],
                lambda: prior_audit._health(output), diagnostics=True,
                yaw_trace=True, gait_trace=True)
            # retain_case writes the complete raw observed payload before either
            # body or gait summary validation. Replays remain descriptive only.
            cases.append(retain_case(output, arm, index, old, observed,
                                     validated["checkpoint_sha256"][arm]))
        result = dict(
            protocol=PROTOCOL, source=args.source, input_source=INPUT_SOURCE,
            input_protocol=recipe.PROTOCOL, evaluation_sha256=evaluation_sha256,
            checkpoint_sha256=validated["checkpoint_sha256"],
            final_iterations=validated["final_iterations"],
            original_decisions=prior["decisions"],
            recomputed_decisions=validated["decisions"], cases=cases,
            diagnostic_case_count=len(cases),
            replay_verified=all(item["replay_verified"] for item in cases),
            decision="diagnostic-only-not-admission", policy_acceptance=False,
            simulator_qualified=False, physical_motion_authorized=False)
        prior_audit.base.write_new(output / "result.json", result)
        prior_audit._health(output)
        return 0
    except Exception as exc:
        prior_audit.base.write_new(output / "failure.json", dict(
            protocol=PROTOCOL, type=type(exc).__name__, error=str(exc),
            source=args.source, policy_acceptance=False,
            simulator_qualified=False, physical_motion_authorized=False))
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
