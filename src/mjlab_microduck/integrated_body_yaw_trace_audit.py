"""Bounded matched replay for body-yaw/orientation diagnostics; no learning."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any

import torch

from mjlab_microduck import integrated_decoupled_yaw_precision_refinement as recipe
from mjlab_microduck import integrated_response_audit as replay
from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck.body_yaw_trace import summarize_trace
from mjlab_microduck.exploratory_turn_pilot import LOCK, shared_host


PROTOCOL = "integrated-body-yaw-trace-audit-v1"
INPUT = base.ROOT / "artifacts/training/decoupled-yaw-ecdf2de4-20261012"
INPUT_SOURCE = "ecdf2de4e08f1f67013ada69b8831c0c8bf651ad"
INPUT_PROTOCOL = "integrated-decoupled-yaw-precision-v1"
ARMS = ("control", "precision")
SEED = 839
CASE_COUNT = len(base.CASES)
UPDATES = 1000
TRAINING_WORLDS = 256
TRAINING_SEED = 1027
PARENT_SHA = recipe.PARENT_SHA
EVALUATION_SHA256 = "2e9b919a94275b791f52d64ccf7d3849f51655e124f195731664ff21a886cbad"
FINAL_SHA256 = {
    "control": "04665a21cd131364a188d965e719ccde4bfdd643f745b8fef8b8828982104899",
    "precision": "52845be16d58461df2b751ae7ff25bc497a341e4a6fd07064a88dc1805fb51a9",
}
RUNTIME_PACKAGES = ("torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "better-actuator-models")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_evaluation_bytes(raw: bytes, expected_sha256: str) -> str:
    require(isinstance(raw, bytes), "evaluation aggregate must be read as raw bytes")
    require(isinstance(expected_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "explicit lowercase evaluation SHA256 required")
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == expected_sha256, "retained full evaluation aggregate SHA must equal pinned CLI SHA")
    return actual


def validate_invocation(source: Any, run_id: Any, evaluation_sha256: Any, *,
                        head: str, status: str, cuda_visible_devices: str | None) -> None:
    require(isinstance(source, str) and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
            "source must be a lowercase 40-character commit")
    require(isinstance(run_id, str) and re.fullmatch(r"[a-z0-9-]{1,64}", run_id) is not None,
            "safe unique run id")
    require(isinstance(evaluation_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", evaluation_sha256) is not None
            and evaluation_sha256 == EVALUATION_SHA256,
            "exact pinned lowercase evaluation SHA256 required")
    require(head == source and not status, "exact clean audit source")
    require(cuda_visible_devices == "0", "literal GPU0 selection")


def _validate_case_rows(prior: dict[str, Any], checkpoints: dict[str, str]) -> dict[str, Any]:
    require(isinstance(prior.get("cases"), dict)
            and isinstance(prior.get("decisions"), dict)
            and isinstance(prior.get("checkpoint_sha256"), dict),
            "retained evaluation aggregate schema")
    recomputed = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        rows = prior["cases"].get(arm)
        require(type(rows) is list and len(rows) == len(base.SEEDS) * CASE_COUNT,
                f"{arm} retained full case matrix")
        require(prior["checkpoint_sha256"].get(arm) == checkpoints[arm],
                f"{arm} aggregate final checkpoint identity")
        for index, row in enumerate(rows):
            seed, case = divmod(index, CASE_COUNT)
            expected_seed = base.SEEDS[seed]
            expected_speed, expected_yaw = base.CASES[case]
            require((row.get("seed"), row.get("speed"), row.get("yaw"))
                    == (expected_seed, expected_speed, expected_yaw),
                    f"{arm} retained case ordering/command identity at index {index}")
            require(row.get("checkpoint_sha256") == checkpoints[arm],
                    f"{arm} retained case checkpoint identity at index {index}")
        recomputed[arm] = recipe.decision(rows)
        require(recomputed[arm] == prior["decisions"].get(arm),
                f"{arm} original deterministic decision changed")
        selected[arm] = rows[:CASE_COUNT]
        require(all(row["seed"] == SEED for row in selected[arm]),
                f"{arm} diagnostic seed839 cases must be first")
    return {"decisions": recomputed, "cases": selected}


def validate_input_bundle(prior: Any, evaluation_sha256: str, *,
                          source: str, runtime: dict[str, str],
                          arm_records: dict[str, dict[str, Any]],
                          evaluation_launch: dict[str, Any]) -> dict[str, Any]:
    """Validate receipt/aggregate/runtime/checkpoint identity without running sim."""
    require(isinstance(evaluation_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", evaluation_sha256) is not None
            and evaluation_sha256 == EVALUATION_SHA256,
            "exact pinned evaluation aggregate SHA256")
    require(source == INPUT_SOURCE, "pinned original training source")
    require(isinstance(prior, dict) and prior.get("source") == INPUT_SOURCE
            and prior.get("protocol") == INPUT_PROTOCOL,
            "exact retained evaluation source/protocol")
    require(prior.get("selection") == "both final999 only; unchanged held-out protocol",
            "evaluation must be final999-only")
    require(evaluation_launch.get("source") == INPUT_SOURCE
            and evaluation_launch.get("protocol") == INPUT_PROTOCOL
            and evaluation_launch.get("mode") == "evaluate"
            and evaluation_launch.get("runtime") == runtime,
            "same-source evaluation launch/runtime")
    require(set(arm_records) == set(ARMS), "exact control/precision arm receipts")
    checkpoints: dict[str, str] = {}
    for arm in ARMS:
        record = arm_records[arm]
        launch = record.get("launch")
        trained = record.get("result")
        actual_sha = record.get("actual_checkpoint_sha256")
        require(isinstance(launch, dict) and isinstance(trained, dict),
                f"{arm} launch/result receipts")
        require(launch.get("source") == INPUT_SOURCE and launch.get("protocol") == INPUT_PROTOCOL
                and launch.get("mode") == arm and launch.get("runtime") == runtime,
                f"{arm} exact source/runtime launch")
        require(launch.get("parent_sha256") == PARENT_SHA,
                f"{arm} exact inherited parent")
        require(trained.get("source") == INPUT_SOURCE
                and trained.get("protocol") == INPUT_PROTOCOL
                and trained.get("mode") == arm
                and trained.get("status") == "training-complete-not-accepted"
                and trained.get("updates") == UPDATES
                and trained.get("worlds") == TRAINING_WORLDS
                and trained.get("seed") == TRAINING_SEED
                and trained.get("warm_start", {}).get("parent_sha256") == PARENT_SHA,
                f"{arm} complete 1000-update final training receipt")
        require(isinstance(trained.get("checkpoint"), str)
                and Path(trained["checkpoint"]).name == "model_999.pt",
                f"{arm} selected checkpoint must be final999")
        digest = trained.get("checkpoint_sha256")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and digest == FINAL_SHA256[arm]
                and digest == actual_sha == prior.get("checkpoint_sha256", {}).get(arm),
                f"{arm} receipt/aggregate/actual checkpoint SHA identity")
        checkpoints[arm] = digest
    require(prior.get("evaluation_sha256", evaluation_sha256) == evaluation_sha256,
            "pinned evaluation aggregate SHA identity")
    return _validate_case_rows(prior, checkpoints)


def case_jobs(prior: dict[str, Any]):
    validated = _validate_case_rows(prior, prior["checkpoint_sha256"])
    for arm in ARMS:
        for index, row in enumerate(validated["cases"][arm]):
            yield arm, index, row


def retain_case(output: Path, arm: str, index: int, original: dict[str, Any],
                observed: dict[str, Any]) -> dict[str, Any]:
    """Write raw evidence first; then audit, compare, and write its hash-bound summary."""
    require(arm in ARMS and type(index) is int and 0 <= index < CASE_COUNT,
            "declared diagnostic case identity")
    require(isinstance(observed, dict) and "body_yaw_trace" in observed,
            "evaluator returned the requested yaw trace")
    trace = observed["body_yaw_trace"]
    case = {key: value for key, value in observed.items() if key != "body_yaw_trace"}
    raw_path = output / f"{arm}-s{SEED}-c{index}.json"
    base.write_new(raw_path, {"case": case, "trace": trace})
    raw_digest = sha256(raw_path)

    expected_speed, expected_yaw = base.CASES[index]
    for label, row in (("original", original), ("observed", case)):
        require((row.get("seed"), row.get("speed"), row.get("yaw"),
                 row.get("checkpoint_sha256"))
                == (SEED, expected_speed, expected_yaw, FINAL_SHA256[arm]),
                f"{label} declared arm/checkpoint/command identity")
    summary = summarize_trace(case, trace)
    differences = replay.replay_differences(case, original)
    summary_path = output / f"{arm}-s{SEED}-c{index}-summary.json"
    base.write_new(summary_path, dict(protocol=PROTOCOL, input_case_trace_sha256=raw_digest,
        checkpoint_sha256=case["checkpoint_sha256"], arm=arm, seed=SEED, case_index=index,
        command=dict(speed=case["speed"], yaw=case["yaw"]), trace_summary=summary,
        original_case_differences=differences, replay_verified=not differences,
        policy_acceptance=False, physical_motion_authorized=False))
    return dict(arm=arm, seed=SEED, case_index=index, case_file=raw_path.name,
        case_trace_sha256=raw_digest, summary_file=summary_path.name,
        summary_sha256=sha256(summary_path), replay_verified=not differences,
        differences=differences)


def _git_identity(source: str) -> tuple[str, str]:
    def git(*argv: str) -> str:
        return subprocess.check_output(("git", "-C", str(base.ROOT), *argv), text=True).strip()
    return git("rev-parse", "HEAD"), git("status", "--porcelain")


def _runtime() -> dict[str, str]:
    return {package: metadata.version(package) for package in RUNTIME_PACKAGES}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def _health(output: Path) -> dict[str, Any]:
    sample = dict(utc=time.time(), **shared_host())
    with (output / "health.jsonl").open("a") as log:
        log.write(json.dumps(sample, allow_nan=False) + "\n")
        log.flush()
        os.fsync(log.fileno())
    return sample


def _validate_gpu_lease():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "literal GPU0 selection")
    require(LOCK.is_file() and not LOCK.is_symlink(), "existing regular GPU lease")
    lease = LOCK.open("r+")
    try:
        info = os.fstat(lease.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                "owned regular GPU lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lease
    except Exception:
        lease.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    args = parser.parse_args(argv)
    # Validate path-bearing input before making the unique local output directory.
    validate_invocation(args.source, args.run_id, args.evaluation_sha256,
        head=args.source, status="", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    output = base.ROOT / "artifacts/evaluations" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    lease = None
    try:
        head, status = _git_identity(args.source)
        validate_invocation(args.source, args.run_id, args.evaluation_sha256,
            head=head, status=status, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        lease = _validate_gpu_lease()
        host = shared_host()
        runtime = _runtime()

        evaluation_path = INPUT / "evaluate/result.json"
        evaluation_bytes = evaluation_path.read_bytes()
        actual_evaluation_sha = validate_evaluation_bytes(evaluation_bytes, args.evaluation_sha256)
        prior = json.loads(evaluation_bytes)
        require(isinstance(prior, dict), "evaluation aggregate JSON object")
        evaluation_launch = _load_json(INPUT / "evaluate/launch.json")
        arm_records = {}
        for arm in ARMS:
            arm_dir = INPUT / arm
            launch = _load_json(arm_dir / "launch.json")
            trained = _load_json(arm_dir / "result.json")
            checkpoint = arm_dir / "model_999.pt"
            require(checkpoint.is_file() and not checkpoint.is_symlink(),
                    f"{arm} final999 checkpoint must be a regular nonsymlink file")
            arm_records[arm] = dict(launch=launch, result=trained,
                                    actual_checkpoint_sha256=sha256(checkpoint))
        validated = validate_input_bundle(prior, actual_evaluation_sha, source=INPUT_SOURCE,
            runtime=runtime, arm_records=arm_records, evaluation_launch=evaluation_launch)

        base.write_new(output / "launch.json", dict(protocol=PROTOCOL, source=args.source,
            run_id=args.run_id, input_source=INPUT_SOURCE, input_protocol=INPUT_PROTOCOL,
            evaluation_sha256=actual_evaluation_sha,
            checkpoint_sha256={arm: arm_records[arm]["actual_checkpoint_sha256"] for arm in ARMS},
            parent_sha256=PARENT_SHA, runtime=runtime, host=host, arms=list(ARMS),
            selected_seed=SEED, selected_commands=[list(case) for case in base.CASES],
            selected_checkpoint="model_999.pt", diagnostic_case_count=2 * CASE_COUNT,
            policy_acceptance=False, physical_motion_authorized=False))
        _health(output)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.20, device=0)
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        summaries = []
        for arm in ARMS:
            checkpoint = INPUT / arm / "model_999.pt"
            old_rows = validated["cases"][arm]
            for index, old in enumerate(old_rows):
                row = base.evaluate_case(checkpoint, SEED, old["speed"], old["yaw"],
                    lambda: _health(output), diagnostics=True, yaw_trace=True)
                # Preserve per-case time-series bytes before any audit can fail.
                summary = retain_case(output, arm, index, old, row)
                summaries.append(summary)
        replay_verified = all(item["replay_verified"] for item in summaries)
        result = dict(protocol=PROTOCOL, source=args.source, input_source=INPUT_SOURCE,
            input_protocol=INPUT_PROTOCOL, evaluation_sha256=actual_evaluation_sha,
            checkpoint_sha256={arm: arm_records[arm]["actual_checkpoint_sha256"] for arm in ARMS},
            original_decisions=prior["decisions"], recomputed_decisions=validated["decisions"],
            cases=summaries, diagnostic_case_count=len(summaries),
            replay_verified=replay_verified,
            decision="diagnostic-only-not-admission", policy_acceptance=False,
            simulator_qualified=False, physical_motion_authorized=False)
        base.write_new(output / "result.json", result)
        _health(output)
        return 0
    except Exception as exc:
        base.write_new(output / "failure.json", dict(protocol=PROTOCOL, type=type(exc).__name__,
            error=str(exc), source=args.source, policy_acceptance=False,
            physical_motion_authorized=False))
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
