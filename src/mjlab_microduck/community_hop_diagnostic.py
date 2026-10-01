"""Bounded native-CPU runner for C1-S; default is static preflight ONLY.

Public weights are not inferred unless the operator supplies a separate run
go-ahead through --execute-approved. No training, video or hardware APIs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from . import community_hop_rehearsal as c1
from .community_hop_baseline import BASELINE_DECISION, SHARED_GATES, score_walk_only_trace


BINDING_KEYS = ("runner_source_commit", "runner_source_sha256", "predeclaration_sha256",
                "policies", "asset_closure", "plant", "bam_parameters", "bam_settings")
BASELINE_GATES = (*SHARED_GATES, "stable_final_zero_command_window",
                  "zero_qualified_airborne_episodes_4_9", "runtime_completed_without_failure")
TRANSFER_GATES = (*SHARED_GATES, "bilateral_30mm_flight_10ms", "exactly_one_qualified_hop",
                  "landed_after_qualified_flight", "landing_100ms_impact_window_complete",
                  "settled_after_landing", "runtime_completed_without_failure")


def encode_numbers(value):
    """Losslessly tag a nonfinite failure value without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: encode_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_numbers(item) for item in value]
    return value


def decode_numbers(value):
    if type(value) is dict and set(value) == {"nonfinite_float"}:
        if value["nonfinite_float"] not in ("nan", "inf", "-inf"):
            raise ValueError("invalid nonfinite trace tag")
        return float(value["nonfinite_float"])
    if isinstance(value, dict):
        return {key: decode_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_numbers(item) for item in value]
    return value


def save_json(path: Path, value) -> None:
    """Exclusive, fsynced evidence: never overwrite an earlier record."""
    with path.open("x") as stream:
        json.dump(encode_numbers(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_record(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError("nonfinite JSON record: " + value)
    record = json.loads(path.read_text(), parse_constant=invalid_constant)
    if type(record) is not dict:
        raise ValueError("evidence record must be a JSON object")
    return record


class Journal:
    def __init__(self, path: Path):
        self.stream = path.open("x")

    def write(self, kind: str, value: dict):
        self.stream.write(json.dumps({"kind": kind, "value": encode_numbers(value)},
                                     sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def read_journal(path: Path) -> tuple[dict, dict]:
    """Recover only fully written records; a killed write cannot be success."""
    rows, incomplete, error = [], 0, None
    if path.exists():
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    incomplete += 1
                    break
                try:
                    record = decode_numbers(json.loads(line))
                    if record["kind"] == "sample":
                        if len(rows) >= c1.TOTAL_SAMPLES:
                            raise ValueError("journal exceeds one fixed attempt")
                        rows.append(record["value"])
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                    break
    return {"schema": c1.TRACE_SCHEMA, "samples": rows}, {
        "complete_records_recovered": len(rows), "partial_lines_dropped": incomplete,
        "journal_error": error,
    }


class CpuPolicy:
    def __init__(self, verified_bytes: bytes):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # The validated bytes, NOT a reopened path, enter this session.
        self.session = ort.InferenceSession(verified_bytes, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.session.disable_fallback()
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("CPU-only provider binding failed")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 61]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].name != "actions" or outputs[0].shape != [1, 14]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("runtime signature differs from validated API1")

    def predict(self, observation: list[float]) -> list[float]:
        import numpy as np

        obs = np.asarray(observation, dtype=np.float32).reshape(1, 61)
        if not np.isfinite(obs).all():
            raise ValueError("float32 observation is nonfinite")
        output = self.session.run(["actions"], {"obs": obs})[0]
        if output.dtype != np.float32 or output.shape != (1, 14) or not np.isfinite(output).all():
            raise ValueError("policy output must be finite float32 [1,14]")
        return output[0].tolist()


def policy_kind(case_id: str, t_s: float) -> str:
    if case_id not in ("walk-only-control", "walk-hop-transition"):
        raise ValueError("unknown predeclared case")
    return "hop" if case_id == "walk-hop-transition" and c1.phase_at(t_s) == "hop" else "walk"


def first_fatal(row: dict, index: int, preceding: list[dict]) -> str | None:
    if not c1._validate_sample(row, index):
        return "nonfinite"
    if row["reset_event"]:
        return "reset-event"
    if row["body_contact"]:
        return "non-foot-body-contact"
    if max(abs(x) for x in row["base_roll_pitch_yaw_rad"][:2]) >= math.radians(60):
        return "fall-tilt"
    if index == 800 and not c1._entry_ready(preceding):
        return "unstable-hop-entry"
    return None


def run_attempt(plant, policies: dict, case_id: str, emit) -> dict:
    """One attempt, no reset or retry. Injectable fixtures are NOT public trials.

    The owning parent enforces a hard wall limit even if inference/physics hangs.
    At an unstable entry the hop policy is not invoked and physics is not stepped.
    """
    delay = c1.OneStepActionDelay()
    velocity_lag = [0.] * 14
    raw = applied = [0.] * 14
    rows, inferences, steps = [], 0, 0
    for index in range(c1.TOTAL_SAMPLES):
        t_s = index * c1.PHYSICS_DT_S
        terminal = index == c1.TOTAL_SAMPLES - 1
        plant.prepare()
        unstable_entry = index == 800 and not c1._entry_ready(rows)
        unsafe = plant.unsafe_for_inference()
        if index % 4 == 0 and not terminal and not unstable_entry and not unsafe:
            state = plant.observation_state()
            live_velocity = [x + y for x, y in zip(state["servo_velocity"], state["backlash_velocity"])]
            state["servo_velocity"], state["backlash_velocity"] = velocity_lag, [0.] * 14
            observation = c1.pack_observation(**state, previous_raw_action=delay.previous_action,
                                             command=c1.command_at(t_s))
            kind = policy_kind(case_id, t_s)
            emit("inference-intent", {"t_s": t_s, "policy": kind, "obs": observation})
            raw = policies[kind].predict(observation)
            applied = delay.submit(raw)
            velocity_lag = live_velocity
            inferences += 1
            emit("control", {"t_s": t_s, "policy": kind, "obs": observation,
                             "action_raw": raw, "action_applied": applied})
        row = {"t_s": t_s, "command": c1.command_at(t_s),
               "action_raw": raw.copy(), "action_applied": applied.copy(),
               **plant.advance_and_measure(applied, terminal=terminal or unstable_entry or unsafe)}
        emit("sample", row)
        fatal = first_fatal(row, index, rows)
        if not terminal and not unstable_entry and not unsafe:
            steps += 1
        if fatal:
            return {"first_runtime_failure": {"sample": index, "t_s": t_s, "reason": fatal},
                    "policy_inferences": inferences, "physics_steps": steps}
        rows.append(row)
    return {"first_runtime_failure": None, "policy_inferences": inferences, "physics_steps": steps}


def bound_child(command: list[str], directory: Path, timeout_s: float) -> dict:
    """Own one process group; kill only it on timeout, retain fsynced prefix."""
    if timeout_s <= 0:
        return {"timed_out": True, "returncode": None, "wall_s": 0., "launched": False}
    start = time.monotonic()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with (directory / "worker.stdout.log").open("xb") as stdout, (directory / "worker.stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                 start_new_session=True)
        timed_out = False
        try:
            child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        except BaseException:
            # Ctrl-C/owner interruption must not orphan our policy/physics child.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            raise
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    return {"timed_out": timed_out, "returncode": child.returncode,
            "wall_s": time.monotonic() - start, "launched": True}


def identity(receipt: dict, case_id: str) -> dict:
    return {**{key: receipt[key] for key in (
        "experiment_id", "measurement_protocol", "measurement_scorer_sha256",
        "predeclaration_sha256", "runner_source_commit", "runner_source_sha256")}, "case_id": case_id}


def finish_case(directory: Path, case_id: str, receipt: dict, child: dict) -> dict:
    trace, recovery = read_journal(directory / "journal.jsonl")
    save_json(directory / "trace.json", trace)
    def score(samples):
        if samples:
            candidate = {"schema": c1.TRACE_SCHEMA, "samples": samples}
            return score_walk_only_trace(candidate) if case_id == "walk-only-control" else c1.score_trace(candidate)
        return {"decision": "empty-attempt-rejected", "gates": {"complete_first_attempt": False},
                "metrics": {}, "first_failure": None, "scored_samples": 0, "ignored_samples": 0}
    validation_error = None
    try:
        measured = score(trace["samples"])
    except (ValueError, TypeError, KeyError) as exc:
        prefix = []
        for index, sample in enumerate(trace["samples"]):
            try:
                c1._validate_sample(sample, index)
            except (ValueError, TypeError, KeyError):
                break
            prefix.append(sample)
        measured = score(prefix)
        validation_error = {"sample": len(prefix), "reason": "invalid-trace-sample", "error": str(exc)}
        measured["first_failure"] = validation_error
        measured["gates"]["trace_schema_valid"] = False
    worker_path = directory / "worker-result.json"
    worker_error = None
    try:
        worker = read_record(worker_path) if worker_path.exists() else {}
    except (ValueError, OSError) as exc:
        worker, worker_error = {}, str(exc)
    attempt_path = directory / "preflight.json"
    attempt_receipt, binding_error = None, None
    try:
        attempt_receipt = read_record(attempt_path)
        if any(attempt_receipt[key] != receipt[key] for key in BINDING_KEYS):
            binding_error = "attempt binding differs from campaign setup"
    except (ValueError, KeyError, OSError) as exc:
        binding_error = "attempt binding incomplete: " + str(exc)
    runtime_ok = (child["returncode"] == 0 and not child["timed_out"]
                  and not validation_error and not binding_error and not worker_error
                  and recovery["journal_error"] is None and recovery["partial_lines_dropped"] == 0
                  and worker.get("first_runtime_failure") is None
                  and worker.get("policy_inferences") == 600 and worker.get("physics_steps") == 2400)
    measured["gates"]["runtime_completed_without_failure"] = runtime_ok
    if not runtime_ok:
        measured["decision"] = "runtime-incomplete-or-failed-rejected"
        if measured.get("first_failure") is None:
            measured["first_failure"] = worker.get("first_runtime_failure") or {
                "sample": len(trace["samples"]), "reason": "wall-timeout" if child["timed_out"] else
                "runtime-or-binding-failure", "error": binding_error or worker_error or recovery["journal_error"]}
    report_identity = receipt
    if attempt_receipt and all(key in attempt_receipt for key in identity(receipt, case_id) if key != "case_id"):
        report_identity = attempt_receipt
    result = {**measured, **identity(report_identity, case_id), "child": child, "recovery": recovery,
              "runtime": worker, "validation_error": validation_error, "binding_error": binding_error,
              "setup_runner_source_commit": receipt["runner_source_commit"],
              "trace_number_encoding": "nonfinite-float-tags-v1",
              **{key: False for key in (
                  "behavioral_acceptance", "training_authorized", "transition_authorized",
                  "physical_motion_authorized", "trace_authenticated", "plant_runtime_binding_verified",
                  "thermal_model_verified", "impact_limit_calibrated", "multi_seed_promotion",
                  "author_entry_replication", "author_training_plant_binding")}}
    save_json(directory / "report.json", result)
    return result


def sequential_decision(control: dict | None, transfer: dict | None) -> str:
    def passed(record: dict, decision: str, required: tuple) -> bool:
        gates = record.get("gates")
        return (record.get("decision") == decision and type(gates) is dict
                and set(gates) == set(required) and all(value is True for value in gates.values()))
    if control is None:
        return "preflight-only-no-policy-execution"
    if not passed(control, BASELINE_DECISION, BASELINE_GATES):
        return "walk-only-rejected-transfer-not-run"
    if transfer is None:
        return "walk-only-passed-transfer-not-run"
    if not passed(transfer, "declared-trace-indicators-pass", TRANSFER_GATES):
        return "substituted-entry-transfer-rejected"
    return "nominal-substituted-entry-diagnostic-candidate-only"


def artifact_manifest(directory: Path) -> dict:
    records = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"schema": "community-hop-c1s-artifact-manifest-v1", "artifacts": records}


def worker(root: Path, output: Path, case_id: str, *, execute_approved: bool):
    from .community_hop_surrogate import preflight

    if case_id != "preflight" and not execute_approved:
        raise ValueError("separate policy execution go-ahead required")
    plan, declaration, payloads, plant, receipt = preflight(root, clean_required=execute_approved)
    save_json(output / "preflight.json", receipt)
    with (output / "compiled-plant.mjb").open("xb") as stream:
        stream.write(plant.binary_model())
        stream.flush()
        os.fsync(stream.fileno())
    with (output / "predeclaration.json").open("xb") as stream:
        stream.write(declaration)
        stream.flush()
        os.fsync(stream.fileno())
    if case_id == "preflight":
        return
    setup = read_record(output.parent / "preflight" / "preflight.json")
    for key in BINDING_KEYS:
        if receipt[key] != setup[key]:
            raise ValueError("case binding drifted from campaign setup: " + key)
    # Fresh independent case; no retries. Only this child owns its simulator.
    if case_id == "walk-hop-transition":
        control = read_record(output.parent / "walk-only-control" / "report.json")
        if sequential_decision(control, None) != "walk-only-passed-transfer-not-run":
            raise ValueError("walker-only baseline did not pass; transfer blocked")
        if any(control.get(key) != receipt[key] for key in identity(receipt, "walk-only-control") if key != "case_id"):
            raise ValueError("walker-only report identity differs from this campaign")
    policies = {"walk": CpuPolicy(payloads["walk"])}
    if case_id == "walk-hop-transition":
        policies["hop"] = CpuPolicy(payloads["hop"])
    journal = Journal(output / "journal.jsonl")
    try:
        result = run_attempt(plant, policies, case_id, journal.write)
        result.update(provider="CPUExecutionProvider", intra_threads=1, inter_threads=1,
                      first_runtime_error=None)
    except Exception as exc:
        journal.write("runtime-error", {"type": type(exc).__name__, "error": str(exc)})
        result = {"first_runtime_failure": {"reason": "runtime-error"},
                  "first_runtime_error": {"type": type(exc).__name__, "error": str(exc)}}
    finally:
        journal.close()
    save_json(output / "worker-result.json", result)


def campaign(root: Path, *, execute_approved: bool = False) -> Path:
    from .community_hop_surrogate import DECLARATION_PATH, load_declaration, source_binding, sha256

    plan, declaration = load_declaration(root)
    sources = source_binding(root, clean_required=False)
    campaign_start = time.monotonic()
    parent = root / plan["evidence"]["artifact_root"]
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    save_json(output / "request.json", {"execute_approved_by_operator": execute_approved,
                                        "declaration_path": DECLARATION_PATH,
                                        "authority_is_not_inferred_from_predeclaration": True})
    def stage(case_id: str, cap: float):
        directory = output / case_id
        directory.mkdir(exist_ok=False)
        command = [sys.executable, "-m", "mjlab_microduck.community_hop_diagnostic",
                   "--repo-root", str(root), "--_worker", case_id, "--_output", str(directory)]
        if execute_approved:
            command.append("--execute-approved")
        remaining = plan["budget"]["max_campaign_wall_s"] - (time.monotonic() - campaign_start)
        outcome = bound_child(command, directory, min(cap, remaining))
        save_json(directory / "child-exit.json", outcome)
        return directory, outcome
    setup, setup_child = stage("preflight", plan["budget"]["max_setup_wall_s"])
    receipt_path = setup / "preflight.json"
    control = transfer = None
    if setup_child["returncode"] != 0 or setup_child["timed_out"] or not receipt_path.exists():
        final = {"experiment_id": plan["experiment_id"], "case_id": "campaign",
                 "measurement_protocol": c1.PROTOCOL, "decision": "preflight-failed-no-policy-execution",
                 "predeclaration_sha256": sha256(declaration), **sources,
                 "measurement_scorer_sha256": plan["measurement"]["scorer_sha256"],
                 "setup": setup_child, **plan["not_claimed"]}
    else:
        receipt = read_record(receipt_path)
        if execute_approved:
            directory, child = stage("walk-only-control", plan["budget"]["max_wall_s_per_attempt"])
            control = finish_case(directory, "walk-only-control", receipt, child)
            if sequential_decision(control, None) == "walk-only-passed-transfer-not-run":
                directory, child = stage("walk-hop-transition", plan["budget"]["max_wall_s_per_attempt"])
                transfer = finish_case(directory, "walk-hop-transition", receipt, child)
        final = {**identity(receipt, "campaign"), "decision": sequential_decision(control, transfer),
                 "policy_execution_requested": execute_approved,
                 "case_decisions": {case["case_id"]: case["decision"] for case in (control, transfer) if case},
                 **plan["not_claimed"]}
    final["wall_s"] = time.monotonic() - campaign_start
    if final["wall_s"] > plan["budget"]["max_campaign_wall_s"]:
        final["decision"] = "campaign-wall-budget-exceeded-rejected"
    save_json(output / "decision.json", final)
    save_json(output / "manifest.json", artifact_manifest(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--execute-approved", action="store_true",
                        help="operator asserts a separate go-ahead for the fixed two-case CPU probe")
    parser.add_argument("--_worker", choices=("preflight", "walk-only-control", "walk-hop-transition"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if not args._output:
            parser.error("internal child output required")
        worker(args.repo_root.resolve(), args._output, args._worker, execute_approved=args.execute_approved)
    else:
        output = campaign(args.repo_root.resolve(), execute_approved=args.execute_approved)
        print(json.dumps({"artifacts": str(output), "decision": json.loads((output / "decision.json").read_text())["decision"]}))


if __name__ == "__main__":
    main()
