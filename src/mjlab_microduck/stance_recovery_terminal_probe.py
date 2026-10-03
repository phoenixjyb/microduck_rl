"""Capped native first-terminal collection and separate recorded-evidence replay.

The unchanged parent is sampled, never updated. Natural simultaneous timeout
and genuine selective reset are distinct integration gates, not skill admission.
"""

import argparse
import io
import os
import time
from hashlib import sha256

import torch

from mjlab_microduck import stance_recovery_optimizer_probe as previous
from mjlab_microduck import stance_recovery_terminal_trace as trace
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cpu-stochastic-first-terminal-probe-v1"
UPDATE_SOURCE = "0f245126837220297d5bd93d58c021997d3c9fb3"
UPDATE_CLOSEOUT_SHA256 = (
    "0e1154de7c2093a1ee9e9bb0fa4433b28577f448edccbbe6b8f973fcdae233ba"
)
UPDATE_LAUNCH_SHA256 = (
    "40bcc1525edb7a49ff65646ed0d1e8a88810e13d328423b1522a3c574662c613"
)
RUN_SECONDS, CLOSEOUT_SECONDS, MARGIN_SECONDS = 360, 300, 60
LAUNCH_RESERVE = RUN_SECONDS + CLOSEOUT_SECONDS + MARGIN_SECONDS
MEMORY_BYTES = 2 * 1024**3
RAW_LIMIT = 128 * 1024**2
OUTPUT_PREFIX = "stance-wsl-cpu-stochastic-first-terminal-"
RUN_FILES = {
    "checkpoint.pt",
    "launch.json",
    "capture.pt",
    "capture.json",
    "report.json",
}
CLOSED_FILES = RUN_FILES | {"independent-closeout.json"}
base, baseline, window = previous.base, previous.baseline, previous.window


def output_path(source):
    base.files.hex_id(source, 40)
    require(
        source not in (UPDATE_SOURCE, previous.AUDIT_SOURCE, previous.ARTIFACT_SOURCE),
        "distinct new first-terminal protocol source",
    )
    return base.host.ROOT / "artifacts/evaluations" / (OUTPUT_PREFIX + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ("run", "closeout"), "fixed first-terminal service mode")
    return f"microduck-cpu-terminal-{mode}-{source[:12]}.service"


def _properties(source, mode):
    name = service_name(source, mode)
    props = {
        key: base.host.read("systemctl", "--user", "show", name, "-p", key, "--value")
        for key in (
            "MainPID",
            "ActiveState",
            "RuntimeMaxUSec",
            "MemoryMax",
            "CPUQuotaPerSecUSec",
            "Nice",
            "KillMode",
        )
    }
    _check_properties(props, mode)
    require(props["MainPID"] == str(os.getpid()), "this exact capped native PID")
    running = base.host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {row.split()[0] for row in running.splitlines()} == {name},
        "only this owned Duck service is active",
    )
    return props


def _check_properties(value, mode):
    require(mode in ("run", "closeout"), "typed recorded service mode")
    seconds = RUN_SECONDS if mode == "run" else CLOSEOUT_SECONDS
    require(
        type(value) is dict
        and type(value.get("MainPID")) is str
        and value["MainPID"].isdecimal()
        and int(value["MainPID"]) > 0
        and value
        == {
            "MainPID": value["MainPID"],
            "ActiveState": "active",
            "RuntimeMaxUSec": f"{seconds // 60}min",
            "MemoryMax": str(MEMORY_BYTES),
            "CPUQuotaPerSecUSec": "2s",
            "Nice": "10",
            "KillMode": "control-group",
        },
        "exact fixed CPU first-terminal service limits",
    )
    return True


def _inventory(root, names):
    root = base.files.native._plain_path(root)
    require(
        {path.name for path in root.iterdir()} == set(names),
        "exact terminal artifact inventory",
    )
    result = {}
    for name in sorted(names):
        raw = base.files.file_bytes(root / name, limit=RAW_LIMIT)
        result[name] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    return result


def _read_closed_update(root=None):
    """Actual pinned prerequisite reader; an optional path serves file-only tests."""
    root = previous.output_path(UPDATE_SOURCE) if root is None else root
    inventory = _inventory(root, previous.CLOSEOUT_FILES)
    require(
        inventory["independent-closeout.json"]["sha256"] == UPDATE_CLOSEOUT_SHA256
        and inventory["launch.json"]["sha256"] == UPDATE_LAUNCH_SHA256,
        "whole closed one-update prerequisite hashes",
    )
    close = base.files.parse(
        base.files.file_bytes(root / "independent-closeout.json", limit=2 * 1024**2)
    )
    require(
        close.get("protocol") == previous.PROTOCOL
        and close.get("artifact_source")
        == close.get("evaluator_source")
        == UPDATE_SOURCE
        and close.get("decision") == "one-update-cpu-optimizer-integration-replayed"
        and close.get("inventory")
        == {k: v for k, v in inventory.items() if k != "independent-closeout.json"}
        and close.get("final_inventory_names") == sorted(previous.CLOSEOUT_FILES)
        and close.get("launch_sha256") == UPDATE_LAUNCH_SHA256
        and close.get("run_report_sha256") == inventory["report.json"]["sha256"]
        and close.get("optimizer_steps") == 20
        and close.get("completed_updates") == 1
        and close.get("training_update_performed") is True
        and close.get("execution_admitted") is False
        and close.get("student_export_available") is False
        and all(close.get(key) is False for key in baseline.FALSE_FLAGS),
        "closed optimizer integration only, never accepted recovery policy",
    )
    previous._check_replay_result(
        close["independent_replay"],
        inventory["transition.pt"]["sha256"],
        inventory["optimizer.pt"]["sha256"],
    )
    previous._recorded_properties(close["service_properties"], "closeout")
    return {
        "source": UPDATE_SOURCE,
        "closeout_sha256": UPDATE_CLOSEOUT_SHA256,
        "inventory": inventory,
        "decision": close["decision"],
        "diagnostic_updated_actor_used": False,
    }


def _fresh(source):
    context, prior, raw_parent, audit = previous._fresh_run(source)
    require(
        sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
        "unchanged parent bytes, not the diagnostic updated actor",
    )
    return context, prior, raw_parent, audit, _read_closed_update()


def _write_raw(path, raw):
    require(
        type(raw) is bytes and 0 < len(raw) <= RAW_LIMIT, "bounded owned raw artifact"
    )
    path = base.files.native._plain_path(path)
    with path.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    base.files.native._fsync_dir(path.parent)
    return {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def _lease():
    return {
        "lock_path": str(base.files.LOCK),
        "mechanism": "advisory-flock-exclusive-nonblocking",
        "scope": ["run", "closeout"],
        "idle_gate": "two-idle-samples-before-and-after",
        "cuda_learner": False,
    }


def _score_decision(score):
    """Numeric runtime gates do not change any existing capability flags."""
    require(
        type(score) is dict and score.get("protocol") == trace.PROTOCOL,
        "exact sibling independent scorer protocol",
    )
    collection = score.get("collection")
    require(
        type(collection) is dict
        and set(collection)
        == {"policy_ticks", "elapsed_seconds", "stop_reason", "failure"}
        and type(collection["policy_ticks"]) is int
        and 0 <= collection["policy_ticks"] <= 250
        and type(score.get("validated_policy_ticks")) is int
        and score.get("validated_policy_ticks") == collection["policy_ticks"]
        and type(collection["elapsed_seconds"]) is float
        and 0 <= collection["elapsed_seconds"] < float("inf"),
        "typed finite independently replayed collection",
    )
    for key in (
        "full_timeout_qualified",
        "timeout_reset_qualified",
        "selective_reset_qualified",
    ):
        require(
            type(score.get(key)) is bool, "typed separate natural-terminal gate " + key
        )
    require(
        score.get("whole_trajectory_physics_resimulated") is False
        and score.get("thermal_model_applied") is False
        and all(score.get(key) is False for key in baseline.FALSE_FLAGS),
        "recorded-evidence integration does not admit a skill",
    )
    qualified = any(
        score[key]
        for key in (
            "full_timeout_qualified",
            "timeout_reset_qualified",
            "selective_reset_qualified",
        )
    )
    if qualified:
        require(
            collection["stop_reason"] == "first-natural-terminal"
            and collection["failure"] is None
            and 0 < collection["elapsed_seconds"] < trace.WALL_LIMIT
            and all(
                score.get(key) is True
                for key in (
                    "exact_policy_replay",
                    "private_rng",
                    "caller_rng",
                    "recorded_force_prefix_checked",
                    "terminal_critic_bootstrap_exact",
                    "compiled_plant_checked",
                    "control_trace_checked",
                    "physical_trace_checked",
                    "collection_cap_respected",
                )
            ),
            "natural terminal qualified only with every independent numeric/provenance gate",
        )
    require(
        not (score["full_timeout_qualified"] and score["selective_reset_qualified"]),
        "one common timeout does not attest an untouched selective-reset sibling",
    )
    require(
        not (score["timeout_reset_qualified"] and score["selective_reset_qualified"]),
        "synchronously started first-terminal timeout and selective reset are distinct outcomes",
    )
    if score["full_timeout_qualified"]:
        force = score.get("force")
        require(
            score["timeout_reset_qualified"]
            and score.get("complete_force_phase_checks") is True
            and collection["policy_ticks"] == 250
            and type(force) is dict
            and force.get("checked_physics_steps") == 2500
            and force.get("window_steps_per_row") == [10, 20]
            and force.get("delivered_nonzero_steps_per_row") == [0, 20],
            "full timeout requires real 250 calls, 2500 steps and both scheduled windows",
        )
    if score["full_timeout_qualified"] and score["timeout_reset_qualified"]:
        return "cpu-natural-full-timeout-and-reset-replayed"
    if score["selective_reset_qualified"]:
        return "cpu-natural-selective-reset-replayed-full-timeout-open"
    if score["timeout_reset_qualified"]:
        return "cpu-natural-timeout-reset-replayed-full-timeout-open"
    return "cpu-first-terminal-prefix-replayed-no-timeout-reset-qualification"


def run(source):
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE)
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        props = _properties(source, "run")
        fresh = _fresh(source)
        context, prior, raw_parent, audit, update = fresh
        root = output_path(source)
        root.mkdir(exist_ok=False)
        checkpoint_record = _write_raw(root / "checkpoint.pt", raw_parent)
        report = dict(
            protocol=PROTOCOL,
            source=source,
            decision="first-terminal-run-failed",
            service_properties=props,
            idle_before=idle_before,
            optimizer_steps=0,
            training_update_performed=False,
            postchecks_passed=False,
            execution_admitted=False,
            student_export_available=False,
            **baseline.FALSE_FLAGS,
        )
        try:
            require(
                os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                and not torch.cuda.is_initialized(),
                "frozen CUDA-hidden CPU first-terminal run",
            )
            from mjlab_microduck.stance_recovery_schedule_runtime import (
                ScheduledRecoveryRuntime,
            )

            declaration = previous.collection_probe.declaration(source)
            env = ScheduledRecoveryRuntime(declaration, device="cpu")
            cpu_math = previous.trace.profile.checked_receipt()
            launch = dict(
                protocol=PROTOCOL,
                source=source,
                **context,
                prerequisite_receipt=prior,
                closed_audit_binding=audit,
                closed_update_binding=update,
                parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
                checkpoint=checkpoint_record,
                declaration=declaration,
                compiled_plant=env.binding,
                cpu_math_profile=cpu_math,
                trace_binding=trace.binding(declaration, env.binding, cpu_math),
                policy_call_limit=250,
                physics_step_limit=2500,
                simulation_seconds=5.0,
                collection_seconds=trace.WALL_LIMIT,
                service_seconds=RUN_SECONDS,
                closeout_seconds=CLOSEOUT_SECONDS,
                launch_reserve_seconds=LAUNCH_RESERVE,
                raw_capture_limit_bytes=RAW_LIMIT,
                campaign_window=window.declaration(),
                service_properties=props,
                lease=_lease(),
                optimizer_steps=0,
                training_update_performed=False,
                execution_admitted=False,
                student_export_available=False,
                **baseline.FALSE_FLAGS,
            )
            base.files.write_json(root / "launch.json", launch)
            value = trace.collect(
                env, raw_parent, deadline_monotonic=time.monotonic() + trace.WALL_LIMIT
            )
            raw = trace.encode(value)
            capture_record = _write_raw(root / "capture.pt", raw)
            metadata = dict(
                protocol=trace.PROTOCOL,
                source=source,
                binding=value["binding"],
                collection=value["collection"],
                capture_sha256=capture_record["sha256"],
                capture_bytes=capture_record["bytes"],
                optimizer_steps=0,
                training_update_performed=False,
                execution_admitted=False,
                student_export_available=False,
                **baseline.FALSE_FLAGS,
            )
            base.files.write_json(root / "capture.json", metadata)
            report.update(
                capture=metadata, raw_bytes_written=len(raw) + len(raw_parent)
            )
            require(
                value["collection"].get("failure") is None,
                "retain and diagnose collection fault without retry",
            )
            require(
                _fresh(source) == fresh
                and _properties(source, "run") == props
                and not torch.cuda.is_initialized(),
                "unchanged native source/profile/prerequisites/service",
            )
            window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
            report["idle_after"] = base.host.wait_idle()
            require(
                _fresh(source) == fresh
                and _properties(source, "run") == props
                and time.monotonic() - started < RUN_SECONDS
                and not torch.cuda.is_initialized(),
                "bounded unchanged CPU run after idle gate",
            )
            window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
            report.update(
                decision="first-terminal-captured-awaiting-independent-closeout",
                postchecks_passed=True,
                source_unchanged=True,
                filmbrain_unchanged=True,
                protected_services_inactive=True,
                cuda_initialized=False,
            )
        except Exception as error:
            report.update(
                error_type=type(error).__name__,
                error=str(error)[:1000],
                cuda_initialized=bool(torch.cuda.is_initialized()),
            )
            raise
        finally:
            report["elapsed_seconds"] = float(time.monotonic() - started)
            report["files"] = _inventory(root, {path.name for path in root.iterdir()})
            base.files.write_json(root / "report.json", report)
        return {
            "output": str(root),
            "decision": report["decision"],
            "launch_sha256": base.host.digest(root / "launch.json"),
            "capture_sha256": report["capture"]["capture_sha256"],
        }


def _check_launch(launch, source, fresh):
    context, prior, raw_parent, audit, update = fresh
    require(
        type(launch) is dict
        and launch.get("protocol") == PROTOCOL
        and launch.get("source") == source
        and {key: launch.get(key) for key in context} == context
        and launch.get("prerequisite_receipt") == prior
        and launch.get("closed_audit_binding") == audit
        and launch.get("closed_update_binding") == update
        and launch.get("parent_checkpoint_sha256") == sha256(raw_parent).hexdigest()
        and launch.get("checkpoint")
        == {"sha256": baseline.CHECKPOINT_SHA256, "bytes": len(raw_parent)}
        and launch.get("policy_call_limit") == 250
        and launch.get("physics_step_limit") == 2500
        and type(launch.get("simulation_seconds")) is float
        and launch["simulation_seconds"] == 5.0
        and launch.get("collection_seconds") == trace.WALL_LIMIT
        and launch.get("service_seconds") == RUN_SECONDS
        and launch.get("closeout_seconds") == CLOSEOUT_SECONDS
        and launch.get("launch_reserve_seconds") == LAUNCH_RESERVE
        and launch.get("raw_capture_limit_bytes") == RAW_LIMIT
        and launch.get("lease") == _lease()
        and launch.get("optimizer_steps") == 0
        and launch.get("training_update_performed") is False
        and launch.get("execution_admitted") is False
        and launch.get("student_export_available") is False
        and all(launch.get(key) is False for key in baseline.FALSE_FLAGS),
        "exact source/context/parent and predeclared first-terminal limits",
    )
    previous.trace.profile.check_recorded(launch["cpu_math_profile"])
    require(
        launch["declaration"] == previous.collection_probe.declaration(source),
        "fixed original dose rows",
    )
    require(
        launch.get("trace_binding")
        == trace.binding(
            launch["declaration"], launch["compiled_plant"], launch["cpu_math_profile"]
        ),
        "exact new independent trace binding",
    )
    require(
        launch.get("campaign_window") == window.declaration(),
        "fixed original 20:00 authorization window",
    )
    _check_properties(launch["service_properties"], "run")
    return True


def _check_report(report, metadata, inventory, launch, source):
    require(
        type(report) is dict
        and type(metadata) is dict
        and report.get("protocol") == PROTOCOL
        and report.get("source") == source
        and report.get("decision")
        == "first-terminal-captured-awaiting-independent-closeout"
        and report.get("files")
        == {k: v for k, v in inventory.items() if k != "report.json"}
        and report.get("capture") == metadata
        and report.get("postchecks_passed") is True
        and report.get("source_unchanged") is True
        and report.get("filmbrain_unchanged") is True
        and report.get("protected_services_inactive") is True
        and report.get("cuda_initialized") is False
        and report.get("optimizer_steps") == 0
        and report.get("training_update_performed") is False
        and report.get("execution_admitted") is False
        and report.get("student_export_available") is False
        and all(report.get(key) is False for key in baseline.FALSE_FLAGS)
        and type(report.get("elapsed_seconds")) is float
        and 0 < report["elapsed_seconds"] < RUN_SECONDS
        and report.get("raw_bytes_written")
        == inventory["capture.pt"]["bytes"] + inventory["checkpoint.pt"]["bytes"]
        and report.get("service_properties") == launch.get("service_properties"),
        "immutable bounded run report from the identical launched PID and caps",
    )
    require(
        set(metadata)
        == {
            "protocol",
            "source",
            "binding",
            "collection",
            "capture_sha256",
            "capture_bytes",
            "optimizer_steps",
            "training_update_performed",
            "execution_admitted",
            "student_export_available",
        }
        | set(baseline.FALSE_FLAGS)
        and metadata.get("source") == source
        and metadata.get("protocol") == trace.PROTOCOL
        and metadata.get("binding") == launch["trace_binding"]
        and metadata.get("capture_sha256") == inventory["capture.pt"]["sha256"]
        and metadata.get("capture_bytes") == inventory["capture.pt"]["bytes"]
        and metadata.get("optimizer_steps") == 0
        and metadata.get("training_update_performed") is False
        and metadata.get("execution_admitted") is False
        and metadata.get("student_export_available") is False
        and all(metadata.get(key) is False for key in baseline.FALSE_FLAGS),
        "exact raw metadata is zero-update and non-admitting",
    )
    _check_properties(report["service_properties"], "run")
    return True


def closeout(source, launch_sha256):
    base.files.hex_id(launch_sha256, 64)
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        props = _properties(source, "closeout")
        fresh = _fresh(source)
        root = output_path(source)
        inventory = _inventory(root, RUN_FILES)
        require(
            inventory["launch.json"]["sha256"] == launch_sha256,
            "externally supplied whole launch hash",
        )
        parsed = {
            name: base.files.parse(
                base.files.file_bytes(root / name, limit=2 * 1024**2)
            )
            for name in ("launch.json", "capture.json", "report.json")
        }
        launch, metadata, report = (
            parsed[name] for name in ("launch.json", "capture.json", "report.json")
        )
        _check_launch(launch, source, fresh)
        require(
            base.files.file_bytes(root / "checkpoint.pt", limit=RAW_LIMIT) == fresh[2],
            "unchanged raw parent",
        )
        _check_report(report, metadata, inventory, launch, source)
        raw = base.files.file_bytes(root / "capture.pt", limit=RAW_LIMIT)
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        require(
            value.get("protocol") == trace.PROTOCOL
            and value.get("binding") == launch["trace_binding"]
            and value.get("declaration") == launch["declaration"]
            and value.get("compiled_plant") == launch["compiled_plant"]
            and value.get("collection") == metadata.get("collection"),
            "actual raw trace binds launch, complete declaration and collection metadata",
        )
        score = trace.verify(
            raw,
            inventory["capture.pt"]["sha256"],
            fresh[2],
            launch["declaration"],
            launch["compiled_plant"],
        )
        decision = _score_decision(score)
        require(
            _fresh(source) == fresh
            and _properties(source, "closeout") == props
            and _inventory(root, RUN_FILES) == inventory
            and not torch.cuda.is_initialized(),
            "pinned source/profile/old failures/evidence after independent replay",
        )
        idle_after = base.host.wait_idle()
        require(
            _fresh(source) == fresh
            and _properties(source, "closeout") == props
            and _inventory(root, RUN_FILES) == inventory
            and not torch.cuda.is_initialized()
            and time.monotonic() - started < CLOSEOUT_SECONDS,
            "unchanged bounded final idle gate",
        )
        window.check()
        receipt = dict(
            protocol=PROTOCOL,
            artifact_source=source,
            evaluator_source=source,
            launch_sha256=launch_sha256,
            inventory=inventory,
            final_inventory_names=sorted(CLOSED_FILES),
            closed_audit_binding=fresh[3],
            closed_update_binding=fresh[4],
            independent_replay=score,
            decision=decision,
            elapsed_seconds=float(time.monotonic() - started),
            service_properties=props,
            lease=_lease(),
            idle_before=idle_before,
            idle_after=idle_after,
            optimizer_steps=0,
            training_update_performed=False,
            execution_admitted=False,
            student_export_available=False,
            physical_resimulation_performed=False,
            **baseline.FALSE_FLAGS,
        )
        base.files.write_json(root / "independent-closeout.json", receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", required=True, choices=("run", "closeout"))
    parser.add_argument("--launch-sha256")
    args = parser.parse_args(argv)
    result = (
        run(args.source)
        if args.mode == "run"
        else closeout(args.source, args.launch_sha256)
    )
    if args.mode == "closeout":
        result = {
            "decision": result["decision"],
            "launch_sha256": result["launch_sha256"],
            "closeout_sha256": base.host.digest(
                output_path(args.source) / "independent-closeout.json"
            ),
        }
    print(
        {"mode": args.mode, "output": str(output_path(args.source)), "result": result}
    )


if __name__ == "__main__":
    main()
