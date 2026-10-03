"""One bounded native CPU PPO update with durable diagnostic progression.

This is an integration probe, not a trainer, resume path, checkpoint export,
admission or physical-recovery qualification. A successful optimizer artifact
is independently replayed offline from the authenticated retained rollout.
"""

import argparse
import io
import os
import random
import time
from hashlib import sha256

import numpy as np
import torch

from mjlab_microduck import stance_recovery_broad_receipt_repair as broad_audit
from mjlab_microduck import stance_recovery_broad_screen as prerequisites
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_optimizer_trace as optimizer_trace
from mjlab_microduck import stance_recovery_ppo_bridge as bridge
from mjlab_microduck import stance_recovery_ppo_probe as collection_probe
from mjlab_microduck import stance_recovery_ppo_receipt_repair as audit
from mjlab_microduck import stance_recovery_ppo_trace as trace
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cpu-scheduled-ppo-one-update-probe-v1"
DIAGNOSTIC_PROTOCOL = PROTOCOL + ":diagnostic-state-v1"
EXCEPTION_PROTOCOL = PROTOCOL + ":exception-state-v1"
ARTIFACT_SOURCE = "b1878b8715efbb7ea6e166bc7d345efecf5285de"
AUDIT_SOURCE = "75ed100d2b8470c86e0327224073013dab393d23"
AUDIT_LAUNCH_SHA256 = "d8cb9f0142e4497eb9c3b9091e54ab63041411a1e50046d89672f89bac8fd701"
AUDIT_SOURCE_INVENTORY_SHA256 = (
    "d1d7423c4d3a9605994d54f3dce3e8f4a079aa1370bb1d0f3991d2312d82aecc"
)
AUDIT_RECEIPT_SHA256 = (
    "8a1d1ed03dcaee29a7d8dbdcaed58c189eecf57a6928075670d27c8fcf75785f"
)
OLD_FAILURES = {
    "microduck-cpu-broad-closeout-1079a104a9e3.service": "5cf35c7873924d9d9218c3d8764c0b56",
    "microduck-cpu-broad-repair-1dc0b3415942.service": "a2bf9ef353dc46db87cf4bbb0a0909b7",
    "microduck-cpu-ppo-closeout-b1878b8715ef.service": "ccd6921149844ace9fa9e12cca4bd816",
}
RUN_SECONDS, CLOSEOUT_SECONDS, MARGIN_SECONDS = 360, 240, 60
LAUNCH_RESERVE = RUN_SECONDS + CLOSEOUT_SECONDS + MARGIN_SECONDS
MEMORY_BYTES, CPU_QUOTA, NICE, KILL_MODE = 2 * 1024**3, "2s", "10", "control-group"
RAW_LIMIT = 64 * 1024 * 1024
OUTPUT_PREFIX = "stance-wsl-cpu-scheduled-ppo-one-update-"
DIAGNOSTIC_FILES = {f"diagnostic-state-step-{step:03d}.pt" for step in range(21)}
PRE_REPORT_FILES = {
    "checkpoint.pt",
    "launch.json",
    "transition.pt",
    "transition.json",
    "capture-attempt.json",
    "preupdate.pt",
    "optimizer.pt",
    "optimizer.json",
} | DIAGNOSTIC_FILES
COMPLETE_FILES = PRE_REPORT_FILES | {"report.json"}
CLOSEOUT_FILES = COMPLETE_FILES | {"independent-closeout.json"}


base = prerequisites.base


def output_path(source):
    base.files.hex_id(source, 40)
    require(
        source not in (ARTIFACT_SOURCE, AUDIT_SOURCE), "distinct fresh evaluator source"
    )
    return base.host.ROOT / "artifacts/evaluations" / (OUTPUT_PREFIX + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ("run", "closeout"), "declared optimizer probe service mode")
    return f"microduck-cpu-ppo-update-{mode}-{source[:12]}.service"


def lease_declaration():
    return {
        "lock_path": str(base.files.LOCK),
        "mechanism": "advisory-flock-exclusive-nonblocking",
        "scope": ["run", "closeout"],
        "idle_gate": "base.host.wait_idle-before-and-after",
        "cuda_learner": False,
    }


def service_properties(source, mode):
    seconds = {"run": RUN_SECONDS, "closeout": CLOSEOUT_SECONDS}[mode]
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
    require(
        props
        == {
            "MainPID": str(os.getpid()),
            "ActiveState": "active",
            "RuntimeMaxUSec": f"{seconds // 60}min",
            "MemoryMax": str(MEMORY_BYTES),
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
        },
        "exact capped CPU optimizer service",
    )
    running = base.host.read(
        "systemctl",
        "--user",
        "list-units",
        "--state=running",
        "--no-legend",
        "microduck*",
    )
    require(
        {line.split()[0] for line in running.splitlines()} == {name},
        "only this owned Duck service runs during optimizer probe",
    )
    return props


def _recorded_properties(value, mode):
    seconds = {"run": RUN_SECONDS, "closeout": CLOSEOUT_SECONDS}[mode]
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
            "CPUQuotaPerSecUSec": CPU_QUOTA,
            "Nice": NICE,
            "KillMode": KILL_MODE,
        },
        "recorded exact CPU service cap",
    )
    return True


def _recorded_failed_state(value, invocation):
    require(
        type(value) is dict
        and value
        == {
            "ActiveState": "failed",
            "MainPID": "0",
            "NRestarts": "0",
            "ExecMainStatus": "1",
            "InvocationID": invocation,
        },
        "exact unchanged prior failure state and invocation",
    )
    return True


def _update_flag_after_failure(*, attempted, optimizer_steps, completed_updates):
    require(
        type(attempted) is bool
        and type(optimizer_steps) is int
        and optimizer_steps >= 0
        and type(completed_updates) is int
        and completed_updates in (0, 1),
        "typed partial optimizer progress",
    )
    if optimizer_steps or completed_updates:
        return True
    return None if attempted else False


def _old_failed_states():
    states = {}
    keys = ("ActiveState", "MainPID", "NRestarts", "ExecMainStatus", "InvocationID")
    for unit, invocation in OLD_FAILURES.items():
        current = {
            key: base.host.read(
                "systemctl", "--user", "show", unit, "-p", key, "--value"
            )
            for key in keys
        }
        _recorded_failed_state(current, invocation)
        states[unit] = current
    return states


def _audit_inventory(root=None, *, check_live=True):
    root = (
        audit.output_path(AUDIT_SOURCE)
        if root is None
        else base.files.native._plain_path(root)
    )
    expected = {
        "launch.json": AUDIT_LAUNCH_SHA256,
        "source-inventory.json": AUDIT_SOURCE_INVENTORY_SHA256,
        "receipt.json": AUDIT_RECEIPT_SHA256,
    }
    require(
        {item.name for item in root.iterdir()} == set(expected),
        "exact closed optimizer prerequisite audit inventory",
    )
    inventory, parsed = {}, {}
    for name, digest in expected.items():
        raw = base.files.file_bytes(root / name, limit=RAW_LIMIT)
        require(sha256(raw).hexdigest() == digest, "pinned closed audit file " + name)
        inventory[name] = {"sha256": digest, "bytes": len(raw)}
        parsed[name] = base.files.parse(raw)
    receipt = parsed["receipt.json"]
    source_inventory = parsed["source-inventory.json"]
    require(
        receipt.get("protocol") == audit.PROTOCOL
        and receipt.get("artifact_source") == audit.ARTIFACT_SOURCE
        and receipt.get("evaluator_source") == AUDIT_SOURCE
        and receipt.get("independent_cpu_replay") is True
        and receipt.get("run_admissible") is True
        and receipt.get("decision")
        == "cpu-ppo-receipt-repair-replayed-complete-cpu2-trace"
        and receipt.get("optimizer_steps") == 0
        and receipt.get("training_update_performed") is False
        and receipt.get("cuda_initialized") is False
        and all(receipt.get(key) is False for key in baseline.FALSE_FLAGS),
        "closed CPU no-update audit receipt and non-admission flags",
    )
    audit._check_replay(receipt.get("replay"))
    require(
        source_inventory.get("artifact_source") == audit.ARTIFACT_SOURCE
        and source_inventory.get("evaluator_source") == AUDIT_SOURCE
        and source_inventory.get("source_hashes") == receipt.get("source_hashes"),
        "closed audit artifact/evaluator source inventory separation",
    )
    context_binding = receipt.get("context_binding")
    require(
        type(context_binding) is dict
        and context_binding.get("evaluator_context", {})
        .get("source_identity", {})
        .get("source")
        == AUDIT_SOURCE,
        "closed audit evaluator context identity",
    )
    result = {
        "artifact_source": audit.ARTIFACT_SOURCE,
        "evaluator_source": AUDIT_SOURCE,
        "inventory": inventory,
    }
    if check_live:
        audit._rehash_original(
            base.host.ROOT / audit.ARTIFACT_ROOT, receipt["original_inventory"]
        )
        require(
            audit._source_inventory() == receipt["source_hashes"],
            "closed audit's complete original dependency source closure",
        )
        old_broad_failure = broad_audit._failure_link()
        old_broad_preflight = broad_audit._preflight_failure_link()
        old_broad_service = broad_audit._old_failed_service_state()
        old_ppo_failure = audit._failure_link()
        old_ppo_service = audit._old_service_state()
        old_failed_states = _old_failed_states()
        require(
            receipt.get("failure") == old_ppo_failure,
            "closed audit retains exact original PPO closeout failure link",
        )
        result.update(
            old_broad_failure=old_broad_failure,
            old_broad_service_state=old_broad_service,
            old_broad_preflight_failure=old_broad_preflight,
            old_ppo_failure=old_ppo_failure,
            old_ppo_service_state=old_ppo_service,
            exact_old_failed_invocations=old_failed_states,
        )
    return result


def _serialize(value):
    target = io.BytesIO()
    torch.save(value, target)
    raw = target.getvalue()
    require(0 < len(raw) <= RAW_LIMIT, "bounded individual optimizer evidence file")
    return raw


def _write_raw(path, raw, budget):
    require(
        type(raw) is bytes
        and 0 < len(raw) <= RAW_LIMIT
        and budget["written"] + len(raw) <= RAW_LIMIT,
        "bounded total raw optimizer artifact budget",
    )
    base.retained.write_capture(path, raw)
    budget["written"] += len(raw)
    return {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def _report_inventory(root):
    return {
        path.name: {"sha256": base.host.digest(path), "bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def _cpu_rng():
    return torch.random.get_rng_state().detach().cpu().clone()


def _named_adam(learner):
    names = optimizer_trace._parameter_names(learner.actor, learner.critic)
    result = {}
    for name, parameter in names.items():
        item = learner.algorithm.optimizer.state.get(parameter, {})
        result[name] = {
            key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
            for key, value in item.items()
        }
    return result


def _rng_snapshot(context, caller_before, private_state, observed_cpu=None):
    require(
        context
        in {
            "outside-update-caller",
            "inside-private-update-fork",
            "after-update-unwind",
        },
        "explicit diagnostic RNG execution context",
    )
    return {
        "context": context,
        "caller_rng_before": caller_before.detach().cpu().clone(),
        "observed_cpu_rng": (
            _cpu_rng() if observed_cpu is None else observed_cpu.detach().cpu().clone()
        ),
        "learner_private_rng_state": private_state.detach().cpu().clone(),
    }


def _diagnostic_value(learner, step, rng_context, caller_before):
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "kind": "diagnostic-only-not-replayable",
        "step_index": step,
        "model_states": optimizer_trace._model_states(learner.actor, learner.critic),
        "adam_by_parameter": _named_adam(learner),
        "counters": {
            "completed_updates": learner.completed_updates,
            "optimizer_steps": learner.optimizer_steps,
            "storage_step": learner.storage.step,
            "phase": learner.phase,
            "faulted": learner.faulted,
        },
        "storage": trace._storage(learner.storage),
        "private_rng_state": learner.private_rng_state.detach().cpu().clone(),
        "rng": _rng_snapshot(
            rng_context, caller_before, learner.private_rng_state, _cpu_rng()
        ),
        "physical_resimulation_performed": False,
        "execution_admitted": False,
        "student_export_available": False,
        **baseline.FALSE_FLAGS,
    }


def _finite_step(learner, *, moments_required):
    optimizer_trace._finite_models(learner.actor, learner.critic)
    optimizer_trace._positive_gaussian(learner.actor)
    for group in learner.algorithm.optimizer.param_groups:
        for parameter in group["params"]:
            require(
                parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                "finite gradients at observed optimizer step",
            )
            state = learner.algorithm.optimizer.state.get(parameter)
            if moments_required:
                require(
                    type(state) is dict and state,
                    "actual Adam state after optimizer step",
                )
            else:
                require(
                    state is None or type(state) is dict,
                    "typed optional fresh Adam state before first step",
                )
            for item in (state or {}).values():
                if torch.is_tensor(item):
                    require(torch.isfinite(item).all(), "finite Adam values after step")
    return True


def _install_diagnostic_hooks(learner, root, budget, caller_before):
    counts = {"pre": 0, "post": 0}
    handles = []

    def before(_optimizer, _args, _kwargs):
        require(
            counts["pre"] < optimizer_trace.UPDATE_STEPS,
            "no twenty-first optimizer step",
        )
        _finite_step(learner, moments_required=False)
        counts["pre"] += 1

    def after(_optimizer, _args, _kwargs):
        _finite_step(learner, moments_required=True)
        counts["post"] += 1
        require(
            counts["post"] <= optimizer_trace.UPDATE_STEPS,
            "no extra optimizer post-step",
        )
        step = counts["post"]
        snapshot = _diagnostic_value(
            learner, step, "inside-private-update-fork", caller_before
        )
        raw = _serialize(snapshot)
        name = f"diagnostic-state-step-{step:03d}.pt"
        _write_raw(root / name, raw, budget)

    optimizer = learner.algorithm.optimizer
    handles.append(optimizer.register_step_pre_hook(before))
    handles.append(optimizer.register_step_post_hook(after))
    return counts, handles


def _check_trace_score(score):
    require(
        type(score) is dict
        and score.get("complete_two_world_transition_qualification") is True
        and score.get("validated_policy_ticks") == trace.HORIZON
        and score.get("storage_step") == trace.HORIZON
        and score.get("collection", {}).get("accepted_complete") is True
        and score.get("collection", {}).get("stop_reason") == "transition-limit"
        and score.get("collection", {}).get("failure") is None
        and score.get("pulse", {}).get("complete_pulse_delivery") is True
        and all(score.get(key) is False for key in baseline.FALSE_FLAGS),
        "complete verified native no-update CPU2 trace before optimizer attempt",
    )


def _check_replay_result(replay, transition_sha256, optimizer_sha256):
    expected = {
        "protocol",
        "optimizer_replay",
        "environment_created",
        "physics_resimulated",
        "exact_gae",
        "exact_updated_model",
        "exact_adam_state",
        "exact_private_rng",
        "exact_metrics",
        "gradient_hook_steps",
        "moment_hook_steps",
        "whole_trajectory_physics_resimulated",
        "thermal_model_applied",
        "cuda_initialized",
        "optimizer_integration_only",
        "source_trace_sha256",
        "capture_sha256",
        "source_trace_score",
    } | set(baseline.FALSE_FLAGS)
    require(
        type(replay) is dict
        and set(replay) == expected
        and replay.get("protocol") == optimizer_trace.PROTOCOL
        and replay.get("optimizer_replay") is True
        and replay.get("environment_created") is False
        and replay.get("physics_resimulated") is False
        and all(
            replay.get(key) is True
            for key in (
                "exact_gae",
                "exact_updated_model",
                "exact_adam_state",
                "exact_private_rng",
                "exact_metrics",
            )
        )
        and replay.get("gradient_hook_steps") == optimizer_trace.UPDATE_STEPS
        and replay.get("moment_hook_steps") == optimizer_trace.UPDATE_STEPS
        and replay.get("whole_trajectory_physics_resimulated") is False
        and replay.get("thermal_model_applied") is False
        and replay.get("cuda_initialized") is False
        and replay.get("optimizer_integration_only") is True
        and replay.get("source_trace_sha256") == transition_sha256
        and replay.get("capture_sha256") == optimizer_sha256
        and type(replay.get("source_trace_score")) is dict
        and all(replay.get(key) is False for key in baseline.FALSE_FLAGS),
        "exact offline optimizer replay schema and identity",
    )
    _check_trace_score(replay["source_trace_score"])
    return True


def _fresh_run(source):
    context = prerequisites._context(source)
    previous, raw_parent = prerequisites.prerequisites()
    require(
        sha256(raw_parent).hexdigest() == baseline.CHECKPOINT_SHA256,
        "unchanged exact parent checkpoint bytes",
    )
    audit_binding = _audit_inventory()
    context_binding = audit.base.files.parse(
        base.files.file_bytes(
            audit.output_path(AUDIT_SOURCE) / "receipt.json", limit=RAW_LIMIT
        )
    )["context_binding"]
    broad_audit._check_source_context(
        context_binding["evaluator_context"],
        context,
        artifact_source=AUDIT_SOURCE,
        evaluator_source=source,
    )
    return context, previous, raw_parent, audit_binding


def _check_transition_binding(value, source, launch, transition_meta):
    expected = trace.binding(
        launch["declaration"], launch["compiled_plant"], launch["cpu_math_profile"]
    )
    require(
        type(value) is dict
        and value.get("protocol") == trace.PROTOCOL
        and value.get("binding") == expected
        and value["binding"].get("source") == source
        and value.get("declaration") == launch["declaration"]
        and value.get("compiled_plant") == launch["compiled_plant"]
        and type(transition_meta) is dict
        and transition_meta.get("source") == source
        and transition_meta.get("binding") == expected,
        "raw trace and metadata bind exact source/declaration/compiled plant",
    )
    return True


def _check_optimizer_binding(value, source, transition_sha256, raw_bytes):
    binding = value.get("binding") if type(value) is dict else None
    require(
        type(value) is dict
        and value.get("protocol") == optimizer_trace.PROTOCOL
        and type(binding) is dict
        and binding.get("source") == source
        and binding.get("source_trace_sha256") == transition_sha256
        and binding.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and binding.get("parent_checkpoint_identity")
        == optimizer_trace.parent.expected_identity()
        and binding.get("learner_seed") == trace.SEED
        and binding.get("worlds") == optimizer_trace.WORLDS
        and binding.get("horizon") == optimizer_trace.HORIZON
        and binding.get("capture_device") == "cpu"
        and binding.get("synthetic_fixture") is False
        and value.get("backend", {}).get("synthetic_fixture") is False
        and value.get("training_update_performed") is True
        and value.get("optimizer_integration_only") is True
        and value.get("execution_admitted") is False
        and value.get("student_export_available") is False
        and all(value.get(key) is False for key in baseline.FALSE_FLAGS)
        and type(raw_bytes) is int
        and 0 < raw_bytes <= RAW_LIMIT,
        "raw optimizer capture binds exact source/trace/parent and CPU update",
    )
    return True


def _invoke_update_once(learner, raw_parent, raw_trace, trace_sha256, *, capture=None):
    capture = optimizer_trace.capture if capture is None else capture
    require(
        callable(capture)
        and learner.completed_updates == 0
        and learner.optimizer_steps == 0,
        "one fresh optimizer capture call",
    )
    result = capture(learner, raw_parent, raw_trace, trace_sha256)
    require(
        learner.completed_updates == 1
        and learner.optimizer_steps == optimizer_trace.UPDATE_STEPS,
        "single bridge update completed exactly twenty steps",
    )
    return result


def _marker_then_capture(write_marker, capture_call, hooks):
    try:
        write_marker()
        return capture_call()
    finally:
        for handle in hooks:
            handle.remove()


def run(source):
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE)
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        return _run_leased(source, idle_before, started)


def _run_leased(source, idle_before, started):
    window.check(reserve_seconds=LAUNCH_RESERVE)
    properties = service_properties(source, "run")
    context, previous, raw_parent, audit_binding = _fresh_run(source)
    root = base.files.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    budget = {"written": 0}
    checkpoint_record = _write_raw(root / "checkpoint.pt", raw_parent, budget)
    report = dict(
        protocol=PROTOCOL,
        source=source,
        decision="optimizer-probe-failed",
        optimizer_steps=0,
        completed_updates=0,
        training_update_performed=False,
        service_properties=properties,
        idle_before=idle_before,
        raw_budget_bytes=RAW_LIMIT,
        audit_binding=audit_binding,
        files={},
        postchecks_passed=False,
        **baseline.FALSE_FLAGS,
    )
    learner = None
    caller_rng_before = None
    try:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not torch.cuda.is_initialized(),
            "CUDA-hidden CPU optimizer run",
        )
        from mjlab_microduck.stance_recovery_schedule_runtime import (
            ScheduledRecoveryRuntime,
        )

        random.seed(trace.SEED)
        np.random.seed(trace.SEED)
        torch.random.default_generator.manual_seed(trace.SEED)
        declaration = collection_probe.declaration(source)
        env = ScheduledRecoveryRuntime(declaration, device="cpu")
        learner = bridge.RecoveryPPOBridge(
            raw_parent, env, declaration, seed=trace.SEED
        )
        caller_rng_before = _cpu_rng()
        launch = dict(
            protocol=PROTOCOL,
            source=source,
            **context,
            prerequisite_receipt=previous,
            audit_binding=audit_binding,
            campaign_window=window.declaration(),
            declaration=declaration,
            compiled_plant=env.binding,
            parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
            learner_seed=trace.SEED,
            worlds=2,
            horizon=trace.HORIZON,
            collection_seconds=trace.WALL_LIMIT,
            optimizer_step_limit=optimizer_trace.UPDATE_STEPS,
            service_seconds=RUN_SECONDS,
            closeout_seconds=CLOSEOUT_SECONDS,
            launch_reserve_seconds=LAUNCH_RESERVE,
            service_properties=properties,
            lease=lease_declaration(),
            checkpoint=checkpoint_record,
            training_update_performed=False,
            optimizer_steps=0,
            **baseline.FALSE_FLAGS,
        )
        base.files.write_json(root / "launch.json", launch)
        require(
            time.monotonic() - started < RUN_SECONDS - trace.WALL_LIMIT - 40,
            "reserve bounded complete collection and scoring",
        )
        value = trace.collect(
            learner, raw_parent, deadline_monotonic=time.monotonic() + trace.WALL_LIMIT
        )
        transition_raw = trace.encode(value)
        transition_record = _write_raw(root / "transition.pt", transition_raw, budget)
        transition_meta = dict(
            protocol=trace.PROTOCOL,
            source=source,
            binding=value["binding"],
            sha256=transition_record["sha256"],
            bytes=transition_record["bytes"],
            collection=value["collection"],
            optimizer_steps=0,
            training_update_performed=False,
            **baseline.FALSE_FLAGS,
        )
        base.files.write_json(root / "transition.json", transition_meta)
        report.update(
            transition=transition_meta,
            transition_decision=collection_probe._capture_decision(value["collection"]),
        )
        require(
            value["collection"].get("accepted_complete") is True,
            "only complete fresh 28x2 trace proceeds to one optimizer update",
        )
        score = trace.verify(
            transition_raw,
            transition_record["sha256"],
            raw_parent,
            declaration,
            env.binding,
        )
        _check_trace_score(score)
        report["trace_score"] = score
        window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
        preupdate = _diagnostic_value(
            learner, 0, "outside-update-caller", caller_rng_before
        )
        preupdate_raw = _serialize(preupdate)
        preupdate_record = _write_raw(root / "preupdate.pt", preupdate_raw, budget)
        zero = _serialize(
            _diagnostic_value(learner, 0, "outside-update-caller", caller_rng_before)
        )
        step_zero = _write_raw(root / "diagnostic-state-step-000.pt", zero, budget)
        caller_before = _cpu_rng()
        counts, hooks = _install_diagnostic_hooks(learner, root, budget, caller_before)
        marker = dict(
            protocol=PROTOCOL,
            state="capture-invocation-entering",
            transition_sha256=transition_record["sha256"],
            preupdate=preupdate_record,
            durable_step_zero=step_zero,
            optimizer_calls_before=learner.completed_updates,
            optimizer_steps_before=learner.optimizer_steps,
            call_limit=1,
            step_limit=optimizer_trace.UPDATE_STEPS,
            **baseline.FALSE_FLAGS,
        )
        update_value = None
        update_value = _marker_then_capture(
            lambda: base.files.write_json(root / "capture-attempt.json", marker),
            lambda: _invoke_update_once(
                learner, raw_parent, transition_raw, transition_record["sha256"]
            ),
            hooks,
        )
        require(
            counts
            == {
                "pre": optimizer_trace.UPDATE_STEPS,
                "post": optimizer_trace.UPDATE_STEPS,
            },
            "twenty actual finite Adam steps observed",
        )
        optimizer_raw = optimizer_trace.encode(update_value)
        optimizer_record = _write_raw(root / "optimizer.pt", optimizer_raw, budget)
        optimizer_meta = dict(
            protocol=optimizer_trace.PROTOCOL,
            source=source,
            binding=update_value["binding"],
            capture_sha256=optimizer_record["sha256"],
            capture_bytes=optimizer_record["bytes"],
            transition_sha256=transition_record["sha256"],
            learner_update_receipt=update_value["learner_update_receipt"],
            completed_updates=1,
            optimizer_steps=optimizer_trace.UPDATE_STEPS,
            hook_counts=counts,
            training_update_performed=True,
            optimizer_integration_only=True,
            execution_admitted=False,
            student_export_available=False,
            **baseline.FALSE_FLAGS,
        )
        base.files.write_json(root / "optimizer.json", optimizer_meta)
        report.update(
            optimizer=optimizer_meta,
            decision="one-update-captured-awaiting-closeout",
            optimizer_steps=optimizer_trace.UPDATE_STEPS,
            completed_updates=1,
            training_update_performed=True,
            diagnostic_state_step_zero=step_zero,
        )
        require(
            prerequisites._context(source) == context
            and service_properties(source, "run") == properties
            and _audit_inventory() == audit_binding
            and not torch.cuda.is_initialized()
            and time.monotonic() - started < RUN_SECONDS,
            "unchanged source, audit, service, CUDA state and run deadline",
        )
        window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
        report["idle_after"] = base.host.wait_idle()
        require(
            time.monotonic() - started < RUN_SECONDS
            and prerequisites._context(source) == context
            and service_properties(source, "run") == properties
            and _audit_inventory() == audit_binding
            and not torch.cuda.is_initialized(),
            "unchanged bounded run after idle closeout gate",
        )
        window.check(reserve_seconds=CLOSEOUT_SECONDS + MARGIN_SECONDS)
        report.update(
            postchecks_passed=True,
            source_unchanged=True,
            filmbrain_unchanged=True,
            protected_services_inactive=True,
            cuda_initialized=False,
            elapsed_seconds=float(time.monotonic() - started),
        )
    except Exception as error:
        report["decision"] = "one-update-probe-failed-partial-retained"
        if learner is not None:
            report["optimizer_steps"] = learner.optimizer_steps
            report["completed_updates"] = learner.completed_updates
            attempted = (root / "capture-attempt.json").is_file()
            report["training_update_performed"] = _update_flag_after_failure(
                attempted=attempted,
                optimizer_steps=learner.optimizer_steps,
                completed_updates=learner.completed_updates,
            )
            report["update_progress"] = (
                "not-started"
                if not attempted
                else "complete"
                if learner.completed_updates == 1
                else "partial-or-in-progress"
                if learner.optimizer_steps
                else "capture-attempted-update-start-unknown"
            )
        report.update(
            error_type=type(error).__name__,
            error=str(error)[:1000],
            error_notes=getattr(error, "__notes__", []),
        )
        if learner is not None:
            report["observed_optimizer_steps"] = learner.optimizer_steps
            report["observed_completed_updates"] = learner.completed_updates
            report["faulted"] = learner.faulted
            report["partial_step_status"] = (
                "a step may be in progress after the last durable snapshot"
            )
            durable = [
                int(path.stem.rsplit("-", 1)[1])
                for path in root.glob("diagnostic-state-step-*.pt")
            ]
            report["last_durable_diagnostic_step"] = max(durable) if durable else None
            try:
                partial = _diagnostic_value(
                    learner,
                    learner.optimizer_steps,
                    "after-update-unwind"
                    if (root / "capture-attempt.json").is_file()
                    else "outside-update-caller",
                    caller_rng_before if caller_rng_before is not None else _cpu_rng(),
                )
                partial["protocol"] = EXCEPTION_PROTOCOL
                partial["kind"] = "exception-diagnostic-only-not-replayable"
                raw = _serialize(partial)
                report["exception_state"] = _write_raw(
                    root / "exception-state.pt", raw, budget
                )
            except Exception as snapshot_error:  # noqa: BLE001 - preserve original failure
                report["exception_snapshot_error"] = (
                    type(snapshot_error).__name__ + ": " + str(snapshot_error)[:300]
                )
        report["cuda_initialized"] = bool(torch.cuda.is_initialized())
        report["elapsed_seconds"] = float(time.monotonic() - started)
        raise
    finally:
        report["raw_bytes_written"] = budget["written"]
        report["files"] = _report_inventory(root)
        base.files.write_json(root / "report.json", report)
    return {
        "output": str(root),
        "decision": report["decision"],
        "launch_sha256": base.host.digest(root / "launch.json"),
        "optimizer_sha256": report["optimizer"]["capture_sha256"],
    }


def _rehash(root, expected):
    require(
        {item.name for item in root.iterdir()} == COMPLETE_FILES,
        "exact immutable optimizer probe artifact inventory",
    )
    inventory = {}
    for name in sorted(expected):
        raw = base.files.file_bytes(root / name, limit=RAW_LIMIT)
        inventory[name] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    return inventory


def closeout(source, launch_sha256):
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS)
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        return _closeout_leased(source, launch_sha256, idle_before, started)


def _closeout_leased(source, launch_sha256, idle_before, started):
    base.files.hex_id(launch_sha256, 64)
    window.check(reserve_seconds=CLOSEOUT_SECONDS)
    properties = service_properties(source, "closeout")
    context, previous, raw_parent, audit_binding = _fresh_run(source)
    root = output_path(source)
    launch_raw = base.files.file_bytes(root / "launch.json", limit=RAW_LIMIT)
    require(
        sha256(launch_raw).hexdigest() == launch_sha256,
        "whole launch hash before parsing",
    )
    launch = base.files.parse(launch_raw)
    require(
        launch.get("protocol") == PROTOCOL
        and launch.get("source") == source
        and {key: launch.get(key) for key in context} == context
        and launch.get("prerequisite_receipt") == previous
        and launch.get("audit_binding") == audit_binding
        and launch.get("campaign_window") == window.declaration()
        and launch.get("lease") == lease_declaration()
        and launch.get("declaration") == collection_probe.declaration(source)
        and launch.get("parent_checkpoint_sha256") == baseline.CHECKPOINT_SHA256
        and launch.get("checkpoint")
        == {"sha256": baseline.CHECKPOINT_SHA256, "bytes": len(raw_parent)}
        and launch.get("learner_seed") == trace.SEED
        and launch.get("worlds") == optimizer_trace.WORLDS
        and launch.get("horizon") == optimizer_trace.HORIZON
        and launch.get("collection_seconds") == trace.WALL_LIMIT
        and launch.get("service_seconds") == RUN_SECONDS
        and launch.get("closeout_seconds") == CLOSEOUT_SECONDS
        and launch.get("launch_reserve_seconds") == LAUNCH_RESERVE
        and _recorded_properties(launch.get("service_properties"), "run") is True
        and launch.get("optimizer_step_limit") == optimizer_trace.UPDATE_STEPS
        and launch.get("optimizer_steps") == 0
        and launch.get("training_update_performed") is False
        and all(launch.get(key) is False for key in baseline.FALSE_FLAGS)
        and base.files.file_bytes(root / "checkpoint.pt", limit=RAW_LIMIT)
        == raw_parent,
        "exact fresh parent, source/context split, schedule and pre-update launch",
    )
    require(
        {item.name for item in root.iterdir()} == COMPLETE_FILES,
        "complete successful update inventory only",
    )
    report_raw = base.files.file_bytes(root / "report.json", limit=RAW_LIMIT)
    report = base.files.parse(report_raw)
    require(
        report.get("postchecks_passed") is True
        and report.get("decision") == "one-update-captured-awaiting-closeout"
        and report.get("optimizer_steps") == optimizer_trace.UPDATE_STEPS
        and report.get("completed_updates") == 1
        and report.get("training_update_performed") is True
        and report.get("cuda_initialized") is False
        and report.get("service_properties") == launch.get("service_properties")
        and all(
            report.get(key) is True
            for key in (
                "source_unchanged",
                "filmbrain_unchanged",
                "protected_services_inactive",
            )
        )
        and type(report.get("elapsed_seconds")) is float
        and 0 < report["elapsed_seconds"] < RUN_SECONDS
        and all(report.get(key) is False for key in baseline.FALSE_FLAGS),
        "successful one-update run receipt without admission",
    )
    inventory = _rehash(root, PRE_REPORT_FILES)
    require(
        report.get("files") == inventory,
        "run receipt pins every complete artifact before closeout",
    )
    require(
        inventory["transition.pt"]["sha256"] == report["transition"]["sha256"]
        and inventory["optimizer.pt"]["sha256"] == report["optimizer"]["capture_sha256"]
        and inventory["checkpoint.pt"]["sha256"] == baseline.CHECKPOINT_SHA256,
        "whole retained transition, optimizer and exact parent hashes",
    )
    transition_raw = base.files.file_bytes(root / "transition.pt", limit=RAW_LIMIT)
    optimizer_raw = base.files.file_bytes(root / "optimizer.pt", limit=RAW_LIMIT)
    transition = base.files.parse(
        base.files.file_bytes(root / "transition.json", limit=RAW_LIMIT)
    )
    metadata = base.files.parse(
        base.files.file_bytes(root / "optimizer.json", limit=RAW_LIMIT)
    )
    marker = base.files.parse(
        base.files.file_bytes(root / "capture-attempt.json", limit=RAW_LIMIT)
    )
    require(
        transition.get("source") == source
        and transition.get("protocol") == trace.PROTOCOL
        and transition.get("sha256") == inventory["transition.pt"]["sha256"]
        and transition.get("bytes") == len(transition_raw)
        and transition.get("training_update_performed") is False
        and metadata.get("capture_sha256") == inventory["optimizer.pt"]["sha256"]
        and metadata.get("completed_updates") == 1
        and metadata.get("optimizer_steps") == optimizer_trace.UPDATE_STEPS
        and metadata.get("training_update_performed") is True
        and all(metadata.get(key) is False for key in baseline.FALSE_FLAGS)
        and marker.get("state") == "capture-invocation-entering"
        and marker.get("transition_sha256") == inventory["transition.pt"]["sha256"]
        and marker.get("preupdate") == inventory["preupdate.pt"]
        and marker.get("durable_step_zero") == inventory["diagnostic-state-step-000.pt"]
        and marker.get("call_limit") == 1
        and marker.get("step_limit") == optimizer_trace.UPDATE_STEPS
        and all(marker.get(key) is False for key in baseline.FALSE_FLAGS),
        "typed transition and one-update metadata",
    )
    raw_trace_value = torch.load(
        io.BytesIO(transition_raw), map_location="cpu", weights_only=True
    )
    _check_transition_binding(raw_trace_value, source, launch, transition)
    optimizer_value = torch.load(
        io.BytesIO(optimizer_raw), map_location="cpu", weights_only=True
    )
    _check_optimizer_binding(
        optimizer_value,
        source,
        inventory["transition.pt"]["sha256"],
        len(optimizer_raw),
    )
    require(
        metadata.get("protocol") == optimizer_trace.PROTOCOL
        and metadata.get("source") == source
        and metadata.get("binding") == optimizer_value.get("binding")
        and metadata.get("learner_update_receipt")
        == optimizer_value.get("learner_update_receipt")
        and metadata["binding"].get("parent_checkpoint_sha256")
        == baseline.CHECKPOINT_SHA256
        and metadata["binding"].get("source_trace_sha256")
        == inventory["transition.pt"]["sha256"]
        and metadata.get("capture_bytes") == len(optimizer_raw)
        and metadata.get("transition_sha256") == inventory["transition.pt"]["sha256"]
        and metadata.get("hook_counts") == {"pre": 20, "post": 20},
        "optimizer metadata binds actual raw capture bytes and source",
    )
    for step in range(optimizer_trace.UPDATE_STEPS + 1):
        name = f"diagnostic-state-step-{step:03d}.pt"
        raw = base.files.file_bytes(root / name, limit=RAW_LIMIT)
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        require(
            type(value) is dict
            and value.get("protocol") == DIAGNOSTIC_PROTOCOL
            and value.get("kind") == "diagnostic-only-not-replayable"
            and value.get("step_index") == step
            and type(value.get("counters")) is dict
            and value["counters"].get("optimizer_steps") == step
            and all(value.get(flag) is False for flag in baseline.FALSE_FLAGS),
            "typed durable diagnostic progression step " + str(step),
        )
    replay = optimizer_trace.replay(
        optimizer_raw,
        inventory["optimizer.pt"]["sha256"],
        transition_raw,
        inventory["transition.pt"]["sha256"],
        raw_parent,
    )
    _check_replay_result(
        replay,
        inventory["transition.pt"]["sha256"],
        inventory["optimizer.pt"]["sha256"],
    )
    require(
        _audit_inventory() == audit_binding
        and prerequisites._context(source) == context
        and service_properties(source, "closeout") == properties
        and not torch.cuda.is_initialized()
        and _rehash(root, PRE_REPORT_FILES) == inventory
        and time.monotonic() - started < CLOSEOUT_SECONDS,
        "pinned prerequisite, unchanged sources/files and bounded CPU closeout",
    )
    window.check()
    idle_after = base.host.wait_idle()
    window.check()
    require(
        time.monotonic() - started < CLOSEOUT_SECONDS
        and _audit_inventory() == audit_binding
        and prerequisites._context(source) == context
        and service_properties(source, "closeout") == properties
        and _rehash(root, PRE_REPORT_FILES) == inventory
        and not torch.cuda.is_initialized(),
        "unchanged bounded source/evidence after final idle gate",
    )
    closed_inventory = dict(inventory)
    closed_inventory["report.json"] = {
        "sha256": sha256(report_raw).hexdigest(),
        "bytes": len(report_raw),
    }
    receipt = dict(
        protocol=PROTOCOL,
        artifact_source=source,
        evaluator_source=source,
        audit_binding=audit_binding,
        launch_sha256=launch_sha256,
        inventory=closed_inventory,
        final_inventory_names=sorted(CLOSEOUT_FILES),
        independent_replay=replay,
        decision="one-update-cpu-optimizer-integration-replayed",
        optimizer_steps=optimizer_trace.UPDATE_STEPS,
        completed_updates=1,
        training_update_performed=True,
        optimizer_integration_only=True,
        execution_admitted=False,
        student_export_available=False,
        physical_resimulation_performed=False,
        elapsed_seconds=float(time.monotonic() - started),
        service_properties=properties,
        lease=lease_declaration(),
        idle_before=idle_before,
        idle_after=idle_after,
        run_report_sha256=closed_inventory["report.json"]["sha256"],
        **baseline.FALSE_FLAGS,
    )
    base.files.write_json(root / "independent-closeout.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("run", "closeout"), required=True)
    parser.add_argument("--launch-sha256")
    args = parser.parse_args(argv)
    result = (
        run(args.source)
        if args.mode == "run"
        else closeout(args.source, args.launch_sha256)
    )
    if args.mode == "run":
        summary = result
    else:
        summary = {
            "protocol": result["protocol"],
            "decision": result["decision"],
            "launch_sha256": result["launch_sha256"],
            "run_report_sha256": result["run_report_sha256"],
            "optimizer_steps": result["optimizer_steps"],
            "independent_replay": result["independent_replay"]["optimizer_replay"],
            "closeout_sha256": base.host.digest(
                output_path(args.source) / "independent-closeout.json"
            ),
        }
    print(
        {"output": str(output_path(args.source)), "mode": args.mode, "result": summary}
    )


if __name__ == "__main__":
    main()
