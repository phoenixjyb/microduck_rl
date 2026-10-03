"""Twenty-one held-out CPU disturbance cases on the unchanged D1 parent.

This is a bounded gap-fill diagnostic only. It never trains, retries, resets,
or admits a recovery policy; complete raw captures are retained before scoring.
"""

import argparse
import gc
import io
import json
import os
import time
from hashlib import sha256

import torch

from mjlab_microduck import stance_recovery_broad_receipt_repair as broad_repair
from mjlab_microduck import stance_recovery_campaign_window as window
from mjlab_microduck import stance_recovery_parent as parent
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_schedule_probe as base
from mjlab_microduck import stance_recovery_schedule_trace as evidence
from mjlab_microduck import stance_recovery_terminal_probe as terminal
from mjlab_microduck import stance_recovery_terminal_receipt_repair as terminal_repair
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-cpu-gentle-gap-screen-v1"
MODULE = "mjlab_microduck.stance_recovery_gentle_gap_screen"
ARTIFACT_SOURCE = "3338b5f381e9a9ae39f73240e5ad49b3b1bd547c"
ZERO_CELL = "zero-wrench"
CARDINALS = ("+x", "-x", "+y", "-y")
DIAGONALS = ("++", "+-", "-+", "--")
CARDINAL_CELLS = tuple(
    f"{direction}-2n-10steps-t{onset}"
    for onset in (250, 500, 750)
    for direction in CARDINALS
)
DIAGONAL_CELLS = tuple(
    f"diagonal-{direction}-2n-10steps-t{onset}"
    for onset in (375, 625)
    for direction in DIAGONALS
)
CELL_IDS = (ZERO_CELL,) + CARDINAL_CELLS + DIAGONAL_CELLS
PREFIX_STEP = 250
POLICY_TICKS = 250
PHYSICS_STEPS = 2500
EVALUATION_SEED = 671
SERVICE_SECONDS = 1440
CLOSEOUT_SECONDS = 480
WINDOW_MARGIN_SECONDS = 60
LAUNCH_RESERVE = SERVICE_SECONDS + CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS
COLLECTION_SECONDS = 60
CASE_RESERVE_SECONDS = 80
TIMING_FACTOR = 1.25
PREVIOUS_COLLECTION_SECONDS = 1040.6777
PREVIOUS_REPLAY_SECONDS = 292.1143
PREVIOUS_COLLECTION_SOURCE = "1079a104a9e3bf83d56c0586cd2dd0374990e259"
PREVIOUS_REPLAY_SOURCE = "b1878b8715efbb7ea6e166bc7d345efecf5285de"
PROJECTED_COLLECTION_SECONDS = (
    PREVIOUS_COLLECTION_SECONDS / 25 * len(CELL_IDS) * TIMING_FACTOR
)
PROJECTED_REPLAY_SECONDS = PREVIOUS_REPLAY_SECONDS / 25 * len(CELL_IDS) * TIMING_FACTOR
PARENT_CHECKPOINT_SHA256 = (
    "2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5"
)
PARENT_STATE_SHA256 = "e5f51035fe5886b32295a9f39ce1a803dc16ac8a12bae13d3433b897f2ca8329"
CLOSED_AUDIT_SHA256 = {
    "launch.json": (
        "47fd7b5ea76550367a080604e38c3266547cd46640bddbf165510bf520f3a699",
        40175,
    ),
    "source-inventory.json": (
        "09e5e579051a651c2e6997a94017df34345b890b0ccfdaba1173e0130e3ca10e",
        22760,
    ),
    "receipt.json": (
        "53af66761f38c0612f929e3b0af1e8587956f6e4b3b33378767a7b18d31076a5",
        50061,
    ),
}
OUTPUT_PREFIX = "stance-wsl-cpu-gentle-gap-"
RUN_SERVICE_PREFIX = "microduck-cpu-gentle-gap-run-"
CLOSEOUT_SERVICE_PREFIX = "microduck-cpu-gentle-gap-closeout-"
MEMORY_BYTES = 2 * 1024**3
CPU_QUOTA, NICE, KILL_MODE = "2s", "10", "control-group"
CASE_FILES = {
    f"case-{i}{suffix}"
    for i in range(len(CELL_IDS))
    for suffix in (".pt", ".json", "-replay.json")
}
PREPARE_FILES = {"checkpoint.pt", "launch.json", "cpu-qualification.json"} | CASE_FILES
COMPLETE_FILES = PREPARE_FILES | {"report.json"}


def declarations(source):
    """Return the exact ordered 21-cell held-out timing matrix."""
    base.files.hex_id(source, 40)
    rows = [
        schedule.declaration(source, "timing", "held-out", [cell]) for cell in CELL_IDS
    ]
    require(
        all(
            row["worlds"] == 1
            and row["stage"] == "timing"
            and row["split"] == "held-out"
            for row in rows
        ),
        "one-world held-out timing declarations",
    )
    return rows


def output_path(source):
    base.files.hex_id(source, 40)
    require(source != ARTIFACT_SOURCE, "fresh gentle-gap evaluator source")
    return base.host.ROOT / "artifacts/evaluations" / (OUTPUT_PREFIX + source[:12])


def service_name(source, mode):
    output_path(source)
    require(mode in ("run", "closeout"), "declared gentle-gap CPU service mode")
    prefix = RUN_SERVICE_PREFIX if mode == "run" else CLOSEOUT_SERVICE_PREFIX
    return f"{prefix}{source[:12]}.service"


def _check_service_properties(props, mode, *, expected_pid=None):
    require(mode in ("run", "closeout"), "typed gentle-gap service mode")
    seconds = SERVICE_SECONDS if mode == "run" else CLOSEOUT_SECONDS
    require(
        type(props) is dict
        and set(props)
        == {
            "MainPID",
            "ActiveState",
            "RuntimeMaxUSec",
            "MemoryMax",
            "CPUQuotaPerSecUSec",
            "Nice",
            "KillMode",
        }
        and type(props["MainPID"]) is str
        and props["MainPID"].isdecimal()
        and int(props["MainPID"]) > 0
        and (expected_pid is None or props["MainPID"] == expected_pid)
        and props["ActiveState"] == "active"
        and props["RuntimeMaxUSec"] == f"{seconds // 60}min"
        and props["MemoryMax"] == str(MEMORY_BYTES)
        and props["CPUQuotaPerSecUSec"] == CPU_QUOTA
        and props["Nice"] == NICE
        and props["KillMode"] == KILL_MODE,
        "exact independently capped gentle-gap CPU service",
    )
    return True


def service_properties(source, mode):
    require(mode in ("run", "closeout"), "typed gentle-gap service mode")
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
    _check_service_properties(props, mode, expected_pid=str(os.getpid()))
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
        "only the owned gentle-gap Duck service is running",
    )
    return props


def _closed_parent_context(root=None):
    """Read the pinned 3338 audit receipt and return its evaluator context."""
    from pathlib import Path

    root = terminal_repair.output_path(ARTIFACT_SOURCE) if root is None else Path(root)
    require(
        {path.name for path in root.iterdir()} == set(CLOSED_AUDIT_SHA256),
        "exact three-file closed 3338 terminal audit inventory",
    )
    parsed = {}
    for name, (digest, byte_count) in CLOSED_AUDIT_SHA256.items():
        raw = base.files.file_bytes(root / name, limit=2 * 1024**2)
        require(
            len(raw) == byte_count and sha256(raw).hexdigest() == digest,
            "pinned closed 3338 terminal audit bytes: " + name,
        )
        parsed[name] = base.files.parse(raw)
    receipt = parsed["receipt.json"]
    context_binding = receipt.get("context_binding")
    require(
        type(context_binding) is dict
        and set(context_binding) == {"original_context", "evaluator_context"}
        and receipt.get("protocol") == terminal_repair.PROTOCOL
        and receipt.get("artifact_source") == terminal_repair.ARTIFACT_SOURCE
        and receipt.get("evaluator_source") == ARTIFACT_SOURCE
        and receipt.get("decision") == "cpu-natural-full-timeout-and-reset-replayed"
        and receipt.get("status") == "complete"
        and context_binding["original_context"].get("source_identity", {}).get("source")
        == terminal_repair.ARTIFACT_SOURCE
        and receipt.get("original_full_timeout_reset_qualified") is True
        and receipt.get("original_selective_reset_qualified") is False
        and receipt.get("audit_simulator_resets") == 0
        and receipt.get("recollection_performed") is False
        and receipt.get("optimizer_steps") == 0
        and receipt.get("training_update_performed") is False
        and receipt.get("cuda_initialized") is False
        and receipt.get("whole_trajectory_physics_resimulated") is False
        and receipt.get("thermal_model_applied") is False
        and all(receipt.get(key) is False for key in base.baseline.FALSE_FLAGS),
        "closed 3338 receipt is complete, saved-record-only and non-admitting",
    )
    context = context_binding["evaluator_context"]
    launch = parsed["launch.json"]
    require(
        type(context) is dict
        and context.get("source_identity", {}).get("source") == ARTIFACT_SOURCE
        and launch.get("protocol") == terminal_repair.PROTOCOL
        and launch.get("artifact_source")
        == receipt.get("artifact_source")
        == terminal_repair.ARTIFACT_SOURCE
        and launch.get("evaluator_source")
        == receipt.get("evaluator_source")
        == ARTIFACT_SOURCE
        and launch.get("evaluator_context") == context
        and launch.get("original_source_context") == context_binding["original_context"]
        and launch.get("source_inventory")
        == receipt.get("source_inventory")
        == parsed["source-inventory.json"]
        and launch.get("failure") == receipt.get("failure")
        and launch.get("original_inventory") == receipt.get("original_inventory")
        and launch.get("service_properties") == receipt.get("service_properties")
        and launch.get("service_seconds") == terminal_repair.SERVICE_SECONDS
        and launch.get("launch_reserve_seconds")
        == terminal_repair.LAUNCH_RESERVE_SECONDS
        and launch.get("memory_bytes") == MEMORY_BYTES
        and launch.get("cpu_quota") == CPU_QUOTA
        and launch.get("nice") == NICE
        and launch.get("kill_mode") == KILL_MODE
        and launch.get("optimizer_steps") == 0
        and launch.get("recollection_performed") is False
        and launch.get("audit_simulator_resets") == 0
        and all(launch.get(key) is False for key in base.baseline.FALSE_FLAGS)
        and receipt.get("output_inventory")
        == {
            name: {"sha256": item[0], "bytes": item[1]}
            for name, item in CLOSED_AUDIT_SHA256.items()
            if name != "receipt.json"
        }
        and receipt.get("final_inventory_names") == sorted(CLOSED_AUDIT_SHA256),
        "closed 3338 launch/context/output inventory binding",
    )
    terminal_repair._recorded_properties(receipt["service_properties"])
    terminal_repair._check_replay(receipt["independent_replay"])
    return context


def _terminal_prerequisites(source):
    """Authenticate current ancestry and the fifth retained failed service."""
    fresh = terminal._fresh(source)
    source_inventory = terminal_repair._source_inventory()
    failure = terminal_repair._read_failure()
    failed_service = terminal_repair._old_service_state()
    failure_binding = terminal_repair._failure_binding(failure, failed_service)
    return {
        "fresh": fresh,
        "source_inventory": source_inventory,
        "failure_binding": failure_binding,
    }


def _context(source):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden gentle-gap CPU screen process",
    )
    identity = base.host.identity(source)
    cpu_profile = base.profile.checked_receipt()
    services = base.protected_state()
    require(
        all(value == "inactive" for value in services.values()),
        "protected services remain inactive",
    )
    return {
        "source_identity": identity,
        "cpu_math_profile": cpu_profile,
        "preserved_filmbrain": base.retained.d0.filmbrain_state(),
        "protected_services": services,
    }


def _postcheck(source, context, service, mode):
    current = _context(source)
    expected = {
        key: value
        for key, value in context.items()
        if key != "terminal_launch_failure_binding"
    }
    require(
        current == expected
        and terminal._launch_failure_binding()
        == context["terminal_launch_failure_binding"]
        and terminal_repair._old_service_state()
        == terminal_repair.FAILURE_SERVICE_STATE,
        "unchanged source/profile/services and retained failed-service bindings",
    )
    require(
        service_properties(source, mode) == service,
        "only the exact owned gentle-gap Duck service remains active",
    )
    return True


def _timing_projection():
    require(
        PROJECTED_COLLECTION_SECONDS < SERVICE_SECONDS
        and PROJECTED_REPLAY_SECONDS < CLOSEOUT_SECONDS,
        "measured collection and replay projections fit their service caps",
    )
    return {
        "previous_collection_source": PREVIOUS_COLLECTION_SOURCE,
        "previous_replay_source": PREVIOUS_REPLAY_SOURCE,
        "prior_cases": 25,
        "planned_cases": len(CELL_IDS),
        "factor": TIMING_FACTOR,
        "previous_collection_seconds": PREVIOUS_COLLECTION_SECONDS,
        "projected_collection_seconds": PROJECTED_COLLECTION_SECONDS,
        "previous_replay_seconds": PREVIOUS_REPLAY_SECONDS,
        "projected_replay_seconds": PROJECTED_REPLAY_SECONDS,
        "run_service_seconds": SERVICE_SECONDS,
        "closeout_service_seconds": CLOSEOUT_SECONDS,
    }


def _screen_inputs(scores, prefixes):
    require(
        type(scores) is list
        and len(scores) == len(CELL_IDS)
        and type(prefixes) is list
        and len(prefixes) == len(CELL_IDS),
        "exact ordered 21-case screen inputs",
    )
    valid_cases, candidate_deficits = [], []
    for index, (cell, score) in enumerate(zip(CELL_IDS, scores)):
        require(
            type(score) is dict
            and score.get("cell") == cell
            and score.get("protocol") == evidence.PROTOCOL
            and score.get("strict_actor_restore") is True
            and score.get("actor_replay_max_abs_error") == 0.0
            and score.get("whole_trajectory_physics_resimulated") is False
            and score.get("thermal_model_applied") is False
            and all(score.get(key) is False for key in base.baseline.FALSE_FLAGS)
            and type(score.get("numerical_diagnostic")) is dict
            and type(score.get("collection")) is dict
            and type(score.get("pulse")) is dict,
            "ordered independently scored gentle-gap row " + str(index),
        )
        numeric, collection, pulse = (
            score["numerical_diagnostic"],
            score["collection"],
            score["pulse"],
        )
        require(
            type(numeric.get("candidate_pass")) is bool
            and type(numeric.get("complete_first_attempt")) is bool
            and type(collection.get("policy_ticks")) is int
            and 1 <= collection["policy_ticks"] <= POLICY_TICKS
            and type(collection.get("elapsed_seconds")) is float
            and 0 < collection["elapsed_seconds"] < COLLECTION_SECONDS
            and collection.get("stop_reason")
            in (
                "policy-tick-limit",
                "all-first-attempts-complete",
                "wall-budget-exhausted",
            )
            and type(pulse.get("checked_physics_steps")) is int
            and 0 < pulse["checked_physics_steps"] <= PHYSICS_STEPS,
            "typed bounded full-attempt gentle-gap score",
        )
        pulse_ok = all(
            pulse.get(key) is True
            for key in (
                "complete_pulse_delivery",
                "complete_phase_checks",
                "recorded_phase_checks_valid",
                "exact_full_force_arrays_checked",
                "unforced_post_arrays_checked",
            )
        )
        terminal_complete = (
            numeric["complete_first_attempt"] is True
            and collection["stop_reason"] == "all-first-attempts-complete"
        )
        duration_complete = (
            collection["policy_ticks"] == POLICY_TICKS
            and pulse["checked_physics_steps"] == PHYSICS_STEPS
            and collection["stop_reason"]
            in ("policy-tick-limit", "all-first-attempts-complete")
        )
        valid_cases.append(bool(pulse_ok and (terminal_complete or duration_complete)))
        if index and numeric["candidate_pass"] is False:
            candidate_deficits.append(cell)
    prefix_valid = all(
        type(item) is str
        and len(item) == 64
        and all(char in "0123456789abcdef" for char in item)
        for item in prefixes
    )
    prefixes_match = bool(prefix_valid and len(set(prefixes)) == 1)
    zero_control_valid = scores[0]["numerical_diagnostic"]["candidate_pass"] is True
    complete = all(valid_cases)
    if not zero_control_valid or not prefixes_match or not complete:
        decision = "cpu-gentle-gap-inconclusive"
    elif candidate_deficits:
        decision = "cpu-gentle-gap-candidate-deficit"
    else:
        decision = "cpu-gentle-gap-no-deficit"
    return dict(
        protocol=PROTOCOL,
        decision=decision,
        candidate_deficit_cells=candidate_deficits,
        zero_control_valid=zero_control_valid,
        complete_declared_force_phases=complete,
        prefixes_identical=prefixes_match,
        complete_cases=sum(valid_cases),
        cases_expected=len(CELL_IDS),
        full_duration_gate_is_acceptance=False,
        promotion_authorized=False,
        optimizer_steps=0,
        simulator_resets=0,
        auto_reset=False,
        **base.baseline.FALSE_FLAGS,
    )


def _capture_metadata(source, index, declaration, raw, value, elapsed):
    return dict(
        protocol=PROTOCOL,
        source=source,
        index=index,
        cell=declaration["cell_ids"][0],
        declaration=declaration,
        capture_sha256=sha256(raw).hexdigest(),
        capture_bytes=len(raw),
        collection=value["collection"],
        collection_wall_seconds=elapsed,
        optimizer_steps=0,
        simulator_resets=0,
        auto_reset=False,
        **base.baseline.FALSE_FLAGS,
    )


def _case_prefix(raw, digest, checkpoint_raw, declaration, compiled_plant):
    score = evidence.verify(raw, digest, checkpoint_raw, declaration, compiled_plant)
    prefix = None
    if score["pulse"]["checked_physics_steps"] > PREFIX_STEP:
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        try:
            prefix = evidence.prefix_hash(value, PREFIX_STEP)
        except ValueError:
            prefix = None
    return score, prefix


def _context_record(source, initial_context, audit_context, prerequisites):
    fresh = prerequisites["fresh"]
    expected_context = dict(initial_context)
    expected_context["terminal_launch_failure_binding"] = fresh[0][
        "terminal_launch_failure_binding"
    ]
    require(
        broad_repair._check_source_context(
            audit_context,
            expected_context,
            artifact_source=ARTIFACT_SOURCE,
            evaluator_source=source,
        ),
        "fresh source differs from exact closed audit context only by source",
    )
    require(
        fresh[0] == expected_context,
        "terminal-probe fresh context agrees with gentle-gap context",
    )
    return {
        "source_identity": initial_context["source_identity"],
        "cpu_math_profile": initial_context["cpu_math_profile"],
        "preserved_filmbrain": initial_context["preserved_filmbrain"],
        "protected_services": initial_context["protected_services"],
        "terminal_launch_failure_binding": fresh[0]["terminal_launch_failure_binding"],
        "terminal_prerequisites": {
            "prior_receipt": fresh[1],
            "closed_audit_binding": fresh[3],
            "closed_update_binding": fresh[4],
            "source_inventory": prerequisites["source_inventory"],
            "failure_binding": prerequisites["failure_binding"],
        },
    }


def run(source):
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        return _run_leased(source, idle_before)


def _lease_declaration():
    return {
        "lock_path": str(base.files.LOCK),
        "mechanism": "advisory-flock-exclusive-nonblocking",
        "scope": "unchanged-parent-gentle-gap-screen",
        "idle_gate": "two-idle-samples-before-and-after",
        "cuda_learner": False,
    }


def _run_leased(source, idle_before):
    started = time.monotonic()
    window.check(reserve_seconds=LAUNCH_RESERVE)
    require(
        len(CELL_IDS) == 21
        and CELL_IDS == (ZERO_CELL,) + CARDINAL_CELLS + DIAGONAL_CELLS
        and base.evidence.EVALUATION_SEED == EVALUATION_SEED
        and PARENT_CHECKPOINT_SHA256 == base.baseline.CHECKPOINT_SHA256,
        "fixed 21-case matrix, seed and unchanged D1 parent",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden CPU gentle-gap screen entry",
    )
    properties = service_properties(source, "run")
    initial_context = _context(source)
    audit_context = _closed_parent_context()
    fresh = _terminal_prerequisites(source)
    terminal_context = fresh["fresh"][0]
    initial_context = dict(
        initial_context,
        terminal_launch_failure_binding=terminal_context[
            "terminal_launch_failure_binding"
        ],
    )
    require(
        terminal_context == initial_context,
        "terminal_probe._fresh current context agrees with the gentle-gap screen",
    )
    broad_repair._check_source_context(
        audit_context,
        initial_context,
        artifact_source=ARTIFACT_SOURCE,
        evaluator_source=source,
    )
    require(
        sha256(fresh["fresh"][2]).hexdigest() == PARENT_CHECKPOINT_SHA256
        and parent.PARENT_STATE_SHA256 == PARENT_STATE_SHA256,
        "exact unchanged D1 parent checkpoint and restored state",
    )
    checkpoint_raw = fresh["fresh"][2]
    prerequisite_record = _context_record(source, initial_context, audit_context, fresh)
    rows = declarations(source)
    require(
        rows
        == [
            schedule.declaration(source, "timing", "held-out", [cell])
            for cell in CELL_IDS
        ],
        "exact schedule declaration binding",
    )
    root = base.files.native._plain_path(output_path(source))
    root.mkdir(exist_ok=False)
    base.retained.write_capture(root / "checkpoint.pt", checkpoint_raw)
    launch_path = root / "launch.json"
    launch = None
    report = dict(
        protocol=PROTOCOL,
        source=source,
        decision="cpu-gentle-gap-incomplete",
        optimizer_steps=0,
        simulator_resets=0,
        auto_reset=False,
        cuda_initialized=False,
        **base.baseline.FALSE_FLAGS,
    )
    scores, prefixes, captures = [], [], []
    try:
        from mjlab_microduck.stance_recovery_schedule_runtime import (
            ScheduledRecoveryRuntime,
        )

        for index, declaration in enumerate(rows):
            window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
            require(
                time.monotonic() - started < SERVICE_SECONDS - CASE_RESERVE_SECONDS,
                "reserve next 60-second CPU case and retention",
            )
            _postcheck(source, initial_context, properties, "run")
            base.seed_cpu()
            env = ScheduledRecoveryRuntime(declaration, device="cpu")
            require(
                type(env) is ScheduledRecoveryRuntime
                and env.n == 1
                and env.live.all()
                and not env.steps.any()
                and str(env.device) == "cpu"
                and not env.wp_device.is_cuda
                and env.forward_graph is None
                and env.solved_field_check == "packed",
                "fresh one-world eager-packed CPU scheduled attempt",
            )
            if launch is None:
                launch = dict(
                    protocol=PROTOCOL,
                    source=source,
                    **prerequisite_record,
                    idle_before=idle_before,
                    lease=_lease_declaration(),
                    parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                    parent_state_sha256=PARENT_STATE_SHA256,
                    parent_identity=parent.expected_identity(),
                    campaign_window=window.declaration(),
                    declarations=rows,
                    cell_ids=list(CELL_IDS),
                    evaluation_seed=EVALUATION_SEED,
                    worlds=1,
                    policy_ticks=POLICY_TICKS,
                    physics_steps=PHYSICS_STEPS,
                    simulation_seconds=5.0,
                    prefix_step=PREFIX_STEP,
                    collection_seconds=COLLECTION_SECONDS,
                    service_seconds=SERVICE_SECONDS,
                    closeout_seconds=CLOSEOUT_SECONDS,
                    launch_reserve_seconds=LAUNCH_RESERVE,
                    per_case_reserve_seconds=CASE_RESERVE_SECONDS,
                    timing_projection=_timing_projection(),
                    service_properties=properties,
                    compiled_plant=env.binding,
                    optimizer_steps=0,
                    simulator_resets=0,
                    auto_reset=False,
                    **base.baseline.FALSE_FLAGS,
                )
                base.files.write_json(launch_path, launch)
            require(
                env.binding == launch["compiled_plant"],
                "same compiled plant across all fresh held-out cases",
            )
            case_started = time.monotonic()
            try:
                value = evidence.collect(
                    env,
                    checkpoint_raw,
                    deadline_monotonic=min(
                        case_started + COLLECTION_SECONDS,
                        started + SERVICE_SECONDS - CASE_RESERVE_SECONDS,
                    ),
                    policy_tick_limit=POLICY_TICKS,
                )
            except Exception as exc:
                base.files.write_json(
                    root / f"case-{index}-failure.json",
                    dict(
                        protocol=PROTOCOL,
                        source=source,
                        index=index,
                        cell=declaration["cell_ids"][0],
                        stage="collection",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        current_case_trace_retained=False,
                        **base.baseline.FALSE_FLAGS,
                    ),
                )
                raise
            elapsed = float(time.monotonic() - case_started)
            raw = evidence.encode(value)
            capture = _capture_metadata(source, index, declaration, raw, value, elapsed)
            base.retained.write_capture(root / f"case-{index}.pt", raw)
            base.files.write_json(root / f"case-{index}.json", capture)
            del env, value
            gc.collect()
            # Whole raw retention and digest precede every independent judgement.
            score, prefix = _case_prefix(
                raw,
                capture["capture_sha256"],
                checkpoint_raw,
                declaration,
                launch["compiled_plant"],
            )
            base.files.write_json(root / f"case-{index}-replay.json", score)
            scores.append(score)
            prefixes.append(prefix)
            captures.append(capture)
            del raw
            gc.collect()
            window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
            _postcheck(source, initial_context, properties, "run")
        screening = _screen_inputs(scores, prefixes)
        _postcheck(source, initial_context, properties, "run")
        require(
            not torch.cuda.is_initialized()
            and time.monotonic() - started < SERVICE_SECONDS,
            "unchanged bounded gentle-gap CPU closeout",
        )
        qualification = dict(
            protocol=PROTOCOL,
            source=source,
            captures=captures,
            scores=scores,
            prefixes=prefixes,
            screening=screening,
            elapsed_seconds=float(time.monotonic() - started),
            cpu_initialized_only=True,
            optimizer_steps=0,
            simulator_resets=0,
            auto_reset=False,
            **base.baseline.FALSE_FLAGS,
        )
        base.files.write_json(root / "cpu-qualification.json", qualification)
        require(launch is not None, "actual compiled plant launch is retained")
        report.update(
            decision=screening["decision"],
            screening=screening,
            launch_sha256=base.host.digest(launch_path),
            qualification_sha256=base.host.digest(root / "cpu-qualification.json"),
            service_properties=properties,
            elapsed_seconds=float(time.monotonic() - started),
            genuine_cpu_first_attempts=len(CELL_IDS),
            source_unchanged=True,
            filmbrain_unchanged=True,
            protected_services_inactive=True,
        )
    except Exception as exc:
        report.update(
            error_type=type(exc).__name__,
            error=str(exc),
            error_notes=getattr(exc, "__notes__", []),
        )
        raise
    finally:
        report["idle_before"] = idle_before
        try:
            report["idle_after"] = base.host.wait_idle()
            if report["decision"] != "cpu-gentle-gap-incomplete":
                _postcheck(source, initial_context, properties, "run")
                require(
                    _terminal_prerequisites(source) == fresh
                    and _closed_parent_context() == audit_context,
                    "all pinned prerequisites unchanged after final idle gate",
                )
                require(
                    time.monotonic() - started < SERVICE_SECONDS,
                    "run remains bounded after final idle and prerequisite checks",
                )
                window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
        except BaseException as exc:
            report.update(
                decision="cpu-gentle-gap-incomplete",
                post_idle_check_passed=False,
                post_idle_error_type=type(exc).__name__,
                post_idle_error=str(exc)[:1000],
            )
            for key in (
                "source_unchanged",
                "filmbrain_unchanged",
                "protected_services_inactive",
            ):
                report.pop(key, None)
            raise
        finally:
            report["elapsed_seconds"] = float(time.monotonic() - started)
            report["files"] = {
                path.name: base.host.digest(path)
                for path in sorted(root.iterdir())
                if path.is_file()
            }
            base.files.write_json(root / "report.json", report)
    return {
        "output": str(root),
        "launch_sha256": report["launch_sha256"],
        "decision": report["screening"]["decision"],
    }


def checked(source, launch_sha256):
    base.files.hex_id(launch_sha256, 64)
    root = output_path(source)
    launch_raw = base.files.file_bytes(root / "launch.json")
    require(
        sha256(launch_raw).hexdigest() == launch_sha256,
        "whole launch bytes before parse",
    )
    launch = base.files.parse(launch_raw)
    current_base = _context(source)
    audit_context = _closed_parent_context()
    fresh = _terminal_prerequisites(source)
    current = dict(
        current_base,
        terminal_launch_failure_binding=fresh["fresh"][0][
            "terminal_launch_failure_binding"
        ],
    )
    require(
        launch["protocol"] == PROTOCOL
        and launch["source"] == source
        and fresh["fresh"][0] == current
        and {key: launch[key] for key in current_base} == current_base
        and launch["terminal_launch_failure_binding"]
        == current["terminal_launch_failure_binding"]
        and launch["terminal_prerequisites"]
        == {
            "prior_receipt": fresh["fresh"][1],
            "closed_audit_binding": fresh["fresh"][3],
            "closed_update_binding": fresh["fresh"][4],
            "source_inventory": fresh["source_inventory"],
            "failure_binding": fresh["failure_binding"],
        },
        "current unchanged source context and authenticated terminal prerequisites",
    )
    broad_repair._check_source_context(
        audit_context, current, artifact_source=ARTIFACT_SOURCE, evaluator_source=source
    )
    require(
        launch["campaign_window"] == window.declaration()
        and launch["declarations"] == declarations(source)
        and launch["cell_ids"] == list(CELL_IDS)
        and launch["evaluation_seed"] == EVALUATION_SEED
        and launch["worlds"] == 1
        and launch["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256
        and launch["parent_state_sha256"] == PARENT_STATE_SHA256
        and launch["parent_identity"] == parent.expected_identity()
        and launch["policy_ticks"] == POLICY_TICKS
        and launch["physics_steps"] == PHYSICS_STEPS
        and launch["simulation_seconds"] == 5.0
        and launch["prefix_step"] == PREFIX_STEP
        and launch["collection_seconds"] == COLLECTION_SECONDS
        and launch["service_seconds"] == SERVICE_SECONDS
        and launch["closeout_seconds"] == CLOSEOUT_SECONDS
        and launch["launch_reserve_seconds"] == LAUNCH_RESERVE
        and launch["per_case_reserve_seconds"] == CASE_RESERVE_SECONDS
        and launch["timing_projection"] == _timing_projection()
        and launch["lease"] == _lease_declaration()
        and launch["optimizer_steps"] == launch["simulator_resets"] == 0
        and launch["auto_reset"] is False
        and all(launch[key] is False for key in base.baseline.FALSE_FLAGS),
        "exact immutable parent, schedule and no-optimizer launch binding",
    )
    _check_service_properties(launch["service_properties"], "run")
    checkpoint_raw = base.files.file_bytes(
        root / "checkpoint.pt", limit=base.retained.evaluation.checkpoint.LIMIT
    )
    require(
        sha256(checkpoint_raw).hexdigest() == PARENT_CHECKPOINT_SHA256
        and checkpoint_raw == fresh["fresh"][2],
        "retained checkpoint is the exact unchanged parent byte stream",
    )
    qualification_raw = base.files.file_bytes(root / "cpu-qualification.json")
    qualification = base.files.parse(qualification_raw)
    require(
        qualification["protocol"] == PROTOCOL
        and qualification["source"] == source
        and len(qualification["captures"])
        == len(qualification["scores"])
        == len(qualification["prefixes"])
        == len(CELL_IDS)
        and type(qualification["elapsed_seconds"]) is float
        and 0 < qualification["elapsed_seconds"] < SERVICE_SECONDS
        and qualification["cpu_initialized_only"] is True
        and qualification["optimizer_steps"] == qualification["simulator_resets"] == 0
        and qualification["auto_reset"] is False
        and all(qualification[key] is False for key in base.baseline.FALSE_FLAGS),
        "complete bounded gentle-gap CPU qualification receipt",
    )
    report = base.files.parse(base.files.file_bytes(root / "report.json"))
    require(
        type(report.get("idle_before")) is dict
        and type(report.get("idle_after")) is dict
        and launch["idle_before"] == report["idle_before"]
        and type(report.get("elapsed_seconds")) is float
        and 0 < report["elapsed_seconds"] < SERVICE_SECONDS
        and report.get("genuine_cpu_first_attempts") == len(CELL_IDS)
        and report.get("source_unchanged") is True
        and report.get("filmbrain_unchanged") is True
        and report.get("protected_services_inactive") is True
        and report.get("service_properties") == launch["service_properties"]
        and "error_type" not in report
        and "post_idle_error_type" not in report,
        "run receipt retains both GPU idle gates",
    )
    return launch, checkpoint_raw, qualification, sha256(qualification_raw).hexdigest()


def replay(source, launch, checkpoint_raw, qualification):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden CPU gentle-gap replay",
    )
    root = output_path(source)
    scores, prefixes = [], []
    for index, (cell, declaration) in enumerate(zip(CELL_IDS, launch["declarations"])):
        raw = base.files.file_bytes(root / f"case-{index}.pt", limit=evidence.LIMIT)
        capture = base.files.parse(base.files.file_bytes(root / f"case-{index}.json"))
        expected_capture = qualification["captures"][index]
        require(
            capture == expected_capture
            and capture["protocol"] == PROTOCOL
            and capture["source"] == source
            and capture["index"] == index
            and capture["cell"] == cell
            and capture["declaration"] == declaration
            and capture["capture_bytes"] == len(raw)
            and sha256(raw).hexdigest() == capture["capture_sha256"]
            and capture["optimizer_steps"] == capture["simulator_resets"] == 0
            and capture["auto_reset"] is False
            and all(capture[key] is False for key in base.baseline.FALSE_FLAGS),
            "exact retained raw gentle-gap case metadata",
        )
        score, prefix = _case_prefix(
            raw,
            capture["capture_sha256"],
            checkpoint_raw,
            declaration,
            launch["compiled_plant"],
        )
        retained_score = base.files.parse(
            base.files.file_bytes(root / f"case-{index}-replay.json")
        )
        require(
            score == retained_score == qualification["scores"][index]
            and score["collection"] == capture["collection"],
            "fresh independent raw/force/actor replay equals retained score",
        )
        scores.append(score)
        prefixes.append(prefix)
        del raw
        gc.collect()
    screening = _screen_inputs(scores, prefixes)
    require(
        prefixes == qualification["prefixes"]
        and screening == qualification["screening"],
        "deterministic common-prefix and gentle-gap screen decision",
    )
    return {"scores": scores, "prefixes": prefixes, "screening": screening}


def closeout(source, launch_sha256):
    with base.files.gpu_lease():
        idle_before = base.host.wait_idle()
        return _closeout_leased(source, launch_sha256, idle_before)


def _closeout_leased(source, launch_sha256, idle_before):
    started = time.monotonic()
    window.check(reserve_seconds=CLOSEOUT_SECONDS + WINDOW_MARGIN_SECONDS)
    properties = service_properties(source, "closeout")
    launch, checkpoint_raw, qualification, qualification_sha = checked(
        source, launch_sha256
    )
    root = output_path(source)
    report_raw = base.files.file_bytes(root / "report.json")
    report = base.files.parse(report_raw)
    require(
        report.get("protocol") == PROTOCOL
        and report.get("source") == source
        and report.get("launch_sha256") == launch_sha256
        and report.get("qualification_sha256") == qualification_sha
        and report.get("optimizer_steps") == report.get("simulator_resets") == 0
        and report.get("auto_reset") is False
        and report.get("cuda_initialized") is False
        and all(report.get(key) is False for key in base.baseline.FALSE_FLAGS)
        and report.get("decision") == qualification["screening"]["decision"]
        and report.get("screening") == qualification["screening"]
        and report["screening"]["decision"]
        in (
            "cpu-gentle-gap-no-deficit",
            "cpu-gentle-gap-candidate-deficit",
            "cpu-gentle-gap-inconclusive",
        ),
        "complete or inconclusive gentle-gap report with no promotion",
    )
    require(
        set(report["files"]) == PREPARE_FILES
        and {path.name for path in root.iterdir()} == COMPLETE_FILES,
        "exact static gentle-gap run inventory",
    )
    inventory = {}
    for name, digest in report["files"].items():
        path = root / name
        require(
            type(name) is str
            and "/" not in name
            and name not in (".", "..")
            and base.host.digest(path) == digest,
            "whole retained gentle-gap file hash " + str(name),
        )
        inventory[name] = {"sha256": digest, "bytes": path.stat().st_size}
    rescored = replay(source, launch, checkpoint_raw, qualification)
    require(
        rescored["screening"] == report["screening"],
        "fresh whole-capture CPU rescore and deterministic decision",
    )
    current = _context(source)
    require(
        current == {key: launch[key] for key in current}
        and terminal._launch_failure_binding()
        == launch["terminal_launch_failure_binding"]
        and terminal_repair._old_service_state()
        == terminal_repair.FAILURE_SERVICE_STATE
        and service_properties(source, "closeout") == properties
        and time.monotonic() - started < CLOSEOUT_SECONDS,
        "unchanged bounded independent CPU closeout",
    )
    window.check()
    inventory["report.json"] = {
        "sha256": sha256(report_raw).hexdigest(),
        "bytes": len(report_raw),
    }
    idle_after = base.host.wait_idle()
    require(
        checked(source, launch_sha256)
        == (launch, checkpoint_raw, qualification, qualification_sha)
        and {path.name for path in root.iterdir()} == COMPLETE_FILES
        and all(
            path.stat().st_size == inventory[path.name]["bytes"]
            and base.host.digest(path) == inventory[path.name]["sha256"]
            for path in root.iterdir()
        )
        and service_properties(source, "closeout") == properties
        and not torch.cuda.is_initialized()
        and time.monotonic() - started < CLOSEOUT_SECONDS,
        "entire run inventory and prerequisites unchanged after final idle gate",
    )
    window.check(reserve_seconds=WINDOW_MARGIN_SECONDS)
    receipt = closeout_result(
        source,
        launch_sha256,
        report_raw,
        rescored,
        inventory,
        properties,
        float(time.monotonic() - started),
        idle_before,
        idle_after,
    )
    base.files.write_json(root / "independent-closeout.json", receipt)
    return receipt


def closeout_result(
    source,
    launch_sha256,
    report_raw,
    rescored,
    inventory,
    properties,
    elapsed,
    idle_before,
    idle_after,
):
    """Pure receipt constructor. Live/native validation belongs to closeout()."""
    base.files.hex_id(source, 40)
    base.files.hex_id(launch_sha256, 64)
    require(
        type(report_raw) is bytes
        and type(rescored) is dict
        and type(rescored.get("screening")) is dict
        and type(inventory) is dict
        and type(properties) is dict
        and type(elapsed) is float
        and type(idle_before) is dict
        and type(idle_after) is dict
        and 0 < elapsed < CLOSEOUT_SECONDS,
        "typed gentle-gap closeout constructor inputs",
    )
    _check_service_properties(properties, "closeout")
    false_flags = dict(base.baseline.FALSE_FLAGS)
    fields = {
        "protocol": PROTOCOL,
        "source": source,
        "launch_sha256": launch_sha256,
        "report_sha256": sha256(report_raw).hexdigest(),
        "decision": rescored["screening"]["decision"],
        "screening": rescored["screening"],
        "files_rehashed": inventory,
        "cases_checked": len(CELL_IDS),
        "whole_cpu_rescore_identical": True,
        "cuda_initialized": False,
        "whole_trajectory_physics_resimulated": False,
        "thermal_model_applied": False,
        "service_properties": properties,
        "optimizer_steps": 0,
        "simulator_resets": 0,
        "auto_reset": False,
        "elapsed_seconds": elapsed,
        "idle_before": idle_before,
        "idle_after": idle_after,
        "lease": _lease_declaration(),
    }
    require(
        not (set(fields) & set(false_flags)),
        "receipt constructor has one owner for each false-flag field",
    )
    return {**fields, **false_flags}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "closeout"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--launch-sha256")
    args = parser.parse_args(argv)
    if args.mode == "run":
        result = run(args.source)
    else:
        require(args.launch_sha256 is not None, "closeout requires exact launch digest")
        result = closeout(args.source, args.launch_sha256)
    print(
        json.dumps(
            {
                "output": str(output_path(args.source)),
                "source": args.source,
                "mode": args.mode,
                "decision": result["decision"],
                "receipt_sha256": base.host.digest(
                    output_path(args.source)
                    / (
                        "report.json"
                        if args.mode == "run"
                        else "independent-closeout.json"
                    )
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
