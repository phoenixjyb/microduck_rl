"""Authenticate the closed CUDA64 preparation before a later sibling probe.

This CUDA-hidden reader does not reopen its directory, allocate CUDA, sample,
create a simulator or admit a learner. Only an explicitly separate evaluator
source ID may differ from the authenticated preparation context.
"""

import os
from copy import deepcopy

import torch

from mjlab_microduck import stance_recovery_cuda_policy_probe as probe
from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = "football-b1d-closed-cuda64-preparation-reader-v1"
ARTIFACT_SOURCE = "48653c58121dec9a2b6ae2db059fc4d0911410cf"
EXTERNAL_SHA256 = "25a0b5fdde74d41578a9ce1c74506f706ddba36085b3b54f779071a8bf09a6b2"
LAUNCH_SHA256 = "cb164a7171d08e0ddac7f7c424aef7b429798ad1ca15406d059be108550222e9"
CLOSEOUT_SHA256 = "9b66dd80be8fd13c175c1385b16bf72b0a83018db13826b8fac54874e02d0e05"
REPORT_SHA256 = "ba94a173940499e2af1cc98cb5688453f826f812d44957f27109153cc9f7e72f"
INVOCATIONS = {
    "source-sync": "d5dd92a752144364909210c838533c70",
    "preflight": "d87e5eb667bf472ea7d95dd8766833a4",
    "tests": "559c118fd8c74a7a8810457fb670d018",
    "run": "3e53413286e54d47bc0a0ad208b5c54b",
    "closeout": "3da42837e1524a5db342d4b5c648ec27",
}


def check_source_delta(original, current, *, evaluator_source):
    """Require full equality except the two declared evaluator source IDs.

    This pure comparison is not provenance authentication. Native ``checked``
    first authenticates whole retained bytes and actual current prerequisites.
    """
    probe._hex(evaluator_source, 40, "separate evaluator source")
    require(evaluator_source != ARTIFACT_SOURCE, "separate new evaluator source")
    require(
        type(original) is dict
        and type(current) is dict
        and set(original)
        == set(current)
        == {
            "source",
            "gap_inventory",
            "gap_receipts",
            "current_context",
            "preflight_failure_binding",
        },
        "exact current and preparation prerequisite binding schema",
    )
    adjusted = deepcopy(original)
    for binding, expected in ((adjusted, ARTIFACT_SOURCE), (current, evaluator_source)):
        require(
            type(binding.get("source")) is dict
            and binding["source"].get("source") == expected
            and type(binding.get("current_context")) is dict
            and type(binding["current_context"].get("source_identity")) is dict
            and binding["current_context"]["source_identity"].get("source") == expected,
            "exact original and separate evaluator source identities",
        )
    adjusted["source"]["source"] = evaluator_source
    adjusted["current_context"]["source_identity"]["source"] = evaluator_source
    require(adjusted == current, "only evaluator source IDs may differ")
    return True


def _validate_external(record):
    require(
        type(record) is dict
        and record.get("protocol") == "cuda64-preparation-external-verification-v1"
        and record.get("source") == ARTIFACT_SOURCE
        and record.get("decision") == "cuda64-preparation-closed-non-admitting"
        and record.get("native_device_preparation_verified") is True
        and record.get("cuda_math_replayed") is False
        and record.get("simulation_steps") == 0
        and record.get("action_samples") == 0
        and record.get("optimizer_steps") == 0
        and record.get("native_tests_passed") == 492
        and type(record.get("native_tests_passed")) is int
        and record.get("service_invocations") == INVOCATIONS
        and type(record.get("inventory")) is dict
        and set(record["inventory"]) == probe.CLOSEOUT_FILES
        and all(record.get(key) is False for key in probe.PREPARATION_FALSE_FLAGS),
        "exact independently closed non-admitting preparation verification",
    )
    return True


def _completed_tool(label):
    units = {
        "source-sync": "microduck-cuda64-source-sync-48653c58121d.service",
        "tests": "microduck-cuda64-tests-48653c58121d.service",
    }
    require(label in units, "one declared original native tool service")
    state = {
        key: probe.host.read(
            "systemctl", "--user", "show", units[label], "-p", key, "--value"
        )
        for key in (
            "MainPID",
            "ActiveState",
            "NRestarts",
            "ExecMainStatus",
            "Result",
            "InvocationID",
        )
    }
    require(
        {k: v for k, v in state.items() if k != "InvocationID"}
        == {
            "MainPID": "0",
            "ActiveState": "inactive",
            "NRestarts": "0",
            "ExecMainStatus": "0",
            "Result": "success",
        }
        and state["InvocationID"] in ("", INVOCATIONS[label]),
        "original successful terminal native tool service",
    )
    return state


def checked(source, *, lease_fd):
    """Rehash old captures and authenticate fresh exact-source CPU context.

    Caller owns the shared lease and must separately qualify new source/tests,
    declare job caps and perform idle/admission checks. ``raw_parent`` is returned
    separately from JSON-safe bindings; never serialize it into a launch.
    """
    probe.training_smoke.inherited_lease(lease_fd)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized()
        and probe.execution.PROFILE.get("name") == probe.execution.WSL,
        "CUDA-hidden exact WSL closed-preparation reader",
    )
    probe._hex(source, 40, "separate evaluator source")
    root = probe.output_path(ARTIFACT_SOURCE)
    external_root = (
        probe.execution.ROOT / "artifacts/tools/cuda64-verified-48653c58121d"
    )
    external_raw = probe._read_file(external_root / "receipt.json", probe.JSON_LIMIT)
    require(
        probe.digest(external_raw) == EXTERNAL_SHA256,
        "whole original external verification",
    )
    external = probe.parse_json(external_raw)
    _validate_external(external)
    probe._exact_inventory(
        external_root,
        {"receipt.json", *(label + ".journal.jsonl" for label in INVOCATIONS)},
    )
    require(
        set(external.get("journals", {}))
        == {label + ".journal.jsonl" for label in INVOCATIONS},
        "exact five original whole journals",
    )
    for name, pin in external["journals"].items():
        raw = probe._read_file(external_root / name, probe.LOG_LIMIT)
        require(
            {"bytes": len(raw), "sha256": probe.digest(raw)} == pin,
            "whole original retained journal " + name,
        )
    expected_tools = {
        "source-sync/receipt.json": "artifacts/tools/cuda64-source-sync-48653c58121d/receipt.json",
        "preflight/receipt.json": "artifacts/tools/cuda64-prerequisites-48653c58121d/receipt.json",
        "preflight/cpu-parent-receipt.json": "artifacts/tools/cuda64-prerequisites-48653c58121d/cpu-parent-receipt.json",
        "tests/receipt.json": "artifacts/tools/cuda64-tests-48653c58121d/receipt.json",
        "tests/pytest.log": "artifacts/tools/cuda64-tests-48653c58121d/pytest.log",
    }
    require(
        set(external.get("tool_receipts", {})) == set(expected_tools),
        "exact prerequisite records",
    )
    tool_records = {}
    for name, relative in expected_tools.items():
        pin = external["tool_receipts"][name]
        require(pin.get("path") == relative, "fixed original tool receipt path")
        raw = probe._read_file(probe.execution.ROOT / relative, probe.JSON_LIMIT)
        require(
            pin == {"path": relative, "bytes": len(raw), "sha256": probe.digest(raw)},
            "whole original tool receipt " + name,
        )
        if name.endswith(".json"):
            tool_records[name] = probe.parse_json(raw)
    inventory = probe._complete_inventory(root, probe.CLOSEOUT_FILES)
    require(inventory == external["inventory"], "whole ten-file preparation inventory")
    _, launch, launch_raw, cpu_receipt = probe._check_launch(
        ARTIFACT_SOURCE, LAUNCH_SHA256
    )
    require(
        probe.digest(launch_raw) == LAUNCH_SHA256, "original source-bound launch bytes"
    )
    report_raw = probe._read_file(root / "report.json", probe.JSON_LIMIT)
    close_raw = probe._read_file(root / "independent-closeout.json", probe.JSON_LIMIT)
    require(
        probe.digest(report_raw) == REPORT_SHA256
        and probe.digest(close_raw) == CLOSEOUT_SHA256,
        "original report and independent-closeout whole bytes",
    )
    report, close = probe.parse_json(report_raw), probe.parse_json(close_raw)
    probe._check_live_gpu_samples(report.get("children"))
    require(
        close.get("files_rehashed")
        == {n: p for n, p in inventory.items() if n != "independent-closeout.json"}
        and close.get("whole_cpu_payload_rescore_identical") is True
        and close.get("cuda_math_replayed") is False,
        "original complete CPU consistency closeout",
    )
    scores = []
    for seed in probe.SEEDS:
        summary = probe.parse_json(
            probe._read_file(root / f"seed-{seed}.json", probe.JSON_LIMIT)
        )
        raw = probe._read_file(root / f"seed-{seed}.pt", probe.RAW_LIMIT)
        scores.append(
            probe._score_payload(raw, summary, launch, LAUNCH_SHA256, cpu_receipt, seed)
        )
    require(
        scores == close.get("scores") == external.get("scores"),
        "fresh whole raw preparation scores",
    )
    run_terminal = probe._completed_service(
        ARTIFACT_SOURCE, "supervise", INVOCATIONS["run"]
    )
    close_terminal = probe._completed_service(
        ARTIFACT_SOURCE, "closeout", INVOCATIONS["closeout"]
    )
    tool_terminals = {
        label: _completed_tool(label) for label in ("source-sync", "tests")
    }
    old_preflight = probe._closed_preflight(
        ARTIFACT_SOURCE, {"launch_binding": launch["native_prerequisites"]}
    )
    require(
        old_preflight
        == launch.get("native_preflight_binding")
        == external.get("native_preflight_binding"),
        "original completed actual CPU preflight remains bound",
    )
    tests = tool_records["tests/receipt.json"]
    require(
        tests.get("status") == "native-source-tests-passed"
        and tests.get("source") == ARTIFACT_SOURCE
        and tests.get("tests_passed") == 492
        and tests.get("native_preflight_binding") == old_preflight
        and tests.get("native_prerequisites") == launch["native_prerequisites"],
        "original native source tests and authenticated prerequisites",
    )
    current = probe._native_prerequisites(source)
    current_binding = probe._json_safe(current["launch_binding"])
    check_source_delta(
        launch["native_prerequisites"], current_binding, evaluator_source=source
    )
    probe.training_smoke.inherited_lease(lease_fd)
    require(
        not torch.cuda.is_initialized(),
        "closed-preparation reading never initializes CUDA",
    )
    return {
        "raw_parent": current["raw_parent"],
        "current_prerequisites": current,
        "launch_binding": {
            "protocol": PROTOCOL,
            "artifact_source": ARTIFACT_SOURCE,
            "evaluator_source": source,
            "external_verification_sha256": EXTERNAL_SHA256,
            "launch_sha256": LAUNCH_SHA256,
            "report_sha256": REPORT_SHA256,
            "closeout_sha256": CLOSEOUT_SHA256,
            "inventory": inventory,
            "scores": scores,
            "original_run_terminal": run_terminal,
            "original_closeout_terminal": close_terminal,
            "original_tool_terminals": tool_terminals,
            "original_preflight": old_preflight,
            "current_native_prerequisites": current_binding,
            "cuda_math_replayed": False,
            **probe.PREPARATION_FALSE_FLAGS,
        },
    }
