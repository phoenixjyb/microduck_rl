"""Capped CPU-only component diagnosis of the immutable full inertia pair."""

import argparse
import os
import re
import subprocess
import sys
import time

from mjlab_microduck import stance_inertia_component_map as components
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
PROTOCOL = "football-b1d-inertia-component-probe-20261005-v1"
PARENT = "65abe930caa16476e4d8ece44f937492da47ce2e"
BASE_SOURCE = "ddd25a43d2714fb52404ecb40a285f819bbb3e46"
DIAGNOSIS_SHA = "b87629d6176e133188bbb735fa0125cf04163e0daadf1570941e504200a2c1b2"
RUN_INVOCATION = "33241079aa88426b96368011eb248527"
CAP = 300
MEMORY = 4 * 1024**3
TEST_FILES = prior.TEST_FILES + (
    "test_stance_inertia_component_map.py",
    "test_stance_inertia_component_probe.py",
)
EXPECTED_TESTS = 1182  # Owner-frozen complete 39-file suite, zero skips.
OWN = (
    "src/mjlab_microduck/stance_inertia_component_map.py",
    "src/mjlab_microduck/stance_inertia_component_probe.py",
    "tests/test_stance_inertia_component_map.py",
    "tests/test_stance_inertia_component_probe.py",
    "docs/experiments/2026-10-05-cuda64-no-update-rollout.md",
)
INPUT_FILES = prior.COMPLETE_FILES | {"independent-failure-diagnosis.json"}
FLAGS = {**prior.FLAGS, **components.FLAGS}


def unit(source, mode):
    base._hex(source, 40, "new constraint reader source")
    require(mode in ("tests", "run"), "new CPU diagnostic mode")
    return f"microduck-inertia-components-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    return (
        base.execution.ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / (f"stance-inertia-components-{mode}-{source[:12]}")
    )


def source_sync_unit(source):
    base._hex(source, 40, "new component bootstrap source")
    return f"microduck-inertia-components-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    """Retarget exactly two frozen template guards, retaining its byte gates."""
    script = prior.source_sync_script(source, bundle_sha256=bundle_sha256)
    substitutions = (
        (
            f'test "$(git rev-parse HEAD)" = {prior.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {PARENT}\n',
        ),
        (
            f'if test "$duck_sync_running" != {prior.source_sync_unit(source)}; then\n',
            f'if test "$duck_sync_running" != {source_sync_unit(source)}; then\n',
        ),
    )
    for old, new in substitutions:
        require(script.count(old) == 1, "exact frozen bootstrap guard occurrence")
        script = script.replace(old, new, 1)
    return script


def source_binding(source):
    root = base.execution.ROOT

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=15
        ).stdout

    require(
        git("rev-parse", "HEAD").decode().strip() == source
        and git("branch", "--show-current").decode().strip()
        == "feat/athletics-obstacle-curriculum"
        and not git("status", "--porcelain"),
        "clean exact new diagnostic branch",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE, source],
        cwd=root,
        check=True,
        timeout=15,
    )
    changed = set(git("diff", "--name-only", BASE_SOURCE, source).decode().splitlines())
    require(
        changed <= set(OWN),
        "only newly declared CPU diagnostic and documentation changes",
    )
    # The old rollout implementation and its whole-byte readers are unchanged.
    stable = set(prior.OWN_FILES) - {OWN[-1]}
    result = {}
    for path in sorted(stable | set(OWN)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(
            raw == git("show", f"{source}:{path}"), "exact new committed leaf " + path
        )
        if path in stable:
            require(
                raw == git("show", f"{PARENT}:{path}"),
                "unchanged original reader " + path,
            )
        result[path] = base.digest(raw)
    return result


def properties(source, mode):
    name = unit(source, mode)
    keys = (
        "MainPID",
        "ActiveState",
        "RuntimeMaxUSec",
        "MemoryMax",
        "CPUQuotaPerSecUSec",
        "Nice",
        "KillMode",
        "InvocationID",
    )
    values = {
        key: base.host.read("systemctl", "--user", "show", name, "-p", key, "--value")
        for key in keys
    }
    prior._recorded_properties(values, "diagnose")  # Same declared 300-s/4-GiB caps.
    require(values["MainPID"] == str(os.getpid()), "CPU diagnostic owns capped service")
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
        "sole Duck CPU service",
    )
    return values


def completed(source, mode, invocation):
    base._hex(invocation, 32, "original CPU test invocation")
    values = {
        key: base.host.read(
            "systemctl", "--user", "show", unit(source, mode), "-p", key, "--value"
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
        values["InvocationID"] in ("", invocation)
        and {k: v for k, v in values.items() if k != "InvocationID"}
        == dict(
            MainPID="0",
            ActiveState="inactive",
            NRestarts="0",
            ExecMainStatus="0",
            Result="success",
        ),
        "successful terminal original CPU test service",
    )
    return values


def authenticate_inputs():
    """Pinned whole diagnosis anchors all original bytes before deserialization."""
    components._hidden()
    root = prior.output_path(PARENT)
    base._exact_inventory(root, INPUT_FILES)
    raw = base._read_file(root / "independent-failure-diagnosis.json", base.JSON_LIMIT)
    require(
        base.digest(raw) == DIAGNOSIS_SHA,
        "predeclared whole independent diagnosis hash",
    )
    diagnosis = base.parse_json(raw)
    require(
        diagnosis.get("protocol") == prior.PROTOCOL + ":failure-diagnosis"
        and diagnosis.get("source") == PARENT
        and diagnosis.get("status") == "failed-pair-independently-diagnosed"
        and diagnosis.get("pair_error") == "paired rollout semantic state exactness"
        and diagnosis.get("pair_accepted") is False
        and all(diagnosis.get(k) is False for k in prior.FLAGS),
        "original non-admitting diagnosis",
    )
    inventory = prior._inventory(root, prior.COMPLETE_FILES)
    require(
        inventory == diagnosis["files_rehashed"],
        "all 22 original files unchanged before any load",
    )
    launch_raw = base._read_file(root / "launch.json", base.JSON_LIMIT)
    sha = base.digest(launch_raw)
    require(sha == diagnosis["launch_sha256"], "original whole launch binding")
    launch = prior._read_launch(PARENT, sha)
    require(
        launch["service_properties"]["InvocationID"] == RUN_INVOCATION,
        "predeclared original failed run identity",
    )
    prior._failed_terminal(PARENT, RUN_INVOCATION)
    return root, launch, sha, inventory


def read_compare():
    components._hidden()
    root, launch, sha, inventory = authenticate_inputs()
    inputs, _scores, traces = prior._score_inputs(root, PARENT, launch, sha)
    pair_error = None
    try:
        prior._strict_pair(inputs, traces)
    except ValueError as error:
        pair_error = str(error)
    require(
        pair_error == "paired rollout semantic state exactness",
        "original strict pair failure reproduced again",
    )
    result = components.compare(
        *traces,
        launch["schedule"],
        [x["archive"]["records"][0] for x in inputs],
        launch["compiled_plant"],
    )
    require(
        prior._inventory(root, prior.COMPLETE_FILES) == inventory
        and base.digest(
            base._read_file(
                root / "independent-failure-diagnosis.json", base.JSON_LIMIT
            )
        )
        == DIAGNOSIS_SHA,
        "original 23 evidence files unchanged after comparison",
    )
    prior._failed_terminal(PARENT, RUN_INVOCATION)
    return result, inventory


def _read_tests(source, binding):
    root = output_path(source, "tests")
    base._exact_inventory(root, {"report.json", "pytest.log"})
    raw = base._read_file(root / "report.json", base.JSON_LIMIT)
    result = base.parse_json(raw)
    require(
        result.get("protocol") == PROTOCOL
        and result.get("source") == source
        and result.get("mode") == "tests"
        and result.get("status") == "passed"
        and result.get("source_binding") == binding
        and type(result.get("passed")) is int
        and result.get("passed") == EXPECTED_TESTS
        and EXPECTED_TESTS > 0
        and all(result.get(k) is False for k in FLAGS),
        "frozen exact new test receipt",
    )
    log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
    require(base.digest(log) == result["pytest_sha256"], "whole new CPU pytest log")
    prior._recorded_properties(result["service_properties"], "diagnose")
    completed(source, "tests", result["service_properties"]["InvocationID"])
    return base.digest(raw)


def execute(source, mode):
    prior._hidden()
    prior.check_window(reserve_seconds=(2 * CAP + 60 if mode == "tests" else CAP + 60))
    service = properties(source, mode)
    binding = source_binding(source)
    host_identity = base.host.identity(source)
    root = output_path(source, mode)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(
        protocol=PROTOCOL,
        source=source,
        parent=PARENT,
        mode=mode,
        status="failed-retained",
        service_properties=service,
        source_binding=binding,
        host_identity=host_identity,
        **FLAGS,
    )
    try:
        with base.gap.base.files.gpu_lease():
            idle = base.gpu_idle_gate.wait_idle()
            filmbrain = base.gap.base.retained.d0.filmbrain_state()
            protected = base.gap.base.protected_state()
            require(
                bool(protected)
                and all(value == "inactive" for value in protected.values()),
                "protected AI Mission services remain inactive",
            )
            report["protected_services"] = protected
            if mode == "tests":
                require(
                    EXPECTED_TESTS > 0, "owner-frozen CPU suite before native checks"
                )
                prior._run_process(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *["tests/" + p for p in TEST_FILES],
                    ],
                    root / "pytest.log",
                    CAP - 45,
                    env=dict(os.environ),
                )
                log = base._read_file(root / "pytest.log", base.LOG_LIMIT)
                totals = re.findall(rb"(?:^|\n)(\d+) passed in ([^\n]+)", log)
                require(
                    len(totals) == 1
                    and int(totals[0][0]) == EXPECTED_TESTS
                    and not re.search(
                        rb"\d+ (?:skipped|failed|xfailed|xpassed|error)", log
                    ),
                    "exact complete new CPU suite, no skips",
                )
                report.update(passed=EXPECTED_TESTS, pytest_sha256=base.digest(log))
            else:
                report["tests_receipt_sha256"] = _read_tests(source, binding)
                result, inventory = read_compare()
                report.update(
                    original_files_rehashed=inventory, diagnosis_sha256=DIAGNOSIS_SHA
                )
                report["comparison_sha256"] = base.write_json(
                    root / "comparison.json", result
                )
                require(
                    authenticate_inputs()[3] == inventory,
                    "original evidence and failed invocation still unchanged",
                )
            idle_after = base.gpu_idle_gate.wait_idle()
            require(
                source_binding(source) == binding
                and base.host.identity(source) == host_identity
                and properties(source, mode) == service
                and base.gap.base.retained.d0.filmbrain_state() == filmbrain
                and base.gap.base.protected_state() == protected
                and time.monotonic() - started < CAP,
                "unchanged bounded CPU diagnostic source and context",
            )
            report.update(
                status="passed",
                idle_before=idle,
                idle_after=idle_after,
                filmbrain=filmbrain,
                protected_services_after=protected,
            )
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = float(time.monotonic() - started)
        base.write_json(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tests", "run"))
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    result = execute(args.source, args.mode)
    print(
        canonical(
            {
                k: result[k]
                for k in ("protocol", "source", "mode", "status", "elapsed_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
