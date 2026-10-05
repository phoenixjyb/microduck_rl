"""Capped CPU-only addition-order plausibility on the immutable full pair."""

import argparse
import os
import re
import subprocess
import sys
import time

from mjlab_microduck import stance_inertia_order_oracle as oracle
from mjlab_microduck import stance_inertia_component_probe as frozen_component
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck.first_attempt_smoke import canonical, require

base = prior.base
PROTOCOL = "football-b1d-inertia-order-probe-20261005-v1"
PARENT = "65abe930caa16476e4d8ece44f937492da47ce2e"
BASE_SOURCE = "3af51c89ee923b46491a3ea9c861876298b003cf"
FREEZE_SOURCE = "f749649d5925f197ee2a995e76aaa1915854f9f9"
MAP_COMPARISON_SHA = "e1e85c0eb76ee579fd15357f3be098cef951434c1ba83fa6846f487c1435d8e9"
MAP_REPORT_SHA = "e1321c1cc3787240e01039f877032869f735ed5fdbb0e93d34726d4c00077e93"
MAP_INVOCATION = "e161d7339d904a4baaac1be2be74c36a"
MAP_TOPOLOGY_SHA = "5b445215f52d61d10bc892a14b0d0b15e3041ff3a5dfb6c1964f06211d0cb43a"
SMOOTH_SHA = "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
SMOOTH_LEAF = ".venv/lib/python3.12/site-packages/mujoco_warp/_src/smooth.py"
SMOOTH_LIMIT = 1024**2
DIAGNOSIS_SHA = "b87629d6176e133188bbb735fa0125cf04163e0daadf1570941e504200a2c1b2"
RUN_INVOCATION = "33241079aa88426b96368011eb248527"
CAP = 300
MEMORY = 4 * 1024**3
TEST_FILES = frozen_component.TEST_FILES + (
    "test_stance_inertia_order_oracle.py",
    "test_stance_inertia_order_probe.py",
)
EXPECTED_TESTS = 1272  # Owner-frozen complete 41-file suite, zero skips.
OWN = (
    "src/mjlab_microduck/stance_inertia_order_oracle.py",
    "src/mjlab_microduck/stance_inertia_order_probe.py",
    "tests/test_stance_inertia_order_oracle.py",
    "tests/test_stance_inertia_order_probe.py",
    "docs/experiments/2026-10-05-cuda64-no-update-rollout.md",
)
INPUT_FILES = prior.COMPLETE_FILES | {"independent-failure-diagnosis.json"}
FLAGS = {**prior.FLAGS, **oracle.FLAGS}


def unit(source, mode):
    base._hex(source, 40, "new addition-order reader source")
    require(mode in ("tests", "run"), "new CPU diagnostic mode")
    return f"microduck-inertia-orders-{mode}-{source[:12]}.service"


def output_path(source, mode):
    unit(source, mode)
    return (
        base.execution.ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / (f"stance-inertia-orders-{mode}-{source[:12]}")
    )


def source_sync_unit(source):
    base._hex(source, 40, "new addition-order bootstrap source")
    return f"microduck-inertia-orders-sync-{source[:12]}.service"


def source_sync_script(source, *, bundle_sha256):
    """Retarget exactly two frozen template guards, retaining its byte gates."""
    script = prior.source_sync_script(source, bundle_sha256=bundle_sha256)
    substitutions = (
        (
            f'test "$(git rev-parse HEAD)" = {prior.SYNC_FROM_SOURCE}\n',
            f'test "$(git rev-parse HEAD)" = {FREEZE_SOURCE}\n',
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
    stable = (set(prior.OWN_FILES) | set(frozen_component.OWN)) - {OWN[-1]}
    result = {}
    for path in sorted(stable | set(OWN)):
        raw = base._read_file(root / path, base.RAW_LIMIT)
        require(
            raw == git("show", f"{source}:{path}"), "exact new committed leaf " + path
        )
        if path in stable:
            require(
                raw == git("show", f"{FREEZE_SOURCE}:{path}"),
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


def authenticate_map():
    """Authenticate the complete prior map and actual installed source bytes."""
    root = frozen_component.output_path(FREEZE_SOURCE, "run")
    base._exact_inventory(root, {"comparison.json", "report.json"})
    comparison_raw = base._read_file(root / "comparison.json", base.JSON_LIMIT)
    report_raw = base._read_file(root / "report.json", base.JSON_LIMIT)
    require(
        base.digest(comparison_raw) == MAP_COMPARISON_SHA
        and base.digest(report_raw) == MAP_REPORT_SHA,
        "both predeclared whole component-map files before any tensor load",
    )
    comparison = base.parse_json(comparison_raw)
    require(
        comparison.get("source") == PARENT
        and comparison.get("worlds") == 64
        and comparison.get("earliest_persistent_differing_event") == 0
        and comparison.get("compiled_map", {}).get("topology_sha256")
        == MAP_TOPOLOGY_SHA,
        "original full component map and compiled topology stay pinned",
    )
    report = base.parse_json(report_raw)
    require(
        report.get("protocol") == frozen_component.PROTOCOL
        and report.get("source") == FREEZE_SOURCE
        and report.get("mode") == "run"
        and report.get("status") == "passed"
        and report.get("comparison_sha256") == MAP_COMPARISON_SHA
        and report.get("diagnosis_sha256") == DIAGNOSIS_SHA
        and report.get("service_properties", {}).get("InvocationID") == MAP_INVOCATION
        and all(report.get(key) is False for key in frozen_component.FLAGS),
        "original successful diagnostic is not an accepted rollout",
    )
    prior._recorded_properties(report["service_properties"], "diagnose")
    frozen_component.completed(FREEZE_SOURCE, "run", MAP_INVOCATION)
    smooth = base._read_file(base.execution.ROOT / SMOOTH_LEAF, SMOOTH_LIMIT)
    require(
        base.digest(smooth) == SMOOTH_SHA, "actual installed smooth source stays pinned"
    )
    return dict(
        comparison_sha256=MAP_COMPARISON_SHA,
        report_sha256=MAP_REPORT_SHA,
        smooth_sha256=SMOOTH_SHA,
        compiled_topology_sha256=MAP_TOPOLOGY_SHA,
    )


def authenticate_inputs():
    """All 25 evidence files and installed source authenticated before loads."""
    oracle._hidden()
    root, launch, sha, inventory = frozen_component.authenticate_inputs()
    anchors = authenticate_map()
    return root, launch, sha, inventory, anchors


def read_compare():
    oracle._hidden()
    root, launch, sha, inventory, anchors = authenticate_inputs()
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
    require(launch["schedule"]["worlds"] == 64, "unchanged full CUDA64 input protocol")
    result = oracle.compare(
        *traces,
        launch["schedule"],
        [x["archive"]["records"][0] for x in inputs],
        launch["compiled_plant"],
    )
    require(
        result.get("source") == PARENT
        and result.get("worlds") == 64
        and result.get("compiled_topology_sha256") == MAP_TOPOLOGY_SHA
        and all(result.get(key) is False for key in oracle.FLAGS),
        "full non-admitting oracle binds to the original compiled component map",
    )
    after = authenticate_inputs()
    require(
        after[3] == inventory and after[4] == anchors,
        "all 25 original files and installed source unchanged after oracle",
    )
    return result, inventory, anchors


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
                result, inventory, anchors = read_compare()
                report.update(
                    original_files_rehashed=inventory,
                    diagnosis_sha256=DIAGNOSIS_SHA,
                    original_map_and_installed_source=anchors,
                    original_pair_accepted=False,
                    original_pair_error="paired rollout semantic state exactness",
                )
                report["comparison_sha256"] = base.write_json(
                    root / "comparison.json", result
                )
                require(
                    authenticate_inputs()[3:] == (inventory, anchors),
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
