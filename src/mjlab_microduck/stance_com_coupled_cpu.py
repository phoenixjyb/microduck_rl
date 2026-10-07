"""Fresh October 7 CPU prerequisite; no CUDA execution or training admission."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_com_entry_probe as retained
from mjlab_microduck.stance_com_entry_receiver import FLAGS, canonical

PROTOCOL = "microduck-com-coupled-cpu-oct7-v1"
MODULE = "mjlab_microduck.stance_com_coupled_cpu"
BASE = "8b032a2779d8bd743fab02bbc596db9bf6116987"
ROOT = retained.ROOT
CUTOFF = datetime(2026, 10, 7, 5, 0, tzinfo=timezone.utc).timestamp()
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 300, 240, 300, 60
CAPS = dict(retained.TEST_SERVICE_CAPS)
OWN = retained.OWN | {
    "src/mjlab_microduck/stance_com_coupled_cpu.py",
    "tests/test_stance_com_coupled_cpu.py",
    "docs/experiments/2026-10-07-com-coupled-forward-diagnostic.md",
}
# Updated only after collecting the exact declared suite; never inferred from exit 0.
EXPECTED_TESTS = 1923


def check_window(reserve, now=None):
    now = time.time() if now is None else now
    retained.need(
        type(reserve) is int
        and reserve > 0
        and type(now) in (float, int)
        and 0 <= now < float("inf")
        and now + reserve < CUTOFF,
        "fresh October 7 complete job and closeout before 13:00",
    )


def unit(source):
    retained.need(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
        "exact CPU source",
    )
    return f"microduck-com-coupled-cpu-{source[:12]}.service"


def output(source):
    unit(source)
    return ROOT / "artifacts/tools" / f"com-coupled-cpu-{source[:12]}"


def source_binding(source):
    unit(source)
    retained.need(
        Path.cwd().resolve() == ROOT
        and Path(__file__).resolve().parents[2] == ROOT
        and retained.read("git", "rev-parse", "HEAD") == source
        and retained.read("git", "branch", "--show-current") == retained.BRANCH
        and retained.read("git", "remote", "get-url", "origin") == retained.ORIGIN
        and not retained.read("git", "status", "--porcelain"),
        "fresh clean exact imported CPU source",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    retained.need(
        set(retained.read("git", "diff", "--name-only", BASE, source).splitlines())
        <= OWN,
        "fresh October 7 source fence",
    )
    leaves = {}
    for path in retained.read(
        "git", "ls-files", "src", "tests", "pyproject.toml", "uv.lock", *sorted(OWN)
    ).splitlines():
        raw = retained.bounded(ROOT / path, 4 * 1024**2, allow_empty=True)
        expected = subprocess.check_output(
            ["git", "show", f"{source}:{path}"], timeout=15
        )
        retained.need(raw == expected, "whole fresh committed source leaf " + path)
        leaves[path] = sha256(raw).hexdigest()
    return dict(
        source=source,
        branch=retained.BRANCH,
        tree=retained.read("git", "rev-parse", "HEAD^{tree}"),
        leaf_count=len(leaves),
        leaves_sha256=sha256(canonical(leaves)).hexdigest(),
    )


def service(source, live_pid):
    values = dict(
        row.split("=", 1)
        for row in retained.read(
            "systemctl", "--user", "show", unit(source)
        ).splitlines()
        if "=" in row
    )
    retained.need(
        type(live_pid) is int
        and live_pid > 0
        and all(values.get(k) == v for k, v in CAPS.items())
        and values.get("MainPID") == str(live_pid)
        and values.get("ActiveState") == "active"
        and values.get("SubState") == "running"
        and re.fullmatch(r"[0-9a-f]{32}", values.get("InvocationID", "")),
        "fresh live capped CPU service",
    )
    return {
        k: values[k]
        for k in (*CAPS, "MainPID", "InvocationID", "ActiveState", "SubState")
    }


def test_files():
    return retained.test_files() + ["tests/test_stance_com_coupled_cpu.py"]


def checked_junit(raw):
    retained.need(type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded whole JUnit")
    suites = list(ET.fromstring(raw).iter("testsuite"))
    retained.need(
        suites
        and sum(int(s.get("tests", "-1")) for s in suites) == EXPECTED_TESTS
        and all(
            s.get(k) == "0" for s in suites for k in ("errors", "failures", "skipped")
        ),
        "exact fresh CPU test count and zero skips/errors/failures",
    )
    return EXPECTED_TESTS


def run(source):
    check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    environment = retained.test_environment()
    binding, live, packages = (
        source_binding(source),
        service(source, os.getpid()),
        retained.packages(),
    )
    predecessors = retained.predecessors()
    fixtures = retained.test_fixture_binding(ROOT)
    retained.need(
        fixtures == retained.test_fixture_binding(retained.OLD_ROOT),
        "pinned fixture copies",
    )
    before = retained.host()
    retained.need(
        not before["processes"] and before["used_mib"] <= 1024,
        "idle before hidden CPU tests",
    )
    directory = output(source)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with (directory / "pytest.log").open("xb") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--junitxml=" + str(directory / "junit.xml"),
                    *test_files(),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=CHILD_SECONDS,
            )
            stream.flush()
            os.fsync(stream.fileno())
        retained.need(result.returncode == 0, "fresh full CPU suite passed")
        xml = retained.bounded(directory / "junit.xml", 1024**2)
        count = checked_junit(xml)
        fd = os.open(directory / "junit.xml", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        after = retained.host()
        retained.need(not after["processes"], "no GPU workload admitted by CPU tests")
        retained.need(
            source_binding(source) == binding
            and retained.packages() == packages
            and retained.predecessors() == predecessors
            and retained.test_environment() == environment
            and retained.test_fixture_binding(ROOT) == fixtures
            and retained.test_fixture_binding(retained.OLD_ROOT) == fixtures
            and service(source, os.getpid()) == live,
            "unchanged CPU prerequisites at closure",
        )
        check_window(CLOSEOUT_SECONDS + MARGIN)
        receipt = dict(
            protocol=PROTOCOL,
            source_binding=binding,
            service_properties=live,
            packages=packages,
            test_environment=environment,
            tests=count,
            test_files=test_files(),
            retained_test_fixtures=fixtures,
            predecessors=predecessors,
            junit_sha256=sha256(xml).hexdigest(),
            log_sha256=sha256(
                retained.bounded(directory / "pytest.log", 1024**2)
            ).hexdigest(),
            host_before=before,
            host_after=after,
            cuda_execution_admitted=False,
            flags=dict(FLAGS),
        )
        retained.write(directory / "receipt.json", canonical(receipt), 128 * 1024)
        print(canonical(receipt).decode(), flush=True)
    except BaseException as error:
        retained.write(
            directory / "failure.json",
            canonical(
                dict(
                    protocol=PROTOCOL + ":failure",
                    source=source,
                    exception_type=type(error).__name__,
                    flags=dict(FLAGS),
                )
            ),
            4096,
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    run(parser.parse_args().source)


if __name__ == "__main__":
    main()
