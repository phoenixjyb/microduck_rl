"""Fresh source-bound RNE observation supervisor, never training admission."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from mjlab_microduck import stance_com_coupled_probe as coupled
from mjlab_microduck import stance_rne_entry_receiver as receiver
from mjlab_microduck import stance_com_entry_probe as old
from mjlab_microduck.stance_com_entry_receiver import FLAGS, SERVICE_CAPS, canonical

PROTOCOL = "microduck-rne-entry-probe-oct7-v1"
MODULE = "mjlab_microduck.stance_rne_entry_probe"
BASE = "98e3b7c1fe26dca8095a3a209961f649daf796d3"
ROOT = old.ROOT
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 300, 240, 300, 60
LEAF_COUNT = 649
OWN = {
    f"src/mjlab_microduck/stance_rne_entry_{name}.py"
    for name in ("capture", "probe", "receiver")
}
OWN |= {
    f"tests/test_stance_rne_entry_{name}.py"
    for name in ("capture", "probe", "receiver")
}
OWN.add("docs/experiments/2026-10-07-rne-actual-entry-diagnostic.md")
COM_SOURCE = "6cd03291581684680e6ab3875a96f88cf8dd83ca"
COM_PROOF = {
    "inventory.json": "1048cd38bd8c68c34bba18ea2af65a499d91e07c92ce60fc090593a313036809",
    "terminal.json": "d1cd1c224625fb6171b0da71b24eeec2ec9396ebbb2c51223783e6837efa993f",
    "receiver.json": "51ec8165d49ed398c2704bbe2ce65ce0b9f3e0778dfcd6056c7375d139e113d0",
}
TEMPORAL_SHA256 = "2a871d75faff2d1da3d78ca644fb24332c92cf1ca5830301d5bd77659c4c3e08"


def check_window(reserve, now=None):
    coupled.check_window(reserve, now)


def unit(source, mode):
    old.need(
        type(source) is str
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and type(mode) is str
        and mode in ("tests", "run"),
        "literal fresh RNE unit identity",
    )
    return f"microduck-rne-entry-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"rne-entry-{mode}-{source[:12]}"
    )


def source_binding(source):
    unit(source, "run")
    old.need(
        Path.cwd().resolve() == ROOT
        and Path(__file__).resolve().parents[2] == ROOT
        and old.read("git", "rev-parse", "HEAD") == source
        and old.read("git", "branch", "--show-current") == old.BRANCH
        and old.read("git", "remote", "get-url", "origin") == old.ORIGIN
        and not old.read("git", "status", "--porcelain"),
        "exact clean fresh RNE source",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    old.need(
        set(old.read("git", "diff", "--name-only", BASE, source).splitlines()) <= OWN,
        "seven-path fresh RNE source fence",
    )
    leaves = {}
    for path in old.read(
        "git",
        "ls-files",
        "src",
        "tests",
        "pyproject.toml",
        "uv.lock",
        *sorted(coupled.OWN | OWN),
    ).splitlines():
        raw = old.bounded(ROOT / path, 4 * 1024**2, allow_empty=True)
        old.need(
            raw
            == subprocess.check_output(["git", "show", f"{source}:{path}"], timeout=15),
            "whole committed RNE leaf " + path,
        )
        leaves[path] = sha256(raw).hexdigest()
    old.need(len(leaves) == LEAF_COUNT, "exact whole RNE leaf inventory")
    return dict(
        source=source,
        branch=old.BRANCH,
        tree=old.read("git", "rev-parse", "HEAD^{tree}"),
        leaf_count=len(leaves),
        leaves_sha256=sha256(canonical(leaves)).hexdigest(),
    )


def properties(source, mode):
    return dict(
        row.split("=", 1)
        for row in old.read(
            "systemctl", "--user", "show", unit(source, mode)
        ).splitlines()
        if "=" in row
    )


def service(source, mode, pid):
    values = properties(source, mode)
    caps = old.TEST_SERVICE_CAPS if mode == "tests" else SERVICE_CAPS
    old.need(
        type(pid) is int
        and pid > 0
        and all(values.get(k) == v for k, v in caps.items())
        and values["MainPID"] == str(pid)
        and values["ActiveState"] == "active"
        and values["SubState"] == "running"
        and re.fullmatch(r"[0-9a-f]{32}", values["InvocationID"]),
        "actual retained capped RNE invocation",
    )
    return {
        k: values[k]
        for k in (*caps, "MainPID", "InvocationID", "ActiveState", "SubState")
    }


def retained_com():
    root = ROOT / "artifacts/tools/com-coupled-run-closeout-6cd032915816"
    raw = {name: old.bounded(root / name, 128 * 1024) for name in COM_PROOF}
    old.need(
        all(
            sha256(raw[name]).hexdigest() == digest
            for name, digest in COM_PROOF.items()
        ),
        "whole immutable paired CoM closeout before JSON",
    )
    old.need(
        json.loads(raw["receiver.json"])["decision"] == "fresh-coupled-control-exact",
        "retained positive paired decision, not temporal admission",
    )
    temporal = old.bounded(
        ROOT / "artifacts/tools/com-coupled-temporal-6cd032915816/native-analysis.json",
        128 * 1024,
    )
    old.need(
        len(temporal) == 53898 and sha256(temporal).hexdigest() == TEMPORAL_SHA256,
        "whole native independent temporal negative before JSON",
    )
    record = json.loads(temporal)
    old.need(
        record["decision"] == "temporal-serial-negative"
        and record["negative_coordinate_count"] == 100
        and record["all_cases_fixed_inputs_temporal_exact"] is True
        and record["serial_temporal_exact"] is False
        and receiver.old.false_flags(record["flags"]),
        "retained separate temporal negative",
    )
    return dict(
        source=COM_SOURCE,
        paired_hashes=dict(COM_PROOF),
        temporal_sha256=TEMPORAL_SHA256,
        paired_decision="fresh-coupled-control-exact",
        temporal_decision="temporal-serial-negative",
    )


def test_files():
    return (
        coupled.test_files()
        + ["tests/test_stance_com_coupled_idempotence.py"]
        + [
            f"tests/test_stance_rne_entry_{name}.py"
            for name in ("capture", "probe", "receiver")
        ]
    )


def checked_junit(raw):
    old.need(
        type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded complete RNE JUnit"
    )
    suites = list(ET.fromstring(raw).iter("testsuite"))
    old.need(
        receiver.EXPECTED_TESTS > 0
        and suites
        and sum(int(s.get("tests", "-1")) for s in suites) == receiver.EXPECTED_TESTS
        and all(
            s.get(k) == "0" for s in suites for k in ("skipped", "errors", "failures")
        ),
        "exact predeclared RNE CPU suite without omissions",
    )


def run_tests(source, directory, binding, live, packages):
    environment = old.test_environment()
    fixtures = old.test_fixture_binding(ROOT)
    descriptor = coupled.descriptor_fixture()
    old.need(
        fixtures == old.test_fixture_binding(old.OLD_ROOT),
        "unchanged copied CPU fixtures",
    )
    old.need(
        sha256(canonical(test_files())).hexdigest() == receiver.TEST_FILES_SHA256,
        "exact ordered current test list",
    )
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
    old.need(result.returncode == 0, "fresh full RNE CPU suite passed")
    xml = old.bounded(directory / "junit.xml", 1024**2)
    checked_junit(xml)
    fd = os.open(directory / "junit.xml", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    old.need(
        source_binding(source) == binding
        and old.packages() == packages
        and service(source, "tests", os.getpid()) == live
        and coupled.descriptor_fixture() == descriptor
        and fixtures
        == old.test_fixture_binding(ROOT)
        == old.test_fixture_binding(old.OLD_ROOT),
        "closed unchanged RNE CPU source/packages/fixtures/caps",
    )
    coupled.phase_a()
    old.predecessors()
    retained_com()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle host at CPU closure",
    )
    check_window(CLOSEOUT_SECONDS + MARGIN)
    receipt = dict(
        protocol=PROTOCOL + ":tests",
        source_binding=binding,
        packages=packages,
        tests=receiver.EXPECTED_TESTS,
        test_files=test_files(),
        test_environment=environment,
        retained_test_fixtures=fixtures,
        retained_descriptor_fixture=descriptor,
        service_properties=live,
        junit_sha256=sha256(xml).hexdigest(),
        log_sha256=sha256(old.bounded(directory / "pytest.log", 1024**2)).hexdigest(),
        flags=dict(FLAGS),
    )
    old.write(directory / "receipt.json", canonical(receipt), 128 * 1024)


def test_prerequisite(source, binding, packages):
    directory = output(source, "tests")
    raw = old.bounded(directory / "receipt.json", 128 * 1024)
    record = json.loads(raw)
    old.need(
        record["source_binding"] == binding
        and record["packages"] == packages
        and type(record["tests"]) is int
        and record["tests"] == receiver.EXPECTED_TESTS
        and record["test_files"] == test_files()
        and record["test_environment"] == old.TEST_ENV
        and record["protocol"] == PROTOCOL + ":tests"
        and receiver.old.false_flags(record["flags"])
        and record["retained_descriptor_fixture"] == coupled.descriptor_fixture()
        and record["retained_test_fixtures"]
        == old.test_fixture_binding(ROOT)
        == old.test_fixture_binding(old.OLD_ROOT),
        "fresh exact-source current CPU prerequisite",
    )
    xml = old.bounded(directory / "junit.xml", 1024**2)
    checked_junit(xml)
    old.need(
        sha256(xml).hexdigest() == record["junit_sha256"]
        and sha256(old.bounded(directory / "pytest.log", 1024**2)).hexdigest()
        == record["log_sha256"],
        "whole current CPU log and JUnit hashes",
    )
    terminal = dict(
        old.TEST_SERVICE_CAPS,
        MainPID="0",
        Result="success",
        ExecMainStatus="0",
        ActiveState="active",
        SubState="exited",
        InvocationID=record["service_properties"]["InvocationID"],
    )
    values = properties(source, "tests")
    old.need(
        all(values.get(k) == v for k, v in terminal.items()),
        "same completed current CPU invocation",
    )
    return (
        raw,
        terminal,
        dict(
            count=receiver.EXPECTED_TESTS,
            files=len(test_files()),
            source_binding=binding,
            receipt_sha256=sha256(raw).hexdigest(),
            terminal=terminal,
        ),
    )


def child(source, lease_fd, owner_pid, declaration_sha):
    old.need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and type(owner_pid) is int
        and os.getppid() == owner_pid,
        "owned visible native child and actual parent",
    )
    old.inherited_lease(lease_fd)
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    binding, packages, live = (
        source_binding(source),
        old.packages(),
        service(source, "run", owner_pid),
    )
    directory = output(source, "run")
    raw = old.bounded(directory / "declaration.json", 128 * 1024)
    old.need(
        sha256(raw).hexdigest() == declaration_sha,
        "whole declaration before native child JSON",
    )
    declaration = json.loads(raw)
    old.need(
        declaration["source_binding"] == binding
        and declaration["packages"] == packages
        and declaration["service_properties"] == live
        and declaration["protocol"] == PROTOCOL + ":declaration"
        and declaration["source"] == source
        and receiver.old.false_flags(declaration["flags"]),
        "exact native RNE child declaration",
    )
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024, "idle before Torch CUDA"
    )
    import torch
    import warp as wp
    from mjlab_microduck import stance_com_entry_capture as com_capture
    from mjlab_microduck import stance_rne_entry_capture as rne_capture
    from mjlab_microduck import stance_crb_runtime_control as crb
    from mjlab_microduck import stance_plant_evidence as plant
    from mjlab_microduck import stance_recovery_schedule as schedule
    from mjlab_microduck import stance_warp_integrator as integrator
    from mjlab_microduck.stance_recovery_schedule_runtime import (
        ScheduledRecoveryRuntime,
    )
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime
    from mjlab_microduck import stance_com_coupled_frames as frames

    old.need(not torch.cuda.is_initialized(), "no early Torch CUDA initialization")
    torch.manual_seed(673)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(677)
    old.need(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "one visible initialized GPU",
    )
    integration_calls = 0
    original = integrator.EulerCandidateCommit.integrate

    def refuse(*_args, **_kwargs):
        nonlocal integration_calls
        integration_calls += 1
        raise ValueError("integration forbidden in RNE observation")

    integrator.EulerCandidateCommit.integrate = refuse
    try:
        states = {
            f"caller_{dev}_before": coupled._rng_state(torch, dev)
            for dev in ("cpu", "cuda")
        }
        com = com_capture.NativeCoupledComControl(mode="serial")
        control = crb.StanceCrbRuntimeControl(mode="serial", max_forward_calls=2)
        rne = rne_capture.RneEntryCapture()
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(977)
            torch.cuda.manual_seed(983)
            states.update(
                {
                    f"private_{dev}_start": coupled._rng_state(torch, dev)
                    for dev in ("cpu", "cuda")
                }
            )
            with wp.ScopedDevice("cuda:0"), control, com, rne:
                env = ScheduledRecoveryRuntime(
                    schedule.declaration(
                        source, "dose", "training", ["zero-wrench"] * 64
                    ),
                    device="cuda:0",
                    solved_field_check="packed",
                )
                control.bind_runtime(env)
                old.need(
                    type(env) is ScheduledRecoveryRuntime
                    and env._forward.__func__ is WarpStanceRuntime._forward
                    and not {"_forward", "_scheduled_forward"} & set(env.__dict__),
                    "original eager scheduled runtime",
                )
                states.update(
                    {
                        f"private_{dev}_ctor_end": coupled._rng_state(torch, dev)
                        for dev in ("cpu", "cuda")
                    }
                )
                first = coupled._frame(env)
                descriptor = plant.describe(env.native)
                old.need(
                    sha256(canonical(descriptor)).hexdigest()
                    == coupled.receiver.DESCRIPTOR_SHA256,
                    "fresh exact whole native descriptor",
                )
                for name, value in dict(
                    vin_tensor=7.5,
                    vin_drop_gain=0.1,
                    kp_scale=1.0,
                    kd_scale=1.0,
                    friction_scale=1.0,
                ).items():
                    tensor = env.motor.nominal_parameters[name]
                    old.need(
                        tensor.shape == (64, 1)
                        and torch.isfinite(tensor).all()
                        and (tensor == value).all(),
                        "fixed nominal motor " + name,
                    )
                fixed = {
                    name: env._view(name).detach().cpu().contiguous().numpy().tobytes()
                    for name in frames.FIXED_INPUT_FIELDS
                }
                env._forward()
                second = coupled._frame(env)
                states.update(
                    {
                        f"private_{dev}_forward_end": coupled._rng_state(torch, dev)
                        for dev in ("cpu", "cuda")
                    }
                )
                unchanged = all(
                    env._view(name).detach().cpu().contiguous().numpy().tobytes()
                    == value
                    for name, value in fixed.items()
                )
                old.need(
                    unchanged
                    and env.forward_graph is None
                    and not torch.count_nonzero(env.steps)
                    and integration_calls == 0
                    and env.model.ntendon == 0,
                    "no changed inputs, tendon postprocessing, integration or graph",
                )
            # All original-call-site hooks are closed before detached dispatch.
            repeats = {}
            packets = {}
            with wp.ScopedDevice("cuda:0"):
                for mode in ("concurrent", "serial"):
                    packets[mode + ".bin"], repeats[mode] = rne_capture.repeat_entry(
                        env.model,
                        env.data,
                        rne.entries[0],
                        sha256(rne.entries[0]).hexdigest(),
                        mode,
                    )
        states.update(
            {
                f"caller_{dev}_after": coupled._rng_state(torch, dev)
                for dev in ("cpu", "cuda")
            }
        )
        rng = b"".join(states[key] for key in coupled.receiver.RNG_KEYS)
        rng_metadata = {
            key: dict(bytes=len(states[key]), sha256=sha256(states[key]).hexdigest())
            for key in coupled.receiver.RNG_KEYS
        }
        coupled.receiver.checked_rng(rng, rng_metadata)
        packets.update(
            {
                "frames.bin": first + second,
                "entries.bin": b"".join(rne.entries),
                "outputs.bin": b"".join(rne.outputs),
                "com.entries.bin": b"".join(com.entries),
                "com.weighted.bin": b"".join(com.weighted_entries),
                "rng.bin": rng,
            }
        )
        for name, packet in packets.items():
            old.write(directory / name, packet, receiver.CAPS[name])
        old.need(
            source_binding(source) == binding
            and old.packages() == packages
            and service(source, "run", owner_pid) == live,
            "closed unchanged native child source/packages/caps",
        )
        old.need(
            integrator.EulerCandidateCommit.integrate is refuse,
            "owned integration refusal remains installed through child closure",
        )
        for name, module in tuple(sys.modules.items()):
            if name.startswith("mjlab_microduck") and getattr(module, "__file__", None):
                old.need(
                    Path(module.__file__)
                    .resolve()
                    .is_relative_to(ROOT / "src/mjlab_microduck"),
                    "bound fresh imports",
                )
        record = dict(
            protocol=PROTOCOL + ":child",
            source=source,
            source_binding=binding,
            packages=packages,
            declaration_sha256=declaration_sha,
            owner_pid=owner_pid,
            child_pid=os.getpid(),
            child_ppid=os.getppid(),
            compiled_descriptor=descriptor,
            ntendon=env.model.ntendon,
            fixed_state_unchanged=unchanged,
            physics_steps=0,
            graph_created=False,
            actor_model_created=False,
            optimizer_created=False,
            storage_created=False,
            integration_calls=integration_calls,
            com_scope=com.receipt,
            crb_scope=control.receipt,
            rne_scope=rne.receipt,
            repeat_scopes=repeats,
            rng_metadata=rng_metadata,
            flags=dict(FLAGS),
        )
        old.write(directory / "child.json", canonical(record), 128 * 1024)
    finally:
        if integrator.EulerCandidateCommit.integrate is refuse:
            integrator.EulerCandidateCommit.integrate = original


def run(source, directory, binding, live, packages):
    test_raw, test_terminal, test_summary = test_prerequisite(source, binding, packages)
    before = os.lstat(old.LOCK)
    old.need(
        stat.S_ISREG(before.st_mode) and before.st_size == 0, "existing shared GPU lock"
    )
    fd = os.open(old.LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process = None
    try:
        old.need(
            (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            == (before.st_dev, before.st_ino),
            "stable shared lease inode",
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = old.host()
        old.need(
            not sample["processes"] and sample["used_mib"] <= 1024, "idle leased GPU"
        )
        check_window(SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN)
        old.write(directory / "tests.receipt.json", test_raw, 128 * 1024)
        old.write(directory / "tests.terminal.json", canonical(test_terminal), 4096)
        declaration = dict(
            protocol=PROTOCOL + ":declaration",
            source=source,
            source_binding=binding,
            packages=packages,
            service_properties=live,
            host=sample,
            phase_a=coupled.phase_a(),
            predecessors=old.predecessors(),
            com_predecessor=retained_com(),
            expected_tests=receiver.EXPECTED_TESTS,
            tests=test_summary,
            flags=dict(FLAGS),
        )
        raw = canonical(declaration)
        old.write(directory / "declaration.json", raw, 128 * 1024)
        with (directory / "child.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "--mode",
                    "child",
                    "--source",
                    source,
                    "--lease-fd",
                    str(fd),
                    "--owner-pid",
                    str(os.getpid()),
                    "--declaration-sha",
                    sha256(raw).hexdigest(),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                pass_fds=(fd,),
            )
            start = time.monotonic()
            observations, seen, observed_ppid = [], False, None
            while process.poll() is None:
                old.need(
                    time.monotonic() - start < CHILD_SECONDS,
                    "bounded native child runtime",
                )
                path = Path(f"/proc/{process.pid}/stat")
                if path.exists():
                    observed_ppid = int(path.read_text().rsplit(") ", 1)[1].split()[1])
                    old.need(observed_ppid == os.getpid(), "actual owned child PPID")
                sample = old.host()
                old.need(
                    all(row["pid"] == process.pid for row in sample["processes"])
                    and sample["used_mib"] <= 12288
                    and sample["free_mib"] >= 10240,
                    "one bounded GPU child, no foreign occupancy",
                )
                seen |= any(row["pid"] == process.pid for row in sample["processes"])
                observations.append(
                    dict(
                        elapsed=time.monotonic() - start,
                        child_pid=process.pid,
                        host=sample,
                    )
                )
                old.need(
                    len(observations) <= 480 and log.tell() <= 1024**2,
                    "bounded monitor/log",
                )
                time.sleep(0.5)
            log.flush()
            os.fsync(log.fileno())
        old.need(
            process.returncode == 0 and seen and observed_ppid == os.getpid(),
            "actually observed successful native child",
        )
        filtered = "\n".join(
            line
            for line in old.bounded(directory / "child.log", 1024**2)
            .decode()
            .splitlines()
            if line != "[mdp] Patches 1-2 active: NaN-safe reward/advantage"
        )
        old.need(
            not old.BAD_LOG.search(filtered), "finite warning-free owned child log"
        )
        old.need(
            source_binding(source) == binding
            and old.packages() == packages
            and service(source, "run", os.getpid()) == live,
            "closed unchanged native owner source/packages/caps",
        )
        coupled.phase_a()
        old.predecessors()
        retained_com()
        closure_host = old.host()
        old.need(
            not closure_host["processes"] and closure_host["used_mib"] <= 1024,
            "fresh idle GPU and preserved host after child exit",
        )
        old.need(
            {path.name for path in directory.iterdir()}
            == set(receiver.CAPS) - {"report.json"}
            and sum(path.stat().st_size for path in directory.iterdir()) < 16 * 1024**2,
            "bounded exact evidence inventory before report",
        )
        report = dict(
            protocol=PROTOCOL + ":report",
            source=source,
            source_binding=binding,
            child_sha256=sha256(
                old.bounded(directory / "child.json", 128 * 1024)
            ).hexdigest(),
            returncode=0,
            observed_child_pid=process.pid,
            observed_child_ppid=observed_ppid,
            gpu_child_observed=True,
            monitor=observations,
            closure_host=closure_host,
            flags=dict(FLAGS),
        )
        old.write(directory / "report.json", canonical(report), 256 * 1024)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", required=True, choices=("tests", "run", "child"))
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--declaration-sha")
    args = parser.parse_args()
    if args.mode == "child":
        child(args.source, args.lease_fd, args.owner_pid, args.declaration_sha)
        return
    old.test_environment()
    check_window(
        (2 if args.mode == "tests" else 1) * SERVICE_SECONDS + CLOSEOUT_SECONDS + MARGIN
    )
    binding, packages, live = (
        source_binding(args.source),
        old.packages(),
        service(args.source, args.mode, os.getpid()),
    )
    coupled.phase_a()
    old.predecessors()
    retained_com()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle before fresh service",
    )
    directory = output(args.source, args.mode)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        (run_tests if args.mode == "tests" else run)(
            args.source, directory, binding, live, packages
        )
    except BaseException as error:
        old.write(
            directory / "failure.json",
            canonical(
                dict(
                    protocol=PROTOCOL + ":failure",
                    source=args.source,
                    exception_type=type(error).__name__,
                    flags=dict(FLAGS),
                )
            ),
            4096,
        )
        raise


if __name__ == "__main__":
    main()
