"""Fresh leased four-case coupled CoM diagnostic; no integration or learner."""

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

from mjlab_microduck import stance_com_coupled_cpu as cpu
from mjlab_microduck import stance_com_coupled_frames as frames
from mjlab_microduck import stance_com_coupled_receiver as receiver
from mjlab_microduck import stance_com_entry_probe as old
from mjlab_microduck.stance_com_entry_receiver import FLAGS, SERVICE_CAPS, canonical

PROTOCOL = receiver.DATA_PROTOCOL
MODULE = "mjlab_microduck.stance_com_coupled_probe"
BASE = "ab88b2621edc14214e96922bc40e7f3685312113"
ROOT = old.ROOT
CUTOFF = cpu.CUTOFF
SERVICE_SECONDS, CHILD_SECONDS, CLOSEOUT_SECONDS, MARGIN = 300, 240, 300, 60
EXPECTED_TESTS = receiver.EXPECTED_TESTS
OWN = cpu.OWN | {
    f"src/mjlab_microduck/stance_com_coupled_{name}.py"
    for name in ("frames", "probe", "receiver")
}
OWN |= {
    f"tests/test_stance_com_coupled_{name}.py"
    for name in ("frames", "probe", "receiver")
}
PHASE_A = {
    "receipt.json": "1728f97adb06662fcca45baaba470b2744bb0b006882886ea386af38de93414b",
    "junit.xml": "937eef453d3e66f12ed5032722310c745a91f478a42fe657cdfbc3446dc8fd47",
    "pytest.log": "febe0fcdf4aadebe52fa248899de69a3835508270359d8abdcbf2b8f7d922dc2",
}
DESCRIPTOR_FIXTURE = "artifacts/evaluations/com-entry-run-cbcc4d88ca3f/child.json"
DESCRIPTOR_FIXTURE_SHA256 = (
    "56370eb3546f19a395a89cf864fc83e25d0690413b6bfbcee459b54b151941b9"
)


def descriptor_fixture():
    raw = old.bounded(ROOT / DESCRIPTOR_FIXTURE, 128 * 1024)
    old.need(
        len(raw) == 9948 and sha256(raw).hexdigest() == DESCRIPTOR_FIXTURE_SHA256,
        "whole immutable descriptor test fixture before decode",
    )
    descriptor = json.loads(raw)["compiled_descriptor"]
    old.need(
        sha256(canonical(descriptor)).hexdigest() == receiver.DESCRIPTOR_SHA256,
        "exact retained compiled descriptor",
    )
    return dict(path=DESCRIPTOR_FIXTURE, bytes=len(raw), sha256=sha256(raw).hexdigest())


def check_window(reserve, now=None):
    cpu.check_window(reserve, now)


def unit(source, mode):
    cpu.unit(source)
    old.need(
        type(mode) is str and mode in ("tests", "run"), "fresh coupled service mode"
    )
    return f"microduck-com-coupled-{mode}-{source[:12]}.service"


def output(source, mode):
    unit(source, mode)
    return (
        ROOT
        / "artifacts"
        / ("tools" if mode == "tests" else "evaluations")
        / f"com-coupled-{mode}-{source[:12]}"
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
        "exact clean fresh coupled source",
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, source], check=True, timeout=15
    )
    old.need(
        set(old.read("git", "diff", "--name-only", BASE, source).splitlines()) <= OWN,
        "fresh coupled source fence",
    )
    leaves = {}
    for path in old.read(
        "git", "ls-files", "src", "tests", "pyproject.toml", "uv.lock", *sorted(OWN)
    ).splitlines():
        raw = old.bounded(ROOT / path, 4 * 1024**2, allow_empty=True)
        old.need(
            raw
            == subprocess.check_output(["git", "show", f"{source}:{path}"], timeout=15),
            "whole committed coupled leaf " + path,
        )
        leaves[path] = sha256(raw).hexdigest()
    old.need(len(leaves) == 640, "exact fresh coupled leaf inventory")
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
        "actual fresh retained service",
    )
    return {
        k: values[k]
        for k in (*caps, "MainPID", "InvocationID", "ActiveState", "SubState")
    }


def phase_a():
    root = cpu.output(BASE)
    for name, digest in PHASE_A.items():
        old.need(
            sha256(old.bounded(root / name, 1024**2)).hexdigest() == digest,
            "immutable native CPU Phase A " + name,
        )
    record = json.loads(old.bounded(root / "receipt.json", 128 * 1024))
    old.need(
        type(record["tests"]) is int
        and record["tests"] == 1923
        and record["source_binding"]["source"] == BASE
        and record["service_properties"]["InvocationID"]
        == "aae498825b9f4814ab3575171e2ff7aa",
        "retained actual Phase A receipt",
    )
    values = dict(
        row.split("=", 1)
        for row in old.read("systemctl", "--user", "show", cpu.unit(BASE)).splitlines()
        if "=" in row
    )
    expected = dict(
        cpu.CAPS,
        InvocationID="aae498825b9f4814ab3575171e2ff7aa",
        MainPID="0",
        Result="success",
        ExecMainStatus="0",
        ActiveState="active",
        SubState="exited",
    )
    old.need(
        all(values.get(k) == v for k, v in expected.items()),
        "immutable completed Phase A invocation",
    )
    return dict(source=BASE, files=PHASE_A, terminal=expected)


def test_files():
    return cpu.test_files() + [
        f"tests/test_stance_com_coupled_{name}.py"
        for name in ("frames", "probe", "receiver")
    ]


def checked_junit(raw):
    old.need(
        type(raw) is bytes and 0 < len(raw) <= 1024**2, "bounded whole current JUnit"
    )
    suites = list(ET.fromstring(raw).iter("testsuite"))
    old.need(
        suites
        and sum(int(s.get("tests", "-1")) for s in suites) == EXPECTED_TESTS
        and all(
            s.get(k) == "0" for s in suites for k in ("skipped", "errors", "failures")
        ),
        "exact current CPU suite without skips/errors/failures",
    )


def run_tests(source, directory, binding, live, packages):
    environment = old.test_environment()
    descriptor = descriptor_fixture()
    fixtures = old.test_fixture_binding(ROOT)
    old.need(
        fixtures == old.test_fixture_binding(old.OLD_ROOT),
        "unchanged copied CPU fixtures",
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
    old.need(result.returncode == 0, "fresh coupled full CPU tests passed")
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
        and descriptor_fixture() == descriptor
        and old.test_fixture_binding(ROOT)
        == fixtures
        == old.test_fixture_binding(old.OLD_ROOT),
        "unchanged current CPU source/package/fixtures",
    )
    phase_a()
    old.predecessors()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle preserved host after CPU tests",
    )
    check_window(CLOSEOUT_SECONDS + MARGIN)
    receipt = dict(
        protocol=PROTOCOL + ":tests",
        source_binding=binding,
        packages=packages,
        tests=EXPECTED_TESTS,
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
        and record["tests"] == EXPECTED_TESTS
        and record["test_files"] == test_files()
        and record["test_environment"] == old.TEST_ENV
        and record["protocol"] == PROTOCOL + ":tests"
        and receiver.old.false_flags(record["flags"])
        and record["retained_descriptor_fixture"] == descriptor_fixture(),
        "whole fresh current CPU prerequisite",
    )
    xml = old.bounded(directory / "junit.xml", 1024**2)
    checked_junit(xml)
    old.need(
        sha256(xml).hexdigest() == record["junit_sha256"]
        and sha256(old.bounded(directory / "pytest.log", 1024**2)).hexdigest()
        == record["log_sha256"],
        "whole CPU log/JUnit hashes",
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
        "same successful current CPU test invocation",
    )
    return (
        raw,
        terminal,
        dict(
            count=EXPECTED_TESTS,
            files=len(test_files()),
            source_binding=binding,
            receipt_sha256=sha256(raw).hexdigest(),
            terminal=terminal,
        ),
    )


def _rng_state(torch, device):
    value = (
        (
            torch.random.get_rng_state()
            if device == "cpu"
            else torch.cuda.get_rng_state(0)
        )
        .detach()
        .cpu()
        .contiguous()
    )
    old.need(
        value.dtype is torch.uint8 and value.ndim == 1 and 0 < value.numel() <= 16384,
        "bounded actual RNG state",
    )
    return value.numpy().tobytes()


def _frame(env):
    import numpy as np

    env._sync()
    chunks = []
    for name, suffix in frames.FRAME_FIELDS:
        value = env._view(name).detach().cpu().contiguous().numpy()
        old.need(
            value.shape == (64, *suffix)
            and value.dtype == np.dtype(np.float32)
            and np.isfinite(value).all(),
            "whole finite solved frame " + name,
        )
        chunks.append(value.astype("<f4", copy=False).tobytes())
    raw = b"".join(chunks)
    old.need(len(raw) == frames.FRAME_BYTES, "complete fixed frame packet")
    return raw


def child(source, lease_fd, owner_pid, declaration_sha):
    old.need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and type(owner_pid) is int
        and os.getppid() == owner_pid,
        "actual inherited native child and parent",
    )
    old.inherited_lease(lease_fd)
    check_window(CHILD_SECONDS + CLOSEOUT_SECONDS + MARGIN)
    binding = source_binding(source)
    packages = old.packages()
    live = service(source, "run", owner_pid)
    directory = output(source, "run")
    raw = old.bounded(directory / "declaration.json", 128 * 1024)
    old.need(
        sha256(raw).hexdigest() == declaration_sha,
        "whole native declaration before child decode",
    )
    declaration = json.loads(raw)
    old.need(
        declaration["source_binding"] == binding
        and declaration["packages"] == packages
        and declaration["service_properties"] == live
        and declaration["protocol"] == PROTOCOL + ":declaration"
        and declaration["source"] == source
        and receiver.old.false_flags(declaration["flags"]),
        "exact native child declaration",
    )
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle before native CUDA initialization",
    )
    import torch
    import warp as wp
    from mjlab_microduck import (
        stance_com_entry_capture as capture,
        stance_crb_runtime_control as crb,
    )
    from mjlab_microduck import (
        stance_plant_evidence as plant,
        stance_recovery_schedule as schedule,
        stance_warp_integrator as integrator,
    )
    from mjlab_microduck.stance_recovery_schedule_runtime import (
        ScheduledRecoveryRuntime,
    )
    from mjlab_microduck.stance_warp_runtime import WarpStanceRuntime

    old.need(not torch.cuda.is_initialized(), "no early Torch CUDA initialization")
    torch.manual_seed(673)
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(677)
    old.need(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "one initialized visible GPU",
    )
    integration_calls = 0
    original = integrator.EulerCandidateCommit.integrate

    def refuse(*_args, **_kwargs):
        nonlocal integration_calls
        integration_calls += 1
        raise ValueError("integration forbidden in coupled CoM diagnostic")

    integrator.EulerCandidateCommit.integrate = refuse
    cases = {}
    try:
        for case in frames.CASES:
            mode = "original" if case.startswith("original") else "serial"
            states = {
                f"caller_{dev}_before": _rng_state(torch, dev)
                for dev in ("cpu", "cuda")
            }
            com = capture.NativeCoupledComControl(mode=mode)
            control = crb.StanceCrbRuntimeControl(mode="serial", max_forward_calls=2)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(977)
                torch.cuda.manual_seed(983)
                states.update(
                    {
                        f"private_{dev}_start": _rng_state(torch, dev)
                        for dev in ("cpu", "cuda")
                    }
                )
                with wp.ScopedDevice("cuda:0"), control, com:
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
                        "exact unmodified scheduled eager runtime",
                    )
                    states.update(
                        {
                            f"private_{dev}_ctor_end": _rng_state(torch, dev)
                            for dev in ("cpu", "cuda")
                        }
                    )
                    first = _frame(env)
                    descriptor = plant.describe(env.native)
                    old.need(
                        sha256(canonical(descriptor)).hexdigest()
                        == receiver.DESCRIPTOR_SHA256
                        and descriptor["selected_fields_sha256"]
                        == "6a4e7578da3b0f4ffd1f710c8d3cffe9d99330d7ee05aa08eee668d922b7f63f",
                        "fresh exact native compiled descriptor",
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
                            "frozen nominal motor " + name,
                        )
                    fixed = {
                        name: env._view(name)
                        .detach()
                        .cpu()
                        .contiguous()
                        .numpy()
                        .tobytes()
                        for name in frames.FIXED_INPUT_FIELDS
                    }
                    env._forward()
                    second = _frame(env)
                    states.update(
                        {
                            f"private_{dev}_forward_end": _rng_state(torch, dev)
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
                        and integration_calls == 0,
                        "unchanged fixed state and no graph/integration",
                    )
            states.update(
                {
                    f"caller_{dev}_after": _rng_state(torch, dev)
                    for dev in ("cpu", "cuda")
                }
            )
            rng_raw = b"".join(states[key] for key in receiver.RNG_KEYS)
            rng_metadata = {
                key: dict(
                    bytes=len(states[key]), sha256=sha256(states[key]).hexdigest()
                )
                for key in receiver.RNG_KEYS
            }
            receiver.checked_rng(rng_raw, rng_metadata)
            for kind, packet in (
                ("frames", first + second),
                ("entries", b"".join(com.entries)),
                ("weighted", b"".join(com.weighted_entries)),
                ("rng", rng_raw),
            ):
                old.write(
                    directory / f"{case}.{kind}.bin",
                    packet,
                    receiver.CAPS[f"{case}.{kind}.bin"],
                )
            cases[case] = dict(
                mode=mode,
                compiled_descriptor=descriptor,
                ntendon=env.model.ntendon,
                fixed_state_unchanged=unchanged,
                physics_steps=int(env.steps.sum()),
                com_scope=com.receipt,
                crb_scope=control.receipt,
                rng_metadata=rng_metadata,
            )
            del env
        old.need(
            source_binding(source) == binding
            and old.packages() == packages
            and service(source, "run", owner_pid) == live,
            "unchanged child source/packages at closure",
        )
        for name, module in tuple(sys.modules.items()):
            if name.startswith("mjlab_microduck") and getattr(module, "__file__", None):
                old.need(
                    Path(module.__file__)
                    .resolve()
                    .is_relative_to(ROOT / "src/mjlab_microduck"),
                    "bound fresh repository imports",
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
            case_order=list(frames.CASES),
            cases=cases,
            integration_calls=integration_calls,
            physics_steps=0,
            graph_created=False,
            actor_model_created=False,
            optimizer_created=False,
            storage_created=False,
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
        stat.S_ISREG(before.st_mode) and before.st_size == 0,
        "existing regular shared GPU lock",
    )
    fd = os.open(old.LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    process = None
    try:
        old.need(
            (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
            == (before.st_dev, before.st_ino),
            "stable existing lease inode",
        )
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sample = old.host()
        old.need(
            not sample["processes"] and sample["used_mib"] <= 1024,
            "fresh leased idle GPU",
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
            phase_a=phase_a(),
            predecessors=old.predecessors(),
            expected_tests=EXPECTED_TESTS,
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
            observations = []
            seen = False
            observed_ppid = None
            while process.poll() is None:
                old.need(
                    time.monotonic() - start < CHILD_SECONDS,
                    "bounded native child time",
                )
                path = Path(f"/proc/{process.pid}/stat")
                if path.exists():
                    observed_ppid = int(path.read_text().rsplit(") ", 1)[1].split()[1])
                    old.need(observed_ppid == os.getpid(), "actual owned child parent")
                sample = old.host()
                old.need(
                    all(row["pid"] == process.pid for row in sample["processes"])
                    and sample["used_mib"] <= 12288
                    and sample["free_mib"] >= 10240,
                    "one bounded GPU workload, no foreign occupancy",
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
                    "bounded monitor and owned log",
                )
                time.sleep(0.5)
            log.flush()
            os.fsync(log.fileno())
        old.need(
            process.returncode == 0 and seen and observed_ppid == os.getpid(),
            "successful actually observed GPU child",
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
            "unchanged owner source/packages at closure",
        )
        phase_a()
        old.predecessors()
        old.need(
            sum(path.stat().st_size for path in directory.iterdir()) < 16 * 1024**2,
            "whole bounded output inventory",
        )
        report = dict(
            protocol=PROTOCOL + ":report",
            source=source,
            source_binding=binding,
            child_sha256=sha256(
                old.bounded(directory / "child.json", 128 * 1024)
            ).hexdigest(),
            returncode=process.returncode,
            observed_child_pid=process.pid,
            observed_child_ppid=observed_ppid,
            gpu_child_observed=seen,
            monitor=observations,
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
    parser.add_argument("--mode", choices=("tests", "run", "child"), required=True)
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
    binding = source_binding(args.source)
    live = service(args.source, args.mode, os.getpid())
    packages = old.packages()
    phase_a()
    old.predecessors()
    sample = old.host()
    old.need(
        not sample["processes"] and sample["used_mib"] <= 1024,
        "idle before fresh owned service work",
    )
    directory = output(args.source, args.mode)
    directory.mkdir(parents=True, exist_ok=False)
    function = run_tests if args.mode == "tests" else run
    try:
        function(args.source, directory, binding, live, packages)
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
